# /// script
# requires-python = ">=3.10"
# dependencies = ["robokit[mjwarp,pt-cu126]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
"""G1 tracking: a Unitree G1 learns to track a LAFAN1 dance on 4096 MuJoCo Warp worlds, scored on tracking errors.

Example usage:
    uv run src/robokit/examples/rl/08_g1_tracking.py --minutes 10
    uv run src/robokit/examples/rl/08_g1_tracking.py --coarse 0.5  # coarse physics mixed in over the first half
"""

import functools
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import mujoco_warp as mjw  # pyright: ignore[reportMissingImports]
import numpy as np
import torch
import tyro

from robokit.assets import ROBOKIT_HOME
from robokit.assets.robots.humanoids import unitree_g1
from robokit.helpers.rl import Env, ppo
from robokit.sim import CoarseConfig, MjWarpSim, mjscene
from robokit.xform.numpy import quaternion_to_matrix


@dataclass(frozen=True)
class TrackingConfig:
    """The tracking task and its training: the reference motion, the worlds and the learner's time."""

    motion_url: str = (
        "https://storage.googleapis.com/mjlab_beta/lafan_dance1_subject1.npz"  # the npz, in the scene's order
    )
    worlds: int = 4096  # MjWarpSim worlds
    minutes: float = 10.0  # wall time to train, compilation included
    coarse: float = 0.0  # the share of PPO's time that mixes in coarse physics (`coarse_physics`)
    control_dt: float = 0.02  # s per policy action; physics steps are `mjm.opt.timestep`
    episode: int = 500  # env steps per episode (10 s)
    # The tracked bodies
    bodies: Tuple[str, ...] = (
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    )
    ee: Tuple[int, ...] = (3, 6, 10, 13)  # the ankles and wrists, whose heights end an episode
    # The tracking errors and success rate
    metrics: Tuple[str, ...] = ("mpkpe", "r_mpkpe", "joint_vel_error", "ee_pos_error", "ee_ori_error", "success")
    # Coarse physics, off by default (`CoarseConfig`): only the feet and the floor collide
    coarse_physics: CoarseConfig = CoarseConfig(collide=r"robot/(left|right)_foot[1-7]_collision|floor", pyramidal=True)


class G1Tracking(Env):
    """G1 motion tracking on flat ground, in the full physics or the `coarse` one."""

    def __init__(self, config: TrackingConfig = TrackingConfig(), coarse: Optional[CoarseConfig] = None):
        n = config.worlds
        self.n = n
        self.config = config
        self.anchor = config.bodies.index("torso_link")  # the body the reference is re-anchored to
        self.ee = list(config.ee)  # a list, for advanced indexing

        # --- build scene ---
        robot_spec = mjscene.build_robot_spec(unitree_g1.MJCF_PATH, prefix="robot/")
        T_world_base = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        scene_spec = mjscene.build_scene_spec([(robot_spec, T_world_base)])
        scene_spec.option.iterations = 10
        scene_spec.option.ls_iterations = 20
        scene_spec.option.ccd_iterations = 50
        # the initial state, standing: each actuator holds its joint's home angle
        leg = [-0.312, 0.0, 0.0, 0.669, -0.363, 0.0]  # hip pitch, roll and yaw, knee, ankle pitch and roll
        arms = [0.2, 0.2, 0, 0.6, 0, 0, 0, 0.2, -0.2, 0, 0.6, 0, 0, 0]  # each shoulder's, elbow's and wrist's
        home = np.array([0, 0, 0.76, 1, 0, 0, 0, *leg, *leg, 0, 0, 0, *arms])  # the base, the legs, the waist, the arms
        model = scene_spec.compile()
        scene_spec.add_key(name="init_state", qpos=home, ctrl=home[model.jnt_qposadr[model.actuator_trnid[:, 0]]])
        for sensor in [s for s in scene_spec.sensors if not s.name.startswith("robot/self_collision")]:
            scene_spec.delete(sensor)  # only the self-collision sensor
        if coarse is not None:
            scene_spec = MjWarpSim.build_coarse_spec(scene_spec, coarse)

        # --- build simulator ---
        mjm = scene_spec.compile()
        self.mjm = mjm
        self.substeps = round(config.control_dt / mjm.opt.timestep)
        sim = MjWarpSim(mjm, n, substeps=self.substeps, key="init_state", njmax=250, nconmax=35)
        self.sim = sim
        sim.opt.contact_sensor_maxmatch = 64  # contacts a contact sensor can match
        sim.opt.warn_overflow &= ~int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)  # capped on purpose
        sim.set_torch_stream()
        robot = sim.build_entity("robot/", soft_joint_pos_limit_factor=0.9)  # 90% of each joint's range is penalty-free
        self.robot = robot
        self.device = robot.device
        self.self_collision = sim.build_contact_sensor("robot/self_collision")  # the G1's, from its MJCF

        # --- set up actions ---
        # a joint target is its home angle + a quarter of its actuator's effort / stiffness per unit of action
        effort = mjm.actuator_forcerange[robot.actuator_ids, 1] / mjm.actuator_gainprm[robot.actuator_ids, 0]  # rad
        self.action_scale = torch.tensor(0.25 * effort, device=self.device, dtype=torch.float32)
        self.action_offset = robot.data.default_joint_pos[robot.actuated_ids]
        lo, hi = robot.data.soft_joint_pos_limits[robot.actuated_ids].T
        self.bound = torch.maximum(self.action_offset - lo, hi - self.action_offset) / self.action_scale  # for FastSAC

        # --- randomize ---
        robot.data.encoder_bias.uniform_(-0.01, 0.01)  # rad
        # foot friction: one draw per world for all feet
        foot_geoms = [
            g for g in range(mjm.ngeom) if mjm.geom(g).name.startswith(("robot/left_foot", "robot/right_foot"))
        ]
        sim.set_random("geom_friction", (foot_geoms, 0), 0.3, 1.2)
        torso_body = robot.body_ids[robot.find_bodies(["torso_link"])[0]]
        sim.set_random("body_ipos", (torso_body,), (-0.025, -0.05, -0.05), (0.025, 0.05, 0.05), add=True)  # m

        # --- load the reference motion ---
        path = ROBOKIT_HOME / "motions" / Path(config.motion_url).name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_suffix(".part")  # a whole file or none
            urllib.request.urlretrieve(config.motion_url, part)
            part.replace(path)
        motion = np.load(path)
        self.bodies = robot.find_bodies(config.bodies)  # the npz's bodies are the robot's, in its order
        self.ref_joint_pos = torch.tensor(motion["joint_pos"], device=self.device, dtype=torch.float32)
        self.ref_joint_vel = torch.tensor(motion["joint_vel"], device=self.device, dtype=torch.float32)
        self.ref_q_wxyz_world_base = torch.tensor(motion["body_quat_w"][:, 0], device=self.device, dtype=torch.float32)
        self.ref_body_pos_w = torch.tensor(
            motion["body_pos_w"][:, self.bodies], device=self.device, dtype=torch.float32
        )
        ref_body_mat_w = quaternion_to_matrix(motion["body_quat_w"][:, self.bodies])
        self.ref_body_mat_w = torch.tensor(ref_body_mat_w, device=self.device, dtype=torch.float32)
        ref_body_lin_vel_w = motion["body_lin_vel_w"][:, self.bodies]
        self.ref_body_lin_vel_w = torch.tensor(ref_body_lin_vel_w, device=self.device, dtype=torch.float32)
        ref_body_ang_vel_w = motion["body_ang_vel_w"][:, self.bodies]
        self.ref_body_ang_vel_w = torch.tensor(ref_body_ang_vel_w, device=self.device, dtype=torch.float32)
        self.bins = len(self.ref_joint_pos) // round(1 / config.control_dt) + 1  # adaptive sampling's: one per second

        # --- set up rewards ---
        # each term's weight; terms are per second
        weights = dict(
            motion_global_root_pos=0.5,
            motion_global_root_ori=0.5,
            motion_body_pos=1.0,
            motion_body_ori=1.0,
            motion_body_lin_vel=1.0,
            motion_body_ang_vel=1.0,
            action_rate_l2=-0.1,
            joint_limit=-10.0,
            self_collisions=-10.0,
        )
        self.names = list(weights)
        self.weights = torch.tensor(list(weights.values()), device=self.device)

        # --- set up resets and pushes ---
        push = (0.5, 0.5, 0.2, 0.52, 0.52, 0.78)  # m/s x, y, z; rad/s roll, pitch, yaw: pushes and reset velocities
        offsets = (0.05, 0.05, 0.01, 0.1, 0.1, 0.2)  # m x, y, z; rad roll, pitch, yaw: reset offsets
        self.push_range = torch.tensor(push, device=self.device)
        self.reset_range = torch.tensor(offsets + push, device=self.device)

        # --- set up observations ---
        joints = len(robot.joint_names)
        actions = len(robot.actuator_ids)
        # the actor's noise on each observation
        noise = [0.0] * 2 * joints  # the reference's joint angles and velocities
        noise += [0.25] * 3 + [0.05] * 6  # the reference anchor's position (m) and orientation in the robot's
        noise += [0.5] * 3 + [0.2] * 3  # base velocity (m/s), angular velocity (rad/s)
        noise += [0.01] * joints + [0.5] * joints + [0.0] * actions  # joint angles (rad), velocities, the last action
        self.obs_noise = torch.tensor(noise, device=self.device)
        self.obs_dims = (len(noise), len(noise) + 9 * len(config.bodies))  # the critic also sees each body's pose
        imu_site = robot.site_ids[robot.find_sites(["imu_in_pelvis"])[0]]
        self.imu_pos_b = torch.tensor(mjm.site_pos[imu_site][None], device=self.device, dtype=torch.float32)

        # --- initialize state ---
        self.action = torch.zeros(n, actions, device=self.device)
        self.last_action = torch.zeros(n, actions, device=self.device)
        self.push_time_left = torch.zeros(n, device=self.device)
        self.errors = torch.zeros(len(config.metrics), n, device=self.device)
        self.fail_rate = torch.zeros(self.bins, device=self.device)
        self.frame = torch.zeros(n, dtype=torch.long, device=self.device)  # the reference's
        self.steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.failed = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.anchored_ref_pos = torch.zeros(n, len(config.bodies), 3, device=self.device)  # re-anchored by `obs`
        self.anchored_ref_mat = torch.zeros(n, len(config.bodies), 3, 3, device=self.device)
        # each tracking error per world, for `ppo.evaluate`
        self.metrics = {
            name: functools.partial(torch.select, self.errors, 0, i) for i, name in enumerate(config.metrics)
        }
        self.reset(torch.ones(n, dtype=torch.bool, device=self.device))

    def curriculum(self, step: int):
        """No curriculum; at step 0 (a fresh start, or `ppo.evaluate`) adaptive sampling forgets its failures."""
        if step == 0:
            self.fail_rate.zero_()

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Target the home angles + scale * `action`, as the biased encoders read them; step; score and push."""
        self.last_action.copy_(self.action)
        self.action.copy_(action)
        bias = self.robot.data.encoder_bias[:, self.robot.actuated_ids]
        self.robot.set_joint_position_target(action * self.action_scale + self.action_offset - bias)
        self.sim.step()
        return self.score()

    def reset(self, mask: torch.Tensor):
        """Reset the masked worlds, advance the motion, then refresh the kinematics the observations read."""
        self.restart(mask)
        self.sim.kinematics()

    @torch.compile
    def score(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The rewards, terminations, timeouts and each term's mean after a step, at the step's reference frame."""
        robot = self.robot.data
        data = self.sim.data.torch
        frame = self.frame
        anchor = self.anchor
        ee = self.ee
        self.steps += 1

        # --- bodies and their references ---
        body_pos = robot.body_link_pos_w[:, self.bodies]
        body_mat = robot.body_link_mat_w[:, self.bodies]
        body_lin_vel = robot.body_link_lin_vel_w[:, self.bodies]  # of each body's origin
        ref_pos = self.ref_body_pos_w[frame]
        ref_mat = self.ref_body_mat_w[frame]
        anchored_pos = self.anchored_ref_pos
        # each body's rotation from its re-anchored reference, then from its reference, R = ref @ body.T, elementwise
        reference = torch.stack([self.anchored_ref_mat, ref_mat])
        twice_sin = torch.linalg.cross(reference.mT, body_mat.mT.expand_as(reference), dim=-1).sum(-2).norm(dim=-1)
        twice_cos = (reference * body_mat).sum((-2, -1)) - 1  # trace(R) - 1
        angle = torch.atan2(twice_sin, twice_cos)  # rad
        body_angle = angle[0]
        anchor_angle = angle[1][:, anchor]

        # --- rewards ---
        joint_pos = robot.joint_pos
        low, high = robot.soft_joint_pos_limits.T
        lin_vel_error = ((self.ref_body_lin_vel_w[frame] - body_lin_vel) ** 2).sum(-1)
        ang_vel_error = ((self.ref_body_ang_vel_w[frame] - robot.body_link_ang_vel_w[:, self.bodies]) ** 2).sum(-1)
        terms = dict(
            motion_global_root_pos=torch.exp(-((ref_pos[:, anchor] - body_pos[:, anchor]) ** 2).sum(1) / 0.3**2),
            motion_global_root_ori=torch.exp(-(anchor_angle**2) / 0.4**2),
            motion_body_pos=torch.exp(-((anchored_pos - body_pos) ** 2).sum(-1).mean(1) / 0.3**2),
            motion_body_ori=torch.exp(-(body_angle**2).mean(1) / 0.4**2),
            motion_body_lin_vel=torch.exp(-lin_vel_error.mean(1) / 1.0**2),
            motion_body_ang_vel=torch.exp(-ang_vel_error.mean(1) / 3.14**2),
            action_rate_l2=((self.action - self.last_action) ** 2).sum(1),
            joint_limit=((low - joint_pos).clamp(min=0.0) + (joint_pos - high).clamp(min=0.0)).sum(1),
            # substeps with a self-contact over 10 N
            self_collisions=(self.self_collision.data.force_history.norm(dim=-1) > 10.0).any(-1).sum(0).float(),
        )
        term_values = torch.nan_to_num(torch.stack([terms[name] for name in self.names]))
        reward = (term_values * self.weights[:, None] * self.config.control_dt).sum(0)

        # --- terminations ---
        qvel = data.qvel
        diverged = ~qvel.isfinite().all(1) | (qvel.abs() > 100.0).any(1)  # the physics blew up
        anchor_off = (ref_pos[:, anchor, 2] - body_pos[:, anchor, 2]).abs() > 0.25  # m
        anchor_tilted = (ref_mat[:, anchor, 2, 2] - body_mat[:, anchor, 2, 2]).abs() > 0.8
        ends_off = ((anchored_pos[:, ee, 2] - body_pos[:, ee, 2]).abs() > 0.25).any(1)  # the ankles' or wrists' heights
        fell = anchor_off | anchor_tilted | ends_off | diverged
        self.failed.copy_(fell)
        timeout = self.steps >= self.config.episode

        # --- tracking errors, for `metrics` ---
        errors = [
            (ref_pos - body_pos).norm(dim=-1).mean(1),
            (anchored_pos - body_pos).norm(dim=-1).mean(1),
            (self.ref_joint_vel[frame] - robot.joint_vel).pow(2).mean(1).sqrt(),
            (anchored_pos - body_pos)[:, ee].norm(dim=-1).mean(1),
            body_angle[:, ee].mean(1),
            (timeout & ~fell) * float(self.config.episode),  # success, summed over an episode by `ppo.evaluate`'s means
        ]
        self.errors.copy_(torch.stack(errors))

        # --- push: every 1 to 3 s, a random twist added to the base's velocity ---
        self.push_time_left -= self.config.control_dt
        push = self.push_time_left < 1e-6
        next_push = 1.0 + 2.0 * torch.rand(self.n, device=self.device)  # s
        self.push_time_left.copy_(torch.where(push, next_push, self.push_time_left))
        kick = (2 * torch.rand(self.n, 6, device=self.device) - 1) * self.push_range
        self.robot.write_root_link_velocity_to_sim(robot.root_link_vel_w + kick, push)
        return reward, fell, timeout, term_values.mean(1)

    @torch.compile
    def restart(self, mask: torch.Tensor):
        """Start the masked worlds at a perturbed frame drawn by adaptive sampling; advance the other frames."""
        robot = self.robot
        data = self.sim.data.torch
        frames = len(self.ref_joint_pos)
        bins = self.bins

        # --- adaptive sampling: count failures (terminated, not timed out) at their frames' bins; draw frames ---
        failed = (mask & self.failed).float()
        failed_bins = (self.frame * bins // frames).clamp(max=bins - 1)
        failures = torch.zeros(bins, device=self.device).scatter_add_(0, failed_bins, failed)
        # sampling weights: 1 s bins of the motion, by their failure rates
        cdf = (self.fail_rate + 0.1 / bins).cumsum(0)
        drawn = torch.searchsorted(cdf, torch.rand(self.n, 1, device=self.device) * cdf[-1]).squeeze(1)
        drawn_bin = drawn.clamp(max=bins - 1)
        start = (
            ((drawn_bin + torch.rand(self.n, device=self.device)) / bins * (frames - 1)).long().clamp(max=frames - 2)
        )
        self.fail_rate.copy_(0.001 * failures + 0.999 * self.fail_rate)
        wrap = (self.frame + 1 >= frames) & ~mask  # past the motion's end: a drawn frame too, in the same episode
        restart = mask | wrap

        # --- robot: the start frame's base pose and velocity and joint state, perturbed ---
        offset = (2 * torch.rand(self.n, 12, device=self.device) - 1) * self.reset_range
        cos_roll, cos_pitch, cos_yaw = (offset[:, 3:6] / 2).cos().T
        sin_roll, sin_pitch, sin_yaw = (offset[:, 3:6] / 2).sin().T
        q_wxyz_tilt = torch.stack(
            [
                cos_yaw * cos_roll * cos_pitch + sin_yaw * sin_roll * sin_pitch,
                cos_yaw * sin_roll * cos_pitch - sin_yaw * cos_roll * sin_pitch,
                cos_yaw * cos_roll * sin_pitch + sin_yaw * sin_roll * cos_pitch,
                sin_yaw * cos_roll * cos_pitch - cos_yaw * sin_roll * sin_pitch,
            ],
            1,
        )
        q_wxyz_ref = self.ref_q_wxyz_world_base[start]
        # the Hamilton product: the reference's orientation, then the tilt
        w_tilt = q_wxyz_tilt[:, :1]
        xyz_tilt = q_wxyz_tilt[:, 1:]
        w_ref = q_wxyz_ref[:, :1]
        xyz_ref = q_wxyz_ref[:, 1:]
        w_base = w_tilt * w_ref - (xyz_tilt * xyz_ref).sum(1, keepdim=True)
        xyz_base = w_tilt * xyz_ref + w_ref * xyz_tilt + torch.cross(xyz_tilt, xyz_ref, dim=1)
        q_wxyz_world_base = torch.cat([w_base, xyz_base], 1)
        t_world_base = self.ref_body_pos_w[start, 0] + offset[:, :3]
        robot.write_root_link_pose_to_sim(torch.cat([t_world_base, q_wxyz_world_base], 1), restart)
        lin_vel_w = self.ref_body_lin_vel_w[start, 0] + offset[:, 6:9]
        ang_vel_w = self.ref_body_ang_vel_w[start, 0] + offset[:, 9:]
        robot.write_root_link_velocity_to_sim(torch.cat([lin_vel_w, ang_vel_w], 1), restart)
        joint_offset = 0.1 * (2 * torch.rand(self.n, len(robot.joint_names), device=self.device) - 1)  # rad
        joint_pos = (self.ref_joint_pos[start] + joint_offset).clamp(*robot.data.soft_joint_pos_limits.T)
        robot.write_joint_state_to_sim(joint_pos, self.ref_joint_vel[start], restart)
        warmstart = data.qacc_warmstart
        warmstart.copy_(torch.where(restart[:, None], 0.0, warmstart))

        # --- episode: actions, the reference frame, steps and the next push ---
        for state in (self.action, self.last_action):
            state.copy_(torch.where(mask[:, None], 0.0, state))
        self.frame.copy_(torch.where(mask, start + 1, torch.where(wrap, start, self.frame + 1)))
        self.steps.masked_fill_(mask, 0)
        next_push = 1.0 + 2.0 * torch.rand(self.n, device=self.device)  # s
        self.push_time_left.copy_(torch.where(mask, next_push, self.push_time_left))

    @torch.compile
    def obs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The actor's (noisy, encoder-biased) and the critic's observations, after re-anchoring the reference."""
        robot = self.robot.data
        frame = self.frame
        anchor = self.anchor
        ref_pos = self.ref_body_pos_w[frame]
        ref_mat = self.ref_body_mat_w[frame]
        body_pos = robot.body_link_pos_w[:, self.bodies]
        body_mat = robot.body_link_mat_w[:, self.bodies]

        # --- re-anchor the reference: to the robot's anchor body's horizontal position and heading ---
        R_heading = body_mat[:, anchor] @ ref_mat[:, anchor].mT
        yaw = torch.atan2(R_heading[:, 1, 0], R_heading[:, 0, 0])
        zero = torch.zeros_like(yaw)
        one = torch.ones_like(yaw)
        R_yaw = torch.stack([yaw.cos(), -yaw.sin(), zero, yaw.sin(), yaw.cos(), zero, zero, zero, one], 1)
        R_yaw = R_yaw.view(self.n, 1, 3, 3)
        origin = torch.cat([body_pos[:, anchor, :2], ref_pos[:, anchor, 2:]], 1)[:, None]
        self.anchored_ref_pos.copy_(origin + (R_yaw @ (ref_pos - ref_pos[:, anchor, None])[..., None]).squeeze(-1))
        self.anchored_ref_mat.copy_(R_yaw @ ref_mat)

        # --- observations ---
        R_anchor_world = body_mat[:, anchor].mT
        anchor_pos = (R_anchor_world @ (ref_pos[:, anchor] - body_pos[:, anchor])[..., None]).squeeze(-1)
        anchor_ori = (R_anchor_world @ ref_mat[:, anchor])[..., :2].flatten(1)  # the reference anchor's, two columns
        body_pos_b = (R_anchor_world[:, None] @ (body_pos - body_pos[:, anchor, None])[..., None]).flatten(1)
        body_ori_b = (R_anchor_world[:, None] @ body_mat)[..., :2].flatten(1)
        imu_lin_vel_b = robot.root_link_lin_vel_b + torch.linalg.cross(robot.root_link_ang_vel_b, self.imu_pos_b)
        command = torch.cat([self.ref_joint_pos[frame], self.ref_joint_vel[frame]], 1)
        actor = [
            command,
            anchor_pos,
            anchor_ori,
            imu_lin_vel_b,
            robot.root_link_ang_vel_b,
            robot.joint_pos_biased - robot.default_joint_pos,
            robot.joint_vel,
            self.action,
        ]
        critic = [
            command,
            anchor_pos,
            anchor_ori,
            body_pos_b,
            body_ori_b,
            imu_lin_vel_b,
            robot.root_link_ang_vel_b,
            robot.joint_pos - robot.default_joint_pos,
            robot.joint_vel,
            self.action,
        ]
        noise = self.obs_noise * (2 * torch.rand(self.n, len(self.obs_noise), device=self.device) - 1)
        return torch.cat(actor, 1) + noise, torch.cat(critic, 1)


if __name__ == "__main__":
    config = tyro.cli(TrackingConfig)
    env = G1Tracking(config)  # the full physics, where the policy is scored
    coarse_env = G1Tracking(config, coarse=config.coarse_physics) if config.coarse else None
    learner = ppo.PPO(env.obs_dims, len(env.bound), ppo.PPOConfig(entropy=0.005))
    actor = learner.train(env, config.minutes, coarse_env, config.coarse)
    print(ppo.evaluate(env, actor, env.metrics))
