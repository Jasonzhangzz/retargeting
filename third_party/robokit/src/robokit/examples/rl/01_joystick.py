# /// script
# requires-python = ">=3.10"
# dependencies = ["robokit[mjwarp,pt-cu126]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
"""Joystick: a Unitree Go1 or G1 learns to track twist commands on flat ground, on 4096 MuJoCo Warp worlds.

Example usage:
    uv run src/robokit/examples/rl/01_joystick.py go1 --minutes 1                   # Go1
    uv run src/robokit/examples/rl/01_joystick.py g1 --minutes 4                    # G1
    uv run src/robokit/examples/rl/01_joystick.py g1 --minutes 4 --energy 0.001     # G1, a smoother gait
    uv run src/robokit/examples/rl/01_joystick.py go1 --coarse 0                    # full physics throughout
"""

import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import mujoco  # pyright: ignore[reportMissingImports]
import mujoco_warp as mjw  # pyright: ignore[reportMissingImports]
import numpy as np
import torch
import tyro

from robokit.assets.robots.humanoids import unitree_g1
from robokit.assets.robots.quadrupeds import unitree_go1
from robokit.helpers.rl import Env, fastsac, ppo
from robokit.sim import CoarseConfig, MjWarpSim, mjscene


@dataclass(frozen=True)
class JoystickConfig:
    """The joystick task and its training: the robot, the worlds, the power penalty, the learner and its time."""

    mjcf_path: Path  # the robot's MJCF
    impratio: float  # solver impratio
    cone: mujoco.mjtCone  # friction cone
    home: List[float]  # initial qpos: the base's pose, then the joint angles
    foot_contact_type: mujoco.mjtObj  # the feet's terrain contacts are those of geoms or of body subtrees
    foot_contacts: List[str]  # their names
    foot_sites: List[str]
    imu_site: str  # where the actor's base velocity is measured
    torso: str  # the body levelled by the upright reward, its center of mass randomized
    foot_geoms: str  # foot collision geoms, a pattern; their friction is randomized
    foot_friction: List[Tuple[float, float]]  # ranges: sliding uniform, torsional and rolling log-uniform
    pose_std: Dict[str, Tuple[float, float, float]]  # pose reward widths by joint name, rad: standing, walking, running
    weights: Dict[str, float] = field(default_factory=dict)  # extra reward weights
    worlds: int = 4096  # MjWarpSim worlds
    energy: float = 0.0  # the weight of the mechanical power penalty
    algo: Literal["ppo", "fastsac"] = "ppo"  # the learner
    minutes: float = 1.0  # wall time to train, compilation included
    coarse: float = 0.5  # the share of PPO's time that mixes in coarse physics (`coarse_physics`), less and less often
    control_dt: float = 0.02  # s per policy action; physics steps are `mjm.opt.timestep`
    # Coarse physics (`CoarseConfig`): only the feet ("{feet}") and terrain collide, 20 ms steps
    coarse_physics: CoarseConfig = CoarseConfig(
        collide="robot/{feet}|(?!robot/).*", pyramidal=True, timestep=0.02, iterations=5, ls_iterations=10
    )


GO1 = JoystickConfig(
    mjcf_path=unitree_go1.MJCF_PATH,
    impratio=10.0,
    cone=mujoco.mjtCone.mjCONE_ELLIPTIC,
    home=[0, 0, 0.278, 1, 0, 0, 0, *[0.1, 0.9, -1.8, -0.1, 0.9, -1.8] * 2],
    foot_contact_type=mujoco.mjtObj.mjOBJ_GEOM,
    foot_contacts=[f"{leg}_foot_collision" for leg in ("FR", "FL", "RR", "RL")],
    foot_sites=["FR", "FL", "RR", "RL"],
    imu_site="imu",
    torso="trunk",
    foot_geoms=r"(FR|FL|RR|RL)_foot_collision",
    foot_friction=[(0.3, 1.5), (1e-4, 2e-2), (1e-5, 5e-3)],
    pose_std=dict(hip=(0.05, 0.3, 0.3), thigh=(0.05, 0.3, 0.3), calf=(0.1, 0.6, 0.6)),
)
G1 = JoystickConfig(
    mjcf_path=unitree_g1.MJCF_PATH,
    impratio=1.0,
    cone=mujoco.mjtCone.mjCONE_PYRAMIDAL,
    # each leg: hip pitch, roll and yaw, knee, ankle pitch and roll
    home=[0, 0, 0.76, 1, 0, 0, 0, *[-0.312, 0, 0, 0.669, -0.363, 0] * 2]
    + [0, 0, 0, 0.2, 0.2, 0, 0.6, 0, 0, 0, 0.2, -0.2, 0, 0.6, 0, 0, 0],
    foot_contact_type=mujoco.mjtObj.mjOBJ_XBODY,
    foot_contacts=["left_ankle_roll_link", "right_ankle_roll_link"],
    foot_sites=["left_foot", "right_foot"],
    imu_site="imu_in_pelvis",
    torso="torso_link",
    foot_geoms=r"(left|right)_foot[1-7]_collision",
    foot_friction=[(0.3, 1.2)],
    pose_std=dict(
        hip_pitch=(0.05, 0.3, 0.5),
        hip_roll=(0.05, 0.15, 0.2),
        hip_yaw=(0.05, 0.15, 0.2),
        knee=(0.05, 0.35, 0.6),
        ankle_pitch=(0.05, 0.25, 0.35),
        ankle_roll=(0.05, 0.1, 0.15),
        waist_yaw=(0.05, 0.2, 0.3),
        waist_roll=(0.05, 0.08, 0.08),
        waist_pitch=(0.05, 0.1, 0.2),
        wrist=(0.05, 0.3, 0.3),
        shoulder_pitch=(0.05, 0.15, 0.5),
        shoulder_roll=(0.05, 0.15, 0.2),
        shoulder_yaw=(0.05, 0.1, 0.15),
        elbow=(0.05, 0.15, 0.35),
    ),
    weights=dict(body_ang_vel=-0.05, angular_momentum=-0.02, self_collisions=-1.0),
)


class Joystick(Env):
    """Velocity tracking on flat ground for `config`'s robot, in the full physics or the `coarse` one."""

    def __init__(self, config: JoystickConfig, coarse: Optional[CoarseConfig] = None):
        n = config.worlds
        self.n = n
        self.config = config

        # --- build scene ---
        robot_spec = mjscene.build_robot_spec(config.mjcf_path, prefix="robot/")
        T_world_base = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        scene_spec = mjscene.build_scene_spec([(robot_spec, T_world_base)], floor=False, timestep=0.005)
        scene_spec.option.iterations = 10
        scene_spec.option.ls_iterations = 20
        scene_spec.option.ccd_iterations = 50
        scene_spec.option.impratio = config.impratio
        scene_spec.option.cone = config.cone
        terrain = scene_spec.worldbody.add_body(name="terrain")
        terrain.add_geom(name="terrain", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.0, 0.0, 0.01])
        # the initial state: each actuator holds its joint's home angle
        home = np.array(config.home)
        model = scene_spec.compile()
        scene_spec.add_key(name="init_state", qpos=home, ctrl=home[model.jnt_qposadr[model.actuator_trnid[:, 0]]])
        for part in config.foot_contacts:  # each foot's contact flag and net force against the terrain
            for reading in ("found", "force"):
                mjscene.build_contact_sensor(
                    scene_spec,
                    f"foot_contact_{part}_{reading}",
                    (config.foot_contact_type, f"robot/{part}"),
                    (mujoco.mjtObj.mjOBJ_BODY, "terrain"),
                    data=(reading,),
                    reduce="netforce",
                )
        if coarse is not None:
            coarse = replace(coarse, collide=coarse.collide.replace("{feet}", config.foot_geoms))
            scene_spec = MjWarpSim.build_coarse_spec(scene_spec, coarse)

        # --- build simulator ---
        mjm = scene_spec.compile()
        self.mjm = mjm
        self.substeps = round(config.control_dt / mjm.opt.timestep)
        assert abs(self.substeps * mjm.opt.timestep - config.control_dt) < 1e-9, (
            "the control step must be a timestep multiple"
        )
        sim = MjWarpSim(mjm, n, substeps=self.substeps, key="init_state", njmax=300)
        self.sim = sim
        sim.opt.contact_sensor_maxmatch = 64  # contacts a contact sensor can match
        sim.opt.warn_overflow &= ~int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)  # capped on purpose
        sim.set_torch_stream()
        robot = sim.build_entity("robot/", soft_joint_pos_limit_factor=0.9)  # 90% of each joint's range is penalty-free
        self.robot = robot
        self.device = robot.device
        self.feet_contact = sim.build_contact_sensor("foot_contact", track_air_time=True)
        self.self_collision = sim.build_contact_sensor("robot/self_collision")  # the G1's, from its MJCF

        # --- set up actions ---
        # a joint target is its home angle + a quarter of its actuator's effort / stiffness per unit of action
        effort = mjm.actuator_forcerange[robot.actuator_ids, 1] / mjm.actuator_gainprm[robot.actuator_ids, 0]  # rad
        self.action_scale = torch.tensor(0.25 * effort, device=self.device, dtype=torch.float32)
        self.action_offset = robot.data.default_joint_pos[robot.actuated_ids]
        lo, hi = robot.data.soft_joint_pos_limits[robot.actuated_ids].T
        self.bound = torch.maximum(self.action_offset - lo, hi - self.action_offset) / self.action_scale  # for FastSAC

        # --- randomize ---
        robot.data.encoder_bias.uniform_(-0.015, 0.015)  # rad
        # foot friction: one draw per world for all feet
        foot_geoms = [g for g in range(mjm.ngeom) if re.fullmatch(f"robot/{config.foot_geoms}", mjm.geom(g).name)]
        for axis, (lo, hi) in enumerate(config.foot_friction):  # sliding uniform, torsional and rolling log-uniform
            sim.set_random("geom_friction", (foot_geoms, axis), lo, hi, log=axis > 0)
        self.torso = robot.find_bodies([config.torso])[0]
        torso_body = robot.body_ids[self.torso]
        sim.set_random("body_ipos", (torso_body,), (-0.025, -0.025, -0.03), (0.025, 0.025, 0.03), add=True)  # m

        # --- set up rewards ---
        # each term's weight; terms are per second
        weights = dict(
            track_linear_velocity=2.0,
            track_angular_velocity=2.0,
            upright=1.0,
            pose=1.0,
            dof_pos_limits=-1.0,
            action_rate_l2=-0.1,
            foot_clearance=-2.0,
            foot_swing_height=-0.25,
            foot_slip=-0.1,
            soft_landing=-1e-5,
            **config.weights,
        )
        if config.energy:
            weights["energy"] = -config.energy
        self.names = list(weights)
        self.weights = torch.tensor(list(weights.values()), device=self.device)
        self.feet = robot.find_sites(config.foot_sites)
        self.angmom = sim.get_sensor("robot/root_angmom") if "angular_momentum" in weights else None  # the G1's
        pose_std = [next(std for key, std in config.pose_std.items() if key in name) for name in robot.joint_names]
        self.pose_std = torch.tensor(pose_std, device=self.device).T  # (standing, walking, running) x joint

        # --- set up resets, pushes and commands ---
        reset_range = torch.tensor(((-0.5, 0.5), (-0.5, 0.5), (0.01, 0.05), (-3.14, 3.14)), device=self.device)
        self.reset_low = reset_range[:, 0]  # m, m, m, rad: offsets of the base's x, y, z and yaw
        self.reset_high = reset_range[:, 1]
        self.push_range = torch.tensor((0.5, 0.5, 0.4, 0.52, 0.52, 0.78), device=self.device)  # m/s x, y, z; rad/s
        self.command_ranges = torch.zeros(3, 2, device=self.device)  # x, y (m/s) and yaw (rad/s), set by `curriculum`
        self.curriculum_ranges = None  # the ranges `curriculum` last set

        # --- set up observations ---
        joints = len(robot.joint_names)
        actions = len(robot.actuator_ids)
        feet = len(self.feet)
        # the actor's noise on each observation
        noise = [0.5] * 3 + [0.2] * 3 + [0.05] * 3  # base velocity (m/s), angular velocity (rad/s), gravity
        noise += [0.01] * joints + [1.5] * joints  # joint angles (rad) and velocities (rad/s)
        noise += [0.0] * (actions + 3)  # the last action and the command
        self.obs_noise = torch.tensor(noise, device=self.device)
        self.obs_dims = (len(noise), len(noise) + 6 * feet)  # the critic also sees each foot's contact state
        imu_site = robot.site_ids[robot.find_sites([config.imu_site])[0]]
        self.imu_pos_b = torch.tensor(
            mjm.site_pos[imu_site][None], device=self.device, dtype=torch.float32
        )  # on the base

        # --- initialize state ---
        self.action = torch.zeros(n, actions, device=self.device)
        self.last_action = torch.zeros(n, actions, device=self.device)
        self.steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.command = torch.zeros(n, 3, device=self.device)
        self.command_time_left = torch.zeros(n, device=self.device)
        self.target_heading = torch.zeros(n, device=self.device)
        self.is_heading = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.is_standing = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.push_time_left = torch.zeros(n, device=self.device)
        self.swing_peak = torch.zeros(n, feet, device=self.device)
        self.foot_height = torch.zeros(n, feet, device=self.device)
        self.curriculum(0)
        self.reset(torch.ones(n, dtype=torch.bool, device=self.device))

    def curriculum(self, step: int):
        """Command curriculum: the x and yaw speed ranges widen after 5000 and 10000 iterations of 24 env steps."""
        ranges = [(-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5)]
        if step >= 5000 * 24:
            ranges[0] = (-1.5, 2.0)
            ranges[2] = (-0.7, 0.7)
        if step >= 10000 * 24:
            ranges[0] = (-2.0, 3.0)
        if ranges != self.curriculum_ranges:
            self.curriculum_ranges = ranges
            self.command_ranges.copy_(torch.tensor(ranges))

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Target the home angles + scale * `action`, as the biased encoders read them; step; score; push."""
        self.last_action.copy_(self.action)
        self.action.copy_(action)
        bias = self.robot.data.encoder_bias[:, self.robot.actuated_ids]
        self.robot.set_joint_position_target(action * self.action_scale + self.action_offset - bias)
        self.sim.step()
        reward, terminated, timeout, terms = self.score()
        self.push()
        return reward, terminated, timeout, terms

    def reset(self, mask: torch.Tensor):
        """Reset the masked worlds, advance the commands, then refresh the kinematics the observations read."""
        self.restart(mask)
        self.sim.kinematics()

    @torch.compile
    def score(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The rewards, terminations, timeouts and each reward term's mean after a step."""
        robot = self.robot.data
        contact = self.feet_contact
        self.steps += 1

        # --- foot contacts ---
        contact.update()
        in_contact = contact.data.found > 0
        landed = contact.compute_first_contact(self.config.control_dt)  # touched down during this step
        swing_peak = torch.maximum(self.swing_peak, self.foot_height)
        self.swing_peak.copy_(torch.where(in_contact, self.swing_peak, swing_peak))
        foot_speed = robot.site_lin_vel_w[:, self.feet, :2].norm(dim=-1)  # horizontal

        # --- rewards ---
        command_speed = self.command[:, :2].norm(dim=1) + self.command[:, 2].abs()
        moving = (command_speed > 0.05).float()  # the foot terms count only when commanded to move
        gait = (command_speed >= 0.05).long() + (command_speed >= 1.5).long()  # standing, walking or running
        lin_vel_b = robot.root_link_lin_vel_b
        ang_vel_b = robot.root_link_ang_vel_b
        lin_vel_error = ((self.command[:, :2] - lin_vel_b[:, :2]) ** 2).sum(1) + lin_vel_b[:, 2] ** 2
        ang_vel_error = (self.command[:, 2] - ang_vel_b[:, 2]) ** 2 + (ang_vel_b[:, :2] ** 2).sum(1)
        low, high = robot.soft_joint_pos_limits.T
        out_of_limits = (low - robot.joint_pos).clamp(min=0.0) + (robot.joint_pos - high).clamp(min=0.0)
        torso_tilt = robot.body_link_mat_w[:, self.torso, 2, :2]  # the z components of the torso's x and y axes
        terms = dict(
            track_linear_velocity=torch.exp(-lin_vel_error / 0.5**2),
            track_angular_velocity=torch.exp(-ang_vel_error / 0.5),
            upright=torch.exp(-(torso_tilt**2).sum(1) / 0.2),
            pose=torch.exp(-((robot.joint_pos - robot.default_joint_pos) ** 2 / self.pose_std[gait] ** 2).mean(1)),
            body_ang_vel=(robot.body_link_ang_vel_w[:, self.torso, :2] ** 2).sum(1),
            dof_pos_limits=out_of_limits.sum(1),
            action_rate_l2=((self.action - self.last_action) ** 2).sum(1),
            foot_clearance=((self.foot_height - 0.1).abs() * foot_speed).sum(1) * moving,
            foot_swing_height=((self.swing_peak / 0.1 - 1) ** 2 * landed).sum(1) * moving,
            foot_slip=(foot_speed**2 * in_contact).sum(1) * moving,
            soft_landing=(contact.data.force.norm(dim=-1) * landed).sum(1) * moving,
            energy=(robot.qfrc_actuator * robot.joint_vel).abs().sum(1),  # W
        )
        if self.angmom is not None:
            terms["angular_momentum"] = (self.angmom**2).sum(1)
        if self.self_collision.primary_names:  # substeps with a self-contact over 10 N
            hits = self.self_collision.data.force_history.norm(dim=-1) > 10.0
            terms["self_collisions"] = hits.any(-1).sum(0).float()
        self.swing_peak.copy_(torch.where(landed, 0.0, self.swing_peak))  # a new swing starts
        term_values = torch.nan_to_num(torch.stack([terms[name] for name in self.names]))
        reward = (term_values * self.weights[:, None] * self.config.control_dt).sum(0)

        # --- terminations ---
        qvel = self.sim.data.torch.qvel
        diverged = ~qvel.isfinite().all(1) | (qvel.abs() > 100.0).any(1)  # the physics blew up
        fell = (-robot.projected_gravity_b[:, 2]).clamp(-1.0, 1.0).acos() > math.radians(70.0)  # tilted past 70°
        timeout = self.steps >= 1000  # 20 s episodes
        return reward, diverged | fell, timeout, term_values.mean(1)

    @torch.compile
    def push(self):
        """Every 1 to 3 s, add a random twist to the base's velocity."""
        self.push_time_left -= self.config.control_dt
        push = self.push_time_left < 1e-6
        next_push = 1.0 + 2.0 * torch.rand(self.n, device=self.device)  # s
        self.push_time_left.copy_(torch.where(push, next_push, self.push_time_left))
        kick = (2 * torch.rand(self.n, 6, device=self.device) - 1) * self.push_range
        self.robot.write_root_link_velocity_to_sim(self.robot.data.root_link_vel_w + kick, push)

    @torch.compile
    def restart(self, mask: torch.Tensor):
        """Reset the masked worlds near home and resample commands, with masked writes only."""
        robot = self.robot

        # --- robot: the base at home, randomly shifted and turned; the joints at home; nothing moving ---
        offset = self.reset_low + (self.reset_high - self.reset_low) * torch.rand(self.n, 4, device=self.device)
        yaw = offset[:, 3]
        zero = torch.zeros_like(yaw)
        q_wxyz_world_base = torch.stack([(yaw / 2).cos(), zero, zero, (yaw / 2).sin()], 1)
        t_world_base = robot.data.default_root_state[:3] + offset[:, :3]
        robot.write_root_link_pose_to_sim(torch.cat([t_world_base, q_wxyz_world_base], 1), mask)
        robot.write_root_link_velocity_to_sim(torch.zeros(self.n, 6, device=self.device), mask)
        home = robot.data.default_joint_pos.clamp(*robot.data.soft_joint_pos_limits.T)
        robot.write_joint_state_to_sim(home, torch.zeros_like(home), mask)
        warmstart = self.sim.data.torch.qacc_warmstart
        warmstart.copy_(torch.where(mask[:, None], 0.0, warmstart))

        # --- episode: steps, foot contacts, actions and the next push ---
        self.steps.masked_fill_(mask, 0)
        self.feet_contact.reset(mask)
        for state in (self.swing_peak, self.action, self.last_action):
            state.copy_(torch.where(mask[:, None], 0.0, state))
        next_push = 1.0 + 2.0 * torch.rand(self.n, device=self.device)  # s
        self.push_time_left.copy_(torch.where(mask, next_push, self.push_time_left))

        # --- commands: new ones on reset and every 3 to 8 s ---
        self.command_time_left.copy_(
            torch.where(mask, self.command_time_left, self.command_time_left - self.config.control_dt)
        )
        new = mask | (self.command_time_left <= 0)
        low, high = self.command_ranges.T
        sampled = low + (high - low) * torch.rand(self.n, 3, device=self.device)
        forward_only = torch.rand(self.n, device=self.device) <= 0.2  # 20%: walk forward at 0.3 m/s or more
        forward = torch.zeros_like(sampled)
        forward[:, 0] = sampled[:, 0].abs().clamp(min=0.3)
        command = torch.where(forward_only[:, None], forward, sampled)
        self.command.copy_(torch.where(new[:, None], command, self.command))
        target_heading = math.pi * (2 * torch.rand(self.n, device=self.device) - 1)
        self.target_heading.copy_(torch.where(new, target_heading, self.target_heading))
        is_heading = (torch.rand(self.n, device=self.device) <= 0.3) & ~forward_only  # 30%: turn to the target heading
        self.is_heading.copy_(torch.where(new, is_heading, self.is_heading))
        is_standing = torch.rand(self.n, device=self.device) <= 0.1  # 10%: stand still
        self.is_standing.copy_(torch.where(new, is_standing, self.is_standing))
        command_time = 3.0 + 5.0 * torch.rand(self.n, device=self.device)  # s
        self.command_time_left.copy_(torch.where(new, command_time, self.command_time_left))

    @torch.compile
    def obs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The actor's (noisy, encoder-biased) and the critic's observations, after steering to the target headings."""
        robot = self.robot.data
        contact = self.feet_contact.data

        # --- heading commands: a yaw speed toward the target heading ---
        heading_error = self.target_heading - robot.heading_w
        heading_error = torch.atan2(heading_error.sin(), heading_error.cos())  # wrapped to [-pi, pi]
        yaw_speed = (0.5 * heading_error).clamp(*self.command_ranges[2])  # rad/s
        self.command[:, 2] = torch.where(self.is_heading, yaw_speed, self.command[:, 2])
        self.command.masked_fill_(self.is_standing[:, None], 0.0)

        # --- observations ---
        imu_lin_vel_b = robot.root_link_lin_vel_b + torch.linalg.cross(robot.root_link_ang_vel_b, self.imu_pos_b)
        self.foot_height.copy_(robot.site_pos_w[:, self.feet, 2].clamp(0.0, 1.0))  # above the plane
        fresh = self.steps.eq(0)[:, None]  # contact sensors are stale after a reset; the feet start in the air
        in_contact = (contact.found > 0) & ~fresh
        contact_force = (contact.force * ~fresh[..., None]).flatten(1)
        actor = [
            imu_lin_vel_b,
            robot.root_link_ang_vel_b,
            robot.projected_gravity_b,
            robot.joint_pos_biased - robot.default_joint_pos,
            robot.joint_vel,
            self.action,
            self.command,
        ]
        critic = [
            imu_lin_vel_b,
            robot.root_link_ang_vel_b,
            robot.projected_gravity_b,
            robot.joint_pos - robot.default_joint_pos,
            robot.joint_vel,
            self.action,
            self.command,
            self.foot_height,
            contact.current_air_time,
            in_contact.float(),
            contact_force.sign() * contact_force.abs().log1p(),
        ]
        noise = self.obs_noise * (2 * torch.rand(self.n, len(self.obs_noise), device=self.device) - 1)
        return torch.cat(actor, 1) + noise, torch.cat(critic, 1)


if __name__ == "__main__":
    config = tyro.cli(tyro.extras.subcommand_type_from_defaults(dict(go1=GO1, g1=G1)))
    env = Joystick(config)  # the full physics, where the policy is scored
    if config.algo == "fastsac":
        actor = fastsac.train(env, config.minutes)
    else:
        coarse_env = Joystick(config, coarse=config.coarse_physics) if config.coarse else None
        actor = ppo.PPO(env.obs_dims, len(env.bound)).train(env, config.minutes, coarse_env, config.coarse)
    print(ppo.evaluate(env, actor))
