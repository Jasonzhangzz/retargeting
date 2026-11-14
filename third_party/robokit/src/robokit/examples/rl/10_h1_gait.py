# /// script
# requires-python = ">=3.10"
# dependencies = ["robokit[mjwarp,pt-cu126]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
"""H1 gait: a Unitree H1 learns to walk at joystick commands with a given step rate and swing height, on 8192 worlds.

Example usage:
    uv run src/robokit/examples/rl/10_h1_gait.py --minutes 5
    uv run src/robokit/examples/rl/10_h1_gait.py --coarse 0.5    # coarse physics mixed in over the first half
"""

import math
import re
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import mujoco  # pyright: ignore[reportMissingImports]
import numpy as np
import torch
import tyro

from robokit.assets.robots.humanoids import unitree_h1
from robokit.helpers.rl import Env, ppo
from robokit.sim import CoarseConfig, MjWarpSim, mjscene


@dataclass(frozen=True)
class GaitConfig:
    """The gait task and its training: the control rate, the observation noise, the worlds and the learner's time."""

    action_scale: float = 0.3  # rad: the joint targets' change per step per unit of action
    noise_level: float = 0.6  # the scale of the actor's observation noise
    worlds: int = 8192  # MjWarpSim worlds
    minutes: float = 5.0  # wall time to train, compilation included
    coarse: float = 0.0  # the share of PPO's time that mixes in coarse physics (`coarse_physics`)
    control_dt: float = 0.02  # s per policy action; physics steps are `mjm.opt.timestep`
    episode: int = 1000  # env steps per episode (20 s)
    # Coarse physics, off by default (`CoarseConfig`): only the feet and the floor collide
    coarse_physics: CoarseConfig = CoarseConfig(collide=r"floor|robot/(left|right)_foot[1-3]", pyramidal=True)
    # Home joint angles: each leg's hip yaw, roll, pitch, knee and ankle, then the torso and the arms
    home_joint_pos: Tuple[float, ...] = (0.0, 0.0, -0.267, 0.8, -0.543) * 2 + (
        0.0,
        0.0,
        0.2,
        0.0,
        1.14,
        0.0,
        -0.2,
        0.0,
        1.14,
    )


class H1Gait(Env):
    """H1 joystick gait tracking on flat ground, in the full physics or the `coarse` one."""

    def __init__(self, config: GaitConfig = GaitConfig(), coarse: Optional[CoarseConfig] = None):
        n = config.worlds
        self.n = n
        self.config = config
        self.action_scale = config.action_scale

        # --- build scene ---
        # the H1 URDF with position actuators; only its foot capsules collide, its meshes dropped
        robot_spec = mjscene.build_robot_spec(unitree_h1.URDF_PATH, prefix="robot/")
        for geom in list(robot_spec.geoms):
            robot_spec.delete(geom)
        for mesh in list(robot_spec.meshes):
            robot_spec.delete(mesh)
        for joint in robot_spec.joints:
            joint.damping = [1.0, 0.0, 0.0]
            joint.armature = 0.1
            leg = any(part in joint.name for part in ("hip", "knee", "ankle"))
            actuator = robot_spec.add_actuator(name=joint.name, target=joint.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            actuator.ctrlrange = joint.range
            actuator.forcerange = joint.actfrcrange
            actuator.set_to_position(kp=800.0 if leg else 400.0, dampratio=1.0)  # the legs, then the torso and arms
        robot_spec.body("robot/pelvis").add_freejoint(name="robot/root")
        robot_spec.body("robot/imu_link").add_site(name="robot/imu")
        capsules = (  # heel, sole, toe in the ankle frame
            (-0.035, 0, -0.056, 0.02, 0, -0.045),
            (0.02, 0, -0.045, 0.115, 0, -0.056),
            (0.14, -0.03, -0.056, 0.14, 0.03, -0.056),
        )
        for side in ("left", "right"):
            ankle = robot_spec.body(f"robot/{side}_ankle_link")
            ankle.add_site(name=f"robot/{side}_foot", pos=[0.02, 0.0, -0.045])
            for i, fromto in enumerate(capsules):
                ankle.add_geom(
                    name=f"robot/{side}_foot{i + 1}",
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    size=[0.014, 0, 0],
                    fromto=fromto,
                    contype=0,
                    conaffinity=1,
                    mass=0.0,
                )
        T_world_base = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        scene_spec = mjscene.build_scene_spec([(robot_spec, T_world_base)], timestep=0.004)
        scene_spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER  # 4 ms Euler steps, 1 solver iteration
        scene_spec.option.iterations = 1
        scene_spec.option.ls_iterations = 5
        scene_spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
        floor = (mujoco.mjtObj.mjOBJ_GEOM, "floor")
        for side in ("left", "right"):  # each foot capsule's contact flag against the floor
            for i in (1, 2, 3):
                capsule = (mujoco.mjtObj.mjOBJ_GEOM, f"robot/{side}_foot{i}")
                name = f"foot_contact_{side}_foot{i}_found"
                mjscene.build_contact_sensor(scene_spec, name, capsule, floor, reduce="mindist")
        home = np.concatenate([[0.0, 0.0, 0.97, 1.0, 0.0, 0.0, 0.0], config.home_joint_pos])
        scene_spec.add_key(name="home", qpos=home, ctrl=config.home_joint_pos)
        if coarse is not None:
            scene_spec = MjWarpSim.build_coarse_spec(scene_spec, coarse)

        # --- build simulator ---
        mjm = scene_spec.compile()
        self.mjm = mjm
        self.substeps = round(config.control_dt / mjm.opt.timestep)
        sim = MjWarpSim(mjm, n, substeps=self.substeps, key="home", njmax=51, nconmax=12)
        self.sim = sim
        sim.set_torch_stream()
        robot = sim.build_entity("robot/")
        self.robot = robot
        self.device = robot.device
        self.feet_contact = sim.build_contact_sensor("foot_contact")  # the left foot's 3 capsules, then the right's

        # --- set up actions ---
        # each step moves the joint targets within the joint ranges, from zero after a reset
        target_range = torch.tensor(mjm.actuator_ctrlrange[robot.actuator_ids], device=self.device, dtype=torch.float32)
        self.target_low = target_range[:, 0]
        self.target_high = target_range[:, 1]
        self.bound = torch.ones(len(robot.actuator_ids), device=self.device)  # actions in [-1, 1]

        # --- set up rewards ---
        # each term's weight; terms are per second
        weights = dict(feet_phase=5.0, tracking_lin_vel=3.5, tracking_ang_vel=0.75, pose=-2.5, action_rate=-0.01)
        self.names = list(weights)
        self.weights = torch.tensor(list(weights.values()), device=self.device)
        self.posed = [i for i, name in enumerate(robot.joint_names) if not re.search("hip_pitch|knee", name)]
        self.pose_weights = torch.tensor([5.0] * 6 + [2.0] + [1.0] * 8, device=self.device)  # legs, torso, arms
        self.feet = robot.find_sites(["left_foot", "right_foot"])
        self.imu_body = robot.find_bodies(["imu_link"])[0]  # on the torso; the IMU site's frame is its
        self.imu_site = robot.find_sites(["imu"])[0]

        # --- set up commands and gaits ---
        self.command_scale = torch.tensor([1.5, 0.5, 1.0], device=self.device)  # m/s forward, lateral; rad/s yaw
        self.start_phase = torch.tensor([0.0, math.pi], device=self.device)  # the left foot's, the right's

        # --- set up observations ---
        joints = len(robot.joint_names)
        # the actor's noise on each observation
        noise = [0.2] * 3 + [0.05] * 3  # gyro (rad/s), up vector
        noise += [0.01] * joints + [1.5] * joints  # joint angles (rad) and velocities (rad/s)
        self.obs_noise = config.noise_level * torch.tensor(noise, device=self.device)
        # then the last action but one, the command, joint velocities, target errors, foot contacts, phases and gait
        self.obs_dims = (len(noise) + 3 * joints + 3 + 2 + 7,) * 2  # the critic's the same

        # --- initialize state ---
        self.joint_target = torch.zeros(n, joints, device=self.device)
        self.action = torch.zeros(n, joints, device=self.device)
        self.last_action = torch.zeros(n, joints, device=self.device)
        self.action_before_last = torch.zeros(n, joints, device=self.device)
        self.steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.command = torch.zeros(n, 3, device=self.device)
        self.phase = torch.zeros(n, 2, device=self.device)  # rad: each foot's
        self.gait = torch.zeros(n, 3, device=self.device)  # step rate (Hz), gait (0: walk), swing height (m)
        # `counted`: the rewards of each world's first episode since a full reset
        self.counted = torch.zeros(n, device=self.device)
        self.alive = torch.zeros(n, dtype=torch.bool, device=self.device)  # in that first episode
        self.reset(torch.ones(n, dtype=torch.bool, device=self.device))

    def curriculum(self, step: int):
        """None: the task has no curriculum."""

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move the joint targets by `action_scale` * `action`, clipped to [-1, 1], within the joint ranges; step."""
        action = action.clamp(-1.0, 1.0)
        self.action_before_last.copy_(self.last_action)
        self.last_action.copy_(self.action)
        self.action.copy_(action)
        moved = self.joint_target + self.action_scale * action
        self.joint_target.copy_(moved.clamp(self.target_low, self.target_high))
        self.robot.set_joint_position_target(self.joint_target)
        self.sim.step()
        return self.score()

    def reset(self, mask: torch.Tensor):
        """Reset the masked worlds, then refresh the kinematics the observations read."""
        self.restart(mask)
        self.sim.kinematics()

    @torch.compile
    def score(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The rewards, terminations, timeouts and each reward term's mean after a step."""
        robot = self.robot.data
        self.steps += 1

        # --- command and phases ---
        command = self.command_scale * (2 * torch.rand(self.n, 3, device=self.device) - 1)
        command = torch.where(command.abs() < 0.1, 0.0, command)  # each zeroed under 0.1
        # resampled once, after step 501
        self.command.copy_(torch.where((self.steps == 502)[:, None], command, self.command))
        phase = self.phase + 2 * math.pi * self.config.control_dt * self.gait[:, :1]
        phase = torch.remainder(phase + math.pi, 2 * math.pi) - math.pi  # wrapped to [-pi, pi)
        self.phase.copy_(torch.where((self.steps > 1)[:, None], phase, self.phase))

        # --- IMU ---
        R_world_imu = robot.body_link_mat_w[:, self.imu_body]
        imu_lin_vel_b = torch.einsum("nji,nj->ni", R_world_imu, robot.site_lin_vel_w[:, self.imu_site])
        imu_ang_vel_b = torch.einsum("nji,nj->ni", R_world_imu, robot.body_link_ang_vel_w[:, self.imu_body])

        # --- rewards ---
        # each foot's swing height: a cubic Bezier up, then down
        swing = (self.phase + math.pi) / (2 * math.pi)
        height = self.gait[:, 2:]
        up = 2 * swing
        down = 2 * swing - 1
        rising = height * (up**3 + 3 * up**2 * (1 - up))
        falling = height - height * (down**3 + 3 * down**2 * (1 - down))
        swing_height = torch.where(swing <= 0.5, rising, falling)
        joint_error = robot.joint_pos - robot.default_joint_pos
        action_rate = ((self.last_action - self.action_before_last) ** 2).sum(1)
        action_change = ((self.last_action - 2 * self.action_before_last + self.action) ** 2).sum(1)
        terms = dict(
            feet_phase=torch.exp(-((robot.site_pos_w[:, self.feet, 2] - swing_height) ** 2).sum(1) / 0.01),
            tracking_lin_vel=torch.exp(-((self.command[:, :2] - imu_lin_vel_b[:, :2]) ** 2).sum(1) / 0.5),
            tracking_ang_vel=torch.exp(-((self.command[:, 2] - imu_ang_vel_b[:, 2]) ** 2) / 0.5),
            pose=(joint_error[:, self.posed] ** 2 * self.pose_weights).sum(1),
            action_rate=action_rate + action_change,
        )
        term_values = torch.nan_to_num(torch.stack([terms[name] for name in self.names]))
        reward = ((term_values * self.weights[:, None]).sum(0) * self.config.control_dt).clamp(min=0.0)

        # --- terminations ---
        data = self.sim.data.torch
        diverged = ~(data.qpos.isfinite().all(1) & data.qvel.isfinite().all(1))  # the physics blew up
        joint_pos = robot.joint_pos
        out_of_range = ((joint_pos < self.target_low) | (joint_pos > self.target_high)).any(1)
        tilted = R_world_imu[:, 2, 2] < 0.85  # the torso's up vector tilted over 32 degrees
        terminated = out_of_range | tilted | diverged
        self.counted.copy_(reward * self.alive)
        return reward, terminated, self.steps >= self.config.episode, term_values.mean(1)

    @torch.compile
    def restart(self, mask: torch.Tensor):
        """Reset the masked worlds with masked writes only: at home and at rest, zero joint targets, a new gait."""
        robot = self.robot

        # --- robot: at home and at rest, the joint targets zero ---
        robot.write_root_link_pose_to_sim(robot.data.default_root_state[:7].expand(self.n, 7), mask)
        robot.write_root_link_velocity_to_sim(torch.zeros(self.n, 6, device=self.device), mask)
        home = robot.data.default_joint_pos
        robot.write_joint_state_to_sim(home, torch.zeros_like(home), mask)
        warmstart = self.sim.data.torch.qacc_warmstart
        warmstart.copy_(torch.where(mask[:, None], 0.0, warmstart))
        for state in (self.joint_target, self.action, self.last_action, self.action_before_last):
            state.copy_(torch.where(mask[:, None], 0.0, state))

        # --- gait: the command, step rate, swing height and phases ---
        command = self.command_scale * (2 * torch.rand(self.n, 3, device=self.device) - 1)
        command = torch.where(command.abs() < 0.1, 0.0, command)  # each zeroed under 0.1
        self.command.copy_(torch.where(mask[:, None], command, self.command))
        step_rate = 0.5 + 1.5 * torch.rand(self.n, device=self.device)  # Hz
        swing_height = 0.08 + 0.32 * torch.rand(self.n, device=self.device)  # m
        gait = torch.stack([step_rate, torch.zeros_like(step_rate), swing_height], 1)
        self.gait.copy_(torch.where(mask[:, None], gait, self.gait))
        self.phase.copy_(torch.where(mask[:, None], self.start_phase, self.phase))
        self.steps.masked_fill_(mask, 0)
        # a full reset (as `ppo.evaluate`'s) starts the counted episodes
        self.alive.copy_(mask.all() | (self.alive & ~mask))

    @torch.compile
    def obs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The actor's observations, noisy sensors then the exact state; the critic's the same."""
        robot = self.robot.data
        fresh = self.steps.eq(0)[:, None]  # after a reset: velocities and contacts are stale; at rest, both feet down
        R_world_imu = robot.body_link_mat_w[:, self.imu_body]
        gyro = torch.einsum("nji,nj->ni", R_world_imu, robot.body_link_ang_vel_w[:, self.imu_body])
        gyro = torch.where(fresh, 0.0, gyro)
        up = R_world_imu[:, :, 2]
        joint_pos = robot.joint_pos
        joint_vel = robot.joint_vel
        sensed = torch.cat([gyro, up, joint_pos - robot.default_joint_pos, joint_vel], 1)
        noisy = sensed + self.obs_noise * (2 * torch.rand_like(sensed) - 1)
        capsule_contact = (self.feet_contact.data.found > 0).view(self.n, 2, 3)
        foot_contact = (capsule_contact.any(2) | fresh).float()  # left, right
        gait = [self.phase.cos(), self.phase.sin(), self.gait]
        target_error = joint_pos - self.joint_target
        obs = torch.cat([noisy, self.last_action, self.command, joint_vel, target_error, foot_contact, *gait], 1)
        return obs, obs


if __name__ == "__main__":
    config = tyro.cli(GaitConfig)
    env = H1Gait(config)  # the full physics, where the policy is scored
    coarse_env = H1Gait(config, coarse=config.coarse_physics) if config.coarse else None
    learner = ppo.PPO(env.obs_dims, len(env.bound), ppo.PPOConfig(gamma=0.97))
    actor = learner.train(env, config.minutes, coarse_env, config.coarse)
    # each metric times the steps: its sum over each world's first episode
    metrics = dict(episode_reward=partial(torch.mul, env.counted, config.episode))
    print(ppo.evaluate(env, actor, metrics, steps=config.episode))
