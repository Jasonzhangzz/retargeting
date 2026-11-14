# /// script
# requires-python = ">=3.12"
# dependencies = ["robokit[mjwarp,pt-cu126]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
"""Leap reorient: a LEAP hand learns to turn a cube to goal after goal with PPO on 4096 MuJoCo Warp worlds.

Example usage:
    uv run src/robokit/examples/rl/06_leap_reorient.py --minutes 30
    uv run src/robokit/examples/rl/06_leap_reorient.py --coarse 0.5   # coarse physics mixed in over the first half
"""

import itertools
import math
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import mujoco  # pyright: ignore[reportMissingImports]
import torch
import tyro
import warp as wp

from robokit.assets.robots.end_effectors import leap_hand
from robokit.helpers.rl import Env, ppo
from robokit.sim import CoarseConfig, MjWarpSim, mjscene
from robokit.xform.warp.torch_wrappers import (
    quaternion_invert,
    quaternion_multiply,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
)


@dataclass(frozen=True)
class LeapReorientConfig:
    """The reorientation: its control step, episode and coarse physics, the worlds and PPO's time."""

    control_dt: float = 0.05  # s per policy action; physics steps are `mjm.opt.timestep`
    episode: int = 1000  # env steps per episode: 50 s
    # Coarse physics (`CoarseConfig`): only the cube, palm, fingertip pads, distal and middle links collide
    coarse_physics: CoarseConfig = CoarseConfig(
        collide=r"cube|robot/(palm_lower_\d+|.._tip|(dip|fingertip)(_\d)?_\d+|thumb_fingertip_\d+)", pyramidal=True
    )
    worlds: int = 4096  # MjWarpSim worlds
    minutes: float = 30.0  # wall time to train, compilation included
    coarse: float = 0.0  # the share of PPO's time that mixes in coarse physics (`coarse_physics`)


class LeapReorient(Env):
    """Cube reorientation on `config.worlds` worlds, in the full physics or the `coarse` one."""

    def __init__(self, config: LeapReorientConfig = LeapReorientConfig(), coarse: Optional[CoarseConfig] = None):
        n = config.worlds
        self.n = n
        self.config = config

        # --- build scene ---
        # the hand: URDF links with soft contacts, padded fingertips, damped joints with geared-motor limits
        hand_spec = mjscene.build_robot_spec(leap_hand.URDF_PATH_RIGHT, prefix="robot/")
        for body in hand_spec.bodies:
            for index, geom in enumerate(list(body.geoms)):
                if geom.type == mujoco.mjtGeom.mjGEOM_MESH:  # the URDF's fingertip meshes: replaced by the pads below
                    hand_spec.delete(geom)
                    continue
                geom.name = f"{body.name}_{index}"
                geom.friction = [0.2, 0.005, 0.0001]
                geom.solref = [0.02, 1.5]
                geom.solimp = [0.9, 0.99, 0.001, 0.5, 2.0]
        # fingertip pads: boxes on the distal links, named as the sites at their tips (pad pos, pad quat, tip pos)
        finger = ((-0.0013, -0.0336, 0.0145), (0.7036, 0.7096, -0.0292, -0.0252), (0, -0.05, 0.015))
        thumb = ((-0.0013, -0.0456, -0.0145), (0.7081, 0.7051, -0.0295, -0.0244), (0, -0.06, -0.015))
        distal = dict(if_tip="fingertip", mf_tip="fingertip_2", rf_tip="fingertip_3", th_tip="thumb_fingertip")
        for tip, link in distal.items():
            pad_pos, pad_quat, tip_pos = thumb if tip == "th_tip" else finger
            body = hand_spec.body(f"robot/{link}")
            body.add_geom(
                name=f"robot/{tip}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.01, 0.012, 0.016],
                pos=pad_pos,
                quat=pad_quat,
                friction=[0.7, 0.05, 0.0002],
                solref=[0.02, 1.5],
                solimp=[0.9, 0.99, 0.001, 0.5, 2.0],
            )
            body.add_site(name=f"robot/{tip}", pos=tip_pos)
        for joint in hand_spec.joints:
            joint.damping = [0.2, 0.0, 0.0]
            joint.armature = 0.00149376
            joint.frictionloss = 0.02
            joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
            joint.actfrcrange = [-0.2196, 0.2196]  # N m
        # the hand palm up, the cube above it, a goal marker, a grasp site; 10 ms Euler steps
        T_world_base = (0.0318, 0.0075, -0.0272, 0.0, 0.8184, 0.0, 0.5746)  # the URDF base, palm up, fingers along +x
        scene_spec = mjscene.build_scene_spec([(hand_spec, T_world_base)], floor=False, timestep=0.01)
        cube_body = scene_spec.worldbody.add_body(name="cube", pos=[0.11, 0.0, 0.1])
        cube_body.add_freejoint(name="cube")
        cube_body.add_geom(
            name="cube",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.035] * 3,
            mass=0.108,
            rgba=[0.8, 0.3, 0.2, 1],
            friction=[0.3, 0.05, 0.0001],
        )
        goal_body = scene_spec.worldbody.add_body(name="goal", mocap=True, pos=[0.325, 0.17, 0.0475])
        goal_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.035] * 3, contype=0, conaffinity=0, rgba=[0.2, 0.6, 0.9, 0.3]
        )
        scene_spec.worldbody.add_site(name="grasp_site", pos=[0.11, 0.0, 0.03])
        for joint in hand_spec.joints:
            actuator = scene_spec.add_actuator(name=joint.name, target=joint.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            actuator.set_to_position(kp=3.0, inheritrange=True)
        # the palm touches only the fingertips, the finger bases never touch each other
        bases = r"robot/(mcp_joint|pip|dip|thumb_temp_base|thumb_pip|thumb_dip)(_\d)?"
        for body in hand_spec.bodies:
            if re.fullmatch(bases, body.name):
                scene_spec.add_exclude(bodyname1="robot/palm_lower", bodyname2=body.name)
        knuckles = ["robot/mcp_joint", "robot/mcp_joint_2", "robot/mcp_joint_3", "robot/thumb_temp_base"]
        for first, second in itertools.combinations(knuckles, 2):
            scene_spec.add_exclude(bodyname1=first, bodyname2=second)
        home = [0.8, 0.0, 0.8, 0.8] * 3 + [0.8, 0.8, 0.8, 0.0]  # rad: index, middle, ring (mcp, rot, pip, dip), thumb
        cube_home = [0.1, 0.0, 0.05, 0.810967, -0.00262895, -0.585086, -0.000254303]
        scene_spec.add_key(
            name="home", qpos=home + cube_home, ctrl=home, mpos=[0.25, 0.16, 0.0], mquat=[1.0, 0.0, 0.0, 0.0]
        )
        scene_spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
        scene_spec.option.iterations = 5
        scene_spec.option.ls_iterations = 8
        scene_spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
        scene_spec.option.ccd_iterations = 10
        if coarse is not None:
            scene_spec = MjWarpSim.build_coarse_spec(scene_spec, coarse)

        # --- build simulator ---
        mjm = scene_spec.compile()
        self.mjm = mjm
        self.substeps = round(config.control_dt / mjm.opt.timestep)
        assert abs(self.substeps * mjm.opt.timestep - config.control_dt) < 1e-9, (
            "the control step must be a timestep multiple"
        )
        sim = MjWarpSim(mjm, n, substeps=self.substeps, key="home", njmax=200, nconmax=30)  # per world
        self.sim = sim
        sim.set_torch_stream()
        robot = sim.build_entity("robot/")  # the hand, no root; every actuator is the hand's, in joint order
        self.robot = robot
        self.device = robot.device
        self.cube = sim.build_entity("cube")
        self.grasp_site = mjm.site("grasp_site").id  # on the world, above the palm
        self.q_wxyz_world_goal = sim.data.torch.mocap_quat[:, 0]  # a view: the goal marker's orientation

        # --- set up actions ---
        # each action moves its joint's target (`ctrl`) by up to 0.5 rad within its range
        joints = len(robot.joint_names)
        ctrl_range = torch.tensor(mjm.actuator_ctrlrange[robot.actuator_ids], device=self.device, dtype=torch.float32)
        self.ctrl_low = ctrl_range[:, 0]
        self.ctrl_high = ctrl_range[:, 1]
        self.bound = torch.ones(len(robot.actuator_ids), device=self.device)  # tanh policies act in [-1, 1]

        # --- randomize ---
        # fingertip friction (one draw for all four), the cube's inertia and center of mass, the hand's joints, masses
        fingertips = ["th_tip", "if_tip", "mf_tip", "rf_tip"]
        sim.set_random("geom_friction", ([mjm.geom(f"robot/{tip}").id for tip in fingertips], 0), 0.5, 1.0)
        cube_inertia_scale = 0.8 + 0.4 * torch.rand(n, 1, device=self.device)
        wp.to_torch(sim.model.body_inertia)[:, self.cube.root_body] *= cube_inertia_scale
        bodies = len(robot.body_ids)
        wp.to_torch(sim.model.body_mass)[:, robot.body_ids] *= 0.9 + 0.2 * torch.rand(n, bodies, device=self.device)
        joint_dofs = robot.data.joint_v_adr
        wp.to_torch(sim.model.dof_frictionloss)[:, joint_dofs] *= 0.5 + 1.5 * torch.rand(n, joints, device=self.device)
        wp.to_torch(sim.model.dof_armature)[:, joint_dofs] *= 1.0 + 0.05 * torch.rand(n, joints, device=self.device)
        wp.to_torch(sim.model.dof_damping)[:, joint_dofs] *= 0.8 + 0.4 * torch.rand(n, joints, device=self.device)
        gain = wp.to_torch(sim.model.actuator_gainprm)
        bias = wp.to_torch(sim.model.actuator_biasprm)
        stiffness = gain[:, :, 0] * (0.8 + 0.4 * torch.rand(n, joints, device=self.device))
        gain[:, :, 0] = stiffness
        bias[:, :, 1] = -stiffness
        sim.set_random("body_ipos", (self.cube.root_body,), [-5e-3] * 3, [5e-3] * 3, add=True)  # m; then set_const

        # --- set up rewards ---
        # each term's weight; terms are per second, success pays 100 once
        weights = dict(
            orientation=5.0,
            position=0.5,
            termination=-100.0,
            hand_pose=-0.5,
            action_rate=-1e-3,
            energy=-1e-3,
            success=100.0 / config.control_dt,
        )
        self.names = list(weights)
        self.weights = torch.tensor(list(weights.values()), device=self.device)

        # --- set up observations ---
        self.fingertips = robot.find_sites(fingertips)
        self.obs_dims = (3 * joints + 9, 3 * joints + 9 + 2 * joints + 12 + 15)  # the critic adds the exact state

        # --- initialize state ---
        self.action = torch.zeros(n, joints, device=self.device)
        self.last_action = torch.zeros(n, joints, device=self.device)
        self.action_before_last = torch.zeros(n, joints, device=self.device)
        self.goal_spin = torch.zeros(n, 3, device=self.device)  # rad/s: the goal's, set on each success
        self.q_wxyz_world_shown_goal = torch.zeros(n, 4, device=self.device)  # the goal the observations show
        self.tally = torch.zeros(2, n, device=self.device)  # first-episode metrics (see `score`)
        self.steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.in_first_episode = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.reset(torch.ones(n, dtype=torch.bool, device=self.device))

    def curriculum(self, step: int):
        """None: the reorientation has no curriculum."""

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move each joint target by 0.5 rad * `action` (in [-1, 1]) within its range; step; score."""
        action = action.clamp(-1.0, 1.0)
        self.action_before_last.copy_(self.last_action)
        self.last_action.copy_(self.action)
        self.action.copy_(action)
        target = self.robot.data.joint_pos_target + 0.5 * action
        self.robot.set_joint_position_target(target.clamp(self.ctrl_low, self.ctrl_high))
        self.sim.step()
        return self.score()

    def reset(self, mask: torch.Tensor):
        """Reset the masked worlds, then refresh the kinematics the observations read."""
        self.restart(mask)
        self.sim.kinematics()

    @torch.compile
    def score(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rewards and terminations from the (one substep stale) cube pose; a reached goal spins on to a new one."""
        robot = self.robot.data
        data = self.sim.data.torch
        self.steps += 1

        # --- rewards ---
        cube_pos = self.cube.data.root_link_pos_w
        q_wxyz_error = quaternion_multiply(
            data.xquat[:, self.cube.root_body], quaternion_invert(self.q_wxyz_world_goal)
        )
        angle_error = quaternion_to_axis_angle(q_wxyz_error).norm(dim=1)  # rad
        success = angle_error < 0.1
        palm_distance = (data.site_xpos[:, self.grasp_site] - cube_pos).norm(dim=1)
        fell = (cube_pos[:, 2] < -0.05) | ~data.qpos.isfinite().all(1) | ~data.qvel.isfinite().all(1)
        action_rate = self.action - self.last_action
        action_accel = self.action - 2 * self.last_action + self.action_before_last
        terms = dict(
            orientation=1 - 0.9 * (angle_error - 0.2).clamp(min=0.0) / math.pi,  # linear tolerances
            position=(1 - 0.9 * (palm_distance - 0.02).clamp(min=0.0) / 0.05).clamp(min=0.0),
            termination=fell.float(),
            hand_pose=((robot.joint_pos - robot.default_joint_pos) ** 2).sum(1),
            action_rate=(action_rate**2).sum(1) + (action_accel**2).sum(1),
            energy=(robot.joint_vel.abs() * data.actuator_force.abs()).sum(1),
            success=success.float(),
        )
        term_values = torch.nan_to_num(torch.stack([terms[name] for name in self.names]))
        reward = (term_values * self.weights[:, None] * self.config.control_dt).sum(0)

        # --- goals: the observations show this one; a reached goal spins on at 1 to 5 rad/s, 20% slower each step ---
        self.q_wxyz_world_shown_goal.copy_(self.q_wxyz_world_goal)
        spin = 1.0 + 4.0 * torch.rand(self.n, 3, device=self.device)  # rad/s
        self.goal_spin.copy_(torch.where(success[:, None], spin, 0.8 * self.goal_spin))
        speed = self.goal_spin.norm(dim=1, keepdim=True)
        half_angle = speed * self.config.control_dt  # half of the 2-step rotation integrated per step
        q_wxyz_turn = torch.cat([half_angle.cos(), self.goal_spin / speed.clamp(min=1e-9) * half_angle.sin()], 1)
        q_wxyz_world_goal = quaternion_multiply(self.q_wxyz_world_goal, q_wxyz_turn)
        self.q_wxyz_world_goal.copy_(q_wxyz_world_goal / q_wxyz_world_goal.norm(dim=1, keepdim=True))

        # --- terminations ---
        timeout = self.steps >= self.config.episode
        # the first episode's per-step shares of the reward and successes: their means over 1000 steps, its totals
        self.tally.copy_(self.config.episode * torch.stack([reward, success.float()]) * self.in_first_episode)
        self.in_first_episode &= ~(fell | timeout)
        return reward, fell, timeout, term_values.mean(1)

    @torch.compile
    def restart(self, mask: torch.Tensor):
        """Masked writes only: the hand near home, cube and goal in random orientations; a full reset starts a tally."""
        robot = self.robot
        data = self.sim.data.torch

        # --- uniform random rotations (Shoemake's): the goal's and the cube's ---
        draw = torch.rand(self.n, 2, 3, device=self.device)
        radius_1 = (1 - draw[..., 0]).sqrt()
        radius_2 = draw[..., 0].sqrt()
        angle_1 = 2 * math.pi * draw[..., 1]
        angle_2 = 2 * math.pi * draw[..., 2]
        q_wxyz_random = torch.stack(
            [radius_1 * angle_1.sin(), radius_1 * angle_1.cos(), radius_2 * angle_2.sin(), radius_2 * angle_2.cos()], -1
        )

        # --- hand: the joints near home, their targets on them, nothing moving ---
        noise = 0.1 * torch.randn(self.n, len(robot.joint_ids), device=self.device)  # rad
        hand = (robot.data.default_joint_pos + noise).clamp(self.ctrl_low, self.ctrl_high)
        robot.write_joint_state_to_sim(hand, torch.zeros_like(hand), mask)
        robot.set_joint_position_target(torch.where(mask[:, None], hand, robot.data.joint_pos_target))
        data.qacc_warmstart.copy_(torch.where(mask[:, None], 0.0, data.qacc_warmstart))

        # --- cube: within 1 cm of its home position, turned at random, at rest ---
        cube_pos = self.cube.data.default_root_state[:3] + (-0.01 + 0.02 * torch.rand(self.n, 3, device=self.device))
        self.cube.write_root_link_pose_to_sim(torch.cat([cube_pos, q_wxyz_random[:, 1]], 1), mask)
        self.cube.write_root_link_velocity_to_sim(torch.zeros(self.n, 6, device=self.device), mask)

        # --- goal and episode ---
        for goal in (self.q_wxyz_world_goal, self.q_wxyz_world_shown_goal):
            goal.copy_(torch.where(mask[:, None], q_wxyz_random[:, 0], goal))
        for state in (self.goal_spin, self.action, self.last_action, self.action_before_last):
            state.copy_(torch.where(mask[:, None], 0.0, state))
        self.steps.masked_fill_(mask, 0)
        self.in_first_episode |= mask.all()

    @torch.compile
    def obs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The actor's noisy joints, target errors, cube pose error and last action; the critic's adds exact state."""
        robot = self.robot.data
        data = self.sim.data.torch
        palm_pos = data.site_xpos[:, self.grasp_site]
        cube_pos = self.cube.data.root_link_pos_w
        q_wxyz_world_cube = data.xquat[:, self.cube.root_body]
        joint_pos = robot.joint_pos
        noisy_joint_pos = joint_pos + 0.05 * (2 * torch.rand_like(joint_pos) - 1)  # rad
        noisy_cube_pos = cube_pos + 0.02 * (2 * torch.rand_like(cube_pos) - 1)  # m
        noisy_q_wxyz = q_wxyz_world_cube + 0.1 * torch.randn_like(q_wxyz_world_cube)
        noisy_q_wxyz = noisy_q_wxyz / noisy_q_wxyz.norm(dim=1, keepdim=True)
        goal_inverse = quaternion_invert(self.q_wxyz_world_shown_goal)
        noisy_rotation_error = quaternion_to_matrix(quaternion_multiply(noisy_q_wxyz, goal_inverse))[:, 1:].flatten(1)
        actor = torch.cat(
            [
                noisy_joint_pos,
                noisy_joint_pos - robot.joint_pos_target,  # from the joint targets
                palm_pos - noisy_cube_pos,
                noisy_rotation_error,
                self.last_action,
            ],
            1,
        )
        fingertips = robot.site_pos_w[:, self.fingertips] - palm_pos[:, None]  # the grasp site's frame is the world's
        critic = [
            actor,
            joint_pos,
            robot.joint_vel,
            fingertips.flatten(1),
            palm_pos - cube_pos,
            quaternion_to_matrix(quaternion_multiply(q_wxyz_world_cube, goal_inverse))[:, 1:].flatten(1),
            self.cube.data.root_link_vel_w,  # linear, then angular
        ]
        return actor, torch.cat(critic, 1)


if __name__ == "__main__":
    config = tyro.cli(LeapReorientConfig)
    env = LeapReorient(config)  # the full physics, where the policy is scored
    coarse_env = LeapReorient(config, coarse=config.coarse_physics) if config.coarse else None
    ppo_config = ppo.PPOConfig(horizon=40, epochs=4, minibatches=32)  # gamma, entropy, MLPs: the defaults
    actor = ppo.PPO(env.obs_dims, len(env.bound), ppo_config).train(env, config.minutes, coarse_env, config.coarse)
    # first episodes span `evaluate`'s 1000 steps: the mean of `tally`'s per-step shares is the episode total
    metrics = dict(episode_reward=env.tally[0].clone, successes=env.tally[1].clone)
    print(ppo.evaluate(env, actor, metrics))
