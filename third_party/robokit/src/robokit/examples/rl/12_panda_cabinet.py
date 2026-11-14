# /// script
# requires-python = ">=3.12"
# dependencies = ["robokit[mjwarp,pt-cu126]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
"""Panda cabinet: a Franka Panda learns with PPO on 4096 MuJoCo Warp worlds to pull a drawer open to a random depth.

Example usage:
    uv run src/robokit/examples/rl/12_panda_cabinet.py --minutes 5
    uv run src/robokit/examples/rl/12_panda_cabinet.py --coarse 0.5  # coarse physics mixed in over the first half
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import mujoco  # pyright: ignore[reportMissingImports]
import numpy as np
import torch
import tyro
import warp as wp

from robokit.assets.robots.arms import franka_panda
from robokit.helpers.rl import Env, ppo
from robokit.sim import CoarseConfig, MjWarpSim, mjscene


@dataclass(frozen=True)
class PandaCabinetConfig:
    """The drawer task's training: the worlds and the learner's time."""

    worlds: int = 4096  # MjWarpSim worlds
    minutes: float = 5.0  # wall time to train, compilation included
    coarse: float = 0.0  # the share of PPO's time that mixes in coarse physics (`coarse_physics`)
    control_dt: float = 0.02  # s per policy action; physics steps are `mjm.opt.timestep`
    episode: int = 150  # steps per episode
    # Coarse physics (`CoarseConfig`): only the hand, drawer, cabinet face and floor collide
    coarse_physics: CoarseConfig = CoarseConfig(
        collide="robot/(left_finger_pad|right_finger_pad|hand_capsule)|handle|shelf|barrier|floor", pyramidal=True
    )


class PandaCabinet(Env):
    """Drawer opening on `config.worlds` worlds, in the full physics or the `coarse` one."""

    def __init__(self, config: PandaCabinetConfig, coarse: Optional[CoarseConfig] = None):
        n = config.worlds
        self.n = n
        self.config = config

        # --- build scene ---
        # the Panda with its hand, a drawer behind a cabinet face 52 cm ahead, 5 ms steps
        robot_spec = mjscene.build_robot_spec(franka_panda.URDF_PATH, prefix="robot/")
        for geom in robot_spec.geoms:  # only the finger pads and a hand capsule collide
            geom.contype = 0
            geom.conaffinity = 0
        for side in ("left", "right"):
            pad = robot_spec.body(f"robot/panda_{side}finger").geoms[3]
            pad.name = f"robot/{side}_finger_pad"
            pad.conaffinity = 3
        for joint, damping in zip(robot_spec.joints, [40, 40, 40, 40, 2, 2, 2, 10, 10]):  # the fingers last
            joint.damping = [damping, 0, 0]
            joint.armature = 0.1
        q_wxyz_link7_hand = [0.9238795, 0, 0, -0.3826834]  # the hand frame, 10.7 cm along link 7
        link7 = robot_spec.body("robot/panda_link7")
        q_wxyz_link7_capsule = np.zeros(4)
        mujoco.mju_mulQuat(q_wxyz_link7_capsule, q_wxyz_link7_hand, [0.7071068, 0.7071068, 0, 0])
        link7.add_geom(
            name="robot/hand_capsule",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[0.04, 0.06, 0],
            pos=[0, 0, 0.137],
            quat=q_wxyz_link7_capsule,
            contype=0,
            conaffinity=3,
            group=3,
        )
        link7.add_site(name="robot/gripper", pos=[0, 0, 0.207], quat=q_wxyz_link7_hand)
        T_world_base = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        scene_spec = mjscene.build_scene_spec([(robot_spec, T_world_base)])
        handle = scene_spec.worldbody.add_body(name="handle", pos=[0.5, 0, 0.5])  # the drawer: a handle on a slide
        handle.add_joint(
            name="drawer", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0], range=[-0.5, 0], damping=[0.1, 0, 0]
        )
        handle.add_geom(name="handle", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.08, 0.01], rgba=[0, 1, 0, 1])
        handle.add_geom(
            name="shelf",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.1, 0.2, 0.01],
            pos=[0.03, 0, 0],
            quat=[0.7071068, 0, -0.7071068, 0],
            rgba=[0, 1, 0, 0.2],
        )
        barrier = scene_spec.worldbody.add_body(name="barrier", pos=[0.52, 0, 0])  # the cabinet face, facing the robot
        barrier.add_geom(
            name="barrier",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0, 0, 0.01],
            quat=[0.7071068, 0, -0.7071068, 0],
            group=5,
        )
        for name in ("handle", "shelf", "barrier"):
            scene_spec.geom(name).contype = 2
            scene_spec.geom(name).friction = [1, 0.03, 0.003]
            scene_spec.geom(name).solref = [0.01, 1]
        scene_spec.option.iterations = 5
        scene_spec.option.ls_iterations = 8
        scene_spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
        gains = [(1000, 20, 100)] * 2 + [(750, 4, 100)] * 2 + [(300, 2, 12)] * 3  # kp, kv, force limit per arm joint
        for i, (kp, kv, force) in enumerate(gains):
            joint = scene_spec.joint(f"robot/panda_joint{i + 1}")
            actuator = scene_spec.add_actuator(
                name=f"actuator{i + 1}",
                target=joint.name,
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                ctrlrange=joint.range,
                forcerange=[-force, force],
            )
            actuator.set_to_position(kp=kp, kv=kv)
        scene_spec.add_actuator(
            name="actuator8",
            target="robot/panda_finger_joint1",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            ctrlrange=[0, 0.04],
            forcerange=[-200, 200],
            gainprm=[350] + [0] * 9,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
            biasprm=[0, -350, -10] + [0] * 7,
        )
        for part in ("left_finger_pad", "right_finger_pad", "hand_capsule"):  # each one's contact with the cabinet face
            mjscene.build_contact_sensor(
                scene_spec,
                f"barrier_{part}_found",
                (mujoco.mjtObj.mjOBJ_GEOM, "barrier"),
                (mujoco.mjtObj.mjOBJ_GEOM, f"robot/{part}"),
                reduce="mindist",
            )
        scene_spec.add_key(name="upright", qpos=[0] * 10, ctrl=[0] * 8)  # the arm upright, aiming there, drawer shut
        if coarse is not None:
            scene_spec = MjWarpSim.build_coarse_spec(scene_spec, coarse)

        # --- build simulator ---
        mjm = scene_spec.compile()
        self.mjm = mjm
        substeps = round(config.control_dt / mjm.opt.timestep)
        sim = MjWarpSim(mjm, n, substeps=substeps, key="upright", njmax=128, nconmax=12)
        self.sim = sim
        sim.set_torch_stream()
        robot = sim.build_entity("robot/")
        self.robot = robot
        self.device = robot.device
        self.barrier_contact = sim.build_contact_sensor("barrier")

        # --- set up actions ---
        # each action moves its actuator's target by up to 0.04 (rad, m) within the control range
        ctrl_range = torch.tensor(mjm.actuator_ctrlrange[robot.actuator_ids], device=self.device, dtype=torch.float32)
        self.ctrl_low = ctrl_range[:, 0]
        self.ctrl_high = ctrl_range[:, 1]
        self.bound = torch.ones(len(robot.actuator_ids), device=self.device)  # tanh-squashed actions

        # --- set up rewards ---
        weights = dict(gripper_box=4.0, box_target=8.0, no_barrier_collision=0.25, robot_target_qpos=0.3)
        self.names = list(weights)
        self.weights = torch.tensor(list(weights.values()), device=self.device)
        self.handle = mjm.body("handle").id
        self.gripper = robot.find_sites(["gripper"])[0]

        # --- set up resets ---
        spawn_range = [[-1.0, -1.0, -1.0, -2.0, -1.0, 0.0, -1.0], [1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 1.0]]  # x 30 degrees
        spawn_range = torch.tensor(spawn_range, device=self.device) * math.radians(30)
        self.spawn_low = spawn_range[0]  # rad: the arm joints' offsets from upright
        self.spawn_high = spawn_range[1]
        self.target_home = torch.tensor([0.3, 0.0, 0.5], device=self.device)  # m: the handle's target; x moves +-0.1

        # --- set up observations ---
        self.eye = torch.tensor([1.0, 0, 0, 0, 1, 0], device=self.device)  # the target orientation's first two rows
        dim = mjm.nq + mjm.nv + 3 + 6 + 6 + 3 + 3 + 6 + mjm.nu
        self.obs_dims = (dim, dim)

        # --- initialize state ---
        self.target = torch.zeros(n, 3, device=self.device)
        self.reached = torch.zeros(n, device=self.device)
        self.opened = torch.zeros(n, device=self.device)
        self.reward = torch.zeros(n, device=self.device)
        self.steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.reset(torch.ones(n, dtype=torch.bool, device=self.device))

    def curriculum(self, step: int):
        """None."""

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move the actuator targets by 0.04 * `action` (clamped to [-1, 1]) within their ranges, step, score."""
        target = self.robot.data.joint_pos_target + 0.04 * action.clamp(-1, 1)
        self.robot.set_joint_position_target(target.clamp(self.ctrl_low, self.ctrl_high))
        self.sim.step()
        return self.score()

    def reset(self, mask: torch.Tensor):
        """Reset the masked worlds to upright with the drawer shut, perturb them, then refresh the kinematics."""
        self.sim.reset(wp.from_torch(mask))
        self.restart(mask)
        self.sim.kinematics()

    @torch.compile
    def score(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-step rewards and `opened`; the episode ends as the handle leaves the 1 m box or the physics turns NaN."""
        data = self.sim.data.torch
        robot = self.robot.data
        self.steps += 1

        # --- rewards ---
        handle = data.xpos[:, self.handle]
        gripper_to_handle = (handle - robot.site_pos_w[:, self.gripper]).norm(dim=1)
        handle_to_target = (self.target - handle).norm(dim=1)
        self.reached.copy_(torch.maximum(self.reached, (gripper_to_handle < 0.012).float()))  # reached it once
        self.opened.copy_((handle_to_target < 0.05).float())
        arm_offset = robot.joint_pos[:, :7] - robot.default_joint_pos[:7]
        terms = dict(
            gripper_box=1 - torch.tanh(5 * gripper_to_handle),
            box_target=(1 - torch.tanh(5 * handle_to_target)) * self.reached,
            no_barrier_collision=1 - (self.barrier_contact.data.found > 0).any(1).float(),
            robot_target_qpos=1 - torch.tanh(arm_offset.norm(dim=1)),
        )
        term_values = torch.nan_to_num(torch.stack([terms[name] for name in self.names]))
        self.reward.copy_((term_values * self.weights[:, None]).sum(0).clamp(-1e4, 1e4))

        # --- terminations ---
        diverged = data.qpos.isnan().any(1) | data.qvel.isnan().any(1)
        out = (handle.abs() > 1.0).any(1) | (handle[:, 2] < 0.0)  # the handle left the 1 m box
        return self.reward, out | diverged, self.steps >= self.config.episode, term_values.mean(1)

    @torch.compile
    def restart(self, mask: torch.Tensor):
        """Masked writes only: the arm joints offset from upright and aiming there, a new handle target."""
        robot = self.robot
        spawn = self.spawn_low + (self.spawn_high - self.spawn_low) * torch.rand(self.n, 7, device=self.device)
        offset = torch.zeros(self.n, len(robot.joint_names), device=self.device)
        offset[:, :7] = spawn  # the arm joints', not the fingers'
        joint_pos = robot.data.default_joint_pos + offset
        robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), mask)
        joint_target = torch.where(mask[:, None], joint_pos[:, robot.actuated_ids], robot.data.joint_pos_target)
        robot.set_joint_position_target(joint_target)  # each target at its joint
        target = self.target_home.repeat(self.n, 1)
        target[:, 0] += 0.2 * torch.rand(self.n, device=self.device) - 0.1
        self.target.copy_(torch.where(mask[:, None], target, self.target))
        self.reached.masked_fill_(mask, 0.0)
        self.steps.masked_fill_(mask, 0)

    @torch.compile
    def obs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The state observation, for the actor and the critic alike."""
        data = self.sim.data.torch
        robot = self.robot.data
        handle = data.xpos[:, self.handle]
        gripper = robot.site_pos_w[:, self.gripper]
        R_world_handle = data.xmat[:, self.handle].flatten(1)
        R_world_gripper = robot.site_mat_w[:, self.gripper].flatten(1)
        obs = [
            data.qpos,
            data.qvel,
            gripper,
            R_world_gripper[:, 3:],
            R_world_handle[:, 3:],
            handle - gripper,
            self.target - handle,
            self.eye - R_world_handle[:, :6],
            robot.joint_pos_target - robot.joint_pos[:, self.robot.actuated_ids],  # each actuator's target error
        ]
        obs = torch.cat(obs, 1)
        return obs, obs


if __name__ == "__main__":
    config = tyro.cli(PandaCabinetConfig)
    env = PandaCabinet(config)  # the full physics, where the policy is scored
    coarse_env = PandaCabinet(config, coarse=config.coarse_physics) if config.coarse else None
    learner = ppo.PPO(env.obs_dims, len(env.bound), ppo.PPOConfig(gamma=0.97, entropy=0.02))
    actor = learner.train(env, config.minutes, coarse_env, config.coarse)
    # the return per episode at each step's reward, and the share of worlds with the drawer open
    metrics = dict(episode_reward=partial(torch.mul, env.reward, config.episode), opened=env.opened.clone)
    print(ppo.evaluate(env, actor, metrics, steps=config.episode))
