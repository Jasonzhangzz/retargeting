# /// script
# requires-python = ">=3.12"
# dependencies = ["robokit[mjwarp,pt-cu126]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
"""Panda pick: a Franka Panda learns with PPO on 4096 MuJoCo Warp worlds to lift a cube to a random target 20-40 cm up.

Example usage:
    uv run src/robokit/examples/rl/05_panda_pick.py --minutes 5
    uv run src/robokit/examples/rl/05_panda_pick.py --coarse 0.5  # coarse physics mixed in over the first half
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import mujoco  # pyright: ignore[reportMissingImports]
import numpy as np
import torch
import tyro

from robokit.assets.robots.arms import franka_panda
from robokit.helpers.rl import Env, ppo
from robokit.sim import CoarseConfig, MjWarpSim, mjscene


@dataclass(frozen=True)
class PandaPickConfig:
    """The pick: its control step, episode and coarse physics, the worlds and PPO's time."""

    control_dt: float = 0.02  # s per policy action; physics steps are `mjm.opt.timestep`
    episode: int = 150  # env steps per episode: 3 s
    # Coarse physics (`CoarseConfig`): only the finger pads, hand capsule, box and floor collide
    coarse_physics: CoarseConfig = CoarseConfig(
        collide="left_finger_pad|right_finger_pad|hand_capsule|box|floor", pyramidal=True
    )
    worlds: int = 4096  # MjWarpSim worlds
    minutes: float = 5.0  # wall time to train, compilation included
    coarse: float = 0.0  # the share of PPO's time that mixes in coarse physics (`coarse_physics`)


class PandaPick(Env):
    """Cube pick and lift on `config.worlds` worlds, in the full physics or the `coarse` one."""

    def __init__(self, config: PandaPickConfig = PandaPickConfig(), coarse: Optional[CoarseConfig] = None):
        n = config.worlds
        self.n = n
        self.config = config

        # --- build scene ---
        # the Panda with its hand, a 4x4x6 cm cube on the floor, 5 ms steps
        robot_spec = mjscene.build_robot_spec(franka_panda.URDF_PATH)
        for geom in robot_spec.geoms:  # only the finger pads and a hand capsule collide
            geom.contype = 0
            geom.conaffinity = 0
        for side in ("left", "right"):
            pad = robot_spec.body(f"panda_{side}finger").geoms[3]
            pad.name = f"{side}_finger_pad"
            pad.conaffinity = 3
        for joint, damping in zip(robot_spec.joints, [40, 40, 40, 40, 2, 2, 2, 10, 10]):  # the fingers last
            joint.damping = [damping, 0, 0]
            joint.armature = 0.1
        q_wxyz_link7_hand = [0.9238795, 0, 0, -0.3826834]  # the hand frame, 10.7 cm along link 7
        link7 = robot_spec.body("panda_link7")
        q_wxyz_link7_capsule = np.zeros(4)
        mujoco.mju_mulQuat(q_wxyz_link7_capsule, q_wxyz_link7_hand, [0.7071068, 0.7071068, 0, 0])
        link7.add_geom(
            name="hand_capsule",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[0.04, 0.06, 0],
            pos=[0, 0, 0.137],
            quat=q_wxyz_link7_capsule,
            contype=0,
            conaffinity=1,
            group=3,
        )
        link7.add_site(name="panda_gripper", pos=[0, 0, 0.207], quat=q_wxyz_link7_hand)  # the robot's "gripper"
        T_world_base = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        scene_spec = mjscene.build_scene_spec([(robot_spec, T_world_base)])
        box_body = scene_spec.worldbody.add_body(name="box", pos=[0.5, 0, 0.03])
        box_body.add_freejoint(name="box")
        box_body.add_geom(
            name="box",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.02, 0.02, 0.03],
            rgba=[0, 1, 0, 1],
            friction=[1, 0.03, 0.003],
            contype=2,
            solref=[0.01, 1],
        )
        scene_spec.option.iterations = 5
        scene_spec.option.ls_iterations = 8
        scene_spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
        gains = [(1000, 20, 100)] * 2 + [(750, 4, 100)] * 2 + [(300, 2, 12)] * 3  # kp, kv, force limit per arm joint
        for i, (kp, kv, force) in enumerate(gains):
            joint = scene_spec.joint(f"panda_joint{i + 1}")
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
            target="panda_finger_joint1",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            ctrlrange=[0, 0.04],
            forcerange=[-200, 200],
            gainprm=[350] + [0] * 9,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
            biasprm=[0, -350, -10] + [0] * 7,
        )
        floor = (mujoco.mjtObj.mjOBJ_GEOM, "floor")
        for geom in ("left_finger_pad", "right_finger_pad", "hand_capsule"):  # contact found, reduced to the closest
            hand_geom = (mujoco.mjtObj.mjOBJ_GEOM, geom)
            mjscene.build_contact_sensor(scene_spec, f"floor_contact_{geom}_found", hand_geom, floor, reduce="mindist")
        home = [0, 0.3, 0, -1.57079, 0, 2.0, -0.7853]  # rad: the arm joint targets
        # the fingers open (m) and the box 70 cm ahead; each actuator holds its joint's home angle
        scene_spec.add_key(name="home", qpos=home + [0.04, 0.04, 0.7, 0, 0.03, 1, 0, 0, 0], ctrl=home + [0.04])
        if coarse is not None:
            scene_spec = MjWarpSim.build_coarse_spec(scene_spec, coarse)

        # --- build simulator ---
        mjm = scene_spec.compile()
        self.mjm = mjm
        self.substeps = round(config.control_dt / mjm.opt.timestep)
        assert abs(self.substeps * mjm.opt.timestep - config.control_dt) < 1e-9, (
            "the control step must be a timestep multiple"
        )
        sim = MjWarpSim(mjm, n, substeps=self.substeps, key="home", njmax=128, nconmax=24)
        self.sim = sim
        sim.set_torch_stream()
        robot = sim.build_entity("panda_")  # every actuator is the Panda's, in joint order
        self.robot = robot
        self.device = robot.device
        self.box = sim.build_entity("box")
        self.floor_contact = sim.build_contact_sensor("floor_contact")

        # --- set up actions ---
        # each action moves its actuator's target (`ctrl`) by up to 0.04 (rad, m) within the control range
        ctrl_range = torch.tensor(mjm.actuator_ctrlrange[robot.actuator_ids], device=self.device, dtype=torch.float32)
        self.ctrl_low = ctrl_range[:, 0]
        self.ctrl_high = ctrl_range[:, 1]
        self.home_target = robot.data.default_joint_pos[robot.actuated_ids]  # the key's `ctrl`
        self.bound = torch.ones(len(robot.actuator_ids), device=self.device)  # tanh-squashed actions

        # --- set up rewards ---
        # each term's weight; the reward is per step
        weights = dict(gripper_box=4.0, box_target=8.0, no_floor_collision=0.25, robot_target_qpos=0.3)
        self.names = list(weights)
        self.weights = torch.tensor(list(weights.values()), device=self.device)
        self.gripper = robot.find_sites(["gripper"])[0]
        # the box's target orientation, upright: the identity's first two rows
        self.box_target_rows = torch.tensor([1.0, 0, 0, 0, 1, 0], device=self.device)

        # --- set up resets ---
        self.box_home = self.box.data.default_root_state[:3]  # m
        # offsets from it (m): the cube's in the plane, the target's 20 to 40 cm above
        self.spawn_low = torch.tensor(((-0.2, -0.2, 0.0), (-0.2, -0.2, 0.2)), device=self.device)
        self.spawn_high = torch.tensor(((0.2, 0.2, 0.0), (0.2, 0.2, 0.4)), device=self.device)

        # --- set up observations ---
        # the whole state, the gripper and the box, their offsets, the box's tilt and the joint targets' errors
        dims = mjm.nq + mjm.nv + 3 + 6 + 6 + 3 + 3 + 6 + mjm.nu
        self.obs_dims = (dims, dims)

        # --- initialize state ---
        self.target = torch.zeros(n, 3, device=self.device)
        self.reached = torch.zeros(n, device=self.device)  # 1 once the gripper reached the box
        self.reward = torch.zeros(n, device=self.device)  # the last step's, per world
        self.steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.reset(torch.ones(n, dtype=torch.bool, device=self.device))

    def curriculum(self, step: int):
        """None: the pick has no curriculum."""

    def at_target(self) -> torch.Tensor:
        """1 where the cube is within 5 cm of the target, else 0, per world."""
        box_pos = self.box.data.root_link_pos_w
        return ((self.target - box_pos).norm(dim=1) < 0.05).float()

    def episode_reward(self) -> torch.Tensor:
        """The per-step reward times the episode length, per world."""
        return self.reward * self.config.episode

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move the joint targets by 0.04 * `action` (clamped to [-1, 1]) within their range; step; score."""
        target = self.robot.data.joint_pos_target + 0.04 * action.clamp(-1, 1)
        self.robot.set_joint_position_target(target.clamp(self.ctrl_low, self.ctrl_high))
        self.sim.step()
        return self.score()

    def reset(self, mask: torch.Tensor):
        """Reset the masked worlds, then refresh the kinematics the observations read."""
        self.restart(mask)
        self.sim.kinematics()

    @torch.compile
    def score(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-step rewards; the episode ends when the cube leaves the 1 m box or the physics turns NaN."""
        robot = self.robot.data
        data = self.sim.data.torch
        self.steps += 1

        # --- rewards ---
        box_pos = self.box.data.root_link_pos_w
        box_rows = self.box.data.body_link_mat_w[:, 0].flatten(1)[:, :6]  # its orientation's first two rows
        gripper_distance = (box_pos - robot.site_pos_w[:, self.gripper]).norm(dim=1)
        self.reached.copy_(torch.maximum(self.reached, (gripper_distance < 0.012).float()))
        target_error = 0.9 * (self.target - box_pos).norm(dim=1) + 0.1 * (self.box_target_rows - box_rows).norm(dim=1)
        arm_error = robot.joint_pos[:, :7] - robot.default_joint_pos[:7]  # the arm's 7 joints, the fingers after
        on_floor = (self.floor_contact.data.found > 0).any(1)  # a finger pad or the hand
        terms = dict(
            gripper_box=1 - torch.tanh(5 * gripper_distance),
            box_target=(1 - torch.tanh(5 * target_error)) * self.reached,
            no_floor_collision=1 - on_floor.float(),
            robot_target_qpos=1 - torch.tanh(arm_error.norm(dim=1)),
        )
        term_values = torch.nan_to_num(torch.stack([terms[name] for name in self.names]))
        self.reward.copy_((term_values * self.weights[:, None]).sum(0).clamp(-1e4, 1e4))

        # --- terminations ---
        diverged = data.qpos.isnan().any(1) | data.qvel.isnan().any(1)
        out_of_bounds = (box_pos.abs() > 1.0).any(1) | (box_pos[:, 2] < 0.0)
        timeout = self.steps >= self.config.episode
        return self.reward, out_of_bounds | diverged, timeout, term_values.mean(1)

    @torch.compile
    def restart(self, mask: torch.Tensor):
        """Reset the masked worlds at home, the cube moved in the plane, a new target above it; masked writes only."""
        robot = self.robot
        data = self.sim.data.torch

        # --- robot: the joints and their targets at home, nothing moving ---
        home = robot.data.default_joint_pos
        robot.write_joint_state_to_sim(home, torch.zeros_like(home), mask)
        robot.set_joint_position_target(torch.where(mask[:, None], self.home_target, robot.data.joint_pos_target))
        data.qacc_warmstart.copy_(torch.where(mask[:, None], 0.0, data.qacc_warmstart))

        # --- box and target: the box at rest in its home orientation ---
        spawn_range = self.spawn_high - self.spawn_low
        spawn = self.box_home + self.spawn_low + spawn_range * torch.rand(self.n, 2, 3, device=self.device)
        q_wxyz_world_box = self.box.data.default_root_state[3:7].expand(self.n, 4)
        self.box.write_root_link_pose_to_sim(torch.cat([spawn[:, 0], q_wxyz_world_box], 1), mask)
        self.box.write_root_link_velocity_to_sim(torch.zeros(self.n, 6, device=self.device), mask)
        self.target.copy_(torch.where(mask[:, None], spawn[:, 1], self.target))

        # --- episode ---
        self.reached.masked_fill_(mask, 0.0)
        self.steps.masked_fill_(mask, 0)

    @torch.compile
    def obs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The state observation, for the actor and the critic alike."""
        robot = self.robot.data
        data = self.sim.data.torch
        box_pos = self.box.data.root_link_pos_w
        gripper_pos = robot.site_pos_w[:, self.gripper]
        R_world_box = self.box.data.body_link_mat_w[:, 0].flatten(1)
        R_world_gripper = robot.site_mat_w[:, self.gripper].flatten(1)
        state = [
            data.qpos,  # the Panda's joints, then the box's free joint
            data.qvel,
            gripper_pos,
            R_world_gripper[:, 3:],
            R_world_box[:, 3:],
            box_pos - gripper_pos,
            self.target - box_pos,
            self.box_target_rows - R_world_box[:, :6],
            robot.joint_pos_target - robot.joint_pos[:, self.robot.actuated_ids],  # the joint targets' errors
        ]
        obs = torch.cat(state, 1)
        return obs, obs


if __name__ == "__main__":
    config = tyro.cli(PandaPickConfig)
    env = PandaPick(config)  # the full physics, where the policy is scored
    coarse_env = PandaPick(config, coarse=config.coarse_physics) if config.coarse else None
    learner = ppo.PPO(env.obs_dims, len(env.bound), ppo.PPOConfig(gamma=0.97, entropy=0.02))
    actor = learner.train(env, config.minutes, coarse_env, config.coarse)
    print(ppo.evaluate(env, actor, dict(episode_reward=env.episode_reward, at_target=env.at_target)))
