# /// script
# requires-python = ">=3.10"
# dependencies = ["robokit[mjwarp,pt-cu126]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
"""Lift: a YAM arm learns to carry a cube to random targets with PPO on 4096 MuJoCo Warp worlds, then is scored.

Example usage:
    uv run src/robokit/examples/rl/03_lift.py --minutes 3
    uv run src/robokit/examples/rl/03_lift.py --coarse 0  # full physics throughout
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple

import mujoco  # pyright: ignore[reportMissingImports]
import mujoco_warp as mjw  # pyright: ignore[reportMissingImports]
import torch
import tyro

from robokit.assets.robots.arms import yam
from robokit.helpers.rl import Env, ppo
from robokit.sim import CoarseConfig, MjWarpSim, mjscene


@dataclass(frozen=True)
class LiftConfig:
    """The cube lift: its control step, its coarse physics, the worlds and PPO's time."""

    control_dt: float = 0.02  # s per policy action; physics steps are `mjm.opt.timestep`
    # Coarse physics (`CoarseConfig`): only finger pads, cube and floor collide, 20 ms steps
    coarse_physics: CoarseConfig = CoarseConfig(
        collide=r"robot/[lr]f_down\d+_collision|cube|floor",
        pyramidal=True,
        timestep=0.02,
        iterations=5,
        ls_iterations=10,
    )
    worlds: int = 4096  # MjWarpSim worlds
    minutes: float = 3.0  # wall time to train, compilation included
    coarse: float = 0.5  # the share of PPO's time that mixes in coarse physics (`coarse_physics`), less and less often


class Lift(Env):
    """The cube lift on `config.worlds` worlds, in the full physics or the `coarse` one."""

    def __init__(self, config: LiftConfig = LiftConfig(), coarse: Optional[CoarseConfig] = None):
        n = config.worlds
        self.n = n
        self.config = config

        # --- build scene ---
        robot_spec = mjscene.build_robot_spec(yam.MJCF_PATH, prefix="robot/")
        T_world_base = (0.0, 0.0, 0.01, 1.0, 0.0, 0.0, 0.0)  # the arm 1 cm above the floor
        scene_spec = mjscene.build_scene_spec([(robot_spec, T_world_base)])
        scene_spec.option.impratio = 10.0
        scene_spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        scene_spec.option.iterations = 10
        scene_spec.option.ls_iterations = 20
        scene_spec.option.ccd_iterations = 50
        cube = scene_spec.worldbody.add_body(name="cube")  # a 4 cm cube
        cube.add_freejoint(name="cube")
        cube.add_geom(name="cube", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02] * 3, mass=0.05, rgba=[0.8, 0.2, 0.2, 1])
        # the initial state: the arm at home (rad; the fingers in m), the cube at the origin
        home = [0.0, 1.047, 1.05, -0.9, 0.0, 0.0, 0.01875, -0.01875]
        scene_spec.add_key(name="init_state", qpos=[*home, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        # the contact force between link 6's subtree, the gripper, and the floor
        link_6 = (mujoco.mjtObj.mjOBJ_XBODY, "robot/link_6")
        floor = (mujoco.mjtObj.mjOBJ_GEOM, "floor")
        mjscene.build_contact_sensor(scene_spec, "floor_contact_link_6_force", link_6, floor, data=("force",))
        if coarse is not None:
            scene_spec = MjWarpSim.build_coarse_spec(scene_spec, coarse)

        # --- build simulator ---
        mjm = scene_spec.compile()
        self.mjm = mjm
        self.substeps = round(config.control_dt / mjm.opt.timestep)
        assert abs(self.substeps * mjm.opt.timestep - config.control_dt) < 1e-9, (
            "the control step must be a timestep multiple"
        )
        sim = MjWarpSim(mjm, n, substeps=self.substeps, key="init_state", njmax=600, nconmax=55)
        self.sim = sim
        sim.opt.contact_sensor_maxmatch = 64  # contacts a contact sensor can match
        sim.opt.warn_overflow &= ~int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)  # capped on purpose
        sim.set_torch_stream()
        robot = sim.build_entity("robot/", soft_joint_pos_limit_factor=0.9)  # 90% of each joint's range is penalty-free
        self.robot = robot
        self.device = robot.device
        self.cube = sim.build_entity("cube")
        self.floor_contact = sim.build_contact_sensor("floor_contact")

        # --- set up actions ---
        # a joint target is its home angle + a quarter of its actuator's effort / stiffness per unit of action
        effort = mjm.actuator_forcerange[robot.actuator_ids, 1] / mjm.actuator_gainprm[robot.actuator_ids, 0]  # rad
        self.action_scale = torch.tensor(0.25 * effort, device=self.device, dtype=torch.float32)
        self.action_offset = robot.data.default_joint_pos[robot.actuated_ids]
        lo, hi = robot.data.soft_joint_pos_limits[robot.actuated_ids].T
        self.bound = torch.maximum(self.action_offset - lo, hi - self.action_offset) / self.action_scale  # for FastSAC

        # --- randomize ---
        # each fingertip pad's friction: sliding uniform, torsional and rolling log-uniform
        pattern = r"robot/[lr]f_down(6|7|8|9|10|11)_collision"
        pads = [g for g in range(mjm.ngeom) if re.fullmatch(pattern, mjm.geom(g).name)]
        for axis, (low, high) in enumerate(((0.3, 1.5), (1e-4, 2e-2), (1e-5, 5e-3))):
            sim.set_random("geom_friction", (pads, axis), [low] * len(pads), [high] * len(pads), log=axis > 0)

        # --- set up rewards ---
        # each term's weight; terms are per second; `curriculum` raises the joint velocity penalty
        weights = dict(lift=1.0, lift_precise=1.0, action_rate_l2=-0.01, joint_pos_limits=-10.0, joint_vel_hinge=-0.01)
        self.names = list(weights)
        self.weights = torch.tensor(list(weights.values()), device=self.device)
        self.joint_vel_weight = None  # the weight `curriculum` last set
        self.grasp_site = robot.find_sites(["grasp_site"])[0]

        # --- set up targets ---
        target_range = torch.tensor(((0.3, 0.5), (-0.2, 0.2), (0.2, 0.4)), device=self.device)
        self.target_low = target_range[:, 0]  # m: the target's x, y, z
        self.target_high = target_range[:, 1]
        cube_range = torch.tensor(((0.2, 0.4), (-0.2, 0.2), (0.02, 0.05), (-3.14, 3.14)), device=self.device)
        self.cube_low = cube_range[:, 0]  # m, m, m, rad: the cube's x, y, z and yaw
        self.cube_high = cube_range[:, 1]

        # --- set up observations ---
        joints = len(robot.joint_names)
        actions = len(robot.actuator_ids)
        # the actor's noise on each observation
        noise = [0.01] * joints + [1.5] * joints  # joint angles (rad) and velocities (rad/s)
        noise += [0.01] * 6 + [0.0] * actions  # the cube from the gripper and to the target (m), the last action
        self.obs_noise = torch.tensor(noise, device=self.device)
        self.obs_dims = (len(noise), len(noise))

        # --- initialize state ---
        self.action = torch.zeros(n, actions, device=self.device)
        self.last_action = torch.zeros(n, actions, device=self.device)
        self.steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.target = torch.zeros(n, 3, device=self.device)
        self.target_time_left = torch.zeros(n, device=self.device)
        self.reset(torch.ones(n, dtype=torch.bool, device=self.device))

    def curriculum(self, step: int):
        """Reward curriculum: the joint velocity penalty grows after 500 and 1000 iterations of 24 env steps."""
        weight = -0.01
        if step >= 500 * 24:
            weight = -0.1
        if step >= 1000 * 24:
            weight = -1.0
        if weight != self.joint_vel_weight:
            self.joint_vel_weight = weight
            self.weights[self.names.index("joint_vel_hinge")] = weight

    def at_goal(self) -> torch.Tensor:
        """1 where the cube is within 5 cm of the target, else 0, per world."""
        cube_pos = self.cube.data.root_link_pos_w
        return ((self.target - cube_pos).norm(dim=1) < 0.05).float()

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Target the home angles + scale * `action`; step; score."""
        self.last_action.copy_(self.action)
        self.action.copy_(action)
        self.robot.set_joint_position_target(action * self.action_scale + self.action_offset)
        self.sim.step()
        return self.score()

    def reset(self, mask: torch.Tensor):
        """Reset the masked worlds, advance the targets, then refresh the kinematics the observations read."""
        self.restart(mask)
        self.sim.kinematics()

    @torch.compile
    def score(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The rewards, terminations, timeouts and each reward term's mean after a step."""
        robot = self.robot.data
        self.steps += 1

        # --- rewards ---
        gripper_pos = robot.site_pos_w[:, self.grasp_site]
        cube_pos = self.cube.data.root_link_pos_w
        target_error = ((self.target - cube_pos) ** 2).sum(1)  # m^2
        reach = torch.exp(-((gripper_pos - cube_pos) ** 2).sum(1) / 0.2**2)  # the gripper at the cube
        low, high = robot.soft_joint_pos_limits.T
        out_of_limits = (low - robot.joint_pos).clamp(min=0.0) + (robot.joint_pos - high).clamp(min=0.0)
        terms = dict(
            lift=reach * (1 + torch.exp(-target_error / 0.3**2)),
            lift_precise=torch.exp(-target_error / 0.05**2),
            action_rate_l2=((self.action - self.last_action) ** 2).sum(1),
            joint_pos_limits=out_of_limits.sum(1),
            joint_vel_hinge=((robot.joint_vel.abs() - 0.5).clamp(min=0.0) ** 2).sum(1),
        )
        term_values = torch.stack([terms[name] for name in self.names])
        reward = (torch.nan_to_num(term_values * self.weights[:, None]) * self.config.control_dt).sum(0)

        # --- terminations ---
        hit_floor = (self.floor_contact.data.force_history.norm(dim=-1) > 10.0).any(-1).any(0)  # over 10 N in a substep
        timeout = self.steps >= 1000  # 20 s episodes
        return reward, hit_floor, timeout, term_values.mean(1)

    @torch.compile
    def restart(self, mask: torch.Tensor):
        """Reset the masked worlds near home and resample targets, with masked writes only."""
        robot = self.robot

        # --- robot: the joints at home, nothing moving ---
        home = robot.data.default_joint_pos.clamp(*robot.data.soft_joint_pos_limits.T)
        robot.write_joint_state_to_sim(home, torch.zeros_like(home), mask)
        warmstart = self.sim.data.torch.qacc_warmstart
        warmstart.copy_(torch.where(mask[:, None], 0.0, warmstart))

        # --- episode: steps and actions ---
        self.steps.masked_fill_(mask, 0)
        for state in (self.action, self.last_action):
            state.copy_(torch.where(mask[:, None], 0.0, state))

        # --- targets: new ones on reset and every 8 to 12 s, each with the cube respawned at rest ---
        self.target_time_left.copy_(
            torch.where(mask, self.target_time_left, self.target_time_left - self.config.control_dt)
        )
        new = mask | (self.target_time_left <= 0)
        target = self.target_low + (self.target_high - self.target_low) * torch.rand(self.n, 3, device=self.device)
        self.target.copy_(torch.where(new[:, None], target, self.target))
        spawn = self.cube_low + (self.cube_high - self.cube_low) * torch.rand(self.n, 4, device=self.device)
        yaw = spawn[:, 3]
        zero = torch.zeros_like(yaw)
        q_wxyz_world_cube = torch.stack([(yaw / 2).cos(), zero, zero, (yaw / 2).sin()], 1)
        self.cube.write_root_link_pose_to_sim(torch.cat([spawn[:, :3], q_wxyz_world_cube], 1), new)
        self.cube.write_root_link_velocity_to_sim(torch.zeros(self.n, 6, device=self.device), new)
        target_time = 8.0 + 4.0 * torch.rand(self.n, device=self.device)  # s
        self.target_time_left.copy_(torch.where(new, target_time, self.target_time_left))

    @torch.compile
    def obs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The actor's (noisy) and the critic's observations: joints, the cube's offsets, the last action."""
        robot = self.robot.data
        cube_pos = self.cube.data.root_link_pos_w
        state = [
            robot.joint_pos - robot.default_joint_pos,
            robot.joint_vel,
            cube_pos - robot.site_pos_w[:, self.grasp_site],
            self.target - cube_pos,
            self.action,
        ]
        critic = torch.cat(state, 1)
        noise = self.obs_noise * (2 * torch.rand(self.n, len(self.obs_noise), device=self.device) - 1)
        return critic + noise, critic


if __name__ == "__main__":
    config = tyro.cli(LiftConfig)
    env = Lift(config)  # the full physics, where the policy is scored
    coarse_env = Lift(config, coarse=config.coarse_physics) if config.coarse else None
    learner = ppo.PPO(env.obs_dims, len(env.bound), ppo.PPOConfig(entropy=0.005))
    actor = learner.train(env, config.minutes, coarse_env, config.coarse)
    print(ppo.evaluate(env, actor, dict(at_goal=env.at_goal)))
