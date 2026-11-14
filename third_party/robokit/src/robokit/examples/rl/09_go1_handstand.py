# /// script
# requires-python = ">=3.12"
# dependencies = ["robokit[mjwarp,pt-cu126]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
"""Go1 handstand: a Unitree Go1 learns to balance on its front feet on 4096 MuJoCo Warp worlds.

Example usage:
    uv run src/robokit/examples/rl/09_go1_handstand.py --minutes 5
    uv run src/robokit/examples/rl/09_go1_handstand.py --coarse 0.5    # coarse physics mixed in over the first half
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import mujoco  # pyright: ignore[reportMissingImports]
import torch
import tyro

from robokit.assets.robots.quadrupeds import unitree_go1
from robokit.helpers.rl import Env, ppo
from robokit.sim import CoarseConfig, MjWarpSim, mjscene


@dataclass(frozen=True)
class HandstandConfig:
    """The handstand task and its training: the control rate, the trunk's height, the worlds and the learner's time."""

    action_scale: float = 0.3  # rad: the joint targets' change per step per unit of action
    height: float = 0.55  # m: the trunk height the height reward grows up to
    worlds: int = 4096  # MjWarpSim worlds
    minutes: float = 5.0  # wall time to train, compilation included
    coarse: float = 0.0  # the share of PPO's time that mixes in coarse physics (`coarse_physics`)
    control_dt: float = 0.02  # s per policy action; physics steps are `mjm.opt.timestep`
    episode: int = 500  # env steps per episode (10 s)
    legs: Tuple[str, ...] = ("FR", "FL", "RR", "RL")  # leg order
    # The front legs' hip, thigh and calf geoms: on the floor, they end an episode
    front: Tuple[str, ...] = tuple(
        f"{leg}_{part}_collision{k}"
        for leg in ("FL", "FR")
        for part, k in (("calf", 1), ("calf", 2), ("thigh", 1), ("thigh", 2), ("thigh", 3), ("hip", ""))
    )
    # Coarse physics, off by default (`CoarseConfig`): only the floor, feet, trunk and front legs
    coarse_physics: CoarseConfig = CoarseConfig(
        collide="floor|robot/(FR|FL|RR|RL)_foot_collision|robot/trunk_collision|robot/F[LR]_.*", pyramidal=True
    )


class Go1Handstand(Env):
    """Go1 handstand on flat ground, in the full physics or the `coarse` one."""

    def __init__(self, config: HandstandConfig = HandstandConfig(), coarse: Optional[CoarseConfig] = None):
        n = config.worlds
        self.n = n
        self.config = config
        self.action_scale = config.action_scale
        self.height = config.height

        # --- build scene ---
        # the Go1 on a floor: 4 ms Euler steps, 1 solver iteration
        robot_spec = mjscene.build_robot_spec(unitree_go1.MJCF_PATH, prefix="robot/")
        T_world_base = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        scene_spec = mjscene.build_scene_spec([(robot_spec, T_world_base)], timestep=0.004)
        scene_spec.geom("floor").friction = (0.6, 0.005, 1e-4)
        scene_spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
        scene_spec.option.iterations = 1
        scene_spec.option.ls_iterations = 5
        scene_spec.option.ccd_iterations = 20
        scene_spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
        for (
            geom
        ) in scene_spec.geoms:  # feet and floor: 3D contacts, the floor's friction winning; the rest frictionless
            geom.condim = 3 if geom.name == "floor" or geom.name.endswith("_foot_collision") else 1
            geom.priority = 1 if geom.name == "floor" else 0
        for actuator in list(scene_spec.actuators):  # replaced by position actuators in joint order
            scene_spec.delete(actuator)
        for leg in config.legs:
            for part, frictionloss, force in (("hip", 0.3, 23.7), ("thigh", 0.3, 23.7), ("calf", 1.0, 35.55)):
                joint = scene_spec.joint(f"robot/{leg}_{part}_joint")
                joint.damping = [0.5, 0.0, 0.0]
                joint.armature = 0.005
                joint.frictionloss = frictionloss
                actuator = scene_spec.add_actuator(
                    name=joint.name, target=joint.name, trntype=mujoco.mjtTrn.mjTRN_JOINT
                )
                actuator.set_to_position(kp=35.0)
                actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
                actuator.forcerange = [-force, force]
                actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
                actuator.ctrlrange = joint.range
        scene_spec.add_sensor(
            name="accelerometer",
            type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname="robot/imu",
        )
        floor = (mujoco.mjtObj.mjOBJ_GEOM, "floor")
        for leg in config.legs:  # each foot's contact flag against the floor
            foot = (mujoco.mjtObj.mjOBJ_GEOM, f"robot/{leg}_foot_collision")
            mjscene.build_contact_sensor(scene_spec, f"foot_contact_{leg}_found", foot, floor, reduce="mindist")
        for part in config.front:  # each front-leg geom's contact flag against the floor
            geom = (mujoco.mjtObj.mjOBJ_GEOM, f"robot/{part}")
            mjscene.build_contact_sensor(scene_spec, f"leg_contact_{part}_found", geom, floor, reduce="mindist")
        home = [0.1, 0.9, -1.8, -0.1, 0.9, -1.8] * 2  # standing joint angles (rad), in leg order
        scene_spec.add_key(name="home", qpos=[0.0, 0.0, 0.278, 1.0, 0.0, 0.0, 0.0, *home], ctrl=home)
        if coarse is not None:
            scene_spec = MjWarpSim.build_coarse_spec(scene_spec, coarse)

        # --- build simulator ---
        mjm = scene_spec.compile()
        self.mjm = mjm
        self.substeps = round(config.control_dt / mjm.opt.timestep)
        sim = MjWarpSim(mjm, n, substeps=self.substeps, key="home", njmax=200, nconmax=30)
        self.sim = sim
        sim.set_torch_stream()
        robot = sim.build_entity("robot/", soft_joint_pos_limit_factor=0.9)  # 90% of each joint's range is penalty-free
        self.robot = robot
        self.device = robot.device
        self.feet_contact = sim.build_contact_sensor("foot_contact")  # touching the floor costs
        self.leg_contact = sim.build_contact_sensor(
            "leg_contact"
        )  # a front leg's hip, thigh or calf on the floor ends it

        # --- randomize ---
        sim.set_random("geom_friction", (mjm.geom("floor").id, 0), 0.4, 1.0)
        dofs = robot.data.joint_v_adr.tolist()
        frictionloss = mjm.dof_frictionloss[dofs]
        sim.set_random("dof_frictionloss", (dofs,), 0.9 * frictionloss, 1.1 * frictionloss)
        armature = mjm.dof_armature[dofs]
        sim.set_random("dof_armature", (dofs,), armature, 1.05 * armature)
        joints = len(robot.joint_names)
        sim.set_random("qpos0", (robot.data.joint_q_adr.tolist(),), [-0.05] * joints, [0.05] * joints, add=True)  # rad
        trunk_body = robot.body_ids[robot.find_bodies(["trunk"])[0]]
        sim.set_random("body_ipos", (trunk_body,), [-0.05] * 3, [0.05] * 3, add=True)  # m: the center of mass
        mass = mjm.body_mass[robot.body_ids]
        sim.set_random("body_mass", (robot.body_ids,), 0.9 * mass, 1.1 * mass)
        sim.set_random("body_mass", (trunk_body,), -1.0, 1.0, add=True)  # kg

        # --- set up rewards ---
        # each term's weight; terms are per second
        weights = dict(height=1.0, orientation=1.0, contact=-0.1, dof_pos_limits=-0.5, pose=-0.1)
        self.names = list(weights)
        self.weights = torch.tensor(list(weights.values()), device=self.device)
        self.trunk = robot.find_bodies(["trunk"])[0]
        self.imu = robot.find_sites(["imu"])[0]  # on the trunk, its frame the trunk's
        self.hind_legs = [i for i, name in enumerate(robot.joint_names) if name.startswith(("RR", "RL"))]

        # --- set up observations ---
        actions = len(robot.actuator_ids)
        # the actor's noise on each observation
        noise = [0.1] * 3 + [0.2] * 3 + [0.05] * 3  # IMU velocity (m/s), angular velocity (rad/s), gravity
        noise += [0.01] * joints + [1.5] * joints  # joint angles (rad) and velocities (rad/s)
        self.obs_noise = torch.tensor(noise, device=self.device)
        actor_dims = len(noise) + actions  # and the last action but one
        # the critic also sees 4 more velocities and accelerations, the joints, their torques and the trunk's height
        self.obs_dims = (actor_dims, actor_dims + 4 * 3 + 2 * joints + actions + 1)
        imu_site = robot.site_ids[self.imu]
        self.imu_pos_b = torch.tensor(mjm.site_pos[imu_site][None], device=self.device, dtype=torch.float32)
        self.accelerometer = sim.get_sensor("accelerometer")  # (nworld, 3) view: the IMU's
        self.bound = torch.ones(actions, device=self.device)  # actions in [-1, 1]

        # --- initialize state ---
        self.joint_target = torch.zeros(n, actions, device=self.device)
        self.action = torch.zeros(n, actions, device=self.device)
        self.last_action = torch.zeros(n, actions, device=self.device)
        self.steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.tally = torch.zeros(n, device=self.device)  # per-step shares of each world's first episode's return
        self.live = torch.zeros(n, dtype=torch.bool, device=self.device)  # in the first episode since a full reset
        self.reset(torch.ones(n, dtype=torch.bool, device=self.device))

    def curriculum(self, step: int):
        """None: the task has no curriculum."""

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move the joint targets by `action_scale` * `action`, clipped to [-1, 1]; step; score."""
        action = action.clamp(-1.0, 1.0)
        self.last_action.copy_(self.action)
        self.action.copy_(action)
        self.joint_target += self.action_scale * action
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
        robot = self.robot.data  # terms and terminations read the same derived quantities, one substep stale
        self.steps += 1

        # --- rewards ---
        R_world_trunk = robot.body_link_mat_w[:, self.trunk]
        joint_pos = robot.joint_pos
        low, high = robot.soft_joint_pos_limits.T
        terms = dict(
            height=torch.exp(robot.site_pos_w[:, self.imu, 2].clamp(max=self.height) - self.height),  # m: the trunk's
            orientation=(0.5 - 0.5 * R_world_trunk[:, 2, 0]) ** 2,  # the trunk's forward axis down
            contact=(self.feet_contact.data.found > 0).any(1).float(),
            dof_pos_limits=((low - joint_pos).clamp(min=0.0) + (joint_pos - high).clamp(min=0.0)).sum(1),
            pose=((joint_pos - robot.default_joint_pos)[:, self.hind_legs] ** 2).sum(1),
        )
        term_values = torch.nan_to_num(torch.stack([terms[name] for name in self.names]))
        reward = ((term_values * self.weights[:, None]).sum(0) * self.config.control_dt).clamp(0.0, 1e4)

        # --- terminations ---
        upside_down = R_world_trunk[:, 2, 2] < -0.25
        leg_down = (self.leg_contact.data.found > 0).any(1)
        diverged = ~self.sim.data.torch.qvel.isfinite().all(1)  # the physics blew up
        fell = upside_down | leg_down | diverged
        timeout = self.steps >= self.config.episode
        self.tally.copy_(self.config.episode * reward * self.live)
        self.live &= ~(fell | timeout)
        return reward, fell, timeout, term_values.mean(1)

    @torch.compile
    def restart(self, mask: torch.Tensor):
        """Reset the masked worlds with masked writes only: standing at home, moved, turned and moving at random."""
        robot = self.robot
        data = self.sim.data.torch

        # --- robot: the base at home, moved, turned and moving; the joints at home and at rest ---
        offset = 0.5 * (2 * torch.rand(self.n, 8, device=self.device) - 1)  # m x, y; m/s and rad/s, world frame
        yaw = 3.14 * (2 * torch.rand(self.n, device=self.device) - 1)
        zero = torch.zeros_like(yaw)
        q_wxyz_world_base = torch.stack([(yaw / 2).cos(), zero, zero, (yaw / 2).sin()], 1)  # home's is the identity
        t_world_base = robot.data.default_root_state[:3] + torch.stack([offset[:, 0], offset[:, 1], zero], 1)
        robot.write_root_link_pose_to_sim(torch.cat([t_world_base, q_wxyz_world_base], 1), mask)
        robot.write_root_link_velocity_to_sim(offset[:, 2:], mask)
        home = robot.data.default_joint_pos
        robot.write_joint_state_to_sim(home, torch.zeros_like(home), mask)
        self.joint_target.copy_(torch.where(mask[:, None], home, self.joint_target))
        warmstart = data.qacc_warmstart
        warmstart.copy_(torch.where(mask[:, None], 0.0, warmstart))

        # --- episode: actions, steps and the tally ---
        for state in (self.action, self.last_action):
            state.copy_(torch.where(mask[:, None], 0.0, state))
        self.steps.masked_fill_(mask, 0)
        self.live |= mask.all()  # a full reset (as `ppo.evaluate`'s) starts tallying first episodes

    @torch.compile
    def obs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The actor's (noisy; its last action a step older) and the critic's observations."""
        robot = self.robot.data
        ang_vel_b = robot.root_link_ang_vel_b
        # the IMU's velocities come from the base's, exact after a reset too
        imu_lin_vel_b = robot.root_link_lin_vel_b + torch.linalg.cross(ang_vel_b, self.imu_pos_b)
        joint_pos = robot.joint_pos
        joint_vel = robot.joint_vel
        sensed = [imu_lin_vel_b, ang_vel_b, robot.projected_gravity_b, joint_pos - robot.default_joint_pos, joint_vel]
        sensed = torch.cat(sensed, 1)
        noise = self.obs_noise * (2 * torch.rand_like(sensed) - 1)
        actor = torch.cat([sensed + noise, self.last_action], 1)
        fresh = self.steps.eq(0)[:, None]  # actuator forces are stale after a reset: zero, at rest
        joint_torque = torch.where(fresh, 0.0, robot.qfrc_actuator)
        acceleration = self.accelerometer
        height = robot.site_pos_w[:, self.imu, 2:]
        ang_vel_w = robot.root_link_vel_w[:, 3:]
        critic = [actor, ang_vel_b, acceleration, imu_lin_vel_b, ang_vel_w, joint_pos, joint_vel, joint_torque, height]
        return actor, torch.cat(critic, 1)


if __name__ == "__main__":
    config = tyro.cli(HandstandConfig)
    env = Go1Handstand(config)  # the full physics, where the policy is scored
    coarse_env = Go1Handstand(config, coarse=config.coarse_physics) if config.coarse else None
    ppo_config = ppo.PPOConfig(entropy=0.001, lr=3e-4, lr_range=(3e-4, 3e-4))  # a fixed rate
    actor = ppo.PPO(env.obs_dims, len(env.bound), ppo_config).train(env, config.minutes, coarse_env, config.coarse)
    # a world's first episode spans at most `config.episode` steps: the mean of its per-step shares is its return
    print(ppo.evaluate(env, actor, dict(episode_reward=env.tally.clone), steps=config.episode))
