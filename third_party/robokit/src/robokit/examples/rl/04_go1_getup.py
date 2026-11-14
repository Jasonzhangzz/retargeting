# /// script
# requires-python = ">=3.12"
# dependencies = ["robokit[mjwarp,pt-cu126]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
"""Go1 getup: a Unitree Go1 dropped in a random pose learns to stand up with PPO on 4096 MuJoCo Warp worlds.

Example usage:
    uv run src/robokit/examples/rl/04_go1_getup.py --minutes 5
    uv run src/robokit/examples/rl/04_go1_getup.py --coarse 0.5    # coarse physics mixed in over the first half
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import mujoco  # pyright: ignore[reportMissingImports]
import torch
import tyro
import warp as wp

from robokit.assets.robots.quadrupeds import unitree_go1
from robokit.helpers.rl import Env, ppo
from robokit.sim import CoarseConfig, MjWarpSim, mjscene


@dataclass(frozen=True)
class GetupConfig:
    """The getup: its control step, episode and coarse physics, the worlds and PPO's time."""

    control_dt: float = 0.02  # s per policy action; physics steps are `mjm.opt.timestep`
    episode: int = 300  # env steps per episode: 6 s
    # Coarse physics (`CoarseConfig`): only the floor, feet and trunk collide, pyramidal cones
    coarse_physics: CoarseConfig = CoarseConfig(
        collide="floor|robot/(FR|FL|RR|RL)_foot_collision|robot/trunk_collision", pyramidal=True
    )
    worlds: int = 4096  # MjWarpSim worlds
    minutes: float = 5.0  # wall time to train, compilation included
    coarse: float = 0.0  # the share of PPO's time that mixes in coarse physics (`coarse_physics`)


class Getup(Env):
    """Go1 getup on `config.worlds` worlds, in the full physics or the `coarse` one."""

    def __init__(self, config: GetupConfig = GetupConfig(), coarse: Optional[CoarseConfig] = None):
        n = config.worlds
        self.n = n
        self.config = config

        # --- build scene ---
        # the Go1 on a floor: 4 ms Euler steps, 1 solver iteration, PD joint targets
        robot_spec = mjscene.build_robot_spec(unitree_go1.MJCF_PATH, prefix="robot/")
        T_world_base = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        scene_spec = mjscene.build_scene_spec([(robot_spec, T_world_base)], timestep=0.004)
        scene_spec.geom("floor").friction = (0.6, 0.005, 1e-4)
        scene_spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
        scene_spec.option.iterations = 1
        scene_spec.option.ls_iterations = 5
        scene_spec.option.ccd_iterations = 20
        scene_spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
        # feet and floor with 3D contacts, the floor's friction winning; the rest frictionless
        for geom in scene_spec.geoms:
            geom.condim = 3 if geom.name == "floor" or geom.name.endswith("_foot_collision") else 1
            geom.priority = 1 if geom.name == "floor" else 0
        for actuator in list(scene_spec.actuators):  # replaced by position actuators in joint order
            scene_spec.delete(actuator)
        for leg in ("FR", "FL", "RR", "RL"):
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
        # the scene's only sensors, at the IMU site: gyro, accelerometer, local linear and global angular velocity
        sensors = [
            mujoco.mjtSensor.mjSENS_GYRO,
            mujoco.mjtSensor.mjSENS_ACCELEROMETER,
            mujoco.mjtSensor.mjSENS_VELOCIMETER,
            mujoco.mjtSensor.mjSENS_FRAMEANGVEL,
        ]
        for sensor_type in sensors:
            scene_spec.add_sensor(type=sensor_type, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="robot/imu")
        stand = [0.1, 0.9, -1.8, -0.1, 0.9, -1.8] * 2  # rad: the standing joint angles
        scene_spec.add_key(name="home", qpos=[0.0, 0.0, 0.278, 1.0, 0.0, 0.0, 0.0, *stand], ctrl=stand)
        if coarse is not None:
            scene_spec = MjWarpSim.build_coarse_spec(scene_spec, coarse)

        # --- build simulator ---
        mjm = scene_spec.compile()
        self.mjm = mjm
        self.substeps = round(config.control_dt / mjm.opt.timestep)
        assert abs(self.substeps * mjm.opt.timestep - config.control_dt) < 1e-9, (
            "the control step must be a timestep multiple"
        )
        sim = MjWarpSim(mjm, n, substeps=self.substeps, key="home", njmax=250, nconmax=30)
        self.sim = sim
        sim.set_torch_stream()
        robot = sim.build_entity(
            "robot/", soft_joint_pos_limit_factor=0.95
        )  # 95% of each joint's range is penalty-free
        self.robot = robot
        self.device = robot.device
        data = sim.data.torch
        # what a reset restores: positions, velocities, sensors and actuator forces, all the Go1's
        self.state = [data.qpos, data.qvel, data.sensordata, data.actuator_force]

        # --- set up actions ---
        joints = len(robot.joint_names)
        actions = len(robot.actuator_ids)
        self.bound = torch.ones(actions, device=self.device)  # actions in [-1, 1]

        # --- randomize ---
        # floor friction, joint friction, armature and zero pose, masses, the trunk's center of mass and mass
        sim.set_random("geom_friction", (mjm.geom("floor").id, 0), 0.4, 1.0)
        joint_dofs = robot.data.joint_v_adr
        wp.to_torch(sim.model.dof_frictionloss)[:, joint_dofs] *= 0.9 + 0.2 * torch.rand(n, joints, device=self.device)
        wp.to_torch(sim.model.dof_armature)[:, joint_dofs] *= 1.0 + 0.05 * torch.rand(n, joints, device=self.device)
        sim.set_random("qpos0", (robot.data.joint_q_adr.tolist(),), [-0.05] * joints, [0.05] * joints, add=True)  # rad
        wp.to_torch(sim.model.body_mass)[:] *= 0.9 + 0.2 * torch.rand(n, mjm.nbody, device=self.device)
        sim.set_random("body_ipos", (robot.root_body,), [-0.05] * 3, [0.05] * 3, add=True)  # m
        sim.set_random("body_mass", (robot.root_body,), -1.0, 1.0, add=True)  # kg; then set_const for all the above

        # --- set up rewards ---
        # each term's weight; terms are per second
        weights = dict(
            orientation=1.0,
            torso_height=1.0,
            posture=1.0,
            stand_still=1.0,
            action_rate=-0.001,
            dof_pos_limits=-0.1,
            torques=-1e-5,
            dof_acc=-2.5e-7,
            dof_vel=-0.1,
        )
        self.names = list(weights)
        self.weights = torch.tensor(list(weights.values()), device=self.device)
        self.imu = robot.find_sites(["imu"])[0]  # on the trunk, its axes the trunk's

        # --- set up observations ---
        # the actor's noise on its gyro (rad/s), gravity, joint angles (rad) and velocities (rad/s); none on the action
        self.obs_noise = torch.tensor([0.2] * 3 + [0.05] * 3 + [0.03] * joints + [1.5] * joints, device=self.device)
        self.obs_dims = (42, 91)  # the critic adds the sensors, joints, torques and the trunk's height

        # --- initialize state ---
        self.action = torch.zeros(n, actions, device=self.device)
        self.last_action = torch.zeros(n, actions, device=self.device)
        self.action_before_last = torch.zeros(n, actions, device=self.device)
        self.steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.reward = torch.zeros(n, device=self.device)  # the last step's, per world
        self.standing = torch.zeros(n, device=self.device)  # 1 where upright at the standing height after the last step

        # --- set up the reset pool ---
        # 16 * n states, 60% dropped from 0.5 m in random poses, the rest at home, each settled for 0.5 s
        joint_low = torch.tensor(mjm.jnt_range[robot.joint_ids, 0], device=self.device, dtype=torch.float32)
        joint_high = torch.tensor(mjm.jnt_range[robot.joint_ids, 1], device=self.device, dtype=torch.float32)
        t_world_drop = torch.tensor([0.0, 0.0, 0.5], device=self.device).expand(n, 3)
        pool = []
        for _ in range(16):
            dropped = torch.rand(n, 1, device=self.device) < 0.6
            q_wxyz_world_drop = torch.randn(n, 4, device=self.device)  # a uniform rotation, once normalized
            q_wxyz_world_drop = q_wxyz_world_drop / (q_wxyz_world_drop.norm(dim=1, keepdim=True) + 1e-6)
            T_world_drop = torch.cat([t_world_drop, q_wxyz_world_drop], 1)
            robot.write_root_link_pose_to_sim(torch.where(dropped, T_world_drop, robot.data.default_root_state[:7]))
            joint_pos = joint_low + (joint_high - joint_low) * torch.rand(n, joints, device=self.device)
            joint_pos = torch.where(dropped, joint_pos, robot.data.default_joint_pos)
            robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
            robot.write_root_link_velocity_to_sim(-0.5 + torch.rand(n, 6, device=self.device))  # m/s, rad/s
            robot.set_joint_position_target(joint_pos)
            data.qacc_warmstart.zero_()
            for _ in range(round(0.5 / config.control_dt)):
                sim.step()
            pool.append(torch.cat([field.clone() for field in self.state], 1))
        self.pool = torch.cat(pool)
        self.state_widths = [field.shape[1] for field in self.state]
        self.reset(torch.ones(n, dtype=torch.bool, device=self.device))

    def curriculum(self, step: int):
        """None: the getup has no curriculum."""

    def episode_reward(self) -> torch.Tensor:
        """The per-step reward times the episode length, per world."""
        return self.reward * self.config.episode

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Target the current joint angles + 0.5 * `action`, clipped to [-1, 1]; step; score."""
        action = action.clamp(-1.0, 1.0)
        self.action_before_last.copy_(self.last_action)
        self.last_action.copy_(self.action)
        self.action.copy_(action)
        self.robot.set_joint_position_target(self.robot.data.joint_pos + 0.5 * action)
        self.sim.step()
        return self.score()

    def reset(self, mask: torch.Tensor):
        """Reset the masked worlds, then refresh the kinematics the observations read."""
        self.restart(mask)
        self.sim.kinematics()

    @torch.compile
    def score(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The rewards from the (one substep stale) derived quantities; terminated only where the physics blew up."""
        robot = self.robot.data
        data = self.sim.data.torch
        self.steps += 1

        # --- rewards ---
        gravity_b = robot.projected_gravity_b  # in the IMU's frame, the trunk's
        tilt_error = (gravity_b[:, :2] ** 2).sum(1) + (gravity_b[:, 2] + 1.0) ** 2  # from (0, 0, -1), upright
        height = robot.site_pos_w[:, self.imu, 2].clamp(max=0.275)  # m: the IMU's, up to the standing height
        upright = tilt_error < 0.01
        standing = upright & (0.275 - height < 0.005)
        joint_error = robot.joint_pos - robot.default_joint_pos
        low, high = robot.soft_joint_pos_limits.T
        out_of_limits = (low - robot.joint_pos).clamp(min=0.0) + (robot.joint_pos - high).clamp(min=0.0)
        action_rate = self.action - self.last_action
        action_accel = self.action - 2 * self.last_action + self.action_before_last
        torque = data.actuator_force
        terms = dict(
            orientation=torch.exp(-2.0 * tilt_error),
            torso_height=height.exp() - 1.0,
            posture=upright * torch.exp(-0.5 * (joint_error**2).sum(1)),
            stand_still=standing * torch.exp(-0.5 * (self.action**2).sum(1)),
            action_rate=(action_rate**2).sum(1) + (action_accel**2).sum(1),
            dof_pos_limits=out_of_limits.sum(1),
            torques=torque.norm(dim=1) + torque.abs().sum(1),
            dof_acc=(data.qacc[:, robot.joint_v_adr] ** 2).sum(1),
            dof_vel=((robot.joint_vel.abs() - 2 * math.pi).clamp(min=0.0) ** 2).sum(1),
        )
        term_values = torch.nan_to_num(torch.stack([terms[name] for name in self.names]))
        reward = ((term_values * self.weights[:, None]).sum(0) * self.config.control_dt).clamp(0.0, 1e4)
        self.reward.copy_(reward)
        self.standing.copy_(standing.float())

        # --- terminations ---
        diverged = ~data.qvel.isfinite().all(1)  # the physics blew up
        timeout = self.steps >= self.config.episode
        return reward, diverged, timeout, term_values.mean(1)

    @torch.compile
    def restart(self, mask: torch.Tensor):
        """Draw the masked worlds' states from the pool, with masked writes only."""
        # --- simulator: a pooled state, no warm start ---
        drawn = self.pool[torch.randint(len(self.pool), (self.n,), device=self.device)]
        for field, value in zip(self.state, drawn.split(self.state_widths, 1)):
            field.copy_(torch.where(mask[:, None], value, field))
        warmstart = self.sim.data.torch.qacc_warmstart
        warmstart.copy_(torch.where(mask[:, None], 0.0, warmstart))

        # --- episode: steps and actions ---
        self.steps.masked_fill_(mask, 0)
        for state in (self.action, self.last_action, self.action_before_last):
            state.copy_(torch.where(mask[:, None], 0.0, state))

    @torch.compile
    def obs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The actor's (noisy; its last action lags a step) and the critic's observations."""
        robot = self.robot.data
        data = self.sim.data.torch
        imu = data.sensordata  # gyro, accelerometer, local linear and global angular velocity
        joint_error = robot.joint_pos - robot.default_joint_pos
        clean = torch.cat([imu[:, :3], robot.projected_gravity_b, joint_error, robot.joint_vel], 1)
        actor = torch.cat([clean + self.obs_noise * (2 * torch.rand_like(clean) - 1), self.last_action], 1)
        height = robot.site_pos_w[:, self.imu, 2:]  # m: the IMU's
        critic = torch.cat([actor, imu, robot.joint_pos, robot.joint_vel, data.actuator_force, height], 1)
        return actor, critic


if __name__ == "__main__":
    config = tyro.cli(GetupConfig)
    env = Getup(config)  # the full physics, where the policy is scored
    coarse_env = Getup(config, coarse=config.coarse_physics) if config.coarse else None
    # PPO settings for this task; tanh-squashed actions, their entropy keeping the means off saturation
    ppo_config = ppo.PPOConfig(
        std="state",
        squash_entropy=True,
        gamma=0.97,
        entropy=0.01,
        lr=3e-4,
        lr_range=(3e-4, 3e-4),
        horizon=20,
        minibatches=32,
        epochs=4,
    )
    actor = ppo.PPO(env.obs_dims, len(env.bound), ppo_config).train(env, config.minutes, coarse_env, config.coarse)
    metrics = dict(episode_reward=env.episode_reward, standing=env.standing.clone)
    print(ppo.evaluate(env, actor, metrics, steps=3 * config.episode))
