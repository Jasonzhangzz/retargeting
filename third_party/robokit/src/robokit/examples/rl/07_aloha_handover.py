# /// script
# requires-python = ">=3.12"
# dependencies = ["robokit[mjwarp,pt-cu126]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
"""Aloha handover: two ALOHA arms learn to pass a box between their grippers and carry it to a target, on 2048 worlds.

Example usage:
    uv run src/robokit/examples/rl/07_aloha_handover.py --minutes 10
    uv run src/robokit/examples/rl/07_aloha_handover.py --coarse 0.5  # coarse physics mixed in over the first half
"""

from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import mujoco  # pyright: ignore[reportMissingImports]
import torch
import tyro

from robokit.assets.robots.assembly import agilex_aloha
from robokit.helpers.rl import Env, ppo
from robokit.sim import CoarseConfig, MjWarpSim, mjscene


@dataclass(frozen=True)
class HandoverConfig:
    """The handover task and its training: control, gains, home, the box's start, the worlds and the learner's time."""

    control_dt: float = 0.02  # s per policy action; physics steps are `mjm.opt.timestep`
    episode: int = 250  # env steps per episode
    # Arm joints 1 to 6 and the gripper: position gains (N m/rad, N/m) and joint damping
    gains: Tuple[Tuple[float, float], ...] = (
        (100.0, 6.0),
        (150.0, 8.0),
        (120.0, 6.0),
        (40.0, 2.0),
        (20.0, 1.0),
        (20.0, 1.0),
        (200.0, 10.0),
    )
    # Home: each arm's joints 1 to 6 put its gripper 13 cm above the box (x -/+ 0.16 m), pointing down, fingers open
    home_arm: Tuple[float, ...] = (-0.0138, 1.715, 1.7168, -1.5721, 0.0002, -0.0115)
    # Coarse physics, off by default (`CoarseConfig`): only the colliding geoms, pyramidal cones
    coarse_physics: CoarseConfig = CoarseConfig(
        collide=r"table|box|(left|right)/(finger7|finger8|gripper)", pyramidal=True
    )
    action_scale: float = 0.015  # the joint targets' change per step per unit of action
    box_range: Tuple[float, float] = (0.05, 0.1)  # m: the box starts up to this far from home in x and y
    worlds: int = 2048  # MjWarpSim worlds
    minutes: float = 10.0  # wall time to train, compilation included
    coarse: float = 0.0  # the share of PPO's time that mixes in coarse physics (`coarse_physics`)


class AlohaHandOver(Env):
    """Box handover between two ALOHA arms, in the full physics or the `coarse` one."""

    def __init__(self, config: HandoverConfig = HandoverConfig(), coarse: Optional[CoarseConfig] = None):
        n = config.worlds
        self.n = n
        self.config = config
        self.action_scale = config.action_scale

        # --- build scene ---
        # two ALOHA arms facing each other across a table
        robots = []
        for side, T_world_base in (
            ("left", (-0.469, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
            ("right", (0.469, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
        ):
            arm_spec = mjscene.build_robot_spec(agilex_aloha.MJCF_PATH, prefix=f"{side}/", body="fr_link1")
            base = arm_spec.body(f"{side}/fr_link1")
            base.pos = [0.0, 0.0, 0.0]
            base.quat = [1.0, 0.0, 0.0, 0.0]
            robots.append((arm_spec, T_world_base))
        scene_spec = mjscene.build_scene_spec(robots, floor=False, timestep=0.005)
        table = scene_spec.worldbody.add_body(name="table", pos=[0, 0, -0.05])
        table.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.61, 0.37, 0.05], rgba=[0.6, 0.5, 0.4, 1.0])
        scene_spec.option.iterations = 10
        scene_spec.option.ls_iterations = 10
        scene_spec.option.impratio = 5.0
        for actuator in list(scene_spec.actuators):  # replaced by joint position actuators
            scene_spec.delete(actuator)
        for side in ("left", "right"):
            for body in scene_spec.bodies:
                if body.name.startswith(f"{side}/"):
                    body.gravcomp = 1.0
                    for geom in body.geoms:  # only the gripper collides
                        geom.contype = 0
                        geom.conaffinity = int(body.name[-1] in "678")
            for i, (kp, damping) in enumerate(config.gains, 1):
                joint = scene_spec.joint(f"{side}/fr_joint{i}")
                joint.damping = [damping, 0.0, 0.0]
                joint.armature = 0.01
                scene_spec.add_actuator(
                    name=f"{side}/joint{i}",
                    target=joint.name,
                    trntype=mujoco.mjtTrn.mjTRN_JOINT,
                    gainprm=[kp, *[0.0] * 9],
                    biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                    biasprm=[0.0, -kp, *[0.0] * 8],
                    ctrlrange=joint.range,
                    ctrllimited=mujoco.mjtLimited.mjLIMITED_TRUE,
                )
            finger = scene_spec.joint(f"{side}/fr_joint8")
            finger.damping = [config.gains[-1][1], 0.0, 0.0]
            finger.armature = 0.01
            scene_spec.add_equality(
                type=mujoco.mjtEq.mjEQ_JOINT,
                objtype=mujoco.mjtObj.mjOBJ_JOINT,
                name1=finger.name,
                name2=f"{side}/fr_joint7",
                data=[0.0, 1.0, *[0.0] * 10],  # the second finger follows the first
            )
            scene_spec.body(f"{side}/fr_link6").add_site(name=f"{side}/gripper", pos=[0.135, 0.0, 0.0])
            scene_spec.body(f"{side}/fr_link6").first_geom().name = f"{side}/gripper"
            for link in (7, 8):  # each finger's contact flag against the table
                scene_spec.body(f"{side}/fr_link{link}").first_geom().name = f"{side}/finger{link}"
                mjscene.build_contact_sensor(
                    scene_spec,
                    f"table_contact_{side}/finger{link}_found",
                    (mujoco.mjtObj.mjOBJ_GEOM, "table"),
                    (mujoco.mjtObj.mjOBJ_GEOM, f"{side}/finger{link}"),
                    reduce="mindist",
                )
        # a 4 x 3 x 6 cm box on the table
        box_home = (-0.16, 0.0, 0.03, 1.0, 0.0, 0.0, 0.0)
        box = scene_spec.worldbody.add_body(name="box", pos=box_home[:3])
        box.add_freejoint(name="box")
        box.add_geom(
            name="box",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.02, 0.015, 0.03],
            mass=0.044,
            rgba=[1.0, 0.0, 0.0, 1.0],
            friction=[1.0, 0.05, 0.001],
            condim=4,
            solref=[0.01, 1.0],
        )
        box.add_site(name="box_top", pos=[0.0, 0.0, 0.02])
        box.add_site(name="box_bottom", pos=[0.0, 0.0, -0.02])
        # the initial state: the arms at home with their fingers open (m), each actuator holding its joint's angle
        arm_home = [*config.home_arm, 0.037, 0.037]
        arm_ctrl = [*config.home_arm, 0.037]
        scene_spec.add_key(name="home", qpos=[*arm_home, *arm_home, *box_home], ctrl=[*arm_ctrl, *arm_ctrl])
        if coarse is not None:
            scene_spec = MjWarpSim.build_coarse_spec(scene_spec, coarse)

        # --- build simulator ---
        mjm = scene_spec.compile()
        self.mjm = mjm
        self.substeps = round(config.control_dt / mjm.opt.timestep)
        assert abs(self.substeps * mjm.opt.timestep - config.control_dt) < 1e-9, (
            "the control step must be a timestep multiple"
        )
        sim = MjWarpSim(mjm, n, substeps=self.substeps, key="home", njmax=128, nconmax=32)
        self.sim = sim
        sim.set_torch_stream()
        self.left = sim.build_entity("left/")
        self.right = sim.build_entity("right/")
        self.box = sim.build_entity("box")  # its sites: "_top", "_bottom"
        self.device = self.left.device
        self.table_contact = sim.build_contact_sensor("table_contact")  # the fingers'

        # --- set up actions ---
        # each step moves the joint targets, the left arm's then the right's, within their ranges
        actuators = self.left.actuator_ids + self.right.actuator_ids
        target_range = torch.tensor(mjm.actuator_ctrlrange[actuators], device=self.device, dtype=torch.float32)
        self.target_low = target_range[:, 0]
        self.target_high = target_range[:, 1]
        self.target_low[6] = 0.01  # an aid to finger exploration: the left gripper never closes fully
        self.home_target = torch.cat([arm.data.default_joint_pos[arm.actuated_ids] for arm in (self.left, self.right)])
        self.bound = torch.ones(len(actuators), device=self.device)  # actions in [-1, 1]

        # --- set up rewards ---
        weights = dict(gripper_box=1.0, box_handover=4.0, handover_target=8.0, no_table_collision=0.3)
        self.names = list(weights)
        self.weights = torch.tensor(list(weights.values()), device=self.device)
        self.gripper = self.left.find_sites(["gripper"])[0]  # the arms are alike: the same index on both
        self.wrist = self.left.find_bodies(["fr_link6"])[0]  # the gripper site's body, whose frame is the site's
        self.fingers = [self.left.joint_names.index(f"fr_joint{i}") for i in (7, 8)]
        self.box_top, self.box_bottom = self.box.find_sites(["_top", "_bottom"])
        self.box_width = float(mjm.geom("box").size[1])
        self.handover = torch.tensor((0.0, 0.0, 0.24), device=self.device)  # m

        # --- set up resets ---
        self.box_home = torch.tensor(box_home, device=self.device)
        self.box_range = torch.tensor(config.box_range, device=self.device)
        self.target_center = torch.tensor((0.2, 0.0, 0.25), device=self.device)  # m

        # --- set up observations ---
        # the whole state, the finger openings and 38 values on the box, the grippers, the target and the time
        self.obs_dims = (mjm.nq + mjm.nv + 38,) * 2  # the critic's the same

        # --- initialize state ---
        self.joint_target = torch.zeros(n, len(actuators), device=self.device)
        self.box_target = torch.zeros(n, 3, device=self.device)
        self.best_potential = torch.zeros(n, device=self.device)  # this episode's
        self.steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.picked = torch.zeros(n, dtype=torch.bool, device=self.device)  # the box lifted this episode
        # `counted`, `success`: the rewards and end state of each world's first episode since a full reset
        self.counted = torch.zeros(n, device=self.device)
        self.success = torch.zeros(n, device=self.device)
        self.alive = torch.zeros(n, dtype=torch.bool, device=self.device)  # in that first episode
        self.reset(torch.ones(n, dtype=torch.bool, device=self.device))

    def curriculum(self, step: int):
        """None: the task has no curriculum."""

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move the joint targets by `action_scale` * `action`, clipped to [-1, 1], within their ranges; step; score."""
        moved = self.joint_target + self.action_scale * action.clamp(-1.0, 1.0)
        self.joint_target.copy_(torch.clamp(moved, self.target_low, self.target_high))
        self.left.set_joint_position_target(self.joint_target[:, : len(self.config.gains)])
        self.right.set_joint_position_target(self.joint_target[:, len(self.config.gains) :])
        self.sim.step()
        return self.score()

    def reset(self, mask: torch.Tensor):
        """Reset the masked worlds, then refresh the kinematics the observations read."""
        self.restart(mask)
        self.sim.kinematics()

    @torch.compile
    def score(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The rewards (potential rise, handover bonus, drop penalty), terminations, timeouts and terms' means."""
        data = self.sim.data.torch
        box = self.box.data
        box_pos = box.root_link_pos_w
        left_gripper = self.left.data.site_pos_w[:, self.gripper]
        right_gripper = self.right.data.site_pos_w[:, self.gripper]

        # --- rewards ---
        before = (box_pos[:, 0] < -0.1).float()  # the box left of the handover point
        past = (box_pos[:, 0] >= 0.0).float()  # the box right of it
        between = (1 - before) * (1 - past)
        left_grasp = torch.exp(-10 * (box.site_pos_w[:, self.box_top] - left_gripper).norm(dim=-1)) * (before + between)
        right_near = torch.exp(-10 * (box.site_pos_w[:, self.box_bottom] - right_gripper).norm(dim=-1))
        right_grasp = right_near * (between + 2 * past)  # biased past the handover point
        box_at_handover = torch.exp(-10 * (box_pos - self.handover).norm(dim=-1))
        left_at_handover = torch.exp(-10 * (left_gripper - self.handover).norm(dim=-1))
        box_at_target = torch.exp(-10 * (self.box_target - box_pos).norm(dim=-1))
        left_withdrawn = torch.sigmoid(-100 * left_gripper[:, 0])  # a logistic step: the left gripper left of x 0
        box_across = torch.sigmoid(100 * (box_pos[:, 0] - 0.1))  # the box right of x 0.1 m
        terms = dict(
            gripper_box=left_grasp + right_grasp,
            box_handover=torch.maximum(box_at_handover, left_at_handover * past),
            handover_target=box_at_target * right_grasp * left_withdrawn,
            no_table_collision=1 - (self.table_contact.data.found > 0).any(1).float(),
        )
        term_values = torch.nan_to_num(torch.stack([terms[name] for name in self.names])) * self.weights[:, None]
        potential = term_values.sum(0) / self.weights.sum()  # the weighted terms' mean
        bonus = 0.02 * potential * left_withdrawn * box_across
        rise = (potential - self.best_potential).clamp(min=0.0)  # over its best this episode
        reward = torch.where(self.steps == 0, 0.0, rise + bonus)  # none on an episode's first step
        self.best_potential.copy_(torch.maximum(potential, self.best_potential))
        self.picked |= box_pos[:, 2] > 0.15  # m
        dropped = self.picked & (box_pos[:, 2] < 0.05)  # penalized, and ends the episode
        reward = reward - 0.1 * dropped.float()

        # --- terminations ---
        diverged = ~(data.qpos.isfinite().all(1) & data.qvel.isfinite().all(1))  # the physics blew up
        off_table = (box_pos.abs() > 1.0).any(1) | (box_pos[:, 2] < 0.0)
        terminated = off_table | diverged | dropped
        self.steps += 1
        timeout = self.steps >= self.config.episode
        self.counted.copy_(reward * self.alive)
        at_target = (self.box_target - box_pos).norm(dim=1) < 0.05  # m
        self.success.copy_((self.alive & (terminated | timeout) & at_target).float())
        return reward, terminated, timeout, term_values.mean(1)

    @torch.compile
    def restart(self, mask: torch.Tensor):
        """Reset the masked worlds with masked writes only: the arms at home, the box moved, a new target."""
        data = self.sim.data.torch

        # --- arms: at home and at rest, their targets at home ---
        for arm in (self.left, self.right):
            home = arm.data.default_joint_pos
            arm.write_joint_state_to_sim(home, torch.zeros_like(home), mask)
        self.joint_target.copy_(torch.where(mask[:, None], self.home_target, self.joint_target))

        # --- box: at home, moved up to `box_range` in x and y, at rest ---
        box_offset = self.box_range * (2 * torch.rand(self.n, 2, device=self.device) - 1)
        box_pose = torch.cat([self.box_home[:2] + box_offset, self.box_home[2:].expand(self.n, 5)], 1)
        self.box.write_root_link_pose_to_sim(box_pose, mask)
        self.box.write_root_link_velocity_to_sim(torch.zeros(self.n, 6, device=self.device), mask)
        warmstart = data.qacc_warmstart
        warmstart.copy_(torch.where(mask[:, None], 0.0, warmstart))

        # --- episode: the target, the best potential, steps, the lift and the counted episodes ---
        box_target = self.target_center + 0.15 * (2 * torch.rand(self.n, 3, device=self.device) - 1)
        box_target[:, 0].clamp_(min=0.15)  # up to 15 cm from the center along each axis, at x >= 0.15 m
        self.box_target.copy_(torch.where(mask[:, None], box_target, self.box_target))
        self.best_potential.masked_fill_(mask, 0.0)
        self.steps.masked_fill_(mask, 0)
        self.picked.masked_fill_(mask, False)
        self.alive.copy_(mask.all() | (self.alive & ~mask))  # a full reset starts the episodes the metrics count

    @torch.compile
    def obs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The actor's observations, the critic's the same: the state, finger openings, sites, frames, progress."""
        data = self.sim.data.torch
        left = self.left.data
        right = self.right.data
        box = self.box.data
        obs = torch.cat(
            [
                data.qpos,  # both arms' joints, then the box's free joint
                data.qvel,
                left.joint_pos[:, self.fingers] - self.box_width,  # the finger openings past the box's width
                right.joint_pos[:, self.fingers] - self.box_width,
                box.site_pos_w[:, self.box_top],
                box.site_pos_w[:, self.box_bottom],
                left.site_pos_w[:, self.gripper],
                left.body_link_mat_w[:, self.wrist, 1:].flatten(1),  # the frames' rows 2 and 3
                right.site_pos_w[:, self.gripper],
                right.body_link_mat_w[:, self.wrist, 1:].flatten(1),
                box.body_link_mat_w[:, 0, 1:].flatten(1),
                box.root_link_pos_w - self.box_target,  # the box's offset from the target
                self.steps[:, None] / self.config.episode,  # the progress
            ],
            1,
        )
        return obs, obs


if __name__ == "__main__":
    config = tyro.cli(HandoverConfig)
    env = AlohaHandOver(config)  # the full physics, where the policy is scored
    coarse_env = AlohaHandOver(config, coarse=config.coarse_physics) if config.coarse else None
    learner = ppo.PPO(env.obs_dims, len(env.bound), ppo.PPOConfig(gamma=0.97, hidden=(256, 256, 256)))
    actor = learner.train(env, config.minutes, coarse_env, config.coarse)
    # each metric times the steps: its sum over each world's first episode
    episode_reward = partial(torch.mul, env.counted, config.episode)
    success = partial(torch.mul, env.success, config.episode)
    print(ppo.evaluate(env, actor, dict(episode_reward=episode_reward, success=success), steps=config.episode))
