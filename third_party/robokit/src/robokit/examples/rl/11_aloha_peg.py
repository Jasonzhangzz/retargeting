# /// script
# requires-python = ">=3.12"
# dependencies = ["robokit[mjwarp,pt-cu126]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
"""Aloha peg insertion: two ALOHA arms learn with PPO to lift a socket and a peg and push the peg in, in the air.

Example usage:
    uv run src/robokit/examples/rl/11_aloha_peg.py --minutes 15
    uv run src/robokit/examples/rl/11_aloha_peg.py --coarse 0.5  # coarse physics mixed in over the first half
"""

from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import mujoco  # pyright: ignore[reportMissingImports]
import torch
import tyro
import warp as wp

from robokit.assets.robots.assembly import agilex_aloha
from robokit.helpers.rl import Env, ppo
from robokit.sim import CoarseConfig, MjWarpSim, mjscene


@dataclass(frozen=True)
class AlohaPegConfig:
    """The peg insertion's training: the worlds and the learner's time."""

    worlds: int = 1024  # MjWarpSim worlds
    minutes: float = 15.0  # wall time to train, compilation included
    coarse: float = 0.0  # the share of PPO's time that mixes in coarse physics (`coarse_physics`)
    physics_dt: float = 0.0025  # s per physics step, each with its own control update
    repeat: int = 2  # physics steps per action
    episode: int = 500  # actions per episode
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
    # Home: each arm's joints 1 to 6 put its gripper 13 cm above x -/+ 0.16 m, pointing down, fingers open
    home_arm: Tuple[float, ...] = (-0.0138, 1.715, 1.7168, -1.5721, 0.0002, -0.0115)
    # Coarse physics (`CoarseConfig`): only the colliding geoms (grippers, peg, socket, table) and pyramidal cones
    coarse_physics: CoarseConfig = CoarseConfig(
        collide=r"table|peg|wall|socket-[BTLR]|(left|right)/(finger7|finger8|gripper)", pyramidal=True
    )


class AlohaPeg(Env):
    """Peg insertion on `config.worlds` worlds, in the full physics or the `coarse` one."""

    def __init__(self, config: AlohaPegConfig, coarse: Optional[CoarseConfig] = None):
        n = config.worlds
        self.n = n
        self.config = config
        self.control_dt = config.repeat * config.physics_dt  # s per policy action

        # --- build scene ---
        # two arms facing each other across a table, an open socket (9.6 x 4.4 x 4.4 cm, a wall at its back), a peg
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
        scene_spec = mjscene.build_scene_spec(robots, floor=False, timestep=config.physics_dt)
        table = scene_spec.worldbody.add_body(name="table", pos=[0, 0, -0.05])
        table.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.61, 0.37, 0.05], rgba=[0.6, 0.5, 0.4, 1.0])
        scene_spec.option.iterations = 10
        scene_spec.option.ls_iterations = 10
        scene_spec.option.impratio = 5.0
        for actuator in list(scene_spec.actuators):  # the MJCF's, replaced below
            scene_spec.delete(actuator)
        for side in ("left", "right"):
            for body in scene_spec.bodies:
                if body.name.startswith(f"{side}/"):
                    body.gravcomp = 1.0
                    for geom in body.geoms:
                        geom.contype = 0
                        geom.conaffinity = int(body.name[-1] in "678")  # only the gripper collides
            for i, (kp, damping) in enumerate(config.gains, 1):  # joint position actuators
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
            for link in (7, 8):  # each finger's table contact
                scene_spec.body(f"{side}/fr_link{link}").first_geom().name = f"{side}/finger{link}"
                mjscene.build_contact_sensor(
                    scene_spec,
                    f"table_{side}/finger{link}_found",
                    (mujoco.mjtObj.mjOBJ_GEOM, "table"),
                    (mujoco.mjtObj.mjOBJ_GEOM, f"{side}/finger{link}"),
                    reduce="mindist",
                )
        peg_home = (0.136, 0.0, 0.0108, 1.0, 0.0, 0.0, 0.0)
        peg = scene_spec.worldbody.add_body(name="peg", pos=peg_home[:3])
        peg.add_freejoint(name="peg")
        peg.add_geom(
            name="peg",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.048, 0.01, 0.01],
            mass=0.044,
            rgba=[1.0, 0.0, 0.0, 1.0],
            friction=[1.0, 0.05, 0.001],
        )
        peg.add_site(name="peg_end1", pos=[0.048, 0.0, 0.0])
        peg.add_site(name="peg_end2", pos=[-0.048, 0.0, 0.0])
        socket_home = (-0.147, 0.0, 0.0228, 1.0, 0.0, 0.0, 0.0)
        socket = scene_spec.worldbody.add_body(name="socket", pos=socket_home[:3])
        socket.add_freejoint(name="socket")
        walls = dict(
            B=((0, 0, -0.02), (0.048, 0.022, 0.002)),
            T=((0, 0, 0.02), (0.048, 0.022, 0.002)),
            L=((0, -0.02, 0), (0.048, 0.002, 0.018)),
            R=((0, 0.02, 0), (0.048, 0.002, 0.018)),
        )
        for name, (pos, size) in walls.items():
            socket.add_geom(
                name=f"socket-{name}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=pos,
                size=size,
                mass=0.016,
                rgba=[0.2, 0.4, 0.9, 1.0],
            )
        socket.add_geom(
            name="wall",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[-0.044, 0, 0],
            size=[0.004, 0.018, 0.018],
            mass=0.006,
            rgba=[1.0, 0.0, 0.0, 1.0],
        )
        socket.add_site(name="socket_entrance", pos=[0.048, 0.0, 0.0])
        socket.add_site(name="socket_rear", pos=[0.0, 0.0, 0.0])
        for geom in [*socket.geoms, scene_spec.geom("peg")]:
            geom.condim = 4
            geom.solref = [0.01, 1.0]
            geom.friction = [1.0, 0.05, 0.001]
        arm_home = [*config.home_arm, 0.037, 0.037]  # m: fingers open
        arm_ctrl = [*config.home_arm, 0.037]
        home_qpos = [*arm_home, *arm_home, *peg_home, *socket_home]  # the arms, the peg, the socket
        scene_spec.add_key(name="home", qpos=home_qpos, ctrl=[*arm_ctrl, *arm_ctrl])
        if coarse is not None:
            scene_spec = MjWarpSim.build_coarse_spec(scene_spec, coarse)

        # --- build simulator ---
        mjm = scene_spec.compile()
        self.mjm = mjm
        self.repeat = max(1, round(self.control_dt / mjm.opt.timestep))  # physics steps per action
        # control steps per physics step: each one's reward and control change
        self.step_scale = config.repeat / self.repeat
        sim = MjWarpSim(mjm, n, key="home", njmax=256, nconmax=32)
        self.sim = sim
        sim.set_torch_stream()
        self.left = sim.build_entity("left/")
        self.right = sim.build_entity("right/")
        self.device = self.left.device
        self.table_contact = sim.build_contact_sensor("table")  # the fingers'

        # --- set up actions ---
        # each control step moves the controls 0.005 times the action (in [-1, 1]) within their ranges
        ctrl_range = torch.tensor(mjm.actuator_ctrlrange, device=self.device, dtype=torch.float32)
        self.ctrl_low = ctrl_range[:, 0]
        self.ctrl_high = ctrl_range[:, 1]
        self.bound = torch.ones(mjm.nu, device=self.device)

        # --- set up rewards ---
        weights = dict(
            left_reward=1.0,
            right_reward=1.0,
            left_target_qpos=0.3,
            right_target_qpos=0.3,
            no_table_collision=0.3,
            socket_z_up=0.5,
            peg_z_up=0.5,
            socket_entrance_reward=4.0,
            peg_end2_reward=4.0,
            peg_insertion_reward=8.0,
        )
        self.names = list(weights)
        self.weights = torch.tensor(list(weights.values()), device=self.device)
        self.socket = mjm.body("socket").id
        self.peg = mjm.body("peg").id
        self.left_gripper = self.left.find_sites(["gripper"])[0]
        self.right_gripper = self.right.find_sites(["gripper"])[0]
        self.socket_entrance = mjm.site("socket_entrance").id
        self.socket_rear = mjm.site("socket_rear").id
        self.peg_end = mjm.site("peg_end2").id
        self.socket_goal = torch.tensor((-0.05, 0.0, 0.15), device=self.device)  # m
        self.peg_goal = torch.tensor((0.05, 0.0, 0.15), device=self.device)

        # --- set up resets ---
        self.socket_q_adr = int(mjm.joint("socket").qposadr[0])
        self.peg_q_adr = int(mjm.joint("peg").qposadr[0])

        # --- set up observations ---
        dim = mjm.nq + mjm.nv + 24
        self.obs_dims = (dim, dim)  # the critic's the same

        # --- initialize state ---
        # `counted`, `success`: each first episode's rewards and end state since a full reset
        self.counted = torch.zeros(n, device=self.device)
        self.success = torch.zeros(n, device=self.device)
        self.steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.alive = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.reset(torch.ones(n, dtype=torch.bool, device=self.device))

    def curriculum(self, step: int):
        """None: the task has no curriculum."""

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per repeated control step: move the controls by 0.005 x `action` (clipped to [-1, 1]), step, score."""
        delta = self.step_scale * 0.005 * action.clamp(-1.0, 1.0)
        ctrl = self.sim.data.torch.ctrl  # the targets: every actuator is an arm's
        reward = torch.zeros(self.n, device=self.device)
        for _ in range(self.repeat):
            ctrl.copy_(torch.clamp(ctrl + delta, self.ctrl_low, self.ctrl_high))
            self.sim.step()
            scored = self.score()
            reward += self.step_scale * scored[0]  # the rewards add up
        # the last physics step's terminations count
        _, terminated, inserted, term_values = scored  # pyright: ignore[reportPossiblyUnbound]
        self.steps += 1
        truncated = self.steps >= self.config.episode
        self.counted.copy_(reward * self.alive)
        self.success.copy_((self.alive & (terminated | truncated) & inserted).float())
        return reward, terminated, truncated, term_values

    def reset(self, mask: torch.Tensor):
        """Reset the masked worlds to the home keyframe, move the socket and the peg, then refresh the kinematics."""
        self.sim.reset(wp.from_torch(mask))
        self.restart(mask)
        self.sim.kinematics()

    @torch.compile
    def score(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """A control step's reward, the weighted terms' mean; terminations; whether the peg is in; each term's mean."""
        data = self.sim.data.torch
        left = self.left.data
        right = self.right.data
        socket = data.xpos[:, self.socket]
        peg = data.xpos[:, self.peg]
        up = data.xmat[:, :, 2, 2]  # each body's z axis' z
        rear = data.site_xpos[:, self.socket_rear]
        peg_end = data.site_xpos[:, self.peg_end]
        axis = data.site_xpos[:, self.socket_entrance] - rear
        along = ((peg_end - rear) * axis).sum(1, keepdim=True) / ((axis * axis).sum(1, keepdim=True) + 1e-6)
        aligned = (peg_end - rear - along * axis).norm(dim=1) < 0.005  # m: the peg's end on the socket's axis
        depth = (peg_end - rear).norm(dim=1)  # m: from the socket's middle; its mouth is 4.8 cm out

        # --- rewards ---
        errors = dict(  # each error (m, or 1 - a z axis' z), its tolerance and its margin
            left_gripper=((socket - left.site_pos_w[:, self.left_gripper]).norm(dim=1), 0.001, 0.3),
            right_gripper=((peg - right.site_pos_w[:, self.right_gripper]).norm(dim=1), 0.001, 0.3),
            socket_lift=((self.socket_goal - socket).norm(dim=1), 0.01, 0.15),
            peg_lift=((self.peg_goal - peg).norm(dim=1), 0.01, 0.15),
            insertion=(depth, 0.001, 0.1),
            socket_tilt=(1.0 - up[:, self.socket], 0.01, 0.03),
            peg_tilt=(1.0 - up[:, self.peg], 0.01, 0.03),
        )
        near = {}  # 1 for an error within its tolerance, falling linearly to 0.1 a margin past it, then to 0
        for name, (error, tolerance, margin) in errors.items():
            near[name] = (1 - 0.9 * (error - tolerance).clamp(min=0.0) / margin).clamp(min=0.0)
        left_offset = (left.joint_pos[:, :6] - left.default_joint_pos[:6]).norm(dim=1)  # rad: joints 1 to 6 off home
        right_offset = (right.joint_pos[:, :6] - right.default_joint_pos[:6]).norm(dim=1)
        home_offset = torch.stack([left_offset, right_offset], 1)
        at_home = 0.1 ** (((home_offset - 0.01).clamp(min=0.0) / 2.0) ** 2)  # 1 within 0.01, 0.1 at 2 past (Gaussian)
        both_near = near["left_gripper"] * near["right_gripper"]
        terms = dict(
            left_reward=near["left_gripper"],
            right_reward=near["right_gripper"],
            left_target_qpos=at_home[:, 0] * both_near,
            right_target_qpos=at_home[:, 1] * both_near,
            no_table_collision=1 - (self.table_contact.data.found > 0).any(1).float(),
            socket_z_up=near["socket_tilt"] * near["socket_lift"],
            peg_z_up=near["peg_tilt"] * near["peg_lift"],
            socket_entrance_reward=near["socket_lift"],
            peg_end2_reward=near["peg_lift"],
            peg_insertion_reward=near["insertion"] * aligned,
        )
        term_values = torch.nan_to_num(torch.stack([terms[name] for name in self.names])) * self.weights[:, None]

        # --- terminations ---
        diverged = ~(data.qpos.isfinite().all(1) & data.qvel.isfinite().all(1))
        out = (socket.abs() > 1.0).any(1) | (peg.abs() > 1.0).any(1)  # the socket or the peg left 1 m bounds
        reward = term_values.sum(0) / self.weights.sum()
        return reward, out | diverged, aligned & (depth < 0.04), term_values.mean(1)

    @torch.compile
    def restart(self, mask: torch.Tensor):
        """Move the socket and the peg each up to 10 cm in x and y from home; masked writes only."""
        qpos = self.sim.data.torch.qpos
        for q_adr in (self.socket_q_adr, self.peg_q_adr):
            offset = 0.2 * torch.rand(self.n, 2, device=self.device) - 0.1  # m
            qpos[:, q_adr : q_adr + 2] += mask[:, None] * offset
        self.steps.masked_fill_(mask, 0)
        self.alive.copy_(mask.all() | (self.alive & ~mask))  # a full reset starts the episodes the metrics count

    @torch.compile
    def obs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Joints, grippers, socket and peg, the socket's mouth, the peg's end, the socket's and the peg's z rows."""
        data = self.sim.data.torch
        z_row = data.xmat[:, :, 2, :]  # each body's
        obs = [
            data.qpos,
            data.qvel,
            self.left.data.site_pos_w[:, self.left_gripper],
            data.xpos[:, self.socket],
            self.right.data.site_pos_w[:, self.right_gripper],
            data.xpos[:, self.peg],
            data.site_xpos[:, self.socket_entrance],
            data.site_xpos[:, self.peg_end],
            z_row[:, self.socket],
            z_row[:, self.peg],
        ]
        obs = torch.cat(obs, 1)
        return obs, obs  # the critic's the same


if __name__ == "__main__":
    config = tyro.cli(AlohaPegConfig)
    env = AlohaPeg(config)  # the full physics, where the policy is scored
    coarse_env = AlohaPeg(config, coarse=config.coarse_physics) if config.coarse else None
    # discount, policy widths, unroll length and a state-dependent, tanh-squashed std
    learner_config = ppo.PPOConfig(gamma=0.97, hidden=(256, 256, 256, 256), horizon=40, std="state")
    actor = ppo.PPO(env.obs_dims, len(env.bound), learner_config).train(env, config.minutes, coarse_env, config.coarse)
    # each step's metric x steps: sums over each episode
    episode_reward = partial(torch.mul, env.counted, config.episode)
    success = partial(torch.mul, env.success, config.episode)
    print(ppo.evaluate(env, actor, dict(episode_reward=episode_reward, success=success), steps=config.episode))
