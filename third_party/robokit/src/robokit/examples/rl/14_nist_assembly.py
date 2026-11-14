# /// script
# requires-python = ">=3.10"
# dependencies = ["robokit[mjwarp,pt-cu126]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
# pyright: reportOptionalMemberAccess=false
# pyright: reportCallIssue=false
# pyright: reportArgumentType=false
"""AssembleNIST's six NIST board assemblies (suniique/assembly-nist): PPO on coarse physics, scored on fine.

Example usage:
    uv run src/robokit/examples/rl/14_nist_assembly.py --minutes 75
    uv run src/robokit/examples/rl/14_nist_assembly.py --eval runs/nist/model_001150.pt
    uv run src/robokit/examples/rl/14_nist_assembly.py --eval runs/nist/model_001150.pt --eval-mean  # mean actions
"""

import copy
import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Tuple

import mujoco  # pyright: ignore[reportMissingImports]
import numpy as np
import torch
import tyro
import warp as wp
from torch import nn

from robokit.assets.datasets import nist_assembly
from robokit.helpers.rl import Env, ppo
from robokit.sim import CoarseConfig, MjWarpSim
from robokit.xform.warp.torch_wrappers import quaternion_to_matrix


@dataclass(frozen=True)
class NistConfig:
    """Train on coarse physics, saving checkpoints, or score checkpoint `eval` on fine physics."""

    data: Path = nist_assembly.DIR  # AssembleNIST's task scenes, point tables and reset banks
    control_dt: float = 0.04  # s per action
    grip_open: float = 0.04  # gripper targets, m per finger
    grip_close: float = -0.01
    history: int = 5  # proprio frames observed
    # Coarse physics: only the part, fixture, obstacles within 8 cm, board, table, hand and fingers collide
    coarse_physics: CoarseConfig = CoarseConfig(
        collide=r"(\w+_)?(part\d+|fix\d+|obst_near_\w+)|board|table|hand|(left|right)_finger",
        condim=3,
        condim_geoms="(left|right)_finger",
        timestep=0.0025,
        min_timeconst=0.005,  # the gears' stiff contacts back to the others'
        min_impedance_width=0.0005,
        pyramidal=True,
        iterations=4,
        ls_iterations=4,
    )
    eval: str = ""  # a checkpoint to score instead of training, by AssembleNIST's evaluation
    eval_seeds: int = 5  # the evaluation's draws: per seed, `eval_draws` episodes per task
    eval_draws: int = 32
    eval_mean: bool = False  # mean actions; AssembleNIST's evaluation samples them
    run_dir: str = "runs/nist"  # where training saves a checkpoint every 50 iterations
    worlds: int = 8192  # MjWarpSim worlds to train on
    minutes: float = 75.0  # wall time to train
    seed: int = 0  # torch's


class Nist(Env):
    """The tasks on `config.worlds` worlds, fine or `coarse` physics, episodes from bank `states` of tasks `task_of`."""

    def __init__(
        self, config: NistConfig, states: np.ndarray, task_of: np.ndarray, coarse: Optional[CoarseConfig] = None
    ):
        self.config = config
        n = config.worlds
        self.n = n
        tables = np.load(config.data / "tables.npz")
        self.tasks = [str(task) for task in tables["tasks"]]

        # --- build scene ---
        # the scenes at 0.5 ms: one with every task's geoms, then each task's; a world takes its task's rows on reset
        roots = []
        task_geoms = {"fix": ".//body[@name='fixture']", "part": ".//body[@name='part']", "obst": "worldbody"}
        for task in self.tasks:
            root = ET.fromstring((config.data / f"{task}.xml").read_text())
            root.find("compiler").set("meshdir", str(config.data / "assets"))
            meshes = {mesh.get("name"): mesh for mesh in root.find("asset").findall("mesh") if mesh.get("name")}
            # the task's geoms and meshes get its name, and join the first scene
            for prefix, path in task_geoms.items():
                for geom in root.find(path).findall("geom"):
                    if not geom.get("name", "").startswith(prefix):  # the table and board are in every scene
                        continue
                    mesh = meshes.get(geom.get("mesh"))
                    geom.set("name", f"{task}_{geom.get('name')}")
                    if prefix != "obst" and mesh is not None:  # obstacle meshes are named by asset already
                        mesh.attrib["name"] = f"{task}_{mesh.attrib['name']}"
                        geom.attrib["mesh"] = mesh.attrib["name"]
                    if roots:
                        roots[0].find(path).append(copy.deepcopy(geom))
                        if mesh is not None and roots[0].find(f"asset/mesh[@name='{mesh.get('name')}']") is None:
                            roots[0].find("asset").append(copy.deepcopy(mesh))
            sensor = ET.SubElement(root, "sensor")  # the arm joints' positions, then velocities, after each substep
            for kind in ("jointpos", "jointvel"):
                for i in range(7):
                    ET.SubElement(sensor, kind, joint=f"joint{i + 1}")
            if not roots:  # the first task's scene also starts the scene with every task's geoms
                roots.append(copy.deepcopy(root))
            roots.append(root)
        scene_specs = [mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode")) for root in roots]
        if coarse is not None:
            scene_specs = [MjWarpSim.build_coarse_spec(scene_spec, coarse) for scene_spec in scene_specs]

        # --- build simulator ---
        self.horizon = round(15.0 / config.control_dt)  # 15 s episodes
        substeps = round(config.control_dt / scene_specs[0].option.timestep)
        worlds = [n] + [0] * len(self.tasks)  # all in the first scene's model
        njmax = 2048 if coarse is None else 480
        nconmax = 512 if coarse is None else 96
        sim = MjWarpSim(scene_specs, worlds, substeps=substeps, njmax=njmax, nconmax=nconmax)
        self.sim = sim
        if coarse is not None:
            sim.opt.warn_overflow = 0  # constraints past njmax are dropped, as in AssembleNIST's run
        sim.set_torch_stream()
        self.device = sim.data.torch.qpos.device
        mjm = sim.mjm

        # --- set up actions ---
        # arm joint targets: the joints + scale x action, within the joint ranges; a latched gripper target
        arm_joints = [mjm.joint(f"joint{i + 1}") for i in range(7)]
        self.arm_joints = [joint.id for joint in arm_joints]
        self.arm_q_adr = torch.tensor([joint.qposadr[0] for joint in arm_joints], device=self.device)
        self.arm_v_adr = torch.tensor([joint.dofadr[0] for joint in arm_joints], device=self.device)
        joint_range = np.array([joint.range for joint in arm_joints])
        joint_range = torch.tensor(joint_range, dtype=torch.float32, device=self.device)
        self.joint_low = joint_range[:, 0]
        self.joint_high = joint_range[:, 1]
        self.action_scale = torch.tensor([0.05, 0.05, 0.05, 0.06, 0.07, 0.1, 0.17], device=self.device)  # rad
        self.bound = torch.ones(8, device=self.device)

        # --- set up rewards ---
        # the arm gains, for the control cost's mean torques, and limits
        self.stiffness = torch.tensor([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0], device=self.device)
        self.damping = torch.tensor([50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0], device=self.device)
        self.torque_limit = torch.tensor([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], device=self.device)  # N m
        self.speed_limit = torch.tensor([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61], device=self.device)  # rad/s
        # each task's success keypoints in the part's frame, and their assembled positions
        self.keypoints = torch.tensor(tables["keypoints"], device=self.device)
        self.keypoint_goals = torch.tensor(tables["keypoint_goals"], device=self.device)
        self.part_q_adr = int(mjm.joint("part_free").qposadr[0])
        self.names = ["success", "cost", "safety"]

        # --- set up resets ---
        # the reset bank and Success Guided Sampling, its CDF for draws on the GPU; diverged worlds go to `qpos0`
        self.bank = torch.tensor(states[:, 1 : 1 + mjm.nq + mjm.nv], dtype=torch.float32, device=self.device)
        self.bank_task = torch.as_tensor(task_of, device=self.device)
        self.outcomes = np.zeros((len(states), 50), dtype=np.uint8)  # each entry's last 50 episodes' successes
        self.visits = np.zeros(len(states), dtype=np.int64)
        self.cdf = torch.zeros(len(states), dtype=torch.float64, device=self.device)
        self.log = []  # (entry, success, ended) per reset since the last `curriculum`
        self.qpos0 = wp.to_torch(sim.model.qpos0)  # each world its task's, set on reset

        # --- set up observations ---
        # the points `Net` places: each task's part and mating geometry (in its frame and the world's), the fingers'
        self.points = {k: torch.tensor(v, device=self.device) for k, v in tables.items() if k.endswith("_points")}
        # observation: task, tcp position and rotation, part's position and rotation, fingers, proprio frames
        self.split = (1, 3, 9, 12, 2, config.history * 30)  # a proprio frame: tcp pose 7, twist 6, joints 9, action 8
        self.obs_dims = (sum(self.split), sum(self.split))
        finger_joints = [mjm.joint(f"finger_joint{i + 1}") for i in range(2)]
        self.finger_q_adr = torch.tensor([joint.qposadr[0] for joint in finger_joints], device=self.device)
        self.hand = mjm.body("hand").id
        self.tcp = mjm.site("tcp").id
        self.events = [torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)]  # rollout, update
        self.episodes = 0  # those ended in the last rollout
        self.success_rate = 0.0  # of the episodes ended in the last rollout

        # --- initialize state ---
        # per world: task, reset-bank entry, step, success, gripper latch, last action, proprio history
        self.task = torch.zeros(n, dtype=torch.long, device=self.device)
        self.entry = torch.zeros_like(self.task)
        self.steps = torch.zeros_like(self.task)
        self.succeeded = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.gripper_target = torch.zeros(n, device=self.device)
        self.last_action = torch.zeros(n, 8, device=self.device)
        self.proprio_history = torch.zeros(n, config.history, 30, device=self.device)
        self.curriculum(0)
        self.reset(torch.ones(n, dtype=torch.bool, device=self.device))
        self.log.clear()
        desync = np.random.default_rng(7).integers(0, self.horizon, n)  # staggered episode clocks
        self.steps = torch.as_tensor(desync, device=self.device)

    def curriculum(self, step: int):
        """Record the episodes ended since the last call; weight entries 1e-4 + sqrt(p (1 - p)), p their success."""
        if self.log:
            entries, outcomes, ended = torch.cat(self.log, 1).cpu().numpy()  # the one host sync, once per update
            done = ended > 0
            for entry, outcome in zip(entries[done], outcomes[done]):  # in order: an entry may end several episodes
                self.outcomes[entry, self.visits[entry] % 50] = outcome
                self.visits[entry] += 1
            self.episodes = int(ended.sum())
            self.success_rate = float(outcomes[done].mean()) if done.any() else 0.0
            self.log.clear()
        success = self.outcomes.sum(1) / np.clip(self.visits, 1, 50)
        weights = 1e-4 + np.sqrt(success * (1 - success))
        self.cdf.copy_(torch.as_tensor(np.cumsum(weights) / weights.sum()))
        self.events[0].record()

    def proprio(self) -> torch.Tensor:
        """The proprio frame: tcp position, quaternion and twist, joints, fingers, last action."""
        data = self.sim.data.torch
        joint_vel = data.qvel[:, self.arm_v_adr]
        tcp = data.site_xpos[:, self.tcp]
        axis = data.xaxis[:, self.arm_joints]
        anchor = data.xanchor[:, self.arm_joints]
        tcp_lin_vel = (joint_vel[..., None] * torch.linalg.cross(axis, tcp[:, None] - anchor)).sum(1)
        tcp_ang_vel = (axis * joint_vel[..., None]).sum(1)
        q_wxyz_world_hand = data.xquat[:, self.hand]
        q_wxyz_world_hand = q_wxyz_world_hand * torch.where(q_wxyz_world_hand[:, 1:2] < 0, -1.0, 1.0)  # x >= 0
        joints = [data.qpos[:, self.arm_q_adr], data.qpos[:, self.finger_q_adr]]
        return torch.cat([tcp, q_wxyz_world_hand, tcp_lin_vel, tcp_ang_vel, *joints, self.last_action], 1)

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The control step: reward success minus control cost; end on success or a safety failure, or at 15 s."""
        data = self.sim.data.torch

        # --- act ---
        arm_target = data.qpos[:, self.arm_q_adr] + self.action_scale * action[:, :7]
        arm_target = torch.clamp(arm_target, self.joint_low, self.joint_high)
        data.ctrl[:, :7] = arm_target
        gripper_target = torch.where(
            action[:, 7] < -0.5, self.config.grip_close, self.gripper_target
        )  # |action| <= 0.5 keeps it
        self.gripper_target = torch.where(action[:, 7] > 0.5, self.config.grip_open, gripper_target)
        data.ctrl[:, 7] = self.gripper_target
        self.sim.step()
        # the arm joints after each substep: the sensors after substep k hold the state after substep k - 1
        sensors = data.substep_sensordata
        joint_pos = torch.cat([sensors[1:, :, :7].transpose(0, 1), data.qpos[:, None, self.arm_q_adr]], 1)
        joint_vel = torch.cat([sensors[1:, :, 7:].transpose(0, 1), data.qvel[:, None, self.arm_v_adr]], 1)

        # --- diverged physics (NaN or past 1e10) back to the defaults, as CPU MuJoCo does ---
        diverged = ~(torch.cat([data.qpos, data.qvel], 1).abs() < 1e10).all(1)
        data.qpos.copy_(torch.where(diverged[:, None], self.qpos0, data.qpos))
        data.qvel.masked_fill_(diverged[:, None], 0.0)
        self.sim.kinematics()

        # --- rewards ---
        # the cost of actions, joint speeds and mean torques; success: every keypoint within 1 mm of its assembled place
        torque = self.stiffness * (arm_target[:, None] - joint_pos) - self.damping * joint_vel
        torque = torch.clamp(torque, -self.torque_limit, self.torque_limit).mean(1)
        torque = torch.where(diverged[:, None], 0.0, torque)
        speed = data.qvel[:, self.arm_v_adr]
        effort = (
            (action**2).sum(1) + ((speed / self.speed_limit) ** 2).sum(1) + ((torque / self.torque_limit) ** 2).sum(1)
        )
        cost = 2e-5 * effort
        part = data.qpos[:, self.part_q_adr : self.part_q_adr + 7]
        R_world_part = quaternion_to_matrix(part[:, 3:])
        keypoints = part[:, None, :3] + self.keypoints[self.task] @ R_world_part.transpose(1, 2)
        distance = (keypoints - self.keypoint_goals[self.task]).norm(dim=2).max(1).values
        unsafe = (joint_vel.abs() > self.speed_limit).any(2).any(1) | diverged  # an arm joint over its speed limit
        self.succeeded = (distance < 0.001) & ~unsafe

        # --- terminations ---
        self.steps = self.steps + 1
        self.last_action = action
        self.proprio_history = torch.cat([self.proprio_history[:, 1:], self.proprio()[:, None]], 1)
        timeout = (self.steps >= self.horizon) & ~self.succeeded & ~unsafe
        terms = torch.stack([self.succeeded.float().mean(), cost.mean(), unsafe.float().mean()])
        return self.succeeded.float() - cost, self.succeeded | unsafe, timeout, terms

    def reset(self, mask: torch.Tensor, entry: Optional[torch.Tensor] = None):
        """Start the masked worlds from reset-bank entries (`entry`, else the sampler's), logging the ended episodes."""
        data = self.sim.data.torch
        if entry is None:
            draw = torch.rand(self.n, dtype=torch.float64, device=self.device)
            entry = torch.searchsorted(self.cdf, draw).clamp(max=len(self.cdf) - 1)
        self.log.append(torch.stack([self.entry, self.succeeded.long(), mask.long()]))
        self.entry = torch.where(mask, entry, self.entry)
        self.task = self.bank_task[self.entry]

        # --- scene: the task's model rows (scene 0 has every task's geoms); `reset` then places its static geoms ---
        self.sim.set_scene(wp.from_torch(mask), wp.from_torch((self.task + 1).int()))
        self.sim.reset(wp.from_torch(mask))

        # --- state: the bank's state, gripper latch (closed unless open), cleared last action and history ---
        nq = data.qpos.shape[1]
        state = self.bank[self.entry]
        data.qpos.copy_(torch.where(mask[:, None], state[:, :nq], data.qpos))
        data.qvel.copy_(torch.where(mask[:, None], state[:, nq:], data.qvel))
        self.sim.kinematics()
        open = data.qpos[:, self.finger_q_adr].mean(1) > 0.036
        gripper_target = torch.where(open, self.config.grip_open, self.config.grip_close)
        self.gripper_target = torch.where(mask, gripper_target, self.gripper_target)
        self.last_action = self.last_action * ~mask[:, None]
        self.steps = self.steps * ~mask
        self.proprio_history = torch.where(mask[:, None, None], self.proprio()[:, None], self.proprio_history)

    def obs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The compact observation (`split`), which `Net` turns into point clouds; the critic's is the same."""
        data = self.sim.data.torch
        self.events[1].record()
        obs = [
            self.task[:, None],
            data.site_xpos[:, self.tcp],
            data.site_xmat[:, self.tcp].flatten(1),
            data.qpos[:, self.part_q_adr : self.part_q_adr + 3],
            quaternion_to_matrix(data.qpos[:, self.part_q_adr + 3 : self.part_q_adr + 7]).flatten(1),
            data.qpos[:, self.finger_q_adr],
            self.proprio_history.flatten(1),
        ]
        obs = torch.cat(obs, 1)
        return obs, obs


class RunningNorm(nn.Module):
    """The running normalization of the proprio, clipped to 10 standard deviations."""

    mean: torch.Tensor
    var: torch.Tensor
    count: torch.Tensor

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / torch.sqrt(self.var + 1e-5)).clamp(-10.0, 10.0)

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        """Merge the statistics of the proprio of observations `x`, in place and without host syncs."""
        proprio = x[:, -self.mean.shape[0] :]
        samples = proprio.shape[0]
        delta = proprio.mean(0) - self.mean
        total = self.count + samples
        batch_var = proprio.var(0, unbiased=False)
        self.var.copy_((self.var * self.count + batch_var * samples + delta**2 * self.count * samples / total) / total)
        self.mean.add_(delta * samples / total)
        self.count.copy_(total)


class ResBlock(nn.Module):
    """SimBa's residual block: x + MLP(LayerNorm(x))."""

    def __init__(self, dim: int = 256, expansion: int = 1024):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, expansion)
        self.fc2 = nn.Linear(expansion, dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(self.ln(x))))


class SimBa(nn.Module):
    """AlphaNIST's network (SimBa, Lee et al. 2025): a point encoder, then residual blocks on points and proprio."""

    def __init__(self, out: int, points: int, proprio: int, hidden: int = 256):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(3 * points, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU())
        self.inp = nn.Linear(128 + proprio, hidden)
        self.blocks = nn.Sequential(ResBlock(hidden), ResBlock(hidden))
        self.ln = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, out)


class Policy(nn.Module):
    """AssembleNIST's actor-critic, named as its checkpoints: SimBa actor and critic, a log-std."""

    def __init__(self, points: int, proprio: int, init_std: float = 1.0):
        super().__init__()
        self.norm = RunningNorm(proprio)
        self.actor = SimBa(8, points, proprio)
        self.critic = SimBa(1, points, proprio)
        self.log_std = nn.Parameter(torch.full((8,), math.log(init_std)))
        nn.init.orthogonal_(self.actor.head.weight, 0.01)
        nn.init.zeros_(self.actor.head.bias)
        nn.init.orthogonal_(self.critic.head.weight, 1.0)
        nn.init.zeros_(self.critic.head.bias)


class Net(nn.Module):
    """The policy's actor (with its log-std) or critic, on `Nist`'s observations, for robokit's PPO."""

    def __init__(self, policy: Policy, env: "Nist", critic: bool = False):
        super().__init__()
        self.obs_normalizer = policy.norm  # shared; PPO updates it after each update
        self.net = policy.critic if critic else policy.actor
        self.log_std = policy.log_std
        self.points = env.points
        self.split = env.split
        self.finger_axis = env.points["left_finger_points"].new_tensor([0.0, 1.0, 0.0])  # the fingers open along y
        self.state_std = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        task, tcp_pos, R_world_tcp, part, fingers, proprio = x.split(self.split, 1)
        # the part's and mating geometry's points, then the fingers', in the tcp frame, in dm
        task = task[:, 0].long()
        R_world_part = part[:, 3:].reshape(-1, 3, 3)
        part_points = part[:, None, :3] + self.points["part_points"][task] @ R_world_part.transpose(1, 2)
        world_points = torch.cat([part_points, self.points["mating_points"][task]], 1)
        left = self.points["left_finger_points"] + fingers[:, :1, None] * self.finger_axis
        right = self.points["right_finger_points"] - fingers[:, 1:, None] * self.finger_axis
        tcp_points = (world_points - tcp_pos[:, None]) @ R_world_tcp.reshape(-1, 3, 3)
        points = 10.0 * torch.cat([tcp_points, left, right], 1).flatten(1)
        hidden = self.net.inp(torch.cat([self.net.enc(points), self.obs_normalizer(proprio)], -1))
        with torch.autocast("cuda", torch.bfloat16):  # only the residual blocks, as AssembleNIST's run
            hidden = self.net.blocks(hidden)
        return self.net.head(self.net.ln(hidden))

    def dist(self, x: torch.Tensor, std_range: Tuple[float, float]) -> Tuple[torch.Tensor, torch.Tensor]:
        std = self.log_std.clamp(math.log(std_range[0]), math.log(std_range[1])).exp()  # clamped in log space
        return self(x), std


if __name__ == "__main__":
    config = tyro.cli(NistConfig)
    learner_config = ppo.PPOConfig(  # AssembleNIST's
        gamma=0.995,
        lam=0.95,
        clip=0.2,
        value_coef=1.0,
        entropy=0.005,
        lr=1e-4,
        lr_range=(1e-5, 1e-3),
        kl=0.08,  # 8x AssembleNIST's target: the rate stays at its cap
        epochs=5,
        minibatches=4,
        max_grad=1.0,
        grad_clip="joint",
        reshuffle=True,
        horizon=32,
        std_range=(1e-3, 2.0),
    )
    torch.manual_seed(config.seed)
    tasks = [str(task) for task in np.load(config.data / "tables.npz")["tasks"]]
    if config.eval:
        # AssembleNIST's evaluation: per seed s, rng(10_000 + s) draws each task's eval-bank entries in turn; one
        # episode each on fine physics, a success if assembled within 15 s without a safety failure; ± the seeds' std
        banks = [np.load(config.data / "banks" / f"{task}_eval.npz") for task in tasks]
        states = np.concatenate([bank["states"] for bank in banks])
        task_of = np.concatenate([np.full(len(bank["states"]), k) for k, bank in enumerate(banks)])
        offsets = np.cumsum([0] + [len(bank["states"]) for bank in banks])
        entries = []
        for seed in range(config.eval_seeds):
            rng = np.random.default_rng(10_000 + seed)
            entries += [
                offsets[k] + rng.integers(0, len(bank["states"]), config.eval_draws) for k, bank in enumerate(banks)
            ]
        entries = np.concatenate(entries)
        env = Nist(replace(config, worlds=len(entries)), states, task_of)
        points = sum(len(points) if points.ndim == 2 else points.shape[1] for points in env.points.values())
        policy = Policy(points, env.split[-1]).cuda()
        policy.load_state_dict(torch.load(config.eval, map_location="cuda", weights_only=True)["model"])
        actor = Net(policy, env)
        env.reset(torch.ones(env.n, dtype=torch.bool, device=env.device), torch.as_tensor(entries, device=env.device))
        ended = torch.zeros(env.n, dtype=torch.bool, device=env.device)
        success = torch.zeros_like(ended)
        with torch.no_grad():
            for _ in range(env.horizon):  # an ended episode keeps stepping; its outcome is kept
                mean, std = actor.dist(env.obs()[0], learner_config.std_range)
                _, terminated, timeout, _ = env.step(mean if config.eval_mean else mean + std * torch.randn_like(mean))
                success |= ~ended & env.succeeded
                ended |= terminated | timeout
        family = np.concatenate([bank["family"] for bank in banks])[entries]
        success = success.cpu().numpy()
        per_seed = success.reshape(config.eval_seeds, -1).mean(1)
        actions = "mean" if config.eval_mean else "sampled"
        print(f"{actions} actions, {len(success)} episodes: success {per_seed.mean():.3f} ± {per_seed.std():.3f}")
        print("tasks:", {task: round(float(success[task_of[entries] == k].mean()), 3) for k, task in enumerate(tasks)})
        families = ("loose", "pickup", "inhand", "partial", "near")  # AssembleNIST's reset families, bank["family"]
        print("families:", {name: round(float(success[family == i].mean()), 3) for i, name in enumerate(families)})
    else:
        states = []  # the training banks: the first 640 entries of each task's reset family
        task_of = []
        for k, name in enumerate(tasks):
            bank = np.load(config.data / "banks" / f"{name}_train.npz")
            keep = np.zeros(len(bank["family"]), dtype=bool)
            for family in np.unique(bank["family"]):
                keep[np.flatnonzero(bank["family"] == family)[:640]] = True
            states.append(bank["states"][keep])
            task_of.append(np.full(keep.sum(), k))
        env = Nist(config, np.concatenate(states), np.concatenate(task_of), coarse=config.coarse_physics)
        points = sum(len(points) if points.ndim == 2 else points.shape[1] for points in env.points.values())
        policy = Policy(points, env.split[-1]).cuda()
        actor = Net(policy, env)
        critic = Net(policy, env, critic=True)
        learner = ppo.PPO(env.obs_dims, len(env.bound), learner_config, actor=actor, critic=critic)
        end = torch.cuda.Event(enable_timing=True)
        Path(config.run_dir).mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        while time.perf_counter() - start < config.minutes * 60:  # `PPO.train`'s loop, with logs and checkpoints
            learner.update(env)
            end.record()
            end.synchronize()
            it = learner.it
            if it % 10 == 0 or it <= 3:
                rollout = env.events[0].elapsed_time(env.events[1]) / 1e3
                update = env.events[1].elapsed_time(end) / 1e3
                lr = float(learner.opt.param_groups[0]["lr"])
                std = float(policy.log_std.exp().mean())
                print(
                    f"it {it} {time.perf_counter() - start:.0f} s: rollout {rollout:.2f} s update {update:.2f} s, "
                    f"success {env.success_rate:.3f} of {env.episodes} episodes, lr {lr:.1e}, std {std:.3f}",
                    flush=True,
                )
            if it % 50 == 0:
                torch.save(dict(model=policy.state_dict(), it=it), Path(config.run_dir) / f"model_{it:06d}.pt")
