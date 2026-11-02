"""MJLab scene/variant simulation for cached, fixed-wrist Wuji2 grasps."""
from __future__ import annotations

from functools import partial
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch
import warp as wp

from dex_supermimic.rl.dex_supermimic.rotations import quat_to_mat, so3_log, mm
from dex_supermimic.rl.dex_supermimic.state import pack, unpack

from .contact import ContactMeter
from .dataset import GraspDataset

QREF = slice(177, 197)
TARGET_P = slice(197, 200)
TARGET_R = slice(200, 209)
CONTEXT_STATIC = 210
CONTEXT_DIM = 215
DT = .02


def hand_spec():
    from retarget.env.grasp.graspadmm_scene import canonical_hand_xml
    return mujoco.MjSpec.from_string(canonical_hand_xml())


def object_spec(mesh):
    root = ET.Element("mujoco", model="grasp_primitive")
    ET.SubElement(root, "compiler", angle="radian")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", name="primitive_mesh", file=str(mesh))
    body = ET.SubElement(ET.SubElement(root, "worldbody"), "body", name="grasp_object", pos="0 0 .5")
    ET.SubElement(body, "freejoint", name="object_joint")
    ET.SubElement(body, "geom", name="object_collision", type="mesh", mesh="primitive_mesh",
                  mass=".1", friction=".6 .02 .001", condim="4", rgba=".3 .6 .9 1")
    return mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))


def make_env_cfg(num_envs=644, control_hz=50, play=False, manifest=None, case_ids=None):
    """Return an explicit config consumed by GraspHoldEnv, not cube managers."""
    if control_hz != 50:
        raise ValueError("Cached grasp hold uses 50 Hz control and 2 ms physics")
    if manifest is None:
        from retarget.config.run import ROOT
        manifest = ROOT / "output/supermimic_grasp_dataset_v2/manifest.json"
    return dict(num_envs=num_envs, manifest=str(manifest), case_ids=case_ids, play=play)


def observation(x, context, previous, aux):
    q, qd, *_ = unpack(x)
    return torch.cat((context[..., :CONTEXT_STATIC], q - context[..., QREF], qd * .25,
                      x[..., 40:], previous, torch.log1p(aux[..., :12].clamp_min(0))), -1)


def control(action, previous_target, context, lower, upper):
    raw = (context[..., QREF] + .5 * action.clamp(-1, 1)).clamp(lower, upper)
    return (.5 * raw + .5 * previous_target).clamp(lower, upper)


def rewards(x, context, aux, action, previous):
    q, qd, p, r, _, _ = unpack(x)
    dp = (p - context[..., TARGET_P]).norm(dim=-1)
    target = context[..., TARGET_R].reshape(*context.shape[:-1], 3, 3)
    angle = so3_log(mm(target.transpose(-1, -2), r)).norm(dim=-1)
    required = context[..., 171:177]
    shortage = (torch.relu(1 - aux[..., :6] / .05).square() * required).sum(-1) / required.sum(-1).clamp_min(1)
    forbidden = torch.relu(aux[..., 6:12] / .01 - 1).square().clamp(max=100).sum(-1)
    deficit = torch.relu(.05 - aux[..., 12:268]).square() / .05**2
    terms = dict(position=4 * torch.exp(-(dp / .002).square()),
                 orientation=2 * torch.exp(-(angle / np.deg2rad(5)).square()),
                 contact=-2 * shortage, forbidden=-forbidden,
                 support=-(deficit.amax(-1) + .25 * deficit.mean(-1)),
                 penetration=-2 * torch.relu(aux[..., 268] - .5).square().clamp(max=100),
                 collision=-aux[..., 269],
                 action_rate=-.01 * (action - previous).square().sum(-1),
                 pose=-.05 * ((q - context[..., QREF]) / .5).square().mean(-1),
                 velocity=-.001 * qd.square().mean(-1))
    return sum(terms.values()) * DT, terms


def physical_checks(x, context, aux):
    _, _, p, r, _, _ = unpack(x)
    position = (p - context[..., TARGET_P]).norm(dim=-1)
    angle = so3_log(mm(context[..., TARGET_R].reshape(-1, 3, 3).transpose(-1, -2), r)).norm(dim=-1)
    required = context[..., 171:177] > 0
    loaded = ((aux[..., :6] >= .05) | ~required).all(-1)
    forbidden = (aux[..., 6:12] <= .01).all(-1)
    finite = torch.isfinite(x).all(-1) & torch.isfinite(aux).all(-1)
    return dict(position=position <= .002, rotation=angle <= np.deg2rad(5),
                required=loaded, forbidden=forbidden, finite=finite)


class GraspHoldEnv:
    def __init__(self, cfg, device="cuda:3", seed=42):
        from mjlab.entity import EntityCfg
        from mjlab.entity.variants import VariantEntityCfg
        from mjlab.scene import Scene, SceneCfg
        from mjlab.sim import SimulationCfg
        from mjlab.sim.sim import Simulation
        self.device = torch.device(device)
        wp.init()
        print("[grasp] reading verified cache", flush=True)
        self.dataset = ds = GraspDataset(cfg["manifest"])
        selected = cfg.get("case_ids") or list(range(len(ds.cases)))
        self.num_envs = n = int(cfg["num_envs"])
        if n < len(selected) or n % len(selected):
            raise ValueError("num_envs must be a positive multiple of selected cases")
        assignment = np.tile(selected, n // len(selected))
        self.case_index = torch.as_tensor(assignment, device=device)
        self.assignment = assignment
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        variant_ids = {case: i for i, case in enumerate(selected)}
        variants = {ds.cases[i]["case_id"]: partial(object_spec, ds.root / ds.cases[i]["mesh"]) for i in selected}
        scene_cfg = SceneCfg(num_envs=n, entities={
            "robot": EntityCfg(spec_fn=hand_spec),
            "object": VariantEntityCfg(variants=variants, assignment=lambda count: [variant_ids[i] for i in assignment]),
        })
        print(f"[grasp] authoring {len(selected)} variants, {n} worlds", flush=True)
        self.scene = Scene(scene_cfg, device=str(device))
        sim_cfg = SimulationCfg(nconmax=128, njmax=512)
        sim_cfg.mujoco.timestep = .002
        sim_cfg.mujoco.iterations = 50
        sim_cfg.mujoco.ls_iterations = 50
        sim_cfg.mujoco.tolerance = 1e-10
        print("[grasp] compiling MJLab simulation", flush=True)
        self.sim = Simulation(num_envs=n, cfg=sim_cfg, spec=self.scene.spec,
                              variant_info=self.scene.collect_variant_info(), device=str(device))
        # MJLab captures graphs on its registered Warp stream. Use that stream
        # from Torch after capture; replacing it with Torch's legacy default
        # stream before graph construction is not supported by Warp.
        previous_stream = torch.cuda.current_stream(self.device)
        self.torch_stream = torch.cuda.ExternalStream(wp.get_stream(str(self.device)).cuda_stream, device=self.device)
        self.torch_stream.wait_stream(previous_stream)
        torch.cuda.set_stream(self.torch_stream)
        self.scene.initialize(mj_model=self.sim.mj_model, model=self.sim.model, data=self.sim.data)
        print("[grasp] scene initialized", flush=True)
        cpu = self.sim.mj_model
        joints = np.flatnonzero(cpu.jnt_type != mujoco.mjtJoint.mjJNT_FREE)
        if len(joints) != 20 or cpu.nu != 20:
            raise ValueError(f"Expected 20 finger joints and native actuators; got {len(joints)}, {cpu.nu}")
        self.joints = torch.as_tensor(cpu.jnt_qposadr[joints], device=device, dtype=torch.long)
        self.dofs = torch.as_tensor(cpu.jnt_dofadr[joints], device=device, dtype=torch.long)
        free = np.flatnonzero(cpu.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1:
            raise ValueError("Object must be the only free joint; wrist must be fixed")
        self.object_q = int(cpu.jnt_qposadr[free[0]])
        self.object_v = int(cpu.jnt_dofadr[free[0]])
        self.object_body = int(cpu.jnt_bodyid[free[0]])
        actuator_joints = cpu.actuator_trnid[:, 0].tolist()
        self.ctrl_ids = torch.as_tensor([actuator_joints.index(int(j)) for j in joints], device=device)
        self.lower = torch.as_tensor(cpu.jnt_range[joints, 0], device=device, dtype=torch.float32)
        self.upper = torch.as_tensor(cpu.jnt_range[joints, 1], device=device, dtype=torch.float32)
        self.qpos, self.qvel, self.ctrl = self.sim.data.qpos, self.sim.data.qvel, self.sim.data.ctrl
        self.group_masks = torch.as_tensor(ds.groups[assignment], device=device)
        self.allowed = torch.as_tensor(ds.allowed[assignment], device=device)
        types = sorted({c["type_id"] for c in ds.cases})
        kinds = ("cylinder", "sphere", "disk", "plate", "cuboid")
        identity = []
        for case in ds.cases:
            identity.append(np.r_[np.eye(31)[types.index(case["type_id"])], np.eye(5)[kinds.index(case["primitive"]["kind"])],
                                  np.asarray(case["primitive"]["dimensions_m"]) / .1])
        identity = torch.as_tensor(np.asarray(identity)[assignment], device=device, dtype=torch.float32)
        self.base = torch.cat((identity, self.group_masks.flatten(1), self.allowed,
                               torch.as_tensor(ds.required[assignment], device=device)), -1)
        radii = []
        for case in ds.cases:
            vertices = np.asarray([[float(x) for x in line.split()[1:4]]
                                   for line in (ds.root / case["mesh"]).read_text().splitlines() if line.startswith("v ")])
            radii.append(max(float(np.linalg.norm(vertices, axis=1).max()), .01))
        self.radii = torch.as_tensor(np.asarray(radii)[assignment], device=device, dtype=torch.float32)
        self.meter = ContactMeter(self.sim, self.group_masks, self.allowed, self.radii)
        self.age = torch.zeros(n, device=device, dtype=torch.long)
        self.reference_ids = np.zeros(n, np.int64)
        self.reference_valid = torch.ones(n, device=device, dtype=torch.bool)
        self.qref = torch.zeros((n, 20), device=device)
        self.target_pose = torch.zeros((n, 7), device=device)
        self.target_rotation = torch.zeros((n, 9), device=device)
        self.previous = torch.zeros((n, 20), device=device)
        self.previous_target = self.previous.clone()
        self.direction = torch.zeros((n, 3), device=device)
        self.force_scale = 0.
        print("[grasp] resetting cached references", flush=True)
        self.reset()
        print("[grasp] environment ready", flush=True)

    @torch.no_grad()
    def reset(self, ids=None, split="train", reference_ids=None, perturb=False, directions=None):
        ids = torch.arange(self.num_envs, device=self.device) if ids is None else ids
        if not len(ids):
            return
        cpu_ids = ids.cpu().numpy()
        pools = self.dataset.pools(split)
        refs = np.array([self.rng.choice(pools[self.assignment[i]]) for i in cpu_ids]) if reference_ids is None else np.asarray(reference_ids)
        self.reference_ids[cpu_ids] = refs
        qref = torch.as_tensor(self.dataset.arrays["qref"][refs], device=self.device)
        pose = torch.as_tensor(self.dataset.arrays["object_pose"][refs], device=self.device)
        self.qref[ids] = qref
        self.target_pose[ids] = pose
        self.target_rotation[ids] = quat_to_mat(pose[:, 3:]).flatten(1)
        self.reference_valid[ids] = torch.as_tensor([self.dataset.records[i]["valid"] for i in refs], device=self.device)
        self.reference_valid[ids] &= ((qref >= self.lower - 1e-5) & (qref <= self.upper + 1e-5)).all(-1)
        self.sim.reset(ids)
        self.qpos[ids[:, None], self.joints] = qref
        self.qpos[ids, self.object_q:self.object_q + 7] = pose
        if perturb:
            self.qpos[ids[:, None], self.joints] = (qref + torch.empty_like(qref).uniform_(-.01, .01)).clamp(self.lower, self.upper)
            self.qpos[ids, self.object_q:self.object_q + 3] += torch.empty_like(pose[:, :3]).uniform_(-.0005, .0005)
            from dex_supermimic.rl.dex_supermimic.rotations import so3_exp
            delta = torch.empty((len(ids), 3), device=self.device).uniform_(-1, 1) * np.deg2rad(1) / np.sqrt(3)
            rot = quat_to_mat(pose[:, 3:]) @ so3_exp(delta)
            from scipy.spatial.transform import Rotation
            quat = Rotation.from_matrix(rot.cpu().numpy()).as_quat(scalar_first=True)
            self.qpos[ids, self.object_q + 3:self.object_q + 7] = torch.as_tensor(quat, device=self.device, dtype=pose.dtype)
        self.qvel[ids] = 0
        self.ctrl[ids[:, None], self.ctrl_ids] = qref.clamp(self.lower, self.upper)
        self.previous[ids] = 0
        self.previous_target[ids] = qref.clamp(self.lower, self.upper)
        self.age[ids] = 0
        axes = torch.cat((torch.eye(3, device=self.device), -torch.eye(3, device=self.device)))
        self.direction[ids] = axes[torch.randint(6, (len(ids),), device=self.device)] if directions is None else directions
        self.sim.forward()
        self.aux = self.meter.sample(self.state()[:, 40:43])

    def state(self):
        o = self.qpos[:, self.object_q:self.object_q + 7]
        v = self.qvel[:, self.object_v:self.object_v + 6]
        r = quat_to_mat(o[:, 3:])
        return pack(self.qpos[:, self.joints], self.qvel[:, self.dofs], o[:, :3], r,
                    v[:, :3], (r @ v[:, 3:, None]).squeeze(-1))

    def context(self):
        return torch.cat((self.base, self.qref, self.target_pose[:, :3], self.target_rotation,
                          self.radii[:, None], self.direction,
                          self.qref.new_full((self.num_envs, 1), self.force_scale), self.age[:, None].float() / 110), -1)

    def observe(self):
        return observation(self.state(), self.context(), self.previous, self.aux)

    @torch.no_grad()
    def step(self, action, *, training=True):
        x, ctx, old_action = self.state(), self.context(), self.previous.clone()
        old_target = self.previous_target.clone()
        old_aux = self.aux.clone()
        target = control(action, self.previous_target, ctx, self.lower, self.upper)
        self.ctrl[:, self.ctrl_ids] = target
        applied = self.sim.data.xfrc_applied
        applied.zero_()
        applied[:, self.object_body, :3] = ctx[:, 210:213] * ctx[:, 213:214] * .98 * (self.age >= 10)[:, None]
        for _ in range(10):
            self.sim.step()
        self.age += 1
        x1 = self.state()
        self.aux = self.meter.sample(x1[:, 40:43])
        reward, terms = rewards(x1, ctx, self.aux, action, old_action)
        finite = torch.isfinite(x1).all(-1) & torch.isfinite(self.aux).all(-1) & self.meter.valid & self.reference_valid
        displacement = (x1[:, 40:43] - ctx[:, TARGET_P]).norm(dim=-1)
        terminated = (~finite) | (displacement > .05)
        timeout = self.age >= 110
        self.previous.copy_(action)
        self.previous_target.copy_(target)
        return dict(x=x, x1=x1, context=ctx, next_context=self.context(), previous=old_action,
                    previous_target=old_target, action=action, u=target, aux=self.aux.clone(), old_aux=old_aux,
                    reward=reward, done=terminated | timeout, terminated=terminated, valid=finite,
                    terms=terms)

    def close(self):
        # The current MJLab Simulation owns no external process or viewer.
        pass
