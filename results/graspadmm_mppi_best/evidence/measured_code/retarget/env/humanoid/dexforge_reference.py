"""Use the existing G1 reference as ForceAware's kinematic input, without IK.

The original arrays are retained unchanged. Official ForceAware interpolation,
window grids and MuJoCo quaternion-aware initial velocity construction are used
at runtime. The free root is a state/reference, separate from the 29 controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .dexforge_scene import OBSERVATION_SITES, _upstream_paths


@dataclass(frozen=True)
class HumanoidWindowTargets:
    """An ordinary upstream window plus the unactuated free-root reference."""

    upstream: Any
    root_qpos: np.ndarray

    def __getattr__(self, name):
        return getattr(self.upstream, name)


@dataclass(frozen=True)
class HumanoidReference:
    model: mujoco.MjModel
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    ref_dt: float
    index: Any
    sequence: Any
    root_qpos_adr: int
    root_dof_adr: int
    source: Path

    @property
    def duration_seconds(self):
        return (len(self.qpos)-1) * self.ref_dt

    @property
    def end_frame(self):
        return len(self.qpos)-1

    def root_at(self, frames):
        """Interpolate root [position, WXYZ] with upstream quaternion nlerp."""
        from forceaware.targets import _interpolate_pose

        root = self.qpos[:, self.root_qpos_adr:self.root_qpos_adr+7]
        xyzw = root[:, [4, 5, 6, 3, 0, 1, 2]]
        pose = _interpolate_pose(xyzw, np.asarray(frames, np.float32).reshape(-1))
        return pose[:, [4, 5, 6, 3, 0, 1, 2]].copy()

    def qpos_at(self, frames):
        from forceaware.targets import _interpolate, _object_qpos

        values = np.asarray(frames, np.float32).reshape(-1)
        positions = np.repeat(np.asarray(self.model.qpos0, np.float32)[None], len(values), axis=0)
        positions[:, self.root_qpos_adr:self.root_qpos_adr+7] = self.root_at(values)
        positions[:, self.index.hand_qpos] = _interpolate(self.sequence.hand, values)
        positions[:, self.index.object_qpos:self.index.object_qpos+7] = _object_qpos(self.sequence.object_pose_xyzw, values)
        return positions

    def initial_state(self, *, start_frame=0., hold=False):
        """Official initial velocity policy, extended to the full free-root q.

        Raw NPZ velocities are zero placeholders. As in official initial_state,
        differentiate the next reference frame with mj_differentiatePos; holding
        a frame explicitly requests zero velocity. No source data is rewritten.
        """
        start = float(start_frame)
        if not np.isfinite(start) or not 0 <= start <= self.end_frame:
            raise ValueError("reference start_frame must lie within the original clip")
        qpos = self.qpos_at([start])[0]
        velocity = np.zeros(self.model.nv, np.float64)
        following = min(np.floor(start)+1., self.end_frame)
        if not hold and following > start:
            next_qpos = self.qpos_at([following])[0]
            mujoco.mj_differentiatePos(self.model, velocity, (following-start)*self.ref_dt,
                                     qpos.astype(np.float64), next_qpos.astype(np.float64))
        return qpos, velocity.astype(np.float32)

    def make_window(self, *, executed_steps, horizon, time_grid, start_frame=0., end_frame=None, hold_frame=-1):
        from forceaware.targets import make_window

        window = make_window(self.sequence, executed_steps=executed_steps, horizon=horizon,
                             time_grid=time_grid, start_frame=start_frame,
                             end_frame=self.end_frame if end_frame is None else end_frame,
                             hold_frame=hold_frame)
        return HumanoidWindowTargets(window, self.root_at(window.frames))


def load_reference(npz, model, *, ref_dt=.0333333, observation_sites=OBSERVATION_SITES):
    """Map the two-freejoint G1 model to official ForceAware target structures."""
    if not np.isfinite(ref_dt) or ref_dt <= 0:
        raise ValueError("reference timestep must be finite and positive")
    path = Path(npz).expanduser().resolve()
    with np.load(path, allow_pickle=False) as raw:
        values = {key: np.asarray(raw[key], np.float32).copy() for key in ("qpos", "qvel", "ctrl")}
    qpos, qvel, ctrl = (values[key] for key in ("qpos", "qvel", "ctrl"))
    if qpos.ndim != 2 or len(qpos) < 2 or qpos.shape[1] != model.nq or qvel.shape != (len(qpos), model.nv) or ctrl.shape != (len(qpos), model.nu):
        raise ValueError("raw reference/model qpos, qvel and ctrl dimensions must agree")
    if not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError("raw reference must contain only finite states and controls")
    free = np.flatnonzero(model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE))
    if len(free) != 2:
        raise ValueError("humanoid ForceAware requires one robot and one object free joint")
    root, obj = map(int, free)
    qadr, vadr = int(model.jnt_qposadr[root]), int(model.jnt_dofadr[root])
    obj_qadr = int(model.jnt_qposadr[obj])
    if qadr != 0 or obj_qadr+7 != model.nq:
        raise ValueError("humanoid reference requires the root first and object last")
    joint = model.actuator_trnid[:, 0].astype(np.int32)
    if len(set(map(int, joint))) != model.nu or np.any(joint < 0) or np.any(model.jnt_type[joint] != int(mujoco.mjtJoint.mjJNT_HINGE)):
        raise ValueError("ForceAware actions must map one-to-one to scalar robot joint actuators")
    hand_qpos = model.jnt_qposadr[joint].astype(np.int32)
    if set(map(int, hand_qpos)) != set(range(7, obj_qadr)):
        raise ValueError("all scalar robot coordinates must have exactly one actuator")
    for address in (qadr+3, obj_qadr+3):
        if np.any(np.linalg.norm(qpos[:, address:address+4], axis=-1) < 1e-8):
            raise ValueError("raw reference contains a zero root/object quaternion")
    _upstream_paths()
    from forceaware.targets import ContactGuidance, ModelIndex, SequenceData

    index = ModelIndex(hand_qpos, np.arange(model.nu, dtype=np.int32), obj_qadr,
                       int(model.jnt_bodyid[obj]))
    site_ids = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in observation_sites], np.int32)
    if len(site_ids) != 4 or np.any(site_ids < 0):
        raise ValueError("raw humanoid reference requires four existing observation sites")
    shape = (len(qpos), len(site_ids))
    # There are no CA anchors in this dataset. Inactive guidance keeps the
    # official window contract without inventing projected contacts or phases.
    contact = ContactGuidance(mask=np.zeros(shape, np.float32), weight=np.zeros(shape, np.float32),
                              position_object=np.zeros(shape+(3,), np.float32),
                              output_index=np.zeros(shape, np.int32),
                              query_body=np.broadcast_to(model.site_bodyid[site_ids], shape).astype(np.int32).copy(),
                              query_local=np.broadcast_to(model.site_pos[site_ids], shape+(3,)).astype(np.float32).copy())
    object_pose = qpos[:, obj_qadr:obj_qadr+7][:, [4,5,6,3,0,1,2]].copy()
    sequence = SequenceData(qpos[:, hand_qpos].copy(), object_pose, contact,
                            hand_index_per_frame=1., hand_source="dataset")
    return HumanoidReference(model, qpos, qvel, ctrl, float(ref_dt), index, sequence, qadr, vadr, path)
