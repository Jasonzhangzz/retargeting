"""Load AMASS-style SMPL-X motions as ordered transform arrays."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from robokit.assets.body_models import get_body_models_dir
from robokit.smplx import (
    SMPLX_NUM_JOINTS,
    BodyModelSpec,
    BodyModelSpecTensors,
    BodyModelState,
    Gender,
    body_fk_warp,
    body_lbs_warp,
    from_amass_dict,
    load_smplx,
)
from robokit.utils.warp_utils import resolve_device


_VALID_GENDERS = ("neutral", "male", "female")
_spec_cache: Dict[Tuple[str, Gender], BodyModelSpec] = {}


def _coerce_gender(gender: str) -> Gender:
    assert gender in _VALID_GENDERS, f"unknown gender {gender!r}; expected one of {_VALID_GENDERS}"
    return gender  # type: ignore[return-value]


def _get_spec(smplx_body_model_path: str, gender: Gender, num_betas: int = 10) -> BodyModelSpec:
    key = (smplx_body_model_path, gender)
    if key not in _spec_cache:
        path = f"{smplx_body_model_path.rstrip('/')}/smplx/SMPLX_{gender.upper()}.npz"
        _spec_cache[key] = load_smplx(path, gender=gender, num_betas=num_betas)
    return _spec_cache[key]


def _compute_tpose_height(spec: BodyModelSpec, betas_trimmed: Any, device: str) -> float:
    """T-pose mesh Y-extent in meters (SMPL-Anthropometry convention). Warp LBS, no torch."""
    betas_np = np.ascontiguousarray(np.asarray(betas_trimmed, dtype=np.float32).reshape(1, -1))
    tpose_state = BodyModelState(
        betas=betas_np,
        full_pose_aa=np.zeros((1, SMPLX_NUM_JOINTS, 3), dtype=np.float32),
        transl=np.zeros((1, 3), dtype=np.float32),
    )
    spec_tensors = BodyModelSpecTensors(spec=spec, device=device)
    verts = body_lbs_warp(spec_tensors, tpose_state, return_landmarks=False)["vertices"].numpy()[0]
    return float(verts[:, 1].max() - verts[:, 1].min())


@dataclass(frozen=True)
class SmplxMotion:
    """Loaded motion payload: raw NPZ + spec + state + posed vertices/joints."""

    raw: Dict[str, Any]
    spec: BodyModelSpec
    state: BodyModelState
    vertices: np.ndarray
    """`[N, V, 3]` posed mesh vertices in world coordinates."""
    joints: np.ndarray
    """`[N, J, 3]` posed skeletal joints in world coordinates."""
    T_world_joint_xyz_wxyz: np.ndarray
    """`[N, J, 7]` per-joint world pose in `wp_vec7` layout (xyz + wxyz)."""
    human_height: float
    """T-pose mesh Y-extent in meters."""
    device: str

    @property
    def num_frames(self) -> int:
        return int(self.state.full_pose_aa.shape[0])

    @property
    def mocap_frame_rate(self) -> float:
        # AMASS uses `mocap_framerate`; other sources use the underscored form; default to 60 fps.
        for key in ("mocap_frame_rate", "mocap_framerate"):
            if key in self.raw:
                return float(self.raw[key].item())
        return 60.0

    @property
    def faces(self) -> np.ndarray:
        return self.spec.faces


def load_smplx_file(
    smplx_file: str,
    smplx_body_model_path: Optional[str] = None,
    device: Optional[str] = None,
    compute_vertices: bool = True,
) -> SmplxMotion:
    """Load an AMASS-style SMPL-X NPZ using full GPU LBS, or FK only when `compute_vertices=False`."""
    smplx_body_model_path = smplx_body_model_path or str(get_body_models_dir())
    device = str(resolve_device(device))
    raw = dict(np.load(smplx_file, allow_pickle=True))
    gender = _coerce_gender(str(raw["gender"]) if "gender" in raw else "neutral")
    spec = _get_spec(smplx_body_model_path, gender)

    state = from_amass_dict(raw, spec)

    spec_tensors = BodyModelSpecTensors(spec=spec, device=device)
    if compute_vertices:
        out = body_lbs_warp(spec_tensors, state, return_landmarks=False)
        vertices = out["vertices"].numpy()
        T_np = out["T_world_joint"].numpy()
    else:
        T_np = body_fk_warp(spec_tensors, state).numpy()
        vertices = np.empty((0, 0, 3), dtype=np.float32)
    joints = T_np[..., :3].copy()

    human_height = _compute_tpose_height(spec, state.betas[0], device)
    return SmplxMotion(
        raw=raw,
        spec=spec,
        state=state,
        vertices=vertices,
        joints=joints,
        T_world_joint_xyz_wxyz=T_np,
        human_height=human_height,
        device=device,
    )


def get_smplx_motion(
    motion: SmplxMotion,
    human_joint_names: Sequence[str],
    tgt_fps: int = 30,
) -> Tuple[np.ndarray, float]:
    """Return contiguous `[T, J, 7]` transforms in `human_joint_names` order."""
    src_fps = motion.mocap_frame_rate

    T_np = motion.T_world_joint_xyz_wxyz
    if tgt_fps < src_fps and src_fps % tgt_fps == 0:
        step = int(src_fps // tgt_fps)
        T_np = T_np[::step]
        aligned_fps = float(tgt_fps)
    else:
        aligned_fps = float(src_fps)

    joint_indices = [motion.spec.joint_names.index(name) for name in human_joint_names]
    return np.ascontiguousarray(T_np[:, joint_indices], dtype=np.float32), aligned_fps


def fetch_smplx_clip(
    motion_path: str,
    human_joint_names: Sequence[str],
    tgt_fps: int = 30,
    device: Optional[str] = None,
) -> Tuple[SmplxMotion, np.ndarray, float]:
    """Fetch and return the rich SMPL-X payload plus ordered transforms and FPS."""
    from pathlib import Path

    from robokit.assets import fetch

    parts = Path(motion_path).parts
    if len(parts) < 3 or parts[0] != "datasets":
        raise ValueError("Expected datasets/<name>/<clip> for an HF motion path")
    dataset = parts[1]
    body_models = str(get_body_models_dir())
    motion = load_smplx_file(str(fetch([f"datasets/{dataset}/**"]) / motion_path), body_models, device=device)
    T_world_human, fps = get_smplx_motion(motion, human_joint_names, tgt_fps=tgt_fps)
    return motion, T_world_human, fps
