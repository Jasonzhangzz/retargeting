"""Load local body models (NPZ or automatically converted PKL) and dataset poses."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from robokit.assets.body_models import get_model_path
from robokit.smplx.constants import (
    MANO_FINGERTIP_NAMES,
    MANO_JOINT_NAMES,
    MANO_NUM_JOINTS,
    MANO_NUM_PCA_COMPONENTS,
    MANO_NUM_POSE_DIRS,
    MANO_NUM_VERTICES,
    SMPLX_JOINT_NAMES,
    SMPLX_NUM_JOINTS,
    SMPLX_NUM_POSE_DIRS,
    SMPLX_NUM_VERTICES,
    SMPLX_STATIC_LANDMARK_NAMES,
    SMPLX_STATIC_LANDMARK_VERTEX_IDS,
    Gender,
    Side,
    mano_fingertip_vertex_ids,
)
from robokit.smplx.spec import BodyModelSpec, BodyModelState


def _parse_lbs_npz(data: Any, num_vertices: int, num_joints: int, num_pose_dirs: int, num_betas: int) -> Dict[str, Any]:
    """Parse the LBS fields shared by SMPL-X and MANO `.npz` files into `BodyModelSpec` kwargs."""
    v_template = data["v_template"].astype(np.float32)
    assert v_template.shape == (num_vertices, 3), (
        f"v_template {v_template.shape} != ({num_vertices}, 3); wrong model file?"
    )

    shapedirs_all = data["shapedirs"].astype(np.float32)
    assert shapedirs_all.shape[0] == num_vertices and shapedirs_all.shape[1] == 3
    assert num_betas <= shapedirs_all.shape[2], f"num_betas={num_betas} > shapedirs components={shapedirs_all.shape[2]}"
    shapedirs = np.ascontiguousarray(shapedirs_all[:, :, :num_betas])

    # NPZ stores posedirs as (V, 3, P); LBS consumes (P, V*3).
    posedirs_vxp = data["posedirs"].astype(np.float32)
    assert posedirs_vxp.shape == (num_vertices, 3, num_pose_dirs)
    posedirs = np.ascontiguousarray(posedirs_vxp.reshape(num_vertices * 3, num_pose_dirs).T)

    J_regressor = data["J_regressor"].astype(np.float32)
    assert J_regressor.shape == (num_joints, num_vertices)

    # NPZ stores the root sentinel as MAX_UINT32-cast-to-int64; force to -1.
    parents = data["kintree_table"][0].astype(np.int64).copy()
    parents[0] = -1

    lbs_weights = data["weights"].astype(np.float32)
    assert lbs_weights.shape == (num_vertices, num_joints)

    faces = data["f"].astype(np.int32)
    assert faces.ndim == 2 and faces.shape[1] == 3

    return {
        "v_template": v_template,
        "shapedirs": shapedirs,
        "posedirs": posedirs,
        "J_regressor": J_regressor,
        "parents": parents,
        "lbs_weights": lbs_weights,
        "faces": faces,
        "num_betas": num_betas,
    }


def load_smplx(
    npz_path: Optional[Union[str, Path]] = None,
    gender: Gender = "neutral",
    num_betas: int = 10,
) -> BodyModelSpec:
    """Load local SMPL-X data with gender/hands_meanl/hands_meanr metadata; never download missing models.

    Path resolution:
    1. Use `npz_path` (NPZ or official PKL converted once beside the source).
    2. Otherwise use ROBOKIT_BODY_MODELS or ~/.robokit/body_models.
    3. If absent, raise FileNotFoundError with setup instructions.

    Examples:
        >>> # xdoctest: +REQUIRES(env:ROBOKIT_BODY_MODELS)
        >>> spec = load_smplx()
        >>> spec.v_template.shape
        (10475, 3)
        >>> int(spec.parents[0])
        -1
        >>> spec.J_regressor.shape
        (55, 10475)
        >>> spec.num_betas
        10
        >>> spec.metadata["gender"]
        'neutral'
    """
    data = np.load(get_model_path("smplx", gender, npz_path), allow_pickle=False)
    common = _parse_lbs_npz(data, SMPLX_NUM_VERTICES, SMPLX_NUM_JOINTS, SMPLX_NUM_POSE_DIRS, num_betas)

    hands_meanl = data["hands_meanl"].astype(np.float32)
    hands_meanr = data["hands_meanr"].astype(np.float32)
    assert hands_meanl.shape == (45,) and hands_meanr.shape == (45,)

    return BodyModelSpec(
        name=f"smplx_{gender}",
        joint_names=list(SMPLX_JOINT_NAMES),
        static_landmark_names=list(SMPLX_STATIC_LANDMARK_NAMES),
        static_landmark_vertex_ids=np.array(SMPLX_STATIC_LANDMARK_VERTEX_IDS, dtype=np.int64),
        metadata={
            "gender": gender,
            "hands_meanl": hands_meanl,
            "hands_meanr": hands_meanr,
        },
        **common,
    )


def load_mano(
    npz_path: Optional[Union[str, Path]] = None,
    side: Side = "right",
    num_betas: int = 10,
) -> BodyModelSpec:
    """Load local MANO data with side/hands_components/hands_mean metadata; never download missing models.

    Path resolution:
    1. Use `npz_path` (NPZ or official PKL converted once beside the source).
    2. Otherwise use ROBOKIT_BODY_MODELS or ~/.robokit/body_models.
    3. If absent, raise FileNotFoundError with setup instructions.

    Examples:
        >>> # xdoctest: +REQUIRES(env:ROBOKIT_BODY_MODELS)
        >>> spec = load_mano(side="right")
        >>> spec.v_template.shape
        (778, 3)
        >>> int(spec.parents[0])
        -1
        >>> spec.J_regressor.shape
        (16, 778)
        >>> spec.num_betas
        10
        >>> spec.metadata["side"]
        'right'
        >>> spec.static_landmark_vertex_ids.shape
        (5,)
    """
    data = np.load(get_model_path("mano", side, npz_path), allow_pickle=False)
    common = _parse_lbs_npz(data, MANO_NUM_VERTICES, MANO_NUM_JOINTS, MANO_NUM_POSE_DIRS, num_betas)

    hands_components = data["hands_components"].astype(np.float32)
    hands_mean = data["hands_mean"].astype(np.float32)
    assert hands_components.shape == (45, 45)
    assert hands_mean.shape == (45,)

    recorded = str(data["side"])
    assert recorded == side, f"side mismatch: NPZ recorded {recorded!r} but caller passed {side!r}"

    return BodyModelSpec(
        name=f"mano_{side}",
        joint_names=list(MANO_JOINT_NAMES),
        static_landmark_names=list(MANO_FINGERTIP_NAMES),
        static_landmark_vertex_ids=np.array(mano_fingertip_vertex_ids(side), dtype=np.int64),
        metadata={
            "side": side,
            "hands_components": hands_components,
            "hands_mean": hands_mean,
        },
        **common,
    )


def from_amass_dict(
    smplx_data: Dict[str, Any],
    spec: BodyModelSpec,
) -> BodyModelState:
    """Convert an AMASS-style dict (`poses` or `root_orient`+`pose_body`) into a state; face/hand DoFs zeroed.

    Examples:
        >>> # xdoctest: +REQUIRES(env:ROBOKIT_BODY_MODELS)
        >>> import numpy as np
        >>> from robokit.smplx import load_smplx
        >>> spec = load_smplx()
        >>> data = {
        ...     "poses": np.zeros((4, 165), dtype=np.float32),
        ...     "trans": np.zeros((4, 3), dtype=np.float32),
        ...     "betas": np.zeros((10,), dtype=np.float32),
        ... }
        >>> state = from_amass_dict(data, spec)
        >>> state.full_pose_aa.shape
        (4, 55, 3)
        >>> state.betas.shape
        (4, 10)
        >>> state.transl.shape
        (4, 3)
    """
    if "pose_body" in smplx_data and "root_orient" in smplx_data:
        root_orient = np.asarray(smplx_data["root_orient"], dtype=np.float32)
        pose_body = np.asarray(smplx_data["pose_body"], dtype=np.float32)
    else:
        assert "poses" in smplx_data, "Expected either ('root_orient', 'pose_body') or 'poses' in smplx_data"
        poses = np.asarray(smplx_data["poses"], dtype=np.float32)
        root_orient = poses[:, :3]
        pose_body = poses[:, 3:66]
    assert root_orient.shape[1] == 3 and pose_body.shape[1] == 63, (
        f"Unexpected shapes: root_orient={root_orient.shape}, pose_body={pose_body.shape}"
    )
    num_frames = root_orient.shape[0]
    assert pose_body.shape[0] == num_frames

    betas_raw = np.asarray(smplx_data["betas"], dtype=np.float32).reshape(-1)
    nb = spec.num_betas
    if betas_raw.size >= nb:
        betas_trim = betas_raw[:nb]
    else:
        betas_trim = np.pad(betas_raw, (0, nb - betas_raw.size))
    betas_frames = np.ascontiguousarray(np.broadcast_to(betas_trim, (num_frames, nb)))

    transl = np.ascontiguousarray(np.asarray(smplx_data["trans"], dtype=np.float32))
    assert transl.shape == (num_frames, 3)

    full_pose_aa = np.zeros((num_frames, SMPLX_NUM_JOINTS, 3), dtype=np.float32)
    full_pose_aa[:, 0] = root_orient
    full_pose_aa[:, 1:22] = pose_body.reshape(num_frames, 21, 3)

    return BodyModelState(
        betas=betas_frames,
        full_pose_aa=full_pose_aa,
        transl=transl,
    )


def from_dexycb_pose(
    pose_m: np.ndarray,
    betas: np.ndarray,
    spec: BodyModelSpec,
    flat_hand_mean: bool = False,
) -> BodyModelState:
    """Convert DexYCB `pose_m` `[B, 51]` (orient(3) + PCA(45) + transl(3), or `[B, 1, 51]`) into a state.

    Examples:
        >>> # xdoctest: +REQUIRES(env:ROBOKIT_BODY_MODELS)
        >>> import numpy as np
        >>> from robokit.smplx import load_mano
        >>> spec = load_mano(side="right")
        >>> pose_m = np.zeros((4, 1, 51), dtype=np.float32)
        >>> betas = np.zeros((10,), dtype=np.float32)
        >>> state = from_dexycb_pose(pose_m, betas, spec, flat_hand_mean=True)
        >>> state.full_pose_aa.shape
        (4, 16, 3)
        >>> state.betas.shape
        (4, 10)
    """
    pose_m = np.asarray(pose_m, dtype=np.float32)
    if pose_m.ndim == 3:
        assert pose_m.shape[1] == 1, f"unexpected pose_m shape {pose_m.shape}"
        pose_m = pose_m[:, 0, :]
    assert pose_m.ndim == 2 and pose_m.shape[1] == 51, f"pose_m must be [B, 51] (or [B, 1, 51]); got {pose_m.shape}"
    B = pose_m.shape[0]

    global_orient = pose_m[:, :3]
    pca_coeffs = pose_m[:, 3 : 3 + MANO_NUM_PCA_COMPONENTS]
    transl = np.ascontiguousarray(pose_m[:, 3 + MANO_NUM_PCA_COMPONENTS :])

    hands_components = spec.metadata["hands_components"]
    hands_mean = spec.metadata["hands_mean"]
    finger_aa = pca_coeffs @ hands_components
    if not flat_hand_mean:
        finger_aa = finger_aa + hands_mean

    full_pose_aa = np.zeros((B, MANO_NUM_JOINTS, 3), dtype=np.float32)
    full_pose_aa[:, 0] = global_orient
    full_pose_aa[:, 1:] = finger_aa.reshape(B, MANO_NUM_JOINTS - 1, 3)

    betas = np.asarray(betas, dtype=np.float32)
    if betas.ndim == 1:
        betas = np.ascontiguousarray(np.broadcast_to(betas, (B, betas.shape[0])))
    assert betas.shape == (B, spec.num_betas), f"betas shape {betas.shape} != ({B}, {spec.num_betas})"

    return BodyModelState(
        betas=betas,
        full_pose_aa=full_pose_aa,
        transl=transl,
    )
