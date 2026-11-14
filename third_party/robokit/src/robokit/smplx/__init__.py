"""Parametric body-model engine (LBS / FK / landmarks) + SMPL-X and MANO adapters; torch exports are lazy."""

from typing import TYPE_CHECKING

from robokit.smplx.constants import (
    MANO_FINGERTIP_NAMES,
    MANO_FINGERTIP_VERTEX_IDS_LEFT,
    MANO_FINGERTIP_VERTEX_IDS_RIGHT,
    MANO_JOINT_NAMES,
    MANO_NUM_JOINTS,
    MANO_NUM_PCA_COMPONENTS,
    MANO_NUM_POSE_DIRS,
    MANO_NUM_SHAPE_COEFFS,
    MANO_NUM_VERTICES,
    MANOPTH_KEYPOINT_PERMUTATION,
    SMPLX_JOINT_NAMES,
    SMPLX_NUM_JOINTS,
    SMPLX_NUM_POSE_DIRS,
    SMPLX_NUM_SHAPE_COEFFS,
    SMPLX_NUM_VERTICES,
    SMPLX_STATIC_LANDMARK_NAMES,
    SMPLX_STATIC_LANDMARK_VERTEX_IDS,
    Gender,
    Side,
    mano_fingertip_vertex_ids,
)
from robokit.smplx.loaders import from_amass_dict, from_dexycb_pose, load_mano, load_smplx
from robokit.smplx.spec import BodyModelSpec, BodyModelState
from robokit.smplx.warp_lbs import BodyModelSpecTensors, body_fk_warp, body_lbs_warp


if TYPE_CHECKING:
    from robokit.smplx.torch_lbs import body_lbs_torch, mano_forward_pca_mm


__all__ = [
    "BodyModelSpec",
    "BodyModelSpecTensors",
    "BodyModelState",
    "body_fk_warp",
    "body_lbs_torch",
    "body_lbs_warp",
    "from_amass_dict",
    "from_dexycb_pose",
    "Gender",
    "load_mano",
    "load_smplx",
    "mano_fingertip_vertex_ids",
    "mano_forward_pca_mm",
    "MANO_FINGERTIP_NAMES",
    "MANO_FINGERTIP_VERTEX_IDS_LEFT",
    "MANO_FINGERTIP_VERTEX_IDS_RIGHT",
    "MANO_JOINT_NAMES",
    "MANO_NUM_JOINTS",
    "MANO_NUM_PCA_COMPONENTS",
    "MANO_NUM_POSE_DIRS",
    "MANO_NUM_SHAPE_COEFFS",
    "MANO_NUM_VERTICES",
    "MANOPTH_KEYPOINT_PERMUTATION",
    "Side",
    "SMPLX_JOINT_NAMES",
    "SMPLX_NUM_JOINTS",
    "SMPLX_NUM_POSE_DIRS",
    "SMPLX_NUM_SHAPE_COEFFS",
    "SMPLX_NUM_VERTICES",
    "SMPLX_STATIC_LANDMARK_NAMES",
    "SMPLX_STATIC_LANDMARK_VERTEX_IDS",
]


def __getattr__(name: str):
    if name in ("body_lbs_torch", "mano_forward_pca_mm"):
        from robokit.smplx import torch_lbs

        return getattr(torch_lbs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
