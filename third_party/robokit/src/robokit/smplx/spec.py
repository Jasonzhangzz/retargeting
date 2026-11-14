"""Parametric body-model spec + per-call state (SMPL / SMPL-H / SMPL-X / MANO / FLAME)."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
from jaxtyping import Float, Int


if TYPE_CHECKING:
    import torch

    _ArrayLike = Union[np.ndarray, "torch.Tensor"]
else:
    _ArrayLike = np.ndarray


@dataclass(frozen=True)
class BodyModelSpec:
    """Parsed, device-agnostic body-model data; format extras (`gender`, `side`, ...) live in `metadata`."""

    name: str
    """Identifier such as `"smplx_neutral"` or `"mano_right"`."""

    v_template: Float[np.ndarray, "V 3"]
    shapedirs: Float[np.ndarray, "V 3 num_betas"]
    posedirs: Float[np.ndarray, "P Vx3"]
    """`(P, V*3)` layout consumed by `pose_feature @ posedirs`; `P = (J - 1) * 9`."""

    J_regressor: Float[np.ndarray, "J V"]
    parents: Int[np.ndarray, "J"]
    """`parents[0] == -1` marks the root."""

    lbs_weights: Float[np.ndarray, "V J"]
    faces: Int[np.ndarray, "F 3"]
    joint_names: List[str]
    num_betas: int

    static_landmark_names: Optional[List[str]] = None
    static_landmark_vertex_ids: Optional[Int[np.ndarray, "L"]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BodyModelState:
    """Batched pose + shape + root translation; numpy on the warp path, torch also accepted by `body_lbs_torch`."""

    betas: _ArrayLike
    """`[B, num_betas]` shape coefficients."""
    full_pose_aa: _ArrayLike
    """`[B, J, 3]` raw axis-angle pose per joint."""
    transl: _ArrayLike
    """`[B, 3]` root translation."""

    @property
    def batch_size(self) -> int:
        return int(self.full_pose_aa.shape[0])
