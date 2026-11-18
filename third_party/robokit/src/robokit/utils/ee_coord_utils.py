"""Semantic coordinate systems for hands, grippers, and attachment frames."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union

import numpy as np


if TYPE_CHECKING:
    import torch


AXIS = Literal["x", "y", "z", "-x", "-y", "-z"]

AXIS_TO_VEC: Dict[AXIS, Tuple[int, int, int]] = {
    "x": (1, 0, 0),
    "-x": (-1, 0, 0),
    "y": (0, 1, 0),
    "-y": (0, -1, 0),
    "z": (0, 0, 1),
    "-z": (0, 0, -1),
}
VEC_TO_AXIS: Dict[Tuple[int, int, int], AXIS] = {v: k for k, v in AXIS_TO_VEC.items()}


@dataclass(frozen=True)
class MountCoordinateSpec:
    """Define a right-handed attachment frame; the third axis follows from the first two.

    Frame fields:
    - ``outward_axis``: wrist/mount toward the end effector.
    - ``right_axis``: rotation around the outward axis.
    - ``mount_point_xyz``: attachment point in link coordinates; None leaves it unspecified.
    """

    outward_axis: AXIS
    right_axis: AXIS
    mount_point_xyz: Optional[Tuple[float, float, float]] = None

    def __post_init__(self):
        if self.outward_axis[-1] == self.right_axis[-1]:
            raise ValueError("outward_axis must be orthogonal to right_axis")
        if self.mount_point_xyz is not None and (
            np.shape(self.mount_point_xyz) != (3,) or not np.isfinite(self.mount_point_xyz).all()
        ):
            raise ValueError("mount_point_xyz must contain three finite coordinates")


@dataclass(frozen=True)
class GripperCoordinateSpec:
    """Map gripper directions to mounting-link axes.

    Frame fields:
    - ``outward_axis``: mount toward fingertips.
    - ``opening_axis``: left jaw toward right jaw; symmetric claws designate a reference finger as the right side.
    - ``mount_point_xyz``: attachment point in mounting-link coordinates, independent of its origin.
    """

    outward_axis: AXIS
    opening_axis: AXIS
    mount_point_xyz: Optional[Tuple[float, float, float]] = None

    def __post_init__(self):
        MountCoordinateSpec(self.outward_axis, self.opening_axis, self.mount_point_xyz)

    @property
    def mount_spec(self) -> MountCoordinateSpec:
        return MountCoordinateSpec(self.outward_axis, self.opening_axis, self.mount_point_xyz)


@dataclass(frozen=True)
class HandCoordinateSpec:
    """Express hand axes in the actual base-link frame, which may differ from the wrist (e.g., Shadow Hand).

    If wrist and base differ:
    - Define the spec in the actual base-link frame.
    - Or truncate the URDF so the wrist becomes the base.

    Frame fields:
    - palm_forward_axis: outward palm normal.
    - four_fingers_up_axis: palm toward fingertips.
    - palm_right_axis: right when looking along palm_forward_axis.
    - mount_point_xyz: optional attachment point in this link's frame.

    Constraints:
    - All axes are distinct.
    - palm_forward_axis and four_fingers_up_axis are orthogonal.
    - cross(palm_forward_axis, four_fingers_up_axis) == palm_right_axis.
    """

    palm_forward_axis: AXIS
    four_fingers_up_axis: AXIS
    mount_point_xyz: Optional[Tuple[float, float, float]] = None

    def __post_init__(self):
        if self.palm_forward_axis[-1] == self.four_fingers_up_axis[-1]:
            raise ValueError("palm_forward_axis must be orthogonal to four_fingers_up_axis")
        MountCoordinateSpec(self.four_fingers_up_axis, self.palm_right_axis, self.mount_point_xyz)

    @property
    def palm_right_axis(self) -> AXIS:
        up = AXIS_TO_VEC[self.four_fingers_up_axis]
        fwd = AXIS_TO_VEC[self.palm_forward_axis]
        return VEC_TO_AXIS[tuple(np.cross(fwd, up))]

    @property
    def palm_forward_vec(self) -> np.ndarray:
        return np.array(AXIS_TO_VEC[self.palm_forward_axis], dtype=np.float32)

    @property
    def four_fingers_up_vec(self) -> np.ndarray:
        return np.array(AXIS_TO_VEC[self.four_fingers_up_axis], dtype=np.float32)

    @property
    def palm_right_vec(self) -> np.ndarray:
        return np.array(AXIS_TO_VEC[self.palm_right_axis], dtype=np.float32)

    @property
    def mount_spec(self) -> MountCoordinateSpec:
        """Mount axes: wrist-to-fingertips is outward and palm-right fixes roll."""
        return MountCoordinateSpec(self.four_fingers_up_axis, self.palm_right_axis, self.mount_point_xyz)


EECoordinateSpec = Union[HandCoordinateSpec, GripperCoordinateSpec]


def ee_coord_conversion(
    src_spec: Union[MountCoordinateSpec, EECoordinateSpec],
    dst_spec: Union[MountCoordinateSpec, EECoordinateSpec],
    return_tensors: Literal["np", "pt"] = "np",
) -> "Union[np.ndarray, torch.Tensor]":
    """Align semantic axes with ``v_dst = R_dst_src @ v_src``; this rotation is independent of mount points.

    >>> ee_coord_conversion(HandCoordinateSpec("y", "z"), GripperCoordinateSpec("z", "x"))
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]], dtype=float32)
    >>> ee_coord_conversion(MountCoordinateSpec("x", "y"), MountCoordinateSpec("z", "x")).shape
    (3, 3)
    """
    if not isinstance(src_spec, MountCoordinateSpec):
        src_spec = src_spec.mount_spec
    if not isinstance(dst_spec, MountCoordinateSpec):
        dst_spec = dst_spec.mount_spec
    src_out = AXIS_TO_VEC[src_spec.outward_axis]
    src_right = AXIS_TO_VEC[src_spec.right_axis]
    dst_out = AXIS_TO_VEC[dst_spec.outward_axis]
    dst_right = AXIS_TO_VEC[dst_spec.right_axis]
    src_basis = np.column_stack([src_out, src_right, np.cross(src_out, src_right)]).astype(np.float32)
    dst_basis = np.column_stack([dst_out, dst_right, np.cross(dst_out, dst_right)]).astype(np.float32)
    R_dst_src = dst_basis @ src_basis.T
    if return_tensors == "pt":
        import torch

        return torch.from_numpy(R_dst_src).float()
    return R_dst_src


MANOPTH_HAND_COORD_SPEC = HandCoordinateSpec(palm_forward_axis="-y", four_fingers_up_axis="-x")
