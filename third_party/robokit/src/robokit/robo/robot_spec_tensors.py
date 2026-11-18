# pyright: reportArgumentType=false
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Optional, cast

import numpy as np
import warp as wp

from robokit.robo.robot_spec import RobotSpec
from robokit.utils.warp_utils import wp_vec6, wp_vec7


if TYPE_CHECKING:
    import torch


def narrow_limits_to_float32(limits: np.ndarray) -> np.ndarray:
    """Cast `[lo, hi]` inward to float32 so clamping respects the original float64 bounds.

    Example:
        >>> limits = np.array([[-0.1, 1.0471975511965976]])
        >>> bool(narrow_limits_to_float32(limits)[0, 1] <= limits[0, 1])
        True
    """
    narrowed = limits.astype(np.float32)
    if narrowed.size == 0:
        return narrowed
    lo, hi = narrowed[:, 0], narrowed[:, 1]
    np.copyto(lo, np.nextafter(lo, np.float32(np.inf)), where=lo < limits[:, 0])
    np.copyto(hi, np.nextafter(hi, np.float32(-np.inf)), where=hi > limits[:, 1])
    return narrowed


@dataclass(eq=False)  # hashed by identity: keys the FK buffer cache
class RobotSpecTensors:
    spec: RobotSpec
    device: Optional[str] = None

    # --- joint tensors ---
    @cached_property
    def joint_axes(self) -> wp.array:
        return wp.from_numpy(self.spec.joint_axes.astype(np.float32), device=self.device, dtype=wp.vec3)

    @cached_property
    def joint_types(self) -> wp.array:
        return wp.from_numpy(self.spec.joint_types.astype(np.int32), device=self.device, dtype=wp.int32)

    @cached_property
    def joint_twists(self) -> wp.array:
        return wp.from_numpy(self.spec.joint_twists.astype(np.float32), device=self.device, dtype=wp_vec6)

    @cached_property
    def actuated_joint_indices(self) -> wp.array:
        return wp.from_numpy(self.spec.actuated_joint_indices.astype(np.int32), device=self.device, dtype=wp.int32)

    @cached_property
    def parent_joint_indices(self) -> wp.array:
        return wp.from_numpy(self.spec.parent_joint_indices.astype(np.int32), device=self.device, dtype=wp.int32)

    @cached_property
    def parent_joint_transforms(self) -> wp.array:
        return wp.from_numpy(self.spec.parent_joint_transforms.astype(np.float32), device=self.device, dtype=wp_vec7)

    @cached_property
    def parent_joint_transforms_matrix(self) -> wp.array:
        return wp.from_numpy(
            self.spec.parent_joint_transforms_matrix.astype(np.float32), device=self.device, dtype=wp.mat44
        )

    @cached_property
    def mimic_actuated_joint_indices(self) -> wp.array:
        return wp.from_numpy(
            self.spec.mimic_actuated_joint_indices.astype(np.int32), device=self.device, dtype=wp.int32
        )

    @cached_property
    def mimic_multipliers(self) -> wp.array:
        return wp.from_numpy(self.spec.mimic_multipliers.astype(np.float32), device=self.device, dtype=wp.float32)

    @cached_property
    def mimic_offsets(self) -> wp.array:
        return wp.from_numpy(self.spec.mimic_offsets.astype(np.float32), device=self.device, dtype=wp.float32)

    @cached_property
    def joint_limits(self) -> wp.array:
        return wp.from_numpy(narrow_limits_to_float32(self.spec.joint_limits), device=self.device, dtype=wp.float32)

    @cached_property
    def joint_velocity_limits(self) -> wp.array:
        return wp.from_numpy(self.spec.joint_velocity_limits.astype(np.float32), device=self.device, dtype=wp.float32)

    @cached_property
    def actuated_joint_limits(self) -> wp.array:
        return wp.from_numpy(
            narrow_limits_to_float32(self.spec.actuated_joint_limits), device=self.device, dtype=wp.float32
        )

    @cached_property
    def actuated_joint_velocity_limits(self) -> wp.array:
        return wp.from_numpy(
            self.spec.actuated_joint_velocity_limits.astype(np.float32), device=self.device, dtype=wp.float32
        )

    # --- link and mapping tensors ---
    @cached_property
    def link_parent_joint_indices(self) -> wp.array:
        return wp.from_numpy(self.spec.link_parent_joint_indices.astype(np.int32), device=self.device, dtype=wp.int32)

    @cached_property
    def link_ancestor_joints_mask(self) -> wp.array:
        return wp.from_numpy(self.spec.link_ancestor_joints_mask.astype(np.bool_), device=self.device, dtype=wp.bool)

    @cached_property
    def joints_to_actuated_mapping(self) -> wp.array:
        return wp.from_numpy(
            self.spec.joints_to_actuated_mapping.astype(np.float32), device=self.device, dtype=wp.float32
        )

    @cached_property
    def topological_order_joint_indices(self) -> wp.array:
        return wp.from_numpy(
            self.spec.topological_order_joint_indices.astype(np.int32), device=self.device, dtype=wp.int32
        )

    # --- default configurations ---
    @cached_property
    def zero_q(self) -> wp.array:
        return wp.from_numpy(self.spec.zero_q.astype(np.float32), device=self.device, dtype=wp.float32)

    @cached_property
    def midrange_q(self) -> wp.array:
        return wp.from_numpy(self.spec.midrange_q.astype(np.float32), device=self.device, dtype=wp.float32)

    # --- collision geometry tensors ---
    @cached_property
    def local_collision_sphere_centers(self) -> wp.array:
        return wp.from_numpy(
            self.spec.local_collision_sphere_centers.astype(np.float32), device=self.device, dtype=wp.vec3
        )

    @cached_property
    def collision_sphere_radii(self) -> wp.array:
        return wp.from_numpy(self.spec.collision_sphere_radii.astype(np.float32), device=self.device, dtype=wp.float32)

    @cached_property
    def collision_spheres_link_indices(self) -> wp.array:
        return wp.from_numpy(
            self.spec.collision_spheres_link_indices.astype(np.int32), device=self.device, dtype=wp.int32
        )

    @cached_property
    def local_link_bounding_sphere_centers(self) -> wp.array:
        return wp.from_numpy(
            self.spec.local_link_bounding_sphere_centers.astype(np.float32), device=self.device, dtype=wp.vec3
        )

    @cached_property
    def link_bounding_sphere_radii(self) -> wp.array:
        return wp.from_numpy(
            self.spec.link_bounding_sphere_radii.astype(np.float32), device=self.device, dtype=wp.float32
        )

    @cached_property
    def local_link_capsule_endpoint_a(self) -> wp.array:
        return wp.from_numpy(
            self.spec.local_link_capsule_endpoints[:, 0].astype(np.float32), device=self.device, dtype=wp.vec3
        )

    @cached_property
    def local_link_capsule_endpoint_b(self) -> wp.array:
        return wp.from_numpy(
            self.spec.local_link_capsule_endpoints[:, 1].astype(np.float32), device=self.device, dtype=wp.vec3
        )

    @cached_property
    def link_capsule_radii(self) -> wp.array:
        return wp.from_numpy(self.spec.link_capsule_radii.astype(np.float32), device=self.device, dtype=wp.float32)

    @cached_property
    def link_identity_indices(self) -> wp.array:
        return wp.from_numpy(np.arange(self.spec.num_links, dtype=np.int32), device=self.device, dtype=wp.int32)

    # --- inertial tensors ---
    @cached_property
    def link_masses(self) -> wp.array:
        return wp.from_numpy(self.spec.link_masses.astype(np.float32), device=self.device, dtype=wp.float32)

    @cached_property
    def link_local_com_positions(self) -> wp.array:
        return wp.from_numpy(self.spec.link_local_com_positions.astype(np.float32), device=self.device, dtype=wp.vec3)


@dataclass
class TorchRobotSpecTensors:
    spec: RobotSpec
    device: object = None  # torch.types.Device

    # --- casting helpers ---
    def _to_float(self, arr: np.ndarray) -> "torch.Tensor":
        import torch

        return torch.from_numpy(arr).float().to(device=self.device)

    def _to_long(self, arr: np.ndarray) -> "torch.Tensor":
        import torch

        return torch.from_numpy(arr).long().to(device=self.device)

    def _to_bool(self, arr: np.ndarray) -> "torch.Tensor":
        import torch

        return torch.from_numpy(arr).bool().to(device=self.device)

    # --- joint tensors ---
    @cached_property
    def joint_twists(self) -> "torch.Tensor":
        return self._to_float(self.spec.joint_twists)

    @cached_property
    def joint_types(self) -> "torch.Tensor":
        return self._to_long(self.spec.joint_types)

    @cached_property
    def joint_axes(self) -> "torch.Tensor":
        return self._to_float(self.spec.joint_axes)

    @cached_property
    def actuated_joint_indices(self) -> "torch.Tensor":
        return self._to_long(self.spec.actuated_joint_indices)

    @cached_property
    def parent_joint_indices(self) -> "torch.Tensor":
        return self._to_long(self.spec.parent_joint_indices)

    @cached_property
    def parent_joint_transforms(self) -> "torch.Tensor":
        return self._to_float(self.spec.parent_joint_transforms)

    @cached_property
    def parent_joint_transforms_matrix(self) -> "torch.Tensor":
        from robokit.lie.se3_torch_wrappers import SE3ToMatrix

        # older torch stubs type Function.apply as returning Optional[Any]
        return cast("torch.Tensor", SE3ToMatrix.apply(self.parent_joint_transforms))

    @cached_property
    def mimic_actuated_joint_indices(self) -> "torch.Tensor":
        return self._to_long(self.spec.mimic_actuated_joint_indices)

    @cached_property
    def mimic_multipliers(self) -> "torch.Tensor":
        return self._to_float(self.spec.mimic_multipliers)

    @cached_property
    def mimic_offsets(self) -> "torch.Tensor":
        return self._to_float(self.spec.mimic_offsets)

    @cached_property
    def joint_limits(self) -> "torch.Tensor":
        return self._to_float(self.spec.joint_limits)

    @cached_property
    def joint_velocity_limits(self) -> "torch.Tensor":
        return self._to_float(self.spec.joint_velocity_limits)

    @cached_property
    def actuated_joint_limits(self) -> "torch.Tensor":
        return self._to_float(self.spec.actuated_joint_limits)

    @cached_property
    def actuated_joint_velocity_limits(self) -> "torch.Tensor":
        return self._to_float(self.spec.actuated_joint_velocity_limits)

    # --- link and mapping tensors ---
    @cached_property
    def link_parent_joint_indices(self) -> "torch.Tensor":
        return self._to_long(self.spec.link_parent_joint_indices)

    @cached_property
    def link_ancestor_joints_mask(self) -> "torch.Tensor":
        return self._to_bool(self.spec.link_ancestor_joints_mask)

    @cached_property
    def joints_to_actuated_mapping(self) -> "torch.Tensor":
        return self._to_float(self.spec.joints_to_actuated_mapping)

    @cached_property
    def topological_order_joint_indices(self) -> "torch.Tensor":
        return self._to_long(self.spec.topological_order_joint_indices)

    # --- default configurations ---
    @cached_property
    def zero_q(self) -> "torch.Tensor":
        return self._to_float(self.spec.zero_q)

    @cached_property
    def midrange_q(self) -> "torch.Tensor":
        return self._to_float(self.spec.midrange_q)
