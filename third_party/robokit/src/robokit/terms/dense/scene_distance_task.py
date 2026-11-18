# pyright: reportArgumentType=false
# pyright: reportOperatorIssue=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportIncompatibleVariableOverride=false
# pyright: reportOptionalMemberAccess=false
"""Contact-point distance to scene geometry."""

from enum import Enum
from typing import TYPE_CHECKING, Literal, Optional, Sequence, Union, cast

import numpy as np
import warp as wp

from robokit.geom import WarpScene
from robokit.lie.se3_kernels import se3_adjoint_func
from robokit.opt.var_values import VarValues
from robokit.terms.robot_task import RobotTask
from robokit.terms.task import GradientTask, ResidualTask
from robokit.utils.warp_utils import wp_device_type, wp_vec6, wp_vec7
from robokit.xform.warp.rotation_conversions import quaternion_apply_func


def _scene_offsets_for(task: "SceneDistanceTask", batch_size: int) -> wp.array:
    """Cache scene-major contact offsets by batch width, preserving ratios across prefixes and draws."""
    cache = task.__dict__.setdefault("_scene_offsets_by_batch", {})
    if batch_size not in cache:
        num_scenes = task.warp_meshes.num_scenes
        cache[batch_size] = wp.from_numpy(
            np.arange(num_scenes + 1, dtype=np.int32) * (batch_size // num_scenes) * task.num_contact_points,
            dtype=wp.int32,
            device=task._device,
        )
    return cache[batch_size]


if TYPE_CHECKING:
    from robokit.robo import Robot, RobotState


# --- device code ------------------------------------------------------------
@wp.func
def _compute_point_jacobian_column_func(
    S_world: wp.array3d(dtype=wp.float32),
    T_world_base: wp.array1d(dtype=wp_vec7),
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    instance: int,
    link_idx: int,
    col_idx: int,
    has_floating_base: bool,
    point: wp.vec3,
) -> wp.vec3:
    base_dofs = wp.int32(6) if has_floating_base else wp.int32(0)
    twist = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if has_floating_base and col_idx < 6:
        unit = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        unit[col_idx] = wp.float32(1.0)
        twist = se3_adjoint_func(T_world_base[instance]) * unit
    else:
        actuated_idx = col_idx - base_dofs
        for joint_idx in range(S_world.shape[2]):
            if link_ancestor_joints_mask[link_idx, joint_idx]:
                joint_weight = joints_to_actuated[joint_idx, actuated_idx]
                for row in range(6):
                    twist[row] += S_world[instance, row, joint_idx] * joint_weight
    linear = wp.vec3(twist[0], twist[1], twist[2])
    angular = wp.vec3(twist[3], twist[4], twist[5])
    return linear + wp.cross(angular, point)


@wp.kernel
def _transform_contact_points_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),
    local_contact_points: wp.array2d(dtype=wp.vec3),
    contact_link_indices: wp.array2d(dtype=wp.int32),
    world_contact_points: wp.array2d(dtype=wp.vec3),
):
    instance, contact_idx = wp.tid()  # pyright: ignore[reportAssignmentType]

    link_idx = contact_link_indices[instance, contact_idx]
    T_world_link_i = T_world_link[instance, link_idx]

    translation = wp.vec3(T_world_link_i[0], T_world_link_i[1], T_world_link_i[2])
    quat_wxyz = wp.vec4(T_world_link_i[3], T_world_link_i[4], T_world_link_i[5], T_world_link_i[6])

    local_pt = local_contact_points[instance, contact_idx]
    world_pt = quaternion_apply_func(quat_wxyz, local_pt) + translation

    world_contact_points[instance, contact_idx] = world_pt


@wp.kernel
def _compute_closest_point_normals_kernel(
    contact_points: wp.array2d(dtype=wp.vec3),
    closest_points: wp.array2d(dtype=wp.vec3),
    signed_dists: wp.array2d(dtype=wp.float32),
    normals: wp.array2d(dtype=wp.vec3),
):
    """Compute outward contact normals from closest surface points."""
    batch, c = wp.tid()  # pyright: ignore[reportAssignmentType]
    direction = contact_points[batch, c] - closest_points[batch, c]
    dir_len = wp.length(direction)
    if dir_len > 1.0e-10:
        sign = wp.float32(1.0)
        if signed_dists[batch, c] < 0.0:
            sign = wp.float32(-1.0)
        normals[batch, c] = direction * (sign / dir_len)
    else:
        normals[batch, c] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def compute_distance_energy_residual_kernel(
    contact_points: wp.array2d(dtype=wp.vec3),
    closest_points: wp.array2d(dtype=wp.vec3),
    residual_weight: wp.array1d(dtype=wp.float32),
    use_sqrt: wp.bool,
    eps: float,
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),
):
    """Compute each contact point's distance to its closest surface point."""
    instance, point_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    dist = wp.length(contact_points[instance, point_idx] - closest_points[instance, point_idx])
    residual = dist
    if use_sqrt:
        residual = wp.sqrt(dist + eps)
    out_residual[instance, row_offset + point_idx] = residual_weight[point_idx] * residual


@wp.kernel
def compute_scene_distance_jacobian_kernel(
    S_world: wp.array3d(dtype=wp.float32),
    T_world_base: wp.array1d(dtype=wp_vec7),
    contact_points_world: wp.array2d(dtype=wp.vec3),
    contact_points_link_indices: wp.array2d(dtype=wp.int32),
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    signed_dists: wp.array2d(dtype=wp.float32),
    normals: wp.array2d(dtype=wp.vec3),
    has_floating_base: wp.bool,
    residual_weight: wp.array1d(dtype=wp.float32),
    use_sqrt: wp.bool,
    eps: float,
    row_offset: int,
    out_jacobian: wp.array3d(dtype=wp.float32),
):
    """Compute the weighted contact-distance Jacobian."""
    instance, contact_idx, col_idx = wp.tid()  # pyright: ignore[reportAssignmentType]

    contact_point = contact_points_world[instance, contact_idx]
    signed_dist = signed_dists[instance, contact_idx]
    normal = normals[instance, contact_idx]

    sign = wp.sign(signed_dist)
    abs_dist = wp.abs(signed_dist)
    scale = wp.float32(1.0)
    if use_sqrt:
        scale = wp.float32(0.5) / wp.sqrt(abs_dist + eps)

    point_column = _compute_point_jacobian_column_func(
        S_world,
        T_world_base,
        link_ancestor_joints_mask,
        joints_to_actuated,
        instance,
        contact_points_link_indices[instance, contact_idx],
        col_idx,
        has_floating_base,
        contact_point,
    )

    jacobian_value = scale * sign * wp.dot(normal, point_column)

    out_jacobian[instance, row_offset + contact_idx, col_idx] = residual_weight[contact_idx] * jacobian_value


@wp.kernel
def _compute_distance_cost_and_gradient_kernel(
    S_world: wp.array3d(dtype=wp.float32),
    T_world_base: wp.array1d(dtype=wp_vec7),
    contact_points_world: wp.array2d(dtype=wp.vec3),
    contact_points_link_indices: wp.array2d(dtype=wp.int32),
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    signed_dists: wp.array2d(dtype=wp.float32),
    normals: wp.array2d(dtype=wp.vec3),
    has_floating_base: wp.bool,
    residual_weight: wp.array1d(dtype=wp.float32),
    out_residual: wp.array2d(dtype=wp.float32),
    use_sqrt: wp.bool,
    eps: float,
    col_offset: int,
    out_cost: wp.array1d(dtype=wp.float32),
    out_gradient: wp.array2d(dtype=wp.float32),
):
    instance, contact_idx, col_idx = wp.tid()  # pyright: ignore[reportAssignmentType]

    contact_point = contact_points_world[instance, contact_idx]
    signed_dist = signed_dists[instance, contact_idx]
    normal = normals[instance, contact_idx]

    sign = wp.sign(signed_dist)
    abs_dist = wp.abs(signed_dist)
    scale = wp.float32(1.0)
    if use_sqrt:
        scale = wp.float32(0.5) / wp.sqrt(abs_dist + eps)

    point_column = _compute_point_jacobian_column_func(
        S_world,
        T_world_base,
        link_ancestor_joints_mask,
        joints_to_actuated,
        instance,
        contact_points_link_indices[instance, contact_idx],
        col_idx,
        has_floating_base,
        contact_point,
    )

    weighted_jac = residual_weight[contact_idx] * scale * sign * wp.dot(normal, point_column)
    r = out_residual[instance, contact_idx]
    contribution = weighted_jac * r
    if contribution != wp.float32(0.0):
        wp.atomic_add(out_gradient, instance, col_offset + col_idx, contribution)

    if col_idx == 0:
        wp.atomic_add(out_cost, instance, wp.float32(0.5) * r * r)


# --- task -------------------------------------------------------------------
class SceneDistanceTask(RobotTask, ResidualTask, GradientTask):
    """Minimize hand contact-point distance to a scene surface."""

    def __init__(
        self,
        robot: Optional["Robot"] = None,
        warp_meshes: Optional[WarpScene] = None,
        num_contact_points: int = 0,
        weight: Optional[Union[float, Sequence[float]]] = None,
        residual_mode: Literal["abs", "sqrt_abs"] = "abs",
        residual_eps: float = 1e-6,
        points_var_key: Optional[Enum] = None,  # read the points from this variable instead of the stored array
        local_contact_points: Optional[wp.array] = None,
        contact_points_link_indices: Optional[wp.array] = None,
    ):
        self.robot = robot
        self.warp_meshes = warp_meshes
        self.num_contact_points = num_contact_points
        self.weight = weight
        self.residual_mode = residual_mode
        self.residual_eps = residual_eps
        self.points_var_key = points_var_key
        self.local_contact_points = local_contact_points
        self.contact_points_link_indices = contact_points_link_indices

        self._device: Optional[wp_device_type] = None
        self._cached_batch_size = 0
        self._residual_weight_np: Optional[np.ndarray] = None
        self.residual_weight: Optional[wp.array] = None
        self._direct_signed_dists: Optional[wp.array] = None
        self._direct_normals: Optional[wp.array] = None
        self.last_query: Optional[tuple] = None
        self._direct_residual_buf: Optional[wp.array] = None

        if self.warp_meshes is None or self.num_contact_points == 0:
            return  # config-time stub
        if self.warp_meshes.num_elements == 0:
            raise ValueError("warp_meshes must contain at least one mesh.")
        if self.residual_mode not in {"abs", "sqrt_abs"}:
            raise ValueError(f"Unsupported residual_mode: {self.residual_mode}")

        residual_weight = np.ones((self.num_contact_points,), dtype=np.float32)
        if self.weight is not None:
            if isinstance(self.weight, (float, int)):
                residual_weight[:] = float(self.weight)
            else:
                weight_arr = np.asarray(self.weight, dtype=np.float32)
                if weight_arr.shape != (self.num_contact_points,):
                    raise ValueError(f"Expected weight shape ({self.num_contact_points},), got {weight_arr.shape}.")
                residual_weight[:] = weight_arr
        self._residual_weight_np = residual_weight

    def set_robot(self, robot: "Robot"):
        self.robot = robot

    def _ensure_allocation(self, var: "RobotState") -> None:
        """Allocate once at the first batch seen; smaller batches run as a prefix, larger ones raise."""
        if self._device is None:
            self._init_buffers(var.batch_size, var.q.device)
        elif var.batch_size > self._cached_batch_size:
            raise ValueError(
                f"batch size {var.batch_size} exceeds the {self._cached_batch_size} this term was built for"
            )

    def _init_buffers(self, batch_size: int, device: wp_device_type):
        self._device = device
        self._cached_batch_size = batch_size
        self.residual_weight = wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device=device)
        self._contact_points_world_buf = wp.empty(
            (batch_size, self.num_contact_points), dtype=wp.vec3, device=device, requires_grad=True
        )
        self._closest_points_buf = wp.empty((batch_size, self.num_contact_points), dtype=wp.vec3, device=device)
        self._direct_signed_dists = wp.empty((batch_size, self.num_contact_points), dtype=wp.float32, device=device)
        self._direct_normals = wp.empty((batch_size, self.num_contact_points), dtype=wp.vec3, device=device)
        self._direct_residual_buf = wp.empty((batch_size, self.num_contact_points), dtype=wp.float32, device=device)

    @property
    def residual_dim(self) -> int:
        return self.num_contact_points

    def _transform_contact_points(
        self,
        var: "RobotState",
        local_contact_points: wp.array,
        contact_points_link_indices: wp.array,
    ) -> wp.array:
        batch_size = var.batch_size
        num_links = self.robot.spec.num_links
        device = var.q.device

        if self._device is not None and batch_size == self._cached_batch_size:
            contact_points_world = self._contact_points_world_buf
        else:
            contact_points_world = wp.empty(
                (batch_size, self.num_contact_points), dtype=wp.vec3, device=device, requires_grad=True
            )

        wp.launch(
            kernel=_transform_contact_points_kernel,
            dim=(batch_size, self.num_contact_points),
            inputs=[
                var.T_world_link.reshape((batch_size, num_links)),
                local_contact_points,
                contact_points_link_indices,
            ],
            outputs=[contact_points_world],
            device=device,
        )

        return contact_points_world

    def compute_weighted_residual(
        self,
        var_values: VarValues,
        local_contact_points: Optional[wp.array] = None,
        contact_points_link_indices: Optional[wp.array] = None,
        scene_offsets: Optional[wp.array] = None,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        var = var_values.get(self.var_key)
        self._ensure_allocation(var)
        if not var.is_fk_computed:
            var = self.robot.forward_kinematics(var)

        if local_contact_points is not None and contact_points_link_indices is not None:
            contact_points_world = self._transform_contact_points(
                var, local_contact_points, contact_points_link_indices
            )
        elif self.points_var_key is not None:
            chosen = var_values.get(self.points_var_key)
            contact_points_world = self._transform_contact_points(var, chosen.points, chosen.link_indices)
        elif self.local_contact_points is not None and self.contact_points_link_indices is not None:
            contact_points_world = self._transform_contact_points(
                var, self.local_contact_points, self.contact_points_link_indices
            )
        else:
            raise ValueError(
                "Contact data not provided. Pass local_contact_points and contact_points_link_indices "
                "as arguments or set them on the task."
            )

        b = var.batch_size
        sdf_scene_offsets = scene_offsets if scene_offsets is not None else _scene_offsets_for(self, b)
        signed_dists, normals, closest_points = (
            cast(wp.array, a) if b == self._cached_batch_size else cast(wp.array, a)[:b]
            for a in (self._direct_signed_dists, self._direct_normals, self._closest_points_buf)
        )
        self.warp_meshes.query_sdf(
            contact_points_world,
            sdf_scene_offsets,
            out_signed_dists=signed_dists,
            out_normals=normals,
            out_closest_points=closest_points,
            record_tape=False,  # the query has no adjoint
        )
        # A ForceClosureTask built with `distance_task=self` reads this query instead of repeating it.
        self.last_query = (var_values, contact_points_world, signed_dists, normals, closest_points)

        kernel_device = out_residual.device if out_residual is not None else self._device
        if out_residual is None:
            out_residual = wp.empty((var.batch_size, self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        use_sqrt = self.residual_mode == "sqrt_abs"
        wp.launch(
            kernel=compute_distance_energy_residual_kernel,
            dim=(var.batch_size, self.num_contact_points),
            inputs=[
                contact_points_world,
                closest_points,
                self.residual_weight,
                use_sqrt,
                self.residual_eps,
                row_offset,
                out_residual,
            ],
            device=kernel_device,
        )

        return out_residual

    def compute_weighted_cost_and_gradient(
        self,
        var_values: VarValues,
        out_cost: wp.array,
        out_gradient: Optional[wp.array] = None,
        local_contact_points: Optional[wp.array] = None,
        contact_points_link_indices: Optional[wp.array] = None,
        scene_offsets: Optional[wp.array] = None,
        precomputed_contact_points_world: Optional[wp.array] = None,
        precomputed_signed_dists: Optional[wp.array] = None,
        precomputed_closest_points: Optional[wp.array] = None,
        precomputed_normals: Optional[wp.array] = None,
    ):
        """Compute cost and optional gradient.

        Lifecycle:
            1. Transform local contact points or reuse supplied world points.
            2. Query or reuse closest points, signed distances, and normals.
            3. Compute residuals and accumulate cost and `Jᵀr`.
        """
        col_offset = var_values.tangent_offset(self.var_key)
        var = var_values.get(self.var_key)
        self._ensure_allocation(var)
        if not var.is_fk_computed:
            var = self.robot.forward_kinematics(var)
        if not var.is_motion_subspace_computed:
            var = self.robot.compute_motion_subspace(var)

        if precomputed_contact_points_world is not None:
            contact_points_world = precomputed_contact_points_world
            if contact_points_link_indices is not None:
                link_indices = contact_points_link_indices
            else:
                link_indices = self.contact_points_link_indices
        elif local_contact_points is not None and contact_points_link_indices is not None:
            contact_points_world = self._transform_contact_points(
                var, local_contact_points, contact_points_link_indices
            )
            link_indices = contact_points_link_indices
        elif self.points_var_key is not None:
            chosen = var_values.get(self.points_var_key)
            contact_points_world = self._transform_contact_points(var, chosen.points, chosen.link_indices)
            link_indices = chosen.link_indices
        else:
            contact_points_world = self._transform_contact_points(
                var, self.local_contact_points, self.contact_points_link_indices
            )
            link_indices = self.contact_points_link_indices

        batch_size = var.batch_size
        if (
            precomputed_signed_dists is not None
            and precomputed_closest_points is not None
            and precomputed_normals is not None
        ):
            signed_dists = precomputed_signed_dists
            closest_points = precomputed_closest_points
            normals = precomputed_normals
        else:
            sdf_scene_offsets = scene_offsets if scene_offsets is not None else _scene_offsets_for(self, var.batch_size)
            self.warp_meshes.query_sdf(
                contact_points_world,
                sdf_scene_offsets,
                out_signed_dists=self._direct_signed_dists,
                out_normals=self._direct_normals,
                out_closest_points=self._closest_points_buf,
            )
            wp.launch(
                kernel=_compute_closest_point_normals_kernel,
                dim=(batch_size, self.num_contact_points),
                inputs=[contact_points_world, self._closest_points_buf, self._direct_signed_dists],
                outputs=[self._direct_normals],
                device=self._device,
            )
            signed_dists = self._direct_signed_dists
            closest_points = self._closest_points_buf
            normals = self._direct_normals

        use_sqrt = self.residual_mode == "sqrt_abs"
        wp.launch(
            kernel=compute_distance_energy_residual_kernel,
            dim=(batch_size, self.num_contact_points),
            inputs=[
                contact_points_world,
                closest_points,
                self.residual_weight,
                use_sqrt,
                self.residual_eps,
                0,
                self._direct_residual_buf,
            ],
            device=self._device,
        )

        spec_tensors = var.spec_tensors
        wp.launch(
            kernel=_compute_distance_cost_and_gradient_kernel,
            dim=(batch_size, self.num_contact_points, var.tangent_dim),
            inputs=[
                var.S_world,
                var.T_world_base.flatten(),
                contact_points_world,
                link_indices,
                spec_tensors.link_ancestor_joints_mask,
                spec_tensors.joints_to_actuated_mapping,
                signed_dists,
                normals,
                var.has_floating_base,
                self.residual_weight,
                self._direct_residual_buf,
                use_sqrt,
                self.residual_eps,
                col_offset,
            ],
            outputs=[out_cost, out_gradient],
            device=self._device,
        )

    def compute_weighted_jacobian_analytic(
        self,
        var_values: VarValues,
        local_contact_points: Optional[wp.array] = None,
        contact_points_link_indices: Optional[wp.array] = None,
        scene_offsets: Optional[wp.array] = None,
        out_jacobian: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        var = var_values.get(self.var_key)
        assert var_values.tangent_offset(self.var_key) == 0
        self._ensure_allocation(var)

        if not var.is_fk_computed:
            var = self.robot.forward_kinematics(var)
        if not var.is_motion_subspace_computed:
            var = self.robot.compute_motion_subspace(var)

        batch_size = var.batch_size

        if local_contact_points is not None and contact_points_link_indices is not None:
            contact_points_world = self._transform_contact_points(
                var, local_contact_points, contact_points_link_indices
            )
        elif self.points_var_key is not None:
            chosen = var_values.get(self.points_var_key)
            contact_points_world = self._transform_contact_points(var, chosen.points, chosen.link_indices)
            contact_points_link_indices = chosen.link_indices
        elif self.local_contact_points is not None and self.contact_points_link_indices is not None:
            contact_points_world = self._transform_contact_points(
                var,
                self.local_contact_points,
                self.contact_points_link_indices,
            )
            contact_points_link_indices = self.contact_points_link_indices
        else:
            raise ValueError(
                "Contact data not provided. Pass local_contact_points and contact_points_link_indices "
                "as arguments or set them on the task."
            )

        total_dofs = var.tangent_dim
        kernel_device = out_jacobian.device if out_jacobian is not None else self._device

        if out_jacobian is None:
            out_jacobian = wp.zeros(
                (batch_size, self.residual_dim, total_dofs),
                dtype=wp.float32,
                device=kernel_device,
            )
            row_offset = 0

        sdf_scene_offsets = scene_offsets if scene_offsets is not None else _scene_offsets_for(self, var.batch_size)
        signed_dists = wp.empty((batch_size, self.num_contact_points), dtype=wp.float32, device=kernel_device)
        closest_points = wp.empty((batch_size, self.num_contact_points), dtype=wp.vec3, device=kernel_device)
        self.warp_meshes.query_sdf(
            contact_points_world,
            sdf_scene_offsets,
            out_signed_dists=signed_dists,
            out_closest_points=closest_points,
        )

        normals = wp.empty((batch_size, self.num_contact_points), dtype=wp.vec3, device=kernel_device)
        wp.launch(
            kernel=_compute_closest_point_normals_kernel,
            dim=(batch_size, self.num_contact_points),
            inputs=[contact_points_world, closest_points, signed_dists],
            outputs=[normals],
            device=kernel_device,
        )

        spec_tensors = var.spec_tensors

        use_sqrt = self.residual_mode == "sqrt_abs"
        wp.launch(
            kernel=compute_scene_distance_jacobian_kernel,
            dim=(batch_size, self.num_contact_points, total_dofs),
            inputs=[
                var.S_world,
                var.T_world_base.flatten(),
                contact_points_world,
                contact_points_link_indices,
                spec_tensors.link_ancestor_joints_mask,
                spec_tensors.joints_to_actuated_mapping,
                signed_dists,
                normals,
                var.has_floating_base,
                self.residual_weight,
                use_sqrt,
                self.residual_eps,
                row_offset,
                out_jacobian,
            ],
            device=kernel_device,
        )

        return out_jacobian


__all__ = ["SceneDistanceTask"]
