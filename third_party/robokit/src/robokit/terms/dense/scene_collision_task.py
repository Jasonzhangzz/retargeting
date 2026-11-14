# pyright: reportArgumentType=false
# pyright: reportOperatorIssue=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportIncompatibleVariableOverride=false
"""Sphere-based robot collision against a `WarpScene`."""

from typing import Dict, Literal, Optional, Sequence, Tuple, Union, cast

import numpy as np
import warp as wp

from robokit.geom import BaseGeom, WarpScene
from robokit.opt.var_values import VarValues
from robokit.robo.robot import Robot
from robokit.robo.robot_state import RobotState
from robokit.terms.dense.scene_distance_task import _compute_point_jacobian_column_func
from robokit.terms.robot_task import RobotTask
from robokit.terms.sphere_collision_kernels import (
    PENALTY_PLAIN,
    PENALTY_SMOOTH,
    PENALTY_SURFACE_DISTANCE,
    _compute_collision_penalty_func,
    sphere_collision_residual_apply_kernel,
    sphere_to_scene_signed_dist_from_query,
)
from robokit.terms.task import Context, GradientTask, ResidualTask
from robokit.utils.warp_utils import wp_device_type, wp_vec7


# --- device code ------------------------------------------------------------
@wp.kernel
def _sphere_collision_jacobian_kernel(
    S_world: wp.array3d(dtype=wp.float32),
    T_world_base: wp.array1d(dtype=wp_vec7),
    sphere_centers_world: wp.array2d(dtype=wp.vec3),
    sphere_indices: wp.array1d(dtype=wp.int32),
    collision_spheres_link_indices: wp.array1d(dtype=wp.int32),
    raw_signed_dists: wp.array2d(dtype=wp.float32),
    raw_normals: wp.array2d(dtype=wp.vec3),
    closest_points: wp.array2d(dtype=wp.vec3),
    radii_global: wp.array1d(dtype=wp.float32),
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    has_floating_base: wp.bool,
    residual_weight: wp.array1d(dtype=wp.float32),
    activation_dist: float,
    use_sqrt: wp.bool,
    eps: float,
    penalty_mode: wp.int32,
    row_offset: int,
    out_jacobian: wp.array3d(dtype=wp.float32),
):
    instance, sphere_idx, col_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    global_idx = sphere_indices[sphere_idx]
    signed_dist = sphere_to_scene_signed_dist_from_query(
        raw_signed_dists, sphere_centers_world, closest_points, radii_global, False, instance, global_idx
    )
    if penalty_mode != PENALTY_SURFACE_DISTANCE and signed_dist > wp.float32(activation_dist):
        out_jacobian[instance, row_offset + sphere_idx, col_idx] = wp.float32(0.0)
        return
    center = sphere_centers_world[instance, global_idx]
    penalty = _compute_collision_penalty_func(signed_dist, wp.float32(activation_dist), penalty_mode)
    scale = penalty[1]
    if use_sqrt and penalty[0] > wp.float32(0.0):
        scale *= wp.float32(0.5) / wp.sqrt(penalty[0] + eps)
    column = _compute_point_jacobian_column_func(
        S_world,
        T_world_base,
        link_ancestor_joints_mask,
        joints_to_actuated,
        instance,
        collision_spheres_link_indices[global_idx],
        col_idx,
        has_floating_base,
        center,
    )
    out_jacobian[instance, row_offset + sphere_idx, col_idx] = (
        residual_weight[sphere_idx] * scale * wp.dot(raw_normals[instance, global_idx], column)
    )


@wp.kernel
def _compute_sphere_skip_mask_kernel(
    link_signed_dists: wp.array2d(dtype=wp.float32),
    link_radii: wp.array1d(dtype=wp.float32),
    sphere_link_indices: wp.array1d(dtype=wp.int32),
    sphere_mask: wp.array1d(dtype=wp.int32),
    activation_dist: float,
    sphere_skip_mask_flat: wp.array1d(dtype=wp.int32),
):
    """Mark sphere queries that this term reads and that pass the link bounding-sphere filter."""
    instance, global_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    n_sph = sphere_link_indices.shape[0]
    link = sphere_link_indices[global_idx]
    signed_dist = link_signed_dists[instance, link]
    r = link_radii[link]
    if sphere_mask[global_idx] == 0 or signed_dist - r > wp.float32(activation_dist):
        sphere_skip_mask_flat[instance * n_sph + global_idx] = 0
    else:
        sphere_skip_mask_flat[instance * n_sph + global_idx] = 1


@wp.kernel
def _expand_scene_indices_kernel(
    scene_indices: wp.array1d(dtype=wp.int32),
    points_per_instance: int,
    scene_per_point: wp.array1d(dtype=wp.int32),
):
    point_idx = wp.tid()
    scene_per_point[point_idx] = scene_indices[point_idx // points_per_instance]


@wp.kernel
def _compute_collision_jtr_cost_and_gradient_kernel(
    S_world: wp.array3d(dtype=wp.float32),
    T_world_base: wp.array1d(dtype=wp_vec7),
    sphere_centers_world: wp.array2d(dtype=wp.vec3),
    local_collision_sphere_radii: wp.array1d(dtype=wp.float32),
    sphere_indices: wp.array1d(dtype=wp.int32),
    collision_spheres_link_indices: wp.array1d(dtype=wp.int32),
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    signed_dists: wp.array2d(dtype=wp.float32),
    normals: wp.array2d(dtype=wp.vec3),
    has_floating_base: wp.bool,
    residual_weight: wp.array1d(dtype=wp.float32),
    out_residual: wp.array2d(dtype=wp.float32),
    activation_dist: float,
    use_sqrt: wp.bool,
    eps: float,
    penalty_mode: wp.int32,
    col_offset: int,
    out_cost: wp.array1d(dtype=wp.float32),
    out_gradient: wp.array2d(dtype=wp.float32),
):
    instance, sphere_idx, col_idx = wp.tid()  # pyright: ignore[reportAssignmentType]

    sphere_global_idx = sphere_indices[sphere_idx]
    radius = local_collision_sphere_radii[sphere_global_idx]
    signed_dist = signed_dists[instance, sphere_global_idx]
    normal = normals[instance, sphere_global_idx]
    signed_dist = signed_dist - radius

    penalty = _compute_collision_penalty_func(signed_dist, wp.float32(activation_dist), penalty_mode)
    dres_dsd = penalty[1]
    if use_sqrt and penalty[0] > wp.float32(0.0):
        dres_dsd = dres_dsd * wp.float32(0.5) / wp.sqrt(penalty[0] + eps)

    r = out_residual[instance, sphere_idx]

    if col_idx == 0:
        wp.atomic_add(out_cost, instance, wp.float32(0.5) * r * r)

    if dres_dsd == wp.float32(0.0):
        return

    center = sphere_centers_world[instance, sphere_global_idx]

    point_column = _compute_point_jacobian_column_func(
        S_world,
        T_world_base,
        link_ancestor_joints_mask,
        joints_to_actuated,
        instance,
        collision_spheres_link_indices[sphere_global_idx],
        col_idx,
        has_floating_base,
        center,
    )
    dsd_dv = wp.dot(normal, point_column)
    weighted_jac = residual_weight[sphere_idx] * dres_dsd * dsd_dv
    contribution = weighted_jac * r
    if contribution != wp.float32(0.0):
        wp.atomic_add(out_gradient, instance, col_offset + col_idx, contribution)


@wp.kernel
def _compute_collision_direct_per_batch_sum_kernel(
    signed_dists: wp.array2d(dtype=wp.float32),
    local_collision_sphere_radii: wp.array1d(dtype=wp.float32),
    sphere_indices: wp.array1d(dtype=wp.int32),
    margin: float,
    use_sqrt: wp.bool,
    eps: float,
    penalty_mode: wp.int32,
    num_frames: int,
    per_batch_sum: wp.array1d(dtype=wp.float32),
):
    b, t, s = wp.tid()  # type: ignore[misc]
    sphere_global_idx = sphere_indices[s]
    radius = local_collision_sphere_radii[sphere_global_idx]
    frame_global_idx = b * num_frames + t
    sd = signed_dists[frame_global_idx, sphere_global_idx]
    residual = _compute_collision_penalty_func(sd - radius, margin, penalty_mode)[0]
    if use_sqrt:
        residual = wp.sqrt(residual + eps)
    wp.atomic_add(per_batch_sum, b, residual)


@wp.kernel
def _accumulate_collision_direct_cost_kernel(
    per_batch_sum: wp.array1d(dtype=wp.float32),
    weight_over_B: float,
    out_cost: wp.array1d(dtype=wp.float32),
):
    b = wp.tid()
    s = per_batch_sum[b]
    out_cost[b] = out_cost[b] + weight_over_B * s * s


@wp.kernel
def _compute_collision_direct_gradient_kernel(
    S_world: wp.array3d(dtype=wp.float32),
    T_world_base: wp.array1d(dtype=wp_vec7),
    sphere_centers_world: wp.array2d(dtype=wp.vec3),
    local_collision_sphere_radii: wp.array1d(dtype=wp.float32),
    sphere_indices: wp.array1d(dtype=wp.int32),
    collision_spheres_link_indices: wp.array1d(dtype=wp.int32),
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    signed_dists: wp.array2d(dtype=wp.float32),
    normals: wp.array2d(dtype=wp.vec3),
    has_floating_base: wp.bool,
    margin: float,
    use_sqrt: wp.bool,
    eps: float,
    penalty_mode: wp.int32,
    two_weight_over_B: float,
    per_batch_sum: wp.array1d(dtype=wp.float32),
    num_frames: int,
    single_tangent_dim: int,
    skip_endpoints: int,
    col_offset: int,
    out_gradient: wp.array2d(dtype=wp.float32),
):
    b, t, s, col = wp.tid()  # type: ignore[misc]
    if skip_endpoints != 0:
        if t == 0 or t == num_frames - 1:
            return

    sphere_global_idx = sphere_indices[s]
    frame_global_idx = b * num_frames + t
    radius = local_collision_sphere_radii[sphere_global_idx]
    sd = signed_dists[frame_global_idx, sphere_global_idx]
    penalty = _compute_collision_penalty_func(sd - radius, margin, penalty_mode)
    dres_dsd = penalty[1]
    if use_sqrt and penalty[0] > wp.float32(0.0):
        dres_dsd = dres_dsd * wp.float32(0.5) / wp.sqrt(penalty[0] + eps)
    if dres_dsd == wp.float32(0.0):
        return

    center = sphere_centers_world[frame_global_idx, sphere_global_idx]
    normal = normals[frame_global_idx, sphere_global_idx]
    point_column = _compute_point_jacobian_column_func(
        S_world,
        T_world_base,
        link_ancestor_joints_mask,
        joints_to_actuated,
        frame_global_idx,
        collision_spheres_link_indices[sphere_global_idx],
        col,
        has_floating_base,
        center,
    )

    sum_b = per_batch_sum[b]
    grad_contrib = two_weight_over_B * sum_b * dres_dsd * wp.dot(normal, point_column)

    grad_col = col_offset + t * single_tangent_dim + col
    wp.atomic_add(out_gradient, b, grad_col, grad_contrib)


# --- task -------------------------------------------------------------------
class SceneCollisionTask(RobotTask, ResidualTask, GradientTask):
    """Penalize robot collision spheres that approach scene geometry."""

    precompute_collision_geometry = "sphere"

    def __init__(
        self,
        robot: Optional[Robot] = None,
        scene: Optional[Union[WarpScene, Context]] = None,
        geometry: Optional[BaseGeom] = None,  # restrict queries to one scene geometry
        scene_indices: Optional[Union[wp.array, Context]] = None,
        use_link_bounding_sphere_filter: bool = False,
        weight: Union[float, Sequence[float]] = 1.0,
        margin: Optional[float] = 0.02,
        sphere_indices: Optional[Sequence[int]] = None,
        residual_mode: Literal["abs", "sqrt_abs"] = "abs",
        residual_eps: float = 1e-6,
        penalty: Literal["smooth", "plain", "surface_distance"] = "smooth",
        skip_trajectory_endpoints: bool = True,  # preserve pinned trajectory endpoints
        movable_only: bool = False,  # without `sphere_indices`: only the spheres off the base link, which never moves
        name: str = "",
    ):
        self.name = name
        self.robot = robot
        self.scene = scene
        self.geometry = geometry
        self.scene_indices = scene_indices
        self.use_link_bounding_sphere_filter = use_link_bounding_sphere_filter
        self.weight = weight
        self.margin = margin
        self.sphere_indices = sphere_indices
        self.residual_mode = residual_mode
        self.residual_eps = residual_eps
        self.penalty = penalty
        self.skip_trajectory_endpoints = skip_trajectory_endpoints
        self.movable_only = movable_only

        self._device: Optional[wp_device_type] = None
        self._sphere_indices_np: Optional[np.ndarray] = None
        self._residual_weight_np: Optional[np.ndarray] = None
        self._sphere_indices_wp: Optional[wp.array] = None
        self._sphere_mask_np: Optional[np.ndarray] = None  # 1 for spheres this term reads; None when it reads all
        self._sphere_mask_wp: Optional[wp.array] = None
        self._residual_weight_wp: Optional[wp.array] = None
        self._scene_indices_wp: Optional[wp.array] = None
        self._sphere_scene_per_point_wp: Optional[wp.array] = None
        self._scene_input_batch_size = 0
        self._link_scene_per_point_wp: Optional[wp.array] = None
        self._raw_signed_dists: Optional[wp.array] = None
        self._raw_normals: Optional[wp.array] = None
        self._raw_closest_points: Optional[wp.array] = None
        self._link_signed_dists: Optional[wp.array] = None
        self._sphere_skip_mask_flat: Optional[wp.array] = None
        self._num_spheres = 0
        self._cached_batch_size = 0
        # direct GD/L-BFGS cost and gradient buffers
        self._direct_signed_dists: Optional[wp.array] = None
        self._direct_normals: Optional[wp.array] = None
        self._direct_closest_points: Optional[wp.array] = None
        self._direct_scene_per_point_wp: Optional[wp.array] = None
        self._direct_link_scene_per_point_wp: Optional[wp.array] = None
        self._direct_per_batch_sum: Optional[wp.array] = None
        self._direct_dummy_T_world_base: Optional[wp.array] = None
        self._cached_direct_shape: Tuple[int, int] = (0, 0)
        self._jtr_residual_bufs: Dict[int, wp.array] = {}
        self._direct_link_signed_dists: Optional[wp.array] = None
        self._direct_skip_mask_flat: Optional[wp.array] = None

        if self.penalty not in {"smooth", "plain", "surface_distance"}:
            raise ValueError(f"Unsupported penalty: {self.penalty}")
        if self.robot is None or not self.active:
            return  # config-time stub, or nothing to check
        if not self.robot.spec.has_collision_spheres:
            raise RuntimeError("Robot has no collision spheres. Load with load_collision_spheres=True.")
        if self.scene_indices is not None and (self.scene_indices.dtype != wp.int32 or self.scene_indices.ndim != 1):
            raise TypeError("scene_indices must be a 1D wp.int32 array.")
        num_spheres = int(self.robot.spec.local_collision_sphere_centers.shape[0])
        self._num_spheres = num_spheres
        if self.sphere_indices is None and self.movable_only:
            self.sphere_indices = np.flatnonzero(self.robot.spec.collision_spheres_link_indices > 0)
        if self.sphere_indices is None:
            sphere_indices_np = np.arange(num_spheres, dtype=np.int32)
        else:
            sphere_indices_np = np.asarray(self.sphere_indices, dtype=np.int32)
            if sphere_indices_np.ndim != 1:
                raise ValueError("sphere_indices must be 1D.")
            if np.any(sphere_indices_np < 0) or np.any(sphere_indices_np >= num_spheres):
                raise ValueError("sphere_indices contains out-of-range indices.")
        self._sphere_indices_np = sphere_indices_np
        if len(np.unique(sphere_indices_np)) < num_spheres:  # a subset queries only its own spheres
            self._sphere_mask_np = np.zeros(num_spheres, dtype=np.int32)
            self._sphere_mask_np[sphere_indices_np] = 1
        residual_weight = np.ones((sphere_indices_np.shape[0],), dtype=np.float32)
        if isinstance(self.weight, (float, int)):
            residual_weight[:] = float(self.weight)
        else:
            weight_arr = np.asarray(self.weight, dtype=np.float32)
            if weight_arr.shape != (sphere_indices_np.shape[0],):
                raise ValueError(f"Expected weight shape ({sphere_indices_np.shape[0]},), got {weight_arr.shape}.")
            residual_weight[:] = weight_arr
        self._residual_weight_np = residual_weight

    @property
    def active(self) -> bool:
        """Whether the term adds cost: nonzero weight and a scene with geometry, or a helper's scene to come."""
        scene = self.scene
        return bool(np.any(self.weight)) and scene is not None and (isinstance(scene, Context) or bool(scene.geoms))

    def set_robot(self, robot: Robot):
        """Build a config-time stub, with fresh query buffers."""
        SceneCollisionTask.__init__(
            self,
            robot,
            self.scene,
            self.geometry,
            self.scene_indices,
            self.use_link_bounding_sphere_filter,
            self.weight,
            self.margin,
            self.sphere_indices,
            self.residual_mode,
            self.residual_eps,
            self.penalty,
            self.skip_trajectory_endpoints,
            self.movable_only,
            self.name,
        )

    @property
    def _penalty_mode(self) -> int:
        if self.penalty == "plain":
            return PENALTY_PLAIN
        if self.penalty == "surface_distance":
            return PENALTY_SURFACE_DISTANCE
        return PENALTY_SMOOTH

    def set_scene_indices(self, scene_indices: wp.array):
        """Copy a per-query device scene mapping into stable query buffers."""
        if scene_indices.dtype != wp.int32 or scene_indices.ndim != 1:
            raise TypeError("scene_indices must be a 1D wp.int32 array.")
        batch_size = self._cached_batch_size or self._cached_direct_shape[0]
        if batch_size != 0 and (scene_indices.shape[0] == 0 or batch_size % scene_indices.shape[0]):
            raise ValueError(
                "scene_indices must have shape (batch_size,) or divide it (one scene per candidate group)."
            )
        self.scene_indices = scene_indices
        if self._device is not None:
            if scene_indices.device != wp.get_device(self._device):
                raise ValueError(f"scene_indices must be on {wp.get_device(self._device)}, got {scene_indices.device}.")
            if self._scene_indices_wp is None:
                self._scene_indices_wp = wp.empty(self._cached_batch_size, dtype=wp.int32, device=self._device)
                self._sphere_scene_per_point_wp = wp.empty(
                    self._cached_batch_size * self._num_spheres, dtype=wp.int32, device=self._device
                )
                if self.use_link_bounding_sphere_filter:
                    self._link_scene_per_point_wp = wp.empty(
                        self._cached_batch_size * self.robot.spec.num_links, dtype=wp.int32, device=self._device
                    )
                self._cached_direct_shape = (0, 0)
            self._scene_input_batch_size = scene_indices.shape[0]
            wp.copy(self._scene_indices_wp[: self._scene_input_batch_size], scene_indices)

    @property
    def residual_dim(self) -> int:
        if self.sphere_indices is None:
            return self._num_spheres
        return len(self.sphere_indices)

    def _ensure_allocation(self, var: RobotState) -> None:
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
        n_sph, n_link = self._num_spheres, self.robot.spec.num_links

        self._scene_indices_wp = wp.zeros(batch_size, dtype=wp.int32, device=device)
        self._scene_input_batch_size = batch_size
        if self.scene_indices is not None:  # the caller's buffer, read directly
            n = self.scene_indices.shape[0]
            if n == 0 or batch_size % n:
                raise ValueError(
                    "scene_indices must have shape (batch_size,) or divide it (one scene per candidate group)."
                )
            if self.scene_indices.device != wp.get_device(device):
                raise ValueError(f"scene_indices must be on {wp.get_device(device)}, got {self.scene_indices.device}.")
            self._scene_input_batch_size = n
            self._scene_indices_wp = self.scene_indices
        self._sphere_scene_per_point_wp = wp.empty(batch_size * n_sph, dtype=wp.int32, device=device)

        self._sphere_indices_wp = wp.from_numpy(self._sphere_indices_np, dtype=wp.int32, device=device)
        self._residual_weight_wp = wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device=device)
        self._raw_signed_dists = wp.empty((batch_size, n_sph), dtype=wp.float32, device=device)
        self._raw_normals = wp.empty((batch_size, n_sph), dtype=wp.vec3, device=device)
        self._raw_closest_points = wp.empty((batch_size, n_sph), dtype=wp.vec3, device=device)
        mask = self._sphere_mask_np if self._sphere_mask_np is not None else np.ones(n_sph, dtype=np.int32)
        self._sphere_mask_wp = wp.from_numpy(mask, dtype=wp.int32, device=device)
        if self._sphere_mask_np is not None:  # static unless the link filter rewrites it per query
            self._sphere_skip_mask_flat = wp.from_numpy(np.tile(mask, batch_size), dtype=wp.int32, device=device)
        if self.use_link_bounding_sphere_filter:
            self._link_signed_dists = wp.empty((batch_size, n_link), dtype=wp.float32, device=device)
            self._sphere_skip_mask_flat = wp.empty((batch_size * n_sph,), dtype=wp.int32, device=device)
            self._link_scene_per_point_wp = wp.empty(batch_size * n_link, dtype=wp.int32, device=device)

    def _run_query(self, var: RobotState):
        spec_t = var.spec_tensors
        b, n_sph = var.batch_size, self._num_spheres
        candidates = max(
            1, b // self._scene_input_batch_size
        )  # elements per scene index: inputs broadcast over the candidates
        sphere_skip_mask: Optional[wp.array] = None
        if self._sphere_mask_np is not None:
            sphere_skip_mask = cast(wp.array, cast(wp.array, self._sphere_skip_mask_flat)[: b * n_sph])
        # buffers hold the capacity batch; a narrower live batch queries a prefix
        sphere_scene = cast(wp.array, self._sphere_scene_per_point_wp)[: b * n_sph]
        wp.launch(
            _expand_scene_indices_kernel,
            dim=b * n_sph,
            inputs=[self._scene_indices_wp, candidates * n_sph],
            outputs=[sphere_scene],
            device=self._device,
            record_tape=False,
        )

        if self.use_link_bounding_sphere_filter:
            n_link = self.robot.spec.num_links
            link_scene = cast(wp.array, self._link_scene_per_point_wp)[: b * n_link]
            wp.launch(
                _expand_scene_indices_kernel,
                dim=b * n_link,
                inputs=[self._scene_indices_wp, candidates * n_link],
                outputs=[link_scene],
                device=self._device,
                record_tape=False,
            )
            (self.geometry or self.scene).query_sdf(
                var.link_bounding_sphere_centers_world,
                scene_indices=link_scene,
                out_signed_dists=cast(wp.array, self._link_signed_dists)[:b],
                distance_only=True,
                record_tape=False,
            )
            wp.launch(
                kernel=_compute_sphere_skip_mask_kernel,
                dim=(b, n_sph),
                inputs=[
                    self._link_signed_dists,
                    spec_t.link_bounding_sphere_radii,
                    spec_t.collision_spheres_link_indices,
                    self._sphere_mask_wp,
                    self.margin,
                ],
                outputs=[self._sphere_skip_mask_flat],
                device=self._device,
                record_tape=False,
            )
            sphere_skip_mask = cast(wp.array, cast(wp.array, self._sphere_skip_mask_flat)[: b * n_sph])

        (self.geometry or self.scene).query_sdf(
            var.collision_sphere_centers_world,
            scene_indices=sphere_scene,
            out_signed_dists=cast(wp.array, self._raw_signed_dists)[:b],
            out_normals=cast(wp.array, self._raw_normals)[:b],
            out_closest_points=cast(wp.array, self._raw_closest_points)[:b],
            query_mask=sphere_skip_mask,
            record_tape=False,
        )

    def compute_weighted_residual(
        self,
        var_values: VarValues,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        var = var_values.get(self.var_key)
        if var.is_trajectory:
            raise NotImplementedError("SceneCollisionTask does not support trajectory states yet.")
        self._ensure_allocation(var)
        if not var.is_fk_computed:
            self.robot.forward_kinematics(var)
        if not var.is_collision_spheres_computed:
            self.robot.transform_collision_spheres(var)

        if out_residual is None:
            out_residual = wp.empty((var.batch_size, self.residual_dim), dtype=wp.float32, device=self._device)
            row_offset = 0

        # Rebuild distance from closest points for gradients; scene queries have no adjoint and stay off tape.
        self._run_query(var)

        wp.launch(
            kernel=sphere_collision_residual_apply_kernel,
            dim=(var.batch_size, self.residual_dim),
            inputs=[
                self._raw_signed_dists,
                var.collision_sphere_centers_world,
                self._raw_closest_points,
                self._sphere_indices_wp,
                var.spec_tensors.collision_sphere_radii,
                var.collision_sphere_centers_world.requires_grad,  # a gradient probe wants the rebuilt distance
                self._residual_weight_wp,
                self.margin,
                self.residual_mode == "sqrt_abs",
                self.residual_eps,
                self._penalty_mode,
                row_offset,
            ],
            outputs=[out_residual],
            device=self._device,
        )
        return out_residual

    def compute_weighted_jacobian_analytic(
        self,
        var_values: VarValues,
        out_jacobian: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        var = var_values.get(self.var_key)
        assert var_values.tangent_offset(self.var_key) == 0
        if var.is_trajectory:
            raise NotImplementedError("SceneCollisionTask does not support trajectory states yet.")
        self._ensure_allocation(var)
        if not var.is_fk_computed:
            self.robot.forward_kinematics(var)
        if not var.is_collision_spheres_computed:
            self.robot.transform_collision_spheres(var)
        if not var.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(var)

        total_dofs = var.tangent_dim
        if out_jacobian is None:
            out_jacobian = wp.zeros(
                (var.batch_size, self.residual_dim, total_dofs), dtype=wp.float32, device=self._device
            )
            row_offset = 0

        spec_t = var.spec_tensors
        wp.launch(
            kernel=_sphere_collision_jacobian_kernel,
            dim=(var.batch_size, self.residual_dim, total_dofs),
            inputs=[
                var.S_world,
                var.T_world_base.flatten(),
                var.collision_sphere_centers_world,
                self._sphere_indices_wp,
                spec_t.collision_spheres_link_indices,
                self._raw_signed_dists,
                self._raw_normals,
                self._raw_closest_points,
                spec_t.collision_sphere_radii,
                spec_t.link_ancestor_joints_mask,
                spec_t.joints_to_actuated_mapping,
                var.has_floating_base,
                self._residual_weight_wp,
                self.margin,
                self.residual_mode == "sqrt_abs",
                self.residual_eps,
                self._penalty_mode,
                row_offset,
            ],
            outputs=[out_jacobian],
            device=self._device,
        )
        return out_jacobian

    # --- analytical cost and gradient ---
    def _init_direct_buffers(self, batch_size: int, num_frames: int, device: wp_device_type):
        BT = batch_size * num_frames
        self._device = device
        if self._sphere_indices_wp is None:
            self._sphere_indices_wp = wp.from_numpy(self._sphere_indices_np, dtype=wp.int32, device=device)
        if self._residual_weight_wp is None:
            self._residual_weight_wp = wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device=device)
        mask = self._sphere_mask_np if self._sphere_mask_np is not None else np.ones(self._num_spheres, dtype=np.int32)
        if self._sphere_mask_wp is None:
            self._sphere_mask_wp = wp.from_numpy(mask, dtype=wp.int32, device=device)
        self._direct_skip_mask_flat = None
        if self._sphere_mask_np is not None:  # static unless the link filter rewrites it per query
            self._direct_skip_mask_flat = wp.from_numpy(np.tile(mask, BT), dtype=wp.int32, device=device)
        self._direct_signed_dists = wp.empty((BT, self._num_spheres), dtype=wp.float32, device=device)
        self._direct_normals = wp.empty((BT, self._num_spheres), dtype=wp.vec3, device=device)
        self._direct_closest_points = wp.empty((BT, self._num_spheres), dtype=wp.vec3, device=device)
        self._direct_per_batch_sum = wp.zeros((batch_size,), dtype=wp.float32, device=device)
        if self.scene_indices is not None:  # the caller's buffer, read directly
            if self.scene_indices.shape[0] == 0 or batch_size % self.scene_indices.shape[0]:
                raise ValueError(
                    "scene_indices must have shape (batch_size,) or divide it (one scene per candidate group)."
                )
            if self.scene_indices.device != wp.get_device(device):
                raise ValueError(f"scene_indices must be on {wp.get_device(device)}, got {self.scene_indices.device}.")
            self._scene_indices_wp = self.scene_indices
        elif self._scene_indices_wp is None or self._scene_indices_wp.shape != (batch_size,):
            self._scene_indices_wp = wp.zeros(batch_size, dtype=wp.int32, device=device)
        self._direct_scene_per_point_wp = wp.empty(BT * self._num_spheres, dtype=wp.int32, device=device)
        self._direct_dummy_T_world_base = wp.empty((1,), dtype=wp_vec7, device=device)
        self._cached_direct_shape = (batch_size, num_frames)
        if self.use_link_bounding_sphere_filter:
            n_link = self.robot.spec.num_links
            self._direct_link_signed_dists = wp.empty((BT, n_link), dtype=wp.float32, device=device)
            self._direct_skip_mask_flat = wp.empty((BT * self._num_spheres,), dtype=wp.int32, device=device)
            self._direct_link_scene_per_point_wp = wp.empty(BT * n_link, dtype=wp.int32, device=device)

    def _ensure_direct_state(self, var: RobotState) -> Tuple[int, int]:
        B, T = (var.batch_size, var.shape[1]) if var.is_trajectory else (var.batch_size, 1)
        if self._cached_direct_shape != (B, T):
            self._init_direct_buffers(B, T, var.q.device)
        if not var.is_fk_computed:
            self.robot.forward_kinematics(var)
        if not var.is_collision_spheres_computed:
            self.robot.transform_collision_spheres(var)
        return B, T

    def _query_direct_sdf(self, var: RobotState, B: int, T: int):
        BT, per_scene = B * T, T * (B // self._scene_indices_wp.shape[0])  # a scene index covers a candidate group
        wp.launch(
            _expand_scene_indices_kernel,
            dim=BT * self._num_spheres,
            inputs=[self._scene_indices_wp, per_scene * self._num_spheres],
            outputs=[self._direct_scene_per_point_wp],
            device=self._device,
        )
        extra_kwargs: dict = {}
        if self._sphere_mask_np is not None:
            extra_kwargs["query_mask"] = self._direct_skip_mask_flat
        if self.use_link_bounding_sphere_filter:
            spec_t = var.spec_tensors
            n_link = self.robot.spec.num_links
            wp.launch(
                _expand_scene_indices_kernel,
                dim=BT * n_link,
                inputs=[self._scene_indices_wp, per_scene * n_link],
                outputs=[self._direct_link_scene_per_point_wp],
                device=self._device,
            )
            (self.geometry or self.scene).query_sdf(
                var.link_bounding_sphere_centers_world,
                scene_indices=self._direct_link_scene_per_point_wp,
                out_signed_dists=self._direct_link_signed_dists,
                distance_only=True,
            )
            wp.launch(
                kernel=_compute_sphere_skip_mask_kernel,
                dim=(BT, self._num_spheres),
                inputs=[
                    self._direct_link_signed_dists,
                    spec_t.link_bounding_sphere_radii,
                    spec_t.collision_spheres_link_indices,
                    self._sphere_mask_wp,
                    self.margin,
                ],
                outputs=[self._direct_skip_mask_flat],
                device=self._device,
            )
            extra_kwargs["query_mask"] = self._direct_skip_mask_flat
        (self.geometry or self.scene).query_sdf(
            var.collision_sphere_centers_world,
            out_signed_dists=self._direct_signed_dists,
            out_normals=self._direct_normals,
            out_closest_points=self._direct_closest_points,
            scene_indices=self._direct_scene_per_point_wp,
            **extra_kwargs,
        )

    def compute_weighted_cost_and_gradient(
        self,
        var_values: VarValues,
        out_cost: wp.array,
        out_gradient: Optional[wp.array] = None,
        cost_form: Literal["squared_sum", "sum_squared"] = "squared_sum",
    ):
        """Compute collision cost and optional direct gradient.

        Lifecycle:
            1. Prepare trajectory-shaped collision buffers and kinematics.
            2. Query the scene SDF for every selected sphere.
            3. Accumulate the selected cost form and optional gradient.
        """
        col_offset = var_values.tangent_offset(self.var_key)
        var = var_values.get(self.var_key)
        if out_gradient is None:
            # --- cost-only step ---
            B, T = self._ensure_direct_state(var)
            self._query_direct_sdf(var, B, T)
            spec_t = var.spec_tensors
            pen_weight = float(self._residual_weight_np[0]) if self._residual_weight_np is not None else 1.0
            weight_over_B = pen_weight / float(B)
            self._direct_per_batch_sum.zero_()
            wp.launch(
                kernel=_compute_collision_direct_per_batch_sum_kernel,
                dim=(B, T, self.residual_dim),
                inputs=[
                    self._direct_signed_dists,
                    spec_t.collision_sphere_radii,
                    self._sphere_indices_wp,
                    float(self.margin),
                    self.residual_mode == "sqrt_abs",
                    self.residual_eps,
                    self._penalty_mode,
                    int(T),
                    self._direct_per_batch_sum,
                ],
                device=self._device,
            )
            wp.launch(
                kernel=_accumulate_collision_direct_cost_kernel,
                dim=B,
                inputs=[self._direct_per_batch_sum, float(weight_over_B), out_cost],
                device=self._device,
            )
            return
        if cost_form == "sum_squared":
            self._compute_weighted_cost_and_gradient_jtr(
                var_values, out_cost=out_cost, out_gradient=out_gradient, col_offset=col_offset
            )
            return
        B, T = self._ensure_direct_state(var)
        if not var.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(var)
        self._query_direct_sdf(var, B, T)
        spec_t = var.spec_tensors
        pen_weight = float(self._residual_weight_np[0]) if self._residual_weight_np is not None else 1.0
        weight_over_B = pen_weight / float(B)
        self._direct_per_batch_sum.zero_()
        wp.launch(
            kernel=_compute_collision_direct_per_batch_sum_kernel,
            dim=(B, T, self.residual_dim),
            inputs=[
                self._direct_signed_dists,
                spec_t.collision_sphere_radii,
                self._sphere_indices_wp,
                float(self.margin),
                self.residual_mode == "sqrt_abs",
                self.residual_eps,
                self._penalty_mode,
                int(T),
                self._direct_per_batch_sum,
            ],
            device=self._device,
        )
        wp.launch(
            kernel=_accumulate_collision_direct_cost_kernel,
            dim=B,
            inputs=[self._direct_per_batch_sum, float(weight_over_B), out_cost],
            device=self._device,
        )
        single_tangent_dim = var.tangent_dim // T
        BT = B * T
        T_world_base_arg = var.T_world_base.reshape((BT,)) if var.has_floating_base else self._direct_dummy_T_world_base
        skip_endpoints = 1 if (var.is_trajectory and self.skip_trajectory_endpoints and T > 2) else 0
        wp.launch(
            kernel=_compute_collision_direct_gradient_kernel,
            dim=(B, T, self.residual_dim, single_tangent_dim),
            inputs=[
                var.S_world.reshape((BT, 6, var.spec.num_joints)),
                T_world_base_arg,
                var.collision_sphere_centers_world.reshape((BT, self._num_spheres)),
                spec_t.collision_sphere_radii,
                self._sphere_indices_wp,
                spec_t.collision_spheres_link_indices,
                spec_t.link_ancestor_joints_mask,
                spec_t.joints_to_actuated_mapping,
                self._direct_signed_dists,
                self._direct_normals,
                var.has_floating_base,
                float(self.margin),
                self.residual_mode == "sqrt_abs",
                self.residual_eps,
                self._penalty_mode,
                float(2.0 * weight_over_B),
                self._direct_per_batch_sum,
                int(T),
                int(single_tangent_dim),
                int(skip_endpoints),
                int(col_offset),
            ],
            outputs=[out_gradient],
            device=self._device,
        )

    def _compute_weighted_cost_and_gradient_jtr(
        self,
        var_values: VarValues,
        out_cost: wp.array,
        out_gradient: wp.array,
        col_offset: int = 0,
    ):
        """Compute the per-pose Gauss-Newton cost and `Jᵀr`."""
        var = var_values.get(self.var_key)
        if var.is_trajectory:
            raise NotImplementedError("cost_form='sum_squared' requires per-pose state.")
        if not var.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(var)
        # one buffer per batch size, so alternating solver widths never reallocate inside a graph capture
        if var.batch_size not in self._jtr_residual_bufs:
            self._jtr_residual_bufs[var.batch_size] = wp.empty(
                (var.batch_size, self.residual_dim), dtype=wp.float32, device=var.q.device
            )
        residual = self._jtr_residual_bufs[var.batch_size]
        self.compute_weighted_residual(var_values, out_residual=residual)
        spec_t = var.spec_tensors
        wp.launch(
            kernel=_compute_collision_jtr_cost_and_gradient_kernel,
            dim=(var.batch_size, self.residual_dim, var.tangent_dim),
            inputs=[
                var.S_world,
                var.T_world_base.flatten(),
                var.collision_sphere_centers_world,
                spec_t.collision_sphere_radii,
                self._sphere_indices_wp,
                spec_t.collision_spheres_link_indices,
                spec_t.link_ancestor_joints_mask,
                spec_t.joints_to_actuated_mapping,
                self._raw_signed_dists,
                self._raw_normals,
                var.has_floating_base,
                self._residual_weight_wp,
                residual,
                self.margin,
                self.residual_mode == "sqrt_abs",
                self.residual_eps,
                self._penalty_mode,
                col_offset,
            ],
            outputs=[out_cost, out_gradient],
            device=self._device,
        )


__all__ = ["SceneCollisionTask"]
