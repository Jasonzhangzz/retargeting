# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportOptionalOperand=false
# pyright: reportCallIssue=false
# pyright: reportIncompatibleVariableOverride=false
"""Point-pair residuals for position, vector, and direction targets."""

from typing import Any, List, Literal, Mapping, Optional, Sequence, Union, cast

import numpy as np
import warp as wp

from robokit.lie.se3_kernels import se3_adjoint_func
from robokit.opt.var_values import VarValues
from robokit.robo.robot import Robot
from robokit.terms.robot_task import RobotTask
from robokit.terms.task import Context, ResidualTask
from robokit.utils.warp_utils import wp_device_type, wp_vec6, wp_vec7


# --- device functions -------------------------------------------------------
@wp.func
def _compute_huber_residual_signed_func(error: float, delta: float) -> float:
    abs_error = wp.abs(error)
    if abs_error <= delta:
        return error
    scale = wp.sqrt(wp.max(2.0 * delta * abs_error - delta * delta, 0.0))
    return scale if error >= 0.0 else -scale


@wp.func
def _compute_huber_gradient_scale_func(error: float, delta: float) -> float:
    abs_error = wp.abs(error)
    if abs_error <= delta:
        return 1.0
    denom = wp.sqrt(wp.max(2.0 * delta * abs_error - delta * delta, 1e-12))
    return delta / denom


@wp.func
def _compute_frame_vector_soft_gate_func(
    target_vector: wp.vec3,
    soft_gate_start_distance: float,
    soft_gate_full_distance: float,
    soft_gate_enabled: wp.int32,
) -> float:
    if soft_gate_enabled == 0:
        return 1.0
    target_norm = wp.length(target_vector)
    width = soft_gate_start_distance - soft_gate_full_distance
    if width <= 1e-8:
        return 1.0
    center = 0.5 * (soft_gate_start_distance + soft_gate_full_distance)
    normalized = (center - target_norm) / width
    sigmoid_gate = 1.0 / (1.0 + wp.exp(-2.5 * normalized))
    return sigmoid_gate * sigmoid_gate * (3.0 - 2.0 * sigmoid_gate)


# --- task -------------------------------------------------------------------
class FrameVectorTask(RobotTask, ResidualTask):
    """Drive point-pair deltas to target vectors; `weight_mode` "cost" takes sqrt, "normalized" divides by sqrt(rows).

    Direction metrics:
    - Default: robot-length-scaled vector error.
    - ``direction_only=True`` with ``direction_metric="cosine"``: ``1 - dot(normalized_robot, normalized_target)``.

    Named weights pick their rows when the task is built:
    - A target name tracks that target's row of `targets`, ordered by `target_names`, from the world origin to its link in `target_links`.
    - `(origin link, task link, origin target, task target)` reads the pair row `origin * len(target_names) + task` of a pair buffer.
    """

    num_pairs = 1  # a template counts as active until `resolve` sizes it

    def __init__(
        self,
        robot: Optional[Robot] = None,
        origin_link_indices: Optional[Union[Sequence[int], Context]] = None,
        task_link_indices: Optional[Union[Sequence[int], Context]] = None,
        targets: Optional[Union[np.ndarray, wp.array, Context]] = None,
        weight: Union[float, Sequence[float], np.ndarray, Mapping[Any, float]] = 1.0,
        target_indices: Optional[Sequence[int]] = None,
        scale: Union[float, Sequence[float], np.ndarray] = 1.0,
        huber_delta: Optional[float] = None,
        huber_on_norm: bool = False,
        direction_only: bool = False,
        soft_gate_start_distance: Optional[float] = None,
        soft_gate_full_distance: Optional[float] = None,
        soft_gate_pair_mask: Optional[List[bool]] = None,
        direction_metric: Literal["length_scaled", "cosine"] = "length_scaled",
        name: str = "",
        weight_mode: Literal["residual", "cost", "normalized"] = "residual",
        target_names: Optional[Union[Sequence[str], Context]] = None,
        target_links: Optional[Union[Mapping[str, str], Context]] = None,
        soft_gate_targets: Optional[Union[Sequence[int], Context]] = None,
    ):
        self.name = name
        self.weight_mode = weight_mode
        self.target_names, self.target_links = target_names, target_links
        self.soft_gate_targets = soft_gate_targets
        self.robot: Optional[Robot] = robot
        self.device: Optional[wp_device_type] = None
        self.weight = weight
        self.scale = scale
        self.huber_delta_arg = huber_delta
        self.huber_on_norm_arg = huber_on_norm
        self.direction_only = direction_only
        self.direction_metric = direction_metric
        self.soft_gate_start_distance = soft_gate_start_distance
        self.soft_gate_full_distance = soft_gate_full_distance
        self.soft_gate_pair_mask = soft_gate_pair_mask
        self.targets_arg = targets  # a stub may name its helper's targets with a Context member
        self.origin_link_indices_arg, self.task_link_indices_arg = origin_link_indices, task_link_indices
        self.target_indices_arg = target_indices
        if any(arg is None or isinstance(arg, Context) for arg in (origin_link_indices, task_link_indices, targets)):
            return  # config-time stub
        if len(origin_link_indices) != len(task_link_indices):
            raise ValueError("origin_link_indices and task_link_indices must have the same length.")

        self.num_pairs = len(origin_link_indices)
        self.huber_delta = float(huber_delta) if huber_delta is not None else 0.0
        self.use_huber = huber_delta is not None
        self.huber_on_norm = bool(huber_on_norm) and self.use_huber
        self.direction_only = bool(direction_only)
        self._direction_mode = (2 if direction_metric == "cosine" else 1) if direction_only else 0
        self.channels = 1 if self.huber_on_norm or self._direction_mode == 2 else 3

        self._origin_indices_np = np.asarray(origin_link_indices, dtype=np.int32)
        self._task_indices_np = np.asarray(task_link_indices, dtype=np.int32)

        targets_np = np.zeros(targets.shape) if isinstance(targets, wp.array) else targets
        targets_np = np.asarray(targets_np, dtype=np.float32)
        self._targets_np = targets_np[None] if targets_np.ndim == 2 else targets_np
        self._target_indices_np = (
            np.arange(self.num_pairs, dtype=np.int32)
            if target_indices is None
            else np.asarray(target_indices, dtype=np.int32)
        )
        if self._target_indices_np.shape != (self.num_pairs,) or not np.all(
            (self._target_indices_np >= 0) & (self._target_indices_np < self._targets_np.shape[1])
        ):
            raise ValueError("target_indices must hold one valid target row per pair.")

        self._scale_np = np.broadcast_to(np.asarray(scale, dtype=np.float32), (self.num_pairs,)).astype(np.float32)
        weight_np = np.asarray(weight, dtype=np.float32)
        if weight_mode == "cost":
            weight_np = np.sqrt(weight_np)
        elif weight_mode == "normalized":
            weight_np = weight_np * np.float32(np.sqrt(1.0 / max(self.num_pairs, 1)))
        self._weight_np = np.ascontiguousarray(
            np.atleast_2d(np.broadcast_to(weight_np, (self.num_pairs,))) if weight_np.ndim < 2 else weight_np
        )
        if self._weight_np.shape[-1] != self.num_pairs:
            raise ValueError(f"weight length {self._weight_np.shape[-1]} != num_pairs {self.num_pairs}")

        if (soft_gate_start_distance is None) != (soft_gate_full_distance is None):
            raise ValueError("soft_gate_start_distance and soft_gate_full_distance must both be set or both None.")
        start_scalar, full_scalar = 0.0, 0.0
        if soft_gate_start_distance is not None and soft_gate_full_distance is not None:
            if soft_gate_start_distance <= soft_gate_full_distance:
                raise ValueError("soft_gate_start_distance must be greater than soft_gate_full_distance.")
            if soft_gate_full_distance < 0:
                raise ValueError("soft_gate_full_distance must be non-negative.")
            start_scalar, full_scalar = float(soft_gate_start_distance), float(soft_gate_full_distance)
        self._soft_gate_start_np = np.full(self.num_pairs, start_scalar, dtype=np.float32)
        self._soft_gate_full_np = np.full(self.num_pairs, full_scalar, dtype=np.float32)
        if soft_gate_pair_mask is not None:
            self._soft_gate_enabled_np = np.array([1 if m else 0 for m in soft_gate_pair_mask], dtype=np.int32)
        else:
            enabled = 1 if soft_gate_start_distance is not None else 0
            self._soft_gate_enabled_np = np.full(self.num_pairs, enabled, dtype=np.int32)

        self.targets_wp: Optional[wp.array] = targets if isinstance(targets, wp.array) else None  # read directly
        self.weights_wp: Optional[wp.array] = None

    def set_robot(self, robot: Robot):
        self.robot = robot

    @property
    def active(self) -> bool:
        weight = list(self.weight.values()) if isinstance(self.weight, Mapping) else self.weight
        return bool(np.any(np.asarray(weight, dtype=np.float32))) and self.num_pairs > 0

    def resolve(self, var_values: VarValues, context: Mapping[Context, Any]):
        """Attach the robot and helper links/targets, reading Warp buffers in place; named weights pick the rows."""
        super().resolve(var_values, context)
        origins, tasks, rows, weight = self.origin_link_indices_arg, self.task_link_indices_arg, None, self.weight
        if isinstance(weight, Mapping):
            links, names = cast(Robot, self.robot).link_names, list(cast(Sequence[str], self.target_names))
            keys = list(weight)
            origins = [-1 if isinstance(k, str) else links.index(k[0]) for k in keys]
            tasks = [links.index(cast(Mapping, self.target_links)[k] if isinstance(k, str) else k[1]) for k in keys]
            rows = [
                names.index(k) if isinstance(k, str) else names.index(k[2]) * len(names) + names.index(k[3])
                for k in keys
            ]
            weight = list(weight.values())
        soft_gate_pair_mask = self.soft_gate_pair_mask
        if self.soft_gate_targets is not None and rows is not None:
            gated, count = set(cast(Sequence[int], self.soft_gate_targets)), len(cast(Sequence[str], self.target_names))
            soft_gate_pair_mask = [row // count in gated and row % count in gated for row in rows]
        FrameVectorTask.__init__(
            self,
            self.robot,
            origins,
            tasks,
            self.targets_arg,
            weight,
            rows if rows is not None else self.target_indices_arg,
            self.scale,
            self.huber_delta_arg,
            self.huber_on_norm_arg,
            self.direction_only,
            self.soft_gate_start_distance,
            self.soft_gate_full_distance,
            soft_gate_pair_mask,
            self.direction_metric,
            self.name,
            self.weight_mode,
        )
        self.init_buffers(var_values.get(self.var_key).q.device)

    @property
    def residual_dim(self) -> int:
        return self.num_pairs * self.channels

    def init_buffers(self, device: wp_device_type):
        self.device = device
        self.origin_indices_wp = wp.from_numpy(self._origin_indices_np, dtype=wp.int32, device=device)
        self.task_indices_wp = wp.from_numpy(self._task_indices_np, dtype=wp.int32, device=device)
        self.target_indices_wp = wp.from_numpy(self._target_indices_np, dtype=wp.int32, device=device)
        self.scale_wp = wp.from_numpy(self._scale_np, dtype=wp.float32, device=device)
        self.soft_gate_start_wp = wp.from_numpy(self._soft_gate_start_np, dtype=wp.float32, device=device)
        self.soft_gate_full_wp = wp.from_numpy(self._soft_gate_full_np, dtype=wp.float32, device=device)
        self.soft_gate_enabled_wp = wp.from_numpy(self._soft_gate_enabled_np, dtype=wp.int32, device=device)
        self.weights_wp = wp.from_numpy(self._weight_np, dtype=wp.float32, device=device)
        if self.targets_wp is None or self.targets_wp.device != wp.get_device(device):
            self.targets_wp = wp.from_numpy(self._targets_np, dtype=wp.float32, device=device)

    def set_targets(self, targets: Union[np.ndarray, wp.array], weight: Optional[np.ndarray] = None):
        """Update targets (and optionally weights) in place; shapes must match the built buffers."""
        if isinstance(targets, np.ndarray):
            targets_np = np.asarray(targets, dtype=np.float32)
            self._targets_np = targets_np[None] if targets_np.ndim == 2 else targets_np
            if self.targets_wp is not None:
                self.targets_wp.assign(self._targets_np)
        else:
            wp.copy(self.targets_wp, targets)
        if weight is not None:
            weight_np = np.asarray(weight, dtype=np.float32)
            self._weight_np = np.ascontiguousarray(np.atleast_2d(weight_np))
            if self.weights_wp is not None:
                self.weights_wp.assign(self._weight_np)

    def compute_weighted_residual(
        self,
        var_values: VarValues,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        var = var_values.get(self.var_key)
        if not var.is_fk_computed:
            self.robot.forward_kinematics(var)
        kernel_device = out_residual.device if out_residual is not None else var.q.device
        if self.device is None or self.device != kernel_device:
            self.init_buffers(kernel_device)
        if out_residual is None:
            out_residual = wp.empty((var.batch_size, self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        wp.launch(
            kernel=compute_frame_vector_residual_kernel,
            dim=(var.batch_size, self.num_pairs),
            inputs=[
                var.T_world_link,
                self.origin_indices_wp,
                self.task_indices_wp,
                self.target_indices_wp,
                self.targets_wp,
                self.weights_wp,
                self.scale_wp,
                self.soft_gate_start_wp,
                self.soft_gate_full_wp,
                self.soft_gate_enabled_wp,
                self.huber_delta,
                wp.int32(1 if self.use_huber else 0),
                wp.int32(1 if self.huber_on_norm else 0),
                wp.int32(self._direction_mode),
                wp.int32(self.channels),
                row_offset,
            ],
            outputs=[out_residual],
            device=kernel_device,
        )
        return out_residual

    def compute_weighted_jacobian_analytic(
        self,
        var_values: VarValues,
        out_jacobian: Optional[wp.array] = None,
        row_offset: int = 0,
        col_offset: int = 0,
    ) -> wp.array:
        var = var_values.get(self.var_key)
        if not var.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(var)
        kernel_device = out_jacobian.device if out_jacobian is not None else var.q.device
        if self.device is None or self.device != kernel_device:
            self.init_buffers(kernel_device)
        if out_jacobian is None:
            out_jacobian = wp.zeros(
                (var.batch_size, self.residual_dim, var.tangent_dim), dtype=wp.float32, device=kernel_device
            )
            row_offset = 0
        base_dim = 6 if var.has_floating_base else 0

        wp.launch(
            kernel=compute_frame_vector_jacobian_kernel,
            dim=(var.batch_size, self.num_pairs, self.robot.spec.num_actuated_joints),
            inputs=[
                var.S_world,
                var.T_world_link,
                self.origin_indices_wp,
                self.task_indices_wp,
                self.target_indices_wp,
                self.targets_wp,
                self.weights_wp,
                self.scale_wp,
                var.spec_tensors.link_ancestor_joints_mask,
                var.spec_tensors.joints_to_actuated_mapping,
                self.soft_gate_start_wp,
                self.soft_gate_full_wp,
                self.soft_gate_enabled_wp,
                self.huber_delta,
                wp.int32(1 if self.use_huber else 0),
                wp.int32(1 if self.huber_on_norm else 0),
                wp.int32(self._direction_mode),
                wp.int32(self.channels),
                base_dim,
                row_offset,
                col_offset,
            ],
            outputs=[out_jacobian],
            device=kernel_device,
        )
        if var.has_floating_base:
            wp.launch(
                kernel=compute_frame_vector_base_jacobian_kernel,
                dim=(var.batch_size, self.num_pairs, 6),
                inputs=[
                    var.T_world_link,
                    var.T_world_base,
                    self.origin_indices_wp,
                    self.task_indices_wp,
                    self.target_indices_wp,
                    self.targets_wp,
                    self.weights_wp,
                    self.scale_wp,
                    self.soft_gate_start_wp,
                    self.soft_gate_full_wp,
                    self.soft_gate_enabled_wp,
                    self.huber_delta,
                    wp.int32(1 if self.use_huber else 0),
                    wp.int32(1 if self.huber_on_norm else 0),
                    wp.int32(self._direction_mode),
                    wp.int32(self.channels),
                    row_offset,
                    col_offset,
                ],
                outputs=[out_jacobian],
                device=kernel_device,
            )
        return out_jacobian


# --- kernels ----------------------------------------------------------------
@wp.kernel
def compute_frame_vector_residual_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),
    origin_indices: wp.array1d(dtype=wp.int32),
    task_indices: wp.array1d(dtype=wp.int32),
    target_indices: wp.array1d(dtype=wp.int32),
    targets: wp.array3d(dtype=wp.float32),
    weights: wp.array2d(dtype=wp.float32),
    scales: wp.array1d(dtype=wp.float32),
    soft_gate_start: wp.array1d(dtype=wp.float32),
    soft_gate_full: wp.array1d(dtype=wp.float32),
    soft_gate_enabled: wp.array1d(dtype=wp.int32),
    huber_delta: float,
    use_huber: wp.int32,
    huber_on_norm: wp.int32,
    direction_mode: wp.int32,
    channels: wp.int32,
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),
):
    batch_idx, pair_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    target_batch_idx = batch_idx // (wp.max(1, T_world_link.shape[0] // targets.shape[0]))
    weight_batch_idx = batch_idx // (wp.max(1, T_world_link.shape[0] // weights.shape[0]))

    origin_link = origin_indices[pair_idx]
    origin_pos = wp.vec3(0.0, 0.0, 0.0)
    if origin_link >= 0:
        T_world_origin_link = T_world_link[batch_idx, origin_link]
        origin_pos = wp.vec3(T_world_origin_link[0], T_world_origin_link[1], T_world_origin_link[2])
    T_world_task_link = T_world_link[batch_idx, task_indices[pair_idx]]
    task_pos = wp.vec3(T_world_task_link[0], T_world_task_link[1], T_world_task_link[2])

    delta = (task_pos - origin_pos) * scales[pair_idx]
    target_idx = target_indices[pair_idx]
    target = wp.vec3(
        targets[target_batch_idx, target_idx, 0],
        targets[target_batch_idx, target_idx, 1],
        targets[target_batch_idx, target_idx, 2],
    )
    gate = _compute_frame_vector_soft_gate_func(
        target, soft_gate_start[pair_idx], soft_gate_full[pair_idx], soft_gate_enabled[pair_idx]
    )
    if direction_mode == 1:
        target = target / (wp.length(target) + 1e-6) * wp.length(delta)
    error = delta - target

    w = weights[weight_batch_idx, pair_idx] * gate
    base = row_offset + pair_idx * channels
    if direction_mode == 2:
        robot_dir = delta / (wp.length(delta) + 1e-6)
        target_dir = target / (wp.length(target) + 1e-6)
        out_residual[batch_idx, base] = w * (1.0 - wp.dot(robot_dir, target_dir))
    elif huber_on_norm == 1:
        dist = wp.length(error)
        residual = dist
        if dist > huber_delta:
            residual = wp.sqrt(wp.max(2.0 * huber_delta * dist - huber_delta * huber_delta, 0.0))
        out_residual[batch_idx, base] = w * residual
    else:
        for i in range(3):
            value = error[i]
            if use_huber == 1:
                value = _compute_huber_residual_signed_func(error[i], huber_delta)
            out_residual[batch_idx, base + i] = w * value


@wp.kernel
def compute_frame_vector_jacobian_kernel(
    S_world: wp.array3d(dtype=wp.float32),
    T_world_link: wp.array2d(dtype=wp_vec7),
    origin_indices: wp.array1d(dtype=wp.int32),
    task_indices: wp.array1d(dtype=wp.int32),
    target_indices: wp.array1d(dtype=wp.int32),
    targets: wp.array3d(dtype=wp.float32),
    weights: wp.array2d(dtype=wp.float32),
    scales: wp.array1d(dtype=wp.float32),
    link_ancestor_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    soft_gate_start: wp.array1d(dtype=wp.float32),
    soft_gate_full: wp.array1d(dtype=wp.float32),
    soft_gate_enabled: wp.array1d(dtype=wp.int32),
    huber_delta: float,
    use_huber: wp.int32,
    huber_on_norm: wp.int32,
    direction_mode: wp.int32,
    channels: wp.int32,
    base_dim: int,
    row_offset: int,
    col_offset: int,
    out_jacobian: wp.array3d(dtype=wp.float32),
):
    batch_idx, pair_idx, actuated_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    target_batch_idx = batch_idx // (wp.max(1, T_world_link.shape[0] // targets.shape[0]))
    weight_batch_idx = batch_idx // (wp.max(1, T_world_link.shape[0] // weights.shape[0]))

    origin_link = origin_indices[pair_idx]
    task_link = task_indices[pair_idx]
    origin_pos = wp.vec3(0.0, 0.0, 0.0)
    if origin_link >= 0:
        T_world_origin_link = T_world_link[batch_idx, origin_link]
        origin_pos = wp.vec3(T_world_origin_link[0], T_world_origin_link[1], T_world_origin_link[2])
    T_world_task_link = T_world_link[batch_idx, task_link]
    task_pos = wp.vec3(T_world_task_link[0], T_world_task_link[1], T_world_task_link[2])

    scale = scales[pair_idx]
    delta = (task_pos - origin_pos) * scale
    target_idx = target_indices[pair_idx]
    target = wp.vec3(
        targets[target_batch_idx, target_idx, 0],
        targets[target_batch_idx, target_idx, 1],
        targets[target_batch_idx, target_idx, 2],
    )
    gate = _compute_frame_vector_soft_gate_func(
        target, soft_gate_start[pair_idx], soft_gate_full[pair_idx], soft_gate_enabled[pair_idx]
    )
    target_dir = target / (wp.length(target) + 1e-6)
    if direction_mode == 1:
        target = target_dir * wp.length(delta)
    error = delta - target

    d_pair_dq = wp.vec3(0.0, 0.0, 0.0)
    for joint_idx in range(joints_to_actuated.shape[0]):
        mapping = joints_to_actuated[joint_idx, actuated_idx]
        if mapping == 0.0:
            continue
        affects_origin = False
        if origin_link >= 0:
            affects_origin = link_ancestor_mask[origin_link, joint_idx]
        affects_task = link_ancestor_mask[task_link, joint_idx]
        if not affects_origin and not affects_task:
            continue
        v = wp.vec3(
            S_world[batch_idx, 0, joint_idx], S_world[batch_idx, 1, joint_idx], S_world[batch_idx, 2, joint_idx]
        )
        omega = wp.vec3(
            S_world[batch_idx, 3, joint_idx], S_world[batch_idx, 4, joint_idx], S_world[batch_idx, 5, joint_idx]
        )
        dp_origin = wp.vec3(0.0, 0.0, 0.0)
        dp_task = wp.vec3(0.0, 0.0, 0.0)
        if affects_origin:
            dp_origin = mapping * (v + wp.cross(omega, origin_pos))
        if affects_task:
            dp_task = mapping * (v + wp.cross(omega, task_pos))
        d_pair_dq += dp_task - dp_origin

    d_error_dq = d_pair_dq * scale
    if direction_mode == 1:
        delta_dir = delta / (wp.length(delta) + 1e-6)
        d_error_dq = d_error_dq - target_dir * wp.dot(delta_dir, d_error_dq)

    w = weights[weight_batch_idx, pair_idx] * gate
    base = row_offset + pair_idx * channels
    col = col_offset + base_dim + actuated_idx
    if direction_mode == 2:
        length = wp.length(delta) + 1e-6
        robot_dir = delta / length
        gradient = -(target_dir - wp.dot(robot_dir, target_dir) * wp.normalize(delta)) / length
        out_jacobian[batch_idx, base, col] = w * wp.dot(gradient, d_error_dq)
    elif huber_on_norm == 1:
        dist = wp.length(error)
        value = float(0.0)
        if dist >= 1e-8:
            value = w * _compute_huber_gradient_scale_func(dist, huber_delta) * wp.dot(error / dist, d_error_dq)
        out_jacobian[batch_idx, base, col] = value
    else:
        for i in range(3):
            grad = float(1.0)
            if use_huber == 1:
                grad = _compute_huber_gradient_scale_func(error[i], huber_delta)
            out_jacobian[batch_idx, base + i, col] = w * grad * d_error_dq[i]


@wp.kernel
def compute_frame_vector_base_jacobian_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),
    T_world_base: wp.array1d(dtype=wp_vec7),
    origin_indices: wp.array1d(dtype=wp.int32),
    task_indices: wp.array1d(dtype=wp.int32),
    target_indices: wp.array1d(dtype=wp.int32),
    targets: wp.array3d(dtype=wp.float32),
    weights: wp.array2d(dtype=wp.float32),
    scales: wp.array1d(dtype=wp.float32),
    soft_gate_start: wp.array1d(dtype=wp.float32),
    soft_gate_full: wp.array1d(dtype=wp.float32),
    soft_gate_enabled: wp.array1d(dtype=wp.int32),
    huber_delta: float,
    use_huber: wp.int32,
    huber_on_norm: wp.int32,
    direction_mode: wp.int32,
    channels: wp.int32,
    row_offset: int,
    col_offset: int,
    out_jacobian: wp.array3d(dtype=wp.float32),
):
    batch_idx, pair_idx, base_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    target_batch_idx = batch_idx // (wp.max(1, T_world_link.shape[0] // targets.shape[0]))
    weight_batch_idx = batch_idx // (wp.max(1, T_world_link.shape[0] // weights.shape[0]))

    origin_link = origin_indices[pair_idx]
    origin_pos = wp.vec3(0.0, 0.0, 0.0)
    if origin_link >= 0:
        T_world_origin_link = T_world_link[batch_idx, origin_link]
        origin_pos = wp.vec3(T_world_origin_link[0], T_world_origin_link[1], T_world_origin_link[2])
    T_world_task_link = T_world_link[batch_idx, task_indices[pair_idx]]
    task_pos = wp.vec3(T_world_task_link[0], T_world_task_link[1], T_world_task_link[2])

    scale = scales[pair_idx]
    delta = (task_pos - origin_pos) * scale
    target_idx = target_indices[pair_idx]
    target = wp.vec3(
        targets[target_batch_idx, target_idx, 0],
        targets[target_batch_idx, target_idx, 1],
        targets[target_batch_idx, target_idx, 2],
    )
    gate = _compute_frame_vector_soft_gate_func(
        target, soft_gate_start[pair_idx], soft_gate_full[pair_idx], soft_gate_enabled[pair_idx]
    )
    target_dir = target / (wp.length(target) + 1e-6)
    if direction_mode == 1:
        target = target_dir * wp.length(delta)
    error = delta - target

    unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    unit_vec[base_idx] = 1.0
    base_twist = se3_adjoint_func(T_world_base[batch_idx]) * unit_vec
    v = wp.vec3(base_twist[0], base_twist[1], base_twist[2])
    omega = wp.vec3(base_twist[3], base_twist[4], base_twist[5])

    d_pair_dq = v + wp.cross(omega, task_pos)
    if origin_link >= 0:
        d_pair_dq = d_pair_dq - (v + wp.cross(omega, origin_pos))
    d_error_dq = d_pair_dq * scale
    if direction_mode == 1:
        delta_dir = delta / (wp.length(delta) + 1e-6)
        d_error_dq = d_error_dq - target_dir * wp.dot(delta_dir, d_error_dq)

    w = weights[weight_batch_idx, pair_idx] * gate
    base = row_offset + pair_idx * channels
    col = col_offset + base_idx
    if direction_mode == 2:
        length = wp.length(delta) + 1e-6
        robot_dir = delta / length
        gradient = -(target_dir - wp.dot(robot_dir, target_dir) * wp.normalize(delta)) / length
        out_jacobian[batch_idx, base, col] = w * wp.dot(gradient, d_error_dq)
    elif huber_on_norm == 1:
        dist = wp.length(error)
        value = float(0.0)
        if dist >= 1e-8:
            value = w * _compute_huber_gradient_scale_func(dist, huber_delta) * wp.dot(error / dist, d_error_dq)
        out_jacobian[batch_idx, base, col] = value
    else:
        for i in range(3):
            grad = float(1.0)
            if use_huber == 1:
                grad = _compute_huber_gradient_scale_func(error[i], huber_delta)
            out_jacobian[batch_idx, base + i, col] = w * grad * d_error_dq[i]
