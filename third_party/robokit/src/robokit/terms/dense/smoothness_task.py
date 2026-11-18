# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
from typing import Any, Mapping, Optional, Sequence, Union, cast

import numpy as np
import warp as wp

from robokit.lie.se3_kernels import (
    se3_adjoint_func,
    se3_compose_func,
    se3_inverse_func,
    se3_jlog_func,
    se3_log_map_func,
)
from robokit.opt.var_values import VarValues
from robokit.robo.robot import Robot, RobotState
from robokit.terms.task import Context, GradientTask, ResidualTask
from robokit.utils.warp_utils import wp_device_type, wp_vec6, wp_vec7


# --- task -------------------------------------------------------------------
class SmoothnessTask(ResidualTask, GradientTask):
    """Penalize change from a previous joint and optional base state.

    Example:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.opt.var_values import VarKey
        >>> from robokit.robo.robot import Robot
        >>> robot = Robot.load(load_robot_description("panda_description"))
        >>> state_prev = robot.state(q=robot.spec.zero_q)
        >>> state_curr = robot.state(q=robot.spec.zero_q + 0.1)
        >>> smooth_task = SmoothnessTask(robot=robot, weight=0.5)
        >>> values = VarValues({VarKey.ROBOT: state_curr})
        >>> cost = wp.zeros(1, device=state_curr.q.device)
        >>> gradient = wp.zeros((1, state_curr.tangent_dim), device=state_curr.q.device)
        >>> smooth_task.compute_weighted_cost_and_gradient(values, state_prev, out_cost=cost, out_gradient=gradient)
        >>> np.allclose(gradient.numpy(), 0.025)
        True
    """

    def __init__(
        self,
        robot: Optional[Robot] = None,
        prev_var: Optional[Union[RobotState, Context]] = None,
        weight: Union[float, Sequence[float]] = 1.0,
        base_weight: Optional[Union[float, Sequence[float]]] = None,
        name: str = "",
        warmup_frames: int = 0,  # with `frame`, ramp the weights in over this many frames of a stream
        frame: Optional[Union[wp.array, Context]] = None,  # a helper's `(1,)` count of frames solved
    ):
        self.name = name
        self.warmup_frames, self.frame = warmup_frames, frame
        self.weight = weight
        # a placeholder until `resolve`
        self.prev_var = cast(Optional[RobotState], prev_var.clone() if isinstance(prev_var, RobotState) else prev_var)
        self.base_weight = base_weight
        self._include_base_residual = base_weight is not None or (
            isinstance(prev_var, RobotState) and prev_var.has_floating_base
        )
        self.robot: Optional[Robot] = None
        self.device = None
        self.residual_weight = None
        if robot is not None:
            self.set_robot(robot)

    @property
    def active(self) -> bool:
        # once built, base residuals count only on a floating base
        base = self.base_weight is not None and (self.robot is None or self._include_base_residual)
        return bool(np.any(self.weight) or (base and np.any(np.asarray(self.base_weight))))

    def resolve(self, var_values: VarValues, context: Mapping[Context, Any]):
        """Build from the robot; a previous state from the helper is its buffer, read directly."""
        super().resolve(var_values, context)
        # base residuals follow a floating base; a fixed-base robot ignores base_weight
        self._include_base_residual = isinstance(self.prev_var, RobotState) and self.prev_var.has_floating_base
        state = cast(RobotState, var_values.get(self.var_key))
        self.set_robot(state.robot)
        self.init_buffers(state.q.device)

    def set_robot(self, robot: Robot):
        self.robot = robot
        num_joints = robot.spec.num_actuated_joints

        residual_weight = np.ones(num_joints, dtype=np.float32)
        residual_weight[:] = self.weight

        self._joint_weight_np = residual_weight.copy()
        if self._include_base_residual:
            self._residual_dim = num_joints + 6
            residual_weight_full = np.ones(self._residual_dim, dtype=np.float32)
            residual_weight_full[:num_joints] = self._joint_weight_np
            if self.base_weight is not None:
                residual_weight_full[num_joints:] = self.base_weight
            else:
                residual_weight_full[num_joints:] = self._joint_weight_np[0] if len(self._joint_weight_np) > 0 else 1.0
            self._residual_weight_np = residual_weight_full
        else:
            self._residual_dim = num_joints
            self._residual_weight_np = residual_weight

    def set_weight(self, weight: Union[float, Sequence[float]]):
        """Update joint-block residual weights in-place. Base-block weights are preserved."""
        num_joints = self.robot.spec.num_actuated_joints
        self._joint_weight_np[:] = weight
        self._residual_weight_np[:num_joints] = weight
        self.weight = weight
        if self.residual_weight is not None:
            wp.copy(self.residual_weight, wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device="cpu"))

    def set_prev_state(self, prev_state: RobotState):
        if self.prev_var is None:
            self.prev_var = prev_state.clone()
            return
        if self.prev_var.q.shape != prev_state.q.shape:
            self.prev_var = prev_state.clone()
            return
        wp.copy(self.prev_var.q, prev_state.q)
        if self.prev_var.has_floating_base and prev_state.has_floating_base:
            wp.copy(self.prev_var.T_world_base, prev_state.T_world_base)

    def init_buffers(self, device: wp_device_type):
        self.device = device
        self.residual_weight = wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device=device)
        # a helper's frame counter ramps the weights in; without one the kernels read a zero count and no ramp
        ramp = isinstance(self.frame, wp.array) and self.warmup_frames > 0
        self._frame_wp = self.frame if ramp else wp.zeros(1, dtype=wp.int32, device=device)
        self._ramp_frames = float(self.warmup_frames) if ramp else 0.0

    @property
    def residual_dim(self) -> int:
        return self._residual_dim

    def compute_weighted_cost_and_gradient(
        self,
        var_values: VarValues,
        prev_var: Optional[RobotState] = None,
        *,
        out_cost: wp.array,
        out_gradient: Optional[wp.array] = None,
    ):
        """Accumulate half the squared residual norm and its direct gradient."""
        var = var_values.get(self.var_key)
        if prev_var is None:
            prev_var = self.prev_var
        if prev_var is None:
            return
        if self._include_base_residual and (not var.has_floating_base or not prev_var.has_floating_base):
            raise ValueError("SmoothnessTask is configured with base residuals but var/prev_var has no floating base.")
        if self.device is None:
            self.init_buffers(var.q.device)
        wp.launch(
            compute_smoothness_task_cost_and_gradient_kernel,
            dim=var.batch_size,
            inputs=[
                var.q,
                prev_var.q,
                var.T_world_base,
                prev_var.T_world_base,
                self.residual_weight,
                self._frame_wp,
                self._ramp_frames,
                var.has_floating_base,
                self._include_base_residual,
                var_values.tangent_offset(self.var_key),
            ],
            outputs=[out_cost, out_gradient],
            device=out_cost.device,
        )

    def compute_weighted_residual(
        self,
        var_values: VarValues,
        prev_var: Optional[RobotState] = None,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        var = var_values.get(self.var_key)
        if prev_var is None:
            prev_var = self.prev_var

        batch_size = var.batch_size
        if prev_var is None:
            if out_residual is None:
                return wp.zeros((batch_size, self.residual_dim), dtype=wp.float32, device=var.q.device)
            else:
                return out_residual
        if self._include_base_residual and (not var.has_floating_base or not prev_var.has_floating_base):
            raise ValueError("SmoothnessTask is configured with base residuals but var/prev_var has no floating base.")

        if self.device is None:
            self.init_buffers(var.q.device)

        kernel_device = out_residual.device if out_residual is not None else self.device
        if out_residual is None:
            out_residual = wp.empty((batch_size, self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        if self._include_base_residual:
            wp.launch(
                kernel=compute_smoothness_task_weighted_residual_with_base_kernel,
                dim=batch_size,
                inputs=[
                    var.q,
                    prev_var.q,
                    var.T_world_base,
                    prev_var.T_world_base,
                    self.residual_weight,
                    self._frame_wp,
                    self._ramp_frames,
                    row_offset,
                ],
                outputs=[out_residual],
                device=kernel_device,
            )
        else:
            wp.launch(
                kernel=compute_smoothness_task_weighted_residual_kernel,
                dim=(batch_size, self.robot.spec.num_actuated_joints),
                inputs=[
                    var.q,
                    prev_var.q,
                    self.residual_weight,
                    self._frame_wp,
                    self._ramp_frames,
                    row_offset,
                ],
                outputs=[out_residual],
                device=kernel_device,
            )

        return out_residual

    def compute_weighted_jacobian_analytic(
        self,
        var_values: VarValues,
        prev_var: Optional[RobotState] = None,
        out_jacobian: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        var = var_values.get(self.var_key)
        col_offset = var_values.tangent_offset(self.var_key)
        if prev_var is None:
            prev_var = self.prev_var

        if self.device is None:
            self.init_buffers(var.q.device)

        kernel_device = out_jacobian.device if out_jacobian is not None else self.device
        total_dofs = var.tangent_dim

        batch_size = var.batch_size
        if out_jacobian is None:
            out_jacobian = wp.zeros((batch_size, self.residual_dim, total_dofs), dtype=wp.float32, device=kernel_device)
            row_offset = 0
            col_offset = 0

        if prev_var is None:
            return out_jacobian
        if self._include_base_residual and (not var.has_floating_base or not prev_var.has_floating_base):
            raise ValueError("SmoothnessTask is configured with base residuals but var/prev_var has no floating base.")

        if self._include_base_residual:
            wp.launch(
                kernel=compute_smoothness_task_weighted_jacobian_with_base_kernel,
                dim=(batch_size, total_dofs),
                inputs=[
                    var.T_world_base,
                    prev_var.T_world_base,
                    self.residual_weight,
                    self._frame_wp,
                    self._ramp_frames,
                    row_offset,
                    col_offset,
                    self.robot.spec.num_actuated_joints,
                ],
                outputs=[out_jacobian],
                device=kernel_device,
            )
        else:
            wp.launch(
                kernel=compute_smoothness_task_weighted_jacobian_kernel,
                dim=(batch_size, self.robot.spec.num_actuated_joints),
                inputs=[
                    var.has_floating_base,
                    self.residual_weight,
                    self._frame_wp,
                    self._ramp_frames,
                    row_offset,
                    col_offset,
                ],
                outputs=[out_jacobian],
                device=kernel_device,
            )

        return out_jacobian


# --- kernels ----------------------------------------------------------------
@wp.kernel
def compute_smoothness_task_cost_and_gradient_kernel(
    q_curr: wp.array2d(dtype=wp.float32),
    q_prev: wp.array2d(dtype=wp.float32),
    T_world_base_curr: wp.array1d(dtype=wp_vec7),
    T_world_base_prev: wp.array1d(dtype=wp_vec7),
    residual_weight: wp.array1d(dtype=wp.float32),
    frame: wp.array1d(dtype=wp.int32),
    frames: float,
    has_floating_base: wp.bool,
    include_base: wp.bool,
    col_offset: int,
    out_cost: wp.array1d(dtype=wp.float32),
    out_gradient: wp.array2d(dtype=wp.float32),
):
    batch_idx = wp.tid()
    ramp_scale = wp.min(float(frame[0]) / frames, 1.0) if frames > 0.0 else 1.0
    num_joints = q_curr.shape[1]
    prev_idx = batch_idx // wp.max(1, q_curr.shape[0] // q_prev.shape[0])  # pyright: ignore[reportOperatorIssue]
    base_offset = 6 if has_floating_base else 0
    cost = wp.float32(0.0)
    for joint_idx in range(num_joints):
        residual = (ramp_scale * residual_weight[joint_idx]) * (
            q_curr[batch_idx, joint_idx] - q_prev[prev_idx, joint_idx]
        )
        cost += wp.float32(0.5) * residual * residual
        if out_gradient:
            wp.atomic_add(
                out_gradient,
                batch_idx,
                col_offset + base_offset + joint_idx,
                (ramp_scale * residual_weight[joint_idx]) * residual,
            )
    if include_base:
        T_prev = T_world_base_prev[prev_idx]
        T_error = se3_compose_func(T_world_base_curr[batch_idx], se3_inverse_func(T_prev))
        r_base = se3_log_map_func(T_error, wp.float32(1e-4))
        weighted_base = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for i in range(6):
            residual = (ramp_scale * residual_weight[num_joints + i]) * r_base[i]
            cost += wp.float32(0.5) * residual * residual
            weighted_base[i] = (ramp_scale * residual_weight[num_joints + i]) * residual
        if out_gradient:
            jlog = se3_jlog_func(T_error, wp.float32(1e-4))
            base_gradient = wp.transpose(se3_adjoint_func(T_prev)) * (wp.transpose(jlog) * weighted_base)
            for i in range(6):
                wp.atomic_add(out_gradient, batch_idx, col_offset + i, base_gradient[i])
    wp.atomic_add(out_cost, batch_idx, cost)


@wp.kernel
def compute_smoothness_task_weighted_residual_kernel(
    q_curr: wp.array2d(dtype=wp.float32),  # [batch, num_joints]
    q_prev: wp.array2d(dtype=wp.float32),  # [prev_batch, num_joints]
    residual_weight: wp.array1d(dtype=wp.float32),  # [num_joints]
    frame: wp.array1d(dtype=wp.int32),
    frames: float,
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),  # [batch, total_residual_dim]
):
    batch_idx, joint_idx = wp.tid()  # type: ignore
    ramp_scale = wp.min(float(frame[0]) / frames, 1.0) if frames > 0.0 else 1.0
    n_repeat = wp.max(1, q_curr.shape[0] // q_prev.shape[0])
    prev_idx = batch_idx // n_repeat
    diff = q_curr[batch_idx, joint_idx] - q_prev[prev_idx, joint_idx]
    out_residual[batch_idx, row_offset + joint_idx] = (ramp_scale * residual_weight[joint_idx]) * diff


@wp.kernel
def compute_smoothness_task_weighted_residual_with_base_kernel(
    q_curr: wp.array2d(dtype=wp.float32),  # [batch, num_joints]
    q_prev: wp.array2d(dtype=wp.float32),  # [batch, num_joints]
    T_world_base_curr: wp.array1d(dtype=wp_vec7),  # [batch]
    T_world_base_prev: wp.array1d(dtype=wp_vec7),  # [prev_batch]
    residual_weight: wp.array1d(dtype=wp.float32),  # [num_joints + 6]
    frame: wp.array1d(dtype=wp.int32),
    frames: float,
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),  # [batch, total_residual_dim]
):
    batch_idx = wp.tid()  # type: ignore
    ramp_scale = wp.min(float(frame[0]) / frames, 1.0) if frames > 0.0 else 1.0
    num_joints = q_curr.shape[1]
    n_repeat = wp.max(1, q_curr.shape[0] // q_prev.shape[0])
    prev_idx = batch_idx // n_repeat  # pyright: ignore[reportOperatorIssue]

    # joint residuals
    for joint_idx in range(num_joints):
        diff = q_curr[batch_idx, joint_idx] - q_prev[prev_idx, joint_idx]
        out_residual[batch_idx, row_offset + joint_idx] = (ramp_scale * residual_weight[joint_idx]) * diff

    # base residuals
    T_prev_inv = se3_inverse_func(T_world_base_prev[prev_idx])
    T_error = se3_compose_func(T_world_base_curr[batch_idx], T_prev_inv)
    r_base = se3_log_map_func(T_error, wp.float32(1e-4))

    out_residual[batch_idx, row_offset + num_joints + 0] = (ramp_scale * residual_weight[num_joints + 0]) * r_base[0]
    out_residual[batch_idx, row_offset + num_joints + 1] = (ramp_scale * residual_weight[num_joints + 1]) * r_base[1]
    out_residual[batch_idx, row_offset + num_joints + 2] = (ramp_scale * residual_weight[num_joints + 2]) * r_base[2]
    out_residual[batch_idx, row_offset + num_joints + 3] = (ramp_scale * residual_weight[num_joints + 3]) * r_base[3]
    out_residual[batch_idx, row_offset + num_joints + 4] = (ramp_scale * residual_weight[num_joints + 4]) * r_base[4]
    out_residual[batch_idx, row_offset + num_joints + 5] = (ramp_scale * residual_weight[num_joints + 5]) * r_base[5]


@wp.kernel
def compute_smoothness_task_weighted_jacobian_kernel(
    has_floating_base: wp.bool,
    residual_weight: wp.array1d(dtype=wp.float32),  # [num_joints]
    frame: wp.array1d(dtype=wp.int32),
    frames: float,
    row_offset: int,
    col_offset: int,
    out_jacobian: wp.array3d(dtype=wp.float32),  # [batch, total_residual_dim, total_dofs]
):
    batch_idx, joint_idx = wp.tid()  # type: ignore
    ramp_scale = wp.min(float(frame[0]) / frames, 1.0) if frames > 0.0 else 1.0

    base_col_offset = 6 if has_floating_base else 0

    # joint identity block
    out_jacobian[batch_idx, row_offset + joint_idx, col_offset + base_col_offset + joint_idx] = (
        ramp_scale * residual_weight[joint_idx]
    )


@wp.kernel
def compute_smoothness_task_weighted_jacobian_with_base_kernel(
    T_world_base_curr: wp.array1d(dtype=wp_vec7),  # [batch]
    T_world_base_prev: wp.array1d(dtype=wp_vec7),  # [prev_batch]
    residual_weight: wp.array1d(dtype=wp.float32),  # [num_joints + 6]
    frame: wp.array1d(dtype=wp.int32),
    frames: float,
    row_offset: int,
    col_offset: int,
    num_joints: int,
    out_jacobian: wp.array3d(dtype=wp.float32),  # [batch, total_residual_dim, total_dofs]
):
    batch_idx, col_idx = wp.tid()  # type: ignore
    ramp_scale = wp.min(float(frame[0]) / frames, 1.0) if frames > 0.0 else 1.0
    n_repeat = wp.max(1, T_world_base_curr.shape[0] // T_world_base_prev.shape[0])
    prev_idx = batch_idx // n_repeat

    # joint identity block starts after the base columns
    if col_idx >= 6 and col_idx < (6 + num_joints):
        joint_idx = col_idx - 6
        out_jacobian[batch_idx, row_offset + joint_idx, col_offset + col_idx] = ramp_scale * residual_weight[joint_idx]

    # base block is Jlog @ Ad
    if col_idx < 6:
        T_prev_inv = se3_inverse_func(T_world_base_prev[prev_idx])
        T_error = se3_compose_func(T_world_base_curr[batch_idx], T_prev_inv)

        # compute Jlog @ Ad column by column
        jlog = se3_jlog_func(T_error, wp.float32(1e-4))

        # adjoint column for the selected basis vector
        unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        unit_vec[col_idx] = 1.0
        ad_col = se3_adjoint_func(T_world_base_prev[prev_idx]) * unit_vec

        # apply Jlog
        result_col = wp.mul(jlog, ad_col)

        out_jacobian[batch_idx, row_offset + num_joints + 0, col_offset + col_idx] = (
            ramp_scale * residual_weight[num_joints + 0]
        ) * result_col[0]
        out_jacobian[batch_idx, row_offset + num_joints + 1, col_offset + col_idx] = (
            ramp_scale * residual_weight[num_joints + 1]
        ) * result_col[1]
        out_jacobian[batch_idx, row_offset + num_joints + 2, col_offset + col_idx] = (
            ramp_scale * residual_weight[num_joints + 2]
        ) * result_col[2]
        out_jacobian[batch_idx, row_offset + num_joints + 3, col_offset + col_idx] = (
            ramp_scale * residual_weight[num_joints + 3]
        ) * result_col[3]
        out_jacobian[batch_idx, row_offset + num_joints + 4, col_offset + col_idx] = (
            ramp_scale * residual_weight[num_joints + 4]
        ) * result_col[4]
        out_jacobian[batch_idx, row_offset + num_joints + 5, col_offset + col_idx] = (
            ramp_scale * residual_weight[num_joints + 5]
        ) * result_col[5]
