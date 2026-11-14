# pyright: reportArgumentType=false
from typing import Any, Mapping, Optional, Sequence, Union, cast

import numpy as np
import warp as wp

from robokit.opt.var_values import VarValues
from robokit.robo.robot import Robot, RobotState
from robokit.terms.task import Context, ResidualTask
from robokit.utils.warp_utils import wp_device_type


# --- task -------------------------------------------------------------------
class VelocityLimitTask(ResidualTask):
    """Penalize joint velocity limit violations relative to a previous state.

    Example:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.robo.robot import Robot
        >>> robot = Robot.load(load_robot_description("panda_description"))
        >>> state_prev = robot.state(q=robot.spec.zero_q)
        >>> state_curr = robot.state(q=robot.spec.zero_q + 0.5)
        >>> vel_task = VelocityLimitTask(robot=robot, dt=0.1, weight=1.0)
        >>> # Residual should be positive if velocity exceeds limits
    """

    def __init__(
        self,
        robot: Optional[Robot] = None,
        dt: Union[float, Context] = 1.0,
        prev_state_var: Optional[Union[RobotState, Context]] = None,
        velocity_limits: Optional[Union[np.ndarray, Context]] = None,
        weight: Optional[Union[float, Sequence[float]]] = None,
        name: str = "",
    ):
        self.name = name
        assert isinstance(dt, Context) or dt > 0, "dt must be positive"
        self.dt = dt
        prev = prev_state_var.clone() if isinstance(prev_state_var, RobotState) else prev_state_var
        self.prev_state_var = cast(Optional[RobotState], prev)  # a placeholder until `resolve`
        self._velocity_limits_arg = velocity_limits
        self.weight = weight
        self.robot: Optional[Robot] = None
        self.device = None
        self.velocity_limits = None
        self.residual_weight = None
        if robot is not None:
            self.set_robot(robot)

    def resolve(self, var_values: VarValues, context: Mapping[Context, Any]):
        """Build from the robot; a previous state from the helper is its buffer, read directly."""
        super().resolve(var_values, context)
        self.set_robot(cast(RobotState, var_values.get(self.var_key)).robot)

    def set_robot(self, robot: Robot):
        self.robot = robot
        velocity_limits_np = (
            self._velocity_limits_arg
            if self._velocity_limits_arg is not None
            else robot.spec.actuated_joint_velocity_limits
        )
        self._velocity_limits_np = velocity_limits_np.astype(np.float32)

        num_joints = robot.spec.num_actuated_joints
        residual_weight = np.ones(num_joints, dtype=np.float32)
        if self.weight is not None:
            if isinstance(self.weight, (float, int)):
                residual_weight[:] = self.weight
            else:
                residual_weight[:] = self.weight

        self._residual_weight_np = residual_weight

    def set_weight(self, weight: Union[float, Sequence[float]]):
        """Update residual weights in-place. Accepts a scalar or per-joint sequence."""
        self._residual_weight_np[:] = weight
        self.weight = weight
        if self.residual_weight is not None:
            wp.copy(self.residual_weight, wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device="cpu"))

    def set_prev_state(self, prev_state: RobotState):
        if self.prev_state_var is None:
            self.prev_state_var = prev_state.clone()
            return
        if self.prev_state_var.q.shape != prev_state.q.shape:
            self.prev_state_var = prev_state.clone()
            return
        wp.copy(self.prev_state_var.q, prev_state.q)

    def init_buffers(self, device: wp_device_type):
        self.device = device
        self.velocity_limits = wp.from_numpy(self._velocity_limits_np, dtype=wp.float32, device=device)
        self.residual_weight = wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device=device)

    @property
    def residual_dim(self) -> int:
        return self.robot.spec.num_actuated_joints

    def compute_weighted_residual(
        self,
        var_values: VarValues,
        prev_state_var: Optional[Union[RobotState, Context]] = None,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        var = var_values.get(self.var_key)
        if prev_state_var is None:
            prev_state_var = self.prev_state_var

        if self.device is None:
            self.init_buffers(var.q.device)

        batch_size = var.batch_size
        kernel_device = out_residual.device if out_residual is not None else self.device
        if out_residual is None:
            out_residual = wp.empty((batch_size, self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        if prev_state_var is None:
            return out_residual

        wp.launch(
            kernel=compute_velocity_limit_weighted_residual_kernel,
            dim=(batch_size, self.robot.spec.num_actuated_joints),
            inputs=[
                var.q,
                prev_state_var.q,
                self.velocity_limits,
                self.residual_weight,
                self.dt,
                row_offset,
            ],
            outputs=[out_residual],
            device=kernel_device,
        )

        return out_residual

    def compute_weighted_jacobian_analytic(
        self,
        var_values: VarValues,
        prev_state_var: Optional[Union[RobotState, Context]] = None,
        out_jacobian: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        var = var_values.get(self.var_key)
        assert var_values.tangent_offset(self.var_key) == 0
        if prev_state_var is None:
            prev_state_var = self.prev_state_var

        if self.device is None:
            self.init_buffers(var.q.device)

        kernel_device = out_jacobian.device if out_jacobian is not None else self.device
        total_dofs = var.tangent_dim

        batch_size = var.batch_size
        if out_jacobian is None:
            out_jacobian = wp.zeros((batch_size, self.residual_dim, total_dofs), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        if prev_state_var is None:
            return out_jacobian

        base_col_offset = 6 if var.has_floating_base else 0

        wp.launch(
            kernel=compute_velocity_limit_weighted_jacobian_kernel,
            dim=(batch_size, self.robot.spec.num_actuated_joints),
            inputs=[
                var.q,
                prev_state_var.q,
                self.velocity_limits,
                self.residual_weight,
                self.dt,
                row_offset,
                base_col_offset,
            ],
            outputs=[out_jacobian],
            device=kernel_device,
        )

        return out_jacobian


# --- kernels ----------------------------------------------------------------
@wp.kernel
def compute_velocity_limit_weighted_residual_kernel(
    q_curr: wp.array2d(dtype=wp.float32),  # [batch, num_joints]
    q_prev: wp.array2d(dtype=wp.float32),  # [prev_batch, num_joints]
    velocity_limits: wp.array1d(dtype=wp.float32),  # [num_joints]
    residual_weight: wp.array1d(dtype=wp.float32),  # [num_joints]
    dt: float,
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),  # [batch, total_residual_dim]
):
    batch_idx, joint_idx = wp.tid()  # type: ignore

    n_repeat = wp.max(1, q_curr.shape[0] // q_prev.shape[0])
    prev_idx = batch_idx // n_repeat
    q_dot = (q_curr[batch_idx, joint_idx] - q_prev[prev_idx, joint_idx]) / dt
    abs_q_dot = wp.abs(q_dot)
    v_limit = velocity_limits[joint_idx]

    violation = wp.max(0.0, abs_q_dot - v_limit)
    out_residual[batch_idx, row_offset + joint_idx] = residual_weight[joint_idx] * violation


@wp.kernel
def compute_velocity_limit_weighted_jacobian_kernel(
    q_curr: wp.array2d(dtype=wp.float32),  # [batch, num_joints]
    q_prev: wp.array2d(dtype=wp.float32),  # [prev_batch, num_joints]
    velocity_limits: wp.array1d(dtype=wp.float32),  # [num_joints]
    residual_weight: wp.array1d(dtype=wp.float32),  # [num_joints]
    dt: float,
    row_offset: int,
    base_col_offset: int,
    out_jacobian: wp.array3d(dtype=wp.float32),  # [batch, total_residual_dim, total_dofs]
):
    batch_idx, joint_idx = wp.tid()  # type: ignore

    n_repeat = wp.max(1, q_curr.shape[0] // q_prev.shape[0])
    prev_idx = batch_idx // n_repeat
    q_dot = (q_curr[batch_idx, joint_idx] - q_prev[prev_idx, joint_idx]) / dt
    abs_q_dot = wp.abs(q_dot)
    v_limit = velocity_limits[joint_idx]

    # write only active limit columns
    if abs_q_dot > v_limit:
        sign = 1.0 if q_dot > 0.0 else -1.0
        J_value = residual_weight[joint_idx] * sign / dt
        out_jacobian[batch_idx, row_offset + joint_idx, base_col_offset + joint_idx] = J_value
