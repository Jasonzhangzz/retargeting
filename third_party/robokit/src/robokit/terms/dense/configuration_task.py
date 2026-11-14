# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
from enum import Enum
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple, Union, cast

import numpy as np
import warp as wp

from robokit.lie.se3 import SE3Var, se3_identity
from robokit.lie.se3_kernels import (
    se3_adjoint_func,
    se3_compose_func,
    se3_inverse_func,
    se3_jlog_func,
    se3_log_map_func,
)
from robokit.opt.var_values import Var, VarKey, VarValues, VectorVar
from robokit.robo.robot import Robot
from robokit.robo.robot_state import RobotState
from robokit.terms.task import Context, ResidualTask
from robokit.utils.warp_utils import wp_device_type, wp_vec6, wp_vec7


# --- task -------------------------------------------------------------------
class ConfigurationTask(ResidualTask):
    """Bias joints/base toward rest; ``rest_q="midrange"`` uses joint-limit midpoints as the rest state.

    Supported variables:
    - RobotState: joints and optional floating base.
    - VectorVar: joint-only block, sized by `rest_q` with no robot.
    - SE3Var: pose-only block, including model-offset regularization.

    Example:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.opt.var_values import VarKey, VarValues
        >>> from robokit.robo.robot import Robot
        >>> robot = Robot.load(load_robot_description("panda_description"))
        >>> rest_q = robot.spec.midrange_q
        >>> configuration_task = ConfigurationTask(robot=robot, rest_q=rest_q, weight=0.1)
        >>> q_init = wp.from_numpy(rest_q + 0.1, dtype=wp.float32)
        >>> state = robot.state(q=q_init)
        >>> residual = configuration_task.compute_weighted_residual(VarValues({VarKey.ROBOT: state}))
        >>> # Residual should be ~0.1 * weight
    """

    def __init__(
        self,
        robot: Optional[Robot] = None,
        rest_q: Optional[Union[np.ndarray, wp.array, Literal["midrange"], Context]] = None,
        T_world_base_rest: Optional[Union[wp.array, np.ndarray, Context]] = None,
        weight: Union[float, Sequence[float], np.ndarray] = 1.0,
        base_weight: Optional[Union[float, Sequence[float], np.ndarray]] = None,
        include_joints: bool = True,
        var_key: Enum = VarKey.ROBOT,
        name: str = "",
        mask: Optional[Union[np.ndarray, Context]] = None,
    ):
        self.name = name
        self.mask = mask  # scales each joint's weight, e.g. a helper's mask of the joints it rests
        self.var_key = var_key
        self.rest_q_arg: Optional[Union[np.ndarray, wp.array, Literal["midrange"], Context]] = rest_q
        self._T_world_base_rest_arg = T_world_base_rest
        self.weight = weight
        self.base_weight = base_weight
        self.include_joints = include_joints
        self._has_base = T_world_base_rest is not None
        self.T_world_base_rest: Optional[wp.array] = None
        self.robot: Optional[Robot] = None
        self.device = None
        self.rest_q = None
        self.residual_weight = None
        self.set_robot(robot)

    @property
    def active(self) -> bool:
        if self.robot is None:  # a template counts its configured weights
            return bool(np.any(self.weight) or (self.base_weight is not None and np.any(self.base_weight)))
        return bool(
            np.any(self._residual_weight_np[: self._joint_rows]) or np.any(self._residual_weight_np[self._num_joints :])
        )

    def resolve(self, var_values: VarValues, context: Mapping[Context, Any]):
        """Build from the robot; Warp rest values from the helper are its buffers, read directly."""
        super().resolve(var_values, context)
        var = var_values.get(self.var_key)
        state = var if isinstance(var, RobotState) else None
        # base residuals follow a floating base; a fixed-base robot drops them
        self._has_base = self._T_world_base_rest_arg is not None and (state is None or state.has_floating_base)
        self.set_robot(state.robot if state is not None else None)
        self.init_buffers(var.device)
        if isinstance(self.rest_q_arg, wp.array):
            self.rest_q = self.rest_q_arg
        if isinstance(self._T_world_base_rest_arg, wp.array):
            self.T_world_base_rest = self._T_world_base_rest_arg

    def set_robot(self, robot: Optional[Robot] = None):
        self.robot = robot
        rest_q = self.rest_q_arg
        if isinstance(rest_q, str):
            if rest_q != "midrange":
                raise ValueError(f"Unsupported rest_q: {rest_q}")
            rest_q = robot.spec.midrange_q.astype(np.float32) if robot is not None else None
        if robot is not None:
            self.rest_q_np = rest_q if rest_q is not None else robot.spec.zero_q
        else:
            self.rest_q_np = rest_q if rest_q is not None else np.zeros(0, dtype=np.float32)
        self._num_joints = robot.num_actuated_joints if robot is not None else int(np.size(self.rest_q_np))

        num_joints = self._num_joints
        has_base = self._has_base

        residual_weight = np.ones(num_joints + 6 * has_base, dtype=np.float32)
        if num_joints:  # a config-time stub has no joints yet; it keeps `weight` until built
            residual_weight[:num_joints] = self.weight
            if robot is not None and self.mask is not None:
                residual_weight[:num_joints] *= np.asarray(self.mask, dtype=np.float32)
        if has_base and self.base_weight is not None:
            residual_weight[num_joints:] = self.base_weight
        self._joint_rows = num_joints if self.include_joints or not has_base else 0
        residual_dim = self._joint_rows + 6 * has_base

        self._residual_weight_np = residual_weight
        self._residual_dim = residual_dim

    def init_buffers(self, device: wp_device_type):
        self.device = device
        # (D,), (N, D) and (B, F, D) all collapse to (N, D): one rest configuration per batch element
        if isinstance(self.rest_q_np, wp.array):
            self.rest_q = wp.clone(self.rest_q_np, device=device)
        elif self._num_joints:
            rest_q = np.asarray(self.rest_q_np, dtype=np.float32).reshape(-1, self._num_joints)
            self.rest_q = wp.from_numpy(rest_q, dtype=wp.float32, device=device)
        self.residual_weight = wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device=device)
        if self._has_base:
            base_rest = self._T_world_base_rest_arg
            if isinstance(base_rest, np.ndarray):
                base_rest = wp.array(
                    np.ascontiguousarray(base_rest, dtype=np.float32).reshape(-1, 7), dtype=wp_vec7, device=device
                )
            if base_rest is not None:
                self.T_world_base_rest = wp.clone(base_rest)
            else:
                self.T_world_base_rest = se3_identity(shape=(1,), device=device)

    def set_weight(self, weight: Union[float, Sequence[float]]):
        """Update joint-block residual weights in-place. Base-block weights are preserved."""
        if self._num_joints:
            self._residual_weight_np[: self._num_joints] = weight
        self.weight = weight
        if self.residual_weight is not None:
            wp.copy(self.residual_weight, wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device="cpu"))

    def set_target(
        self,
        rest_q: Optional[Union[wp.array, np.ndarray]] = None,
        T_world_base_rest: Optional[Union[wp.array, np.ndarray]] = None,
    ):
        """Update the rest references; a config-time stub keeps NumPy references until its buffers are built."""
        if self.robot is None:  # config-time stub: the constructor arguments are the references
            if rest_q is not None:
                self.rest_q_arg = np.asarray(rest_q, dtype=np.float32)
            if T_world_base_rest is not None:
                self._T_world_base_rest_arg = np.asarray(T_world_base_rest, dtype=np.float32)
                self._has_base = True
            return
        if self.rest_q is None:
            self.init_buffers((rest_q if rest_q is not None else T_world_base_rest).device)
        if rest_q is not None:
            if rest_q.shape != self.rest_q.shape:
                self.rest_q = wp.empty_like(rest_q)
            wp.copy(self.rest_q, rest_q)
        if T_world_base_rest is not None and self.T_world_base_rest is not None:
            if T_world_base_rest.shape != self.T_world_base_rest.shape:
                self.T_world_base_rest = wp.empty_like(T_world_base_rest)
            wp.copy(self.T_world_base_rest, T_world_base_rest)

    @property
    def residual_dim(self) -> int:
        return self._residual_dim

    def _blocks(self, var: Var) -> Tuple[Optional[wp.array], Optional[wp.array], bool]:
        """Joints, base pose, and whether the base applies: a plain SE3Var is all pose, a VectorVar all joints."""
        if isinstance(var, RobotState):
            return (var.q if self._num_joints else None), var.T_world_base, var.has_floating_base
        if isinstance(var, SE3Var):
            return None, var.xyz_wxyz, True
        return cast(VectorVar, var).x, None, False

    def compute_weighted_residual(
        self,
        var_values: VarValues,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        var = var_values.get(self.var_key)
        var_q, var_T, has_base = self._blocks(var)
        if self.device is None:
            self.init_buffers((var_q if var_q is not None else var_T).device)

        kernel_device = out_residual.device if out_residual is not None else self.device
        batch_size = var.batch_size
        if out_residual is None:
            out_residual = wp.empty((batch_size, self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        if self._has_base and has_base:
            if self.include_joints:
                wp.launch(
                    kernel=compute_configuration_task_weighted_residual_with_base_kernel,
                    dim=batch_size,
                    inputs=[
                        var_q,
                        var_T,
                        self.rest_q,
                        self.T_world_base_rest,
                        self.residual_weight,
                        row_offset,
                    ],
                    outputs=[out_residual],
                    device=kernel_device,
                )
            else:
                wp.launch(
                    kernel=compute_base_configuration_task_weighted_residual_kernel,
                    dim=batch_size,
                    inputs=[
                        var_T,
                        self.T_world_base_rest,
                        self.residual_weight,
                        self._num_joints,
                        row_offset,
                    ],
                    outputs=[out_residual],
                    device=kernel_device,
                )
        else:
            wp.launch(
                kernel=compute_configuration_task_weighted_residual_kernel,
                dim=(batch_size, self._num_joints),
                inputs=[
                    var_q,
                    self.rest_q,
                    self.residual_weight,
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
    ) -> wp.array:
        var = var_values.get(self.var_key)
        var_q, var_T, has_base = self._blocks(var)
        col_offset = var_values.tangent_offset(self.var_key)
        if self.device is None:
            self.init_buffers((var_q if var_q is not None else var_T).device)

        kernel_device = out_jacobian.device if out_jacobian is not None else self.device
        total_dofs = var.tangent_dim

        batch_size = var.batch_size
        if out_jacobian is None:
            out_jacobian = wp.zeros((batch_size, self.residual_dim, total_dofs), dtype=wp.float32, device=kernel_device)
            row_offset = 0
            col_offset = 0

        if self._has_base and has_base:
            if self.include_joints:
                wp.launch(
                    kernel=compute_configuration_task_weighted_jacobian_with_base_kernel,
                    dim=(batch_size, total_dofs),
                    inputs=[
                        var_T,
                        self.T_world_base_rest,
                        self.residual_weight,
                        row_offset,
                        col_offset,
                        self._num_joints,
                    ],
                    outputs=[out_jacobian],
                    device=kernel_device,
                )
            else:
                wp.launch(
                    kernel=compute_base_configuration_task_weighted_jacobian_kernel,
                    dim=(batch_size, 6),
                    inputs=[
                        var_T,
                        self.T_world_base_rest,
                        self.residual_weight,
                        self._num_joints,
                        row_offset,
                        col_offset,
                    ],
                    outputs=[out_jacobian],
                    device=kernel_device,
                )
        else:
            wp.launch(
                kernel=compute_configuration_task_weighted_jacobian_kernel,
                dim=(batch_size, self._num_joints),
                inputs=[
                    has_base,
                    self.residual_weight,
                    row_offset,
                    col_offset,
                ],
                outputs=[out_jacobian],
                device=kernel_device,
            )

        return out_jacobian


# --- kernels ----------------------------------------------------------------
@wp.kernel
def compute_configuration_task_weighted_residual_kernel(
    q: wp.array2d(dtype=wp.float32),  # [batch, num_joints]
    rest_q: wp.array2d(dtype=wp.float32),  # [rest_batch, num_joints]
    residual_weight: wp.array1d(dtype=wp.float32),  # [num_joints]
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),  # [batch, total_residual_dim]
):
    batch_idx, joint_idx = wp.tid()  # type: ignore
    n_repeat = wp.max(1, q.shape[0] // rest_q.shape[0])
    rest_idx = batch_idx // n_repeat
    diff = q[batch_idx, joint_idx] - rest_q[rest_idx, joint_idx]
    out_residual[batch_idx, row_offset + joint_idx] = residual_weight[joint_idx] * diff


@wp.kernel
def compute_configuration_task_weighted_residual_with_base_kernel(
    q: wp.array2d(dtype=wp.float32),  # [batch, num_joints]
    T_world_base: wp.array1d(dtype=wp_vec7),  # [batch]
    rest_q: wp.array2d(dtype=wp.float32),  # [rest_batch, num_joints]
    T_world_base_rest: wp.array1d(dtype=wp_vec7),  # [rest_batch]
    residual_weight: wp.array1d(dtype=wp.float32),  # [num_joints + 6]
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),  # [batch, total_residual_dim]
):
    batch_idx = wp.tid()  # type: ignore
    num_joints = q.shape[1]
    n_repeat = wp.max(1, q.shape[0] // rest_q.shape[0])
    rest_idx = batch_idx // n_repeat  # pyright: ignore[reportOperatorIssue]

    # joint residuals
    for joint_idx in range(num_joints):
        diff = q[batch_idx, joint_idx] - rest_q[rest_idx, joint_idx]
        out_residual[batch_idx, row_offset + joint_idx] = residual_weight[joint_idx] * diff

    # base residuals
    T_ref_inv = se3_inverse_func(T_world_base_rest[rest_idx])
    T_error = se3_compose_func(T_world_base[batch_idx], T_ref_inv)
    r_base = se3_log_map_func(T_error, wp.float32(1e-4))

    out_residual[batch_idx, row_offset + num_joints + 0] = residual_weight[num_joints + 0] * r_base[0]
    out_residual[batch_idx, row_offset + num_joints + 1] = residual_weight[num_joints + 1] * r_base[1]
    out_residual[batch_idx, row_offset + num_joints + 2] = residual_weight[num_joints + 2] * r_base[2]
    out_residual[batch_idx, row_offset + num_joints + 3] = residual_weight[num_joints + 3] * r_base[3]
    out_residual[batch_idx, row_offset + num_joints + 4] = residual_weight[num_joints + 4] * r_base[4]
    out_residual[batch_idx, row_offset + num_joints + 5] = residual_weight[num_joints + 5] * r_base[5]


@wp.kernel
def compute_base_configuration_task_weighted_residual_kernel(
    T_world_base: wp.array1d(dtype=wp_vec7),
    T_world_base_rest: wp.array1d(dtype=wp_vec7),
    residual_weight: wp.array1d(dtype=wp.float32),
    num_joints: int,
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),
):
    batch_idx = wp.tid()
    rest_idx = batch_idx // (wp.max(1, T_world_base.shape[0] // T_world_base_rest.shape[0]))  # pyright: ignore[reportOperatorIssue]
    T_error = se3_compose_func(T_world_base[batch_idx], se3_inverse_func(T_world_base_rest[rest_idx]))
    r_base = se3_log_map_func(T_error, wp.float32(1e-4))
    for i in range(6):
        out_residual[batch_idx, row_offset + i] = residual_weight[num_joints + i] * r_base[i]


@wp.kernel
def compute_configuration_task_weighted_jacobian_kernel(
    has_floating_base: wp.bool,
    residual_weight: wp.array1d(dtype=wp.float32),  # [num_joints]
    row_offset: int,
    col_offset: int,
    out_jacobian: wp.array3d(dtype=wp.float32),  # [batch, total_residual_dim, total_dofs]
):
    batch_idx, joint_idx = wp.tid()  # type: ignore

    base_col_offset = 6 if has_floating_base else 0

    # joint identity block
    out_jacobian[batch_idx, row_offset + joint_idx, col_offset + base_col_offset + joint_idx] = residual_weight[
        joint_idx
    ]


@wp.kernel
def compute_configuration_task_weighted_jacobian_with_base_kernel(
    T_world_base: wp.array1d(dtype=wp_vec7),  # [batch]
    T_world_base_rest: wp.array1d(dtype=wp_vec7),  # [rest_batch]
    residual_weight: wp.array1d(dtype=wp.float32),  # [num_joints + 6]
    row_offset: int,
    col_offset: int,
    num_joints: int,
    out_jacobian: wp.array3d(dtype=wp.float32),  # [batch, total_residual_dim, total_dofs]
):
    batch_idx, col_idx = wp.tid()  # type: ignore
    n_repeat = wp.max(1, T_world_base.shape[0] // T_world_base_rest.shape[0])
    rest_idx = batch_idx // n_repeat

    # joint identity block starts after the base columns
    if col_idx >= 6 and col_idx < (6 + num_joints):
        joint_idx = col_idx - 6
        out_jacobian[batch_idx, row_offset + joint_idx, col_offset + col_idx] = residual_weight[joint_idx]

    # base block is Jlog @ Ad
    if col_idx < 6:
        T_ref_inv = se3_inverse_func(T_world_base_rest[rest_idx])
        T_error = se3_compose_func(T_world_base[batch_idx], T_ref_inv)

        # compute Jlog @ Ad column by column
        jlog = se3_jlog_func(T_error, wp.float32(1e-4))

        # adjoint column for the selected basis vector
        unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        unit_vec[col_idx] = 1.0
        ad_col = se3_adjoint_func(T_world_base_rest[rest_idx]) * unit_vec

        # apply Jlog
        result_col = wp.mul(jlog, ad_col)

        out_jacobian[batch_idx, row_offset + num_joints + 0, col_offset + col_idx] = (
            residual_weight[num_joints + 0] * result_col[0]
        )
        out_jacobian[batch_idx, row_offset + num_joints + 1, col_offset + col_idx] = (
            residual_weight[num_joints + 1] * result_col[1]
        )
        out_jacobian[batch_idx, row_offset + num_joints + 2, col_offset + col_idx] = (
            residual_weight[num_joints + 2] * result_col[2]
        )
        out_jacobian[batch_idx, row_offset + num_joints + 3, col_offset + col_idx] = (
            residual_weight[num_joints + 3] * result_col[3]
        )
        out_jacobian[batch_idx, row_offset + num_joints + 4, col_offset + col_idx] = (
            residual_weight[num_joints + 4] * result_col[4]
        )
        out_jacobian[batch_idx, row_offset + num_joints + 5, col_offset + col_idx] = (
            residual_weight[num_joints + 5] * result_col[5]
        )


@wp.kernel
def compute_base_configuration_task_weighted_jacobian_kernel(
    T_world_base: wp.array1d(dtype=wp_vec7),
    T_world_base_rest: wp.array1d(dtype=wp_vec7),
    residual_weight: wp.array1d(dtype=wp.float32),
    num_joints: int,
    row_offset: int,
    col_offset: int,
    out_jacobian: wp.array3d(dtype=wp.float32),
):
    batch_idx, col_idx = wp.tid()  # type: ignore
    rest_idx = batch_idx // (wp.max(1, T_world_base.shape[0] // T_world_base_rest.shape[0]))
    T_ref = T_world_base_rest[rest_idx]
    T_error = se3_compose_func(T_world_base[batch_idx], se3_inverse_func(T_ref))
    unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    unit_vec[col_idx] = 1.0
    result_col = wp.mul(se3_jlog_func(T_error, wp.float32(1e-4)), se3_adjoint_func(T_ref) * unit_vec)
    for i in range(6):
        out_jacobian[batch_idx, row_offset + i, col_offset + col_idx] = residual_weight[num_joints + i] * result_col[i]
