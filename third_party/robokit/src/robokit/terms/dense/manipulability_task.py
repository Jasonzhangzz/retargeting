# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportOperatorIssue=false
"""Single-frame Yoshikawa manipulability task."""

from typing import Optional, Union

import warp as wp

from robokit.opt.var_values import VarValues
from robokit.robo.robot import Robot
from robokit.terms.robot_task import RobotTask
from robokit.terms.task import Context, ResidualTask
from robokit.utils.warp_utils import wp_device_type, wp_vec7


# --- device functions -------------------------------------------------------
@wp.func
def _compute_det_3x3_func(
    a00: float,
    a01: float,
    a02: float,
    a10: float,
    a11: float,
    a12: float,
    a20: float,
    a21: float,
    a22: float,
) -> float:
    """Compute determinant of 3x3 matrix."""
    return a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)


# --- task -------------------------------------------------------------------
class ManipulabilityTask(RobotTask, ResidualTask):
    """Maximize Yoshikawa `sqrt(det(J @ J.T))` at `frame_index`, using the end-effector translational Jacobian.

    Analytic derivative columns:
    - Fixed-base joints: computed normally.
    - Floating base: left zero.

    Example:
        >>> import numpy as np
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.opt.var_values import VarKey, VarValues
        >>> from robokit.robo.robot import Robot
        >>> robot = Robot.load(load_robot_description("ur10_description"))
        >>> frame_index = robot.spec.link_names.index("ee_link")
        >>> random_q = wp.from_numpy(
        ...     np.array([[-0.5, 0.3, -0.2, 0.4, -0.1, 0.25]], dtype=np.float32), dtype=wp.float32
        ... )
        >>> state = robot.state(q=random_q)
        >>> task = ManipulabilityTask(robot, frame_index, weight=0.01)
        >>> task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy().shape[1] == 1
        True
        >>> task.compute_weighted_jacobian(VarValues({VarKey.ROBOT: state})).numpy().shape[1] == 1
        True
    """

    def __init__(
        self,
        robot: Optional[Robot] = None,
        frame_index: Optional[Union[int, Context]] = None,
        weight: float = 0.01,
        epsilon: float = 1e-6,
    ):
        self.weight = float(weight)
        self.epsilon = float(epsilon)
        self.robot: Optional[Robot] = robot
        self.frame_index = frame_index
        self.device = None
        # keep weight on the device so graph replay can use updates
        self.weight_wp: Optional[wp.array] = None

    def set_weight(self, weight: float):
        """Update the manipulability weight in-place (live-tunable, CUDA-graph safe)."""
        self.weight = float(weight)
        if self.weight_wp is not None:
            self.weight_wp.assign([self.weight])

    def _weight_array(self, device: wp_device_type) -> wp.array:
        if self.weight_wp is None:
            self.weight_wp = wp.array([self.weight], dtype=wp.float32, device=device)
        return self.weight_wp

    @property
    def residual_dim(self) -> int:
        return 1

    def compute_weighted_residual(
        self, var_values: VarValues, out_residual: Optional[wp.array] = None, row_offset: int = 0
    ) -> wp.array:
        var = var_values.get(self.var_key)
        if not var.is_fk_computed:
            self.robot.forward_kinematics(var)
        if not var.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(var)
        kernel_device = out_residual.device if out_residual is not None else var.q.device
        batch_size = var.batch_size
        if out_residual is None:
            out_residual = wp.zeros((batch_size, self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        wp.launch(
            kernel=compute_manipulability_residual_kernel,
            dim=batch_size,
            inputs=[
                var.S_world,
                var.T_world_link,
                var.spec_tensors.link_ancestor_joints_mask,
                var.spec_tensors.joints_to_actuated_mapping,
                self.frame_index,
                self.robot.spec.num_actuated_joints,
                self.epsilon,
                self._weight_array(kernel_device),
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
        col_offset = var_values.tangent_offset(self.var_key)
        if not var.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(var)
        kernel_device = out_jacobian.device if out_jacobian is not None else var.q.device
        batch_size = var.batch_size
        if out_jacobian is None:
            out_jacobian = wp.zeros(
                (batch_size, self.residual_dim, var.tangent_dim), dtype=wp.float32, device=kernel_device
            )
            row_offset = 0
        num_actuated = self.robot.spec.num_actuated_joints
        base_dim = 6 if bool(var.has_floating_base) else 0

        wp.launch(
            kernel=compute_manipulability_jacobian_kernel,
            dim=(batch_size, num_actuated),
            inputs=[
                var.S_world,
                var.T_world_link,
                var.spec_tensors.link_ancestor_joints_mask,
                var.spec_tensors.joints_to_actuated_mapping,
                var.spec_tensors.parent_joint_indices,
                self.frame_index,
                num_actuated,
                base_dim,
                self.epsilon,
                self._weight_array(kernel_device),
                row_offset,
                col_offset,
            ],
            outputs=[out_jacobian],
            device=kernel_device,
        )
        return out_jacobian


# --- device code ------------------------------------------------------------
@wp.func
def _compute_end_effector_jacobian_column_func(
    S_world: wp.array3d(dtype=wp.float32),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    ancestor_mask: wp.array2d(dtype=wp.bool),
    batch_idx: int,
    link_idx: int,
    act_idx: int,
    p: wp.vec3,
) -> wp.vec3:
    """End-effector translational Jacobian column for actuated DOF `act_idx`: v + w x p."""
    num_joints = S_world.shape[2]
    v = wp.vec3(0.0, 0.0, 0.0)
    w = wp.vec3(0.0, 0.0, 0.0)
    for joint_idx in range(num_joints):
        if ancestor_mask[link_idx, joint_idx]:
            jw = joints_to_actuated[joint_idx, act_idx]
            if jw != 0.0:
                v = (
                    v
                    + wp.vec3(
                        S_world[batch_idx, 0, joint_idx],
                        S_world[batch_idx, 1, joint_idx],
                        S_world[batch_idx, 2, joint_idx],
                    )
                    * jw
                )
                w = (
                    w
                    + wp.vec3(
                        S_world[batch_idx, 3, joint_idx],
                        S_world[batch_idx, 4, joint_idx],
                        S_world[batch_idx, 5, joint_idx],
                    )
                    * jw
                )
    return v + wp.cross(w, p)


@wp.kernel
def compute_manipulability_residual_kernel(
    S_world: wp.array3d(dtype=wp.float32),  # [batch, 6, num_joints]
    T_world_link: wp.array2d(dtype=wp_vec7),  # [batch, num_links]
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    target_link_index: int,
    num_actuated: int,
    epsilon: float,
    weight: wp.array1d(dtype=wp.float32),
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),
):
    batch_idx = wp.tid()
    T = T_world_link[batch_idx, target_link_index]
    p = wp.vec3(T[0], T[1], T[2])

    jjt_00 = float(0.0)
    jjt_01 = float(0.0)
    jjt_02 = float(0.0)
    jjt_11 = float(0.0)
    jjt_12 = float(0.0)
    jjt_22 = float(0.0)

    for act_idx in range(num_actuated):
        a = _compute_end_effector_jacobian_column_func(
            S_world, joints_to_actuated, link_ancestor_joints_mask, batch_idx, target_link_index, act_idx, p
        )
        jjt_00 = jjt_00 + a[0] * a[0]
        jjt_01 = jjt_01 + a[0] * a[1]
        jjt_02 = jjt_02 + a[0] * a[2]
        jjt_11 = jjt_11 + a[1] * a[1]
        jjt_12 = jjt_12 + a[1] * a[2]
        jjt_22 = jjt_22 + a[2] * a[2]

    det_jjt = _compute_det_3x3_func(jjt_00, jjt_01, jjt_02, jjt_01, jjt_11, jjt_12, jjt_02, jjt_12, jjt_22)
    manip = wp.sqrt(wp.max(0.0, det_jjt))
    out_residual[batch_idx, row_offset] = weight[0] / (manip + epsilon)


@wp.kernel
def compute_manipulability_jacobian_kernel(
    S_world: wp.array3d(dtype=wp.float32),  # [batch, 6, num_joints]
    T_world_link: wp.array2d(dtype=wp_vec7),  # [batch, num_links]
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    parent_joint_indices: wp.array1d(dtype=wp.int32),
    target_link_index: int,
    num_actuated: int,
    base_dim: int,
    epsilon: float,
    weight: wp.array1d(dtype=wp.float32),
    row_offset: int,
    col_offset: int,
    out_jacobian: wp.array3d(dtype=wp.float32),  # [batch, residual_dim, total_dofs]
):
    batch_idx, dq_idx = wp.tid()  # pyright: ignore  # derivative w.r.t. actuated DOF dq_idx
    num_joints = S_world.shape[2]

    T = T_world_link[batch_idx, target_link_index]
    p = wp.vec3(T[0], T[1], T[2])
    # d(link position)/dq_dq_idx = end-effector Jacobian column dq_idx
    dp = _compute_end_effector_jacobian_column_func(
        S_world, joints_to_actuated, link_ancestor_joints_mask, batch_idx, target_link_index, dq_idx, p
    )

    jjt_00 = float(0.0)
    jjt_01 = float(0.0)
    jjt_02 = float(0.0)
    jjt_11 = float(0.0)
    jjt_12 = float(0.0)
    jjt_22 = float(0.0)

    dA_00 = float(0.0)
    dA_01 = float(0.0)
    dA_02 = float(0.0)
    dA_11 = float(0.0)
    dA_12 = float(0.0)
    dA_22 = float(0.0)

    for act_idx in range(num_actuated):
        v_k = wp.vec3(0.0, 0.0, 0.0)
        w_k = wp.vec3(0.0, 0.0, 0.0)
        dv_k = wp.vec3(0.0, 0.0, 0.0)  # d(v_k)/dq_dq_idx
        dw_k = wp.vec3(0.0, 0.0, 0.0)  # d(w_k)/dq_dq_idx

        for joint_idx in range(num_joints):
            if link_ancestor_joints_mask[target_link_index, joint_idx]:
                jw = joints_to_actuated[joint_idx, act_idx]
                if jw != 0.0:
                    v_j = wp.vec3(
                        S_world[batch_idx, 0, joint_idx],
                        S_world[batch_idx, 1, joint_idx],
                        S_world[batch_idx, 2, joint_idx],
                    )
                    w_j = wp.vec3(
                        S_world[batch_idx, 3, joint_idx],
                        S_world[batch_idx, 4, joint_idx],
                        S_world[batch_idx, 5, joint_idx],
                    )
                    v_k = v_k + v_j * jw
                    w_k = w_k + w_j * jw

                    # d(S_joint)/dq = sum over ancestors of joint mapped to dq_idx of ad_{S_anc} S_joint
                    dvj = wp.vec3(0.0, 0.0, 0.0)
                    dwj = wp.vec3(0.0, 0.0, 0.0)
                    anc = parent_joint_indices[joint_idx]
                    while anc >= 0:
                        aw = joints_to_actuated[anc, dq_idx]
                        if aw != 0.0:
                            v_a = wp.vec3(
                                S_world[batch_idx, 0, anc], S_world[batch_idx, 1, anc], S_world[batch_idx, 2, anc]
                            )
                            w_a = wp.vec3(
                                S_world[batch_idx, 3, anc], S_world[batch_idx, 4, anc], S_world[batch_idx, 5, anc]
                            )
                            dvj = dvj + (wp.cross(w_a, v_j) + wp.cross(v_a, w_j)) * aw
                            dwj = dwj + wp.cross(w_a, w_j) * aw
                        anc = parent_joint_indices[anc]
                    dv_k = dv_k + dvj * jw
                    dw_k = dw_k + dwj * jw

        a = v_k + wp.cross(w_k, p)
        # d(a_k)/dq = d(v_k) + d(w_k) x p + w_k x d(p)
        da = dv_k + wp.cross(dw_k, p) + wp.cross(w_k, dp)

        jjt_00 = jjt_00 + a[0] * a[0]
        jjt_01 = jjt_01 + a[0] * a[1]
        jjt_02 = jjt_02 + a[0] * a[2]
        jjt_11 = jjt_11 + a[1] * a[1]
        jjt_12 = jjt_12 + a[1] * a[2]
        jjt_22 = jjt_22 + a[2] * a[2]

        dA_00 = dA_00 + 2.0 * a[0] * da[0]
        dA_11 = dA_11 + 2.0 * a[1] * da[1]
        dA_22 = dA_22 + 2.0 * a[2] * da[2]
        dA_01 = dA_01 + da[0] * a[1] + a[0] * da[1]
        dA_02 = dA_02 + da[0] * a[2] + a[0] * da[2]
        dA_12 = dA_12 + da[1] * a[2] + a[1] * da[2]

    det_jjt = _compute_det_3x3_func(jjt_00, jjt_01, jjt_02, jjt_01, jjt_11, jjt_12, jjt_02, jjt_12, jjt_22)
    if det_jjt <= 0.0:
        return

    manip = wp.sqrt(det_jjt)
    dres_dmanip = -weight[0] / ((manip + epsilon) * (manip + epsilon))
    dmanip_ddet = 0.5 / manip

    ddet_00 = jjt_11 * jjt_22 - jjt_12 * jjt_12
    ddet_11 = jjt_00 * jjt_22 - jjt_02 * jjt_02
    ddet_22 = jjt_00 * jjt_11 - jjt_01 * jjt_01
    ddet_01 = -2.0 * (jjt_01 * jjt_22 - jjt_12 * jjt_02)
    ddet_02 = 2.0 * (jjt_01 * jjt_12 - jjt_11 * jjt_02)
    ddet_12 = -2.0 * (jjt_00 * jjt_12 - jjt_01 * jjt_02)

    ddet_dq = ddet_00 * dA_00 + ddet_11 * dA_11 + ddet_22 * dA_22 + ddet_01 * dA_01 + ddet_02 * dA_02 + ddet_12 * dA_12

    col = col_offset + base_dim + dq_idx
    out_jacobian[batch_idx, row_offset, col] = dres_dmanip * dmanip_ddet * ddet_dq
