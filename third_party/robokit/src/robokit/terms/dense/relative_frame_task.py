# pyright: reportArgumentType=false, reportIndexIssue=false, reportGeneralTypeIssues=false, reportOperatorIssue=false
from typing import Optional, Sequence, Tuple

import warp as wp

from robokit.lie.se3_kernels import se3_adjoint_func, se3_jlog_func, se3_log_map_func
from robokit.lie.se3_kernels import se3_inverse_func as inv
from robokit.lie.se3_kernels import se3_multiply_func as mul
from robokit.opt.var_values import VarValues
from robokit.robo.robot import Robot
from robokit.terms.robot_task import RobotTask
from robokit.terms.task import ResidualTask
from robokit.utils.warp_utils import wp_vec6, wp_vec7


@wp.kernel
def _relative_frame_kernel(
    S_world: wp.array3d(dtype=wp.float32),  # (poses, 6, joints), read with the Jacobian
    T_world_link: wp.array2d(dtype=wp_vec7),  # (poses, links), `reference_frames.shape[0]` poses per trajectory
    link_pairs: wp.array2d(dtype=wp.int32),
    reference_frames: wp.array2d(dtype=wp.int32),  # (frames, pairs): each pair's reference frame, -1 off
    ancestor_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    weight: wp.vec2,
    offsets: wp.vec3i,  # Jacobian (1) or residual (0), row offset, column offset
    out: wp.array3d(dtype=wp.float32),  # (poses, rows, columns), one column for residuals
):
    pose, pair, col = wp.tid()  # pyright: ignore[reportAssignmentType]
    frames, a, b = reference_frames.shape[0], link_pairs[pair, 0], link_pairs[pair, 1]
    reference = reference_frames[pose % frames, pair]
    start = wp.where(reference < 0, pose, pose - pose % frames + reference)  # off: its own pose, a zero residual
    T_a_bref = mul(inv(T_world_link[start, a]), T_world_link[start, b])  # b at the reference frame, in a's frame
    error = mul(inv(T_a_bref), mul(inv(T_world_link[pose, a]), T_world_link[pose, b]))  # T_bref_b
    value = se3_log_map_func(error, wp.float32(1e-4))
    if offsets[0] != 0:  # b's spatial velocity relative to a's, in b's frame: d log(error) = jlog Ad(T_b_world) dS
        spatial = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for joint in range(S_world.shape[2]):
            sign = wp.where(ancestor_mask[b, joint], 1.0, 0.0) - wp.where(ancestor_mask[a, joint], 1.0, 0.0)
            for row in range(6):
                spatial[row] += sign * joints_to_actuated[joint, col] * S_world[pose, row, joint]
        value = se3_jlog_func(error, wp.float32(1e-4)) * (se3_adjoint_func(inv(T_world_link[pose, b])) * spatial)
    for row in range(6):
        on = wp.where(reference < 0, 0.0, weight[row // 3])
        out[pose, offsets[1] + 6 * pair + row, wp.where(offsets[0] != 0, offsets[2] + col, 0)] = on * value[row]


class RelativeFrameTask(RobotTask, ResidualTask):
    """Hold each link pair's relative pose `T_a_b` at its value on a reference frame of the same trajectory: wrapped
    in `TrajectoryTask`, frame `t` reads pair `p`'s reference frame `reference_frames[t, p]`; -1 leaves its rows zero,
    else the residual is `log(T_a_b(reference)^-1 T_a_b(t))`, with the reference held fixed.

    Example:
        >>> import numpy as np
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.terms.trajectory_task import TrajectoryTask
        >>> robot = Robot.load(load_robot_description("panda_description"))
        >>> reference = wp.array([[0], [0]], dtype=wp.int32, device="cpu")  # both frames hold frame 0's pose
        >>> task = TrajectoryTask(RelativeFrameTask(robot, [(0, robot.link_names.index("panda_hand"))], reference), 2)
        >>> q = wp.array(np.stack([robot.spec.zero_q, robot.spec.zero_q + 0.1])[None], dtype=wp.float32, device="cpu")
        >>> residual = task.compute_weighted_residual(VarValues({task.var_key: robot.state(q=q)})).numpy()
        >>> bool(np.allclose(residual[0, :6], 0.0)), bool(np.linalg.norm(residual[0, 6:]) > 0.1)
        (True, True)
    """

    def __init__(
        self,
        robot: Robot,
        link_pairs: Sequence[Tuple[int, int]],
        reference_frames: wp.array,  # (frames, pairs) int32, which the caller rewrites
        position_weight: float = 1.0,
        orientation_weight: float = 1.0,
    ):
        self.robot, self.reference_frames, self.device = robot, reference_frames, reference_frames.device
        self.weight = (position_weight, orientation_weight)
        self.link_pairs = wp.array(link_pairs, dtype=wp.int32, device=self.device)  # which the caller may rewrite

    @property
    def residual_dim(self) -> int:
        return 6 * self.link_pairs.shape[0]

    def compute_weighted_residual(
        self, var_values: VarValues, out_residual: Optional[wp.array] = None, row_offset: int = 0
    ) -> wp.array:
        return self.compute_weighted_jacobian_analytic(var_values, out_residual, row_offset, jacobian=False)

    def compute_weighted_jacobian_analytic(
        self, var_values: VarValues, out_jacobian: Optional[wp.array] = None, row_offset: int = 0, jacobian: bool = True
    ) -> wp.array:
        var, rows, out = var_values.get(self.var_key), self.residual_dim, out_jacobian
        self.precompute(var_values, need_gradient=jacobian)
        if out is None:
            shape = (var.batch_size, rows, var.tangent_dim) if jacobian else (var.batch_size, rows)
            out, row_offset = wp.zeros(shape, dtype=wp.float32, device=self.device), 0
        spec, base = var.spec_tensors, var_values.tangent_offset(self.var_key) + (6 if var.has_floating_base else 0)
        wp.launch(
            _relative_frame_kernel,
            dim=(var.batch_size, self.link_pairs.shape[0], spec.joints_to_actuated_mapping.shape[1] if jacobian else 1),
            inputs=[var.S_world if jacobian else None, var.T_world_link, self.link_pairs, self.reference_frames]
            + [spec.link_ancestor_joints_mask, spec.joints_to_actuated_mapping, wp.vec2(*self.weight)]
            + [wp.vec3i(int(jacobian), row_offset, base), out if jacobian else out.reshape((*out.shape, 1))],
            device=self.device,
        )
        return out
