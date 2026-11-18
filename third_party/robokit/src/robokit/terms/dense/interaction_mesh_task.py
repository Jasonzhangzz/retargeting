# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportOperatorIssue=false
# pyright: reportCallIssue=false
"""Interaction-mesh Laplacian residual for retargeting."""

from typing import Any, Mapping, Optional, Union

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
def _compute_vertex_position_func(
    v: int,
    batch: int,
    vertex_link_index: wp.array1d(dtype=wp.int32),
    T_world_link: wp.array2d(dtype=wp_vec7),
    reference_vertex_positions: wp.array1d(dtype=wp.vec3),
) -> wp.vec3:
    li = vertex_link_index[v]
    if li < 0:
        return reference_vertex_positions[v]
    T = T_world_link[batch, li]
    return wp.vec3(T[0], T[1], T[2])


@wp.func
def _compute_vertex_position_jacobian_func(
    v: int,
    col: int,
    batch: int,
    base_dim: int,
    vertex_link_index: wp.array1d(dtype=wp.int32),
    T_world_link: wp.array2d(dtype=wp_vec7),
    T_world_base: wp.array1d(dtype=wp_vec7),
    S_world: wp.array3d(dtype=wp.float32),
    link_ancestor_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
) -> wp.vec3:
    """Return one vertex-position Jacobian column."""
    li = vertex_link_index[v]
    if li < 0:
        return wp.vec3(0.0, 0.0, 0.0)
    T = T_world_link[batch, li]
    p = wp.vec3(T[0], T[1], T[2])

    if col < base_dim:
        unit = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        unit[col] = 1.0
        twist = se3_adjoint_func(T_world_base[batch]) * unit
        v_lin = wp.vec3(twist[0], twist[1], twist[2])
        omega = wp.vec3(twist[3], twist[4], twist[5])
        return v_lin + wp.cross(omega, p)

    actuated_idx = col - base_dim
    num_joints = joints_to_actuated.shape[0]
    out = wp.vec3(0.0, 0.0, 0.0)
    for joint_idx in range(num_joints):
        if link_ancestor_mask[li, joint_idx]:
            weight_val = joints_to_actuated[joint_idx, actuated_idx]
            if weight_val != 0.0:
                v_lin = wp.vec3(
                    S_world[batch, 0, joint_idx], S_world[batch, 1, joint_idx], S_world[batch, 2, joint_idx]
                )
                omega = wp.vec3(
                    S_world[batch, 3, joint_idx], S_world[batch, 4, joint_idx], S_world[batch, 5, joint_idx]
                )
                out = out + weight_val * (v_lin + wp.cross(omega, p))
    return out


# --- kernels ----------------------------------------------------------------
@wp.kernel
def _interaction_mesh_residual_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),
    vertex_link_index: wp.array1d(dtype=wp.int32),
    reference_vertex_positions: wp.array1d(dtype=wp.vec3),
    neighbor_offsets: wp.array1d(dtype=wp.int32),
    neighbor_indices: wp.array1d(dtype=wp.int32),
    neighbor_weights: wp.array1d(dtype=wp.float32),
    target_laplacian: wp.array1d(dtype=wp.vec3),
    weight: float,
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),
):
    batch_idx, i = wp.tid()  # pyright: ignore
    p_i = _compute_vertex_position_func(i, batch_idx, vertex_link_index, T_world_link, reference_vertex_positions)
    centroid = wp.vec3(0.0, 0.0, 0.0)
    for k in range(neighbor_offsets[i], neighbor_offsets[i + 1]):
        p_j = _compute_vertex_position_func(
            neighbor_indices[k], batch_idx, vertex_link_index, T_world_link, reference_vertex_positions
        )
        centroid = centroid + neighbor_weights[k] * p_j
    lap = p_i - centroid
    res = lap - target_laplacian[i]
    off = row_offset + i * 3
    out_residual[batch_idx, off + 0] = weight * res[0]
    out_residual[batch_idx, off + 1] = weight * res[1]
    out_residual[batch_idx, off + 2] = weight * res[2]


@wp.kernel
def _interaction_mesh_jacobian_kernel(
    S_world: wp.array3d(dtype=wp.float32),
    T_world_link: wp.array2d(dtype=wp_vec7),
    T_world_base: wp.array1d(dtype=wp_vec7),
    vertex_link_index: wp.array1d(dtype=wp.int32),
    neighbor_offsets: wp.array1d(dtype=wp.int32),
    neighbor_indices: wp.array1d(dtype=wp.int32),
    neighbor_weights: wp.array1d(dtype=wp.float32),
    link_ancestor_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    weight: float,
    base_dim: int,
    row_offset: int,
    out_jacobian: wp.array3d(dtype=wp.float32),
):
    batch_idx, i, col = wp.tid()  # pyright: ignore
    # differentiate the Laplacian coordinate
    d = _compute_vertex_position_jacobian_func(
        i,
        col,
        batch_idx,
        base_dim,
        vertex_link_index,
        T_world_link,
        T_world_base,
        S_world,
        link_ancestor_mask,
        joints_to_actuated,
    )
    for k in range(neighbor_offsets[i], neighbor_offsets[i + 1]):
        dj = _compute_vertex_position_jacobian_func(
            neighbor_indices[k],
            col,
            batch_idx,
            base_dim,
            vertex_link_index,
            T_world_link,
            T_world_base,
            S_world,
            link_ancestor_mask,
            joints_to_actuated,
        )
        d = d - neighbor_weights[k] * dj
    off = row_offset + i * 3
    out_jacobian[batch_idx, off + 0, col] = weight * d[0]
    out_jacobian[batch_idx, off + 1, col] = weight * d[1]
    out_jacobian[batch_idx, off + 2, col] = weight * d[2]


# --- task -------------------------------------------------------------------
class InteractionMeshTask(RobotTask, ResidualTask):
    """Preserve interaction-mesh Laplacian coordinates from human to robot.

    Args:
        robot: Robot model. May be assigned later by the solver.
        robot_link_indices: Robot link for each dynamic vertex.
        weight: Residual multiplier.

    Example:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> import numpy as np
        >>> from robokit.helpers.humanoid_retargeting import build_interaction_mesh_frame
        >>> from robokit.robo.robot import Robot
        >>> robot = Robot.load(load_robot_description("panda_description"))
        >>> state = robot.state(q=robot.zero_q)
        >>> robot.forward_kinematics(state)
        >>> links = [robot.link_names.index(n) for n in robot.link_names[1:5]]
        >>> P = state.T_world_link.numpy()[0][links, :3]
        >>> pts = np.vstack([P, P.mean(0) + np.array([[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]], np.float32)])
        >>> task = InteractionMeshTask(robot, np.array(links, dtype=np.int32))
        >>> T_world_object = np.array([0, 0, 0, 1, 0, 0, 0], np.float32)
        >>> task.set_frame(*build_interaction_mesh_frame(P, pts[len(P):], T_world_object))
        >>> task.residual_dim
        21
    """

    def __init__(
        self,
        robot: Optional[Robot] = None,
        robot_link_indices: Optional[Union[np.ndarray, Context]] = None,
        weight: float = 1.0,
        name: str = "",
    ):
        self.name = name
        self.robot: Optional[Robot] = robot
        self.weight = float(weight)
        self.robot_link_indices = robot_link_indices
        self._robot_link_indices_np = (
            None
            if robot_link_indices is None or isinstance(robot_link_indices, Context)
            else np.asarray(robot_link_indices, dtype=np.int32)
        )
        self.num_vertices = 0

        self.device: Optional[wp_device_type] = None
        self._vertex_link_indices_wp: Optional[wp.array] = None
        self._neighbor_offsets_wp: Optional[wp.array] = None
        self._neighbor_indices_wp: Optional[wp.array] = None
        self._neighbor_weights_wp: Optional[wp.array] = None
        self._reference_vertex_positions_wp: Optional[wp.array] = None
        self._target_laplacian_wp: Optional[wp.array] = None

    def set_robot(self, robot: Robot):
        self.robot = robot

    def resolve(self, var_values: VarValues, context: Mapping[Context, Any]):
        """Build from the robot and the helper's links, the first vertices of every frame."""
        super().resolve(var_values, context)
        self._robot_link_indices_np = np.asarray(self.robot_link_indices, dtype=np.int32)

    @property
    def residual_dim(self) -> int:
        if self.num_vertices == 0:
            raise RuntimeError("call set_frame() before building the solver")
        return 3 * self.num_vertices

    def set_frame(
        self,
        neighbor_offsets: np.ndarray,
        neighbor_indices: np.ndarray,
        neighbor_weights: np.ndarray,
        reference_vertex_positions: np.ndarray,
        target_laplacian: np.ndarray,
    ):
        """Set adjacency, reference vertices, and target Laplacian for one frame."""
        if self._robot_link_indices_np is None:
            raise RuntimeError("robot_link_indices are required before set_frame()")

        reference_vertex_positions = np.asarray(reference_vertex_positions, dtype=np.float32)
        target_laplacian = np.asarray(target_laplacian, dtype=np.float32)
        num_vertices = len(reference_vertex_positions)
        if len(target_laplacian) != num_vertices:
            raise ValueError("reference_vertex_positions and target_laplacian must have the same length")
        if self.num_vertices == 0:
            if num_vertices < len(self._robot_link_indices_np):
                raise ValueError("frame has fewer vertices than robot_link_indices")
            self.num_vertices = num_vertices
            self._vertex_link_indices_np = np.full(num_vertices, -1, dtype=np.int32)
            self._vertex_link_indices_np[: len(self._robot_link_indices_np)] = self._robot_link_indices_np
        elif num_vertices != self.num_vertices:
            raise ValueError(f"frame vertex count must remain {self.num_vertices}, got {num_vertices}")

        self._neighbor_offsets_np = np.asarray(neighbor_offsets, dtype=np.int32)
        self._neighbor_indices_np = np.asarray(neighbor_indices, dtype=np.int32)
        self._neighbor_weights_np = np.asarray(neighbor_weights, dtype=np.float32)
        self._reference_vertex_positions_np = reference_vertex_positions
        self._target_laplacian_np = target_laplacian
        if self.device is not None:
            self.init_buffers(self.device)

    def init_buffers(self, device: wp_device_type):
        _ = self.residual_dim
        self.device = device
        self._vertex_link_indices_wp = wp.from_numpy(self._vertex_link_indices_np, dtype=wp.int32, device=device)
        self._neighbor_offsets_wp = wp.from_numpy(self._neighbor_offsets_np, dtype=wp.int32, device=device)
        self._neighbor_indices_wp = wp.from_numpy(self._neighbor_indices_np, dtype=wp.int32, device=device)
        self._neighbor_weights_wp = wp.from_numpy(self._neighbor_weights_np, dtype=wp.float32, device=device)
        self._reference_vertex_positions_wp = wp.from_numpy(
            self._reference_vertex_positions_np, dtype=wp.vec3, device=device
        )
        self._target_laplacian_wp = wp.from_numpy(self._target_laplacian_np, dtype=wp.vec3, device=device)

    def compute_weighted_residual(
        self, var_values: VarValues, out_residual: Optional[wp.array] = None, row_offset: int = 0
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
            kernel=_interaction_mesh_residual_kernel,
            dim=(var.batch_size, self.num_vertices),
            inputs=[
                var.T_world_link,
                self._vertex_link_indices_wp,
                self._reference_vertex_positions_wp,
                self._neighbor_offsets_wp,
                self._neighbor_indices_wp,
                self._neighbor_weights_wp,
                self._target_laplacian_wp,
                self.weight,
                row_offset,
            ],
            outputs=[out_residual],
            device=kernel_device,
        )
        return out_residual

    def compute_weighted_jacobian_analytic(
        self, var_values: VarValues, out_jacobian: Optional[wp.array] = None, row_offset: int = 0
    ) -> wp.array:
        var = var_values.get(self.var_key)
        assert var_values.tangent_offset(self.var_key) == 0
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
        spec_tensors = var.spec_tensors
        wp.launch(
            kernel=_interaction_mesh_jacobian_kernel,
            dim=(var.batch_size, self.num_vertices, var.tangent_dim),
            inputs=[
                var.S_world,
                var.T_world_link,
                var.T_world_base,
                self._vertex_link_indices_wp,
                self._neighbor_offsets_wp,
                self._neighbor_indices_wp,
                self._neighbor_weights_wp,
                spec_tensors.link_ancestor_joints_mask,
                spec_tensors.joints_to_actuated_mapping,
                self.weight,
                base_dim,
                row_offset,
            ],
            outputs=[out_jacobian],
            device=kernel_device,
        )
        return out_jacobian


__all__ = ["InteractionMeshTask"]
