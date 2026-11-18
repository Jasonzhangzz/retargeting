# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportOperatorIssue=false
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import warp as wp

from robokit.lie.se3_kernels import se3_adjoint_func
from robokit.opt.var_values import VarValues
from robokit.robo.robot import Robot
from robokit.terms.robot_task import RobotTask
from robokit.terms.task import Context, ResidualTask
from robokit.utils.warp_utils import resolve_device, wp_device_type, wp_vec6, wp_vec7
from robokit.xform.warp.transforms import project_point_func


class PixelTask(RobotTask, ResidualTask):
    """Match the image pixels of link origins seen by a fixed pinhole camera (`intrinsic`, `T_camera_world`); the residual is `weight * (uv - project_points(transform_points(p_world_link, T_camera_world), intrinsic))` in pixels (OpenCV camera axes; identity `T_camera_world` by default); a point at or behind the camera is projected at depth 1e-3 m; `uv` is `(batch, frames, 2)` and is shared by the candidates; `set_target` refills it in place, so a built solver sees new pixels; `frame_index` may be `IKContext.TARGET_LINKS`.

    Example:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.opt.var_values import VarKey, VarValues
        >>> robot = Robot.load(load_robot_description("ur10_description"))
        >>> link = robot.link_names.index("ee_link")
        >>> K, T_camera_world = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]]), np.eye(4)
        >>> T_camera_world[2, 3] = 2.0  # the robot 2 m in front of the camera
        >>> task = PixelTask(robot, link, np.zeros((1, 1, 2)), K, T_camera_world, weight=0.05, device="cpu")
        >>> state = robot.state(q=wp.zeros((1, 6), dtype=wp.float32, device="cpu"))
        >>> task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).shape
        (1, 2)
    """

    def __init__(
        self,
        robot: Optional[Robot] = None,
        frame_index: Optional[Union[int, Sequence[int], Context]] = None,
        uv: Optional[np.ndarray] = None,
        intrinsic: Optional[np.ndarray] = None,
        T_camera_world: Optional[np.ndarray] = None,
        weight: Union[float, Sequence[float]] = 0.05,
        name: str = "",
        device: Optional[wp_device_type] = None,
    ):
        if uv is None or intrinsic is None:
            raise ValueError("PixelTask needs uv (batch, frames, 2) and a 3x3 intrinsic")
        uv, intrinsic = np.asarray(uv, np.float32), np.asarray(intrinsic, np.float32)
        T = np.eye(4, dtype=np.float32) if T_camera_world is None else np.asarray(T_camera_world, np.float32)
        if uv.ndim != 3 or uv.shape[2] != 2 or intrinsic.shape != (3, 3) or T.shape != (4, 4):
            raise ValueError(f"uv must be (batch, frames, 2), intrinsic 3x3, T_camera_world 4x4; got {uv.shape}, "
                             f"{intrinsic.shape}, {T.shape}")  # fmt: skip
        self.name, self.weight, self.robot, self.frame_index = name, weight, None, frame_index
        self.num_frames, self.device = uv.shape[1], resolve_device(device)
        self.uv = wp.array(uv, dtype=wp.vec2, device=self.device)
        self.K = wp.array([intrinsic], dtype=wp.mat33, device=self.device)
        self.T = wp.array([T], dtype=wp.mat44, device=self.device)
        self.residual_weight = wp.zeros(2 * self.num_frames, dtype=wp.float32, device=self.device)
        self.set_weight(weight)
        if robot is not None and not isinstance(frame_index, Context):
            self.resolve(None, {})  # pyright: ignore[reportArgumentType]
            self.set_robot(robot)

    def set_weight(self, weight: Union[float, Sequence[float]]):
        """Update residual weights in-place: scalar or per frame."""
        values = np.broadcast_to(np.asarray(weight, np.float32), (self.num_frames,))
        wp.copy(self.residual_weight, wp.array(np.repeat(values, 2), dtype=wp.float32, device="cpu"))

    def set_target(self, uv: np.ndarray):
        """Refill target pixels `(batch, frames, 2)` in place."""
        wp.copy(self.uv, wp.array(np.asarray(uv, np.float32), dtype=wp.vec2, device="cpu"))

    def resolve(self, var_values: VarValues, context: Mapping[Context, Any]):
        if var_values is not None:
            super().resolve(var_values, context)
        frames = np.atleast_1d(np.asarray(self.frame_index, np.int32))
        if frames.shape != (self.num_frames,):
            raise ValueError(f"{len(frames)} frame indices for {self.num_frames} pixel targets")
        self.frame_indices = wp.array(frames, dtype=wp.int32, device=self.device)

    @property
    def residual_dim(self) -> int:
        return 2 * self.num_frames

    def compute_weighted_residual(
        self, var_values: VarValues, out_residual: Optional[wp.array] = None, row_offset: int = 0
    ) -> wp.array:
        return self._launch(var_values, out_residual, row_offset, False)

    def compute_weighted_jacobian_analytic(
        self, var_values: VarValues, out_jacobian: Optional[wp.array] = None, row_offset: int = 0
    ) -> wp.array:
        return self._launch(var_values, out_jacobian, row_offset, True)

    def _launch(self, var_values: VarValues, out: Optional[wp.array], row_offset: int, jacobian: bool) -> wp.array:
        var = var_values.get(self.var_key)
        if not var.is_fk_computed:
            self.robot.forward_kinematics(var)
        if jacobian and not var.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(var)
        device = out.device if out is not None else self.device
        shape = (var.batch_size, self.residual_dim) + ((var.tangent_dim,) if jacobian else ())
        if out is None:
            out, row_offset = wp.zeros(shape, dtype=wp.float32, device=device), 0
        spec = var.spec_tensors
        num_actuated = spec.joints_to_actuated_mapping.shape[1]
        inputs = [var.T_world_link, self.frame_indices, self.uv, self.K, self.T, self.residual_weight, row_offset]
        if not jacobian:
            wp.launch(_pixel_residual_kernel, dim=(var.batch_size, self.num_frames), inputs=inputs, outputs=[out],
                      device=device)  # fmt: skip
            return out
        inputs += [var.S_world, var.T_world_base, spec.link_ancestor_joints_mask, spec.joints_to_actuated_mapping,
                   var.has_floating_base, var_values.tangent_offset(self.var_key)]  # fmt: skip
        wp.launch(_pixel_jacobian_kernel, dim=(var.batch_size, self.num_frames, num_actuated), inputs=inputs,
                  outputs=[out], device=device)  # fmt: skip
        return out


@wp.kernel
def _pixel_residual_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),
    frame_indices: wp.array1d(dtype=wp.int32),
    uv: wp.array2d(dtype=wp.vec2),
    K: wp.array1d(dtype=wp.mat33),
    T: wp.array1d(dtype=wp.mat44),
    weight: wp.array1d(dtype=wp.float32),
    row_offset: int,
    residual: wp.array2d(dtype=wp.float32),
):
    idx, f = wp.tid()  # pyright: ignore
    p = T_world_link[idx, frame_indices[f]]
    pixel = project_point_func(wp.transform_point(T[0], wp.vec3(p[0], p[1], p[2])), K[0])  # pyright: ignore[reportCallIssue]
    error = uv[idx // wp.max(1, T_world_link.shape[0] // uv.shape[0]), f] - pixel
    for i in range(2):
        residual[idx, row_offset + 2 * f + i] = weight[2 * f + i] * error[i]


@wp.kernel
def _pixel_jacobian_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),
    frame_indices: wp.array1d(dtype=wp.int32),
    uv: wp.array2d(dtype=wp.vec2),
    K: wp.array1d(dtype=wp.mat33),
    T: wp.array1d(dtype=wp.mat44),
    weight: wp.array1d(dtype=wp.float32),
    row_offset: int,
    S_world: wp.array3d(dtype=wp.float32),
    T_world_base: wp.array1d(dtype=wp_vec7),
    ancestor_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    has_floating_base: wp.bool,
    col_offset: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    instance, f, actuated = wp.tid()  # pyright: ignore
    frame = frame_indices[f]
    T_world_frame = T_world_link[instance, frame]
    p = wp.vec3(T_world_frame[0], T_world_frame[1], T_world_frame[2])
    p_camera = wp.transform_point(T[0], p)  # pyright: ignore[reportCallIssue]
    pixel, s = project_point_func(p_camera, K[0]), wp.max(p_camera[2], 1e-3)
    column = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for j in range(S_world.shape[2]):
        if ancestor_mask[frame, j] and joints_to_actuated[j, actuated] != 0.0:
            for r in range(6):
                column[r] += S_world[instance, r, j] * joints_to_actuated[j, actuated]
    base = 6 if has_floating_base else 0
    for k in range(2):
        if k == 1 and not (has_floating_base and actuated < 6):
            continue
        if k == 1:  # the floating base's twist about the same column
            unit = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            unit[actuated] = 1.0
            column = se3_adjoint_func(T_world_base[instance]) * unit
        v = wp.vec3(column[0], column[1], column[2]) + wp.cross(wp.vec3(column[3], column[4], column[5]), p)
        d = wp.transform_vector(T[0], v)
        if p_camera[2] <= 1e-3:
            d[2] = 0.0  # the clamped depth does not move
        dh = K[0] * d
        col = col_offset + (actuated if k == 1 else base + actuated)
        for i in range(2):
            jacobian[instance, row_offset + 2 * f + i, col] = -weight[2 * f + i] * (dh[i] - pixel[i] * dh[2]) / s
