# pyright: reportArgumentType=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportIndexIssue=false
"""Replay a per-pose dense task at every frame of a trajectory."""

import copy
from typing import Any, Dict, Mapping, Optional, Sequence, Union, cast

import numpy as np
import warp as wp

from robokit.opt.var_values import VarValues
from robokit.robo.robot_state import RobotState
from robokit.terms.dense.frame_task import FrameTask
from robokit.terms.robot_task import RobotTask
from robokit.terms.task import (
    Context,
    GradientTask,
    ResidualTask,
    SparseTask,
    SparsityPattern,
    Task,
    sparse_cost_and_gradient,
)
from robokit.utils.warp_utils import wp_device_type


# --- kernels ----------------------------------------------------------------
@wp.kernel
def _pattern_kernel(
    frames: wp.array1d(dtype=wp.int32),
    residual_dim: int,
    tangent_dim: int,
    row_offset: int,
    nnz_offset: int,
    row_indices: wp.array1d(dtype=wp.int32),
    col_indices: wp.array1d(dtype=wp.int32),
):
    slot, r, d = wp.tid()  # type: ignore
    row = slot * residual_dim + r
    row_indices[nnz_offset + row * tangent_dim + d] = row_offset + row
    col_indices[nnz_offset + row * tangent_dim + d] = frames[slot] * tangent_dim + d


@wp.kernel
def _select_kernel(
    src: wp.array2d(dtype=wp.float32),
    frames: wp.array1d(dtype=wp.int32),
    block: int,
    col_offset: int,
    frame_mask: wp.array2d(dtype=wp.uint8),
    use_mask: wp.bool,
    dst: wp.array2d(dtype=wp.float32),
):
    batch_idx, slot, i = wp.tid()  # type: ignore
    frame_idx = frames[slot]
    value = src[batch_idx, frame_idx * block + i]
    if use_mask:
        if frame_mask[batch_idx, frame_idx] == wp.uint8(0):
            value = 0.0
    dst[batch_idx, col_offset + slot * block + i] = value


# --- task -------------------------------------------------------------------
class TrajectoryTask(RobotTask, SparseTask, GradientTask):
    """Apply independent per-pose tasks to trajectory frames, with one dense tangent block per selected frame.

    Args:
        dense_task: Per-pose residual task.
        num_frames: Number of trajectory frames, or omitted so the trajectory optimizer sets it.
        frame_indices: Frames to include. Defaults to all frames.
        name: Optional planner input name (start, goal, target, path_goal, or path_target).
    """

    compute_weighted_cost_and_gradient = sparse_cost_and_gradient
    # the wrapper owns the frames; these setters belong to the per-pose task and are forwarded to it
    _FORWARD = ("set_target", "set_weight", "set_frames")

    def __getattr__(self, name: str) -> Any:
        if name not in TrajectoryTask._FORWARD or "_dense_task" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self._dense_task, name)

    @property
    def dense_task(self) -> ResidualTask:
        return self._dense_task

    def __init__(
        self,
        dense_task: ResidualTask,
        num_frames: Optional[int] = None,
        frame_indices: Optional[Union[Sequence[int], Context]] = None,
        name: str = "",
    ):
        self._dense_task = dense_task
        self.frames = frame_indices
        if num_frames is not None:
            self.num_frames = num_frames
            self.frames = list(range(num_frames)) if frame_indices is None else [f % num_frames for f in frame_indices]
        self.name = name
        self.var_key = dense_task.var_key
        self.robot = dense_task.robot
        self.precompute_collision_geometry = (
            dense_task.precompute_collision_geometry if isinstance(dense_task, RobotTask) else ""
        )
        self.device: Optional[wp_device_type] = None
        self._frames_wp: Optional[wp.array] = None
        # Cache per batch width; alternating stage sizes otherwise reallocate inside CUDA capture and break replay.
        self._residual_bufs: Dict[int, wp.array] = {}
        self._jacobian_bufs: Dict[int, wp.array] = {}

    @property
    def active(self) -> bool:
        return self._dense_task.active and self.frames != []

    def resolve(self, var_values: VarValues, context: Mapping[Context, Any]):
        """Build for the trajectory in `var_values`; a built per-pose task keeps its identity and live targets."""
        Task.resolve(self, var_values, context)
        state = cast(RobotState, var_values.get(self.var_key))
        dense = self._dense_task
        if dense.robot is None:
            dense = copy.deepcopy(dense)
            dense.resolve(var_values, context)
        TrajectoryTask.__init__(self, dense, state.q.shape[1], self.frames, self.name)
        # a single-link pose cost on the last frame evaluates each trajectory's last frame only
        if isinstance(dense, FrameTask) and dense.num_frames == 1 and self.frames == [self.num_frames - 1]:
            dense.trajectory_frames = self.num_frames
        self.init_buffers(state.q.device)  # allocate now: per-query inputs may be written inside graph capture

    def init_buffers(self, device: wp_device_type):
        if self.device is not None:
            if wp.get_device(self.device) != wp.get_device(device):
                raise ValueError("A built trajectory task cannot move between devices.")
            return
        self.device = device
        self._frames_wp = wp.from_numpy(np.array(self.frames, dtype=np.int32), dtype=wp.int32, device=device)
        if self.dense_task.device is None:
            self.dense_task.init_buffers(device)

    @property
    def residual_dim(self) -> int:
        return len(self.frames) * self.dense_task.residual_dim

    def _build_flat_var_values(self, state: Any) -> VarValues:
        if self.device is None:
            self.init_buffers(state.q.device)
        return VarValues.from_var(state.flatten(), self.var_key)

    def fill_residual_and_jacobian(
        self,
        var_values: VarValues,
        out_residual: wp.array,
        row_offset: int,
        out_jacobian: Optional[wp.array] = None,
        nnz_offset: int = 0,
    ):
        """A terminal cost whose dense task writes the trajectory layout itself (`fill_terminal`) takes one launch."""
        state, dense = var_values.get(self.var_key), self.dense_task
        terminal = (
            isinstance(dense, FrameTask)
            and dense.residual_mode != "distance"
            and dense.trajectory_frames == self.num_frames
            and self.frames == [self.num_frames - 1]
            and not self.use_mask
            and not state.has_floating_base
        )
        if not terminal:
            super().fill_residual_and_jacobian(
                var_values, out_residual, row_offset, out_jacobian=out_jacobian, nnz_offset=nnz_offset
            )
            return
        dense.fill_terminal(
            self._build_flat_var_values(state),
            out_residual,
            row_offset,
            out_jacobian=out_jacobian,
            nnz_offset=nnz_offset,
        )

    def compute_weighted_residual(
        self,
        var_values: VarValues,
        *args: Any,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
        **kwargs: Any,
    ) -> wp.array:
        state = var_values.get(self.var_key)
        flat = self._build_flat_var_values(state)
        batch_size, rows = state.batch_size, self.dense_task.residual_dim
        device = state.q.device

        if out_residual is None:
            out_residual = wp.empty((batch_size, self.residual_dim), dtype=wp.float32, device=device)
            row_offset = 0
        residual_buf = self._residual_bufs.get(flat.batch_size)
        if residual_buf is None:
            residual_buf = wp.zeros(
                (flat.batch_size, rows), dtype=wp.float32, device=device, requires_grad=state.q.requires_grad
            )
            self._residual_bufs[flat.batch_size] = residual_buf

        self.dense_task.compute_weighted_residual(flat, out_residual=residual_buf)
        wp.launch(
            _select_kernel,
            dim=(batch_size, len(self.frames), rows),
            inputs=[
                residual_buf.reshape((batch_size, self.num_frames * rows)),
                self._frames_wp,
                rows,
                row_offset,
                self._get_frame_mask(),
                self.use_mask,
            ],
            outputs=[out_residual],
            device=device,
        )
        return out_residual

    def compute_sparse_jacobian_pattern(
        self,
        var_values: VarValues,
        *args: Any,
        offset: int = 0,
        out_row_indices: Optional[wp.array] = None,
        out_col_indices: Optional[wp.array] = None,
        nnz_offset: int = 0,
        **kwargs: Any,
    ) -> SparsityPattern:
        state = var_values.get(self.var_key)
        if self.device is None:
            self.init_buffers(state.q.device)
        device = state.q.device
        rows = self.dense_task.residual_dim
        tangent_dim = (6 if state.has_floating_base else 0) + self.robot.spec.num_actuated_joints
        nnz = len(self.frames) * rows * tangent_dim

        pattern = SparsityPattern()
        pattern.row_indices = (
            wp.empty(nnz, dtype=wp.int32, device=device) if out_row_indices is None else out_row_indices
        )
        pattern.col_indices = (
            wp.empty(nnz, dtype=wp.int32, device=device) if out_col_indices is None else out_col_indices
        )
        wp.launch(
            _pattern_kernel,
            dim=(len(self.frames), rows, tangent_dim),
            inputs=[self._frames_wp, rows, tangent_dim, offset, nnz_offset],
            outputs=[pattern.row_indices, pattern.col_indices],
            device=device,
        )
        return pattern

    def compute_weighted_sparse_jacobian_values(
        self,
        var_values: VarValues,
        *args: Any,
        out_jacobian_values: Optional[wp.array] = None,
        offset: int = 0,
        nnz_offset: int = 0,
        **kwargs: Any,
    ) -> wp.array:
        state = var_values.get(self.var_key)
        flat = self._build_flat_var_values(state)
        batch_size, rows = state.batch_size, self.dense_task.residual_dim
        tangent_dim = flat.tangent_dim
        block = rows * tangent_dim
        device = state.q.device

        if out_jacobian_values is None:
            out_jacobian_values = wp.empty((batch_size, len(self.frames) * block), dtype=wp.float32, device=device)
            nnz_offset = 0
        jacobian_buf = self._jacobian_bufs.get(flat.batch_size)
        if jacobian_buf is None:
            jacobian_buf = wp.zeros((flat.batch_size, rows, tangent_dim), dtype=wp.float32, device=device)
            self._jacobian_bufs[flat.batch_size] = jacobian_buf

        # dense kernels only write their nonzeros, so the block starts clean every call
        jacobian_buf.zero_()
        self.dense_task.compute_weighted_jacobian_analytic(flat, out_jacobian=jacobian_buf)
        wp.launch(
            _select_kernel,
            dim=(batch_size, len(self.frames), block),
            inputs=[
                jacobian_buf.reshape((batch_size, self.num_frames * block)),
                self._frames_wp,
                block,
                nnz_offset,
                self._get_frame_mask(),
                self.use_mask,
            ],
            outputs=[out_jacobian_values],
            device=device,
        )
        return out_jacobian_values


__all__ = ["TrajectoryTask"]
