# pyright: reportArgumentType=false, reportIndexIssue=false
import copy
from typing import Any, Mapping, Optional, Sequence, Union, cast

import numpy as np
import warp as wp

from robokit.opt.var_values import VarValues
from robokit.terms.task import Context, ResidualTask


@wp.kernel
def _threshold_score_kernel(
    rows: wp.array2d(dtype=wp.float32),
    thresholds: wp.array1d(dtype=wp.float32),
    num_candidates: int,
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),
    valid_count: wp.array1d(dtype=wp.int32),
):
    i = wp.tid()
    valid = int(1)
    score = float(0.0)
    for j in range(rows.shape[1]):
        if thresholds[j] < 0.0:
            score += rows[i, j] * rows[i, j]
        elif not (wp.abs(rows[i, j]) <= thresholds[j]):
            valid = int(0)
    if valid != 0:
        wp.atomic_add(valid_count, i // num_candidates, 1)  # pyright: ignore[reportOperatorIssue]
        out_residual[i, row_offset] = wp.sqrt(score)
    else:
        out_residual[i, row_offset] = wp.inf


class CompositeTask(ResidualTask):
    """Concatenate weighted residuals, with optional threshold-based selection scoring.

    Modes:
    - Without `thresholds`: ordinary residual concatenation for optimization.
    - With `thresholds`: bounded children are constraints; children with None thresholds are scores.

    Selection sequence:
    1. Require each constraint's absolute residuals to satisfy scalar, per-row or repeating bounds.
    2. Return score norm for valid rows and infinity otherwise, preserving ordinary squared costs on valid rows.
    3. Count valid rows per `num_candidates` group in `valid_count`; use stable selection to preserve ties.

    Example:
        >>> from robokit.terms.dense.configuration_task import ConfigurationTask
        >>> task = CompositeTask([ConfigurationTask()], thresholds=[None])
        >>> task.residual_dim
        1
    """

    def __init__(
        self,
        tasks: Sequence[ResidualTask],
        *,
        thresholds: Optional[Sequence[Optional[Union[float, Sequence[float]]]]] = None,
        num_candidates: Union[int, Context] = 1,
    ):
        self.tasks = list(tasks)
        self.thresholds = None if thresholds is None else list(thresholds)
        self.num_candidates = cast(int, num_candidates)  # a placeholder until `resolve`
        if isinstance(num_candidates, int) and num_candidates < 1:
            raise ValueError("num_candidates must be positive")
        if thresholds is not None and len(thresholds) != len(tasks):
            raise ValueError("thresholds must have one entry per task")
        self.SUPPORTS_CUDA_GRAPH = all(task.SUPPORTS_CUDA_GRAPH for task in tasks)
        self._rows = None
        self.valid_count: Optional[wp.array] = None

    def resolve(self, var_values: VarValues, context: Mapping[Context, Any]):
        """Build a copy of each unbuilt child, dropping one the context empties, and take the candidate count."""
        super().resolve(var_values, context)
        kept = []
        for task, threshold in zip(self.tasks, self.thresholds or [None] * len(self.tasks)):
            if task.robot is None:
                built = copy.copy(task)
                built.resolve(var_values, context)
                if task.active and not built.active:  # e.g. collision in a scene without geometry
                    continue
                task = built
            kept.append((task, threshold))
        thresholds = None if self.thresholds is None else [threshold for _, threshold in kept]
        CompositeTask.__init__(
            self, [task for task, _ in kept], thresholds=thresholds, num_candidates=self.num_candidates
        )

    @property
    def residual_dim(self) -> int:
        return 1 if self.thresholds is not None else sum(task.residual_dim for task in self.tasks)

    def precompute(self, var_values: VarValues, need_gradient: bool = False):
        """Prepare every child task."""
        if self.thresholds is not None:
            if need_gradient:
                raise ValueError("Threshold scoring is for selection, not optimization")
            return  # prepare once, after invalidation in compute_weighted_residual
        for task in self.tasks:
            task.precompute(var_values, need_gradient=need_gradient)

    def compute_weighted_residual(
        self,
        var_values: VarValues,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        if self.thresholds is not None:
            batch = var_values.batch_size
            if batch % self.num_candidates:
                raise ValueError("batch size must be divisible by num_candidates")
            if self._rows is None or self._rows.shape[0] != batch:
                limits = []
                for task, threshold in zip(self.tasks, self.thresholds):
                    bound = np.full(task.residual_dim, -1.0, dtype=np.float32)
                    if threshold is not None:  # a sequence repeats over the rows, e.g. (position, angle) per link
                        bound[:] = np.resize(np.asarray(threshold, dtype=np.float32), task.residual_dim)
                        if np.any(bound < 0) or np.any(np.isnan(bound)):
                            raise ValueError("constraint thresholds must be nonnegative")
                    limits.extend(bound)
                self._limits = wp.array(limits, dtype=wp.float32, device=var_values.device)
                self._rows = wp.empty((batch, len(limits)), dtype=wp.float32, device=var_values.device)
                self.valid_count = wp.zeros(batch // self.num_candidates, dtype=wp.int32, device=var_values.device)
            if out_residual is None:
                out_residual = wp.empty((batch, 1), dtype=wp.float32, device=var_values.device)
                row_offset = 0
            var_values.invalidate()  # constraints must inspect the final values, including fresh derived state
            offset = 0
            for task in self.tasks:
                task.precompute(var_values, need_gradient=False)
                task.compute_weighted_residual(var_values, out_residual=self._rows, row_offset=offset)
                offset += task.residual_dim
            self.valid_count.zero_()
            wp.launch(
                _threshold_score_kernel,
                dim=batch,
                inputs=[self._rows, self._limits, self.num_candidates, row_offset],
                outputs=[out_residual, self.valid_count],
                device=var_values.device,
            )
            return out_residual
        if out_residual is None:
            raise ValueError("out_residual is required.")
        for task in self.tasks:
            task.compute_weighted_residual(var_values, out_residual=out_residual, row_offset=row_offset)
            row_offset += task.residual_dim
        return out_residual
