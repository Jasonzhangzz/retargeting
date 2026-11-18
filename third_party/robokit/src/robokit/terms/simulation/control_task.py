# pyright: reportGeneralTypeIssues=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false
# pyright: reportOperatorIssue=false
"""Penalize the applied controls, or their offsets from a reference."""

from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import warp as wp

from robokit.opt.var_values import VarValues
from robokit.terms.simulation.sim_context import SimContext, SimResidualTask, _get_instance_step_func, _instanced
from robokit.terms.task import Context


@wp.kernel
def _compute_control_kernel(
    ctrl: wp.array2d(dtype=float),
    ctrl_ref: wp.array3d(dtype=float),
    weight: wp.array(dtype=float),
    norm: bool,
    clock: wp.array(dtype=int),
    worlds: int,
    world_steps: wp.array(dtype=wp.vec2i),
    row_offset: int,
    out: wp.array2d(dtype=float),
    out_cost: wp.array(dtype=float),
):
    """`sqrt(2 weight) (ctrl - ctrl_ref[k])` per control, or with `norm` one `sqrt(2 cost)`; or the cost, if wanted."""
    w = wp.tid()
    bt = _get_instance_step_func(w, worlds, clock, world_steps)
    r = wp.min(clock[0] + bt[1], ctrl_ref.shape[1] - 1)
    cost_only = out_cost.shape[0] > 0
    total = float(0.0)
    for u in range(ctrl.shape[1]):
        e = ctrl[w, u] - ctrl_ref[bt[0], r, u]
        total += wp.where(norm, weight[u] * weight[u], weight[u]) * e * e
        if not norm and not cost_only:
            out[w, row_offset + u] = wp.sqrt(2.0 * weight[u]) * e
    value = wp.where(norm, wp.sqrt(total), total)
    if cost_only:
        out_cost[w] += value
    elif norm:
        out[w, row_offset] = wp.sqrt(2.0 * value)


class ControlTask(SimResidualTask):
    """Per step, `sum(weight * (ctrl - ctrl_ref[k])**2)`, or with `norm` its unsquared norm; `ctrl_ref` defaults to 0.

    Example:
        >>> context = {
        ...     SimContext.CLOCK: wp.array([0, 0, 1], dtype=int, device="cpu"),
        ...     SimContext.NUM_INSTANCES: 1,
        ...     SimContext.WORLDS_PER_INSTANCE: 1,
        ...     SimContext.CTRL: wp.array([[1.0, 2.0]], dtype=float, device="cpu"),
        ... }
        >>> out = wp.zeros(2, dtype=float, device="cpu")
        >>> for i, term in enumerate([ControlTask([0.5, 1.0]), ControlTask(2.0, np.ones((10, 2)), norm=True)]):
        ...     term.resolve(None, context)
        ...     term.compute_weighted_cost(None, out_cost=out[i : i + 1])
        >>> out.numpy().tolist()
        [4.5, 2.0]
    """

    def __init__(
        self,
        weight: Union[float, Sequence[float]],
        ctrl_ref: Optional[np.ndarray] = None,
        norm: bool = False,
    ):
        self.weight, self.ctrl_ref, self.norm = weight, ctrl_ref, norm
        self.ctrl, self.clock = SimContext.CTRL, SimContext.CLOCK
        self.instances, self.worlds = SimContext.NUM_INSTANCES, SimContext.WORLDS_PER_INSTANCE

    def resolve(self, var_values: Optional[VarValues], context: Mapping[Context, Any]):
        """Broadcast the weight to every control and build the device reference."""
        super().resolve(var_values, context)
        nu, device = self.ctrl.shape[1], self.ctrl.device
        self.weight = wp.array(np.broadcast_to(self.weight, nu).astype(np.float32), dtype=float, device=device)
        ctrl_ref = np.zeros((1, nu)) if self.ctrl_ref is None else self.ctrl_ref
        self.ctrl_ref = _instanced(ctrl_ref, 2, self.instances, device)
        self._residual_dim = 1 if self.norm else nu
        self._resolve_world_steps(context, device)

    def _launch(self, out_residual: wp.array, row_offset: int, out_cost: wp.array):
        inputs = [self.ctrl, self.ctrl_ref, self.weight, self.norm, self.clock, self.worlds, self.world_steps]
        wp.launch(
            _compute_control_kernel,
            dim=self.ctrl.shape[0],
            inputs=[*inputs, row_offset],
            outputs=[out_residual, out_cost],
            device=self.ctrl.device,
        )
