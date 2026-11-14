# pyright: reportGeneralTypeIssues=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false
# pyright: reportOperatorIssue=false
"""Point one sensor from another toward a target."""

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import warp as wp

from robokit.opt.var_values import VarValues
from robokit.terms.simulation.sim_context import SimContext, SimResidualTask, _get_instance_step_func, _instanced
from robokit.terms.task import Context


@wp.kernel
def _compute_align_kernel(
    sensordata: wp.array2d(dtype=float),
    target_ref: wp.array3d(dtype=float),
    origin: int,
    sensor: int,
    components: wp.array(dtype=int),
    weight: float,
    freeze_reference: bool,
    clock: wp.array(dtype=int),
    worlds: int,
    world_steps: wp.array(dtype=wp.vec2i),
    row_offset: int,
    out: wp.array2d(dtype=float),
    out_cost: wp.array(dtype=float),
):
    w = wp.tid()
    bt = _get_instance_step_func(w, worlds, clock, world_steps)
    b = bt[0]
    r = wp.min(clock[0] + wp.where(freeze_reference, 0, bt[1]), target_ref.shape[1] - 1)
    uv, uu, vv = float(0.0), float(0.0), float(0.0)
    for j in range(components.shape[0]):
        i = components[j]
        u = sensordata[w, sensor + i] - sensordata[w, origin + i]
        v = target_ref[b, r, i] - sensordata[w, origin + i]
        uv += u * v
        uu += u * u
        vv += v * v
    cost = float(0.0)
    if uu >= 1.0e-12 and vv >= 1.0e-12:
        cost = weight * (1.0 + uv / wp.sqrt(uu * vv))
    if out_cost.shape[0] > 0:
        out_cost[w] += cost
    else:
        out[w, row_offset] = wp.sqrt(2.0 * wp.max(cost, 0.0))


class AlignTask(SimResidualTask):
    """Per step, `weight * (1 + u.v / (|u||v|))` with `u = sensor - origin` and `v = target_ref[k] - origin`.

    Example:
        >>> context = {
        ...     SimContext.CLOCK: wp.array([0, 0, 1], dtype=int, device="cpu"),
        ...     SimContext.NUM_INSTANCES: 1,
        ...     SimContext.WORLDS_PER_INSTANCE: 1,
        ...     SimContext.SENSORDATA: wp.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=float, device="cpu"),
        ...     SimContext.SENSORS: [("o", 0, 3), ("s", 3, 3)],
        ... }
        >>> term, out = AlignTask("o", "s", np.tile([0.0, 1.0, 0.0], (10, 1)), weight=2.0), wp.zeros(1, device="cpu")
        >>> term.resolve(None, context)
        >>> term.compute_weighted_cost(None, out_cost=out)
        >>> assert np.isclose(out.numpy()[0], 2.0)  # u = (1, 0) is perpendicular to v = (0, 1)
    """

    def __init__(
        self,
        origin: str,
        sensor: str,
        target_ref: np.ndarray,
        weight: float,
        components: Sequence[int] = (0, 1),
        freeze_reference: bool = False,
    ):
        self.origin, self.sensor, self.target_ref, self.weight = origin, sensor, target_ref, weight
        self.components, self.freeze_reference = components, freeze_reference
        self.sensordata, self.sensors, self.clock = SimContext.SENSORDATA, SimContext.SENSORS, SimContext.CLOCK
        self.instances, self.worlds = SimContext.NUM_INSTANCES, SimContext.WORLDS_PER_INSTANCE

    def resolve(self, var_values: Optional[VarValues], context: Mapping[Context, Any]):
        """Look up both sensors and build the device target."""
        super().resolve(var_values, context)
        table, device = {name: adr for name, adr, _ in self.sensors}, self.clock.device
        self.origin, self.sensor = int(table[self.origin]), int(table[self.sensor])
        self.target_ref = _instanced(self.target_ref, 2, self.instances, device)
        self.components = wp.array(np.asarray(self.components, np.int32), dtype=int, device=device)
        self._residual_dim = 1  # sqrt(2 cost)
        self._resolve_world_steps(context, device)

    def _launch(self, out_residual: wp.array, row_offset: int, out_cost: wp.array):
        inputs = [self.sensordata, self.target_ref, self.origin, self.sensor, self.components, self.weight]
        inputs += [self.freeze_reference, self.clock, self.worlds, self.world_steps, row_offset]
        wp.launch(
            _compute_align_kernel,
            dim=self.sensordata.shape[0],
            inputs=inputs,
            outputs=[out_residual, out_cost],
            device=self.sensordata.device,
        )
