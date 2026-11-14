# pyright: reportGeneralTypeIssues=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false
# pyright: reportOperatorIssue=false
"""The values `SimMPC` and `ILQROptimizer` give simulation terms, and the terms' shared parts."""

from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

import numpy as np
import warp as wp

from robokit.opt.var_values import VarValues
from robokit.terms.task import Context, ResidualTask


if TYPE_CHECKING:
    from robokit.sim import Sim


class SimContext(Context):
    """Values a planner gives its terms for one world set (`sim_context`); arrays are refilled every step."""

    CLOCK = 0
    """Device ints `[t0, t, horizon]`: the rollout started at sim step `t0` and has just taken step `t`."""
    NUM_INSTANCES = 1
    """Instances planned at once."""
    WORLDS_PER_INSTANCE = 2
    """Consecutive worlds of one instance: world `w` belongs to instance `w // WORLDS_PER_INSTANCE`."""
    QPOS = 3
    """`(nworld, nq)` positions."""
    QVEL = 4
    """`(nworld, nv)` velocities."""
    CTRL = 5
    """`(nworld, nu)` controls applied during the step."""
    SENSORDATA = 6
    """`(nworld, nsensordata)` sensor readings, evaluated before integrating."""
    BODY_POSITION = 7
    """`(nworld, nbody)` vec3 body positions in the world frame."""
    BODY_ROTATION = 8
    """`(nworld, nbody)` mat33 body rotations, world from body."""
    CONTACTS = 9
    """Device arrays `(count, world, geom, dist, position, frame)` of the contact slots (frame: normal first); a negative geom pair is empty."""
    JOINTS = 10
    """Device ints `(type, qpos_address, dof_address)` per joint; MuJoCo's types: 0 free, 1 ball, 2 slide, 3 hinge."""
    SENSORS = 11
    """`(name, address, dimension)` of every sensor, in `sensordata` order."""
    BODIES = 12
    """Body names by id."""
    GEOMS = 13
    """Geom names by id."""
    WORLD_STEPS = 14
    """Optional device vec2i `(instance, step)` per world, for worlds of different instances and steps (iLQR's finite differences); without it, world `w` is instance `w // WORLDS_PER_INSTANCE` at step `CLOCK[1]`."""


@wp.func
def _get_instance_step_func(
    w: int, worlds: int, clock: wp.array(dtype=int), world_steps: wp.array(dtype=wp.vec2i)
) -> wp.vec2i:
    """World `w`'s instance and step since the rollout's start (`SimContext.WORLD_STEPS`)."""
    if world_steps.shape[0] > 0:
        return world_steps[w]
    return wp.vec2i(w // worlds, clock[1])


def sim_context(sim: "Sim") -> Dict[Context, Any]:
    """The `SimContext` values of a simulator's worlds, with its `mjm`'s layout, except `CLOCK`, `NUM_INSTANCES` and `WORLDS_PER_INSTANCE`."""
    data = sim.data
    model = sim.mjm
    device = data.qpos.device
    return {
        SimContext.QPOS: data.qpos,
        SimContext.QVEL: data.qvel,
        SimContext.CTRL: data.ctrl,
        SimContext.SENSORDATA: data.sensordata,
        SimContext.BODY_POSITION: data.xpos,
        SimContext.BODY_ROTATION: data.xmat,
        SimContext.JOINTS: tuple(
            wp.array(np.asarray(x, np.int32), dtype=int, device=device)
            for x in (model.jnt_type, model.jnt_qposadr, model.jnt_dofadr)
        ),
        SimContext.SENSORS: [
            (model.sensor(i).name, int(model.sensor_adr[i]), int(model.sensor_dim[i])) for i in range(model.nsensor)
        ],
        SimContext.BODIES: [model.body(i).name for i in range(model.nbody)],
        SimContext.GEOMS: [model.geom(i).name for i in range(model.ngeom)],
        SimContext.CONTACTS: (
            data.nacon,
            *(getattr(data.contact, k) for k in ["worldid", "geom", "dist", "pos", "frame"]),
        ),
    }


class SimResidualTask(ResidualTask):
    """A simulation term with weighted residuals: `ILQROptimizer` reads them, `SimMPC` their cost `0.5 |r|^2`, which the term's kernel adds up directly; subclasses set `_residual_dim` and call `_resolve_world_steps` in `resolve`, then implement `_launch`: one kernel over the context's worlds that writes the residuals, or with a nonempty `out_cost`, adds each world's cost."""

    _residual_dim: int = 0

    @property
    def residual_dim(self) -> int:
        return self._residual_dim

    def _resolve_world_steps(self, context: Mapping[Context, Any], device: Any):
        """Each world's instance and step (`SimContext.WORLD_STEPS`, else none), and the kernel's unused outputs."""
        steps = context.get(SimContext.WORLD_STEPS)
        self.world_steps = steps if steps is not None else wp.zeros(0, dtype=wp.vec2i, device=device)
        self._no_cost, self._no_residual = wp.zeros(0, device=device), wp.zeros((0, 0), device=device)

    def _launch(self, out_residual: wp.array, row_offset: int, out_cost: wp.array):
        raise NotImplementedError

    def compute_weighted_residual(
        self, var_values: Optional[VarValues], *, out_residual: wp.array, row_offset: int = 0
    ) -> wp.array:
        self._launch(out_residual, row_offset, self._no_cost)
        return out_residual

    def compute_weighted_cost(self, var_values: Optional[VarValues], *, out_cost: wp.array):
        """Add each world's `0.5 |r|^2`."""
        self._launch(self._no_residual, 0, out_cost)


def _instanced(x: Any, ndim: int, instances: int, device: Any, dtype: Any = float) -> wp.array:
    """`x` with a leading instance axis on `device`; an array of `ndim` axes is shared by every instance; a Warp array that already has the instance axis is used in place, so its owner can rewrite it between plans."""
    if isinstance(x, wp.array) and x.ndim == ndim + 1 and x.shape[0] == instances:
        return x
    x = np.asarray(x, np.float32)
    x = x if x.ndim > ndim else x[None]
    return wp.array(np.ascontiguousarray(np.broadcast_to(x, (instances, *x.shape[1:]))), dtype=dtype, device=device)
