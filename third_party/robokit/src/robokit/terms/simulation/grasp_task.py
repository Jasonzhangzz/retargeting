# pyright: reportGeneralTypeIssues=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false
# pyright: reportOperatorIssue=false
"""Keep a gripper body at an offset from an object."""

from typing import Any, Mapping, Optional, Union

import numpy as np
import warp as wp

from robokit.opt.var_values import VarValues
from robokit.terms.simulation.sim_context import SimContext, SimResidualTask, _get_instance_step_func, _instanced
from robokit.terms.task import Context


@wp.kernel
def _compute_grasp_kernel(
    body_position: wp.array2d(dtype=wp.vec3),
    body_rotation: wp.array2d(dtype=wp.mat33),
    clock: wp.array(dtype=int),
    worlds: int,
    world_steps: wp.array(dtype=wp.vec2i),
    gripper: int,
    obj: int,
    offset: wp.array2d(dtype=wp.vec3),
    weight: wp.array2d(dtype=float),
    tolerance: wp.array2d(dtype=float),
    object_frame: bool,
    row_offset: int,
    out: wp.array2d(dtype=float),
    out_cost: wp.array(dtype=float),
):
    w = wp.tid()
    bt = _get_instance_step_func(w, worlds, clock, world_steps)
    b = bt[0]
    r = wp.min(clock[0] + bt[1] + 1, weight.shape[1] - 1)
    relative = offset[b, r]
    if object_frame:
        relative = body_rotation[w, obj] @ relative
    distance = wp.length(body_position[w, gripper] - (body_position[w, obj] + relative))
    cost = weight[b, r] * (distance - wp.min(distance, tolerance[b, r]))
    if out_cost.shape[0] > 0:
        out_cost[w] += cost
    else:
        out[w, row_offset] = wp.sqrt(2.0 * wp.max(cost, 0.0))


class GraspTask(SimResidualTask):
    """Per step, `weight[k]` times the gripper's distance to body `obj` plus `offset[k]`, less `tolerance`, at least 0.

    Example:
        >>> context = {  # the hand at (0.1, 0, 0.2), a box at the origin turned 90 degrees about z
        ...     SimContext.CLOCK: wp.array([0, 0, 1], dtype=int, device="cpu"),
        ...     SimContext.NUM_INSTANCES: 1,
        ...     SimContext.WORLDS_PER_INSTANCE: 1,
        ...     SimContext.BODIES: ["world", "hand", "box"],
        ...     SimContext.BODY_POSITION: wp.array([[[0, 0, 0], [0.1, 0, 0.2], [0, 0, 0]]], dtype=wp.vec3, device="cpu"),
        ...     SimContext.BODY_ROTATION: wp.array(
        ...         [[np.eye(3), np.eye(3), [[0, -1, 0], [1, 0, 0], [0, 0, 1]]]], dtype=wp.mat33, device="cpu"
        ...     ),
        ... }
        >>> offset, out = np.tile([0.0, -0.1, 0.2], (2, 1)), wp.zeros(2, dtype=float, device="cpu")
        >>> terms = [GraspTask("hand", "box", offset, np.ones(2)), GraspTask(1, 2, offset, np.ones(2), 0.01, False)]
        >>> for i, term in enumerate(terms):  # in the box's frame the hand is on target
        ...     term.resolve(None, context)
        ...     term.compute_weighted_cost(None, out_cost=out[i : i + 1])
        >>> assert np.allclose(out.numpy(), [0.0, np.hypot(0.1, 0.1) - 0.01], atol=1e-6)
    """

    def __init__(
        self,
        gripper: Union[str, int],
        obj: Union[str, int],
        offset: np.ndarray,
        weight: np.ndarray,
        tolerance: Union[float, np.ndarray] = 0.0,
        object_frame: bool = True,
    ):
        self.gripper, self.obj, self.offset, self.weight = gripper, obj, offset, weight
        self.tolerance, self.object_frame = tolerance, object_frame
        self.body_position, self.body_rotation = SimContext.BODY_POSITION, SimContext.BODY_ROTATION
        self.bodies, self.clock = SimContext.BODIES, SimContext.CLOCK
        self.instances, self.worlds = SimContext.NUM_INSTANCES, SimContext.WORLDS_PER_INSTANCE

    def resolve(self, var_values: Optional[VarValues], context: Mapping[Context, Any]):
        """Look up both bodies and build the device schedule."""
        super().resolve(var_values, context)
        n, device = self.instances, self.clock.device
        self.gripper, self.obj = [
            self.bodies.index(b) if isinstance(b, str) else int(b) for b in (self.gripper, self.obj)
        ]
        if np.shape(self.offset)[-2] != np.shape(self.weight)[-1]:
            raise ValueError(f"offset has {np.shape(self.offset)[-2]} steps, weight {np.shape(self.weight)[-1]}")
        tolerance = np.broadcast_to(self.tolerance, np.shape(self.weight))
        self.offset = _instanced(self.offset, 2, n, device, wp.vec3)
        self.weight, self.tolerance = _instanced(self.weight, 1, n, device), _instanced(tolerance, 1, n, device)
        self._residual_dim = 1  # sqrt(2 cost)
        self._resolve_world_steps(context, device)

    def _launch(self, out_residual: wp.array, row_offset: int, out_cost: wp.array):
        inputs = [self.body_position, self.body_rotation, self.clock, self.worlds, self.world_steps, self.gripper]
        inputs += [self.obj, self.offset, self.weight, self.tolerance, self.object_frame, row_offset]
        wp.launch(
            _compute_grasp_kernel,
            dim=self.body_position.shape[0],
            inputs=inputs,
            outputs=[out_residual, out_cost],
            device=self.body_position.device,
        )
