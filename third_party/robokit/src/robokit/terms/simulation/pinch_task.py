# pyright: reportGeneralTypeIssues=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false
# pyright: reportOperatorIssue=false
"""Score two-finger pinches: opposed contact normals and an aligned gripper, as MJPC's bimanual grasp score.

MJPC: https://arxiv.org/abs/2212.00541.
"""

from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import warp as wp

from robokit.opt.var_values import VarValues
from robokit.terms.simulation.sim_context import SimContext
from robokit.terms.task import Context, Task


@wp.kernel
def _compute_pinch_normals_kernel(
    count: wp.array(dtype=int),
    world: wp.array(dtype=int),
    geom: wp.array(dtype=wp.vec2i),
    dist: wp.array(dtype=float),
    frame: wp.array(dtype=wp.mat33),
    role: wp.array(dtype=int),
    depth: float,
    normal: wp.array3d(dtype=wp.vec3),
    touches: wp.array3d(dtype=int),
):
    c = wp.tid()
    if c >= count[0] or geom[c][0] < 0 or geom[c][1] < 0:
        return
    r0, r1 = role[geom[c][0]], role[geom[c][1]]
    if r0 < 0 or r1 < 0 or r0 // 3 != r1 // 3 or (r0 % 3 == 2) == (r1 % 3 == 2):
        return
    finger, sign = wp.where(r0 % 3 == 2, r1, r0), wp.where(r0 % 3 == 2, -1.0, 1.0)  # normal: finger to object
    scale = wp.where(dist[c] < depth, 10.0, 1.0)  # touching counts 10 times a contact within the margin
    wp.atomic_add(normal, world[c], finger // 3, finger % 3, sign * scale * frame[c][0])
    wp.atomic_add(touches, world[c], finger // 3, finger % 3, 1)


@wp.kernel
def _compute_pinch_cost_kernel(
    sensordata: wp.array2d(dtype=float),
    axes: wp.array2d(dtype=int),
    weight: float,
    normal: wp.array3d(dtype=wp.vec3),
    touches: wp.array3d(dtype=int),
    out: wp.array(dtype=float),
):
    w = wp.tid()
    total = float(0.0)
    for p in range(axes.shape[0]):
        score = float(1.0)  # a finger without contact
        if touches[w, p, 0] > 0 and touches[w, p, 1] > 0:
            g, o = axes[p, 0], axes[p, 1]
            gripper = wp.vec3(sensordata[w, g], sensordata[w, g + 1], sensordata[w, g + 2])
            obj = wp.vec3(sensordata[w, o], sensordata[w, o + 1], sensordata[w, o + 2])
            fingers = wp.dot(wp.normalize(normal[w, p, 0]), wp.normalize(normal[w, p, 1]))
            score = (fingers + 2.0 * wp.dot(gripper, obj) + 3.0) / 6.0
        total += weight * score
        for f in range(2):
            normal[w, p, f] = wp.vec3(0.0)
            touches[w, p, f] = 0
    out[w] += total


class PinchTask(Task):
    """Per step and pinch, `weight` times the bimanual grasp score: 1 while a finger touches nothing of the object, else `(n_0 . n_1 + 2 x_g . x_o + 3) / 6`, 0 for opposed fingers and opposed gripper and object axes; each pinch is `(finger_0, finger_1, object, gripper_axis, object_axis)`: three geoms and two 3-axis sensors; `n_f` sums finger f's contact normals toward the object, one with dist below `depth` (touching) counting 10 times one within the margin.

    Example:
        >>> contacts = [([0, 2], [0.0, 0, 1]), ([2, 1], [0.0, 0, 1])]  # both fingers on the box, from either side
        >>> context = {
        ...     SimContext.CLOCK: wp.array([0, 0, 1], dtype=int, device="cpu"),
        ...     SimContext.GEOMS: ["finger_0", "finger_1", "box"],
        ...     SimContext.SENSORS: [("gripper_x", 0, 3), ("box_x", 3, 3)],
        ...     SimContext.SENSORDATA: wp.array([[1.0, 0, 0, -1, 0, 0]], dtype=float, device="cpu"),
        ...     SimContext.CONTACTS: (
        ...         wp.array([2], dtype=int, device="cpu"),
        ...         wp.zeros(2, dtype=int, device="cpu"),
        ...         wp.array([g for g, _ in contacts], dtype=wp.vec2i, device="cpu"),
        ...         wp.zeros(2, dtype=float, device="cpu"),
        ...         wp.zeros(2, dtype=wp.vec3, device="cpu"),
        ...         wp.array([[n, [1.0, 0, 0], [0, 1.0, 0]] for _, n in contacts], dtype=wp.mat33, device="cpu"),  # normal 1st
        ...     ),
        ... }
        >>> out = wp.zeros(1, dtype=float, device="cpu")
        >>> term = PinchTask([("finger_0", "finger_1", "box", "gripper_x", "box_x")])
        >>> term.resolve(None, context)
        >>> term.compute_weighted_cost(None, out_cost=out)
        >>> assert np.allclose(out.numpy(), [0.0])
    """

    def __init__(self, pinches: Sequence[Tuple[str, str, str, str, str]], weight: float = 1.0, depth: float = 0.0):
        self.pinches, self.weight, self.depth = [tuple(p) for p in pinches], weight, depth
        self.contact_data, self.geoms, self.sensors = SimContext.CONTACTS, SimContext.GEOMS, SimContext.SENSORS
        self.sensordata, self.clock = SimContext.SENSORDATA, SimContext.CLOCK

    def resolve(self, var_values: Optional[VarValues], context: Mapping[Context, Any]):
        """Look up the pinches' geoms and axis sensors and build the per-world accumulators."""
        super().resolve(var_values, context)
        device, adr = self.clock.device, {name: a for name, a, _ in self.sensors}
        role = np.full(len(self.geoms), -1, np.int32)  # pinch * 3 + (0, 1: a finger; 2: the object)
        for p, pinch in enumerate(self.pinches):
            role[[self.geoms.index(g) for g in pinch[:3]]] = 3 * p + np.arange(3)
        self.role = wp.array(role, dtype=int, device=device)
        self.axes = wp.array([[adr[p[3]], adr[p[4]]] for p in self.pinches], dtype=int, device=device)
        shape = (self.sensordata.shape[0], len(self.pinches), 2)
        self.normal, self.touches = wp.zeros(shape, wp.vec3, device), wp.zeros(shape, int, device)

    def compute_weighted_cost(self, var_values: Optional[VarValues], *, out_cost: wp.array):
        count, world, geom, dist, _, frame = self.contact_data
        wp.launch(
            _compute_pinch_normals_kernel,
            dim=world.shape[0],
            inputs=[count, world, geom, dist, frame, self.role, self.depth],
            outputs=[self.normal, self.touches],
            device=out_cost.device,
        )
        wp.launch(
            _compute_pinch_cost_kernel,
            dim=self.sensordata.shape[0],
            inputs=[self.sensordata, self.axes, self.weight],
            outputs=[self.normal, self.touches, out_cost],
            device=out_cost.device,
        )
