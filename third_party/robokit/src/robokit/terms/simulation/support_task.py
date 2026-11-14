# pyright: reportGeneralTypeIssues=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false
# pyright: reportOperatorIssue=false
"""Penalize steps where too few finger groups touch an object."""

from typing import Any, Mapping, Optional, Sequence

import warp as wp

from robokit.opt.var_values import VarValues
from robokit.terms.simulation.sim_context import SimContext
from robokit.terms.task import Context, Task


@wp.kernel
def _compute_touch_kernel(
    count: wp.array(dtype=int),
    world: wp.array(dtype=int),
    geom: wp.array(dtype=wp.vec2i),
    dist: wp.array(dtype=float),
    obj: int,
    group: wp.array(dtype=int),
    margin: float,
    touch: wp.array2d(dtype=int),
):
    c = wp.tid()
    if c >= count[0] or dist[c] > margin:
        return
    a, b = geom[c][0], geom[c][1]
    other = wp.where(a == obj, b, wp.where(b == obj, a, -1))
    if other >= 0 and group[other] >= 0:
        wp.atomic_max(touch, world[c], group[other], 1)


@wp.kernel
def _compute_support_cost_kernel(
    touch: wp.array2d(dtype=int), need: int, weight: float, free: int, out: wp.array(dtype=float)
):
    w, last = wp.tid(), touch.shape[1] - 1
    n = int(0)
    for g in range(touch.shape[1] - free):
        n += touch[w, g]
    if free == 0 or touch[w, last] == 0:
        out[w] += weight * float(wp.max(need - n, 0))
    for g in range(touch.shape[1]):
        touch[w, g] = 0


class SupportTask(Task):
    """Per step, `weight` times how many groups short of `need` touch geom `obj`, at least 0; a group is every geom whose name contains one of `groups`, and it touches when any of its contacts with `obj` is within `margin`; with `free`, the last group (say a thumb) is not counted and the cost applies only while it is off `obj`.

    Example:
        >>> contacts = [([4], int), ([0, 0, 1, 1], int), ([[0, 1], [2, 0], [1, 0], [0, 3]], wp.vec2i)]
        >>> contacts += [([0.0, 0.001, 0.0, 0.0], float), ([[0, 0, 0]] * 4, wp.vec3)]
        >>> context = {  # world 0: index and middle on the pen; world 1: index and thumb
        ...     SimContext.GEOMS: ["pen", "index", "middle", "thumb"],
        ...     SimContext.CTRL: wp.zeros((2, 1), dtype=float, device="cpu"),
        ...     SimContext.CONTACTS: tuple(wp.array(x, dtype=t, device="cpu") for x, t in contacts),
        ... }
        >>> costs = []
        >>> for term in [SupportTask("pen", ["index", "middle"], 2, 3.0, 0.002),
        ...              SupportTask("pen", ["index", "middle", "thumb"], 2, 3.0, 0.002, free=True)]:  # fmt: skip
        ...     term.resolve(None, context)
        ...     out = wp.zeros(2, dtype=float, device="cpu")
        ...     term.compute_weighted_cost(None, out_cost=out)
        ...     costs.append(out.numpy().tolist())
        >>> costs  # world 1 is one finger short, which `free` forgives while the thumb is on the pen
        [[0.0, 3.0], [0.0, 0.0]]
    """

    def __init__(
        self, obj: str, groups: Sequence[str], need: int, weight: float, margin: float = 0.0, free: bool = False
    ):
        self.obj, self.groups, self.need, self.weight = obj, list(groups), need, weight
        self.margin, self.free = margin, free
        self.contact_data, self.geoms, self.ctrl = SimContext.CONTACTS, SimContext.GEOMS, SimContext.CTRL

    def resolve(self, var_values: Optional[VarValues], context: Mapping[Context, Any]):
        """Map every geom to its group, look up the object and make each world's touch flags."""
        super().resolve(var_values, context)
        group = [next((i for i, g in enumerate(self.groups) if g in (name or "")), -1) for name in self.geoms]
        self.group, self.obj = wp.array(group, dtype=int, device=self.ctrl.device), self.geoms.index(self.obj)
        self.touch = wp.zeros((self.ctrl.shape[0], len(self.groups)), dtype=int, device=self.ctrl.device)

    def compute_weighted_cost(self, var_values: Optional[VarValues], *, out_cost: wp.array):
        count, world, geom, dist = self.contact_data[:4]
        inputs = [count, world, geom, dist, self.obj, self.group, self.margin]
        wp.launch(
            _compute_touch_kernel, dim=world.shape[0], inputs=inputs, outputs=[self.touch], device=out_cost.device
        )
        inputs = [self.touch, self.need, self.weight, int(self.free)]
        wp.launch(
            _compute_support_cost_kernel,
            dim=out_cost.shape[0],
            inputs=inputs,
            outputs=[out_cost],
            device=out_cost.device,
        )
