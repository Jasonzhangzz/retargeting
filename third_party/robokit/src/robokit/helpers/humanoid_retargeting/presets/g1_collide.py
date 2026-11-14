"""G1 aligned preset with collision-sphere self-collision avoidance."""

import dataclasses
from copy import deepcopy

from robokit.helpers.humanoid_retargeting.presets.g1_aligned import g1_aligned
from robokit.terms.dense.self_collision_task import SelfCollisionTask


g1_collide = dataclasses.replace(
    g1_aligned,
    terms=[*deepcopy(g1_aligned.terms[:-1]), SelfCollisionTask(weight=100.0, margin=0.01, name="self_collision")],
)


__all__ = ["g1_collide"]
