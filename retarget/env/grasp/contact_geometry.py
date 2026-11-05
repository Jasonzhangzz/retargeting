"""Shared contact geometry, variable bounds and penetration penalties."""
from dataclasses import dataclass, asdict
import math
import torch

from retarget.solver.cem import ContactSettings as PoseObjectiveSettings

def penetration_terms(depth, settings):
    """Full-hand clearance deficits [hand/object, hand/hand], in metres.

    Only hand/object compression gets a soft allowance. The independent self
    term still penalizes any overlap; touching at zero distance is allowed.
    """
    object_depth = (depth[:, 0] - settings.collision_clearance_m).clamp_min(0.0)
    excess = (object_depth - settings.hand_object_allowance_m).clamp_min(0.0)
    self_depth = depth[:, 1]
    strict = (excess / settings.penetration_scale_m).square() + (
        self_depth / settings.penetration_scale_m
    ).square()
    soft_scale = max(settings.hand_object_allowance_m, settings.penetration_scale_m)
    soft = (object_depth / soft_scale).square()
    return strict, soft, excess
