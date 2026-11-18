"""Shared units for the Robokit contact Task and native review bridge."""
import numpy as np
from retarget.solver.contact_defaults import DEFAULTS


def force_penalty_parameters(model) -> tuple:
    # H1 support must regularly carry its full weight. A dimensionless excess
    # above one body weight per point regularizes impacts without taxing stance.
    # Keep the existing manipulation scale for the fixed-base Sharpa tasks.
    if any(model.body(i).name == 'pelvis' for i in range(model.nbody)):
        scale = max(float(model.body_mass.sum() * np.linalg.norm(model.opt.gravity)), 1.)
        return scale, DEFAULTS.humanoid_force_peak_weight
    return 100., .1
