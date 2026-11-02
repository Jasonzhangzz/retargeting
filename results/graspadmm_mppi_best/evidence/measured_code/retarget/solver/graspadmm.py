"""Configuration only: GraspADMM runs through its dedicated native runner."""

import math


DEFAULTS = dict(
    step=50, substep=10, step_obj=1, step_hand=1,
    rho_admm=1000.0, lr_obj=0.0002, square_res=True, cdist=0.001,
    fix_hand_local_cp=False, opt_miu_coef=[0.1, 0.0], device="cpu",
    init_epoch=1, init_n_final=1, init_points=1024, init_inplane=8,
    init_iterations=100,
)


def settings(values=None):
    if values is not None and not isinstance(values, dict):
        raise ValueError("solver.graspadmm must be a mapping")
    unknown = set(values or {}) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown solver.graspadmm fields: {sorted(unknown)}")
    result = {**DEFAULTS, **(values or {})}
    for key in ("step", "substep", "step_obj", "step_hand", "init_epoch",
                "init_n_final", "init_points", "init_inplane", "init_iterations"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"solver.graspadmm.{key} must be a positive integer")
    if result["init_epoch"] != 1 or result["init_n_final"] != 1:
        raise ValueError("Independent GraspADMM trials require init_epoch=1 and init_n_final=1")
    for key in ("rho_admm", "lr_obj", "cdist"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"solver.graspadmm.{key} must be finite and positive")
    for key in ("square_res", "fix_hand_local_cp"):
        if not isinstance(result[key], bool):
            raise ValueError(f"solver.graspadmm.{key} must be boolean")
    if result["device"] != "cpu":
        raise ValueError("GraspADMM retains the upstream CPU optimization path; initialization uses environment.device")
    friction = result["opt_miu_coef"]
    if (not isinstance(friction, (list, tuple)) or len(friction) != 2
            or any(isinstance(v, bool) or not isinstance(v, (float, int))
                   or not math.isfinite(v) or v < 0 for v in friction)):
        raise ValueError("solver.graspadmm.opt_miu_coef must contain two finite nonnegative coefficients")
    result["opt_miu_coef"] = list(friction)
    return result
