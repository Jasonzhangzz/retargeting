"""Explicit actuator preload as an initial condition for stationary FA holds."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from functools import wraps

import numpy as np


def seeded_initial_state(initial, model, index, command, reference, horizon):
    """Keep CA state/reference; seed additive control knots and previous input."""
    command, reference = np.asarray(command, np.float64), np.asarray(reference, np.float64)
    hand, actuators = np.asarray(index.hand_qpos), np.asarray(index.hand_ctrl)
    if command.shape != (len(hand),) or reference.shape != command.shape or not np.isfinite(command).all():
        raise ValueError("Preload must contain finite controls for every hand coordinate")
    if not np.allclose(initial.qpos[hand], reference, rtol=0, atol=2e-6):
        raise ValueError("Control preload belongs to a different ContactAware initial pose")
    bounds = model.actuator_ctrlrange[actuators]
    limited = model.actuator_ctrllimited[actuators].astype(bool)
    if np.any(limited & ((command < bounds[:, 0]-1e-6) | (command > bounds[:, 1]+1e-6))):
        raise ValueError("Control preload exceeds actual actuator control limits")
    raw_force = model.actuator_gainprm[actuators, 0]*command + model.actuator_biasprm[actuators, 0]
    raw_force += model.actuator_biasprm[actuators, 1]*reference
    dofs = model.jnt_dofadr[model.actuator_trnid[actuators, 0]]
    raw_force += model.actuator_biasprm[actuators, 2]*initial.qvel[dofs]
    limits = model.actuator_forcerange[actuators]
    force_limited = model.actuator_forcelimited[actuators].astype(bool)
    if np.any(force_limited & ((raw_force < limits[:, 0]-1e-6) | (raw_force > limits[:, 1]+1e-6))):
        raise ValueError("Control preload exceeds actual actuator force limits")
    dimension = len(hand)
    if initial.warm.shape != (dimension*(1+horizon),):
        raise ValueError("Preload and original FA control-knot dimensions disagree")
    warm = initial.warm.copy()
    warm[dimension:] = np.tile(command-reference, horizon)
    return replace(initial, warm=warm, previous_action=command.astype(np.float32))


def activate_control_preload(root):
    """Supply a feasible, separately recorded initial input to original FA."""
    root = Path(root)
    payload, report_path = root/"control_preload.npz", root/"control_preload.json"
    report = json.loads(report_path.read_text())
    if report.get("feasible") is not True:
        raise ValueError("Cannot initialize FA with an infeasible static preload")
    with np.load(payload, allow_pickle=False) as raw:
        command, reference = raw["command"].copy(), raw["hand_reference_qpos"].copy()
    from forceaware import workflow
    original = workflow._initial_run

    def initialize(cfg, model, index, *, sequence):
        if cfg.sequence.hold_frame < 0 or not np.allclose(sequence.hand, sequence.hand[:1], rtol=0, atol=2e-6):
            raise ValueError("Static preload initialization requires a stationary CA hold")
        initial = original(cfg, model, index, sequence=sequence)
        return seeded_initial_state(initial, model, index, command, reference, cfg.optimizer.horizon)

    workflow._initial_run = initialize
    return dict(mode="static_load_preload", optimizer_modified=False, objective_modified=False,
                initial_qpos_modified=False, initial_qvel_modified=False, hand_reference_modified=False,
                initial_control_knots_seeded=True, previous_action_is_preload=True,
                payload_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(), report=str(report_path))


def activate_fixed_initial_state():
    """Use the original solver's allow_init=False for a known CA initial state.

    The upstream rolling driver always enables first-window state fitting.
    A static preload is computed for a known state, so that input condition
    must also stay fixed. Only this existing solve-call option is selected;
    the optimizer, objective, control variables and update code are unchanged.
    """
    from forceaware.servo_prior import ServoPriorOptimizer
    original = ServoPriorOptimizer.solve

    @wraps(original)
    def solve(self, *args, **kwargs):
        kwargs["allow_init"] = False
        return original(self, *args, **kwargs)

    ServoPriorOptimizer.solve = solve
    return dict(mode="fixed_CA_state", initial_state_correction_enabled=False,
                original_solve_option=dict(allow_init=False), objective_modified=False,
                optimizer_update_algorithm_modified=False, control_optimization_enabled=True,
                reason="Static preload belongs to the accepted CA initial pose; the measured initial state stays fixed")
