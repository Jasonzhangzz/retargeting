"""Read-only directional derivatives of the complete original FA objective.

The diagnostic evaluates the original optimizer's raw parameterization and
objective without Adam updates.  A numerical derivative is accepted only on
a resolved finite-difference plateau with unchanged native program branches.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
from pathlib import Path

import numpy as np

from retarget.common.results import write_json


DEFAULT_EPSILONS = (.001, .0003, .0001, .00003, .00001, .000003)


def objective_scores(trajectory_terms, regularization):
    """Accumulate existing per-step objective terms in host float64."""
    terms, regularizer = np.asarray(trajectory_terms), np.asarray(regularization)
    if terms.ndim != 2 or regularizer.shape != (terms.shape[0],):
        raise ValueError("Objective terms must be [world,term] with regularization[world]")
    return np.sum(terms.astype(np.float64), axis=1, dtype=np.float64) + regularizer.astype(np.float64)


def objective_difference(high, low):
    """Subtract component values before reduction, avoiding scalar cancellation."""
    if np.shape(high.trajectory_terms) != np.shape(low.trajectory_terms):
        raise ValueError("Finite-difference objective terms changed shape")
    return float(np.sum(np.asarray(high.trajectory_terms, np.float64)
        - np.asarray(low.trajectory_terms, np.float64), dtype=np.float64)
        + (float(high.regularization) - float(low.regularization)))


def unscaled_raw_gradient(gradient, *, gradient_scale):
    values, scale = np.asarray(gradient), float(gradient_scale)
    if values.ndim != 2 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Expected raw gradients[world,parameter] and a positive finite scale")
    return values.astype(np.float64) / scale


@dataclass(frozen=True)
class ObjectiveSample:
    raw: np.ndarray
    trajectory_terms: np.ndarray
    regularization: float
    controls: np.ndarray
    contact_signature: tuple
    solver_signature: tuple
    actuator_signature: tuple
    contact_counts: tuple = ()
    simulator_valid: bool = True

    @property
    def score(self):
        return float(objective_scores(self.trajectory_terms[None], [self.regularization])[0])


def _directions(raw, *, action_dim, count, seed, include_initial_raw):
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("directions must be a positive integer")
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((count, raw.size))
    if not include_initial_raw:
        values[:, :action_dim] = 0.
    norm = np.linalg.norm(values, axis=1)
    if np.any(norm == 0.):
        raise ValueError("No raw control dimensions are available to perturb")
    return values / norm[:, None]


def directional_checks(evaluate, base, gradient, vectors, *, expected_control_direction,
                       epsilons=DEFAULT_EPSILONS, rtol=.05, atol=.02):
    """Compare an already evaluated gradient to branch-aware central differences.

    This numerical driver accepts a sampler so its accumulation, quantization
    and branch rejection can be tested without pretending synthetic dynamics
    are an actual native-objective validation.
    """
    from .forceaware_validation import native_fd_assessment

    epsilons = tuple(float(value) for value in epsilons)
    if (len(epsilons) < 2 or not np.isfinite(epsilons).all() or min(epsilons) <= 0
            or any(a <= b for a, b in zip(epsilons, epsilons[1:]))):
        raise ValueError("FD epsilons must be a strictly decreasing positive ladder")
    raw, gradient, vectors = np.asarray(base.raw, np.float64), np.asarray(gradient, np.float64), np.asarray(vectors, np.float64)
    if gradient.shape != raw.shape or vectors.ndim != 2 or vectors.shape[1:] != raw.shape:
        raise ValueError("Raw point, gradient and directional vectors disagree")
    if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("Directional vectors must be finite and nonzero")
    tests = []
    for number, direction in enumerate(vectors):
        expected = np.asarray(expected_control_direction(direction), np.float64)
        derivative = float(np.dot(gradient, direction))
        trials = []
        for epsilon in epsilons:
            low = evaluate((raw-epsilon*direction).astype(np.float32))
            high = evaluate((raw+epsilon*direction).astype(np.float32))
            effective_raw = (np.asarray(high.raw, np.float64)-np.asarray(low.raw, np.float64))/(2*epsilon)
            effective_control = (np.asarray(high.controls, np.float64)-np.asarray(low.controls, np.float64))/(2*epsilon)
            raw_error = float(np.linalg.norm(effective_raw-direction))
            control_error = float(np.linalg.norm(effective_control-expected))
            same_contact = (low.contact_signature == high.contact_signature == base.contact_signature)
            valid = bool(low.simulator_valid and high.simulator_valid)
            trials.append(dict(epsilon=epsilon, derivative=objective_difference(high, low)/(2*epsilon),
                same_contact_active=bool(same_contact and valid), same_contact_features=bool(same_contact and valid),
                same_contact_solver_branch=bool(low.solver_signature == high.solver_signature == base.solver_signature),
                same_actuator_branch=bool(low.actuator_signature == high.actuator_signature == base.actuator_signature),
                simulator_valid=valid, raw_direction_error_norm=raw_error,
                dense_control_direction_error_norm=control_error,
                control_direction_error_norm=max(raw_error, control_error),
                quantized_raw_direction_adjoint=float(np.dot(gradient, effective_raw))))
        assessment = native_fd_assessment(derivative, trials, rtol=rtol, atol=atol)
        assessment["direction"] = number
        tests.append(assessment)
    return tests


def _sample(optimizer, raw, *, allow_init, with_gradient=False):
    from .forceaware_validation import native_contact_signature, native_program_branch_signatures
    from forceaware.numerics import LOSS_GRADIENT_SCALE

    optimizer.control.raw.assign(np.asarray(raw, np.float32)[None])
    optimizer.control.valid.fill_(1)
    optimizer._clear_contact_history()
    optimizer._set_hard_contact()
    optimizer._forward(allow_init=allow_init)
    optimizer._validate_simulator_candidates()
    optimizer._evaluate_losses(allow_init=allow_init)
    training = optimizer.objective.training
    contact, solver, actuator, counts, controls = [], [], [], [], []
    for step in optimizer.steps:
        signature, count = native_contact_signature(optimizer.compiled, step.workspace)
        branches = native_program_branch_signatures(optimizer.compiled, step.workspace, step.inputs)
        contact.append(signature); counts.append(count)
        solver.append(branches[0]); actuator.append(branches[1])
        controls.append(step.inputs.dynamics.ctrl.numpy()[0].copy())
    sample = ObjectiveSample(raw=optimizer.control.raw.numpy()[0].copy(),
        trajectory_terms=training.trajectory_terms.numpy()[0].copy(),
        regularization=float(training.regularization.numpy()[0]), controls=np.asarray(controls),
        contact_signature=tuple(contact), solver_signature=tuple(solver), actuator_signature=tuple(actuator),
        contact_counts=tuple(counts), simulator_valid=bool(optimizer.control.valid.numpy()[0]))
    gradient = None
    if with_gradient and sample.simulator_valid and np.isfinite(sample.score):
        optimizer._backward(allow_init=allow_init)
        gradient = unscaled_raw_gradient(optimizer.control.gradient.numpy(), gradient_scale=LOSS_GRADIENT_SCALE)[0]
    return sample, gradient


def _dense_direction(optimizer, direction):
    action_dim = optimizer.index.action_dim
    knots = np.asarray(direction[action_dim:]).reshape(optimizer.settings.horizon, action_dim)
    result = np.zeros((len(optimizer.dense_knot_map), optimizer.cpu_model.nu), np.float64)
    for step, (left, right, fraction) in enumerate(optimizer.dense_knot_map):
        first = np.zeros(action_dim) if left < 0 else knots[left]
        result[step, optimizer.index.hand_ctrl] = first + fraction*(knots[right]-first)
    return result


def probe_window(optimizer, qpos, qvel, targets, *, raw=None, previous_action=None,
                 allow_init=False, include_initial_raw=False, directions=3, seed=177,
                 epsilons=DEFAULT_EPSILONS, rtol=.05, atol=.02, start_time=0.):
    """Evaluate an original, one-candidate optimizer without solving or updating."""
    if optimizer.settings.worlds != 1:
        raise ValueError("Full-objective diagnostic requires exactly one candidate")
    if include_initial_raw and not allow_init:
        raise ValueError("Initial raw directions require the original allow_init policy to be enabled")
    dim = optimizer.index.action_dim
    raw = np.zeros(dim*(1+optimizer.settings.horizon), np.float32) if raw is None else np.asarray(raw, np.float32)
    if raw.shape != (dim*(1+optimizer.settings.horizon),) or not np.isfinite(raw).all():
        raise ValueError("Invalid finite full-window raw parameter point")
    previous = np.asarray(qpos)[optimizer.index.hand_qpos] if previous_action is None else previous_action
    optimizer._prepare_window(qpos, qvel, targets, warm=raw, previous_action=previous,
                              seed=seed, start_time=start_time)
    base, gradient = _sample(optimizer, raw, allow_init=allow_init, with_gradient=True)
    report = dict(kind="original_forceaware_complete_objective", optimizer_run=False,
        objective="original MpcObjective trajectory, terminal, contact, init and all regularization terms",
        score_accumulation="float64 sum of original float32 trajectory_terms and regularization; FD subtracts components before reduction",
        gradient_units="optimizer.control.gradient / original LOSS_GRADIENT_SCALE",
        branch_policy="all dense-step collision features, row activity, hard solver branches, actuator/control clipping",
        raw_dimension=raw.size, allow_init=bool(allow_init), include_initial_raw=bool(include_initial_raw),
        initial_state_optimized=False, parameters_updated=False,
        horizon_knots=optimizer.settings.horizon, native_steps=optimizer.settings.dense_steps,
        active_contacts_per_step=list(base.contact_counts), has_actual_contact=bool(any(base.contact_counts)),
        score=base.score, trajectory_terms=np.asarray(base.trajectory_terms, np.float64).tolist(),
        regularization=base.regularization, raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
        epsilons=list(epsilons), seed=seed, directions=directions, rtol=rtol, atol=atol)
    vectors = _directions(raw, action_dim=dim, count=directions, seed=seed, include_initial_raw=include_initial_raw)
    if gradient is None or not np.isfinite(gradient).all():
        report.update(certified=False, outcome="inconclusive", directional_tests=[],
                      reason="base complete objective/gradient or native simulator status is not finite/valid")
    else:
        evaluate = lambda point: _sample(optimizer, point, allow_init=allow_init)[0]
        try:
            checks = directional_checks(evaluate, base, gradient, vectors,
                expected_control_direction=lambda direction: _dense_direction(optimizer, direction),
                epsilons=epsilons, rtol=rtol, atol=atol)
        finally:
            _sample(optimizer, raw, allow_init=allow_init)
        certified = all(row["passed"] for row in checks)
        report.update(certified=certified, outcome="passed" if certified else
            "mismatch_on_same_branch" if any(row["outcome"] == "mismatch_on_same_branch" for row in checks)
            else "inconclusive", directional_tests=checks, gradient_norm=float(np.linalg.norm(gradient)),
            reason="all requested directional tests passed" if certified else "complete objective derivative remains uncertified")
    return report, dict(raw=raw.copy(), gradient=np.asarray([]) if gradient is None else gradient,
                        directions=vectors, qpos=np.asarray(qpos), qvel=np.asarray(qvel))


def diagnostic_config(cfg):
    """Reduce candidate/iteration budgets only; preserve every cost and time grid."""
    return replace(cfg, optimizer=replace(cfg.optimizer, multistart_count=1, n_iter=1))


def validate_objective(artifacts, cfg, output_dir, *, servo_integration="upstream_euler",
                       device="cuda:0", **probe_options):
    """Load actual native FA models, then audit one complete original window."""
    from retarget.dynamics.dexforge_upstream import activate_upstream, require_environment, verify_upstream
    require_environment(); activate_upstream()
    integration = dict(mode=servo_integration, upstream_runtime_modified=False)
    if servo_integration == "implicit_actuator":
        from retarget.dynamics.dexforge_servo_integration import activate_servo_integration
        integration.update(activate_servo_integration(), upstream_runtime_modified=True)
    elif servo_integration != "upstream_euler":
        raise ValueError("Unknown native servo integration diagnostic mode")
    from .forceaware_reference import create_optimizer, load_reference, register_wuji2_hand
    register_wuji2_hand()
    from forceaware.runtime import setup_runtime
    from forceaware.workflow import _load_runtime_models
    import comfree_warp
    import warp as wp
    setup_runtime(cfg); wp.set_device(device)
    checked_cfg = diagnostic_config(cfg)
    models = _load_runtime_models(checked_cfg, comfree_warp)
    reference = load_reference(artifacts, models.planner_cpu, models.planner_device, checked_cfg)
    optimizer = create_optimizer(models.planner_cpu, models.planner_device, reference.index,
        cfg=checked_cfg, guidance_count=reference.sequence.contact.mask.shape[1])
    report, arrays = probe_window(optimizer, reference.qpos, reference.qvel, reference.targets, **probe_options)
    report.update(upstream_commit=verify_upstream()["commit"], integration=integration,
        optimizer_class=f"{type(optimizer).__module__}.{type(optimizer).__name__}",
        objective_class=f"{type(optimizer.objective).__module__}.{type(optimizer.objective).__name__}",
        diagnostic_budget=dict(original_multistart_count=cfg.optimizer.multistart_count,
            original_n_iter=cfg.optimizer.n_iter, checked_multistart_count=1, checked_n_iter=1,
            adam_iterations_executed=0, horizon_unchanged=checked_cfg.optimizer.horizon == cfg.optimizer.horizon),
        original_loss=asdict(cfg.loss), original_smoothing=asdict(cfg.smoothing),
        original_bounds=asdict(cfg.bounds), original_contact=asdict(cfg.contact),
        time_grid=asdict(cfg.simulator.time_grid), source_reference=str(artifacts.root))
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "objective_probe.npz", **arrays)
    write_json(destination / "objective_validation.json", report)
    return report
