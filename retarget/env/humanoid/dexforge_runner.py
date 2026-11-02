"""Run the pinned DexForge ForceAware controller from an existing G1 reference.

No ContactAware/IK, ADMM, IPOPT or alternate contact solver is invoked here.
Planning, adjoints, multistart updates and execution belong to upstream FA;
this module supplies the G1 input/output and independent validation boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from retarget.common.results import clean_json, write_json


@dataclass
class PreparedRun:
    artifacts: object
    reference: object
    cfg: object
    models: object
    optimizer: object
    environment_versions: dict
    upstream: dict


class GradientValidationError(RuntimeError):
    """An actual native gradient probe failed before controller execution."""


def _environment_options(config):
    values = config.environment.get("dexforge", {})
    if not isinstance(values, dict):
        raise ValueError("environment.dexforge must be a mapping")
    unknown = set(values) - {"gradient_check"}
    if unknown:
        raise ValueError(f"Unknown environment.dexforge fields: {sorted(unknown)}")
    check = values.get("gradient_check", {})
    if isinstance(check, bool):
        check = {"enabled": check}
    if not isinstance(check, dict):
        raise ValueError("environment.dexforge.gradient_check must be a mapping")
    unknown = set(check) - {"enabled", "report", "directions", "epsilons", "rtol", "atol"}
    if unknown:
        raise ValueError(f"Unknown DexForge gradient check fields: {sorted(unknown)}")
    if not isinstance(check.get("enabled", True), bool):
        raise ValueError("DexForge gradient_check.enabled must be a boolean")
    directions = check.get("directions", 3)
    if isinstance(directions, bool) or not isinstance(directions, int) or directions < 2:
        raise ValueError("DexForge gradient_check.directions must be an integer >= 2")
    epsilons = tuple(check.get("epsilons", (.01, .003, .001, .0003, .0001, .00003, .00001, .000003)))
    if len(epsilons) < 3 or any(not np.isfinite(e) or e <= 0 for e in epsilons):
        raise ValueError("DexForge gradient checks require at least three positive epsilon values")
    if len(set(epsilons)) != len(epsilons):
        raise ValueError("DexForge gradient check epsilon values must be distinct")
    for name, default in (("rtol", .05), ("atol", .002)):
        if not np.isfinite(check.get(name, default)) or check.get(name, default) <= 0:
            raise ValueError(f"DexForge gradient_check.{name} must be positive")
    return dict(enabled=check.get("enabled", True), report=check.get("report"),
                directions=directions, epsilons=tuple(sorted(epsilons, reverse=True)),
                rtol=float(check.get("rtol", .05)), atol=float(check.get("atol", .002)))


def validate(config):
    """Validate source inputs without importing the retired humanoid physics."""
    if config.solver.type != "dexforge" or config.task.get("type") != "humanoid":
        raise ValueError("DexForge requires solver.type: dexforge and task.type: humanoid")
    if config.task.get("name") != "move_largebox":
        raise ValueError("DexForge humanoid currently requires task.name: move_largebox")
    if config.environment.get("dynamics", "dexforge") != "dexforge":
        raise ValueError("DexForge uses environment.dynamics: dexforge")
    if not str(config.device).startswith("cuda"):
        raise ValueError("The pinned DexForge native FA backend requires CUDA")
    if config.environment.get("scsp_contact") is not None:
        raise ValueError("DexForge does not use environment.scsp_contact")
    frames = config.environment.get("frames")
    if frames is not None and (isinstance(frames, bool) or not isinstance(frames, int) or frames < 1):
        raise ValueError("environment.frames must be a positive source-frame count")
    check = _environment_options(config)
    if frames is None and not check["enabled"]:
        raise ValueError("A full DexForge trajectory requires the actual native gradient validation gate")
    data = Path(config.task["data"])
    for name in ("scene.xml", "reference.npz", "task_info.json"):
        if not (data / name).is_file():
            raise ValueError(f"Missing humanoid input: {data / name}")
    with np.load(data / "reference.npz", allow_pickle=False) as raw:
        for name in ("qpos", "qvel", "ctrl"):
            if name not in raw or raw[name].ndim != 2 or len(raw[name]) < 2 or not np.isfinite(raw[name]).all():
                raise ValueError(f"Invalid humanoid reference field: {name}")
        if not check["enabled"] and frames is not None and frames >= len(raw["qpos"])-1:
            raise ValueError("A full DexForge trajectory requires the actual native gradient validation gate")
    info = json.loads((data / "task_info.json").read_text())
    if not np.isfinite(info.get("ref_dt", 0)) or info["ref_dt"] <= 0:
        raise ValueError("task_info.json requires a finite positive ref_dt")
    from retarget.dynamics.dexforge_upstream import require_environment, verify_upstream

    require_environment()
    verify_upstream()
    return data


def prepare(config, output):
    """Construct native assets/models and an objective-only FA subclass."""
    import warp as wp
    from retarget.dynamics.dexforge_upstream import activate_upstream, require_environment, verify_upstream

    versions = require_environment()
    activate_upstream()
    wp.init()
    wp.set_device(config.device)
    from forceaware.workflow import _load_runtime_models
    import comfree_warp
    from retarget.solver.dexforge import make_config, create_optimizer
    from .dexforge_reference import load_reference
    from .dexforge_scene import prepare_scene

    data = Path(config.task["data"])
    settings = config.solver.dexforge
    artifacts = prepare_scene(data / "scene.xml", Path(output) / "native_scene", timestep=settings.mpc_dt)
    ref_dt = float(json.loads((data / "task_info.json").read_text())["ref_dt"])
    cfg = make_config(settings, xml=artifacts.xml, ref_dt=ref_dt, device=config.device)
    with wp.ScopedDevice(config.device):
        models = _load_runtime_models(cfg, comfree_warp)
        reference = load_reference(data / "reference.npz", models.planner_cpu, ref_dt=ref_dt)
        optimizer = create_optimizer(models.planner_cpu, models.planner_device, reference.index,
                                     cfg=cfg, settings=settings)
    return PreparedRun(artifacts, reference, cfg, models, optimizer, versions, verify_upstream())


def _identity(config, prepared):
    """A gate report applies only to these exact inputs, settings and runtime."""
    data = Path(config.task["data"])
    settings = asdict(config.solver.dexforge)
    value = dict(upstream_commit=prepared.upstream["commit"], versions=prepared.environment_versions,
                 source_scene_sha256=hashlib.sha256((data / "scene.xml").read_bytes()).hexdigest(),
                 reference_sha256=hashlib.sha256((data / "reference.npz").read_bytes()).hexdigest(),
                 task_info_sha256=hashlib.sha256((data / "task_info.json").read_bytes()).hexdigest(),
                 solver_settings=settings, device=str(config.device))
    root = Path(__file__).resolve().parents[3]
    adapters = ("retarget/env/humanoid/dexforge_runner.py", "retarget/env/humanoid/dexforge_reference.py",
                "retarget/env/humanoid/dexforge_scene.py", "retarget/dynamics/dexforge_loss.py",
                "retarget/dynamics/dexforge_upstream.py", "retarget/solver/dexforge.py",
                "retarget/dynamics/dexforge_contact_validation.py")
    value["adapter_sha256"] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in adapters}
    value["upstream_file_manifest_sha256"] = hashlib.sha256(json.dumps(prepared.upstream["sha256"], sort_keys=True).encode()).hexdigest()
    value["fingerprint"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value


def _fd_assessment(adjoint, trials, *, rtol, atol):
    """Select a stable finite-difference plateau independently of the adjoint.

    A passing point is never chosen merely because it agrees with the supplied
    derivative. Two adjacent epsilon values must first agree with each other
    while preserving actual active engine contact slots.
    """
    stable = []
    for first, second in zip(trials, trials[1:]):
        if not first["same_contact_active"] or not second["same_contact_active"] or not first.get("same_contact_features", True) or not second.get("same_contact_features", True):
            continue
        a, b = first["derivative"], second["derivative"]
        if np.isfinite(a) and np.isfinite(b) and abs(a-b) <= atol + rtol*max(abs(a), abs(b)):
            stable.append(first)
    chosen = stable[0] if stable else None
    finite = bool(np.isfinite(adjoint))
    error = None if chosen is None else float(abs(adjoint-chosen["derivative"]))
    tolerance = None if chosen is None else float(atol + rtol*max(abs(adjoint), abs(chosen["derivative"])))
    return dict(passed=bool(finite and chosen is not None and error <= tolerance),
                adjoint=float(adjoint), selected_epsilon=None if chosen is None else chosen["epsilon"],
                finite_difference=None if chosen is None else chosen["derivative"],
                absolute_error=error, tolerance=tolerance, trials=trials,
                reason=("nonfinite adjoint" if not finite else "no stable finite-difference plateau with unchanged real contacts" if chosen is None else "passed" if error <= tolerance else "adjoint does not match stable finite differences"))


def _contact_signature(compiled, workspace, *, world=0):
    """Hash only active contact IDs and used body-local discrete features."""
    collision = workspace.forward.base.collision
    active = collision.contacts.active.numpy()[world].astype(bool)
    digest = hashlib.sha256(active.tobytes())
    for batch, state in zip(compiled.base.collision.batches, collision.batches):
        selected = state.selection
        count = int(batch.model.contact_count)
        local = active[batch.contact_offset:batch.contact_offset+count]
        for name in ("source_id", "target_id"):
            values = getattr(selected, name).numpy()[world]
            digest.update(np.where(local, values, -1).tobytes())
        pairs = int(batch.source_target_count)
        if not pairs:
            continue
        offset = int(batch.model.target_contact_offset)
        enabled = local[offset:offset+pairs, None]
        # Hard selection depends only on its winning feature. Fused modes also
        # depend on their selected/boundary ranks; inactive padding is ignored.
        ranks = 1 if int(batch.model.contact_topk) == 1 else int(batch.model.contact_topk)+1
        source = selected.candidate_source.numpy()[world, :pairs, :ranks]
        target = selected.candidate_target.numpy()[world, :pairs, :ranks]
        used = enabled & (source >= 0) & (target >= 0)
        digest.update(np.where(used, source, -1).tobytes())
        digest.update(np.where(used, target, -1).tobytes())
        for name in ("candidate_anchor", "candidate_axis"):
            values = getattr(selected, name).numpy()[world, :pairs, :ranks]
            digest.update(np.where(used[..., None], values, 0.).tobytes())
    return digest.hexdigest(), int(active.sum())


def _native_probe(prepared, *, horizon, directions, epsilons, rtol, atol, seed):
    """Check the real native state/control VJP over 1 or 30 contact steps."""
    import warp as wp
    from comfree_warp.native_adjoint.dynamics import DynamicsInput
    from comfree_warp.native_adjoint.fast_step import compile_fast_step, clear_contact_status, check_contact_status
    from comfree_warp.native_adjoint.fast_step_adjoint import (
        BackwardCall, StepCotangent, allocate_workspace, record, forward, backward,
    )
    from comfree_warp.native_adjoint.runtime import StepInput

    cfg, reference = prepared.cfg, prepared.reference
    compiled = compile_fast_step(prepared.models.planner_cpu, prepared.models.planner_device)
    device, cpu = compiled.base.device, prepared.models.planner_cpu
    qbase, vbase = reference.initial_state()
    qbase, vbase = qbase.astype(np.float32), vbase.astype(np.float32)
    for address in (3, reference.index.object_qpos + 3):
        qbase[address:address+4] /= np.linalg.norm(qbase[address:address+4])
    frames = (np.arange(horizon)+1) * cfg.simulator.time_grid.mpc_dt / reference.ref_dt
    targets = reference.qpos_at(frames)[:, reference.index.hand_qpos]
    ubase = np.zeros((horizon, cpu.nu), np.float32)
    ubase[:, reference.index.hand_ctrl] = targets
    matrix = lambda values: wp.array(np.asarray(values, np.float32), dtype=float, device=device, requires_grad=True, retain_grad=True)
    initial_q, initial_v = matrix(qbase[None]), matrix(vbase[None])
    current_q, current_v = initial_q, initial_v
    current_time = matrix(np.zeros(1, np.float32))
    stages = []
    for step in range(horizon):
        dynamics = DynamicsInput()
        dynamics.qpos, dynamics.qvel = current_q, current_v
        dynamics.ctrl = matrix(ubase[step:step+1])
        dynamics.qfrc_applied = matrix(np.zeros((1, cpu.nv), np.float32))
        inputs = StepInput(dynamics, current_time, matrix(np.zeros(1, np.float32)))
        workspace = allocate_workspace(compiled, 1)
        recorded = record(compiled, inputs, workspace, freeze_frame_vjp=False)
        zeros = lambda shape, dtype=float: wp.zeros(shape, dtype=dtype, device=device)
        base = compiled.base
        cotangent = StepCotangent(
            zeros((1, cpu.nq)), zeros((1, cpu.nv)), zeros((1, cpu.nv)), zeros(1),
            zeros((1, cpu.nv)), zeros((1, cpu.nbody), wp.vec3), zeros((1, cpu.nbody), wp.mat33),
            zeros((1, base.collision.contact_count)), zeros((1, base.collision.contact_count), wp.vec3),
            zeros((1, base.collision.contact_count), wp.mat33),
        )
        stages.append((inputs, workspace, recorded, cotangent))
        current_q, current_v, current_time = recorded.result.qpos, recorded.result.qvel, recorded.result.time

    rng = np.random.default_rng(seed)
    qseed = rng.choice([-1., 1.], cpu.nq).astype(np.float32)
    vseed = (.125*rng.choice([-1., 1.], cpu.nv)).astype(np.float32)

    def evaluate(qpos, qvel, controls):
        initial_q.assign(qpos[None].astype(np.float32))
        initial_v.assign(qvel[None].astype(np.float32))
        signatures, contact_counts = [], []
        for step, (inputs, workspace, recorded, cotangent) in enumerate(stages):
            inputs.dynamics.ctrl.assign(controls[step:step+1].astype(np.float32))
            clear_contact_status(compiled, workspace.forward)
            forward(compiled, inputs, workspace)
            check_contact_status(compiled, workspace.forward)
            signature, contact_count = _contact_signature(compiled, workspace)
            signatures.append(signature)
            contact_counts.append(contact_count)
        final = stages[-1][2].result
        # A host double dot avoids adding float32 loss-reduction cancellation
        # to the engine's unavoidable float32 state quantization.
        score = float(np.dot(final.qpos.numpy()[0].astype(np.float64), qseed.astype(np.float64))
                      + np.dot(final.qvel.numpy()[0].astype(np.float64), vseed.astype(np.float64)))
        return score, np.asarray(signatures), contact_counts

    score, signature, contacts = evaluate(qbase, vbase, ubase)
    stages[-1][3].qpos.assign(qseed[None])
    stages[-1][3].qvel.assign(vseed[None])
    gradients = [None]*horizon
    initial_gradient = None
    for step in reversed(range(horizon)):
        inputs, workspace, recorded, cotangent = stages[step]
        gradient = backward(BackwardCall(compiled, inputs, workspace, recorded, cotangent))
        gradients[step] = gradient.ctrl.numpy()[0].copy()
        if step:
            wp.copy(stages[step-1][3].qpos, gradient.qpos)
            wp.copy(stages[step-1][3].qvel, gradient.qvel)
        else:
            initial_gradient = (gradient.qpos.numpy()[0].copy(), gradient.qvel.numpy()[0].copy())
    tests = []
    for direction in range(directions):
        dq = rng.standard_normal(cpu.nq)
        dv = rng.standard_normal(cpu.nv)
        du = rng.standard_normal((horizon, cpu.nu))
        # Quaternion perturbations stay in the tangent plane and are normalized
        # symmetrically; the first derivative remains exactly dq at the base.
        for address in (3, reference.index.object_qpos+3):
            quat = qbase[address:address+4].astype(np.float64)
            dq[address:address+4] -= np.dot(dq[address:address+4], quat)*quat
        dq /= max(np.linalg.norm(dq), 1e-30)
        dv /= max(np.linalg.norm(dv), 1e-30)
        du /= max(np.linalg.norm(du), 1e-30)
        derivative = float(np.dot(initial_gradient[0], dq) + np.dot(initial_gradient[1], dv)
                           + np.sum(np.asarray(gradients)*du))
        trials = []
        for epsilon in epsilons:
            values, masks = [], []
            for sign in (-1., 1.):
                qtrial = qbase.astype(np.float64) + sign*epsilon*dq
                for address in (3, reference.index.object_qpos+3):
                    qtrial[address:address+4] /= np.linalg.norm(qtrial[address:address+4])
                value, active, _ = evaluate(qtrial, vbase+sign*epsilon*dv, ubase+sign*epsilon*du)
                values.append(value)
                masks.append(active)
            trials.append(dict(epsilon=float(epsilon), derivative=float((values[1]-values[0])/(2*epsilon)),
                               same_contact_active=bool(np.array_equal(masks[0], signature) and np.array_equal(masks[1], signature)),
                               same_contact_features=bool(np.array_equal(masks[0], signature) and np.array_equal(masks[1], signature))))
        assessment = _fd_assessment(derivative, trials, rtol=rtol, atol=atol)
        assessment["direction"] = direction
        tests.append(assessment)
    return dict(kind="native_state_chain", horizon=horizon, score=score, active_contacts_per_step=contacts,
                has_actual_contact=bool(any(contacts)), passed=bool(any(contacts) and all(t["passed"] for t in tests)),
                directional_tests=tests, objective="linear final qpos/qvel state cotangent; full native q/v/control chain")


def _objective_probe(prepared, *, directions, epsilons, rtol, atol, seed):
    """Check the actual G1 loss, regularizers and upstream raw-control chain."""
    from forceaware.numerics import LOSS_GRADIENT_SCALE

    optimizer, reference, cfg = prepared.optimizer, prepared.reference, prepared.cfg
    grid = cfg.simulator.time_grid
    qpos, qvel = reference.initial_state()
    dimension = reference.index.action_dim*(cfg.optimizer.horizon+1)
    targets = reference.make_window(executed_steps=0, horizon=cfg.optimizer.horizon,
                                    time_grid=grid, start_frame=0., end_frame=reference.end_frame)
    optimizer._prepare_window(qpos, qvel, targets, warm=np.zeros(dimension, np.float32),
                              previous_action=qpos[reference.index.hand_qpos], seed=seed, start_time=0.)
    control = optimizer.control
    raw = control.raw.numpy().copy()

    def evaluate(values, *, gradient=False):
        control.raw.assign(values)
        optimizer._clear_contact_history()
        optimizer._set_hard_contact()
        optimizer._forward(allow_init=True)
        optimizer._evaluate_losses(allow_init=True)
        if gradient:
            optimizer._backward(allow_init=True)
        score = float(optimizer.loss.total.numpy()[0])*LOSS_GRADIENT_SCALE
        signatures = [_contact_signature(optimizer.compiled, step.workspace) for step in optimizer.steps]
        masks = np.asarray([value[0] for value in signatures])
        counts = [value[1] for value in signatures]
        derivative = control.gradient.numpy()[0].copy() if gradient else None
        return score, masks, derivative, counts

    score, signature, derivative, contacts = evaluate(raw, gradient=True)
    rng = np.random.default_rng(seed+31)
    tests = []
    for index in range(directions):
        direction = rng.standard_normal(dimension)
        direction /= np.linalg.norm(direction)
        adjoint = float(np.dot(derivative, direction))
        trials = []
        for epsilon in epsilons:
            values, masks = [], []
            for sign in (-1., 1.):
                trial = raw.copy()
                trial[0] += (sign*epsilon*direction).astype(np.float32)
                value, mask, _, _ = evaluate(trial)
                values.append(value)
                masks.append(mask)
            trials.append(dict(epsilon=float(epsilon), derivative=float((values[1]-values[0])/(2*epsilon)),
                               same_contact_active=bool(np.array_equal(masks[0], signature) and np.array_equal(masks[1], signature)),
                               same_contact_features=bool(np.array_equal(masks[0], signature) and np.array_equal(masks[1], signature))))
        check = _fd_assessment(adjoint, trials, rtol=rtol, atol=atol)
        check["direction"] = index
        tests.append(check)
    control.raw.assign(raw)
    return dict(kind="forceaware_objective", horizon=len(optimizer.steps), score=score,
                raw_gradient_norm=float(np.linalg.norm(derivative)),
                active_contacts_per_step=contacts,
                has_actual_contact=bool(any(contacts)),
                passed=bool(any(contacts) and np.isfinite(derivative).all() and all(t["passed"] for t in tests)),
                directional_tests=tests, objective="actual G1 root/joint/object/penetration/smoothing FA loss; first-window raw variables",
                objective_scale=float(LOSS_GRADIENT_SCALE))


def check_gradients(config, prepared, *, output=None):
    """Mandatory actual-contact single-step and 30-step derivative gate."""
    options = _environment_options(config)
    identity = _identity(config, prepared)
    report = dict(identity=identity, passed=False, probes=[], check="native full-state directional central differences",
                  finite_difference_dtype="float64 host reduction of actual float32 native states", options=options)
    for horizon in (1, 30):
        print(f"[dexforge gradients] native {horizon}-step contacts/VJP", flush=True)
        report["probes"].append(_native_probe(
            prepared, horizon=horizon, directions=options["directions"], epsilons=options["epsilons"],
            rtol=min(options["rtol"], .01 if horizon == 1 else .05),
            atol=options["atol"], seed=config.seed+177+horizon,
        ))
        if output is not None:
            write_json(output, report)
    print("[dexforge gradients] actual G1 objective/raw-control VJP", flush=True)
    report["probes"].append(_objective_probe(
        prepared, directions=options["directions"], epsilons=options["epsilons"],
        rtol=min(options["rtol"], .05), atol=options["atol"], seed=config.seed+713,
    ))
    if output is not None:
        write_json(output, report)
    print("[dexforge gradients] actual hand-object control-to-object VJP", flush=True)
    from retarget.dynamics.dexforge_contact_validation import validate_hand_object_control

    report["probes"].append(validate_hand_object_control(
        prepared, directions=options["directions"], epsilons=options["epsilons"],
        rtol=min(options["rtol"], .01), atol=options["atol"], seed=config.seed+917,
    ))
    report["passed"] = all(probe["passed"] for probe in report["probes"])
    if output is not None:
        write_json(output, report)
    return report


def _gradient_gate(config, prepared, output):
    options = _environment_options(config)
    destination = Path(output) / "gradient_validation.json"
    if not options["enabled"]:
        report = dict(passed=False, skipped=True, reason="explicit diagnostic prefix without gradient certification",
                      identity=_identity(config, prepared))
        write_json(destination, report)
        return report
    cached = options["report"]
    if cached is not None:
        path = Path(cached).expanduser()
        if not path.is_absolute():
            path = Path(config.source).parent / path
        report = json.loads(path.read_text())
        if report.get("identity", {}).get("fingerprint") != _identity(config, prepared)["fingerprint"]:
            raise GradientValidationError("The supplied gradient report belongs to different inputs, settings or numerical runtime")
        if report.get("options") is None or any(report["options"].get(key) != clean_json(options[key]) for key in ("directions", "epsilons", "rtol", "atol")):
            raise GradientValidationError("The supplied gradient report used different validation tolerances or probe budgets")
        write_json(destination, report)
    else:
        report = check_gradients(config, prepared, output=destination)
    native = sorted(p.get("horizon") for p in report.get("probes", []) if p.get("kind") == "native_state_chain")
    objective = [p for p in report.get("probes", []) if p.get("kind") == "forceaware_objective"]
    hand_object = [p for p in report.get("probes", []) if p.get("kind") == "hand_object_control"]
    if (not report.get("passed") or native != [1, 30] or len(objective) != 1
            or len(hand_object) != 1 or not all(p.get("passed") for p in report.get("probes", []))):
        raise GradientValidationError(f"DexForge native gradient validation failed; controller execution stopped. Inspect {destination}")
    return report


def _duration(reference, source_frames):
    if source_frames is None:
        return reference.duration_seconds
    # frames is a count of source intervals, matching the existing runner's
    # interpretation of frames*ref_steps while allowing a partial final step.
    return min(reference.duration_seconds, source_frames*reference.ref_dt)


def run(config, output):
    """Execute upstream FA solve/execution/warm-start contracts unchanged."""
    import warp as wp
    from retarget.dynamics.dexforge_upstream import activate_upstream

    activate_upstream()
    from forceaware.control import resample_warm_corrections
    from forceaware.executor import WarpActionExecutor
    from forceaware.time_grid import SolveRequest
    from retarget.dynamics.dexforge_diagnostics import collect_window_diagnostics
    from retarget.task.humanoid.metrics import assess_trajectory, timing_summary

    started = time.perf_counter()
    prepared = prepare(config, output)
    gate = _gradient_gate(config, prepared, output)
    cfg, reference, models, optimizer = prepared.cfg, prepared.reference, prepared.models, prepared.optimizer
    grid = cfg.simulator.time_grid
    requested_duration = _duration(reference, config.environment.get("frames"))
    if requested_duration >= reference.duration_seconds and not gate.get("passed"):
        raise GradientValidationError("A complete trajectory cannot bypass the native gradient validation gate")
    request = SolveRequest.create(grid, duration_seconds=requested_duration)
    executor = WarpActionExecutor(models.executor.cpu_model, models.executor.device_model,
                                  index=reference.index, cfg=cfg)
    qpos, qvel = reference.initial_state()
    warm = np.zeros(reference.index.action_dim*(1+cfg.optimizer.horizon), np.float32)
    previous = qpos[reference.index.hand_qpos].copy()
    positions, velocities = [qpos.copy()], [qvel.copy()]
    controls, records = [], []
    initialized = time.perf_counter()-started
    executed = 0
    for window, count in enumerate(request.execution_steps_per_window):
        targets = reference.make_window(executed_steps=executed, horizon=cfg.optimizer.horizon,
                                        time_grid=grid, start_frame=0., end_frame=reference.end_frame)
        tick = time.perf_counter()
        plan = optimizer.solve(qpos, qvel, targets, warm=warm, previous_action=previous,
                               allow_init=window == 0, seed=cfg.optimizer.multistart_seed+window,
                               execution_steps=count, start_time=executor.elapsed)
        solve_seconds = time.perf_counter()-tick
        diagnostics, arrays = collect_window_diagnostics(
            optimizer, plan, targets, qpos_before=qpos, settings=config.solver.dexforge,
            observation_site_ids=prepared.artifacts.observation_site_ids,
            allow_init=window == 0,
        )
        diagnostic_file = f"native_contacts_{window:05d}.npz"
        np.savez_compressed(Path(output)/diagnostic_file, **arrays)
        tick = time.perf_counter()
        if window == 0:
            # Preserve FA's bounded initial joint correction. Floating root and
            # object qpos are absent from hand_qpos and cannot be teleported.
            positions[0] = plan.initial_qpos.copy()
        execution = executor.execute_action_sequence(plan.initial_qpos if window == 0 else qpos,
                                                       qvel, actions=plan.control_sequence)
        execute_seconds = time.perf_counter()-tick
        positions.extend(execution.qpos_sequence)
        velocities.extend(execution.qvel_sequence)
        controls.extend(plan.control_sequence)
        qpos, qvel = execution.next_qpos, execution.next_qvel
        previous = plan.action
        warm = resample_warm_corrections(plan.raw, action_dim=reference.index.action_dim,
                                         horizon=cfg.optimizer.horizon, executed_dense_steps=count,
                                         physics_steps_per_knot=grid.knot_substeps,
                                         warm_scale=cfg.optimizer.warm_shift_scale)
        executed += count
        record = dict(window=window, executed_steps=executed, solve_seconds=solve_seconds,
                      optimization_seconds=solve_seconds, execute_seconds=execute_seconds,
                      iterations=cfg.optimizer.n_iter, cost=float(plan.loss), drift=float(plan.drift),
                      gradient_norm=float(plan.grad_norm), candidates=cfg.optimizer.multistart_count,
                      candidate_diagnostics=clean_json(plan.diag), sampled=cfg.optimizer.multistart_count,
                      physical_diagnostics=diagnostics, contact_prediction_file=diagnostic_file,
                      planned_executed_qpos_max=float(np.max(np.abs(plan.qpos_sequence-execution.qpos_sequence))),
                      evaluated=cfg.optimizer.multistart_count*(cfg.optimizer.n_iter+1), io_seconds=0.)
        records.append(record)
        with (Path(output)/"iterations.jsonl").open("a") as stream:
            stream.write(json.dumps(clean_json(record), allow_nan=False)+"\n")
        actual = np.asarray(positions)
        times = np.arange(len(actual))*grid.mpc_dt
        target_qpos = reference.qpos_at(times/reference.ref_dt)
        np.savez_compressed(Path(output)/"trajectory.npz", qpos=actual, qvel=np.asarray(velocities),
                            ctrl=np.asarray(controls), time=times, qpos_ref=target_qpos)
        write_json(Path(output)/"status.json", dict(state="running", completed_windows=window+1,
                                                   requested_windows=len(request.execution_steps_per_window),
                                                   executed_seconds=executed*grid.mpc_dt))
        print(f"[dexforge {window+1}/{len(request.execution_steps_per_window)}] cost={plan.loss:.6g} solve={solve_seconds:.3f}s", flush=True)
    if executed != request.total_physics_steps:
        raise RuntimeError("DexForge execution disagrees with the official solve request")
    complete = config.environment.get("frames") is None or requested_duration >= reference.duration_seconds
    summary = assess_trajectory(models.planner_cpu, np.asarray(positions), target_qpos, complete, True)
    summary.update(solver="dexforge", upstream_commit=prepared.upstream["commit"],
                   execution_model="DexForge WarpActionExecutor", rollout_model="DexForge native ComFree",
                   input_reference="original dataset joint angles; ContactAware/IK skipped",
                   gradient_validation_passed=bool(gate["passed"]),
                   full_clip_completed=bool(complete), completed_windows=len(records),
                   initialize_and_gradient_seconds=initialized, timing=timing_summary(records),
                   requested_duration_seconds=requested_duration, executed_duration_seconds=executed*grid.mpc_dt,
                   reference_tail_hold_seconds=request.tail_hold_seconds,
                   total_wall_seconds=time.perf_counter()-started)
    if not gate["passed"]:
        summary["task_success"] = False
    from .dexforge_replay import replay_controls

    independent = replay_controls(prepared.artifacts.viewer_xml, Path(output)/"trajectory.npz",
                                  output, complete=complete)
    summary["independent_mujoco_summary"] = "independent_mujoco/summary.json"
    summary["independent_mujoco_task_success"] = bool(independent.get("task_success", False))
    write_json(Path(output)/"summary.json", summary)
    write_json(Path(output)/"model.json", dict(upstream=prepared.upstream,
                                               versions=prepared.environment_versions,
                                               solver_settings=asdict(config.solver.dexforge),
                                               scene=prepared.artifacts.manifest,
                                               reference=dict(source=str(reference.source),
                                                              original_qpos_preserved=True,
                                                              raw_qvel_all_zero=bool(np.all(reference.qvel == 0)),
                                                              initial_velocity_policy="MuJoCo mj_differentiatePos of adjacent original reference frames",
                                                              initial_joint_correction="official bounded first-window FA correction; free root/object excluded"),
                                               algorithm="unmodified upstream ServoPriorOptimizer and native adjoint/executor; G1 objective hook",
                                               observation_sites=list(prepared.artifacts.manifest["observation_sites"])))
    return [dict(task="humanoid", name=config.task["name"], method="dexforge",
                 model=str(prepared.artifacts.viewer_xml), trajectory="trajectory.npz",
                 summary="summary.json", dt=grid.mpc_dt)]


def validate_only(config, output):
    """Prepare the exact task and write a reusable native gradient report."""
    validate(config)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepared = prepare(config, output)
    return check_gradients(config, prepared, output=output/"gradient_validation.json")
