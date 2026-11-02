"""Static, fixed-template MPPI grasp synthesis without an ADMM reference."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import time
import traceback

import numpy as np

from retarget.common.results import write_json


OBJECTIVE = "template_comfree_soft_contact_v4"
SUCCESS_CRITERION = "projected_soft_contact_closure_v2"


def _positive(value, name, *, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or value <= 0
            or (integer and not isinstance(value, int))):
        kind = "a positive integer" if integer else "finite and positive"
        raise ValueError(f"{name} must be {kind}")
    return value


def _options(config):
    """Validate this runner's options without loading a GPU model."""
    from .template_mppi import TemplateMPPISettings

    values = config.environment.get("template_mppi", {})
    if not isinstance(values, dict):
        raise ValueError("environment.template_mppi must be a mapping")
    unknown = set(values) - {"objective", "initialization", "size_ik", "evaluate_native", "object_mass", "parallel"}
    if unknown:
        raise ValueError(f"Unknown template_mppi options: {sorted(unknown)}")
    objective = values.get("objective", {})
    if not isinstance(objective, dict):
        raise ValueError("template_mppi.objective must be a mapping")
    try:
        objective = TemplateMPPISettings(**objective)
    except TypeError as error:
        raise ValueError(f"Invalid template_mppi.objective: {error}") from error
    initialization = values.get("initialization", {})
    if not isinstance(initialization, dict):
        raise ValueError("template_mppi.initialization must be a mapping")
    unknown = set(initialization) - {"points", "inplane", "iterations", "selection"}
    if unknown:
        raise ValueError(f"Unknown template_mppi.initialization options: {sorted(unknown)}")
    initialization = {"points": 24, "inplane": 8, "iterations": 60, **initialization}
    for key, value in initialization.items():
        if key == "selection":
            if value not in ("matcher", "collision_aware"):
                raise ValueError("template_mppi.initialization.selection must be matcher or collision_aware")
        else:
            _positive(value, f"template_mppi.initialization.{key}", integer=True)
    evaluate_native = values.get("evaluate_native", False)
    if evaluate_native is not False:
        raise ValueError("Native physical evaluation was removed from template MPPI; use evaluate_native: false")
    if "parallel" in values:
        from .template_mppi_parallel import options
        options(values["parallel"])
    mass = _positive(values.get("object_mass", .1), "template_mppi.object_mass")
    from .template_mppi_size_ik import SizeIKSettings
    try:
        size_ik = SizeIKSettings(**values.get('size_ik', {}))
    except TypeError as error:
        raise ValueError(f'Invalid template_mppi.size_ik: {error}') from error
    return SimpleNamespace(objective=objective, initialization=initialization,
                           size_ik=size_ik, evaluate_native=evaluate_native, object_mass=float(mass))


def validate(config):
    """Resolve exact template names and keep legacy grasp adapters out."""
    from .primitives import catalog

    if config.task.get("type") != "grasp" or config.solver.type != "mppi":
        raise ValueError("Template MPPI requires task grasp and solver mppi")
    if config.solver.gradient_steps:
        raise ValueError("Template MPPI optimizes q by sampling; use objective.closure for inner GN settings")
    if not str(config.device).startswith("cuda"):
        raise ValueError("Template MPPI requires a CUDA environment device")
    settings = _options(config)
    values = config.task.get("options", {})
    if not isinstance(values, dict):
        raise ValueError("task.options must be a mapping")
    unknown = set(values) - {"hand_model", "dexonomy_root", "primary_shape_only",
                             "primitive", "radius", "height", "scale"}
    if unknown:
        raise ValueError(f"Unknown template MPPI input options: {sorted(unknown)}")
    if values.get("hand_model", "wuji2") != "wuji2":
        raise ValueError("Template MPPI supports only wuji2")
    data = Path(config.task["data"])
    checkout = Path(values.get("dexonomy_root", data / "templates")).expanduser()
    if not checkout.is_absolute():
        checkout = data / checkout
    checkout = checkout.resolve()
    inventory = catalog(checkout)
    by_id = {str(row["type_id"]): row["template"] for row in inventory}
    supplied = config.task.get("templates", ["1"])
    if not isinstance(supplied, (list, tuple)) or not supplied:
        raise ValueError("task.templates must be a nonempty list")
    templates = []
    for supplied_name in supplied:
        name = by_id.get(str(supplied_name), str(supplied_name))
        annotation = checkout / "assets/hand/shadow/raw_anno" / f"{name}.yaml"
        if not annotation.is_file():
            raise ValueError(f"Missing grasp template: {annotation}")
        if name in templates:
            raise ValueError("task.templates must not contain duplicates")
        templates.append(name)
    scale = _positive(values.get("scale", 1.), "task.options.scale")
    for key in ("radius", "height"):
        if values.get(key) is not None:
            _positive(values[key], f"task.options.{key}")
    primary = values.get("primary_shape_only", True)
    if not isinstance(primary, bool):
        raise ValueError("task.options.primary_shape_only must be boolean")
    radial = config.task.get("radius_scales", [1.])
    axial = config.task.get("height_scales", [1.])
    for key, scales in (("radius_scales", radial), ("height_scales", axial)):
        if not isinstance(scales, (list, tuple)) or not scales:
            raise ValueError(f"task.{key} must be a nonempty list")
        for value in scales:
            _positive(value, f"task.{key}")
    object_path = config.task.get("object")
    if object_path is not None and not Path(object_path).is_file():
        raise ValueError(f"Missing grasp object: {object_path}")
    return SimpleNamespace(
        dexonomy_root=checkout, templates=templates, settings=settings,
        solver=config.solver,
        init_device=config.device, scale=float(scale), primary_shape_only=primary,
        primitive=values.get("primitive"), radius=values.get("radius"), height=values.get("height"),
        radius_scales=list(radial), height_scales=list(axial),
        primitive_grid_explicit=("radius_scales" in config.task or "height_scales" in config.task
                                 or values.get("radius") is not None or values.get("height") is not None),
    )


def _native_qpos(q):
    from scipy.spatial.transform import Rotation

    q = np.asarray(q, float).reshape(26)
    return np.r_[q[:3], Rotation.from_euler("XYZ", q[3:6]).as_quat(scalar_first=True), q[6:]]


def _choose_root(objective, template, candidates, device):
    """Choose root geometry before the fingers and force magnitudes adapt."""
    import torch

    candidates = np.asarray(candidates, float)
    if candidates.ndim != 2 or candidates.shape[1] != 26 or not len(candidates):
        raise ValueError("Root initializer must return a nonempty [N,26] array")
    if not np.isfinite(candidates).all():
        raise ValueError("Root initializer returned nonfinite configurations")
    objective.bind(template, candidates[0])
    geometry_only = callable(getattr(objective, "evaluate_initialization", None))
    evaluate = objective.evaluate_initialization if geometry_only else objective.evaluate
    with torch.no_grad():
        controls = torch.as_tensor(candidates, dtype=torch.float32, device=device)[:, None]
        costs = evaluate(controls).detach().cpu().numpy()
    if costs.shape != (len(candidates),):
        raise ValueError("Root candidate costs have an unexpected shape")
    finite = np.isfinite(costs)
    if not finite.any():
        raise FloatingPointError("No finite root initialization candidate")
    index = int(np.argmin(np.where(finite, costs, np.inf)))
    return candidates[index].copy(), dict(selected=index, candidate_costs=costs.tolist(),
                                         selection_objective="template_root_geometry_v1" if geometry_only else OBJECTIVE,
                                         root_selection_cost="geometry_only" if geometry_only else "evaluate_fallback",
                                         actual_contact_or_force_closure_required=False)


def run(config, output, *, initialization_reference=None, primitive_override=None, objective_factory=None):
    """Generate templates, initialize only the root, and solve one q per object."""
    import torch
    import warp as wp

    from .initialization import matching_mesh
    from .inputs import build_scene
    from .primitives import jobs
    from .template_mppi import TemplatePoseObjective
    from .template_mppi_inputs import load_solver_template_contacts, load_fixed_reference_template
    from .template_mppi_reference import prepare_reference, load_reference, reference_key, POLICY

    args = validate(config)
    settings = args.settings
    if config.environment["template_mppi"].get("parallel"):
        from .template_mppi_parallel import run_parallel
        return run_parallel(config, output, args)
    if not torch.cuda.is_available():
        raise RuntimeError("Template MPPI requires CUDA; no GPU solve was performed")
    torch.set_num_threads(1)
    wp.init()
    wp.set_device(config.device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if config.task.get("object"):
        tasks = [(Path(config.task["object"]), name, primitive_override) for name in args.templates]
        skipped = []
    else:
        tasks, skipped = jobs(output, args.templates, args)
    write_json(output / "skipped.json", skipped)
    if not tasks:
        raise ValueError("The requested selection has no supported single-object cases")
    if initialization_reference is not None and len(tasks) != 1:
        raise ValueError("A worker initialization reference requires exactly one case")
    references = {}
    manifest, rows = [], []
    for index, (mesh, name, primitive) in enumerate(tasks):
        case = output / f"{index:03d}_{name}"
        case.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        print(f"[template_mppi] {index + 1}/{len(tasks)} {name}: starting", flush=True)
        summary = dict(method="mppi", objective=OBJECTIVE, template=name, hand="wuji2",
                       primitive=primitive, seed=config.seed, state="failed",
                       solver_completed=False, task_success=False, native_evaluated=False,
                       input_policy=POLICY,
                       admm_reference_used=False, success_criterion=SUCCESS_CRITERION)
        stage = "template_input"
        try:
            model, surface, input_xml, offset = build_scene(mesh, args.scale, "wuji2")
            (case / "scene.xml").write_text(input_xml)
            centered_mesh = case / "input_object.obj"
            matching_mesh(surface).export(centered_mesh, digits=17)
            write_json(case / "object_frame.json", dict(
                source_mesh=str(Path(mesh).resolve()), scale=args.scale,
                recenter_offset_m=np.asarray(offset).tolist(),
                convention="scaled source vertices minus recenter_offset_m; fixed object pose identity"))
            if initialization_reference is not None:
                template = load_fixed_reference_template(initialization_reference, args.dexonomy_root, model, name)
            else:
                template = load_solver_template_contacts(name, args.dexonomy_root, model,
                                                 device=config.device, cache_root=output / "template_cache")
            write_json(case / "template_contacts.json", template.to_dict())
            canonical = np.r_[np.zeros(6), template.finger_qpos]
            np.savez_compressed(case / "trajectory.npz", qpos=canonical[None], time=[0.])
            stage = "root_initialization"
            before = time.perf_counter()
            key = reference_key(name, primitive)
            if key not in references:
                references[key] = initialization_reference or prepare_reference(
                    mesh, name, primitive, args, output, device=config.device,
                    seed=config.seed, template=template)
            initial, initialization = load_reference(references[key], template, name)
            reference_path = Path(references[key])
            with np.load(reference_path.parent / "candidates.npz") as proposals:
                candidates = proposals["qpos"].copy()
            objective = (objective_factory or TemplatePoseObjective)(centered_mesh, settings.objective,
                                             device=config.device, object_mass=settings.object_mass)
            if hasattr(objective, "set_batch_size"):
                objective.set_batch_size(config.solver.worlds)
            objective.bind(template, initial)
            if hasattr(objective, 'configure_object_regions'):
                objective.configure_object_regions(name, primitive)
            size_ik_report = None
            reference_initial = initial.copy()
            if settings.size_ik.enabled and hasattr(objective, 'initialize_size'):
                stage = 'size_ik_initialization'
                initial, size_ik_report = objective.initialize_size(reference_path, primitive,
                    initial, settings.size_ik, config.seed)
                write_json(case / 'size_ik.json', size_ik_report)
                # Recenter only the local root search box; all finger bounds
                # remain their physical limits, independent of 1x angles.
                objective._search_bounds(initial)
            initialization = dict(initialization,
                                  seconds=time.perf_counter() - before,
                                  initial_qpos=initial.tolist(),
                                  template_finger_angles_preserved=bool(np.array_equal(initial[6:], template.finger_qpos)),
                                  template_contacts_preserved=True,
                                  opening_prefit=False)
            if size_ik_report is not None:
                initialization.update(size_ik_policy=size_ik_report['policy'], size_ik=size_ik_report,
                                      reference_initial_qpos=reference_initial.tolist())
            write_json(case / "initialization.json", initialization)
            np.savez_compressed(case / "initialization_candidates.npz", qpos=candidates)
            (case / "scene.xml").write_text(objective.xml)
            stage = "mppi_optimization"
            q, detail = objective.optimize(initial, config.solver, config.seed)
            q = np.asarray(q, float).reshape(26)
            if not np.isfinite(q).all():
                raise FloatingPointError("Template MPPI returned a nonfinite hand configuration")
            trajectory = np.asarray(detail.pop("trajectory"), float).reshape(-1, 26)
            if not len(trajectory) or not np.isfinite(trajectory).all():
                raise FloatingPointError("Template MPPI returned an invalid optimization trajectory")
            if not np.allclose(trajectory[-1], q, atol=1e-6, rtol=0):
                raise ValueError("Optimization trajectory does not end at the exported grasp")
            native = _native_qpos(q)
            write_json(case / "optimization.json", detail)
            diagnostics = getattr(objective, "final_diagnostics", objective.diagnostics)(q)
            if hasattr(objective, "patch_diagnostics"):
                from .template_mppi_patch_coverage import attach_patch_coverage
                patch_coverage = objective.patch_diagnostics(q)
                write_json(case / "patch_contact_coverage.json", patch_coverage)
                diagnostics = attach_patch_coverage(diagnostics, patch_coverage)
                diagnostics["patch_coverage_file"] = "patch_contact_coverage.json"
            write_json(case / "projection_force_closure.json", diagnostics)
            np.savez_compressed(case / "grasp.npz", qpos=q, grasp_qpos=native)
            np.savez_compressed(case / "trajectory.npz", qpos=trajectory,
                                time=np.arange(len(trajectory), dtype=float))
            summary.update(solver_completed=True, state="completed", geometry_diagnostics=diagnostics,
                           initialization_seconds=initialization["seconds"],
                           initial_cost=detail.get("initial_cost"), final_cost=detail.get("final_cost"),
                           optimization_seconds=detail.get("optimization_seconds"),
                           objective_settings=asdict(settings.objective), failure_stage=None,
                           task_success=bool(diagnostics["soft_contact_success"]))
        except Exception as error:
            (case / "error.log").write_text(traceback.format_exc())
            summary.update(state="failed", failure_stage=stage,
                           reason="initialization_failed" if not summary["solver_completed"] else "result_export_failed",
                           error=f"{type(error).__name__}: {error}",
                           error_log=str((case / "error.log").relative_to(output)))
            print(f"[template_mppi] {name}: {summary['error']}", flush=True)
        summary["wall_seconds"] = time.perf_counter() - started
        write_json(case / "summary.json", summary)
        rows.append(summary)
        if (case / "scene.xml").is_file() and (case / "trajectory.npz").is_file():
            relative = case.relative_to(output)
            entry = dict(task="grasp", name=case.name, method="mppi", objective=OBJECTIVE,
                         hand="wuji2", template=name, primitive=primitive, seed=config.seed,
                         model=str(relative / "scene.xml"), trajectory=str(relative / "trajectory.npz"),
                         summary=str(relative / "summary.json"), dt=1., phase="refinement",
                         task_success=summary["task_success"], success_criterion=SUCCESS_CRITERION)
            manifest.append(entry)
        print(f"[template_mppi] {index + 1}/{len(tasks)} {name}: "
              f"solver={summary['solver_completed']} geometry={summary['task_success']}", flush=True)
    write_json(output / "summary.json", dict(
        method="mppi", objective=OBJECTIVE, planned=len(tasks), finished=len(rows),
        solver_completed=sum(bool(row["solver_completed"]) for row in rows),
        native_evaluated=sum(bool(row["native_evaluated"]) for row in rows),
        successes=sum(bool(row["task_success"]) for row in rows), cases=rows,
        acceptance="Projected friction closure and separate self-penetration limit; hand-object compression softly penalized; coverage diagnostic only; no physical evaluation"))
    return manifest
