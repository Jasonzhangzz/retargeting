"""Full-objective sampling of static hand pose and independent servo offsets."""

from pathlib import Path
from types import SimpleNamespace
import json, time
import numpy as np
import torch
import warp as wp
import yaml
from retarget.config.run import ROOT
from retarget.solver.base import Problem, solve, NoValidCandidate
from retarget.common.results import write_json, clean_json
from .inputs import initialize_cases
from .runtime import GraspRuntime, prepare_experiment
from .rollout import GraspProblem


def tracking_snapshot(problem):
    tracking = getattr(problem, "tracking", None)
    if tracking is None:
        return None
    return dict(
        parts=tracking.parts.numpy()[0].copy(),
        step_parts=tracking.step_parts.numpy()[0].copy(),
        metrics=tracking.metrics.numpy()[0].copy(),
    )


def contact_solve_diagnostics(problem):
    """Check the final contact iterate of the selected candidate, not its cost."""
    explicit = problem.explicit
    inverse = explicit.parameters.inverse.numpy()
    model = problem.runtime.model
    names = [model.body(int(body)).name for body in explicit.parameters.bodies.numpy()]
    frames = []
    for step, stage in enumerate(explicit.stages):
        force = stage.force.numpy()[0]
        jacobian = stage.jac.numpy()[0]
        displacement = stage.free.numpy()[0] + inverse * (jacobian.T @ force)
        normal_gap = (stage.phi.numpy()[0] + jacobian @ displacement).reshape(-1, 4).mean(1)
        normal_force = force.reshape(-1, 4).sum(1)
        loaded = normal_force > 1.0e-4
        residual = max(
            float(np.maximum(-normal_gap, 0.0).max()),
            float(np.abs(normal_gap[loaded]).max()) if loaded.any() else 0.0,
        ) * 1000.0
        if not all(np.isfinite(value).all() for value in (
            force, jacobian, displacement, stage.phi.numpy()[0], normal_gap,
        )):
            residual = float("inf")
        grouped = {name: 0.0 for name in ("thumb", "index", "middle", "ring", "pinky", "palm")}
        for body_name, pressure in zip(names, normal_force):
            name = (
                ("thumb", "index", "middle", "ring", "pinky")[int(body_name[len("right_finger")]) - 1]
                if body_name.startswith("right_finger") else "palm"
            )
            grouped[name] += float(pressure)
        resultant = (force[:, None] * stage.rays.numpy()[0]).sum(0)
        frames.append(dict(
            time_s=step * explicit.parameters.dt,
            normal_constraint_residual_mm=residual,
            iterations_used=getattr(stage, "iterations_used", explicit.settings["contact_iterations"]),
            predicted_normal_force_N=grouped,
            predicted_contact_force_world_N=resultant,
        ))
    maximum = max(row["normal_constraint_residual_mm"] for row in frames)
    tolerance = getattr(problem.runtime.args, "contact_solver_tolerance_mm", 0.1)
    return dict(
        scope="normal feasibility/complementarity check in the lightweight predictor; not full friction KKT or physical hold certification",
        iterations=explicit.settings["contact_iterations"],
        maximum_normal_constraint_residual_mm=maximum,
        tolerance_mm=tolerance,
        converged=bool(np.isfinite(maximum) and maximum <= tolerance),
        object_weight_N=float(model.body_mass[problem.runtime.index.object_body]) * 9.81,
        frames=frames,
    )


def arguments(config):
    defaults = yaml.safe_load(
        (ROOT / "retarget/task/grasp/input_defaults.yaml").read_text()
    )
    options = config.task.get("options", {})
    unknown = set(options) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown grasp input settings: {sorted(unknown)}")
    defaults.update(options)
    defaults["primitive_grid_explicit"] = any(k in config.task for k in ("radius_scales", "height_scales")) or any(options.get(k) is not None for k in ("radius", "height"))
    for key in ("dexonomy_root", "physics_reference", "wrist_pose_reference"):
        if defaults[key] is not None:
            defaults[key] = str((Path(config.task["data"]) / defaults[key]).resolve())
    defaults.update(
        init_device=config.device,
        seed=config.seed,
        explicit_pose=True,
        solver_config=Path(config.task["data"]) / "model.yaml",
        worlds=config.solver.worlds,
    )
    defaults["dexonomy_root"] = Path(defaults["dexonomy_root"])
    if defaults["physics_reference"] is not None:
        defaults["physics_reference"] = Path(defaults["physics_reference"])
    defaults["wrist_pose_reference"] = Path(
        config.task.get("wrist_reference", defaults["wrist_pose_reference"])
    )
    if config.task.get("object"):
        defaults["primitives"] = False
        defaults["object"] = [Path(config.task["object"])]
    else:
        defaults["primitives"] = True
    args = SimpleNamespace(**defaults)
    from .physics import grasp_physics

    physics = grasp_physics(args)
    for key, value in dict(
        friction=physics["friction"][0],
        impratio=physics["impratio"],
        contact_solref=physics["solref"],
        contact_solimp=physics["solimp"],
    ).items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    from .primitives import catalog

    supplied = config.task.get("templates", args.templates)
    args.templates = [
        next(
            row["template"]
            for row in catalog(args.dexonomy_root)
            if str(row["type_id"]) == str(t)
        )
        if str(t).isdigit()
        else t
        for t in supplied
    ]
    args.primitive = getattr(args, "primitive", None)
    args.radius_scales = config.task.get("radius_scales", [1.0])
    args.height_scales = config.task.get("height_scales", [1.0])
    return args


def validate(config):
    args = arguments(config)
    if args.hand_model not in ("wuji1", "wuji2"):
        raise ValueError("hand_model must be wuji1 or wuji2")
    if args.objective not in ("card", "dex_tracking"):
        raise ValueError("Grasp objective must be card or dex_tracking")
    if getattr(args, "contact_mapping", "nearest") not in ("nearest", "semantic_patch"):
        raise ValueError("contact_mapping must be nearest or semantic_patch")
    if getattr(args, "reference_method", "template") not in ("template", "dexforge_contactaware"):
        raise ValueError("reference_method must be template or dexforge_contactaware")
    if getattr(args, "reference_method", "template") == "dexforge_contactaware":
        if args.contact_mapping != "semantic_patch":
            raise ValueError("ContactAware primitive references require semantic_patch mapping")
        from .contactaware import validate_settings, require_upstream

        validate_settings(args)
        require_upstream()
    if args.objective == "dex_tracking":
        from retarget.task.grasp.tracking_cost import configuration

        configuration(args)
    if config.solver.gradient_steps:
        raise ValueError("Static grasp sampling does not support gradient_steps")
    if min(args.starts, args.init_points, args.init_inplane, args.init_iterations) < 1:
        raise ValueError("Grasp initialization budgets must be positive")
    if any(
        not np.isfinite(v) or v <= 0 for v in args.radius_scales + args.height_scales
    ):
        raise ValueError("Grasp dimensions must be positive and finite")
    for template in args.templates:
        path = args.dexonomy_root / "assets/hand/shadow/raw_anno" / f"{template}.yaml"
        if not path.is_file():
            raise ValueError(f"Missing grasp template: {path}")
    required = [
        args.solver_config,
        args.dexonomy_root / "assets/hand/shadow/right.xml",
        args.dexonomy_root / "assets/hand/shadow/keypoint.yaml",
        args.dexonomy_root / "dexonomy/config/op/init.yaml",
    ]
    if args.physics_reference is not None:
        required.append(args.physics_reference)
    if args.wrist_pose_prior:
        required.append(args.wrist_pose_reference)
    from .hand import asset_root, HAND_BODIES

    required.append(asset_root(args.hand_model) / "right.xml")
    required.extend(
        asset_root(args.hand_model) / "qp_right" / f"{name}.npz"
        for name in HAND_BODIES
    )
    for path in required:
        if not path.exists():
            raise ValueError(f"Missing grasp input: {path}")
    if config.task.get("object") and not Path(config.task["object"]).is_file():
        raise ValueError("Grasp object mesh does not exist")
    return args


def run(config, output):
    args = validate(config)
    cases = []
    torch.set_num_threads(1)
    wp.init()
    wp.set_device(config.device)
    for hand, guesses, case, input_xml in initialize_cases(args, output):
        relative = case.relative_to(output)
        started = time.perf_counter()
        write_json(case / "initialization.json", [g.metadata for g in guesses])
        valid = [g for g in guesses if g.metadata.get("initialization_valid", False)]
        if not valid:
            (case / "scene.xml").write_text(input_xml)
            np.savez_compressed(
                case / "trajectory.npz", qpos=guesses[0].hand_qpos[None], time=[0.0]
            )
            write_json(
                case / "summary.json",
                dict(
                    task_success=False,
                    reason="contactaware_reference_failed"
                    if getattr(args, "reference_method", "template") == "dexforge_contactaware"
                    else "initialization_failed",
                    reference_method=getattr(args, "reference_method", "template"),
                    case_id=str(relative),
                    method=config.solver.type,
                ),
            )
            cases.append(
                dict(
                    task="grasp",
                    name=case.name,
                    method=config.solver.type,
                    hand=args.hand_model,
                    model=str(relative / "scene.xml"),
                    trajectory=str(relative / "trajectory.npz"),
                    summary=str(relative / "summary.json"),
                    dt=0.1,
                )
            )
            continue
        selected = None
        failures = []
        for start, guess in enumerate(valid):
            experiment = prepare_experiment(
                case / f"input_{start:02d}", hand, guess, args
            )
            runtime = GraspRuntime(experiment, args)
            vertices, faces = np.unique(
                hand.surface.triangles.reshape(-1, 3), axis=0, return_inverse=True
            )
            mesh = wp.Mesh(
                points=wp.array(vertices, dtype=wp.vec3, device=config.device),
                indices=wp.array(faces, dtype=int, device=config.device),
            )
            runtime.audit_context = (SimpleNamespace(mesh=mesh), hand, args)
            with wp.ScopedStream(
                wp.stream_from_torch(torch.cuda.current_stream(config.device))
            ):
                problem = GraspProblem(runtime)
                baseline = problem.evaluate(problem.mean[None])
                torch.cuda.synchronize(config.device)
                initial_cost = float(baseline.cost[0])
                initial_tracking = tracking_snapshot(problem)
                tick = time.perf_counter()
                try:
                    result = solve(
                        Problem(
                            problem.mean,
                            problem.sigma,
                            problem.evaluate,
                            problem.project,
                        ),
                        config.solver,
                        config.seed + start,
                    )
                except NoValidCandidate as exc:
                    failures.append(
                        dict(start=start, reason=str(exc), history=exc.history)
                    )
                    continue
                torch.cuda.synchronize(config.device)
                elapsed = time.perf_counter() - tick
                final_evaluation = problem.evaluate(result.parameters[None])
                torch.cuda.synchronize(config.device)
                result.cost = float(final_evaluation.cost[0])
                if not np.isfinite(result.cost):
                    raise FloatingPointError("Selected grasp failed its final geometry/contact solve check")
                q = problem.state.qpos.numpy()[0].copy()
                predicted = np.stack(
                    [state.numpy()[0] for state in problem.explicit.qpos]
                )
                hold = problem.explicit.hold.numpy()[0].copy()
            if selected is None or result.cost < selected[0]:
                selected = (
                    result.cost,
                    result,
                    runtime,
                    q,
                    predicted,
                    hold,
                    elapsed,
                    problem,
                    initial_cost,
                    initial_tracking,
                )
        write_json(case / "failures.json", failures)
        if selected is None:
            write_json(
                case / "summary.json",
                dict(
                    task_success=False,
                    reason="no_feasible_candidate",
                    hard_constraints_passed=False,
                ),
            )
            (case / "scene.xml").write_text(runtime.scene_xml)
            np.savez_compressed(
                case / "trajectory.npz", qpos=runtime.qpos[None], time=[0.0]
            )
            cases.append(
                dict(
                    task="grasp",
                    name=case.name,
                    method=config.solver.type,
                    hand=args.hand_model,
                    template=case.name,
                    status="no_feasible_candidate",
                    model=str(relative / "scene.xml"),
                    trajectory=str(relative / "trajectory.npz"),
                    summary=str(relative / "summary.json"),
                    dt=0.1,
                )
            )
            continue
        (_, result, runtime, q, predicted, hold, elapsed, problem,
         initial_cost, initial_tracking) = selected
        (case / "scene.xml").write_text(runtime.scene_xml)
        write_json(
            case / "model.json",
            dict(
                explicit_prediction=problem.explicit.settings,
                search_dimension=46,
                initial_noise=problem.sigma.cpu().numpy(),
                initial_state_fixed=False,
                independent_servo_offsets=True,
                objective=args.objective,
                contact_mapping=args.contact_mapping,
            ),
        )
        p = runtime.index.object_qpos
        shift = (
            np.linalg.norm(predicted[:, p : p + 3] - predicted[0, p : p + 3], axis=1)
            * 1000
        )
        quat = predicted[:, p + 3 : p + 7]
        dots = np.abs(quat @ quat[0]) / (
            np.linalg.norm(quat, axis=1) * np.linalg.norm(quat[0])
        )
        angles = np.degrees(2 * np.arccos(np.clip(dots, 0, 1)))
        stable = bool(
            shift.max() <= problem.explicit.settings["motion_tolerance_mm"]
            and angles.max() <= problem.explicit.settings["rotation_tolerance_deg"]
        )
        final_tracking = tracking_snapshot(problem)
        tracking_summary = {}
        contact_check = contact_solve_diagnostics(problem)
        write_json(case / "contact_solver.json", contact_check)
        if final_tracking is not None:
            tracking = problem.tracking
            write_json(case / "objective.json", tracking.metadata)
            np.savez_compressed(
                case / "tracking_costs.npz",
                initial_parts=initial_tracking["parts"],
                final_parts=final_tracking["parts"],
                initial_step_parts=initial_tracking["step_parts"],
                final_step_parts=final_tracking["step_parts"],
                initial_metrics=initial_tracking["metrics"],
                final_metrics=final_tracking["metrics"],
                time=np.arange(len(predicted)) * problem.explicit.parameters.dt,
            )
            names = tuple(tracking.metadata["part_names"])
            metrics = final_tracking["metrics"]
            penetration_ok = bool(metrics[:, 0].max() <= args.penetration_tolerance_mm)
            contacts_retained = bool(metrics[:, 4].max() <= args.contact_tolerance_mm)
            tracking_summary = dict(
                objective=args.objective,
                reference_method=getattr(args, "reference_method", "template"),
                initial_cost=initial_cost,
                initial_tracking_cost=float(initial_tracking["parts"].sum()),
                initial_hard_constraints_passed=bool(np.isfinite(initial_cost)),
                initial_cost_parts=dict(zip(names, initial_tracking["parts"])),
                cost_parts=dict(zip(names, final_tracking["parts"])),
                cost_reduction=(initial_cost - result.cost)
                if np.isfinite(initial_cost) else None,
                tracking_cost_reduction=float(initial_tracking["parts"].sum()) - result.cost,
                predicted_contacts_retained=contacts_retained,
                predicted_penetration_acceptable=penetration_ok,
                max_prediction_penetration_mm=float(metrics[:, 0].max()),
                max_guided_gap_mm=float(metrics[:, 4].max()),
                max_template_distance_mm=float(metrics[:, 2].max()),
            )
            task_success = bool(stable and contacts_retained and penetration_ok and contact_check["converged"])
        else:
            task_success = stable
        np.savez_compressed(
            case / "grasp.npz",
            qpos=q,
            initial_qpos=runtime.qpos,
            servo_offsets=result.parameters.cpu().numpy()[26:],
            hold_command=hold,
        )
        np.savez_compressed(
            case / "trajectory.npz",
            qpos=predicted,
            time=np.arange(len(predicted)) * problem.explicit.parameters.dt,
        )
        summary = dict(
            task_success=task_success,
            hard_constraints_passed=True,
            predicted_hold_feasible=task_success,
            predicted_object_stable=stable,
            physical_hold_tested=False,
            contact_solver_converged=contact_check["converged"],
            contact_solver_residual_mm=contact_check["maximum_normal_constraint_residual_mm"],
            cost=result.cost,
            max_displacement_mm=float(shift.max()),
            max_rotation_deg=float(angles.max()),
            solve_seconds=elapsed,
            total_wall_seconds=time.perf_counter() - started,
            sampled=result.sampled,
            evaluated=result.evaluated,
            search_dimension=len(result.parameters),
            model="lightweight compliant contact prediction",
            hand=args.hand_model,
            template=case.name,
            case_id=str(relative),
            contact_mapping=args.contact_mapping,
            **tracking_summary,
        )
        if final_tracking is None:
            summary.update(
                pose_cost=float(problem.explicit.cost.numpy()[0]),
                geometric_parts=problem.cost.parts.numpy()[0],
                wrist_prior_cost=float(problem.cost.wrist_prior_cost.numpy()[0]),
            )
        from .grasp_evaluation import apply_evaluation
        apply_evaluation(summary, case / "scene.xml", q, case, method=config.solver.type,
                         selection=dict(kind="optimized_static_grasp", path=str(case / "grasp.npz")))
        write_json(case / "summary.json", summary)
        with (case / "iterations.jsonl").open("w") as log:
            for row in result.history:
                log.write(json.dumps(clean_json(row)) + "\n")
        cases.append(
            dict(
                task="grasp",
                name=case.name,
                method=config.solver.type,
                hand=args.hand_model,
                template=case.name,
                model=str(relative / "scene.xml"),
                trajectory=str(relative / "trajectory.npz"),
                summary=str(relative / "summary.json"),
                dt=problem.explicit.parameters.dt,
            )
        )
    summaries = [json.loads((output / row["summary"]).read_text()) for row in cases]
    overview = dict(cases=len(cases), solver=config.solver.type,
        evaluation_protocol="graspadmm_native_v1",
        task_successes=sum(bool(row.get("task_success")) for row in summaries),
        all_task_success=bool(summaries) and all(row.get("task_success") for row in summaries))
    if args.objective == "dex_tracking":
        summaries = [
            dict(name=row["name"], **json.loads((output / row["summary"]).read_text()))
            for row in cases
        ]
        overview.update(
            objective=args.objective,
            predicted_successes=sum(bool(row.get("predicted_hold_feasible")) for row in summaries),
            task_successes=sum(bool(row.get("task_success")) for row in summaries),
            evaluation_protocol="graspadmm_native_v1",
            physical_hold_tested=any(row.get("physical_hold_tested",False) for row in summaries),
            results=summaries,
        )
    write_json(output / "summary.json", overview)
    return cases
