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


def arguments(config):
    defaults = yaml.safe_load(
        (ROOT / "retarget/task/grasp/input_defaults.yaml").read_text()
    )
    options = config.task.get("options", {})
    unknown = set(options) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown grasp input settings: {sorted(unknown)}")
    defaults.update(options)
    for key in ("dexonomy_root", "physics_reference", "wrist_pose_reference"):
        defaults[key] = str((Path(config.task["data"]) / defaults[key]).resolve())
    defaults.update(
        init_device=config.device,
        seed=config.seed,
        explicit_pose=True,
        solver_config=Path(config.task["data"]) / "model.yaml",
        worlds=config.solver.worlds,
    )
    defaults["dexonomy_root"] = Path(defaults["dexonomy_root"])
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
    from .physics import reference_physics

    physics = reference_physics(str(args.physics_reference))
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
    for path in (
        args.solver_config,
        args.dexonomy_root,
        args.physics_reference,
        args.wrist_pose_reference,
    ):
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
                    reason="initialization_failed",
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
                problem.evaluate(problem.mean[None])
                torch.cuda.synchronize(config.device)
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
                problem.evaluate(result.parameters[None])
                torch.cuda.synchronize(config.device)
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
        _, result, runtime, q, predicted, hold, elapsed, problem = selected
        (case / "scene.xml").write_text(runtime.scene_xml)
        write_json(
            case / "model.json",
            dict(
                explicit_prediction=problem.explicit.settings,
                search_dimension=46,
                initial_noise=problem.sigma.cpu().numpy(),
                initial_state_fixed=False,
                independent_servo_offsets=True,
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
            task_success=stable,
            hard_constraints_passed=True,
            predicted_hold_feasible=stable,
            physical_hold_tested=False,
            cost=result.cost,
            pose_cost=float(problem.explicit.cost.numpy()[0]),
            geometric_parts=problem.cost.parts.numpy()[0],
            wrist_prior_cost=float(problem.cost.wrist_prior_cost.numpy()[0]),
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
        )
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
    write_json(
        output / "summary.json", dict(cases=len(cases), solver=config.solver.type)
    )
    return cases
