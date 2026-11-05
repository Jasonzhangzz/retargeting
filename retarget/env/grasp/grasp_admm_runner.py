"""Independent GraspADMM primitive runner; no existing grasp costs or solvers."""

import copy
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from retarget.common.results import write_json
from .graspadmm_source import require_upstream
from .graspadmm_scene import write_scene, from_upstream_qpos
from .graspadmm_inputs import fixed_template, initialize_one, seed_trial
from retarget.solver.graspadmm import solve,NativeProblem


def arguments(config):
    options = dict(config.task.get("options", {}))
    data = Path(config.task["data"])
    checkout = Path(options.get("dexonomy_root", "templates"))
    if not checkout.is_absolute():
        checkout = data / checkout
    args = SimpleNamespace(
        dexonomy_root=checkout.resolve(), hand_model="wuji2", init_device=config.device,
        seed=config.seed, scale=options.get("scale", 1.), primitive=options.get("primitive"),
        radius=options.get("radius"), height=options.get("height"),
        radius_scales=config.task.get("radius_scales", [1.]),
        height_scales=config.task.get("height_scales", [1.]),
        primary_shape_only=options.get("primary_shape_only", True),
        primitive_grid_explicit=any(k in options for k in ("radius", "height")) or
            "radius_scales" in config.task or "height_scales" in config.task,
    )
    from .primitives import catalog
    templates = catalog(checkout)
    args.templates = [
        next(row["template"] for row in templates if str(row["type_id"]) == str(name))
        if str(name).isdigit() else str(name)
        for name in config.task.get("templates", ["1"])
    ]
    return args


def validate(config):
    require_upstream()
    args = arguments(config)
    if config.task.get("type") != "grasp" or config.solver.type != "graspadmm":
        raise ValueError("GraspADMM requires task grasp and solver graspadmm")
    if config.task.get("options", {}).get("hand_model", "wuji2") != "wuji2":
        raise ValueError("GraspADMM currently supports only wuji2")
    if not str(config.device).startswith("cuda"):
        raise ValueError("Native GraspADMM initialization requires a CUDA environment device")
    for key, value in (("scale", args.scale), ("radius", args.radius), ("height", args.height)):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"GraspADMM {key} must be finite and positive")
    return args


def _viewer_scene(scene, data):
    from .graspadmm_scene import WujiEvalEnv
    source = require_upstream()
    env = WujiEvalEnv(
        source.sim.HandCfg(str(scene.hand_xml), freejoint=True), scene.scene_cfg,
        source.sim.MuJoCo_EvalCfg(miu_coef=[.6, .02]), debug_view=False,
    )
    env.reset_qpos(data["grasp_qpos"][0])
    return env._spec.to_xml(), np.r_[data["grasp_qpos"][0], [0., 0., 0., 1., 0., 0., 0.]]


def run(config, output):
    import torch
    from .primitives import jobs
    from .initialization import matching_mesh

    args = validate(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    settings = dict(config.solver.graspadmm)
    if config.task.get("object"):
        tasks = [(Path(config.task["object"]), name, None) for name in args.templates]
        skipped = []
    else:
        tasks, skipped = jobs(output, args.templates, args)
    write_json(output / "skipped.json", skipped)
    cases, summaries = [], []
    for index, (mesh_path, name, primitive) in enumerate(tasks):
        supplied=config.task.get('options',{}).get('primitive_spec')
        if supplied is not None:
            if (len(tasks)!=1 or primitive is None or supplied.get('kind')!=primitive['kind']
                or not np.allclose([supplied.get('radius_m',np.nan),supplied.get('height_m',np.nan)],
                                   [primitive['radius_m'],primitive['height_m']],rtol=0,atol=1.e-12)):
                raise ValueError('primitive_spec must describe this single physical mesh exactly')
            primitive=copy.deepcopy(supplied)
        case = output / f"{index:03d}_{name}"
        case.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        print(f"[graspadmm] {index + 1}/{len(tasks)} {name} {primitive}", flush=True)
        summary = dict(method="graspadmm", template=name, primitive=primitive, seed=config.seed,
                       hand="wuji2", task_success=False, solver_success=False)
        stage = "template_adapter"
        try:
            seed_trial(config.seed)
            hand, template_q, template, template_path, template_info = fixed_template(args, name, mesh_path, case)
            object_mesh = case / "input_object.obj"
            matching_mesh(hand.surface).export(object_mesh, digits=17)
            scene = write_scene(case / "native_scene", object_mesh)
            stage = "native_initialization"
            init_started = time.perf_counter()
            initial, init_info = initialize_one(hand, template_q, template, template_path, args, scene,
                                                seed=config.seed, settings=settings)
            init_info["seconds"] = time.perf_counter() - init_started
            write_json(case / "initialization.json", init_info)
            np.save(case / "initial_grasp_raw.npy", initial)
            np.save(case / "initial_grasp.npy", initial)
            stage = "native_grasp_optimization"
            solve_started = time.perf_counter()
            result, status, trace = solve(NativeProblem(initial, scene, case))
            from .orientation import primitive_contact_diagnostics
            number = name.partition("_")[0]
            if primitive is not None and number.isdigit():
                diagnostics = primitive_contact_diagnostics(
                    int(number), primitive, result.get("ho_c"), initial_hand_contacts=initial)
                summary["semantic_diagnostics"] = diagnostics
                write_json(case / "semantic_diagnostics.json", diagnostics)
            summary.update(solver_success=status["success"], native_stages=status,
                           init_seconds=init_info["seconds"], solve_seconds=time.perf_counter() - solve_started,
                           fallback_used=status["fallback_used"], outer_iterations=status["outer_iterations"])
            # The shared evaluator reuses these exact native postprocessing
            # stages and the upstream six-direction physical protocol.
            candidate = case / "native_candidate.npz"
            np.savez_compressed(candidate, grasp_qpos=result["grasp_qpos"][0],
                                hand_qpos=from_upstream_qpos(result["grasp_qpos"][0]))
            candidate_info = dict(mesh_path=str(scene.mesh_path), template=name,
                                  required_cbody=initial["required_cbody"], primitive=primitive,
                                  hand="wuji2", method="graspadmm", seed=config.seed)
            for key in ("contact_regions",):
                if key in initial:
                    candidate_info[key] = initial[key]
            write_json(case / "native_candidate.json", candidate_info)
            stage = "native_evaluation"
            from .grasp_evaluation import evaluate_native_grasp, PROTOCOL, CRITERION
            evaluation = evaluate_native_grasp(
                result["grasp_qpos"][0], scene.mesh_path, case / "graspadmm_evaluation",
                method="graspadmm", required_cbody=initial["required_cbody"],
                contact_regions=initial.get("contact_regions"),
                selection=dict(template=name, primitive=primitive, seed=config.seed),
            )
            summary.update(evaluation_protocol=PROTOCOL, acceptance=CRITERION,
                           graspadmm_evaluation=evaluation, task_success=bool(evaluation["success"]),
                           physical_hold_tested=bool(evaluation.get("six_direction_tested", False)),
                           failure_stage=evaluation.get("failure_stage"))
            if evaluation.get("contact_region_diagnostics") is not None:
                summary["contact_region_diagnostics"] = evaluation["contact_region_diagnostics"]
                summary["semantic_success"] = evaluation["contact_region_diagnostics"]["semantic_success"]
            xml, qpos = _viewer_scene(scene, result)
            (case / "scene.xml").write_text(xml)
            path = np.asarray([row["hand_qpos"] for row in trace]) if trace else result["grasp_qpos"]
            trajectory = np.c_[path, np.tile([0., 0., 0., 1., 0., 0., 0.], (len(path), 1))]
            np.savez_compressed(case / "trajectory.npz", qpos=trajectory,
                                time=np.arange(len(path)) * settings.get("substep", 10) * .002)
            summary.update(native_candidate=str(candidate), native_candidate_metadata=str(case / "native_candidate.json"))
        except Exception as error:
            import traceback
            summary.update(reason="native_pipeline_exception", failure_stage=stage,
                           error_type=type(error).__name__, error=str(error))
            (case / "error.txt").write_text(traceback.format_exc())
            print(f"[graspadmm] failed {name}: {type(error).__name__}: {error}", flush=True)
        summary["total_wall_seconds"] = time.perf_counter() - started
        write_json(case / "summary.json", summary)
        relative = case.relative_to(output)
        cases.append(dict(task="grasp", name=case.name, method="graspadmm", hand="wuji2", template=name,
                          model=str(relative / "scene.xml"), trajectory=str(relative / "trajectory.npz"),
                          summary=str(relative / "summary.json"), dt=.02))
        summaries.append(summary)
    write_json(output / "summary.json", dict(cases=len(cases), solver="graspadmm", results=summaries,
                                             successful=sum(bool(row["task_success"]) for row in summaries)))
    return cases
