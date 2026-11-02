#!/usr/bin/env python3
"""Same-input timing audit of pinned GraspADMM and the recorded local adapter.

The primary timing runs have one warmup and five measurements per isolated
process. Direct vendor runs use only the Wuji geometry adapters, with neither
RecordedFilter nor sys.settrace. Separate timed-call passes provide phase timing;
their instrumented numbers are not used for the primary performance ratio.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    from retarget.common.results import write_json
    write_json(path, value)


def fixture_case(output):
    import numpy as np
    import trimesh
    from retarget.env.grasp.graspadmm_scene import write_scene

    record = json.loads((ROOT / "tests/fixtures/graspadmm_reference.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    mesh = output / "fixture.obj"
    trimesh.Trimesh(record["vertices"], record["faces"], process=False).export(mesh, digits=17)
    scene = write_scene(output / "scene", mesh)
    data = record["data"]
    for key in ("grasp_qpos", "hand_cpn_w", "obj_cpn_w", "ext_wrench", "ext_center", "n_evo"):
        data[key] = np.asarray(data[key], np.float32)
    data["scene_path"] = str(scene.scene_path)
    return data, scene


@contextmanager
def direct_geometry_adapters(source):
    from retarget.env.grasp.graspadmm_scene import WujiOptEnv, project_to_hand
    original_env, original_projection = source.core.MuJoCo_OptEnv, source.core.project_to_hand
    source.core.MuJoCo_OptEnv, source.core.project_to_hand = WujiOptEnv, project_to_hand
    try:
        yield
    finally:
        source.core.MuJoCo_OptEnv, source.core.project_to_hand = original_env, original_projection


def direct_call(source, data, scene, directory):
    import numpy as np
    from retarget.env.grasp.graspadmm_optimizer import operation_config

    cfg = operation_config(scene, directory, dict(step=50, substep=10))
    path = Path(cfg.init_dir) / "candidate.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, data)
    function = source.core._single_grasp.__wrapped__
    result = Path(cfg.grasp_dir) / "candidate.npy"
    result.unlink(missing_ok=True)
    with direct_geometry_adapters(source):
        if sys.gettrace() is not None:
            raise RuntimeError("Direct baseline must have no Python line tracer")
        started = time.perf_counter()
        function((str(path), cfg))
        seconds = time.perf_counter() - started
    return seconds, np.load(result, allow_pickle=True).item() if result.is_file() else None, result


@contextmanager
def phase_timers(source):
    """Extra diagnostic pass: wrap calls with clocks, without altering values."""
    import functools
    import importlib
    import inspect
    import torch
    from retarget.env.grasp import graspadmm_scene, graspadmm_optimizer

    totals, originals = {}, []
    operation = source.core._single_grasp.__wrapped__

    def add(stage, seconds):
        row = totals.setdefault(stage, dict(calls=0, cumulative_seconds=0.))
        row["calls"] += 1
        row["cumulative_seconds"] += seconds

    def patch(owner, attribute, stage):
        original = getattr(owner, attribute)

        @functools.wraps(original)
        def timed(*args, **kwargs):
            name = stage
            if stage == "mujoco_steps":
                caller = inspect.currentframe().f_back
                while caller is not None and caller.f_code is not operation.__code__:
                    caller = caller.f_back
                state = {} if caller is None else caller.f_locals
                name = "pregrasp_mujoco_steps" if "pregrasp_lst" in state else "fallback_mujoco_steps" if state.get("terminated", False) else "admm_mujoco_steps"
                del caller
            elif stage == "filters":
                name = "pregrasp_filter" if args[0].log_name.endswith(" pregrasp") else "grasp_filter"
            tick = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                add(name, time.perf_counter() - tick)

        originals.append((owner, attribute, original))
        setattr(owner, attribute, timed)

    sim = importlib.import_module("dexonomy.sim.mujoco_env")
    patch(source.sim.MuJoCo_OptEnv, "__init__", "optimization_environment_build")
    patch(source.qp.ContactQPTorch, "solve", "object_contact_qp")
    patch(source.qp.ContactQP, "solve", "validation_contact_qp")
    patch(source.core, "get_closest_triangles", "object_closest_triangle_query")
    patch(source.core, "point_to_triangle_simple", "object_triangle_projection")
    patch(torch.Tensor, "backward", "object_metric_backward")
    patch(graspadmm_scene, "project_to_hand", "hand_surface_projection")
    patch(graspadmm_optimizer, "project_to_hand", "hand_surface_projection")
    patch(source.sim.MuJoCo_OptEnv, "apply_contact_forces_no_normal", "hand_contact_spring_force")
    patch(sim.MuJoCo_BaseEnv, "step_sim", "mujoco_steps")
    patch(source.core.GraspFilter, "forward", "filters")
    patch(source.sim.MuJoCo_OptEnv, "get_squeeze_qpos", "squeeze_generation")
    patch(source.core, "load_obj_trimesh", "object_mesh_loading")
    try:
        yield totals
    finally:
        for owner, attribute, original in reversed(originals):
            setattr(owner, attribute, original)


def worker(kind, output, warmup, repeats):
    boot = time.perf_counter()
    import copy
    import importlib.metadata
    import numpy as np
    import torch
    from retarget.env.grasp.graspadmm_source import require_upstream
    from retarget.env.grasp.graspadmm_optimizer import optimize

    torch.set_num_threads(1)
    source = require_upstream()
    imported = time.perf_counter()
    data, scene = fixture_case(output / "shared")
    payload = dict(kind=kind, imports_and_source_verification_seconds=imported - boot,
                   scene_setup_seconds=time.perf_counter() - imported,
                   source_commit=source.manifest["commit"], source_manifest_sha256=digest(ROOT / "third_party/Dexonomy/GRASPADMM_UPSTREAM.json"),
                   python=sys.version, executable=sys.executable, pid=os.getpid(),
                   config=dict(step=50, substep=10, step_obj=1, step_hand=1,
                               rho_admm=1000., lr_obj=.0002, square_res=True, opt_miu_coef=[.1, 0.]),
                   torch_threads=torch.get_num_threads(), fixture_sha256=digest(ROOT / "tests/fixtures/graspadmm_reference.json"))
    payload["versions"] = {name: importlib.metadata.version(name) for name in
                           ("torch", "numpy", "scipy", "mujoco", "warp-lang", "trimesh", "qpsolvers", "clarabel")}
    if kind in ("direct", "local"):
        samples = []
        final_path = None
        for iteration in range(warmup + repeats):
            directory = output / f"repetition_{iteration:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            if kind == "direct":
                seconds, result, result_path = direct_call(source, copy.deepcopy(data), scene, directory)
                extra = dict(line_tracer=False, recorded_filter=False)
            else:
                started = time.perf_counter()
                result, status, trace = optimize(copy.deepcopy(data), scene, directory, dict(step=50, substep=10))
                seconds = time.perf_counter() - started
                result_path = directory / "graspadmm_data.npy"
                extra = dict(line_tracer=True, recorded_filter=True,
                             outer_iterations=status["outer_iterations"], fallback_used=status["fallback_used"])
            if result is None:
                raise RuntimeError(f"The common fixture did not pass native generation: {kind}")
            qpos = np.asarray(result["grasp_qpos"])
            row = dict(repetition=iteration, warmup=iteration < warmup, seconds=seconds,
                       final_qpos=qpos.tolist(), result_path=str(result_path), **extra)
            samples.append(row)
            final_path = str(result_path)
        timings = [row["seconds"] for row in samples if not row["warmup"]]
        payload.update(samples=samples, median_seconds=statistics.median(timings),
                       min_seconds=min(timings), max_seconds=max(timings),
                       final_result=final_path, measured_repeats=repeats, warmup=warmup)
    elif kind.startswith("profile_"):
        directory = output / "profiled_operation"
        directory.mkdir(parents=True, exist_ok=True)
        # Warm caches first; profiling is an extra diagnostic, not a timing
        # sample in either primary series.
        direct_call(source, copy.deepcopy(data), scene, output / "profile_warmup")
        with phase_timers(source) as phases:
            if kind == "profile_direct":
                seconds, _, _ = direct_call(source, copy.deepcopy(data), scene, directory)
                phases["original_single_grasp_total"] = dict(calls=1, cumulative_seconds=seconds)
            else:
                started = time.perf_counter()
                optimize(copy.deepcopy(data), scene, directory, dict(step=50, substep=10))
                phases["recorded_optimize_total"] = dict(calls=1, cumulative_seconds=time.perf_counter() - started)
        payload.update(phases=phases,
                       note="Separate timed-call diagnostic pass. Inclusive stages can overlap; not used for the primary runtime ratio.")
    elif kind == "frontend":
        from retarget.config.run import load
        from retarget.env.grasp.grasp_admm_runner import arguments
        from retarget.env.grasp.graspadmm_inputs import fixed_template, fixed_finger_angles, initialize_one
        from retarget.task.grasp.dexonomy import load_template, anatomical_joints
        from retarget.env.grasp.grasp_evaluation import evaluate_native_grasp

        cfg = load(ROOT / "configs/graspadmm.yaml")
        cfg.task["templates"] = ["11"]
        args = arguments(cfg)
        started = time.perf_counter()
        hand, qpos, template, template_path, metadata = fixed_template(args, args.templates[0], scene.mesh_path, output / "template_first")
        first_seconds = time.perf_counter() - started
        shadow, posed, _, annotation = load_template(args.dexonomy_root, args.templates[0])
        fresh, conversion = fixed_finger_angles(hand.model, anatomical_joints(shadow, posed)[None], args, annotation, cache=False)
        started = time.perf_counter()
        _, cached_qpos, _, _, cached_metadata = fixed_template(args, args.templates[0], scene.mesh_path, output / "template_cached")
        cached_seconds = time.perf_counter() - started
        if not np.array_equal(fresh, qpos[6:]) or not np.array_equal(cached_qpos, qpos):
            raise AssertionError("Cached fixed template differs from a fresh conversion")
        started = time.perf_counter()
        _, initialization = initialize_one(hand, qpos, template, template_path, args, scene, seed=0, settings=cfg.solver.graspadmm)
        init_seconds = time.perf_counter() - started
        result = np.load(output.parent / "direct/repetition_01/grasp/candidate.npy", allow_pickle=True).item()
        started = time.perf_counter()
        evaluation = evaluate_native_grasp(result["grasp_qpos"][0], scene.mesh_path, output / "public_evaluation",
                                           required_cbody=result["required_cbody"])
        payload.update(phases=dict(
            uncached_fixed_finger_conversion_seconds=conversion["template_conversion_seconds"],
            first_template_call_seconds=first_seconds,
            first_template_cache_hit=metadata["template_cache_hit"],
            cached_template_call_seconds=cached_seconds,
            cached_template_cache_hit=cached_metadata["template_cache_hit"],
            native_initialization_1024x8_seconds=init_seconds,
            public_evaluation_seconds=time.perf_counter() - started,
            public_evaluation_runtime_mode=evaluation.get("runtime_mode", "subprocess")),
            cached_and_fresh_joint_angles_bitwise_equal=True, initialization=initialization,
            public_evaluation_success=evaluation["success"],
            note="Preparation/evaluation measured separately; excluded from the original-operation performance ratio.")
    payload["worker_wall_seconds"] = time.perf_counter() - boot
    save(output / "result.json", payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/graspadmm_performance")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--worker", choices=("direct", "local", "profile_direct", "profile_local", "frontend"))
    args = parser.parse_args()
    if args.warmup != 1 or args.repeats != 5:
        parser.error("The comparison protocol fixes warmup=1 and repeats=5")
    if args.worker:
        args.output.mkdir(parents=True, exist_ok=True)
        worker(args.worker, args.output.resolve(), args.warmup, args.repeats)
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    python = ROOT / ".venv-graspadmm/bin/python"
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MUJOCO_GL": "egl"}
    results = {}
    for kind in ("direct", "local", "profile_direct", "profile_local", "frontend"):
        folder = output / kind
        folder.mkdir(parents=True, exist_ok=True)
        command = [str(python), str(Path(__file__).resolve()), "--worker", kind,
                   "--output", str(folder), "--warmup", "1", "--repeats", "5"]
        started = time.perf_counter()
        with (folder / "run.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((folder / "result.json").read_text())
        result["process_wall_seconds"] = time.perf_counter() - started
        result["interpreter_startup_and_exit_seconds"] = result["process_wall_seconds"] - result["worker_wall_seconds"]
        results[kind] = result
        print(kind, result.get("median_seconds", result.get("phases")), flush=True)
    import numpy as np
    direct = results["direct"]
    local = results["local"]
    expected = np.asarray(direct["samples"][-1]["final_qpos"])
    actual = np.asarray(local["samples"][-1]["final_qpos"])
    ratio = local["median_seconds"] / direct["median_seconds"]
    report = dict(
        title="Same-input original GraspADMM / Wuji2 adapter performance audit",
        source_commit=direct["source_commit"], source_manifest_sha256=direct["source_manifest_sha256"],
        fixture_sha256=direct["fixture_sha256"], benchmark_script_sha256=digest(__file__),
        runtime=dict(python=str(python), platform=platform.platform(), processor=platform.processor(),
                     versions=direct["versions"],
                     cpu_affinity=sorted(os.sched_getaffinity(0)), threads=1, environment={key: environment[key] for key in
                     ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MUJOCO_GL")}),
        budget=direct["config"], warmup=1, measured_repeats=5,
        comparison=dict(original_operation_median_seconds=direct["median_seconds"],
                        recorded_adapter_total_median_seconds=local["median_seconds"],
                        adapter_to_original_ratio=ratio, same_order_of_magnitude=.1 <= ratio <= 10.,
                        final_grasp_qpos_bitwise_equal=bool(np.array_equal(expected, actual)),
                        final_grasp_qpos_max_abs_difference=float(np.max(np.abs(expected - actual))),
                        baseline="Pinned unmodified _single_grasp on the same real Wuji2 input; only Wuji root/pad geometry adapters",
                        adapter_scope="The full optimize wrapper, including operation configuration, input/output, filters/trace recording"),
        independent_stage_measurements=results["frontend"], phase_profiles={key: results[key] for key in ("profile_direct", "profile_local")},
        series={key: results[key] for key in ("direct", "local")},
        limitations=["This is an operation-equivalence benchmark on Wuji2, not a Shadow-hand performance reproduction.",
                     "Template conversion, native initialization, interpreter/import startup and public six-direction evaluation are excluded from the primary core comparison.",
                     "Separate phase profiles include timer-wrapper overhead and may contain overlapping inclusive durations."],
    )
    save(output / "benchmark.json", report)
    print(json.dumps(report["comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()
