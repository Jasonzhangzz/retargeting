"""Memory-budgeted, independent CUDA template solves with live manifests.

Workers use spawn, never fork a CUDA context. Each pose population and all
wrench scenarios execute on CUDA; processes only schedule independent cases.
The budget measures total device allocations, including Warp and contexts.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from copy import deepcopy
import gc
import math
import multiprocessing as mp
import os
from pathlib import Path
import time

from retarget.common.results import write_json


def options(values):
    if not isinstance(values, dict):
        raise ValueError("template_mppi.parallel must be a mapping")
    unknown = set(values) - {"memory_fraction", "max_workers", "safety_factor"}
    if unknown:
        raise ValueError(f"Unknown template_mppi.parallel options: {sorted(unknown)}")
    result = {"memory_fraction": .8, "max_workers": 32, "safety_factor": 1.08, **values}
    for key in ("memory_fraction", "safety_factor"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"parallel.{key} must be finite")
    if not 0 < result["memory_fraction"] <= .9:
        raise ValueError("parallel.memory_fraction must lie in (0, 0.9]")
    if result["safety_factor"] < 1.:
        raise ValueError("parallel.safety_factor must be at least one")
    value = result["max_workers"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("parallel.max_workers must be a positive integer")
    return result


def worker_count(budget, measured_peak, maximum, safety_factor):
    if measured_peak <= 0 or budget <= 0:
        raise ValueError("A positive measured CUDA memory footprint is required")
    if measured_peak > budget:
        raise RuntimeError(f"One solve needs {measured_peak / 2**30:.2f} GiB, above the "
                           f"{budget / 2**30:.2f} GiB device memory budget")
    return max(1, min(maximum, math.floor(budget / (measured_peak * safety_factor))))


def _solve(config, root, index, mesh, name, primitive, initialization_reference):
    """One worker entry; all artifacts stay at their original stable paths."""
    import json
    import torch
    import warp as wp
    from .template_mppi_runner import run

    root = Path(root)
    relative = Path("cases") / f"{index:04d}_{name}"
    destination = root / relative
    destination.mkdir(parents=True, exist_ok=True)
    # Redirect native CUDA/Warp diagnostics too, without interleaved pool logs.
    with (destination / "worker.log").open("a", buffering=1) as log:
        saved = (os.dup(1), os.dup(2))
        try:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            wp.config.kernel_cache_dir = os.environ.get("RETARGET_WARP_CACHE", "/tmp/retarget-template-mppi-warp")
            child = deepcopy(config)
            child.environment["template_mppi"].pop("parallel", None)
            child.task["object"] = str(Path(mesh).resolve())
            child.task["templates"] = [name]
            entries = run(child, destination, initialization_reference=initialization_reference,
                          primitive_override=primitive)
            report = json.loads((destination / "summary.json").read_text())
            summary = report["cases"][0]
            summary.update(primitive=primitive, index=index, case_id=relative.name)
            case = destination / f"000_{name}"
            write_json(case / "summary.json", summary)
            for entry in entries:
                entry.update(primitive=primitive, name=relative.name)
                for key in ("model", "trajectory", "summary"):
                    entry[key] = str(relative / entry[key])
            return dict(index=index, entries=entries, summary=summary)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            for target, source in zip((1, 2), saved):
                os.dup2(source, target)
                os.close(source)


def run_parallel(config, output, args):
    import torch
    from .primitives import jobs
    from .template_mppi_runner import OBJECTIVE

    if not torch.cuda.is_available():
        raise RuntimeError("Template MPPI requires CUDA; no GPU solve was performed")
    settings = options(config.environment["template_mppi"]["parallel"])
    output = Path(output)
    if config.task.get("object"):
        tasks = [(Path(config.task["object"]), name, None) for name in args.templates]
        skipped = []
    else:
        tasks, skipped = jobs(output, args.templates, args)
    write_json(output / "skipped.json", skipped)
    if not tasks:
        raise ValueError("The requested selection has no supported single-object cases")
    free, total = torch.cuda.mem_get_info(config.device)
    budget = int(free * settings["memory_fraction"])
    started = time.perf_counter()
    resource = dict(device=torch.cuda.get_device_name(config.device), total_bytes=total,
                    baseline_free_bytes=free, budget_bytes=budget, **settings,
                    peak_incremental_bytes=0, minimum_free_bytes=free,
                    memory_scope="whole device, relative to free memory before workers; includes CUDA contexts and Warp",
                    response_backend=args.settings.objective.closure.equilibrium_backend,
                    physical_evaluation=False)
    from retarget.common.provenance import ROOT, sha256
    sources = list((ROOT / "retarget").rglob("*.py")) + [ROOT / "run.py", config.source]
    write_json(output / "source_sha256_at_start.json", {str(path): sha256(path) for path in sources})
    completed = {}

    def sample_memory():
        available, _ = torch.cuda.mem_get_info(config.device)
        resource["minimum_free_bytes"] = min(resource["minimum_free_bytes"], available)
        resource["peak_incremental_bytes"] = max(resource["peak_incremental_bytes"], free - available)
        resource["peak_fraction_of_initial_free"] = resource["peak_incremental_bytes"] / free

    def save(state="running"):
        ordered = [completed[i] for i in sorted(completed)]
        manifest = [entry for result in ordered for entry in result["entries"]]
        rows = [result["summary"] for result in ordered]
        write_json(output / "manifest.json", dict(version=1, task="grasp", method="mppi", cases=manifest))
        write_json(output / "summary.json", dict(
            method="mppi", objective=OBJECTIVE, planned=len(tasks), finished=len(rows),
            solver_completed=sum(bool(row["solver_completed"]) for row in rows),
            native_evaluated=0, successes=sum(bool(row["task_success"]) for row in rows), cases=rows,
            acceptance="Projected friction closure and separate self-penetration limit; hand-object compression softly penalized; coverage diagnostic only; no physical evaluation"))
        write_json(output / "resources.json", resource)
        write_json(output / "status.json", dict(state=state, planned=len(tasks), finished=len(rows),
                                                wall_seconds=time.perf_counter() - started))
        return manifest

    # Calibrate one actual case per contact count, keeping every result. This
    # allows light pinches to run at higher concurrency than 16-contact grasps.
    import yaml
    from retarget.task.grasp.dexonomy import annotation_entries
    counts = {}
    for _, name, _ in tasks:
        if name not in counts:
            path = args.dexonomy_root / "assets/hand/shadow/raw_anno" / f"{name}.yaml"
            data = yaml.safe_load(path.read_text())
            counts[name] = sum(len(annotation_entries(value)) for value in data["contact"].values())
    representatives = {}
    for i, (mesh, name, _) in enumerate(tasks):
        n = counts[name]
        previous = representatives.get(n)
        if previous is None or Path(mesh).stat().st_size > Path(tasks[previous][0]).stat().st_size:
            representatives[n] = i
    context = mp.get_context("spawn")
    profiles = {}
    save("initializing_templates")
    # Fit once per template/shape at exact 1x before any size-specific solve.
    # A separate process releases all matcher allocations before calibration.
    from .template_mppi_reference import prepare_all_references
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        preparation = pool.submit(prepare_all_references, config, str(output), tasks)
        while not preparation.done():
            wait([preparation], timeout=.2)
            sample_memory()
        references = preparation.result()
    resource["template_initialization_seconds"] = time.perf_counter() - started
    if resource["peak_incremental_bytes"] > budget:
        save("failed")
        raise RuntimeError("Canonical template matching exceeded the CUDA memory budget")
    save()
    print(f"[template_mppi_parallel] {len(tasks)} cases, {len(skipped)} unsupported; "
          f"CUDA budget={budget / 2**30:.2f} GiB; calibrating contact-count profiles", flush=True)
    for n in sorted(representatives, reverse=True):
        i = representatives[n]
        local_free = torch.cuda.mem_get_info(config.device)[0]
        minimum = local_free
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
            future = pool.submit(_solve, config, str(output), i, *tasks[i], references[i])
            while not future.done():
                wait([future], timeout=.1)
                sample_memory()
                minimum = min(minimum, torch.cuda.mem_get_info(config.device)[0])
            result = future.result()
            completed[i] = result
            if not result["summary"]["solver_completed"]:
                save("failed")
                raise RuntimeError(f"CUDA calibration solve failed: {result['summary'].get('error')}")
        measured = local_free - minimum
        worker_count(budget, measured, settings["max_workers"], settings["safety_factor"])
        profiles[n] = measured
        resource["calibration_profiles"] = {str(k): dict(peak_bytes=v, case=representatives[k])
                                             for k, v in profiles.items()}
        print(f"[template_mppi_parallel] calibration {len(completed)}/{len(tasks)} "
              f"n={n}, peak={measured / 2**30:.2f} GiB", flush=True)
        save()
    remaining = [i for i in range(len(tasks)) if i not in completed]
    workers = min(settings["max_workers"], len(remaining) or 1)
    resource.update(maximum_workers=workers, peak_active_workers=0,
                    scheduling="sum of measured per-contact-count worker peaks times safety_factor <= budget",
                    worker_lifetime="one case; releases context and compiled workspaces on exit")
    save()
    # One case per child prevents resident contexts/workspaces from growing as
    # different shapes visit a worker. Admission uses useful predicted work,
    # and actual device-free memory also gates launching the next group.
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, max_tasks_per_child=1) as pool:
        pending = {}
        while remaining or pending:
            committed = sum(profiles[counts[tasks[i][1]]] * settings["safety_factor"]
                            for i in pending.values())
            while remaining and len(pending) < workers:
                available = torch.cuda.mem_get_info(config.device)[0]
                if available < free - budget:
                    break
                eligible = next((i for i in remaining if committed
                    + profiles[counts[tasks[i][1]]] * settings["safety_factor"] <= budget), None)
                if eligible is None:
                    break
                remaining.remove(eligible)
                pending[pool.submit(_solve, config, str(output), eligible, *tasks[eligible],
                                    references[eligible])] = eligible
                committed += profiles[counts[tasks[eligible][1]]] * settings["safety_factor"]
            if not pending:
                raise RuntimeError("GPU memory budget cannot admit an unfinished case; other device allocations may have changed")
            resource["peak_active_workers"] = max(resource["peak_active_workers"], len(pending))
            resource["active_workers"] = len(pending)
            done, _ = wait(pending, timeout=.2, return_when=FIRST_COMPLETED)
            sample_memory()
            for future in done:
                i = pending.pop(future)
                completed[i] = future.result()
                row = completed[i]["summary"]
                print(f"[template_mppi_parallel] {len(completed)}/{len(tasks)} {tasks[i][1]} "
                      f"solver={row['solver_completed']} geometry={row['task_success']} "
                      f"time={row.get('optimization_seconds')}s active={len(pending)}", flush=True)
                save()
    sample_memory()
    resource.update(wall_seconds=time.perf_counter() - started, active_workers=0)
    return save("completed")
