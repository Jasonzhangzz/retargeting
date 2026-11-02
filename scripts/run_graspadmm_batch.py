#!/usr/bin/env python3
"""Resumable 31-type / 161-size / ten-seed native GraspADMM experiment."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retarget.common.results import write_json

PROTOCOL = "graspadmm_native_v1"
TERMINAL = {"completed", "failed", "timeout"}
STOP = threading.Event()
ACTIVE_CHILDREN = {}
ACTIVE_LOCK = threading.Lock()


def read(path):
    return json.loads(Path(path).read_text())


def sample_jobs(seeds=range(10), types=None):
    from retarget.env.grasp.primitives import catalog, size_specifications
    seeds = tuple(seeds)
    if (not seeds or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be distinct nonnegative integers and nonempty")
    jobs, skipped = [], []
    selected = None if types is None else set(map(int, types))
    inventory = catalog(ROOT / "data/grasp/templates")
    if selected is not None and selected - {r["type_id"] for r in inventory}:
        raise ValueError(f"Unknown grasp types: {sorted(selected - {r['type_id'] for r in inventory})}")
    for row in inventory:
        if selected is not None and row["type_id"] not in selected:
            continue
        if not row["supported"]:
            skipped.append(dict(type_id=row["type_id"], template=row["template"],
                                reason="articulated_or_multi_object_asset_required"))
            continue
        specs = size_specifications(row["type_id"], row["families"][0])
        for size_index, primitive in enumerate(specs):
            for seed in seeds:
                jobs.append(dict(id=f"t{row['type_id']:02d}_{primitive['kind']}_s{size_index:02d}_seed{seed:02d}",
                    type_id=row["type_id"], template=row["template"], primitive=primitive,
                    size_index=size_index, seed=int(seed)))
    return jobs, skipped


def make_batch(path, *, seeds=range(10), types=None, timeout=1800., workers=1):
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("workers and timeout must be finite and positive")
    seeds = tuple(seeds)
    path = Path(path).resolve()
    if path.parent != ROOT / "output" or path.exists() or path.is_symlink():
        raise ValueError("Use a new direct child of output/; existing runs are never replaced")
    jobs, skipped = sample_jobs(seeds, types)
    if not jobs:
        raise ValueError("The selection has no supported primitive trials")
    stack = numerical_stack(ROOT/".venv-graspadmm/bin/python")
    path.mkdir(parents=True)
    runtime = path / "runtime"
    runtime.mkdir()
    for name in ("retarget", "third_party"):
        shutil.copytree(ROOT/name, runtime/name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
    (runtime/"scripts").mkdir()
    shutil.copy2(__file__, runtime/"scripts"/Path(__file__).name)
    for name in ("run.py", "requirements-graspadmm.txt"):
        shutil.copy2(ROOT/name, runtime/name)
    for name in ("data", "licenses", ".venv-graspadmm"):
        (runtime/name).symlink_to(ROOT/name, target_is_directory=True)
    source_files = [runtime/"run.py", runtime/"requirements-graspadmm.txt"]
    for name in ("retarget", "third_party", "scripts"):
        source_files.extend(p for p in (runtime/name).rglob("*") if p.is_file())
    source_hashes = {str(p.relative_to(runtime)): digest(p) for p in source_files}
    input_files = [ROOT/"data/grasp/model.yaml"]
    for name in ("assets/wuji2", "templates"):
        input_files.extend(p for p in (ROOT/"data/grasp"/name).rglob("*") if p.is_file())
    input_hashes = {str(p): digest(p) for p in input_files if p.is_file()}
    python = str(runtime/".venv-graspadmm/bin/python")
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(), runtime=str(runtime),
                protocol=PROTOCOL, jobs=jobs, skipped=skipped, workers=workers,
                timeout_seconds=timeout, source_hashes=source_hashes, input_hashes=input_hashes,
                python=python, numerical_stack=stack, physical_model="wuji2",
                trial_policy=dict(epoch=1, n_final=1, input_policy="taxonomy", template_feedback=False,
                                  failure_resampling=False, seeds=list(seeds)))
    write_json(path/"plan.json", plan)
    for job in jobs:
        write_json(path/"jobs"/job["id"]/"status.json", dict(state="queued"))
    refresh(path)
    return plan


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numerical_stack(python):
    """Read package metadata in the worker environment without importing CUDA."""
    packages = ("torch", "numpy", "scipy", "mujoco", "warp-lang", "trimesh", "qpsolvers", "clarabel")
    script = ("import importlib.metadata as m,json,platform; "
              f"packages={packages!r}; "
              "print(json.dumps(dict(python=platform.python_version(), "
              "**{name:m.version(name) for name in packages})))")
    result = subprocess.run([str(python), "-c", script], check=True,
                            capture_output=True, text=True, timeout=30.)
    return json.loads(result.stdout)


def verify_snapshot(plan):
    """Reject drift in copied source or the shared hand/template assets."""
    runtime = Path(plan["runtime"])
    for relative, expected in plan.get("source_hashes", {}).items():
        path = runtime/relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Frozen runtime changed: {relative}")
    for filename, expected in plan.get("input_hashes", {}).items():
        path = Path(filename)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Shared experiment asset changed: {filename}")
    if "numerical_stack" in plan:
        actual = numerical_stack(plan["python"])
        if actual != plan["numerical_stack"]:
            changed = sorted(name for name in actual.keys() | plan["numerical_stack"].keys()
                             if actual.get(name) != plan["numerical_stack"].get(name))
            raise ValueError(f"Frozen numerical stack changed: {', '.join(changed)}")


def job_work(batch, job):
    base = Path(batch)/"jobs"/job["id"]
    return Path(read(base/"status.json").get("work", base))


def worker(batch, job_id):
    import yaml
    from retarget.config.run import load
    from retarget.env.grasp import grasp_admm_runner

    batch = Path(batch).resolve()
    plan = read(batch/"plan.json")
    job = next(j for j in plan["jobs"] if j["id"] == job_id)
    work = job_work(batch, job)
    output = work/"solve"
    output.mkdir(exist_ok=True)
    p = job["primitive"]
    raw = dict(mode="solve", task=dict(type="grasp", name=job_id,
        data=str(ROOT/"data/grasp"), templates=[job["template"]],
        radius_scales=[1.], height_scales=[1.],
        options=dict(hand_model="wuji2", primary_shape_only=True, primitive_seed_policy="taxonomy",
                     primitive=p["kind"], radius=p["radius_m"], height=p["height_m"])),
        solver=dict(type="graspadmm", graspadmm=dict(init_epoch=1, init_n_final=1)),
        environment=dict(device="cuda:0", dynamics="graspadmm"),
        seed=job["seed"], output=dict(name=job_id), visualization=dict(save_video=False, live=False))
    cfg_path = work/"config.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    config = load(cfg_path)
    grasp_admm_runner.validate(config)
    cases = grasp_admm_runner.run(config, output)
    if len(cases) != 1:
        raise ValueError("Each trial must export exactly one candidate")
    for case in cases:
        case.update(primitive=p, size_scale=p["size_scale"], size_axis=p["size_axis"],
                    radius_scale=p["radial_scale"], seed=job["seed"], case_id=job_id)
    write_json(output/"manifest.json", dict(version=1, task="grasp", method="graspadmm", cases=cases))
    return cases


def stop_child(child):
    """A timed out trial must release its process group and CUDA context."""
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=2.)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def execute_job(batch, job, plan):
    base = batch/"jobs"/job["id"]
    status = base/"status.json"
    if STOP.is_set() or read(status).get("state") in TERMINAL:
        return
    started = time.perf_counter()
    number = max((int(p.name[8:]) for p in base.glob("attempt_[0-9]*") if p.name[8:].isdigit()), default=-1)+1
    work = base/f"attempt_{number:03d}"
    work.mkdir()
    running = dict(state="running", started_at=datetime.now(timezone.utc).isoformat(), work=str(work), attempt=number)
    write_json(status, running)
    command = [plan.get("python", str(ROOT/".venv-graspadmm/bin/python")),
               str(Path(plan["runtime"])/"scripts"/Path(__file__).name),
               "--worker", str(batch), "--job", job["id"]]
    env = {**os.environ, "MUJOCO_GL": "egl", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1"}
    try:
        with (work/"run.log").open("w") as log:
            child = subprocess.Popen(command, cwd=plan["runtime"], env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            with ACTIVE_LOCK:
                ACTIVE_CHILDREN[child.pid] = child
            try:
                write_json(status, dict(running, child_pid=child.pid))
                if STOP.is_set():
                    stop_child(child)
                child.wait(timeout=plan["timeout_seconds"])
            except BaseException:
                stop_child(child)
                raise
            finally:
                with ACTIVE_LOCK:
                    ACTIVE_CHILDREN.pop(child.pid, None)
        state = "completed" if child.returncode == 0 else "failed"
        detail = dict(returncode=child.returncode)
    except subprocess.TimeoutExpired:
        state, detail = "timeout", dict(reason="trial_timeout")
    except Exception as exc:
        state, detail = "failed", dict(reason=f"{type(exc).__name__}: {exc}")
    if STOP.is_set():
        state, detail = "queued", dict(reason="supervisor_interrupted", interrupted_attempt=str(work))
    write_json(status, dict(running, state=state, child_pid=None,
                           wall_seconds=time.perf_counter()-started, **detail))


def refresh(batch):
    batch = Path(batch)
    plan = read(batch/"plan.json")
    rows, cases = [], []
    for job in plan["jobs"]:
        status = read(batch/"jobs"/job["id"]/"status.json")
        work = Path(status.get("work", batch/"jobs"/job["id"]))
        entry = dict(job, **status, task_success=False)
        manifest = work/"solve/manifest.json"
        if manifest.is_file() and status["state"] in TERMINAL:
            exported = read(manifest).get("cases", [])
            if len(exported) != 1:
                entry["failure_stage"] = "single_candidate_contract"
            for c in exported if len(exported) == 1 else []:
                c = dict(c)
                summary = work/"solve"/c["summary"]
                info = read(summary) if summary.is_file() else {}
                compatible = info.get("evaluation_protocol") == PROTOCOL
                entry.update(task_success=(status["state"] == "completed" and compatible
                                          and info.get("task_success") is True),
                    evaluation_protocol=info.get("evaluation_protocol"),
                    failure_stage=(info.get("failure_stage", info.get("reason"))
                                   or ("evaluation_protocol" if not compatible else None)),
                    fallback=info.get("fallback", info.get("fallback_used", False)))
                for key in ("model", "trajectory", "summary"):
                    if key in c:
                        source = Path(c[key])
                        c[key] = str(source if source.is_absolute() else (work/"solve"/source).relative_to(batch))
                c.update(primitive=job["primitive"], size_scale=job["primitive"]["size_scale"],
                         size_axis=job["primitive"]["size_axis"], radius_scale=job["primitive"]["radial_scale"],
                         seed=job["seed"], case_id=job["id"])
                cases.append(c)
        if entry["state"] in TERMINAL and not entry.get("failure_stage") and not entry["task_success"]:
            entry["failure_stage"] = (entry.get("reason") or "runtime") if entry["state"] != "completed" else "native_evaluation"
        rows.append(entry)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["type_id"], row["size_index"])].append(row)
    table = []
    for key, values in sorted(grouped.items()):
        first = values[0]
        n = sum(v["state"] in TERMINAL for v in values)
        successes = sum(v["task_success"] for v in values if v["state"] in TERMINAL)
        table.append(dict(type_id=key[0], template=first["template"], size_index=key[1],
            **first["primitive"], trials=len(values), finished=n, successes=successes,
            success_rate=successes/len(values), completed_success_rate=successes/n if n else None))
    counts = Counter(r["state"] for r in rows)
    complete = all(r["state"] in TERMINAL for r in rows)
    result = dict(state="completed" if complete else "stopped" if STOP.is_set() else "running", evaluation_protocol=PROTOCOL,
        planned=len(rows), finished=sum(counts[s] for s in TERMINAL), states=dict(counts),
        successes=sum(r["task_success"] for r in rows),
        success_rate=sum(r["task_success"] for r in rows)/len(rows) if rows else 0.,
        failure_stages=dict(Counter(r.get("failure_stage") for r in rows if r["state"] in TERMINAL and not r["task_success"])),
        skipped=plan["skipped"], by_type_size=table, jobs=rows)
    write_json(batch/"summary.json", result)
    write_json(batch/"status.json", {k: result[k] for k in ("state", "planned", "finished", "successes", "states")})
    write_json(batch/"manifest.json", dict(version=1, task="grasp", method="graspadmm", cases=cases))
    columns = ["type_id", "template", "size_index", "kind", "size_axis", "size_scale",
               "radius_m", "height_m", "trials", "finished", "successes", "success_rate"]
    with (batch/"success_rates.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(table)
    return result


def run(batch):
    batch = Path(batch).resolve()
    plan = read(batch/"plan.json")
    with (batch/"supervisor.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("This batch already has an active supervisor") from exc
        verify_snapshot(plan)
        recover_interrupted(batch, plan)
        STOP.clear()
        previous_handlers = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[sig] = signal.signal(sig, request_stop)
        try:
            with ThreadPoolExecutor(max_workers=plan["workers"]) as pool:
                futures = [pool.submit(execute_job, batch, job, plan) for job in plan["jobs"]]
                from concurrent.futures import as_completed
                last_refresh = time.monotonic()
                for future in as_completed(futures):
                    if STOP.is_set():
                        for pending in futures:
                            pending.cancel()
                        break
                    future.result()
                    if STOP.is_set():
                        for pending in futures:
                            pending.cancel()
                        break
                    now = time.monotonic()
                    if now - last_refresh >= 5.:
                        report = refresh(batch)
                        print(f"{report['finished']}/{report['planned']} success={report['successes']}", flush=True)
                        last_refresh = now
            return refresh(batch)
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)


def request_stop(*_):
    STOP.set()
    with ACTIVE_LOCK:
        children = list(ACTIVE_CHILDREN.values())
    for child in children:
        stop_child(child)


def recover_interrupted(batch, plan):
    """Keep interrupted attempts, terminate surviving matching children, resume seeds.

    Failed and timed-out trials are terminal and are never resampled. A new
    attempt is reserved for a supervisor interruption while a trial was running.
    """
    script = str(Path(plan["runtime"])/"scripts"/Path(__file__).name)
    for job in plan["jobs"]:
        path = batch/"jobs"/job["id"]/"status.json"
        status = read(path)
        if status.get("state") != "running":
            continue
        pid = status.get("child_pid")
        if pid:
            try:
                command = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
                expected = [script, "--worker", str(batch), "--job", job["id"]]
                if command[1:6] == expected and os.getpgid(pid) == pid:
                    os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, FileNotFoundError):
                pass
        write_json(path, dict(status, state="queued", child_pid=None,
                              interrupted_attempt=status.get("work")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="graspadmm_all_types_10seeds")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--types", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800.)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--job")
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args.worker, args.job)
        except BaseException:
            traceback.print_exc()
            raise
        return
    if args.workers < 1 or not math.isfinite(args.timeout) or args.timeout <= 0 or any(s < 0 for s in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("workers/timeout must be positive; seeds must be distinct nonnegative integers")
    batch = args.resume or ROOT/"output"/args.output
    if not args.resume:
        make_batch(batch, seeds=args.seeds, types=args.types, timeout=args.timeout, workers=args.workers)
    else:
        frozen_script = Path(read(batch/"plan.json")["runtime"])/"scripts"/Path(__file__).name
        if frozen_script.resolve() != Path(__file__).resolve():
            os.execv(sys.executable, [sys.executable, str(frozen_script), "--resume", str(batch.resolve())])
    run(batch)


if __name__ == "__main__":
    main()
