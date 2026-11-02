"""Detached, resumable CA -> SPIDER MPPI sweep with independent job processes.

Launch: .venv-contact/bin/python scripts/run_spider_grasp_batch.py --launch
Inspect: output/<batch>/status.json, summary.json, jobs/<id>/status.json.
Each worker slot prepares CA in one process, releases its CUDA context, then
runs the SPIDER backend in another process. Finite CA failures are proposals,
not accepted tracking references. Runtime code is copied once; previous results
stay untouched. Use --pilot before promoting to the complete sweep.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
import hashlib
import json
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
STOP = threading.Event()
TERMINAL = {"completed", "ca_failed", "input_invalid", "solve_failed", "timeout"}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_jobs(scales=None, cylinder_length=.18):
    """Use the same per-type primitive policy as ordinary grasp inputs."""
    from retarget.env.grasp.primitives import CATALOG, catalog, size_specifications
    if scales is not None and (len(scales) != 5 or len(set(scales)) != 5
            or any(not 0 < value < float("inf") for value in scales)):
        raise ValueError("Specify exactly five distinct finite positive radius scales")
    if not 0 < cylinder_length < float("inf"):
        raise ValueError("Cylinder length must be finite and positive")
    jobs, skipped = [], []
    for row in catalog(ROOT/"data/grasp/templates"):
        family, base_radius, base_height = CATALOG[row["type_id"]][0]
        if base_radius is None:
            skipped.append(dict(type_id=row["type_id"], template=row["template"], family=family,
                scales=list(scales) if scales is not None else None,
                reason="articulated or multi-object asset required; no single-primitive substitution"))
            continue
        specifications = size_specifications(row["type_id"], family,
            radius_scales=scales, explicit=scales is not None, cylinder_length=cylinder_length,
            height=cylinder_length if scales is not None and family == "cylinder" and row["type_id"] <= 6 else None)
        for index, primitive in enumerate(specifications):
            jobs.append(dict(id=f"t{row['type_id']:02d}_{family}_s{index:02d}",
                template=row["template"], type_id=row["type_id"], size_index=index,
                primitive=primitive, previous_catalog_height_m=base_height))
    jobs.sort(key=lambda j: (abs(j["primitive"]["size_scale"]-1.), j["size_index"],
                             j["type_id"], j["primitive"]["kind"]))
    return jobs, skipped


PILOT_TYPES = {1, 5, 6, 10, 15, 23, 26, 31}


def promotion_gate(batch):
    """Promote using the shared native GraspADMM evaluation protocol."""
    batch=Path(batch).resolve();plan=read(batch/'plan.json');passed={};pending=[]
    inventory, _ = sample_jobs(plan.get("radius_scales"), plan.get("long_cylinder_height_m", .18))
    expected = set()
    for type_id in PILOT_TYPES:
        indices = sorted(j["size_index"] for j in inventory if j["type_id"] == type_id)
        expected.update((type_id, i) for i in (indices[0], indices[len(indices)//2], indices[-1]))
    actual={(j['type_id'],j['size_index']) for j in plan['jobs']}
    for job in plan['jobs']:
        status=read(batch/'jobs'/job['id']/'status.json')
        if status['state'] not in TERMINAL:pending.append(job['id']);continue
        if not status.get('work'):continue
        output=Path(status['work'])/'solve'
        manifest=output/'manifest.json'
        if not manifest.is_file():continue
        rows=read(manifest)['cases']
        if len(rows)!=1:continue
        case=(output/rows[0]['summary']).parent
        summary=case/'summary.json'
        if not summary.is_file():continue
        result=read(summary)
        if (result.get('evaluation_protocol') == 'graspadmm_native_v1'
                and result.get('task_success') is True):
            passed.setdefault(job['primitive']['kind'],[]).append(job['id'])
    missing=sorted({'cylinder','sphere','disk'}-passed.keys())
    return dict(eligible=expected<=actual and not pending and not missing,
        pilot=str(batch),complete_pilot=expected<=actual, pending=pending,
        successful_cases_by_family=passed,missing_families=missing,
        policy='24 pilot cases complete; cylinder, sphere and disk each require graspadmm_native_v1 success')


def make_batch(path, workers=2, timeout=1800., scales=None, cylinder_length=.18,
               *,pilot=False,cached_batch=None):
    import yaml
    if workers < 1 or timeout <= 0:
        raise ValueError("Worker count and per-stage timeout must be positive")
    jobs, skipped = sample_jobs(scales, cylinder_length)
    if pilot:
        selected = set()
        for type_id in PILOT_TYPES:
            indices = sorted(j["size_index"] for j in jobs if j["type_id"] == type_id)
            selected.update((type_id, i) for i in (indices[0], indices[len(indices)//2], indices[-1]))
        jobs = [j for j in jobs if (j["type_id"], j["size_index"]) in selected]
    if cached_batch is not None:
        cached_batch=Path(cached_batch).resolve()
        cached_plan = read(cached_batch / "plan.json") if (cached_batch / "plan.json").is_file() else {}
        cached_jobs = {j["id"]: j for j in cached_plan.get("jobs", [])}
        for job in jobs:
            previous = cached_jobs.get(job["id"], {}).get("primitive", {})
            fields = ("kind", "radius_m", "height_m", "size_policy_version")
            if any(previous.get(key) != job["primitive"].get(key) for key in fields):
                continue
            status=cached_batch/'jobs'/job['id']/'status.json'
            if status.is_file():
                work=read(status).get('work')
                if work and (Path(work)/'ca_seed').is_dir():job['cached_work']=work
    path = Path(path).resolve()
    if path.parent != ROOT/"output" or path.exists() or path.is_symlink():
        raise ValueError("Use a new direct child of the repository output directory")
    path.mkdir(parents=True)
    runtime = path/"runtime"
    runtime.mkdir()
    shutil.copytree(ROOT/"retarget", runtime/"retarget", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (runtime/"scripts").mkdir()
    shutil.copy2(__file__, runtime/"scripts"/Path(__file__).name)
    for name in ("data", "third_party", "licenses"):
        (runtime/name).symlink_to(ROOT/name, target_is_directory=True)
    ca = yaml.safe_load((ROOT/"configs/grasp_contactaware_mppi.yaml").read_text())
    ca["task"]["data"] = str(ROOT/"data/grasp")
    ca["task"]["options"]["dexonomy_root"] = str(ROOT/"data/grasp/templates")
    ca["task"]["options"]["ca_as_seed"] = True
    intent = path/"grasp_pose_intent_wuji2.json"
    shutil.copy2(ROOT/"configs/grasp_pose_intent_wuji2.json", intent)
    ca["task"]["options"]["grasp_intent_reference"] = str(intent)
    ca["task"]["options"]["contactaware"].update(bone_max_deviation_deg=45., bone_style_weight=.02,
        pose_rotation_limit_deg=15., pose_rotation_weight=1.)
    ca["visualization"] = dict(live=False, save_video=False)
    ca["output"]["name"] = path.name
    spider = yaml.safe_load((ROOT/"configs/grasp_spider_acquisition_mppi.yaml").read_text())
    spider["output"]["name"] = path.name
    for name, config in (("ca.yaml", ca), ("spider.yaml", spider)):
        (path/name).write_text(yaml.safe_dump(config, sort_keys=False))
    source_files = [*runtime.rglob("*.py"), *runtime.rglob("*.json")]
    # Never follow asset/vendor symlinks when snapshotting source fingerprints.
    source_files = [p for p in source_files if p.is_relative_to(runtime/"retarget") or p.is_relative_to(runtime/"scripts")]
    plan = dict(version=1, created_utc=stamp(), workers=workers, stage_timeout_seconds=timeout,
        original_repository=str(ROOT), runtime=str(runtime), python=sys.executable,
        radius_scales=list(scales) if scales is not None else None, size_policy_version=2, long_cylinder_types=list(range(1, 7)),
        long_cylinder_height_m=cylinder_length, total_jobs=len(jobs), jobs=jobs,
        grasp_intent_sha256=digest(intent),
        ca_is_seed=True,
        skipped=skipped, requested_type_count=8 if pilot else 33, runnable_type_count=8 if pilot else 31,
        scope='pilot' if pilot else 'full',cached_batch=str(cached_batch) if cached_batch else None,
        ca_policy="finite CA proposal only; no strict CA gate or refinement; fixed-object MPPI then free-object hold",
        strict_ca_refinement_enabled=False,
        final_ca=dict(contact_tolerance_mm=.2, penetration_tolerance_mm=.5, normal_cosine=.5),
        code_sha256={str(p.relative_to(runtime)): digest(p) for p in source_files},
        template_sha256={row["template"]: digest(ROOT/"data/grasp/templates/assets/hand/shadow/raw_anno"/(row["template"]+".yaml"))
            for row in __import__("retarget.env.grasp.primitives", fromlist=["catalog"]).catalog(ROOT/"data/grasp/templates")})
    write(path/"plan.json", plan)
    for job in jobs:
        write(path/"jobs"/job["id"]/"status.json", dict(id=job["id"], state="queued", updated_utc=stamp()))
    write(path/"status.json", dict(state="queued", total=len(jobs), workers=workers, completed=0, failed=0))
    return plan


def prepare(batch, job, work):
    import numpy as np
    import torch
    import warp as wp
    import yaml
    from retarget.common.results import write_json
    from retarget.config.run import load
    from retarget.env.grasp.runner import validate
    from retarget.env.grasp.inputs import initialize_cases
    from retarget.env.grasp.runtime import prepare_experiment
    from retarget.env.grasp.primitives import primitive_mesh
    from retarget.env.grasp.forceaware_ca_refinement import refine_saved_reference
    from retarget.env.grasp.spider_reference import read_input
    torch.set_num_threads(1)
    wp.init(); wp.set_device("cuda:0")
    config = read_yaml(batch/"ca.yaml")
    config["task"]["templates"] = [job["template"]]
    config["task"]["radius_scales"] = [job["primitive"]["radial_scale"]]
    config["task"]["height_scales"] = [1.]
    source = work/"ca_config.yaml"
    source.write_text(yaml.safe_dump(config, sort_keys=False))
    args = validate(load(source))
    primitive = job["primitive"]
    mesh = primitive_mesh(work/(job["id"]+".obj"), primitive["kind"], primitive["radius_m"], primitive["height_m"])
    args._tasks = [(mesh, job["template"], primitive)]
    if args.ca_as_seed and job.get('cached_work'):
        from retarget.env.grasp.spider_seed import cached_proposal,export_seed
        hand,guess=cached_proposal(job['cached_work'],mesh,job['template'],primitive,args)
        seed=export_seed(work/'spider_seed',hand,guess,args,template=job['template'],primitive=primitive)
        write(work/'prepared.json',dict(state='ready',source=str(seed),ca_is_gate=False,ca_success=guess.metadata['contactaware'].get('success',False),primitive=primitive))
        return 0
    hand, guesses, case, xml = next(initialize_cases(args, work/"ca_seed"))
    write_json(case/"initialization.json", [g.metadata for g in guesses])
    if args.ca_as_seed:
        from retarget.env.grasp.spider_seed import export_seed
        finite=[g for g in guesses if np.asarray(g.hand_qpos).shape==(26,) and np.isfinite(g.hand_qpos).all()]
        if not finite:raise ValueError('input_invalid: no finite geometric/CA proposal')
        seed=export_seed(work/'spider_seed',hand,finite[0],args,template=job['template'],primitive=primitive)
        write(work/'prepared.json',dict(state='ready',source=str(seed),ca_is_gate=False,ca_success=finite[0].metadata['contactaware'].get('success',False),primitive=primitive))
        return 0
    good = [g for g in guesses if g.metadata.get("initialization_valid", False)]
    if not good:
        write(work/"prepared.json", dict(state="ca_failed", reason="initial_CA_not_accepted", diagnostics=str(case/"contactaware.json")))
        return 3
    experiment = prepare_experiment(case/"input_00", hand, good[0], args)
    # Reuse the already-tested refinement stage used for the three-size inputs.
    final = refine_saved_reference(experiment, work/"ca_final", **read(batch/"plan.json")["final_ca"])
    report = read(final/"contactaware_reference.json")
    if report.get("success") is not True:
        write(work/"prepared.json", dict(state="ca_failed", reason="strict_CA_refinement_not_accepted", diagnostics=str(final/"contactaware_reference.json")))
        return 3
    ref = read_input(final)
    write(work/"prepared.json", dict(state="ready", source=str(ref.source), source_sha256=ref.source_hashes,
        active_slots=ref.active_slots.tolist(), ca_reference_qpos=ref.qpos.tolist(), primitive=primitive))
    return 0


def read_yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text())


def solve_job(batch, job, work):
    import torch
    import yaml
    from retarget.common.results import write_json
    from retarget.common.provenance import record
    from retarget.config.run import load
    from retarget.env.grasp import spider_runner
    torch.set_num_threads(1)
    prepared = read(work/"prepared.json")
    if prepared["state"] != "ready":
        raise ValueError("MPPI cannot run without accepted CA")
    raw = read_yaml(batch/"spider.yaml")
    raw["task"]["data"] = prepared["source"]
    cfg_path = work/"spider_config.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    config = load(cfg_path)
    output = work/"solve"
    # Preflight only; a failed static proposal falls back to the existing,
    # explicit CA-configuration control initialization. CA state stays fixed.
    from retarget.env.grasp.spider_reference import read_input, prepare_model, preload
    reference = read_input(prepared["source"])
    if reference.reference_kind == 'ca_proposal':
        raw['environment']['spider']['preload']=True
        cfg_path.write_text(yaml.safe_dump(raw,sort_keys=False));config=load(cfg_path)
        policy=dict(preload=True,policy='optional bounded load warm start after MPPI acquisition; CA not a gate')
    else:
        policy=None
    if policy is None:
        model, path = prepare_model(reference, work/"preload_probe", .002)
        try:
            preload(reference, path, 5)
            policy = dict(preload=True, fallback=False)
        except ValueError as error:
            if "no feasible bounded static control initialization" not in str(error):raise
            raw["environment"]["spider"]["preload"] = False
            cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False));config = load(cfg_path)
            policy = dict(preload=False, fallback=True, reason=str(error),policy="CA qpos control seed")
    write(work/"control_seed_policy.json", policy)
    output.mkdir()
    (output/"config.yaml").write_text(yaml.safe_dump(config.resolved(), sort_keys=False))
    tick = time.monotonic()
    cases = spider_runner.run(config, output)
    write_json(output/"provenance.json", record(config, cases, output))
    write(output/"manifest.json", dict(version=1, task="grasp", method="mppi", cases=cases))
    write(output/"status.json", dict(state="completed", wall_seconds=time.monotonic()-tick))
    return 0


def run_child(batch, job, work, mode, timeout, update):
    runtime = Path(read(batch/"plan.json")["runtime"])
    env = os.environ.copy()
    env.update(PYTHONPATH=str(runtime), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MUJOCO_GL="egl", PYTHONUNBUFFERED="1", FORCEAWARE_DEVICE="cuda:0")
    command = [sys.executable, str(runtime/"scripts"/Path(__file__).name), "--"+mode,
               "--batch", str(batch), "--job", job["id"], "--work", str(work)]
    log = work/(mode+".log")
    tick = time.monotonic()
    with log.open("w") as stream:
        child = subprocess.Popen(command, cwd=runtime, env=env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        update(state="preparing_ca" if mode == "prepare" else "solving", child_pid=child.pid,
               stage=mode, log=str(log), stage_started_utc=stamp())
        while child.poll() is None:
            if STOP.is_set() or time.monotonic()-tick > timeout:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10.)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
                return "cancelled" if STOP.is_set() else "timeout"
            time.sleep(.5)
    return child.returncode


def execute_job(batch, job, timeout):
    directory = batch/"jobs"/job["id"]
    old = read(directory/"status.json")
    if old.get("state") in TERMINAL:
        return
    if STOP.is_set():
        return
    attempt = len(list(directory.glob("attempt_*")))
    work = directory/f"attempt_{attempt:03d}"
    work.mkdir()
    status = dict(id=job["id"], type_id=job["type_id"], template=job["template"],
        primitive=job["primitive"], work=str(work), started_utc=stamp(), supervisor_pid=os.getpid())
    def update(**fields):
        status.update(fields, updated_utc=stamp())
        write(directory/"status.json", status)
    started = time.monotonic()
    try:
        result = run_child(batch, job, work, "prepare", timeout, update)
        if result != 0:
            details = read(work/"prepared.json") if (work/"prepared.json").is_file() else {}
            new_seed=(batch/'plan.json').is_file() and read(batch/'plan.json').get('ca_is_seed',False)
            update(state=result if isinstance(result, str) else 'input_invalid' if new_seed else "ca_failed", reason=details.get("reason", "input_invalid" if new_seed else "CA_process_failed"),
                   returncode=result, diagnostics=details.get("diagnostics"), elapsed_seconds=time.monotonic()-started)
            return
        result = run_child(batch, job, work, "solve", timeout, update)
        if result != 0:
            update(state=result if isinstance(result, str) else "solve_failed", returncode=result,
                   elapsed_seconds=time.monotonic()-started)
            return
        summary_path = work/"solve/summary.json"
        summary = read(summary_path)["cases"][0]
        update(state="completed", summary=str(summary_path), task_success=summary["task_success"],
            object_hold_gpu=summary["native_hold_success"], object_hold_cpu=summary["independent_mujoco_hold_success"],
            optimization_seconds=summary["timing"]["optimization_seconds"],
            ca_required_finger_mask=summary.get("acceptance",{}).get("required_contact_finger_mask"),
            acquisition_success=summary.get('acquisition',{}).get('success'),
            failure_stage=summary.get('failure_stage'),
            elapsed_seconds=time.monotonic()-started, child_pid=None)
    except Exception as error:
        write(work/"failure.json", dict(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()))
        update(state="solve_failed", reason=str(error), elapsed_seconds=time.monotonic()-started, child_pid=None)


def refresh(batch, plan, *, final=False):
    from collections import Counter
    rows = [read(batch/"jobs"/job["id"]/"status.json") for job in plan["jobs"]]
    counts = Counter(row["state"] for row in rows)
    completed = [row for row in rows if row["state"] == "completed"]
    failed = sum(counts[name] for name in ("ca_failed", "input_invalid", "solve_failed", "timeout"))
    state = "running"
    if final:
        state = "stopped" if STOP.is_set() else "completed_with_failures" if failed else "completed"
    status = dict(state=state, pid=os.getpid(), updated_utc=stamp(), total=len(rows), workers=plan["workers"],
        completed=len(completed), failed=failed, finished=len(completed)+failed,
        remaining=len(rows)-len(completed)-failed, counts=dict(counts),
        task_successes=sum(row.get("task_success", False) for row in completed),
        active=[row for row in rows if row["state"] in ("preparing_ca", "solving")],
        skipped_types=plan["skipped"], radius_scales=plan["radius_scales"],
        long_cylinder_height_m=plan["long_cylinder_height_m"])
    write(batch/"status.json", status)
    write(batch/"summary.json", dict(**status, jobs=rows))


def supervise(batch):
    batch = Path(batch).resolve()
    plan = read(batch/"plan.json")
    with (batch/"supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: STOP.set())
        refresh(batch, plan)
        with ThreadPoolExecutor(max_workers=plan["workers"]) as pool:
            pending = {pool.submit(execute_job, batch, job, plan["stage_timeout_seconds"]) for job in plan["jobs"]}
            while pending:
                done, pending = wait(pending, timeout=5., return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                refresh(batch, plan)
        refresh(batch, plan, final=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("launch", "supervise", "prepare", "solve", "watch-promote"):
        mode.add_argument("--"+name, action="store_true")
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1800.)
    parser.add_argument("--scales", type=float, nargs=5, default=None, help="Explicit legacy radius grid; omitted uses the per-type size policy")
    parser.add_argument("--cylinder-length", type=float, default=.18)
    parser.add_argument("--job")
    parser.add_argument("--work", type=Path)
    parser.add_argument('--pilot',action='store_true',help='Run the specified 24-case contact acquisition pilot')
    parser.add_argument('--cached-batch',type=Path,help='Reuse finite geometric CA proposals with provenance')
    parser.add_argument('--promote-from',type=Path,help='Require the three-family independent pilot success gate before a full launch')
    parser.add_argument('--full-batch',type=Path,help='New full-sweep destination for --watch-promote')
    args = parser.parse_args()
    if args.watch_promote:
        if args.batch is None or args.full_batch is None:parser.error('--watch-promote requires --batch and --full-batch')
        while read(args.batch/'status.json')['state']=='running':time.sleep(5.)
        gate=promotion_gate(args.batch);write(args.batch/'promotion_gate.json',gate)
        if not gate['eligible']:
            write(args.batch/'promotion_status.json',dict(state='not_promoted',**gate))
            print(json.dumps(gate),flush=True)
            return
        command=[sys.executable,str(Path(__file__).resolve()),'--launch','--batch',str(args.full_batch),
            '--promote-from',str(args.batch),'--workers',str(args.workers),'--timeout',str(args.timeout)]
        if args.cached_batch:command+=['--cached-batch',str(args.cached_batch)]
        result=subprocess.run(command,check=True,capture_output=True,text=True)
        write(args.batch/'promotion_status.json',dict(state='launched',full_batch=str(args.full_batch),
            launch=json.loads(result.stdout.strip().splitlines()[-1]),**gate))
        print(result.stdout,flush=True)
    elif args.launch:
        gate=None
        if args.promote_from is not None:
            gate=promotion_gate(args.promote_from)
            write(args.promote_from/'promotion_gate.json',gate)
            if not gate['eligible']:
                raise SystemExit('Full sweep not launched: '+json.dumps(gate))
        elif not args.pilot:
            raise SystemExit('Run --pilot first, then launch the full sweep with --promote-from <pilot batch>')
        path = args.batch or ROOT/"output/grasp_spider_all_types_five_sizes"
        if args.batch is None:
            number = 1
            while path.exists():
                path = ROOT/"output"/f"grasp_spider_all_types_five_sizes_{number:02d}"
                number += 1
        plan = make_batch(path, args.workers, args.timeout, args.scales, args.cylinder_length,
                          pilot=args.pilot,cached_batch=args.cached_batch)
        if gate is not None:write(path/'promotion_gate.json',gate)
        path = path.resolve()
        runtime = Path(plan["runtime"])
        with (path/"supervisor.log").open("a") as log:
            child = subprocess.Popen([sys.executable, str(runtime/"scripts"/Path(__file__).name),
                "--supervise", "--batch", str(path)], cwd=runtime, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        write(path/"launch.json", dict(pid=child.pid, batch=str(path), total=plan["total_jobs"], workers=args.workers))
        print(json.dumps(read(path/"launch.json")), flush=True)
    elif args.supervise:
        supervise(args.batch)
    else:
        batch, work = args.batch.resolve(), args.work.resolve()
        job = next(row for row in read(batch/"plan.json")["jobs"] if row["id"] == args.job)
        if not work.is_relative_to(batch/"jobs"/job["id"]):
            raise ValueError("Worker workspace must remain inside its own job directory")
        try:
            code = prepare(batch, job, work) if args.prepare else solve_job(batch, job, work)
        except Exception as error:
            write(work/("prepare_failure.json" if args.prepare else "solve_failure.json"),
                dict(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()))
            raise
        raise SystemExit(code)


if __name__ == "__main__":
    main()
