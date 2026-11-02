#!/usr/bin/env python3
"""Create a new artifact by replacing every trial of one complete grasp type.

No source file, status, result or success verdict is changed. Selection uses
(type_id, size_index, seed), never a comparison of success or cost.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retarget.common.results import write_json

PROTOCOL = "graspadmm_native_v1"
TERMINAL = {"completed", "failed", "timeout"}
PATH_FIELDS = ("model", "trajectory", "summary")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trial_key(job):
    return (int(job["type_id"]), int(job["size_index"]), int(job["seed"]))


def vendor_hashes(plan):
    values = {key: value for key, value in plan.get("source_hashes", {}).items()
              if key.startswith("third_party/Dexonomy/")}
    if not values:
        raise ValueError("Source plan has no recorded Dexonomy vendor hashes")
    return values


def _absolute(path, parent):
    path = Path(path)
    return (path if path.is_absolute() else Path(parent)/path).resolve()


def source_policy(source):
    plan = source["plan"]
    declared = plan.get("input_policy", plan.get("trial_policy", {}).get("input_policy"))
    observed = {row["primitive_seed_policy"] for row in source["rows"].values()
                if row.get("primitive_seed_policy") is not None}
    if len(observed) > 1 or (declared is not None and observed and observed != {declared}):
        raise ValueError("Source input policy declaration conflicts with trial summaries")
    if declared is not None:
        return declared, "source plan input_policy / trial_policy.input_policy"
    if observed:
        return next(iter(observed)), "trial summary primitive_seed_policy"
    runner = Path(plan["runtime"])/"retarget/env/grasp/grasp_admm_runner.py"
    if runner.is_file():
        text = runner.read_text()
        if "primitive_seed_policy" not in text and "initialize_one(" in text:
            return "upstream", "verified legacy frozen runner has original initialize_one and no primitive_seed_policy option"
    raise ValueError("Source input policy cannot be identified from its plan, summaries or legacy runner")


def _verify_source_files(plan):
    runtime = plan.get("runtime")
    if not runtime:
        raise ValueError("Source plan must identify its frozen runtime")
    for relative, expected in plan.get("source_hashes", {}).items():
        path = Path(runtime)/relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Frozen source drift: {path}")
    for filename, expected in plan.get("input_hashes", {}).items():
        path = Path(filename)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Shared input drift: {filename}")


def load_source(batch):
    """Read terminal trial records and their actual exported candidate summaries."""
    batch = Path(batch).resolve()
    plan_path = batch/"plan.json"
    plan = read(plan_path)
    plan_hash = digest(plan_path)
    jobs = plan.get("jobs", [])
    if not jobs:
        raise ValueError("Source plan has no trials")
    keys = [trial_key(j) for j in jobs]
    if len(set(keys)) != len(keys) or len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate source trial key or job ID")
    _verify_source_files(plan)
    geometries = {}
    for job in jobs:
        group = trial_key(job)[:2]
        if group in geometries and not _same_geometry(geometries[group], job["primitive"]):
            raise ValueError(f"One source size group contains different seed geometries: {group}")
        geometries[group] = job["primitive"]
    rows, manifests = {}, {}
    for job in jobs:
        key = trial_key(job)
        base = batch/"jobs"/job["id"]
        status = read(base/"status.json")
        if status.get("state") not in TERMINAL:
            raise ValueError(f"Nonterminal trial: {job['id']} ({status.get('state')})")
        work = _absolute(status.get("work", base), batch)
        row = dict(copy.deepcopy(job), **status)
        row.update(work=str(work), source_batch=str(batch), source_plan_sha256=plan_hash,
                   task_success=False, evaluation_protocol=None)
        manifest_path = work/"solve/manifest.json"
        if manifest_path.is_file():
            exported = read(manifest_path).get("cases", [])
            if len(exported) != 1:
                raise ValueError(f"Single-candidate contract violated: {job['id']}")
            case = copy.deepcopy(exported[0])
            summary_path = _absolute(case["summary"], work/"solve")
            if not summary_path.is_file():
                raise ValueError(f"Missing trial summary: {job['id']}")
            info = read(summary_path)
            protocol = info.get("evaluation_protocol")
            if protocol not in (None, PROTOCOL):
                raise ValueError(f"Foreign evaluation protocol: {job['id']}")
            if info.get("task_success") and protocol != PROTOCOL:
                raise ValueError(f"Unverifiable success protocol: {job['id']}")
            row.update(task_success=status["state"] == "completed" and protocol == PROTOCOL
                       and info.get("task_success") is True,
                       evaluation_protocol=protocol,
                       primitive_seed_policy=info.get("primitive_seed_policy"),
                       primitive_seed_geometry=copy.deepcopy(info.get("primitive_seed_geometry")),
                       failure_stage=info.get("failure_stage", info.get("reason")),
                       fallback=info.get("fallback", info.get("fallback_used", False)))
            for field in PATH_FIELDS:
                if field in case:
                    case[field] = str(_absolute(case[field], work/"solve"))
            primitive = copy.deepcopy(job["primitive"])
            case.update(primitive=primitive, size_scale=primitive["size_scale"],
                        size_axis=primitive["size_axis"], radius_scale=primitive["radial_scale"],
                        seed=job["seed"], case_id=job["id"], source_batch=str(batch),
                        source_plan_sha256=row["source_plan_sha256"])
            manifests[key] = case
        elif status["state"] == "completed":
            raise ValueError(f"Completed trial has no candidate manifest: {job['id']}")
        if not row["task_success"] and not row.get("failure_stage"):
            row["failure_stage"] = status.get("reason", "native_evaluation" if status["state"] == "completed" else "runtime")
        rows[key] = row
    return dict(batch=batch, plan=plan, rows=rows, manifests=manifests)


def _input_fingerprints(plan):
    # Absolute checkout locations may differ; physical input identity may not.
    normalized = {}
    for filename, value in plan.get("input_hashes", {}).items():
        key = str(filename).partition("/data/grasp/")[-1] if "/data/grasp/" in str(filename) else str(filename)
        normalized[key] = value
    return normalized


def _same_geometry(left, right):
    if left.get("kind") != right.get("kind"):
        return False
    for field in ("radius_m", "height_m"):
        try:
            if not math.isclose(float(left[field]), float(right[field]), rel_tol=0., abs_tol=1.e-12):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    if "dimensions_m" in left or "dimensions_m" in right:
        a, b = left.get("dimensions_m", []), right.get("dimensions_m", [])
        if len(a) != 3 or len(b) != 3 or any(not math.isclose(x,y,rel_tol=0.,abs_tol=1.e-12) for x,y in zip(a,b)):
            return False
    return True


def validate_pair(base, replacement, type_id=9):
    left, right = base["plan"], replacement["plan"]
    if left.get("protocol") != PROTOCOL or right.get("protocol") != PROTOCOL:
        raise ValueError("Both source plans must use graspadmm_native_v1")
    if not left.get("numerical_stack") or left["numerical_stack"] != right.get("numerical_stack"):
        raise ValueError("Source numerical stacks differ or are not recorded")
    if left.get("physical_model") != "wuji2" or right.get("physical_model") != "wuji2":
        raise ValueError("Both source plans must use wuji2")
    if vendor_hashes(left) != vendor_hashes(right):
        raise ValueError("Source Dexonomy vendor hashes differ")
    if _input_fingerprints(left) != _input_fingerprints(right):
        raise ValueError("Source shared physical inputs differ")
    expected = {k for k in base["rows"] if k[0] == type_id}
    if not expected or set(replacement["rows"]) != expected:
        raise ValueError("Replacement must contain every size/seed of exactly the selected type")
    for key in expected:
        a, b = base["rows"][key], replacement["rows"][key]
        if a["template"] != b["template"] or not _same_geometry(a["primitive"], b["primitive"]):
            raise ValueError(f"Replacement geometry/template differs: {key}")
    return expected


def require_main_inventory(source):
    rows = source["rows"]
    types = set(range(1,34))-{19,21}
    expected = {(t,i,seed) for t in types for i in range(11 if t == 10 else 5) for seed in range(10)}
    if set(rows) != expected:
        raise ValueError("Base must be the complete 1610-trial, 161-size, ten-seed main sweep")


def _timings(rows):
    values = [float(row["wall_seconds"]) for row in rows if row.get("wall_seconds") is not None]
    return dict(trials=len(rows), process_wall_seconds_count=len(values),
                sum_process_wall_seconds=sum(values),
                mean_process_wall_seconds=sum(values)/len(values) if values else None,
                min_process_wall_seconds=min(values) if values else None,
                max_process_wall_seconds=max(values) if values else None)


def source_record(source, selected):
    batch, plan = source["batch"], source["plan"]
    policy, policy_evidence = source_policy(source)
    record = dict(batch=str(batch), runtime=str(Path(plan["runtime"]).resolve()),
        plan_sha256=digest(batch/"plan.json"), source_hashes=copy.deepcopy(plan["source_hashes"]),
        protocol=plan["protocol"], numerical_stack=copy.deepcopy(plan["numerical_stack"]),
        physical_model=plan["physical_model"], workers=plan.get("workers"),
        input_policy=policy, input_policy_evidence=policy_evidence,
        trial_policy=copy.deepcopy(plan.get("trial_policy", {})),
        source_trial_count=len(source["rows"]), selected_trial_count=len(selected),
        timings_all_source_trials=_timings(list(source["rows"].values())),
        timings_selected_trials=_timings(selected), created_at=plan.get("created_at"))
    if (batch/"summary.json").is_file():
        record["summary_sha256"] = digest(batch/"summary.json")
    if plan.get("created_at") and (batch/"status.json").is_file():
        record["source_batch_wall_seconds"] = (batch/"status.json").stat().st_mtime-datetime.fromisoformat(plan["created_at"]).timestamp()
    return record


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["type_id"],row["size_index"])].append(row)
    table = []
    for (type_id,index), items in sorted(groups.items()):
        successes = sum(row["task_success"] for row in items)
        table.append(dict(type_id=type_id,template=items[0]["template"],size_index=index,
                          **items[0]["primitive"],trials=len(items),finished=len(items),
                          successes=successes,success_rate=successes/len(items),
                          completed_success_rate=successes/len(items)))
    successes = sum(row["task_success"] for row in rows)
    return dict(state="completed", evaluation_protocol=PROTOCOL, planned=len(rows),finished=len(rows),
        states=dict(Counter(row["state"] for row in rows)), successes=successes,
        success_rate=successes/len(rows), failure_stages=dict(Counter(
            row.get("failure_stage") for row in rows if not row["task_success"])),
        by_type_size=table,jobs=rows)


def merge_batches(base_path, replacement_path, output, *, type_id=9):
    """Validate both complete sources before writing one new combined artifact."""
    output = Path(output).resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("Combined output must be a new directory")
    base, replacement = load_source(base_path), load_source(replacement_path)
    if output in (base["batch"], replacement["batch"]) or any(
            output.is_relative_to(source["batch"]) for source in (base,replacement)):
        raise ValueError("Combined output must not be inside a source batch")
    require_main_inventory(base)
    replaced = validate_pair(base,replacement,type_id)
    selected = dict(base["rows"])
    selected.update(replacement["rows"])
    rows = [selected[key] for key in sorted(selected)]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Merged trial IDs are not unique")
    original_selected = [row for key,row in selected.items() if key not in replaced]
    replacement_selected = list(replacement["rows"].values())
    sources = [source_record(base,original_selected),source_record(replacement,replacement_selected)]
    timestamp = datetime.now(timezone.utc).isoformat()
    policy = dict(replace_type_id=type_id,replaced_trials=len(replaced),
        selection="complete type replacement by (type_id,size_index,seed); no cost/success comparison",
        source_paths=[str(base["batch"]),str(replacement["batch"])])
    jobs = [dict(copy.deepcopy(row), source_input_policy=(sources[1] if row["type_id"]==type_id else sources[0])["input_policy"]) for row in rows]
    plan = dict(artifact_created_at=timestamp,created_at=None,combined_experiment=True,
        resumable=False,workers=6,protocol=PROTOCOL,physical_model="wuji2",
        numerical_stack=copy.deepcopy(base["plan"]["numerical_stack"]),
        vendor_hashes=vendor_hashes(base["plan"]),source_runs=sources,merge_policy=policy,
        jobs=jobs,skipped=copy.deepcopy(base["plan"].get("skipped",[])))
    summary = aggregate(jobs)
    summary.update(combined_experiment=True,source_runs=sources,merge_policy=policy,skipped=plan["skipped"])
    cases = []
    for key in sorted(selected):
        source = replacement if key in replaced else base
        if key in source["manifests"]:
            cases.append(source["manifests"][key])
    # All validation and source reads precede the first destination mutation.
    output.mkdir(parents=True)
    write_json(output/"plan.json",plan)
    write_json(output/"summary.json",summary)
    write_json(output/"status.json",{k:summary[k] for k in ("state","planned","finished","successes","states")})
    write_json(output/"manifest.json",dict(version=1,task="grasp",method="graspadmm",
        combined_experiment=True,source_runs=[str(s["batch"]) for s in sources],cases=cases))
    columns=["type_id","template","size_index","kind","size_axis","size_scale","radius_m","height_m","trials","finished","successes","success_rate"]
    with (output/"success_rates.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction="ignore")
        writer.writeheader();writer.writerows(summary["by_type_size"])
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",type=Path,required=True)
    parser.add_argument("--replacement",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--type",type=int,default=9,dest="type_id")
    args=parser.parse_args()
    report=merge_batches(args.base,args.replacement,args.output,type_id=args.type_id)
    print(json.dumps({key:report[key] for key in ("planned","finished","successes","success_rate")}))


if __name__=="__main__":
    main()
