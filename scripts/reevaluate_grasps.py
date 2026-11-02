#!/usr/bin/env python3
"""Re-evaluate all historical grasp cases without rewriting their source results.

Manifest aliases are merged first; byte-identical object/relative-hand candidates
share one native test. Every original case remains in the accounting denominator.
"""
from __future__ import annotations
import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from retarget.common.results import write_json
from retarget.env.grasp.grasp_evaluation import (PROTOCOL, CRITERION, select_grasp,
                                                prepare_candidate, evaluate_candidate, measure_primitive, infer_template, NativeEvaluationWorker)


def resolve(batch, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (batch/path).resolve()


def discover_cases(source, excluded=None):
    source, excluded = Path(source).resolve(), Path(excluded).resolve() if excluded else None
    merged = {};other_tasks=[]
    for manifest in sorted(source.rglob("manifest.json")):
        if excluded and manifest.is_relative_to(excluded):
            continue
        try:
            document = json.loads(manifest.read_text())
        except (OSError, ValueError):
            continue
        if document.get("historical_reevaluation"):
            continue
        if document.get("task") not in (None, "grasp"):
            other_tasks.append(manifest.parent)
            continue
        for case in document.get("cases", []):
            if case.get("task", document.get("task")) != "grasp" or not case.get("model"):
                continue
            batch = manifest.parent
            model = resolve(batch, case["model"])
            trajectory = resolve(batch, case["trajectory"]) if case.get("trajectory") else None
            summary = resolve(batch, case["summary"]) if case.get("summary") else None
            key = (str(model), str(trajectory), str(summary))
            alias = dict(manifest=str(manifest), case_id=case.get("case_id", case.get("name")),
                         batch=batch.name)
            if key in merged:
                merged[key]["aliases"].append(alias)
                continue
            merged[key] = dict(case, model=str(model), trajectory=str(trajectory) if trajectory else None,
                summary=str(summary) if summary else None, method=case.get("method", document.get("method", "unknown")),
                aliases=[alias])
    # Failed FA prerequisite cases were historically omitted from manifests.
    # Recover those recorded trials so rejection cannot improve success rates.
    known_summaries={row.get("summary") for row in merged.values()}
    for path in sorted(source.rglob("summary.json")):
        if (excluded and path.is_relative_to(excluded)) or str(path.resolve()) in known_summaries or any(path.is_relative_to(other) for other in other_tasks):
            continue
        try:
            document=json.loads(path.read_text())
        except (OSError,ValueError):
            continue
        if not isinstance(document,dict) or document.get("historical_reevaluation"):
            continue
        if document.get("method") not in ("dexforge","mppi","cem","contactaware"):
            continue
        if not document.get("failed_stage") or document.get("solver_completed") is not False:
            continue
        key=(None,None,str(path.resolve()))
        merged[key]=dict(task="grasp",method=document["method"],name=document.get("name",path.parent.name),
            model=None,trajectory=None,summary=str(path.resolve()),hand="wuji2",
            aliases=[dict(manifest=None,case_id=path.parent.name,batch=path.parent.parent.name)])
    return list(merged.values())


def _case_dir(row):
    return Path(row["summary"]).parent if row.get("summary") else Path(row["trajectory"]).parent


def refresh(destination, rows, started, *, complete=False, methods=None):
    counts = Counter(row["state"] for row in rows)
    finished = [row for row in rows if row["state"] != "queued"]
    groups = {}
    for row in finished:
        primitive = row.get("primitive") or {}
        key = (row.get("method", "unknown"), row.get("template", "unknown"),
               primitive.get("kind", "unknown"), primitive.get("radius_m"), primitive.get("height_m"))
        group = groups.setdefault(key, dict(method=key[0], template=key[1], primitive=key[2],
            radius_m=key[3], height_m=key[4], successes=0, trials=0))
        group["trials"] += 1
        group["successes"] += int(row.get("task_success", False))
    for group in groups.values():
        group["success_rate"] = group["successes"]/group["trials"]
    report = dict(evaluation_protocol=PROTOCOL, historical_reevaluation=True,
        state="completed" if complete else "running", methods=methods or ["all"], total=len(rows), finished=len(finished),
        successful=sum(row.get("task_success", False) for row in finished), counts=dict(counts),
        wall_seconds=time.perf_counter()-started, groups=list(groups.values()), cases=rows,
        source_results_changed=False, denominator="Every distinct source case, including unavailable solver outputs; aliases merged")
    with (destination/"success_rates.csv").open("w", newline="") as handle:
        writer=csv.DictWriter(handle, fieldnames=["method","template","primitive","radius_m","height_m","successes","trials","success_rate"])
        writer.writeheader();writer.writerows(groups.values())
    write_json(destination/"summary.json", report)
    write_json(destination/"status.json", {key:value for key,value in report.items() if key not in ("cases", "groups")})
    cases=[]
    for row in finished:
        if not row.get("model") or not row.get("trajectory"):
            continue
        cases.append(dict(task="grasp", method=row.get("method", "unknown"), hand=row.get("hand", "wuji2"),
            template=row.get("template", row.get("name", "unknown")), name=row.get("name", row["id"]),
            case_id=row["id"], model=row["model"], trajectory=row["trajectory"],
            summary=str(destination/"cases"/row["id"]/"summary.json"),
            stage="GraspADMM评测", historical_reevaluation=True,
            primitive=row.get("primitive"), radius_scale=row.get("radius_scale"), dt=row.get("dt", .004)))
    write_json(destination/"manifest.json", dict(version=1, task="grasp", method="graspadmm_evaluation",
        evaluation_protocol=PROTOCOL, historical_reevaluation=True, cases=cases))
    return report



def refresh_candidate_metadata(case,row):
    """Refresh grouping metadata from copied geometry, without repeating physics."""
    row["acceptance"]=CRITERION
    if row.get("state")=="input_unavailable" and row.get("has_hand_pose") is False:
        row["error"]="ValueError: No saved finite hand pose; original result is an object-only failure preview"
    if row.get("graspadmm_evaluation"):
        row["graspadmm_evaluation"]["criterion"]=CRITERION
        evaluation_path=Path(case)/"evaluation/evaluation.json"
        if evaluation_path.is_file():
            write_json(evaluation_path,row["graspadmm_evaluation"])
    path=Path(case)/"evaluation/candidate.json"
    if not path.is_file():
        return row
    metadata=json.loads(path.read_text())
    mesh=Path(metadata["mesh"])
    if mesh.is_file():
        import trimesh
        measured=measure_primitive(trimesh.load_mesh(mesh,process=False).vertices)
        if row.get("primitive") and not row["primitive"].get("dimensions_source"):
            # Explicit original taxonomy metadata wins over shape inference;
            # retain its size-axis/scale fields while measuring physical bounds.
            measured.update(row["primitive"])
        row["primitive"]=measured
    if not row.get("template") or row.get("template")=="unknown":
        row["template"]=infer_template(metadata.get("source_model",row.get("model","")),metadata.get("source_selection"))
    return row

def run(source, destination, *, timeout=600., limit=None, resume=False, fresh_processes=False, methods=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() and not resume:
        raise ValueError("Re-evaluation destination already exists; use --resume")
    destination.mkdir(parents=True, exist_ok=True)
    plan_path = destination/"plan.json"
    if resume and plan_path.is_file():
        plan = json.loads(plan_path.read_text())
        rows = plan["cases"]
        methods = plan.get("methods") if methods is None else methods
        if methods == ["all"]:methods=None
    else:
        discovered = discover_cases(source, destination)
        if methods is not None:
            discovered=[row for row in discovered if row["method"] in methods]
        if limit is not None:
            discovered = discovered[:limit]
        rows=[]
        for index, case in enumerate(discovered):
            digest = hashlib.sha256(json.dumps(case["aliases"], sort_keys=True).encode()).hexdigest()[:12]
            rows.append(dict(case, id=f"{index:05d}_{digest}", state="queued"))
        write_json(plan_path, dict(source=str(source), evaluation_protocol=PROTOCOL,
            historical_reevaluation=True, methods=methods or ["all"], cases=rows))
    old_report_path=destination/"summary.json"
    old_report=json.loads(old_report_path.read_text()) if resume and old_report_path.is_file() else {}
    started=time.perf_counter(); cache={}
    worker=None if fresh_processes else NativeEvaluationWorker(destination/"native_worker.log")
    for row in rows:
        case = destination/"cases"/row["id"]
        previous=case/"summary.json"
        if resume and previous.is_file():
            saved=json.loads(previous.read_text());row.update(saved)
            refresh_candidate_metadata(case,row)
            write_json(previous,row)
            if row.get("candidate_sha256"):cache[row["candidate_sha256"]]=saved
            continue
        case.mkdir(parents=True, exist_ok=True)
        try:
            qpos, selection = select_grasp(_case_dir(row), row["method"], row.get("trajectory"))
            metadata = prepare_candidate(row["model"], qpos, case/"evaluation", method=row["method"], selection=selection)
            if not row.get("template"):
                row["template"]=metadata.get("template")
            if not row.get("primitive"):
                row["primitive"]=metadata.get("actual_primitive")
            digest=metadata["candidate_sha256"]
            if digest in cache:
                result=dict(cache[digest]["graspadmm_evaluation"])
                row["evaluation_reused_from"]=cache[digest]["id"]
            else:
                result=(evaluate_candidate(case/"evaluation",timeout=timeout) if worker is None
                        else worker.evaluate(case/"evaluation",timeout=timeout))
            row.update(state="completed" if result.get("state")=="completed" else "evaluation_failed",
                candidate_sha256=digest, evaluation_protocol=PROTOCOL,
                task_success=bool(result.get("success")), graspadmm_evaluation=result, acceptance=CRITERION)
            cache[digest]=dict(row)
        except Exception as error:
            row.update(state="input_unavailable", evaluation_protocol=PROTOCOL, task_success=False,
                error=f"{type(error).__name__}: {error}", acceptance=CRITERION)
        refresh_candidate_metadata(case,row)
        write_json(previous,row)
        refresh(destination,rows,started,methods=methods)
        print(f"[{row['id']}] {row['state']} success={row['task_success']}",flush=True)
    if worker is not None:worker.close()
    report=refresh(destination,rows,started,complete=True,methods=methods)
    report["evaluation_run_wall_seconds"]=old_report.get("evaluation_run_wall_seconds",old_report.get("wall_seconds",report["wall_seconds"]))
    if resume:
        report["metadata_refresh_wall_seconds"]=report["wall_seconds"]
        report["wall_seconds"]=report["evaluation_run_wall_seconds"]
    write_json(destination/"summary.json",report)
    write_json(destination/"status.json",{key:value for key,value in report.items() if key not in ("cases","groups")})
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=ROOT/"output")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--timeout",type=float,default=600.)
    parser.add_argument("--limit",type=int)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--methods",nargs="+",help="Evaluate only these manifest methods (default: all)")
    parser.add_argument("--fresh-processes",action="store_true",help="Use a fresh pinned interpreter per case instead of the default persistent worker")
    args=parser.parse_args()
    report=run(args.source,args.output,timeout=args.timeout,limit=args.limit,resume=args.resume,fresh_processes=args.fresh_processes,methods=args.methods)
    print(json.dumps({key:value for key,value in report.items() if key not in ("cases","groups")}),flush=True)


if __name__=="__main__":
    main()
