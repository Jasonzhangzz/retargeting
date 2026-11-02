#!/usr/bin/env python3
"""Render the actual completed size sweep and upstream runtime comparison."""

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retarget.common.results import write_json


def render(batch, *, history=None, performance=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    batch = Path(batch)
    data = json.loads((batch/"summary.json").read_text())
    groups = data["by_type_size"]
    by_type = defaultdict(list)
    for row in groups:
        by_type[row["type_id"]].append(row)
    types = sorted(by_type)
    width = max(map(len, by_type.values()))
    rates = np.full((len(types), width), np.nan)
    fig, ax = plt.subplots(figsize=(15, 12), layout="constrained")
    for i, type_id in enumerate(types):
        for j, row in enumerate(sorted(by_type[type_id], key=lambda x:x["size_index"])):
            rates[i, j] = row["success_rate"]
    colors = plt.colormaps["YlGn"].copy()
    colors.set_bad("#eeeeee")
    plot = ax.imshow(rates, vmin=0., vmax=1., cmap=colors, aspect="auto")
    labels=[]
    for i, type_id in enumerate(types):
        rows=sorted(by_type[type_id], key=lambda x:x["size_index"])
        labels.append(f"{type_id:02d}  {rows[0]['kind']} / {rows[0]['size_axis']}")
        for j, row in enumerate(rows):
            ax.text(j, i, f"{row['size_scale']:g}x\n{row['successes']}/{row['trials']}",
                    ha="center",va="center",fontsize=7.5,
                    color="white" if row["success_rate"]>.65 else "#253c2a")
    ax.set_yticks(range(len(types)), labels)
    ax.set_xticks(range(width), [str(i+1) for i in range(width)])
    ax.set_xlabel("Size position within each type's own grid (actual multiplier in each cell)")
    ax.set_title(f"WUJI2 GraspADMM: native six-direction success\n{data['finished']}/{data['planned']} trials finished; {data['successes']} successful")
    fig.colorbar(plot,ax=ax,label="Success rate",shrink=.6)
    fig.savefig(batch/"size_success_rates.png",dpi=170)
    plt.close(fig)

    per_type=[]
    for type_id in types:
        rows=by_type[type_id]
        n=sum(r["trials"] for r in rows)
        k=sum(r["successes"] for r in rows)
        rate=k/n
        z=1.959963984540054
        center=(rate+z*z/(2*n))/(1+z*z/n)
        half=z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
        per_type.append(dict(type_id=type_id,template=rows[0]["template"],trials=n,successes=k,
                             success_rate=rate,wilson_95_low=max(0.,center-half),wilson_95_high=min(1.,center+half)))
    fig, ax=plt.subplots(figsize=(13,4),layout="constrained")
    x=np.arange(len(per_type));p=np.array([r["success_rate"] for r in per_type])
    low=np.array([r["wilson_95_low"] for r in per_type]);high=np.array([r["wilson_95_high"] for r in per_type])
    ax.bar(x,p,color="#287f62",width=.7)
    ax.errorbar(x,p,yerr=[np.maximum(p-low,0.),np.maximum(high-p,0.)],
                fmt="none",ecolor="#203c34",capsize=2,lw=.8)
    ax.set_xticks(x,[str(t) for t in types]);ax.set_ylim(0,1)
    ax.set_xlabel("Grasp type");ax.set_ylabel("Success rate")
    ax.set_title("Success across the tested size grid (95% Wilson interval)")
    fig.savefig(batch/"type_success_rates.png",dpi=170);plt.close(fig)
    plan=json.loads((batch/"plan.json").read_text())
    runtime_groups=defaultdict(lambda:defaultdict(list))
    for job in data["jobs"]:
        if job["state"]!="completed":
            continue
        work=Path(job.get("work",batch/"jobs"/job["id"]))
        manifest=work/"solve/manifest.json"
        if not manifest.is_file():
            continue
        cases=json.loads(manifest.read_text()).get("cases",[])
        if len(cases)!=1:
            continue
        info=json.loads((work/"solve"/cases[0]["summary"]).read_text())
        group=runtime_groups[job["type_id"]]
        for key in ("init_seconds","solve_seconds","total_wall_seconds"):
            if info.get(key) is not None:
                group[key].append(info[key])
        if job.get("wall_seconds") is not None:
            group["process_wall_seconds"].append(job["wall_seconds"])
        evaluation=info.get("graspadmm_evaluation",{})
        if evaluation.get("evaluation_wall_seconds") is not None:
            group["evaluation_seconds"].append(evaluation["evaluation_wall_seconds"])
        group["physical_tested"].append(bool(info.get("physical_hold_tested")))
        group["fallback"].append(bool(info.get("fallback_used")))
    runtimes=[]
    for type_id,values in sorted(runtime_groups.items()):
        item=dict(type_id=type_id,scope="Per-case wall times under the recorded concurrent worker count",
                  physical_tests=sum(values.pop("physical_tested",[])),fallback_trials=sum(values.pop("fallback",[])))
        for key,samples in values.items():
            item[key]=dict(count=len(samples),median=float(np.median(samples)),
                           mean=float(np.mean(samples)),min=float(np.min(samples)),max=float(np.max(samples)))
        runtimes.append(item)
    report=dict(state=data["state"],evaluation_protocol=data["evaluation_protocol"],
                planned=data["planned"],finished=data["finished"],successes=data["successes"],
                success_rate=data["success_rate"],by_type=per_type,
                type_macro_success_rate=float(np.mean([r["success_rate"] for r in per_type])),
                failure_stages=data.get("failure_stages"),workers=plan["workers"],runtime_by_type=runtimes)
    if plan.get("combined_experiment"):
        report["combined_experiment"]=True
        report["source_runs"]=plan["source_runs"]
        report["timing_scope"]="Source-run execution times are reported separately; artifact assembly time is excluded"
    if plan.get("created_at"):
        report["batch_wall_seconds"]=(batch/"status.json").stat().st_mtime-datetime.fromisoformat(plan["created_at"]).timestamp()
    if history:
        historic=json.loads((Path(history)/"summary.json").read_text())
        methods=defaultdict(lambda:dict(trials=0,successes=0))
        for row in historic["cases"]:
            methods[row["method"]]["trials"]+=1
            methods[row["method"]]["successes"]+=bool(row.get("task_success"))
        report["historical_reevaluation"]=dict(source=str(history),total=historic["total"],
            successful=historic["successful"],counts=historic["counts"],by_method=dict(methods),
            comparison="Historical geometries and solver budgets differ from the new sweep")
    if performance:
        timing=json.loads((Path(performance)/"benchmark.json").read_text())
        report["runtime_comparison"]=timing["comparison"]
    write_json(batch/"report.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch",type=Path,required=True)
    parser.add_argument("--history",type=Path)
    parser.add_argument("--performance",type=Path)
    args=parser.parse_args()
    report=render(args.batch,history=args.history,performance=args.performance)
    print(json.dumps({k:v for k,v in report.items() if k not in ("by_type","runtime_by_type","historical_reevaluation","source_runs")}))


if __name__=="__main__":
    main()
