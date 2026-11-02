#!/usr/bin/env python3
"""Verify a full 1024-candidate adjacent solve before launching remote shards."""
from dataclasses import replace
import argparse,threading,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retarget.config.run import load
from retarget.common.results import write_json
from retarget.env.grasp.template_mppi_dense_runner import validate,make_plan,solve_node


def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--population',type=int,default=1024);args=p.parse_args()
    config=load(ROOT/'configs/grasp_template_mppi_dense.yaml')
    config.solver=replace(config.solver,population=args.population,worlds=args.population)
    _,values,child=validate(config);values=dict(values,types=[1])
    destination=ROOT/'migration/large_batch_validation'
    destination.mkdir(parents=True,exist_ok=True)
    nodes,anchors=make_plan(config,destination,values)
    node=next(n for n in nodes if n.get('parent_index') in anchors)
    parent=anchors[node['parent_index']]['entries'][0]
    before=torch.cuda.mem_get_info()[0];minimum=[before];stop=threading.Event()
    def sample():
        while not stop.wait(.05):minimum[0]=min(minimum[0],torch.cuda.mem_get_info()[0])
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    try:result=solve_node(child,str(destination),node,parent)
    finally:stop.set();thread.join()
    d=result['summary']['geometry_diagnostics']
    report=dict(population=args.population,worlds=args.population,device=torch.cuda.get_device_name(),
        passed=bool(result['summary']['solver_completed'] and d['valid'] and d['adjacent_motion_valid']),
        peak_incremental_bytes=before-minimum[0],seconds=result['continuity']['seconds'],
        projected_closure=d['projected_closure'],coverage=d['coverage'],motion=result['continuity']['motion'],case=result)
    write_json(destination/'validation.json',report)
    print({k:v for k,v in report.items() if k!='case'},flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
