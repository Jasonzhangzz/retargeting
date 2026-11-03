#!/usr/bin/env python3
"""Cache original Shadow poses and static geometric contacts; render previews."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retarget.env.grasp.dexonomy_grasp_static import run_static_preview,run_static_full
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=ROOT/'data/dexonomy')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['preview','full'],default='preview')
    p.add_argument('--plan',type=Path,default=ROOT/'results/dexonomy_shadow_pilot100_20261103/plan.json')
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--records-per-shard',type=int,default=32)
    a=p.parse_args()
    if min(a.workers,a.records_per_shard)<1:p.error('Counts must be positive')
    result=run_static_full(a.dataset,a.output,a.workers,a.records_per_shard) if a.mode=='full' else run_static_preview(a.dataset,a.plan,a.output,a.workers)
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
