#!/usr/bin/env python3
"""Recompute all paired native Moxian grasps with fixed Wuji2 contact anchors."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from dataclasses import asdict,replace
import fcntl
import hashlib
import json
from pathlib import Path
import sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from visualization.viser_moxian_dataset import Dataset,DEFAULT_DATA,sha256
from retarget.env.grasp.moxian_contact_retarget import Solver,Settings,REGIONS


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(temp,path)


def summary(parts):
    active=np.concatenate([p['contact_active'] for p in parts]);before=np.concatenate([p['initial_anchor_error_m'] for p in parts])[active]
    after=np.concatenate([p['anchor_error_m'] for p in parts])[active]
    valid=np.concatenate([p['contact_correspondence_valid'] for p in parts]);count=np.concatenate([p['source_contact_count'] for p in parts])
    penetration=np.concatenate([p['sampled_penetration_m'] for p in parts])
    normal=np.concatenate([p['normal_dot'] for p in parts]);material=np.concatenate([p['material_side_dot'] for p in parts])
    unexpected=np.concatenate([p['unexpected_contact'] for p in parts])
    errors=np.concatenate([p['anchor_error_m'] for p in parts])
    def quant(a):return dict(mean=float(a.mean()) if len(a) else None,p50=float(np.median(a)) if len(a) else None,p95=float(np.quantile(a,.95)) if len(a) else None,max=float(a.max()) if len(a) else None)
    return dict(samples=len(valid),accepted=int(valid.sum()),no_source_contact=int((count==0).sum()),
                initial_anchor_error_mm=quant(before*1000),final_anchor_error_mm=quant(after*1000),
                dense_surface_penetration_mm=quant(penetration*1000),
                failed_anchor=int((np.where(active,errors,0).max(1)>.005).sum()),
                failed_penetration=int((penetration>.002).sum()),
                failed_normal=int((np.where(active,normal,1).min(1)<.5).sum()),
                failed_material_side=int((np.where(active,material,1).min(1)<.5).sum()),
                unexpected_contact=int(unexpected.sum()))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=DEFAULT_DATA)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--iterations',type=int,default=75)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--pilot',type=int,default=0,help='Spread this many rows over each type, in a separate output')
    ap.add_argument('--resume-compatible-engine',action='store_true',help='Record a tested kernel revision; settings, source and acceptance must match exactly')
    args=ap.parse_args()
    if args.batch_size<=0 or args.iterations<=0 or not 0<=args.pilot<=4000:ap.error('invalid numeric argument')
    if args.pilot and args.output is None:ap.error('--pilot requires a separate --output directory')
    output=args.output or args.data/'contact_retarget_v1';output.mkdir(parents=True,exist_ok=True)
    lock=(output/'campaign.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import torch,warp as wp
    torch.set_num_threads(1);wp.init()
    settings=replace(Settings(),iterations=args.iterations)
    dataset=Dataset(args.data)
    config=dict(schema='moxian_wuji_contact_retarget_v1',settings=asdict(settings),batch_size=args.batch_size,
        pilot=args.pilot,regions=list(REGIONS),source_manifest_sha256=sha256(args.data/'manifest.json'),
        wuji_pairs_manifest_sha256=sha256(args.data/'wuji2_pairs/manifest.json'),
        algorithm='bounded independent GPU LM; native mesh skinning; fixed object anchors; native joint limits; pseudo-normal clearance during optimization and three-ray final audit',
        source_contact_policy='visual mesh samples within 3mm with opposing normals, one anchor per anatomical region',
        acceptance_policy='all active anchors <=5mm; actual deformed face normal dot >=0.5; full skin vertices+face-centers penetration <=2mm; no unexpected sampled regional contact; matching material-side normal dot >=0.5',
        force_closure_verified=False,dynamics_verified=False,self_collision_verified=False,
        code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'retarget/env/grasp/moxian_contact_retarget.py',ROOT/'retarget/env/grasp/moxian_contact_kernels.py']})
    config_path=output/'config.json'
    history_path=output/'engine_history.json'
    history=json.loads(history_path.read_text()) if history_path.exists() else []
    if config_path.exists():
        prior=json.loads(config_path.read_text())
        if prior!=config:
            matching={**prior,'code_sha256':config['code_sha256']}==config
            if not args.resume_compatible_engine or not matching:
                raise ValueError('Resume configuration/code mismatch; use a new output directory')
            known={k for h in history for k in h['existing_files']}
            existing={str(p.relative_to(output)):sha256(p) for p in output.glob('*/*.npz') if str(p.relative_to(output)) not in known}
            history.append(dict(previous_config=prior,existing_files=existing,
                compatibility='Same source, objective, settings and acceptance; optimized geometry kernels and cached residual geometry'))
            write(history_path,history);write(config_path,config)
    else:write(config_path,config)
    engine_id=lambda cfg:hashlib.sha256(json.dumps(cfg['code_sha256'],sort_keys=True).encode()).hexdigest()
    old_engines={name:engine_id(h['previous_config']) for h in history for name in h['existing_files']}
    old_hashes={name:fingerprint for h in history for name,fingerprint in h['existing_files'].items()}
    file_engines={}
    solver=Solver(dataset,args.device,settings)
    order=['24_Tip_Pinch']+[k for k in dataset.types if k!='24_Tip_Pinch']
    all_ids=np.linspace(0,3999,args.pilot,dtype=int) if args.pilot else np.arange(4000)
    total=len(order)*len(all_ids);done=0;new=0;began=time.monotonic();reports={};files={}
    for kind in order:
        grid=dataset.grid(kind);folder=output/kind;folder.mkdir(exist_ok=True);parts=[]
        for start in range(0,len(all_ids),args.batch_size):
            ids=all_ids[start:start+args.batch_size];path=folder/f'{start:07d}.npz'
            if path.exists():
                rel=str(path.relative_to(output))
                if rel in old_hashes and sha256(path)!=old_hashes[rel]:raise ValueError('Changed checkpoint '+rel)
                with np.load(path,allow_pickle=False) as z:part={k:z[k] for k in z.files}
                np.testing.assert_array_equal(part['sample_indices'],ids)
                np.testing.assert_array_equal(part['wuji_pose_id'],grid['wuji_pose_id'][ids])
            else:
                part=solver.solve(grid,ids)
                for key in ('source_case_name','wuji_pose_id','object_scene','object_pose_world','object_scale'):
                    part[key]=grid[key][ids]
                temporary=path.with_suffix('.tmp')
                with temporary.open('wb') as f:np.savez_compressed(f,**part)
                os.replace(temporary,path);new+=len(ids)
            parts.append(part);done+=len(ids)
            report=summary(parts);reports[kind]=report
            elapsed=time.monotonic()-began;eta=(total-done)*elapsed/max(new,1)
            write(output/'status.json',dict(state='running',computed=done,total=total,current_type=kind,
                 elapsed_seconds=elapsed,estimated_remaining_seconds=eta,types=reports,pid=os.getpid()))
            print(f'{kind} {start+len(ids)}/{len(all_ids)} total {done}/{total} accepted {report["accepted"]}/{report["samples"]} elapsed {elapsed:.1f}s ETA {eta:.1f}s',flush=True)
            files[str(path.relative_to(output))]=sha256(path)
            file_engines[str(path.relative_to(output))]=old_engines.get(str(path.relative_to(output)),engine_id(config))
        write(folder/'report.json',reports[kind])
    write(output/'manifest.json',dict(schema=config['schema'],complete=True,computed=done,total=total,
         config_sha256=sha256(config_path),files=files,file_engines=file_engines,types=reports,regions=list(REGIONS),
         engine_history_sha256=sha256(history_path) if history_path.exists() else None))
    elapsed=time.monotonic()-began
    write(output/'status.json',dict(state='complete',computed=done,total=total,elapsed_seconds=elapsed,
         types=reports,accepted=sum(r['accepted'] for r in reports.values()),no_source_contact=sum(r['no_source_contact'] for r in reports.values())))
    print('COMPLETE',done,'accepted',sum(r['accepted'] for r in reports.values()),flush=True)

if __name__=='__main__':main()
