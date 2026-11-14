#!/usr/bin/env python3
"""Non-publishing single-frame latency probe of an accepted or failed candidate."""
import argparse,json,sys,time,threading
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moxian_web.hand_model import DEFAULT_MANO
from moxian_web.mano_runtime import ManoSurface
from moxian_web.mano_fit import holdout_rows
from moxian_web.mano_gpu_runtime import solve_batch
from scripts.calibrate_wuji_mano import CHAINS


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--rows',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--per-stage',type=int,default=3)
    a=ap.parse_args();torch.set_num_threads(1)
    p=json.loads(a.candidate.read_text());source=json.loads(a.rows.read_text())['rows']
    m=ManoSurface(DEFAULT_MANO.with_name('MANO_'+p['side'].upper()+'.pkl'),side=p['side']);m.enable_official_fk();m.build_binding(p['mapping'])
    if p.get('glove_parameters') is not None:m.set_glove_geometry(p['glove_parameters'])
    m.set_shape(p['shape']);m.raw_rotation=np.asarray(p['raw_rotation']);m.raw_translation=np.asarray(p['raw_translation'])
    m.flat_reference=p['flat_reference'];m.sensor_joint_position=np.asarray(p['sensor_joint_position'])
    m.mount_pos=np.asarray(p['mount_pos']);m.mount_rot=np.asarray(p['mount_rot']);m.mounting_calibrated=True # same fixed candidate mounts as the holdout validator
    if p.get('delta_policy'):
        from moxian_web.mano_calibrate_delta import load_policy
        m.delta_artifact=p['delta_policy'];m.delta_policy=load_policy(p['delta_policy'],m.device)
    counts={};rows=[]
    for r in holdout_rows(source):
        if counts.get(r['stage'],0)>=a.per_stage:continue
        counts[r['stage']]=counts.get(r['stage'],0)+1;rows.append(r)
    values=[m.official_fk.parameters(m,r,CHAINS) for r in rows]
    warm_start=time.monotonic();q,root=values[0][:2]
    solve_batch(m,[rows[0]],q[None],root[None],tactile=True,cancel=threading.Event())
    torch.cuda.synchronize();warm=time.monotonic()-warm_start
    previous=None;measurements=[]
    for r,value in zip(rows,values):
        q,root=value[:2]
        torch.cuda.synchronize();start=time.monotonic()
        *_,report,previous=solve_batch(m,[r],q[None],root[None],tactile=True,previous=previous)
        torch.cuda.synchronize();ms=(time.monotonic()-start)*1000
        measurements.append(dict(stage=r['stage'],timestamp_ms=r['timestamp_ms'],milliseconds=ms,constraints_satisfied=report[0]['constraints_satisfied']))
    times=[r['milliseconds'] for r in measurements]
    result=dict(method=p['report'].get('method'),candidate_accepted=p['report']['accepted'],applied=False,mounts_enabled_as_in_holdout_validation=True,
        scope='single_frame_GPU_pose_with_tactile_and_original_triangle_queries; excludes device IO/upstream IK/web transport',
        warmup_s=warm,frames=len(rows),latency_ms=dict(p50=float(np.median(times)),p95=float(np.percentile(times,95)),max=max(times)),measurements=measurements)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='measurements'},ensure_ascii=False))
if __name__=='__main__':main()
