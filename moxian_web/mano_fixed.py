"""Frozen MANO forward replay: no optimizer, tactile feedback or pose fitting."""
import time
import numpy as np
import torch
from .contact import FINGERS
from scripts.calibrate_wuji_mano import CHAINS

BACKEND='fixed_mano_v1'


@torch.no_grad()
def replay(model,selected,sequence,mounting_calibrated,progress=None,cancel=None):
    began=time.monotonic();model.mounting_calibrated=mounting_calibrated
    values=[model.official_fk.parameters(model,row,CHAINS) for row in sequence]
    t=lambda value:torch.as_tensor(np.asarray(value),device=model.device,dtype=torch.float32)
    q=t([v[0] for v in values]);roots=t([v[1] for v in values])
    if getattr(model,'delta_policy',None) is not None:
        from .mano_joint_delta import infer
        q=infer(model,sequence,np.asarray([v[0] for v in values]),np.asarray([v[1] for v in values]))
    v,j,r,_=model.decode(q,roots)
    pred=j[:,[c[-1] for c in CHAINS]]+(r[:,[c[-1] for c in CHAINS]]@t(model.sensor_offsets())[None,:,:,None]).squeeze(-1)
    observed=t([[row.get('fingers',{}).get(f,{}).get('position',[0,0,0]) for f in FINGERS] for row in sequence])@t(model.raw_rotation).T+t(model.raw_translation)
    from .mano_objective import participation
    mask=participation(sequence,training=True)
    error=(pred-observed).norm(dim=-1).cpu().numpy()*1000
    # Collision is a diagnostic query only. It never moves the frozen hand.
    from .mano_gpu_collision import BatchedSurfaceCollision
    cache={};reports=[]
    for start in range(0,len(sequence),32):
        if cancel is not None and cancel.is_set():raise InterruptedError('固定模型验证已取消')
        chunk=v[start:start+32];n=len(chunk)
        if n not in cache:cache[n]=BatchedSurfaceCollision(model,n)
        witness=cache[n].query(chunk);depth=witness['maximum'].cpu().tolist()
        valid=bool(witness['valid'] & ~witness['witness_overflow'].any())
        for k,d in enumerate(depth):
            errors=error[start+k];source={f:dict(position_error_mm=float(errors[i]),satisfied=bool(errors[i]<=15)) for i,f in enumerate(FINGERS)}
            collision=dict(max_penetration_m=d,query_valid=valid,constraints_satisfied=valid and d<=.002)
            reports.append(dict(revision=BACKEND,status='fixed_forward',constraints_satisfied=collision['constraints_satisfied'],source_tracking_satisfied=all(p['satisfied'] for i,p in enumerate(source.values()) if mask[start+k,i]),source_tracking_errors=source,collision=collision,contact_pairs=[],contact_constraints_satisfied=True,optimizer_steps=0,geometry_updated=False,tactile_pose_feedback=False,elapsed_ms=0.))
        if progress:progress(start+n,len(sequence))
    elapsed=(time.monotonic()-began)*1000
    for report in reports:report['elapsed_ms']=elapsed/len(sequence)
    from .mano_fit import bilateral_contact
    trajectory=[dict(stage=row['stage'],timestamp_ms=row['timestamp_ms'],rmse_mm=float(np.sqrt(np.sum(e**2*m)/max(m.sum(),1))),bilateral_contact=bilateral_contact(row),guided_contact=row.get('guided_contact',False),is_approach=row.get('motion_phase') in ('apart','approach','release','trajectory') and not row.get('contact_confirmed')) for row,e,m in zip(sequence,error,mask)]
    model._gpu_replay_reports=reports
    stamps={row['timestamp_ms'] for row in selected};ids=[i for i,row in enumerate(sequence) if row['timestamp_ms'] in stamps]
    return q[ids],roots[ids],trajectory
