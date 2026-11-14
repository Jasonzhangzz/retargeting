#!/usr/bin/env python3
"""Select complete dual-hand episodes using input-only quality/kinematic gates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import h5py


def receiver_fk_batch(model,q):
    """Vectorized input-quality FK only; this never performs an IK/LM solve."""
    positions=[];rotations=[];n=len(q);eye=np.eye(3)
    for chain in model._chains:
        r=np.broadcast_to(model._tx_inverse[:3,:3],(n,3,3)).copy()
        p=np.broadcast_to(model._tx_inverse[:3,3],(n,3)).copy()
        for j in chain:
            p=p+(r@j['origin'][:3,3,None]).squeeze(-1)
            r=r@j['origin'][:3,:3]
            if j['index'] is not None:
                x,y,z=j['axis'];k=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
                angle=q[:,j['index'],None,None]
                r=r@(eye+np.sin(angle)*k+(1-np.cos(angle))*(k@k))
        positions.append(p);rotations.append(r)
    return np.stack(positions,1),np.stack(rotations,1)


def quaternion_matrix(q):
    q=q/np.linalg.norm(q,axis=-1,keepdims=True);x,y,z,w=np.moveaxis(q,-1,0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),
        1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(*q.shape[:-1],3,3)


def inspect_raw(path):
    try:
      with h5py.File(path,'r',locking=False) as h:
        data={};reasons=[];summary={}
        attrs={k:(v.decode() if isinstance(v,bytes) else v) for k,v in h.attrs.items()}
        for side in ('left','right'):
          g=h[side];p=g['emf_poses/position'][:];q=g['emf_poses/orientation_xyzw'][:];c=g['emf_poses/confidence'][:]
          ts=g['source_timestamp_ns'][:];angles=g['hand_joint_angles/angles'][:]
          if len(p)<120 or len(p)>1200:reasons.append(side+':length_outside_120_1200')
          if not all(np.isfinite(x).all() for x in (p,q,c,angles)):reasons.append(side+':nonfinite')
          distance=np.linalg.norm(p,axis=-1);qnorm=np.linalg.norm(q,axis=-1)
          if c.min()<.8:reasons.append(side+':confidence_below_0.8')
          if distance.max()>.25 or distance.min()<.01:reasons.append(side+':implausible_tx_receiver_distance')
          if abs(qnorm-1).max()>.002:reasons.append(side+':nonunit_quaternion')
          dt=np.diff(ts)*1e-9
          if len(dt)==0 or dt.min()<=0 or dt.max()>.05:reasons.append(side+':timestamp_gap')
          dp=np.linalg.norm(np.diff(p,axis=0),axis=-1)
          rn=q/np.maximum(qnorm[...,None],1e-12)
          dr=2*np.arccos(np.clip(abs((rn[1:]*rn[:-1]).sum(-1)),0,1))
          if len(dp) and dp.max()>.025:reasons.append(side+':position_jump_over_25mm')
          if len(dr) and dr.max()>.5:reasons.append(side+':rotation_jump_over_0.5rad')
          summary[side]=dict(frames=len(p),confidence_min=float(c.min()),confidence_mean=c.mean(0).tolist(),
              receiver_distance_max_m=float(distance.max()),position_jump_max_m=float(dp.max()) if len(dp) else None,
              rotation_jump_max_rad=float(dr.max()) if len(dr) else None,timestamp_gap_max_s=float(dt.max()) if len(dt) else None)
          data[side]=dict(position=p,quaternion=q,confidence=c,angles=angles)
        return dict(source_h5=str(path),passed_raw=not reasons,reasons=reasons,summary=summary,data=data,attrs=attrs)
    except Exception as e:return dict(source_h5=str(path),passed_raw=False,reasons=[type(e).__name__+':'+str(e)],summary={})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--source',type=Path,default=Path('/mnt/jfs/x2robot-prod/x2robot-data/collection/18/RP001_VR/XRRD11ZD2604070080'))
    p.add_argument('--max-collections',type=int,default=30);p.add_argument('--candidates-per-collection',type=int,default=40)
    a=p.parse_args();root=a.base/'good_data_v2';root.mkdir(exist_ok=True)
    for name in ('assets','reference','sdk831','runtime','scripts','warp_cache','causal_best.pt'):
        dst=root/name
        if not dst.exists():dst.symlink_to(a.base/name,target_is_directory=(a.base/name).is_dir())
    sys.path[:0]=[str(root/'runtime'),str(root/'reference')]
    from human_ik_reference.kinematics import HumanHandKinematics
    models={}
    for side in ('left','right'):
        human=HumanHandKinematics(side,root/'assets'/f'builtin_{side}.urdf',sdk_tx_rounding=True)
        models[side]=human
    # Name timestamp, rather than file modification time, identifies acquisition order.
    folders=sorted([p for p in a.source.iterdir() if p.is_dir() and p.name[:14].isdigit()],key=lambda p:p.name,reverse=True)
    rng=np.random.default_rng(42);audit=[];selected=[]
    for folder in folders[:a.max_collections]:
      episodes=sorted([p for p in folder.iterdir() if p.is_dir() and '@' in p.name])
      if not episodes:continue
      order=rng.permutation(len(episodes));paths=[episodes[i]/'wuji_glove_offline.h5' for i in order[:a.candidates_per_collection]]
      print('SCREEN_COLLECTION',folder.name,len(paths),flush=True)
      found=None
      # Read-only input screening; no CPU IK/LM optimization occurs here.
      with ThreadPoolExecutor(max_workers=4) as pool:
       for result in pool.map(inspect_raw,paths):
        data=result.pop('data',None);attrs=result.pop('attrs',{})
        if result['passed_raw']:
          kinematic={}
          for side in ('left','right'):
            d=data[side];angles=d['angles'];qhuman=np.concatenate([angles[:,f,:5 if f==0 else 4] for f in range(5)],axis=-1)
            pos,rot=receiver_fk_batch(models[side],qhuman)
            rr=quaternion_matrix(d['quaternion'])
            ep=np.linalg.norm(pos-d['position'],axis=-1)
            relative=rot.swapaxes(-1,-2)@rr
            er=np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)*.5,-1,1))
            detail=dict(position_p99_m=np.quantile(ep,.99,axis=0).tolist(),rotation_p99_rad=np.quantile(er,.99,axis=0).tolist(),
                position_max_m=ep.max(0).tolist(),rotation_max_rad=er.max(0).tolist())
            kinematic[side]=detail
            if np.quantile(ep,.99,axis=0).max()>.01:result['reasons'].append(side+':recorded_human_fk_position_p99_over_10mm')
            if np.quantile(er,.99,axis=0).max()>.7:result['reasons'].append(side+':recorded_human_fk_orientation_p99_over_0.7rad')
          result['kinematic_quality']=kinematic
        result['passed']=not result['reasons'];audit.append(result)
        print('CANDIDATE',Path(result['source_h5']).parent.name,'PASS' if result['passed'] else result['reasons'],flush=True)
        (root/'selection_audit.json').write_text(json.dumps(dict(criteria=dict(confidence_min=.8,receiver_distance_m=[.01,.25],frames=[120,1200],
            timestamp_gap_max_s=.05,position_jump_max_m=.025,rotation_jump_max_rad=.5,
            recorded_human_fk_position_p99_max_m=.01,recorded_human_fk_rotation_p99_max_rad=.7,
            criterion_scope='Input confidence/geometry/timestamps and recording human-kinematic self-consistency only; no network or Wuji2 q error enters selection.'),
            attempted_collections=[f.name for f in folders[:folders.index(folder)+1]],candidates=audit),indent=2))
        if result['passed']:
          found=(result,attrs);break
      if found:
        result,attrs=found;index=len(selected);source=Path(result['source_h5']);dst=root/'inputs'/f'trajectory_{index}'/'wuji_glove_offline.h5'
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dst)
        metadata=None
        for value in attrs.values():
          if isinstance(value,str):
           try:
            parsed=json.loads(value)
            if isinstance(parsed,dict) and 'devices' in parsed:metadata=parsed
           except json.JSONDecodeError:pass
        if metadata is None:
            manifest=source.parent/'wuji_glove_offline_manifest.json'
            obj=json.loads(manifest.read_text());print('METADATA_MANIFEST_KEYS',list(obj),flush=True)
            metadata=obj.get('metadata',obj.get('recording_metadata'))
        if metadata is None:raise ValueError('Recording metadata not found; attrs='+str(list(attrs)))
        record=dict(index=index,collection=folder.name,episode=str(source.parent),source_h5=str(source),cached_h5=str(dst),
            h5_sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),cached_h5_relative=f'inputs/trajectory_{index}/wuji_glove_offline.h5',recording_metadata=metadata,
            quality=result,sides={side:dict(frames=result['summary'][side]['frames']) for side in ('left','right')})
        selected.append(record)
        (root/'selection.json').write_text(json.dumps(dict(selection='Newest collections with an entire dual-hand episode passing declared input-quality gates; deterministic random episode order seed42. Bad collection inputs are replaced, never clipped or selected by final q error.',trajectories=selected),indent=2))
        print('SELECTED',index,folder.name,source.parent.name,flush=True)
      if len(selected)==5:break
    if len(selected)!=5:raise RuntimeError(f'Found only {len(selected)} clean collections; see selection_audit.json')
    (root/'CLEAN_SELECTION_COMPLETE').write_text('complete\n');print('CLEAN_SELECTION_COMPLETE',flush=True)


if __name__=='__main__':main()
