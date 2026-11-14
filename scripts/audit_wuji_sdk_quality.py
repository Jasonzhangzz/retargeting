#!/usr/bin/env python3
"""Frozen, complete-segment SDK confidence and measured-EMF FK residual audit."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
import numpy as np


def summary(a):
    a=np.asarray(a)
    return dict(min=float(a.min()),median=float(np.median(a)),p90=float(np.quantile(a,.9)),p99=float(np.quantile(a,.99)),max=float(a.max()))


def audit_segment(task):
    root,w,out=task
    import wuji_sdk as sdk
    from human_ik_reference.kinematics import HumanHandKinematics
    from select_wuji_clean_trajectories import receiver_fk_batch,quaternion_matrix
    sdk.set_log_level('error');side=w['side'];cal=w['calibration']
    with np.load(root/'sdk_segments'/(w['id']+'.npz')) as f:d={k:f[k] for k in f.files}
    pipeline=sdk.WujiGlove.offline_pipeline(sn=cal['sn'],hand_side=side,urdf_path=cal['urdf'] if cal['override'] else None)
    names=('wrist','thumb_cmc','thumb_mcp','thumb_ip','thumb_tip')+tuple(f+'_'+j for f in ('index_finger','middle_finger','ring_finger','pinky') for j in ('mcp','pip','dip','tip'))
    angles=[];conf=[];skconf=[];replay=0.
    for i in range(len(d['position_tx_m'])):
        frame=sdk.EmfPoseArray(header=sdk.FrameHeader(seq=int(d['source_seq'][i]),timestamp_us=int(d['timestamp_ns'][i]//1000),frame_id=side[0]+'_hand_emf_tx'),
            poses=[sdk.EmfPose(pose=sdk.Pose(position=d['position_tx_m'][i,f].tolist(),orientation=sdk.Quaternion(*d['orientation_xyzw'][i,f].tolist())),confidence=float(d['confidence'][i,f])) for f in range(5)])
        sk=pipeline.hand_skeleton().compute(frame);by_name={j.name:j for j in sk.joints}
        pts=np.asarray([by_name[n].pose.position for n in names],dtype=np.float32)
        replay=max(replay,float(abs(pts-d['skeleton_sdk_m'][i]).max()))
        q=pipeline.hand_joint_angles().compute(frame)
        angles.append(np.concatenate([np.asarray(f.angles)[:5 if k==0 else 4] for k,f in enumerate(q.fingers)]))
        conf.append([f.confidence for f in q.fingers]);skconf.append([by_name[n].confidence for n in names])
    if replay>1e-7:raise ValueError('Reading angle resource changed SDK replay: '+str(replay))
    angles=np.asarray(angles);conf=np.asarray(conf);skconf=np.asarray(skconf)
    assert np.isfinite(angles).all() and np.isfinite(conf).all() and np.isfinite(skconf).all()
    model=HumanHandKinematics(side,cal['urdf'],sdk_tx_rounding=True)
    p,r=receiver_fk_batch(model,angles);pe=np.linalg.norm(p-d['position_tx_m'],axis=-1)
    relative=r.swapaxes(-1,-2)@quaternion_matrix(d['orientation_xyzw'])
    re=np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)*.5,-1,1))
    agreement=0.
    for i in sorted({0,len(angles)//2,len(angles)-1}):
        fk=model.forward(angles[i]);agreement=max(agreement,float(abs(fk['skeleton'][:,:3,3]-d['skeleton_sdk_m'][i]).max()))
    assert agreement<2e-6, 'Independent FK model does not match SDK skeleton'
    np.savez_compressed(out/(w['id']+'.npz'),ik_confidence=conf,skeleton_confidence=skconf,fk_position_error_m=pe,fk_orientation_error_rad=re)
    return dict(id=w['id'],frames=len(angles),side=side,split=w['split'],collection=w['collection'],device=w['device'],
        emf_confidence=summary(d['confidence']),ik_confidence=summary(conf),skeleton_confidence=summary(skconf),
        frames_emf_below_08=int((d['confidence']<.8).any(1).sum()),frames_ik_below_08=int((conf<.8).any(1).sum()),
        frames_skeleton_below_08=int((skconf<.8).any(1).sum()),
        frames_fk_position_over_10mm=int((pe>.01).any(1).sum()),frames_fk_orientation_over_07rad=int((re>.7).any(1).sum()),
        frames_bad_any=int(((conf<.8).any(1)|(pe>.01).any(1)|(re>.7).any(1)).sum()),
        fk_position_error_m=summary(pe),fk_orientation_error_rad=summary(re),
        repeated_sdk_skeleton_max_abs_m=replay,human_fk_skeleton_agreement_max_m=agreement,
        ik_confidence_equals_emf_max_abs=float(abs(conf-d['confidence']).max()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--limit',type=int,default=0);a=p.parse_args();root=a.campaign;out=a.output
    out.mkdir(parents=True,exist_ok=False);conf=json.loads((root/'MANIFEST.json').read_text());sys.path[:0]=[str(root/'scripts'),conf['reference']]
    windows=json.loads((root/'selection.json').read_text())['segments'];windows=windows[:a.limit] if a.limit else windows
    started=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('fork')) as pool:
        futures=[pool.submit(audit_segment,(root,w,out)) for w in windows]
        for f in as_completed(futures):
            r=f.result();results.append(r)
            (out/'progress.json').write_text(json.dumps(dict(complete=False,segments=len(results),frames=sum(r['frames'] for r in results),expected_frames=sum(w['end']-w['start'] for w in windows))))
            print('AUDITED',len(results),sum(r['frames'] for r in results),flush=True)
    arrays={name:[] for name in ['ik_confidence','skeleton_confidence','fk_position_error_m','fk_orientation_error_rad']}
    offsets=[];offset=0
    for w in windows:
        with np.load(out/(w['id']+'.npz')) as f:
            for name in arrays:arrays[name].append(f[name])
        n=w['end']-w['start'];offsets.append(dict(id=w['id'],offset=offset,length=n,split=w['split'],side=w['side'],collection=w['collection'],device=w['device']));offset+=n
    arrays={name:np.concatenate(values) for name,values in arrays.items()};np.savez_compressed(out/'frame_quality.npz',**arrays)
    (out/'segments.json').write_text(json.dumps(offsets))
    counters=['frames_emf_below_08','frames_ik_below_08','frames_skeleton_below_08','frames_fk_position_over_10mm','frames_fk_orientation_over_07rad','frames_bad_any']
    result=dict(complete=True,optimizer_updates=0,frames=offset,segments=len(windows),expected_frames=200000 if not a.limit else offset,
        data_signature=json.loads((root/'sdk_provenance.json').read_text())['label_signature'],
        thresholds=dict(emf_and_sdk_confidence_min=.8,fk_position_error_m=.01,fk_orientation_error_rad=.7),
        threshold_scope='Post-hoc diagnostic only; no data or model is changed',
        sdk_confidence_semantics='Vendor finger/skeleton confidence; not assumed calibrated probabilities',
        counts={k:sum(r[k] for r in results) for k in counters},
        distributions={k:summary(v) for k,v in arrays.items()},
        repeated_sdk_skeleton_max_abs_m=max(r['repeated_sdk_skeleton_max_abs_m'] for r in results),
        independent_human_fk_skeleton_max_abs_m=max(r['human_fk_skeleton_agreement_max_m'] for r in results),
        ik_confidence_equals_emf_max_abs=max(r['ik_confidence_equals_emf_max_abs'] for r in results),
        by_split={s:dict(frames=sum(r['frames'] for r in results if r['split']==s),**{k:sum(r[k] for r in results if r['split']==s) for k in counters}) for s in ['train','val']},
        elapsed_seconds=time.perf_counter()-started,evaluated_at=datetime.now(timezone.utc).isoformat(),per_segment=results)
    (out/'report.json').write_text(json.dumps(result,indent=2));print('AUDIT_COMPLETE',json.dumps({k:v for k,v in result.items() if k!='per_segment'}),flush=True)


if __name__=='__main__':main()
