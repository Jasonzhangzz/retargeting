#!/usr/bin/env python3
"""CPU-only diagnostic of source SLSQP stopping on complete dev records.

SDK skeletons enter this reference-consistency diagnostic only. Not a new
initializer or a production timing result. Source tree is never modified.
"""
import argparse,json,sys,time,collections
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;out=r/'oneway_comparison/source_slsqp_stopping';out.mkdir(exist_ok=True);sys.path.insert(0,str(r/'reference'))
    from retarget_reference.session import RetargetSession
    records=[v for v in json.loads((r/'selection.json').read_text())['records'] if v['split']=='development'];rows=[]
    (out/'BUDGET.json').write_text(json.dumps(dict(source_records=5,groups=4,frames=11540,maximum_function_evaluations=4*5*130+(11540-4*5)*11,neural_optimizer_updates=0,device='CPU',scope='Explicit SDK-skeleton reference diagnostic; never production initialization or GPU speed claim'),indent=2))
    for config in ['builtin','hl_urdf']:
        for side in ['left','right']:
            raw=[];filtered=[];sdk=[];indices=[];offsets=[0];status=[collections.Counter(),collections.Counter()];fallback=[0,0];evaluations=[0,0];per_frame_evaluations=[];per_frame_status=[];begin=time.perf_counter()
            for record in records:
                if side not in record['sides']:continue
                i=record['index'];gt=np.load(r/'sdk'/config/f'trajectory_{i}_{side}.npz');data=np.load(r/'inputs'/f'trajectory_{i}_{side}.npz');n=len(gt['skeleton']);local=data['segment_offsets'];session=RetargetSession.for_hand('WujiHand2',side);starts=set(local[:-1].tolist())
                for t in range(n):
                    if t in starts:session.reset()
                    q=session.step(gt['skeleton'][t]);filtered.append(q);raw.append(session.last_trace['unfiltered']);sdk.append(gt['q_sdk_filtered'][t]);indices.append(i);counts=[];codes=[]
                    for s,trace in enumerate(session.last_trace['stages']):
                        status[s][trace['status']]+=1;fallback[s]+=int(trace['fallback']);evaluations[s]+=len(trace['evaluations']);counts.append(len(trace['evaluations']));codes.append(trace['status'])
                    per_frame_evaluations.append(counts);per_frame_status.append(codes)
                offsets.append(offsets[-1]+n);print('SOURCE_RECORD',config,side,i,n,flush=True)
            q=np.asarray(filtered);gt=np.asarray(sdk);error=q.astype(float)-gt.astype(float);row=dict(configuration=config,side=side,frames=len(q),filtered_bitwise_equal=bool(np.array_equal(q,gt)),max_filtered_difference_rad=float(abs(error).max()),rmse_filtered_difference_rad=float(np.sqrt(np.square(error).mean())),statuses=[dict(v) for v in status],fallback_frames=fallback,function_evaluations=evaluations,wall_s=time.perf_counter()-begin,scope='Source SLSQP matches SDK; wall time is CPU diagnostic, not production benchmark')
            np.savez_compressed(out/f'{config}_{side}.npz',q_source_raw=np.asarray(raw),q_source_filtered=q,q_sdk_filtered=gt,record_indices=np.asarray(indices),record_offsets=np.asarray(offsets),stage_evaluations=np.asarray(per_frame_evaluations),stage_status=np.asarray(per_frame_status));rows.append(row);(out/'summary.json').write_text(json.dumps(dict(complete=len(rows)==4,all_bitwise_equal=all(v['filtered_bitwise_equal'] for v in rows),rows=rows,neural_optimizer_updates=0),indent=2));print('SOURCE_GROUP',json.dumps(row),flush=True)

if __name__=='__main__':main()
