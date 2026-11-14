#!/usr/bin/env python3
"""Time only the original SDK path, excluding reference FK/report generation."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--repeats',type=int,default=3);p.add_argument('--component-timing',action='store_true')
    a=p.parse_args();root=a.root;sys.path.insert(0,str(root/'sdk831'))
    import wuji_sdk as sdk
    sdk.set_log_level('error');assert sdk.__version__=='2026.8.31'
    selection=json.loads((root/'selection.json').read_text());inputs=[]
    for configuration in ('builtin','hl_urdf'):
        for item in selection['trajectories']:
            for side in ('left','right'):
                path=root/'gt'/configuration/f'trajectory_{item["index"]}_{side}.npz'
                arrays=np.load(path);inputs.append((configuration,item,side,{key:arrays[key] for key in arrays.files}))
    report=dict(sdk_version=sdk.__version__,warmup=None,repeats=[],
        timing='CPU-resident same raw EMF. End-to-end includes fresh SDK pipeline/session per episode and frame-object construction, skeleton extraction and retarget step; excludes disk, diagnostics, serialization and independent FK.')
    for repeat in range(a.repeats+1):
        rows=[]
        for configuration,item,side,data in inputs:
            started=time.perf_counter();urdf=root/'assets'/f'{configuration}_{side}.urdf'
            pipeline=sdk.WujiGlove.offline_pipeline(sn=item['recording_metadata']['devices'][side]['sn'],hand_side=side,
                urdf_path=str(urdf) if configuration=='hl_urdf' else None)
            session=sdk.RetargetSession.for_hand(sdk.HandModel.WujiHand2,side=sdk.Handedness.Left if side=='left' else sdk.Handedness.Right)
            assert pipeline.urdf_source==('override' if configuration=='hl_urdf' else 'builtin_default')
            construction_end=time.perf_counter();solve=0.;objects=0.;human_seconds=0.;packing_seconds=0.;robot_seconds=0.;observed=[]
            names=('wrist','thumb_cmc','thumb_mcp','thumb_ip','thumb_tip')+tuple(
                f+'_'+j for f in ('index_finger','middle_finger','ring_finger','pinky') for j in ('mcp','pip','dip','tip'))
            for i in range(len(data['position_tx_m'])):
                start=time.perf_counter()
                frame=sdk.EmfPoseArray(header=sdk.FrameHeader(seq=int(data['source_seq'][i]),timestamp_us=int(data['timestamp_ns'][i]//1000),frame_id=side[0]+'_hand_emf_tx'),
                    poses=[sdk.EmfPose(pose=sdk.Pose(position=data['position_tx_m'][i,j].tolist(),orientation=sdk.Quaternion(*data['orientation_xyzw'][i,j].tolist())),confidence=float(data['confidence'][i,j])) for j in range(5)])
                solve_start=time.perf_counter()
                skeleton=pipeline.hand_skeleton().compute(frame);human_end=time.perf_counter()
                lookup={j.name:j for j in skeleton.joints}
                points=np.asarray([lookup[n].pose.position for n in names],np.float32)
                robot_start=time.perf_counter();q=session.step(points);end=time.perf_counter()
                human_seconds+=human_end-solve_start;packing_seconds+=robot_start-human_end;robot_seconds+=end-robot_start
                objects+=solve_start-start;solve+=end-solve_start;observed.append(q)
            total=construction_end-started+objects+solve
            if not np.array_equal(np.asarray(observed),data['q_sdk_rad']):raise ValueError('SDK timing repeat differs from frozen GT')
            rows.append(dict(configuration=configuration,side=side,trajectory=item['index'],frames=len(observed),
                sdk_solve_seconds=solve,human_ik_seconds=human_seconds,skeleton_packing_seconds=packing_seconds,robot_retarget_seconds=robot_seconds,frame_construction_seconds=objects,initialize_seconds=construction_end-started,end_to_end_seconds=total))
        run=dict(streams=rows,configuration_totals={c:dict(frames=sum(x['frames'] for x in rows if x['configuration']==c),
            solve_seconds=sum(x['sdk_solve_seconds'] for x in rows if x['configuration']==c),
            human_ik_seconds=sum(x['human_ik_seconds'] for x in rows if x['configuration']==c),
            skeleton_packing_seconds=sum(x['skeleton_packing_seconds'] for x in rows if x['configuration']==c),
            robot_retarget_seconds=sum(x['robot_retarget_seconds'] for x in rows if x['configuration']==c),
            end_to_end_seconds=sum(x['end_to_end_seconds'] for x in rows if x['configuration']==c)) for c in ('builtin','hl_urdf')})
        if repeat==0:report['warmup']=run
        else:report['repeats'].append(run)
        (root/('sdk_benchmark_components.json' if a.component_timing else 'sdk_benchmark.json')).write_text(json.dumps(report,indent=2));print('SDK_TIMED',repeat,json.dumps(run['configuration_totals']),flush=True)
    print('SDK_TIMING_COMPLETE',flush=True)


if __name__=='__main__':main()
