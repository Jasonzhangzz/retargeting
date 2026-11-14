"""Replay stored raw observations through the same official calibrated C engine."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moxian_web.emf_slsqp import skeleton_engine

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--config',type=Path,required=True);ap.add_argument('--fingerprint',required=True);ap.add_argument('--runtime-revision',default='');ap.add_argument('--personalized',type=Path);ap.add_argument('--tactile',action='store_true');a=ap.parse_args()
    from qnbot_sdk.glove._resources import builtin_skeleton_package_root
    payload=json.loads(a.input.read_text());side=payload['rows'][0].get('hand_side','left')
    if a.personalized:
        from moxian_web.emf_slsqp import SLSQPSkeleton
        engine=SLSQPSkeleton(builtin_skeleton_package_root(),a.config,side=side,personalized=json.loads(a.personalized.read_text()))
    else:engine=skeleton_engine(builtin_skeleton_package_root(),a.config,side=side)
    try:
        mapper=getattr(engine,'mapper',engine)
        if mapper.lib.qh_engine_physics_enabled(mapper.handle):raise ValueError('离线转换必须关闭动力学')
        began=time.monotonic();total=len(payload['rows']);previous_sequence=None
        for index,row in enumerate(payload['rows']):
            sequence=(row.get('partition'),row.get('stage'),row.get('window_id'))
            if previous_sequence is not None and sequence!=previous_sequence:
                # Independent recording windows cannot seed train poses from a
                # preceding held-out window (notably the appended spread action).
                if hasattr(engine,'previous'):engine.previous=None;engine.previous_timestamp=None
            previous_sequence=sequence
            # Calibration validation uses EMF only; gesture/tactile association
            # must not change the skeleton seeds being evaluated.
            result=engine.step(row if a.tactile else {k:v for k,v in row.items() if k not in ('contact','tactile_timestamp_ms')})
            row['skeleton_original']=row.get('skeleton');row['skeleton']=result['pose'];row['skeleton_targets']=result['targets'];row['calibration_fingerprint']=a.fingerprint;row['skeleton_runtime_revision']=a.runtime_revision
            if index % 20 == 0 or index+1 == total:
                elapsed=time.monotonic()-began
                print(json.dumps(dict(completed=index+1,total=total,elapsed_s=elapsed,estimated_remaining_s=elapsed*(total-index-1)/(index+1))),flush=True)
    finally:engine.close()
    temp=a.output.with_suffix('.tmp');temp.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False));temp.replace(a.output)
if __name__=='__main__':main()
