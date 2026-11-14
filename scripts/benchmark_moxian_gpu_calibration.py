#!/usr/bin/env python3
"""Full-record, non-publishing GPU calibration benchmark (both methods/caches)."""
import argparse,copy,json,sys,threading,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moxian_web.hand_model import DEFAULT_MANO,DEFAULT_MATERIAL,MANO_FIT_REVISION,MANO_SOLVER_REVISION
from moxian_web.mano_runtime import ManoSurface
from moxian_web.official_mano import reference_from_frame,OFFICIAL_MANO_MODE
from moxian_web.mano_calibration import fit_capture
from moxian_web.mano_objective import SolverSettings


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--device-home',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--methods',nargs='+',choices=('parameters','delta','flashsac'),default=['parameters','delta']);ap.add_argument('--runs',type=int,default=3)
    ap.add_argument('--cache',choices=('both','cold','warm'),default='both');ap.add_argument('--max-seconds',type=float,default=30)
    ap.add_argument('--eager',action='store_true');ap.add_argument('--skip-warmup',action='store_true');ap.add_argument('--smoke-steps',type=int)
    ap.add_argument('--resume',action='store_true',help='Keep complete matching trials; rerun interrupted trials')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    doc=json.loads((a.device_home/'motion-capture.json').read_text());rows=doc['rows'];side=rows[0]['hand_side'];mapping=rows[0]['mapping']
    serial=doc.get('serial') or json.loads((a.device_home/'shared-capture.json').read_text()).get('serial','offline')
    path=DEFAULT_MANO.with_name('MANO_'+side.upper()+'.pkl');m=ManoSurface(path,side=side);m.enable_official_fk();m.build_binding(mapping)
    prepared=json.loads((a.device_home/'fit-wrist-targets.json').read_text())['rows']
    profile=dict(schema_version=1,serial=serial,side=side,weights_hash=m.weights_hash,tracking_source='official_qnbot_skeleton_fk_v1',shape=m.shape.tolist(),dimension_mode=OFFICIAL_MANO_MODE,material=dict(DEFAULT_MATERIAL),raw_rotation=m.raw_rotation.tolist(),raw_translation=m.raw_translation.tolist(),mount_pos=m.mount_pos.tolist(),mount_rot=m.mount_rot.tolist(),sensor_joint_position=m.sensor_offsets().tolist(),flat_reference=None,mapping=mapping,penetration_tolerance_m=.002,calibration_pose_offsets=np.zeros((15,3)).tolist(),artifact_home=str(a.device_home.resolve()),calibration_fingerprint=prepared[0]['calibration_fingerprint'],skeleton_runtime_revision=prepared[0]['skeleton_runtime_revision'])
    if not a.eager and not a.skip_warmup:
        from moxian_web.mano_gpu_warmup import warmup
        from moxian_web.mano_calibration import prepare_rows
        began=time.monotonic();prepared,_=prepare_rows(profile,rows,lambda _:None,threading.Event(),use_cache=True)
        warmup(path,profile,mapping,prepared,lambda v:print('WARMUP',json.dumps(v,ensure_ascii=False),flush=True),threading.Event(),methods=a.methods)
        warmup_name='resume-warmup.json' if a.resume and (a.output/'warmup.json').exists() else 'warmup.json'
        (a.output/warmup_name).write_text(json.dumps(dict(seconds=time.monotonic()-began,excluded_from_benchmark=True)))
    previous=json.loads((a.output/'summary.json').read_text()) if a.resume and (a.output/'summary.json').exists() else []
    from moxian_web.mano_fit import choose_rows,holdout_rows
    train_stamps=[r['timestamp_ms'] for r in choose_rows(rows)[0]]
    valid_stamps=[r['timestamp_ms'] for r in holdout_rows(rows)]
    result=[]
    for method in a.methods:
        for cache in (['cold','warm'] if a.cache=='both' else [a.cache]):
            for run in range(a.runs):
                folder=a.output/f'{method}-{cache}-{run+1}';folder.mkdir(exist_ok=True)
                saved=next((r for r in previous if r.get('method')==method and r.get('cache')==cache and r.get('run')==run+1 and not r.get('error')),None)
                if saved and (folder/'candidate.json').exists() and (folder/'report.json').exists():
                    candidate=json.loads((folder/'candidate.json').read_text());report=json.loads((folder/'report.json').read_text())
                    if candidate.get('report')!=report or report.get('fit_revision')!=MANO_FIT_REVISION or report.get('solver_revision')!=MANO_SOLVER_REVISION or report.get('method')!=method or any(candidate.get(k)!=profile.get(k) for k in ('serial','side','weights_hash','calibration_fingerprint','skeleton_runtime_revision')):
                        raise ValueError('已有试验与当前模型／设备不匹配，请使用新的输出目录')
                    if report.get('solver',{}).get('parameter_snapshot',{}).get('frame_timestamps_ms')!=train_stamps or [r['timestamp_ms'] for r in report.get('runtime_solver_frames',[])]!=valid_stamps:
                        raise ValueError('已有试验与当前采集序列不匹配，请使用新的输出目录')
                    result.append(saved);print('RESUME',method,cache,run+1,'complete result preserved',flush=True);continue
                torch.manual_seed(7)
                began=time.monotonic();log=(folder/'progress.jsonl').open('w')
                def progress(v):log.write(json.dumps(v,ensure_ascii=False)+'\n');log.flush();print(method,cache,run+1,json.dumps(v,ensure_ascii=False),flush=True)
                try:
                    opts=dict(max_seconds=a.max_seconds,use_cache=cache=='warm',compile_loss=not a.eager)
                    if a.smoke_steps:opts['solver_settings']=SolverSettings(adam_steps=a.smoke_steps,lm_steps=0)
                    candidate,report=fit_capture(path,copy.deepcopy(profile),mapping,copy.deepcopy(rows),progress,threading.Event(),method=method,**opts)
                    (folder/'candidate.json').write_text(json.dumps(candidate,ensure_ascii=False,allow_nan=False))
                    (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,allow_nan=False))
                    result.append(dict(method=method,cache=cache,run=run+1,accepted=report['accepted'],performance_passed=report['performance_passed'],end_to_end_s=report['end_to_end_s'],compilation_s=report.get('compilation_s',0),rejection_reasons=report['rejection_reasons']))
                except Exception as exc:
                    import traceback;traceback.print_exc();result.append(dict(method=method,cache=cache,run=run+1,accepted=False,performance_passed=False,error=str(exc),end_to_end_s=time.monotonic()-began))
                finally:log.close()
                (a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
