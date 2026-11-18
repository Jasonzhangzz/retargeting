#!/usr/bin/env python3
"""Run one retained-data least-squares fit and register a preview-only branch."""
import argparse,hashlib,json,sys,threading,time,uuid,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from moxian_web.mano_calibration import fit_capture,prepare_rows
from moxian_web.qnbot_least_squares import write_json


def digest(path):
    if not path.exists():return None
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--device-home',type=Path,required=True)
    parser.add_argument('--capture',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--seconds',type=float,default=600.)
    parser.add_argument('--iterations',type=int,default=6)
    parser.add_argument('--shared-batch-size',type=int,default=96)
    parser.add_argument('--stages',nargs='+',choices=('extrinsics','dimensions','kinematics','tactile','surface'))
    a=parser.parse_args();torch.set_num_threads(1)
    home=a.device_home.resolve();source=(a.capture or home/'motion-capture.json').resolve()
    run_id=uuid.uuid4().hex;folder=(a.output or home/'qnbot'/'runs'/'least_squares'/run_id).resolve()
    if folder.exists() and any(folder.iterdir()):raise ValueError('输出目录非空，请为本次求解指定独立目录')
    folder.mkdir(parents=True,exist_ok=True)
    protected=[source,home/'current.json',home/'qnbot/current.json']
    protected.extend(home.glob('methods/*/candidate.json'));protected.extend(p for p in home.glob('qnbot/methods/*/candidate.json') if p.parent.name!='least_squares')
    before={str(p):digest(p) for p in protected}
    write_json(folder/'integrity-before.json',before)
    document=json.loads(source.read_text());rows=document['rows'];last=rows[-1]
    profile=dict(surface_backend='qnbot',side=document.get('side') or last['hand_side'],serial=document['serial'],artifact_home=str(home),run_output=str(folder),run_id=run_id,
        calibration_fingerprint=document.get('calibration_fingerprint') or last.get('calibration_fingerprint'),skeleton_runtime_revision=last.get('skeleton_runtime_revision'),capture_session_id=document.get('motion_session_id'))
    # Stored IK poses are derived caches. Resolve the CURRENT executable seed
    # policy from the matching original initialization before selecting caches.
    root=Path(__file__).resolve().parents[1];ik_home=home.parents[1]/'ik'/home.name
    command=[str(root/'.venv-moxian-ik/bin/python'),'-I',str(root/'moxian_web/qnbot_lsq_policy.py'),'--home',str(ik_home),'--side',profile['side'],'--fingerprint',profile['calibration_fingerprint']]
    resolved=json.loads(subprocess.check_output(command,text=True).strip().splitlines()[-1])
    profile['skeleton_runtime_revision']=resolved['skeleton_runtime_revision']
    write_json(folder/'seed-policy.json',resolved)
    options=dict(max_seconds=a.seconds,max_iterations=a.iterations,shared_batch_size=a.shared_batch_size)
    if a.stages:options['stages']=a.stages
    began=time.monotonic();cancel=threading.Event()
    with (folder/'progress.jsonl').open('w') as log:
        def progress(value):
            event=dict(value,elapsed_s=time.monotonic()-began)
            log.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n');log.flush()
            print(json.dumps({k:v for k,v in event.items() if k!='training_sample'},ensure_ascii=False),flush=True)
        prepared,preparation=prepare_rows(profile,rows,progress,cancel)
        preparation_s=time.monotonic()-began
        candidate,report=fit_capture(None,profile,rows[0]['mapping'],prepared,progress,cancel,method='least_squares',surface_backend='qnbot',solver_options=options)
    report.update(preparation=preparation,raw_pose_preparation_s=preparation_s,request_to_completion_s=time.monotonic()-began,source_capture_file=str(source),source_file_sha256=before[str(source)])
    report['end_to_end_s']=report['request_to_completion_s'];report['time_target_met']=report['end_to_end_s']<=60;report['performance_passed']=report['time_target_met']
    after={str(p):digest(p) for p in protected}
    integrity=dict(unchanged=before==after,before=before,after=after)
    write_json(folder/'integrity.json',integrity)
    if before!=after:raise RuntimeError('原始记录、当前模型或已有分支在运行期间发生变化；结果未注册')
    candidate['report']=report;write_json(folder/'candidate.json',candidate);write_json(folder/'report.json',report)
    branch=home/'qnbot/methods/least_squares'
    if (branch/'candidate.json').exists():write_json(branch/'history'/f'{time.time_ns()}.json',json.loads((branch/'candidate.json').read_text()))
    write_json(branch/'candidate.json',candidate);write_json(branch/'report.json',report)
    trace_home=home/'qnbot/training-traces';trace_home.mkdir(parents=True,exist_ok=True);trace_path=trace_home/f'{run_id}.jsonl'
    trace_path.write_text(''.join(json.dumps(dict(run_id=run_id,method='least_squares',state='completed',training_sample=p),ensure_ascii=False,allow_nan=False)+'\n' for p in report['solver']['training_trace']))
    write_json(trace_home/'least_squares.json',dict(file=trace_path.name,run_id=run_id,method='least_squares',surface_backend='qnbot'))
    print(json.dumps(dict(output=str(folder),candidate=str(branch/'candidate.json'),accepted=report['accepted'],reason=report['reason'],unchanged=integrity['unchanged'],seconds=report['end_to_end_s']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
