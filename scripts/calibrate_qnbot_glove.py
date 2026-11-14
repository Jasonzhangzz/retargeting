#!/usr/bin/env python3
"""Offline native-mesh fit; never applies a candidate to a connected glove."""
import argparse,json,sys,threading
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from moxian_web.qnbot_calibration import fit_capture


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--capture',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--method',choices=('moxian_wuji','wuji_native','bone_extrinsics','parameters','delta','flashsac'),default='moxian_wuji');ap.add_argument('--maxeval',type=int,default=20000);ap.add_argument('--steps',type=int,default=480);ap.add_argument('--seconds',type=float,default=45);ap.add_argument('--batch-size',type=int,default=0);ap.add_argument('--optimizer',choices=('adam','muon'),default='adam');ap.add_argument('--resume',type=Path);ap.add_argument('--serial');ap.add_argument('--side',choices=('left','right'));ap.add_argument('--compile',action='store_true');args=ap.parse_args()
    torch.set_num_threads(1);document=json.loads(args.capture.read_text());rows=document['rows'];args.output.mkdir(parents=True,exist_ok=True)
    serial=args.serial or document.get('serial');side=args.side or document.get('side') or rows[-1].get('hand_side')
    if not serial or side not in ('left','right'):ap.error('记录缺少设备／手侧，请提供 --serial 和 --side')
    profile=dict(surface_backend='qnbot',serial=serial,side=side,artifact_home=str(args.output),calibration_fingerprint=document.get('calibration_fingerprint'),skeleton_runtime_revision=rows[-1].get('skeleton_runtime_revision'),contact_reference=document.get('contact_reference'),capture_session_id=document.get('motion_session_id'))
    if args.method in ('moxian_wuji','wuji_native'):
        from moxian_web.wuji_native_capture import WujiNativeCapture
        WujiNativeCapture.restore(document)
    if args.method=='bone_extrinsics':
        from moxian_web.static_capture import StaticHandCapture
        restored=StaticHandCapture.restore(document,allow_partial=True)
        if restored.status not in ('fit_ready','complete'):ap.error('请先完成17动作拟合采集')
        profile['static_aggregates']=document['aggregates']
        profile['material_binding']=(document.get('contact_reference') or {}).get('surface',{}).get('material_binding')
    with (args.output/'trace.jsonl').open('w') as trace:
        def progress(value):
            trace.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');trace.flush()
            print(json.dumps({k:v for k,v in value.items() if k!='training_sample'},ensure_ascii=False),flush=True)
        if args.method=='moxian_wuji':
            from moxian_web.moxian_six_calibration import fit_capture as fit_moxian_six
            candidate,report=fit_moxian_six(profile,document['mapping'],document,progress,threading.Event())
        elif args.method=='wuji_native':
            from moxian_web.wuji_native_calibration import fit_native_capture
            candidate,report=fit_native_capture(profile,document['mapping'],document,progress,threading.Event())
        else:
            candidate,report=fit_capture(None,profile,rows[0]['mapping'],rows,progress,threading.Event(),method=args.method,compile_loss=args.compile,
            training_options=dict(adam_steps=args.steps,max_seconds=args.seconds,batch_size=args.batch_size,optimizer=args.optimizer),initial_candidate=json.loads(args.resume.read_text()) if args.resume else None,
            **({'solver_options':{'maxeval':args.maxeval}} if args.method=='bone_extrinsics' else {}))
    for name,value in [('candidate.json',candidate),('report.json',report)]:
        (args.output/name).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')))
    print(json.dumps(dict(accepted=report['accepted'],steps=report.get('steps',report.get('solver',{}).get('evaluations')),seconds=report['end_to_end_s'],reasons=report.get('rejection_reasons',[])),ensure_ascii=False))


if __name__=='__main__':main()
