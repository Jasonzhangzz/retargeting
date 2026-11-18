"""CART protocol capture controls, preprocessing, training and evaluation."""
import argparse,json,signal,sys,time,uuid
from pathlib import Path
from retarget.baseline.geort.adapter import ROOT

def preview():
    import contextlib,queue,threading
    import torch
    from .cart_data import SourceGeometry
    from .cart_contacts import transform_contacts,overlay
    init=json.loads(sys.stdin.readline())
    torch.set_num_threads(1)
    with contextlib.redirect_stdout(sys.stderr):
        source=SourceGeometry(init['topology'],init['profile'],'cuda:0' if torch.cuda.is_available() else 'cpu')
    pending=queue.Queue(maxsize=1)
    def read():
        for line in sys.stdin:
            item=json.loads(line)
            try:pending.get_nowait()
            except queue.Empty:pass
            pending.put(item)
        pending.put(None)
    threading.Thread(target=read,daemon=True).start()
    while True:
        item=pending.get()
        if item is None:return
        try:
            with contextlib.redirect_stdout(sys.stderr):
                row,v,j=source.frame(item['qpos'],item['timestamp_ms'],item)
            root=j[source.adapter.indices[0]]
            contacts=transform_contacts(row['contacts'],source.adapter.transform,root,inverse=True)
            hull=overlay(contacts,timestamp_ms=item['timestamp_ms'],geometry_version=init['topology']['geometry_version'],
                         processing_ms=row['processing_ms'],coordinate_frame='wrist_local')
            frame=dict(ready=True,vertices_m=v.tolist(),joints_m=j.tolist(),timestamp_ms=item['timestamp_ms'],
                geometry_version=init['topology']['geometry_version'],binding_version=init['topology']['binding_version'])
            print(json.dumps(dict(generation=item.get('generation'),hull=hull,preview_frame=frame),allow_nan=False),flush=True)
        except Exception as exc:print(json.dumps(dict(error=str(exc))),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT/'output/moxian/cart')
    sub=parser.add_subparsers(dest='action',required=True);sub.add_parser('preview')
    for name in ('preprocess','train','evaluate'):
        p=sub.add_parser(name);p.add_argument('--datasets',nargs='+',required=True);p.add_argument('--job-id')
        if name=='train':
            p.add_argument('--epochs',type=int,default=200);p.add_argument('--smoke-test',action='store_true')
            p.add_argument('--architecture',choices=('attention','fsq'),default='attention')
            p.add_argument('--batch-size',type=int,default=0,help='Simultaneous windows per optimizer step; 0 selects from free memory')
            p.add_argument('--micro-batch-size',type=int,default=0,help='Deprecated alias for a simultaneous-window cap; never gradient accumulation')
            p.add_argument('--attention-precision',choices=('auto','float32','bfloat16'),default='auto',help='Only CUDA attention uses BF16; geometry and output stay FP32')
            p.add_argument('--validate-every',type=int,default=10,help='Periodic true-mesh selection audit; first and final epochs are always checked')
            p.add_argument('--validation-windows',type=int,default=32,help='Fixed true-mesh selection window budget; 0 means all. Final full-session acceptance is never sampled.')
            p.add_argument('--length',type=int,default=16)
            p.add_argument('--resume-run');p.add_argument('--sdf-warmup-steps',type=int)
            p.add_argument('--initialize-from',help='Initialize a new run from a compatible v5 checkpoint; not exact resume')
            p.add_argument('--initial-noise-rad',type=float,default=.1,help='Uniform IK reset noise in [-amplitude, amplitude]; 0 disables')
        if name=='evaluate':p.add_argument('--model',required=True)
    cap=sub.add_parser('capture');cap.add_argument('--dataset',required=True);cap.add_argument('--follow',action='store_true');cap.add_argument('--device')
    imp=sub.add_parser('import-emf');imp.add_argument('--dataset',required=True);imp.add_argument('--raw',type=Path,required=True)
    ctrl=sub.add_parser('control');ctrl.add_argument('command');ctrl.add_argument('--data',default='{}');ctrl.add_argument('--url',default='http://127.0.0.1:8890')
    args=parser.parse_args()
    if args.action=='preview':return preview()
    if args.action=='capture':
        from .cart_capture import follow,capture_recording
        result=follow(args.root,args.dataset,args.device) if args.follow else capture_recording(args.root,args.dataset,device=args.device)
        print(json.dumps(result,ensure_ascii=False));return
    if args.action=='import-emf':
        from .cart_data import import_emf
        from .saved_assets import asset_lock
        with asset_lock(args.root):print(json.dumps(import_emf(args.root,args.dataset,args.raw),ensure_ascii=False))
        return
    if args.action=='control':
        import urllib.request
        request=urllib.request.Request(args.url+'/api/command',data=json.dumps(dict(command=args.command,**json.loads(args.data))).encode(),headers={'Content-Type':'application/json'})
        print(urllib.request.urlopen(request).read().decode());return
    from .saved_assets import asset_lock
    with asset_lock(args.root):return run(args)

def run(args):
    from .geort_service import safe_id
    from retarget.baseline.geort.training import atomic_json
    key=safe_id(args.job_id or time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'_'+uuid.uuid4().hex[:8])
    folder=args.root/'jobs'/key;folder.mkdir(parents=True,exist_ok=True)
    cancelled=False
    def cancel(*_):
        nonlocal cancelled
        cancelled=True
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    def check():
        if cancelled:raise InterruptedError('CART 任务已取消')
    def progress(phase,**values):
        import os
        check();atomic_json(folder/'status.json',dict(id=key,status='running',pid=os.getpid(),action=args.action,phase=phase,**values))
    try:
        progress('starting')
        if args.action=='preprocess':
            from .cart_data import preprocess
            from .cart_initialization import prepare_ik
            for dataset in args.datasets:
                check();preprocess(args.root,safe_id(dataset),progress=progress,check=check)
                prepare_ik(args.root,safe_id(dataset),check=check,progress=progress)
            result=dict(processed=args.datasets)
        elif args.action=='train':
            from .cart_data import manifest
            from .cart_training import train
            ids={s:[] for s in ('train','validation')}
            for dataset in args.datasets:
                _,m=manifest(args.root,dataset);ids[m['split']].append(dataset)
            result=dict(model=str(train(args.root,ids['train'],ids['validation'],safe_id(args.resume_run) if args.resume_run else key,args.epochs,smoke=args.smoke_test,
                architecture=args.architecture,batch_size=args.batch_size,micro_batch_size=args.micro_batch_size,length=args.length,resume=bool(args.resume_run),warmup_steps=args.sdf_warmup_steps,initialize_from=args.initialize_from,
                initial_noise_rad=args.initial_noise_rad,attention_precision=args.attention_precision,
                validation_interval=args.validate_every,validation_window_budget=args.validation_windows)))
        else:
            from .cart_training import evaluate
            result=evaluate(args.root,safe_id(args.model),args.datasets,check=check)
        atomic_json(folder/'status.json',dict(id=key,status='completed',action=args.action,result=result))
    except BaseException as exc:
        atomic_json(folder/'status.json',dict(id=key,status='cancelled' if isinstance(exc,InterruptedError) else 'failed',action=args.action,error=str(exc)));raise

if __name__=='__main__':main()
