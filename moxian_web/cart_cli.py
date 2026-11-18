"""CART protocol capture controls, preprocessing, training and evaluation."""
import argparse,json,signal,sys,time,uuid
from pathlib import Path
from retarget.baseline.geort.adapter import ROOT

def preview():
    import queue,threading
    import numpy as np
    from .cart_data import SourceGeometry
    from .cart_protocol import MotionTracker
    from .cart_geometry import transform_hull
    init=json.loads(sys.stdin.readline());source=SourceGeometry(init['topology'],init['profile'])
    pending=queue.Queue(maxsize=1)
    def read():
        for line in sys.stdin:
            item=json.loads(line)
            try:pending.get_nowait()
            except queue.Empty:pass
            pending.put(item)
        pending.put(None)
    threading.Thread(target=read,daemon=True).start();tracker=MotionTracker();last=None
    while True:
        item=pending.get()
        if item is None:return
        try:
            identity=(item['stage'],item.get('generation'))
            if identity!=last:tracker.reset();last=identity
            row,v,j=source.frame(item['qpos'],item['stage'],item['timestamp_ms'],tracker)
            root=j[source.adapter.indices[0]];hull=transform_hull(row['target'],source.adapter.transform,root)
            if row['target'].get('ready'):
                # Row-vector canonical->wrist uses transform.T.
                hull=transform_hull(row['target'],source.adapter.transform,root)
            hull['coordinate_frame']='wrist_local';hull['timestamp_ms']=item['timestamp_ms']
            frame=dict(ready=True,vertices_m=v.tolist(),joints_m=j.tolist(),cart_hull=hull,
                timestamp_ms=item['timestamp_ms'],geometry_version=init['topology']['geometry_version'],
                binding_version=init['topology']['binding_version'])
            hull['contact_points_m']=[p for pair in row['target'].get('pairs',[]) for p in
                (np.asarray(pair['points_m'])@source.adapter.transform.T+root).tolist()]
            hull['contact_points_m'] += [p for c in hull.get('contacts',[]) for p in (c['point_m'],c['object_point_m'])]
            print(json.dumps(dict(generation=item.get('generation'),hull=hull,motion=dict(phase=row['phase'],cycles=row['cycles']),preview_frame=frame),allow_nan=False),flush=True)
        except Exception as exc:print(json.dumps(dict(error=str(exc))),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT/'output/moxian/cart')
    sub=parser.add_subparsers(dest='action',required=True);sub.add_parser('preview')
    for name in ('preprocess','train','evaluate'):
        p=sub.add_parser(name);p.add_argument('--datasets',nargs='+',required=True);p.add_argument('--job-id')
        if name=='train':p.add_argument('--epochs',type=int,default=200);p.add_argument('--smoke-test',action='store_true')
        if name=='evaluate':p.add_argument('--model',required=True)
    ctrl=sub.add_parser('control');ctrl.add_argument('command');ctrl.add_argument('--data',default='{}');ctrl.add_argument('--url',default='http://127.0.0.1:8890')
    args=parser.parse_args()
    if args.action=='preview':return preview()
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
            for dataset in args.datasets:check();preprocess(args.root,safe_id(dataset),progress=progress,check=check)
            result=dict(processed=args.datasets)
        elif args.action=='train':
            from .cart_data import manifest
            from .cart_training import train
            ids={s:[] for s in ('train','validation')}
            for dataset in args.datasets:
                _,m=manifest(args.root,dataset);ids[m['split']].append(dataset)
            result=dict(model=str(train(args.root,ids['train'],ids['validation'],key,args.epochs,smoke=args.smoke_test)))
        else:
            from .cart_training import evaluate
            result=evaluate(args.root,safe_id(args.model),args.datasets,check=check)
        atomic_json(folder/'status.json',dict(id=key,status='completed',action=args.action,result=result))
    except BaseException as exc:
        atomic_json(folder/'status.json',dict(id=key,status='cancelled' if isinstance(exc,InterruptedError) else 'failed',action=args.action,error=str(exc)));raise

if __name__=='__main__':main()
