"""CLI for the same GeoRT training/evaluation jobs exposed by the WebUI."""
import argparse
import json
import time
import uuid
from pathlib import Path
from retarget.baseline.geort.adapter import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'output/moxian/geort')
    sub=p.add_subparsers(dest='action',required=True)
    train=sub.add_parser('train');train.add_argument('--datasets',nargs='+',required=True)
    train.add_argument('--run-id',default=None);train.add_argument('--epochs',type=int,default=200)
    train.add_argument('--fk-epochs',type=int,default=200);train.add_argument('--seed',type=int,default=714)
    train.add_argument('--mode',choices=['official','contact_v1'],default='official')
    train.add_argument('--smoke-test',action='store_true',help='明确标记为测试；缩小样本量，不作为完整复现结果')
    ev=sub.add_parser('evaluate');ev.add_argument('--model',required=True);ev.add_argument('--datasets',nargs='+',required=True)
    imp=sub.add_parser('import-raw');imp.add_argument('--recording',type=Path,required=True)
    imp.add_argument('--profile',type=Path,required=True);imp.add_argument('--id',required=True)
    ctrl=sub.add_parser('control');ctrl.add_argument('command');ctrl.add_argument('--data',default='{}')
    ctrl.add_argument('--url',default='http://127.0.0.1:8890')
    workspace=sub.add_parser('workspace');workspace.add_argument('--folder',type=Path,required=True)
    args=p.parse_args()
    if args.action=='workspace':
        from retarget.baseline.geort.workspace import build
        build(args.folder);return
    if args.action=='control':
        import urllib.request
        request=urllib.request.Request(args.url+'/api/command',data=json.dumps(dict(command=args.command,**json.loads(args.data))).encode(),headers={'Content-Type':'application/json'})
        print(urllib.request.urlopen(request).read().decode());return
    from moxian_web.saved_assets import asset_lock
    with asset_lock(args.root):return run(args,p)


def run(args,p):
    from retarget.baseline.geort.training import train as run_train,evaluate
    if args.action=='import-raw':
        from retarget.baseline.geort.import_data import import_raw
        print(json.dumps(import_raw(args.root,args.recording,args.profile,args.id),ensure_ascii=False))
    elif args.action=='train':
        if not 1<=args.epochs<=2000 or not 1<=args.fk_epochs<=2000:p.error('轮数必须为1–2000')
        from moxian_web.geort_service import safe_id
        key=safe_id(args.run_id or time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'_'+uuid.uuid4().hex[:8])
        extra=dict(robot_samples=2048,resample_to=4096,batch_size=128) if args.smoke_test else {}
        run_train(args.root,args.datasets,key,args.epochs,args.fk_epochs,args.seed,mode=args.mode,**extra)
    else:
        from moxian_web.geort_service import safe_id
        from retarget.baseline.geort.training import atomic_json
        folder=args.root/'models'/safe_id(args.model)
        atomic_json(folder/'evaluation_status.json',{'status':'running'})
        try:
            result=evaluate(args.root,args.model,args.datasets)
            atomic_json(folder/'evaluation_status.json',{'status':'completed','result':result})
            print(json.dumps(result,ensure_ascii=False,indent=2))
        except BaseException as exc:
            atomic_json(folder/'evaluation_status.json',{'status':'failed','error':str(exc)});raise


if __name__=='__main__':main()
