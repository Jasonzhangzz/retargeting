"""Backfill and follow epoch metrics in W&B without restarting GPU training."""
import argparse
import fcntl
import json
from pathlib import Path
import signal
import time
import uuid

from .data import write_json

STOP=False
TERMINAL={'completed','completed_targets_unmet','failed','paused'}
METRICS={
    'loss':'train/total_loss','reconstruction':'train/reconstruction_mse',
    'cross_entropy':'train/classification_ce','commitment':'train/commitment',
    'lr':'train/learning_rate','val_accuracy':'val/classification_accuracy',
    'val_macro_f1':'val/classification_macro_f1','val_cross_entropy':'val/classification_ce',
    'val_mae_deg':'val/reconstruction_mae_deg','val_rmse_deg':'val/reconstruction_rmse_deg',
    'val_p95_deg':'val/reconstruction_p95_deg','val_normalized_mse':'val/reconstruction_mse',
    'val_joint_score':'val/joint_loss','best_val_joint_score':'val/best_joint_loss',
    'seconds':'performance/seconds_per_epoch','gpu_peak_allocated_mib':'performance/gpu_peak_allocated_mib',
    'classification_weight':'loss_weights/classification','reconstruction_weight':'loss_weights/reconstruction',
}


def read_json(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


def records(output,configuration):
    """Order warmup and RVQ records on a single monotonic epoch axis."""
    rows=[]
    for phase,offset in [('warmup',0),('rvq',configuration['warmup_epochs'])]:
        for record in read_json(Path(output)/phase/'history.json',[]):
            epoch=int(record['epoch']);overall=offset+epoch
            values={'epoch':overall,'phase_epoch':epoch,'phase':phase,
                    'optimizer_step':overall*configuration['steps_per_epoch']}
            for key,target in METRICS.items():
                if key in record:values[target]=record[key]
            rows.append((overall,values))
    return sorted(rows,key=lambda row:row[0])


def scalar_summary(report,prefix='test'):
    return {prefix+'/'+key:value for key,value in report.items()
            if isinstance(value,(int,float,bool)) and key not in {'samples'}}


def sync(args):
    import wandb
    global STOP
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    # Exactly one live writer for this training run.
    lock=(output/'wandb_sync.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('A W&B synchronizer already owns this output directory')
    def stop(*_):
        global STOP
        STOP=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    configuration=read_json(output/'configuration.json')
    if configuration is None:raise ValueError('Training configuration.json is required')
    if configuration.get('task')!='joint':raise ValueError('This synchronizer expects the joint warmup/RVQ campaign')
    status_path=output/'wandb_sync_status.json'
    api=None
    while not STOP:
        try:
            api=wandb.Api(overrides={'base_url':args.base_url},timeout=15)
            entity=args.entity or api.default_entity
            if not entity:raise RuntimeError('W&B account has no default entity; specify --entity')
            break
        except Exception as exc:
            error_type=type(exc).__name__
            write_json(status_path,dict(state='waiting_auth',base_url=args.base_url,project=args.project,error_type=error_type))
            if not args.wait_for_login:raise
            print(json.dumps(dict(state='waiting_auth',error_type=error_type)),flush=True)
            time.sleep(min(args.poll_seconds,15))
    if STOP:return
    descriptor_path=output/'wandb_run.json'
    descriptor=read_json(descriptor_path,{})
    target=dict(base_url=args.base_url.rstrip('/'),entity=entity,project=args.project)
    if descriptor and any(descriptor.get(key)!=value for key,value in target.items()):
        raise ValueError('W&B target differs from existing wandb_run.json; use the recorded entity/project')
    run_id=descriptor.get('id') or uuid.uuid4().hex[:12]
    write_json(descriptor_path,dict(**target,id=run_id,name=args.name or output.name))
    settings=wandb.Settings(base_url=args.base_url,init_timeout=45,disable_git=True,save_code=False,console='off')
    run=wandb.init(project=args.project,entity=entity,id=run_id,resume='allow',name=args.name or output.name,
        config=configuration,dir=str(output),settings=settings,mode='online',save_code=False,
        tags=['wuji2','rvq','joint','muon','4gpu'])
    write_json(descriptor_path,dict(**target,id=run.id,name=run.name,url=run.url))
    run.define_metric('epoch')
    for pattern in ['train/*','val/*','performance/*','loss_weights/*']:
        run.define_metric(pattern,step_metric='epoch')
    # Server-resumed step is authoritative; replay locally buffered epochs if needed.
    next_step=run.step;logged=0;last_epoch=max(0,next_step-1)
    print(json.dumps(dict(state='syncing',url=run.url,resume_step=next_step)),flush=True)
    stable_terminal=None
    try:
        while not STOP:
            for epoch,values in records(output,configuration):
                if epoch<next_step:continue
                run.log(values,step=epoch)
                next_step=epoch+1;last_epoch=epoch;logged+=1
            training=read_json(output/'status.json',{})
            write_json(status_path,dict(state='syncing',url=run.url,last_epoch=last_epoch,uploaded_records_this_session=logged,
                training_state=training.get('state')))
            terminal=training.get('state') in TERMINAL
            # Reporting plots are generated immediately after terminal status is written.
            report=read_json(output/'report.json')
            if terminal:
                if training['state'].startswith('completed'):
                    expected=[output/name for name in ['REPORT.md','report.json','training_curves.png','confusion_matrix.png']]
                    fingerprint=tuple((p.stat().st_mtime_ns,p.stat().st_size) if p.exists() else None for p in expected)
                    if None in fingerprint or fingerprint!=stable_terminal:
                        stable_terminal=fingerprint;time.sleep(args.poll_seconds);continue
                if report:
                    run.summary.update(scalar_summary(report))
                    for level,book in enumerate(report.get('codebooks',[])):
                        for key in ['used','utilization','perplexity']:
                            if key in book:run.summary[f'test/codebook_{level}/{key}']=book[key]
                    for key,value in report.get('fingertips',{}).items():
                        if isinstance(value,(int,float)):run.summary['test/fingertips/'+key]=value
                run.summary['training_state']=training['state']
                # Only metric reports/plots are uploaded, no pose data or model checkpoints.
                for filename in ['report.json','validation.json','continuous_test.json','continuous_validation.json','REPORT.md']:
                    path=output/filename
                    if path.exists():run.save(str(path),base_path=str(output),policy='now')
                for filename in ['training_curves.png','confusion_matrix.png']:
                    path=output/filename
                    if path.exists():run.log({'plots/'+path.stem:wandb.Image(str(path))})
                break
            if args.once:break
            time.sleep(args.poll_seconds)
        run.finish()
        write_json(status_path,dict(state='synced' if not STOP else 'stopped',url=run.url,last_epoch=last_epoch,
            uploaded_records_this_session=logged,training_state=read_json(output/'status.json',{}).get('state')))
    except BaseException:
        run.finish(exit_code=1)
        write_json(status_path,dict(state='error',url=run.url,last_epoch=last_epoch))
        raise
    finally:lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base-url',default='https://wandb.plat.x2robot.com')
    p.add_argument('--project',default='wuji2-rvq')
    p.add_argument('--entity');p.add_argument('--name')
    p.add_argument('--poll-seconds',type=float,default=5.)
    p.add_argument('--wait-for-login',action='store_true');p.add_argument('--once',action='store_true')
    args=p.parse_args()
    if args.poll_seconds<=0:p.error('Polling interval must be positive')
    sync(args)


if __name__=='__main__':main()
