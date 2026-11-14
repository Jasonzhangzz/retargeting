#!/usr/bin/env python3
"""Single-owner four-GPU sequential training queue with frozen source and receipts."""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

VARIANTS=('temporal_causal','temporal_bidirectional','frame_regression','frame_flow')


def stamp():return datetime.now(timezone(timedelta(hours=8),'Asia/Hong_Kong')).isoformat()


def write(path,value):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,path)


def free_memory(indices):
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    values={int(line.split(',')[0]):int(line.split(',')[1]) for line in output.splitlines() if line.strip()}
    return {i:values[i] for i in indices}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',required=True);p.add_argument('--output',required=True)
    p.add_argument('--python',default=sys.executable);p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--frames-per-gpu',type=int,default=8192);p.add_argument('--memory-limit-gib',type=float,default=12)
    p.add_argument('--gpus',default='0,1,2,3');p.add_argument('--workers',type=int,default=2)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--temporal-smoothness-weight',type=float,default=0.,help='Normalized acceleration penalty for both temporal variants')
    p.add_argument('--initialize-causal-from',help='Weight-only causal initialization for a new campaign')
    p.add_argument('--initialize-causal-optimizer',action='store_true',help='Retain moments from the causal initializer; reset its LR schedule to the explicit fine-tuning rates')
    p.add_argument('--causal-muon-lr',type=float,default=.02)
    p.add_argument('--causal-adamw-lr',type=float,default=.0003)
    p.add_argument('--causal-warmup-fraction',type=float,default=.02)
    p.add_argument('--runtime-snapshot',help='Optional local-disk source copy; all Python hashes must match the archived source snapshot')
    p.add_argument('--continue-existing',action='store_true',help='Continue a stopped campaign from its checkpoints, optionally extending epochs')
    args=p.parse_args()
    if args.initialize_causal_optimizer and not args.initialize_causal_from:
        raise ValueError('--initialize-causal-optimizer requires --initialize-causal-from')
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    repo=Path(__file__).resolve().parents[1];indices=[int(v) for v in args.gpus.split(',')]
    if len(indices)!=4 or len(set(indices))!=4:raise ValueError('This campaign requires four distinct GPUs')
    global_lock=repo/'results/wuji_ik_training_queue.lock';global_lock.parent.mkdir(exist_ok=True)
    lock=global_lock.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=None
    if (root/'STATUS.json').exists():
        if not args.continue_existing:raise RuntimeError('Campaign already exists; use --continue-existing after stopping its controller')
        previous=json.loads((root/'configuration.json').read_text())
        previous_state=json.loads((root/'STATUS.json').read_text())
        for job in previous_state['jobs'].values():
            cmdline=Path('/proc')/str(job.get('pid',0))/'cmdline'
            if cmdline.exists() and str(root).encode() in cmdline.read_bytes():
                raise RuntimeError('Previous training process is still alive; stop it before continuation')
        if (root/'STOP').exists():raise RuntimeError('Remove STOP explicitly before continuing')
        if args.epochs < previous['epochs']:raise ValueError('Cannot reduce campaign epochs')
        if (previous['gpus']!=indices or previous['effective_frames_per_gpu']!=args.frames_per_gpu
                or previous['memory_limit_gib_per_rank']!=args.memory_limit_gib
                or previous.get('temporal_smoothness_weight',0.)!=args.temporal_smoothness_weight):
            raise ValueError('Continuation must preserve GPUs, frame budget and memory limit')
        for key,default in [('causal_muon_lr',.02),('causal_adamw_lr',.0003),('causal_warmup_fraction',.02)]:
            if previous.get(key,default)!=getattr(args,key):
                raise ValueError('Exact continuation must preserve causal learning-rate configuration')
        archive=root/'revisions'/str(time.time_ns());archive.mkdir(parents=True)
        for name in ('STATUS.json','configuration.json'):shutil.copy2(root/name,archive/name)
    metadata=json.loads((Path(args.prepared)/'manifest.json').read_text())
    if not metadata.get('complete') or metadata.get('validation_fraction')!=.05:raise ValueError('Expected the verified 5% validation split')
    if previous and previous['data_signature']!=metadata['signature']:raise ValueError('Continuation data signature mismatch')
    snapshot=root/'source_snapshot'
    if previous and snapshot.exists():shutil.move(str(snapshot),str(archive/'source_snapshot'))
    package=snapshot/'retarget/learning';package.mkdir(parents=True)
    (snapshot/'retarget/__init__.py').write_text('')
    (package/'__init__.py').write_text('')
    shutil.copytree(repo/'retarget/learning/wuji_ik',package/'wuji_ik',ignore=shutil.ignore_patterns('__pycache__'))
    hashes={str(f.relative_to(snapshot)):hashlib.sha256(f.read_bytes()).hexdigest() for f in snapshot.rglob('*.py')}
    runtime=Path(args.runtime_snapshot).resolve() if args.runtime_snapshot else snapshot
    if runtime != snapshot:
        runtime_hashes={str(f.relative_to(runtime)):hashlib.sha256(f.read_bytes()).hexdigest() for f in runtime.rglob('*.py')}
        if runtime_hashes != hashes:
            raise ValueError('Local runtime source hashes differ from the archived source snapshot')
    configuration=dict(created=stamp(),prepared=str(Path(args.prepared).resolve()),data_signature=metadata['signature'],
        split_rows=metadata['split_rows'],epochs=args.epochs,variants=list(VARIANTS),gpus=indices,
        temporal_smoothness_weight=args.temporal_smoothness_weight,
        initialize_causal_from=args.initialize_causal_from,
        initialize_causal_optimizer=args.initialize_causal_optimizer,
        causal_muon_lr=args.causal_muon_lr,causal_adamw_lr=args.causal_adamw_lr,
        causal_warmup_fraction=args.causal_warmup_fraction,
        runtime_snapshot=str(runtime),
        effective_frames_per_gpu=args.frames_per_gpu,global_effective_frames=args.frames_per_gpu*4,
        microbatch='one complete trajectory; disjoint loss ranges preserve exact frame count',
        epoch_tail='all remaining frames exactly once; may be less than 8192 on a rank; aligned zero-loss dummy ranks',
        memory_limit_gib_per_rank=args.memory_limit_gib,source_sha256=hashes,source_snapshot=str(snapshot))
    if previous:configuration.update(previous_epochs=previous['epochs'],continued=stamp(),previous_revision=str(archive))
    write(root/'configuration.json',configuration)
    state=dict(pid=os.getpid(),status='prepared',created=stamp(),active=None,
               jobs={v:dict(status='queued') for v in VARIANTS})
    def status(name,**fields):
        state.update(status=name,updated=stamp(),**fields);write(root/'STATUS.json',state)
    stop=False;child=None
    def request_stop(*_):
        nonlocal stop
        stop=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,request_stop)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=args.gpus,PYTHONPATH=str(runtime),
        PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        TORCHINDUCTOR_COMPILE_THREADS='2')
    try:
        for variant in VARIANTS:
            if stop or (root/'STOP').exists():raise InterruptedError('Queue stop requested')
            while True:
                free=free_memory(indices)
                if min(free.values()) >= args.memory_limit_gib*1024+2048:break
                status('waiting_for_memory',free_mib=free)
                time.sleep(30)
                if stop or (root/'STOP').exists():raise InterruptedError('Queue stop requested')
            output=root/variant
            result_path=output/'status.json'
            result=json.loads(result_path.read_text()) if result_path.exists() else {}
            configuration_path=output/'configuration.json'
            old_training=json.loads(configuration_path.read_text())['train'] if configuration_path.exists() else None
            if result.get('complete') and old_training and old_training['epochs']==args.epochs:
                state['jobs'][variant].update(status='complete',result=result);continue
            argv=[args.python,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=4','--max-restarts=0',
                '-m','retarget.learning.wuji_ik','train','--prepared',str(Path(args.prepared).resolve()),
                '--output',str(output),'--variant',variant,'--epochs',str(args.epochs),'--batch-size','1',
                '--effective-frames-per-gpu',str(args.frames_per_gpu),'--accumulation','1',
                '--workers',str(args.workers),'--threads','2','--checkpoint-every','10','--seed',str(args.seed),
                '--gpu-memory-limit-gib',str(args.memory_limit_gib),'--checkpoint-blocks','--backend','auto']
            if (output/'latest.pt').exists():
                if not previous:raise RuntimeError('Unexpected existing checkpoint')
                argv+=['--resume',str(output/'latest.pt'),'--extend-epochs']
            elif variant=='temporal_causal' and args.initialize_causal_from:
                argv+=['--initialize-from',args.initialize_causal_from]
                if args.initialize_causal_optimizer:argv+=['--initialize-optimizer']
            if variant=='temporal_causal':
                argv+=['--muon-lr',str(args.causal_muon_lr),'--adamw-lr',str(args.causal_adamw_lr),
                       '--warmup-fraction',str(args.causal_warmup_fraction)]
            if variant in ('temporal_causal','temporal_bidirectional'):
                argv+=['--smoothness-weight',str(args.temporal_smoothness_weight)]
            state['jobs'][variant].update(status='starting',argv=argv,started=stamp(),log=str(root/(variant+'.log')))
            with (root/(variant+'.log')).open('a') as log:
                child=subprocess.Popen(argv,cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state['jobs'][variant].update(status='running',pid=child.pid)
                status('training',active=variant,free_mib=free)
                print('START',variant,'pid',child.pid,flush=True)
                next_report=0
                while child.poll() is None:
                    if stop or (root/'STOP').exists():
                        os.killpg(child.pid,signal.SIGTERM);child.wait();raise InterruptedError('Queue stop requested')
                    if time.monotonic()>=next_report:
                        progress=output/'train.jsonl'
                        if progress.exists():
                            lines=progress.read_text().splitlines()
                            if lines:state['jobs'][variant]['last_update']=json.loads(lines[-1])
                        status('training',active=variant);next_report=time.monotonic()+30
                    time.sleep(5)
            code=child.returncode
            result=json.loads((output/'status.json').read_text()) if (output/'status.json').exists() else {}
            state['jobs'][variant].update(exit_code=code,finished=stamp(),result=result)
            if code or not result.get('complete'):
                state['jobs'][variant]['status']='failed';status('failed',active=None)
                raise RuntimeError(f'{variant} failed; subsequent tasks remain queued, no concurrent/retry launch')
            state['jobs'][variant]['status']='complete';status('between_jobs',active=None)
            print('COMPLETE',variant,flush=True)
        status('complete',active=None)
    except InterruptedError as exc:
        status('stopped',active=None,reason=str(exc))
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM);child.wait()
        status('failed',active=None,reason=str(exc));raise


if __name__=='__main__':main()
