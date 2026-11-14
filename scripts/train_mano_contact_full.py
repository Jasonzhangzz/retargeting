"""One fixed two-stage MANO run, with GPU admission, receipts and graceful stop.

Run with --manifest RUN/MANIFEST.json. A RUN/STOP file prevents new jobs and
asks the active trainer to commit its complete state. No automatic tuning or
failed-job retries are performed. This controller sends no external messages.
"""
import argparse
import csv
from datetime import datetime, timezone
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
import traceback


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    temporary.replace(path)


def sha(path):
    result=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):result.update(block)
    return result.hexdigest()


def process_alive(pid):
    try:return bool(Path(f'/proc/{pid}/cmdline').read_bytes())
    except FileNotFoundError:return False


def gpu_inventory():
    def query(kind, fields):
        output=subprocess.check_output(['nvidia-smi','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],text=True)
        return [[x.strip() for x in row] for row in csv.reader(output.splitlines()) if row]
    devices=query('gpu','index,uuid,memory.total,memory.used,memory.free,utilization.gpu')
    processes=query('compute-apps','gpu_uuid,pid,process_name,used_memory')
    return [dict(index=int(i),uuid=u,total_mib=int(t),used_mib=int(m),free_mib=int(f),utilization=int(v),
                 processes=[dict(pid=int(p[1]),command=p[2],memory_mib=p[3]) for p in processes if p[0]==u])
            for i,u,t,m,f,v in devices]


def eligible(gpu, policy, minimum_free_mib=12288):
    if policy=='idle_only':
        return not gpu['processes'] and gpu['used_mib']<=1024 and gpu['utilization']<=5
    if policy=='shared':return gpu['free_mib']>=minimum_free_mib
    raise ValueError('Unknown device policy: '+policy)


def preflight(manifest, device):
    """Actual corpus T=1/8 forward/backward; exactly ZERO optimizer updates."""
    import h5py
    import numpy as np
    import torch
    from retarget.learning.mano_contact.model import PoseAutoencoder, ContactModel
    from retarget.learning.mano_contact.train import stage1_loss, stage2_forward
    from retarget.learning.mano_contact.data import collate_windows, train_statistics
    from retarget.learning.mano_contact.losses import HACOLoss
    torch.set_num_threads(4);torch.manual_seed(42)
    if device.startswith('cuda'):
        total=torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(manifest['gpu_memory_limit_mib']*1024**2/total,0)
    examples={}
    with h5py.File(manifest['scratch_corpus'],'r') as f:
        assert f.attrs['preparation_state']=='complete'
        assert f.attrs['corpus_fingerprint']==manifest['corpus_fingerprint']
        for key in sorted(f['records']):
            group=f['records'][key]
            if group.attrs['split']!='train':continue
            temporal=group.attrs['kind']=='temporal'
            frames=group['frame_id'][:];known=group['contact_valid'][:];labels=group['contact'][:].any(-1)
            for end in np.flatnonzero(known):
                start=end-7 if temporal else end
                if start<0 or (temporal and np.any(np.diff(frames[start:end+1])!=1)):continue
                identity=(temporal,bool(labels[end]))
                if identity in examples:continue
                examples[identity]=dict(pose51=torch.from_numpy(group['pose51'][start:end+1]),
                    contact=torch.from_numpy(group['contact'][end].astype(np.float32)),valid=torch.tensor(True),
                    frame=torch.tensor(bool(labels[end])),weight=torch.tensor(1.),dataset=group.attrs['dataset'],row_id=f'{key}:{end}')
            if (False,True) in examples and (True,True) in examples and (True,False) in examples:break
    assert (False,True) in examples and (True,True) in examples and (True,False) in examples
    rows=[examples[(True,False)],examples[(True,True)],examples[(False,True)],examples[(True,True)]]
    rows=(rows*((manifest['batch_size']+3)//4))[:manifest['batch_size']] if device.startswith('cuda') else rows
    groups=[{k:v.to(device) if torch.is_tensor(v) else v for k,v in g.items()} for g in collate_windows(rows)]
    ae=PoseAutoencoder().to(device);loss1=stage1_loss(ae,groups,noise=False);loss1['total'].backward()
    assert torch.isfinite(loss1['total'])
    model=ContactModel.from_autoencoder(ae,manifest['assets']).to(device)
    loss_fn=HACOLoss(model.decoder,train_statistics(manifest['scratch_corpus'])).configure_faces(np.load(Path(manifest['assets'])/'faces.npy')).to(device)
    prediction,target,valid=stage2_forward(model,groups);loss2=loss_fn(prediction,target,valid);loss2['total'].backward()
    assert torch.isfinite(loss2['total'])
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.encoder.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    receipt=dict(device=device,optimizer_updates=0,batch_size=len(rows),input_lengths=[1,8],
                 stage1_loss=float(loss1['total'].detach()),stage2_loss=float(loss2['total'].detach()),
                 encoder_gradients_finite=True,completed_utc=stamp())
    if device.startswith('cuda'):
        torch.cuda.synchronize();receipt['peak_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2
    write(Path(manifest['output'])/('preflight_'+device.replace(':','_')+'.json'),receipt)
    print(json.dumps(receipt),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--mode',choices=['controller','preflight','train'],default='controller')
    parser.add_argument('--device',default='cpu')
    args,remaining=parser.parse_known_args()
    manifest=json.loads(Path(args.manifest).read_text());root=Path(manifest['output'])
    if args.mode=='preflight':return preflight(manifest,args.device)
    if args.mode=='train':
        import torch
        total=torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(manifest['gpu_memory_limit_mib']*1024**2/total,0)
        from retarget.learning.mano_contact.cli import main as cli_main
        sys.argv=['mano_contact']+remaining+['--device',args.device]
        return cli_main()
    if remaining:parser.error('Unexpected arguments')
    lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'STATUS.json').exists():raise RuntimeError('Existing controller state: inspect receipts before an explicit continuation')
    stopped=False;child=None;active_job=None
    state=dict(campaign=manifest['campaign'],control_revision=1,pid=os.getpid(),process_start_ticks=Path('/proc/self/stat').read_text().split()[21],
               stage='prepared',jobs={},created_utc=stamp(),actual_optimizer_updates=0)
    def status(stage,**fields):
        state.update(stage=stage,updated_utc=stamp(),**fields);write(root/'STATUS.json',state)
    def stop(*_):
        nonlocal stopped
        stopped=True;(root/'STOP').touch()
    def check_stop():
        if stopped or (root/'STOP').exists():raise InterruptedError('Campaign stop requested')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    environment=os.environ.copy();environment.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        PYTHONPATH=manifest['source_snapshot'],PYTHONUNBUFFERED='1')
    executable=manifest['python'];script=str(Path(__file__).resolve())
    def run_job(name,argv,env):
        nonlocal child,active_job
        check_stop();active_job=name
        state['jobs'][name]=dict(status='starting',argv=argv,started_utc=stamp());status(name)
        with (root/(name+'.log')).open('a') as log:
            child=subprocess.Popen(argv,cwd=manifest['source_snapshot'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['jobs'][name].update(status='running',pid=child.pid,process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21]);status(name)
            stopping_since=None
            while child.poll() is None:
                if stopped or (root/'STOP').exists():
                    if stopping_since is None:
                        stopping_since=time.monotonic();(root/'STOP').touch();status('stopping',active_job=name)
                    if name not in ('stage1','stage2') and time.monotonic()-stopping_since>2:os.killpg(child.pid,signal.SIGTERM)
                    if time.monotonic()-stopping_since>300:os.killpg(child.pid,signal.SIGKILL)
                status(state['stage'],active_job=name)
                time.sleep(5)
            code=child.returncode;child=None
        state['jobs'][name].update(status='complete' if code==0 else 'failed',exit_code=code,completed_utc=stamp())
        status(state['stage']);check_stop()
        if code:raise RuntimeError(f'{name} exited {code}; see {name}.log')
    try:
        status('preparing_data')
        for path,digest in manifest['source_hashes'].items():
            if sha(Path(manifest['source_snapshot'])/path)!=digest:raise ValueError('Source snapshot changed: '+path)
        scratch=Path(manifest['scratch_corpus']);scratch.parent.mkdir(parents=True,exist_ok=True)
        if not scratch.exists():
            partial=scratch.with_suffix('.partial');shutil.copyfile(manifest['corpus'],partial)
            check_stop()
            if sha(partial)!=manifest['corpus_sha256']:raise ValueError('Corpus copy checksum mismatch')
            partial.replace(scratch)
        elif sha(scratch)!=manifest['corpus_sha256']:raise ValueError('Scratch corpus checksum mismatch')
        write(root/'DATA_READY.json',dict(corpus_sha256=manifest['corpus_sha256'],scratch_corpus=str(scratch),completed_utc=stamp()))
        run_job('cpu_preflight',[executable,script,'--manifest',args.manifest,'--mode','preflight'],environment)
        while any(process_alive(pid) for pid in manifest.get('generation_pids_to_drain',[])):
            check_stop();status('waiting_for_generation_stop');time.sleep(5)
        consecutive={};gpu_lock=None
        while True:
            check_stop();control=json.loads((root/'CONTROL.json').read_text());policy=control['device_policy']
            devices=gpu_inventory();status('waiting_for_gpu',device_policy=policy,gpus=devices)
            selected=None
            for gpu in sorted(devices,key=lambda g:-g['free_mib']):
                if gpu['index'] not in manifest.get('allowed_gpu_indices',[0,1,2,3]):continue
                consecutive[gpu['uuid']]=consecutive.get(gpu['uuid'],0)+1 if eligible(gpu,policy,manifest['gpu_memory_limit_mib']+4096) else 0
                if consecutive[gpu['uuid']]<3:continue
                candidate=Path('/dev/shm')/('mano_contact_'+gpu['uuid']+'.lock');handle=candidate.open('a')
                try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:handle.close();continue
                selected=gpu;gpu_lock=handle;break
            if selected is not None:break
            for _ in range(manifest.get('gpu_poll_seconds',20)//5):check_stop();time.sleep(5)
        check_stop();gpu_env=dict(environment,CUDA_VISIBLE_DEVICES=selected['uuid'])
        status('gpu_reserved',selected_gpu=selected,device_policy=policy)
        run_job('gpu_preflight',[executable,script,'--manifest',args.manifest,'--mode','preflight','--device','cuda:0'],gpu_env)
        common=['--corpus',str(scratch),'--assets',manifest['assets'],'--config',manifest['config'],'--device','cuda:0',
                '--threads','4','--checkpoint-every','100','--stop-file',str(root/'STOP'),'--preload']
        for stage in (1,2):
            command=[executable,script,'--manifest',args.manifest,'--mode','train','train','--stage',str(stage),*common,'--output',str(root/f'stage{stage}')]
            if stage==2:command+=['--stage1',str(root/'stage1'/'last.pt')]
            run_job(f'stage{stage}',command,gpu_env)
            # A successful worker exit is not sufficient evidence of the budget.
            import torch
            checkpoint=torch.load(root/f'stage{stage}'/'last.pt',map_location='cpu',weights_only=False)
            expected=manifest[f'stage{stage}_steps']
            if checkpoint['stage']!=stage or checkpoint['step']!=expected:raise RuntimeError('Incomplete stage checkpoint')
            state['actual_optimizer_updates']+=expected
            write(root/f'STAGE{stage}_COMPLETE.json',dict(step=expected,checkpoint=str(root/f'stage{stage}'/'last.pt'),completed_utc=stamp()))
            del checkpoint
        for split in ('val','test'):
            run_job('evaluate_'+split,[executable,'-m','retarget.learning.mano_contact.cli','evaluate','--corpus',str(scratch),
                '--assets',manifest['assets'],'--checkpoint',str(root/'stage2'/'last.pt'),'--split',split,
                '--output',str(root/(split+'.json')),'--device','cuda:0','--threads','4','--batch-size','32'],gpu_env)
        run_job('export',[executable,'-m','retarget.learning.mano_contact.cli','export','--checkpoint',str(root/'stage2'/'last.pt'),
                          '--output',str(root/'export')],environment)
        status('complete',completed_utc=stamp())
        scratch.unlink()
    except InterruptedError:
        last=root/str(active_job)/'last.pt'
        status('stopped_by_user',completed_utc=stamp(),last_checkpoint=str(last) if last.exists() else None)
    except BaseException:
        status('failed',error=traceback.format_exc(),completed_utc=stamp());raise


if __name__=='__main__':main()
