#!/usr/bin/env python3
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.retarget_dexonomy_mano import read,write,stamp


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight',type=Path)
    parser.add_argument('--wuji-preflight',type=Path)
    parser.add_argument('--raw-source',type=Path,required=True)
    parser.add_argument('--packed',type=Path,required=True)
    parser.add_argument('--mano',type=Path,required=True)
    parser.add_argument('--wuji',type=Path,required=True)
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    processes=[]
    def stop(*_):
        (a.output/'STOP').touch()
        for root in (a.packed,a.mano,a.wuji):
            if root.exists():(root/'STOP').touch()
        for process in processes:
            if process.poll() is None:process.terminate()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def state(stage,**kw):write(a.output/'status.json',dict(stage=stage,updated_utc=stamp(),**kw))
    def launch(name,root,arguments):
        if (a.output/'STOP').exists():raise RuntimeError('Pipeline stopped')
        root.mkdir(parents=True,exist_ok=True)
        receipt=a.output/'launches'/(name+'.json')
        if receipt.exists():raise RuntimeError('Existing launch requires inspection: '+str(receipt))
        argv=[sys.executable,'-u',*arguments]
        with (root/'run.log').open('ab') as log:
            process=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(process)
        write(receipt,dict(pid=process.pid,argv=argv,started_utc=stamp(),stage=name))
        return process
    def link(name,target):
        path=ROOT/'results'/name
        if path.is_symlink():path.unlink()
        elif path.exists():raise RuntimeError('Refuse to replace a real directory: '+str(path))
        path.symlink_to(target.resolve())
    files=['retarget/solver/cem.py','retarget/env/grasp/closure_types.py','retarget/env/grasp/projected_qp.py','retarget/env/grasp/pose_gpu.py','retarget/env/grasp/pose_inputs.py','retarget/env/grasp/qp_projection.py','retarget/env/grasp/pose_objective.py',
        'retarget/env/grasp/mano_adapter.py','retarget/env/grasp/mano_solver.py','scripts/pack_dexonomy_batches.py',
        'scripts/retarget_dexonomy_mano.py','scripts/retarget_mano_wuji2.py','scripts/run_mano_then_wuji.py']
    write(a.output/'manifest.json',dict(pid=os.getpid(),authorized_order=['input_packing_and_mano','wuji2_after_mano_complete'],
        batch_size=2048,mano_devices=['cuda:0','cuda:1','cuda:2','cuda:3'],mano_workers_per_device=1,
        source=str(a.raw_source.resolve()),packed=str(a.packed.resolve()),mano=str(a.mano.resolve()),wuji=str(a.wuji.resolve()),
        objective='template_historical_wrench_qp_axis_v1',solver='repository CEM with independent batch dimension',
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},created_utc=stamp()))
    try:
        if a.preflight:
            state('waiting_for_2048_preflight',preflight=str(a.preflight))
            while True:
                if (a.output/'STOP').exists():raise RuntimeError('Pipeline stopped')
                status=read(a.preflight/'status.json') if (a.preflight/'status.json').exists() else {}
                if status.get('failed_shards',0) or status.get('state') in ('failed','stopped','completed_with_errors'):
                    raise RuntimeError('Batch-2048 preflight failed')
                if status.get('state')=='completed':
                    if status['samples']!=2048:raise RuntimeError('Preflight did not produce 2048 samples')
                    verifications=list(a.preflight.glob('*/shards/*/verification.json'))
                    if len(verifications)!=1 or not all(read(verifications[0])['checks'].values()):raise RuntimeError('Preflight output checks failed')
                    write(a.output/'preflight.json',dict(status=status,verification=read(verifications[0])))
                    break
                process=read(a.preflight/'process.json') if (a.preflight/'process.json').exists() else None
                if process and not Path('/proc/'+str(process['pid'])).exists():raise RuntimeError('Preflight process exited without completion')
                time.sleep(10)
        if a.wuji_preflight:
            state('waiting_for_wuji_gpu_preflight',preflight=str(a.wuji_preflight))
            while True:
                if (a.output/'STOP').exists():raise RuntimeError('Pipeline stopped')
                status=read(a.wuji_preflight/'status.json') if (a.wuji_preflight/'status.json').exists() else {}
                if status.get('state') in ('failed','stopped'):raise RuntimeError('WUJI GPU preflight failed')
                if status.get('state')=='completed':
                    verifications=list(a.wuji_preflight.glob('*/shards/*/verification.json'))
                    if not verifications or not all(read(p).get('complete') for p in verifications):raise RuntimeError('WUJI preflight output missing')
                    write(a.output/'wuji_preflight.json',dict(status=status,verification=[read(p) for p in verifications]))
                    break
                time.sleep(10)
        packer=launch('input_packing',a.packed,['scripts/pack_dexonomy_batches.py','--source',str(a.raw_source),'--output',str(a.packed)])
        state('packing_first_batch',packer_pid=packer.pid)
        while not (a.packed/'status.json').exists():
            if packer.poll() not in (None,0):raise RuntimeError('Input packer failed before first batch')
            if (a.output/'STOP').exists():raise RuntimeError('Pipeline stopped')
            time.sleep(2)
        mano=launch('mano',a.mano,['scripts/retarget_dexonomy_mano.py','--source',str(a.packed),'--output',str(a.mano),
            '--devices','cuda:0,cuda:1,cuda:2,cuda:3','--workers-per-device','1','--batch-size','2048',
            '--candidate-chunk','2048','--population','128','--iterations','24','--watch'])
        link('dexonomy_mano_current',a.mano)
        state('mano_running_wuji_waiting',packer_pid=packer.pid,mano_pid=mano.pid)
        while packer.poll() is None or mano.poll() is None:
            if (a.output/'STOP').exists():raise RuntimeError('Pipeline stopped')
            if packer.poll() not in (None,0) or mano.poll() not in (None,0):raise RuntimeError('MANO/input stage process failed')
            if (a.mano/'status.json').exists() and read(a.mano/'status.json').get('failed_shards',0):raise RuntimeError('MANO shard failed')
            if not (a.output/'first_production_timing.json').exists():
                first=next(a.mano.glob('*/shards/*/verification.json'),None)
                if first:
                    verified=read(first)
                    write(a.output/'first_production_timing.json',dict(samples=verified['samples'],seconds=verified['seconds'],
                        amortized_seconds_per_pose=verified['seconds']/verified['samples'],
                        scope=verified.get('timing_scope'),verification=str(first),updated_utc=stamp()))
            time.sleep(10)
        result=read(a.mano/'status.json')
        if read(a.packed/'status.json')['state']!='completed' or result['state']!='completed' or result['samples']==0 or result['samples']!=read(a.packed/'status.json')['samples'] or result['failed_shards']:
            raise RuntimeError('MANO/input completion gate did not pass')
        wuji=launch('wuji2',a.wuji,['scripts/retarget_mano_wuji2.py','--source',str(a.mano),'--output',str(a.wuji),
            '--workers-per-device','4','--population','1024'])
        link('dexonomy_wuji2_current',a.wuji)
        state('wuji_running_mano_complete',wuji_pid=wuji.pid,mano_samples=result['samples'])
        while wuji.poll() is None:
            if (a.output/'STOP').exists():raise RuntimeError('Pipeline stopped')
            time.sleep(10)
        if wuji.returncode or read(a.wuji/'status.json')['state']!='completed':raise RuntimeError('WUJI2 stage failed')
        state('completed',mano_samples=result['samples'])
    except Exception as error:
        stopped=(a.output/'STOP').exists()
        stop()
        state('stopped' if stopped else 'failed',error=repr(error))
        raise
if __name__=='__main__':main()
