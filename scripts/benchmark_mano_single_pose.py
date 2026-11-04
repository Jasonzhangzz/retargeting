#!/usr/bin/env python3
"""Synchronized batch=1 latency of the active GPU MANO IK/CEM/QP pipeline."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import sys,json,time,signal,subprocess,hashlib
from dataclasses import replace,asdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.retarget_dexonomy_mano import read,write,load_npz,GeometryLoader,stamp


def main():
    import torch,warp as wp
    from retarget.env.grasp.mano_solver import Settings,ManoDecoder,ShadowSkeleton,FixedContacts,SurfaceBatch,MeshCache,optimize
    output=ROOT/'output/mano_single_pose_latency_20261104';output.mkdir(parents=True,exist_ok=True)
    lock=(output/'.lock').open('a+')
    import fcntl
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    device='cuda:1';torch.set_num_threads(1);torch.cuda.set_device(device);wp.init()
    source=ROOT/'results/dexonomy_shadow_b2048_shared_20261104/7_Prismatic_3_Finger/shards/0000000'
    entries=read(source/'manifest.json')['cases'];archive=load_npz(source/'poses.npz')
    mapping=read(ROOT/'output/dexonomy_shadow_mano_mapping_20261103/data.json')['mappings']
    decoder=ManoDecoder(ROOT/'data/grasp/assets/mano/MANO_RIGHT.pkl',device,24)
    decoder.configure_contact_vertices(mapping)
    skeleton=ShadowSkeleton(ROOT/'data/dexonomy/shadow_assets/right.xml',device)
    fixed=FixedContacts(decoder,mapping['7_Prismatic_3_Finger'])
    loader=GeometryLoader(ROOT/'data/dexonomy');_,geo=loader.get(entries[0]);cache=MeshCache(device)
    config=read(ROOT/'results/dexonomy_mano_gpu_qp_b2048_20261104/configuration.json')
    settings=Settings(**config['settings']);settings=replace(settings,batch_size=1)
    state=read(ROOT/'results/dexonomy_mano_gpu_qp_b2048_20261104/status.json')
    worker=next(w for w in state['workers'] if w['device']==device);pid=worker['pid']
    command=Path(f'/proc/{pid}/cmdline').read_bytes()
    if b'spawn_main' not in command:raise RuntimeError('Unexpected worker identity')
    identity=Path(f'/proc/{pid}/stat').read_text().split()[21]
    marker=output/'worker_suspended';marker.write_text(str(pid))
    watchdog='''import time,os,signal
from pathlib import Path
time.sleep(180)
p=Path('/proc/%d/stat')
if Path(%r).exists() and p.exists() and p.read_text().split()[21]==%r:os.kill(%d,signal.SIGCONT)
'''%(pid,str(marker),identity,pid)
    subprocess.Popen([sys.executable,'-c',watchdog],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    measurements=[]
    try:
        os.kill(pid,signal.SIGSTOP)
        # Let already submitted work drain before timing; no model work timed here.
        time.sleep(1)
        gpu_processes=subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
        write(output/'status.json',dict(state='measuring',pid=os.getpid(),paused_worker=pid,settings=asdict(settings),started_utc=stamp(),gpu_processes=gpu_processes))
        for chunk in (128,2048):
            setting=replace(settings,candidate_chunk=chunk)
            for repeat in range(3):
                q=torch.as_tensor(archive['grasp_qpos'][:1],device=device,dtype=torch.float32)
                op=torch.as_tensor(archive['object_pose_world'][:1],device=device,dtype=torch.float32)
                if op.ndim==3:op=op[:,0]
                scale=archive['scene_scale'][:1]
                torch.cuda.synchronize(device);begun=time.perf_counter()
                target=skeleton(q);points,normals=skeleton.contacts(q,mapping['7_Prismatic_3_Finger'])
                surface=SurfaceBatch([geo],[0],op,scale,device,mesh_cache=cache)
                result=optimize(decoder,target,surface,setting,7,fixed,points,normals)
                torch.cuda.synchronize(device);seconds=time.perf_counter()-begun
                measurements.append(dict(candidate_chunk=chunk,repeat=repeat,seconds=seconds,first_call=repeat==0,
                    finite=bool(torch.isfinite(result['variables']).all()),closure=bool(result['projected_closure'][0]),
                    final_cost=float(result['final_cost'][0])))
                write(output/'measurements.json',dict(measurements=measurements,updated_utc=stamp()))
                print(json.dumps(measurements[-1]),flush=True)
        write(output/'report.json',dict(settings=asdict(settings),device=device,pose_batch=1,
            scope='input pose already on device -> Shadow FK and contacts -> surface transforms/BVH -> MANO initialization and IK -> CEM/QP -> final device pose/keypoints/contacts; synchronized; excludes explicit input/output file IO and eagerly loaded model assets; first call includes lazy collision model/kernel-cache/graph setup',
            gpu_processes=gpu_processes,paused_worker=pid,measurements=measurements,completed_utc=stamp()))
    finally:
        if Path(f'/proc/{pid}/stat').exists() and Path(f'/proc/{pid}/stat').read_text().split()[21]==identity:os.kill(pid,signal.SIGCONT)
        marker.unlink(missing_ok=True)
        write(output/'status.json',dict(state='completed' if len(measurements)==6 else 'failed',worker_resumed=True,measurements=len(measurements),updated_utc=stamp()))
if __name__=='__main__':main()
