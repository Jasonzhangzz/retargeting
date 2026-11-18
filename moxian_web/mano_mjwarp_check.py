"""Isolated MJWarp convex diagnostic; original triangles remain authoritative."""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def _scene(model,poses,roots):
    meshes=[];bodies=[]
    for part in range(16):
        points=model.rest_np[np.asarray(model.labels)==part]-model.rest_j_np[part]
        if len(points)<4:raise ValueError('MJWarp 独立检查缺少骨段表面')
        vertices=' '.join(map(str,points.ravel()))
        meshes.append(f'<mesh name="p{part}" vertex="{vertices}"/>')
        bodies.append(f'<body name="b{part}"><freejoint/><geom type="mesh" mesh="p{part}" mass=".01" margin=".002"/></body>')
    xml='<mujoco><option gravity="0 0 0"><flag multiccd="disable"/></option><asset>'+''.join(meshes)+'</asset><worldbody>'+''.join(bodies)+'</worldbody></mujoco>'
    with torch.no_grad():_,j,r,_=model.decode(poses,roots)
    quaternion=Rotation.from_matrix(r.detach().cpu().numpy().reshape(-1,3,3)).as_quat().reshape(len(poses),16,4)[...,[3,0,1,2]]
    qpos=np.concatenate((j.detach().cpu().numpy(),quaternion),-1).reshape(len(poses),-1)
    return xml,qpos


def _run_scene(xml,qpos,device):
    import mujoco
    import mujoco_warp as mjw
    import warp as wp
    began=time.monotonic();count=0;maximum=0.;compilation=0.
    wp.init()
    # Initialization, Torch copies and Warp launches must use one stream.
    # A native diagnostic failure must not poison the display's CUDA context.
    stream=wp.stream_from_torch(torch.cuda.current_stream(device))
    with wp.ScopedDevice(device),wp.ScopedStream(stream):
        cpu=mujoco.MjModel.from_xml_string(xml);gpu=mjw.put_model(cpu)
        batch=min(32,len(qpos));data=mjw.make_data(cpu,nworld=batch,nconmax=512,njmax=512)
        for start in range(0,len(qpos),batch):
            chunk=qpos[start:start+batch];actual=len(chunk)
            if actual<batch:chunk=np.concatenate((chunk,np.repeat(chunk[-1:],batch-actual,0)))
            wp.to_torch(data.qpos).copy_(torch.as_tensor(chunk,device=device,dtype=torch.float32))
            first=time.monotonic();mjw.kinematics(gpu,data);mjw.collision(gpu,data)
            wp.synchronize_stream(stream)
            if start==0:compilation=time.monotonic()-first
            if bool(wp.to_torch(data.overflow).any()):raise ValueError('MJWarp 独立接触检查缓冲区溢出')
            n=int(wp.to_torch(data.nacon).sum())
            if not 0<=n<=data.naconmax:raise ValueError('MJWarp 接触数量越界')
            if n:
                world=wp.to_torch(data.contact.worldid)[:n];dist=wp.to_torch(data.contact.dist)[:n]
                if not bool(torch.isfinite(dist).all()):raise ValueError('MJWarp 接触距离非有限')
                active=world<actual;count+=int(active.sum())
                maximum=max(maximum,float(torch.where(active,-dist,0).clamp_min(0).max()))
    return dict(backend='mujoco_warp',frames=len(qpos),query_valid=True,overflow=False,contact_records=count,
        proxy_max_penetration_mm=maximum*1000,seconds=time.monotonic()-began,compilation_s=compilation,
        representation='rigid_anatomical_segment_convex_hulls',authoritative=False,process_isolated=True,
        note='独立凸包诊断；准确间隙和2mm验收使用原始MANO三角面，不能用本结果替代')


def cross_check(model,poses,roots,cancel=None,timeout_s=90):
    began=time.monotonic();xml,qpos=_scene(model,poses,roots)
    with tempfile.TemporaryDirectory(prefix='mano-mjwarp-') as folder:
        folder=Path(folder);source=folder/'scene.npz';target=folder/'result.json'
        np.savez(source,xml=xml,qpos=qpos)
        with (folder/'worker.log').open('w+') as log:
            process=subprocess.Popen([sys.executable,'-I',str(Path(__file__).resolve()),str(source),str(target),str(model.device)],stdout=log,stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    if cancel is not None and cancel.is_set():raise InterruptedError('MJWarp 接触交叉检查已取消')
                    if time.monotonic()-began>timeout_s:raise TimeoutError('MJWarp 独立进程检查超时')
                    time.sleep(.1)
                if process.returncode or not target.exists():
                    log.seek(0);raise RuntimeError(f'MJWarp 独立进程退出 {process.returncode}: '+log.read()[-2000:])
                result=json.loads(target.read_text());result['seconds']=time.monotonic()-began
                return result
            except InterruptedError:raise
            except (RuntimeError,TimeoutError,ValueError) as exc:
                return dict(backend='mujoco_warp',frames=len(poses),query_valid=False,process_isolated=True,
                    authoritative=False,error=str(exc),seconds=time.monotonic()-began,compilation_s=0.)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=2)
                    except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':
    with np.load(sys.argv[1],allow_pickle=False) as source:
        result=_run_scene(str(source['xml'].item()),source['qpos'],sys.argv[3])
    Path(sys.argv[2]).write_text(json.dumps(result,allow_nan=False))
