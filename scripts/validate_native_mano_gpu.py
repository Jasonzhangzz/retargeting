#!/usr/bin/env python3
"""GPU six-force and six-torque validation after a committed restart solve."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def write(path,data):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def run(a):
    root=a.root;start=time.monotonic()
    write(root/'dynamic_status.json',dict(state='waiting_for_committed_solve',pid=os.getpid(),optimizer_updates=0))
    while not (root/'solve_receipt.json').exists():
        status=json.loads((root/'status.json').read_text())
        if status['state']=='failed' or (root/'STOP').exists():raise RuntimeError('Solve failed or was stopped; no dynamic result is fabricated')
        if time.monotonic()-start>a.wait_timeout:raise TimeoutError('No committed solve within the configured startup wait')
        time.sleep(5)
    import numpy as np
    import torch,warp as wp
    from retarget.env.grasp.native_model_assets import load_model
    from retarget.env.grasp.native_mjwarp import NativePhysics
    from retarget.env.grasp.native_wrench_gpu import evaluate
    torch.set_num_threads(1);torch.cuda.set_device(a.device);wp.init()
    m=json.loads((root/'MANIFEST.json').read_text())
    for name,expected in m['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise ValueError('Solver sources changed after solve')
    pose_sha=hashlib.sha256((root/'poses.npz').read_bytes()).hexdigest()
    state=dict(np.load(root/'poses.npz'));n=len(state['final_qpos'])
    t=lambda x:torch.as_tensor(x,device=a.device,dtype=torch.float32)
    valid=torch.as_tensor(state['native_output'],device=a.device,dtype=torch.bool)
    ids=torch.nonzero(valid,as_tuple=False).flatten();flags=torch.zeros((n,12),device=a.device,dtype=torch.bool)
    values={};load_start=time.perf_counter()
    if len(ids):
        cpu,frozen,meta=load_model(root/'assets','eval')
        object_mask=np.array([cpu.body(b).name.startswith('obj-') for b in cpu.geom_bodyid])
        physics=NativePhysics(cpu,12*len(ids),device=a.device,qpos_to_ctrl=frozen['native_qpos_to_ctrl'],
            object_geom_mask=object_mask,frozen_mesh_arrays=frozen)
        obj=dict(np.load(root/'assets/object.npz'));vertices=t(obj['vertices']);faces=torch.as_tensor(obj['faces'],device=a.device)
        tri=vertices[faces];vol=(tri[:,0]*torch.linalg.cross(tri[:,1],tri[:,2])).sum(-1)/6
        center=(vol[:,None]*tri.sum(1)/4).sum(0)/vol.sum();radius=(vertices-center).norm(dim=-1).max()
        object_pose=t(frozen['qpos0'][27:34])[None].expand(len(ids),-1)
        torch.cuda.synchronize();load_seconds=time.perf_counter()-load_start
        write(root/'dynamic_status.json',dict(state='evaluating',inputs=n,eligible=len(ids),independent_worlds=12*len(ids),optimizer_updates=0))
        begin=time.perf_counter()
        result=evaluate(physics,t(state['final_qpos'])[ids],t(state['squeeze_qpos'])[ids],t(state['pregrasp_qpos'])[ids],
            torch.as_tensor(state['pregrasp_count'],device=a.device,dtype=torch.long)[ids],object_pose,
            radius.expand(len(ids)),torch.ones(len(ids),device=a.device,dtype=torch.bool),object_body=meta['object_body'])
        torch.cuda.synchronize();seconds=time.perf_counter()-begin
        flags[ids]=result['directions_pass'];values={k:v.detach().cpu().numpy() for k,v in result.items()}
    else:load_seconds=0.;seconds=0.
    values.update(evaluated_indices=ids.cpu().numpy(),directions_pass_all=flags.cpu().numpy())
    np.savez_compressed(root/'dynamic_results.npz',**values)
    report=dict(state='complete',inputs=n,evaluated_outputs=len(ids),native_rejected=n-len(ids),
        native_gsr_pass=int(flags[:,:6].all(-1).sum()),torque_pass=int(flags[:,6:].all(-1).sum()),combined_pass=int(flags.all(-1).sum()),
        loading_seconds=load_seconds,rollout_seconds=seconds,pose_sha256=pose_sha,optimizer_updates=0,
        translation_limit_m=.05,rotation_limit_deg=15.,physics_steps_per_direction=500,
        production_launch_allowed=False,scope='Fixed restart preflight; not a dataset-wide success rate')
    write(root/'dynamic_receipt.json',report);write(root/'dynamic_status.json',report);print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--wait-timeout',type=float,default=900.);a=p.parse_args();a.root=a.root.resolve()
    with (a.root/'dynamic.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (a.root/'dynamic_receipt.json').exists():raise ValueError('Already validated; do not replay')
        try:run(a)
        except BaseException:
            write(a.root/'dynamic_failure.json',dict(exception=traceback.format_exc(),optimizer_updates=0))
            write(a.root/'dynamic_status.json',dict(state='failed',optimizer_updates=0));raise


if __name__=='__main__':main()
