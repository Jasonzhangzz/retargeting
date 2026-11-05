#!/usr/bin/env python3
"""Finite GPU end-to-end restart preflight; never launches the CPU bulk runner."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import argparse,fcntl,hashlib,json,sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def prepare(a):
    import mujoco
    import native_graspadmm_retest as native
    import native_mano_graspadmm_variant as variant
    if mujoco.__version__!='3.3.3':raise ValueError('Asset freezing requires the native CPU 3.3.3 environment')
    root=a.root;root.mkdir(parents=True,exist_ok=True)
    if (root/'MANIFEST.json').exists():raise ValueError('Refusing to replace a frozen restart')
    source=native.load_upstream(a.source/'upstream')
    cfg=variant.configuration(source,a.source,'grasp_admm')
    entries=json.loads((a.source/'initialization.json').read_text())['inputs']
    first=np.load(entries[0]['path'],allow_pickle=True).item()
    selected=[e for e in entries if np.load(e['path'],allow_pickle=True).item()['scene_path']==first['scene_path']][:a.samples]
    if len(selected)!=a.samples:raise ValueError('Not enough fixed inputs in the first native scene')
    mappings={r['input']:r for r in json.loads((a.source/'mapping.json').read_text())['rows'] if r['completed']}
    scene=source.files.load_scene_cfg(first['scene_path'])
    if scene['task']['type']!='force_closure':raise ValueError('This preflight implements the original force-closure task only')
    assets=root/'assets';assets.mkdir()
    def freeze(env,stem):
        model=env._model
        (assets/(stem+'.xml')).write_text(env._spec.to_xml())
        arrays={k:getattr(model,k).copy() for k in dir(model) if isinstance(getattr(model,k),np.ndarray)}
        arrays['native_qpos_to_ctrl']=env._hand_qpos2ctrl_mat.copy()
        np.savez(assets/(stem+'.npz'),**arrays)
        flags={}
        for category,enum,prefix,attr in [('disable',mujoco.mjtDisableBit,'mjDSBL_','disableflags'),
                                         ('enable',mujoco.mjtEnableBit,'mjENBL_','enableflags')]:
            flags[category]={k[len(prefix):]:bool(int(getattr(model.opt,attr))&int(getattr(enum,k))) for k in dir(enum) if k.startswith(prefix)}
        keys=['timestep','tolerance','ls_tolerance','impratio','gravity','wind','density','viscosity','integrator',
              'solver','iterations','ls_iterations','noslip_iterations','noslip_tolerance','ccd_iterations','ccd_tolerance','cone','jacobian']
        options={}
        for k in keys:
            v=getattr(model.opt,k)
            options[k]=v.tolist() if isinstance(v,np.ndarray) else float(v) if isinstance(v,float) else int(v)
        write(assets/(stem+'.json'),dict(flags=flags,numeric_options=options,n_hand_body=env.n_hand_body,
            hand_prefix=env._cfg.hand_prefix,object_body=env._obj_active_bodyid,hand_qpos=27,
            body_names=[model.body(i).name for i in range(model.nbody)]))
        return model,env
    opt=source.sim.MuJoCo_OptEnv(source.sim.HandCfg(cfg.hand.xml_path,freejoint=True),scene,
                                source.sim.MuJoCo_OptCfg(obj_margin=cfg.op.pregrasp.cdist))
    opt.set_obj_margin(cfg.op.grasp.cdist);model,_=freeze(opt,'opt')
    mesh,*_=source.core.load_obj_trimesh(first)
    np.savez(assets/'object.npz',vertices=mesh.vertices,faces=mesh.faces,vertex_normals=mesh.vertex_normals)
    # Evaluation has its original mass normalization and friction/NoSlip setup.
    import copy
    eval_scene=copy.deepcopy(scene);eval_cfg=variant.configuration(source,a.source,'eval')
    for obj in eval_scene['scene'].values():
        if obj['type']=='rigid_object':
            info=source.files.load_json(obj['info_path']);coef=info['mass']/(info['density']*info['scale']**3)
            obj['density']=eval_cfg.obj_mass/(coef*np.prod(obj['scale']))
    env=source.sim.MuJoCo_EvalEnv(source.sim.HandCfg(cfg.hand.xml_path,freejoint=True),eval_scene,
        source.sim.MuJoCo_EvalCfg(miu_coef=list(eval_cfg.op.miu_coef)),debug_view=False,hard=eval_cfg.op.hard)
    freeze(env,'eval')
    rows=[]
    for e in selected:
        if sha(e['path'])!=e['sha256']:raise ValueError('Frozen input changed')
        data=np.load(e['path'],allow_pickle=True).item();m=mappings[e['path']]
        if data['hand_cbody']!=first['hand_cbody']:raise ValueError('Contact layouts differ in this homogeneous preflight')
        rows.append(dict(input=e['path'],sha256=e['sha256'],source=m['source'],source_sha256=m['source_sha256'],
                         source_state=m['source_state'],source_state_sha256=m['source_state_sha256'],
                         source_bodies=m['source_hand_cbody']))
    write(root/'inputs.json',rows)
    files=['scripts/restart_native_mano_gpu.py','retarget/env/grasp/native_model_assets.py',
        'retarget/env/grasp/native_mjwarp.py','retarget/env/grasp/native_batched_qp.py','retarget/env/grasp/native_wrench_gpu.py',
        'retarget/env/grasp/native_admm.py','retarget/env/grasp/shadow_mano_seed_gpu.py']
    write(root/'MANIFEST.json',dict(authorization='User: 修改后重新启动求解',stage='GPU_end_to_end_preflight',
        samples=len(rows),source=str(a.source),upstream_commit=native.COMMIT,optimizer_budget=dict(admm=50,substeps=10),
        solver='Original rigid-body ADMM formulas; native mjwarp; GPU six-ray batched QP',
        margin='MuJoCo 3.3.3 max geom margin/gap; native QP filter and physical checks',
        cpu_pose_solver_allowed=False,production_launch_allowed=False,inputs_sha256=sha(root/'inputs.json'),
        source_sha256={f:sha(ROOT/f) for f in files},asset_sha256={str(p.relative_to(root)):sha(p) for p in assets.iterdir()},
        gpu_device=a.device,batch_size=a.samples,seed=20261104))
    write(root/'status.json',dict(state='prepared',completed=0,total=len(rows),optimizer_updates=0))


def run(a):
    import torch,warp as wp
    from retarget.env.grasp.native_model_assets import load_model
    from retarget.env.grasp.native_mjwarp import NativePhysics,native_filter_squeeze_pregrasp
    from retarget.env.grasp.native_batched_qp import NativeBatchedQP
    from retarget.solver.graspadmm import solve,GPUProblem
    from retarget.env.grasp.native_admm import ObjectSurfaces,Settings,plane_projection
    root=a.root;m=json.loads((root/'MANIFEST.json').read_text());rows=json.loads((root/'inputs.json').read_text())
    for f,h in m['source_sha256'].items():
        if sha(ROOT/f)!=h:raise ValueError('Restart code changed: '+f)
    for f,h in m['asset_sha256'].items():
        if sha(root/f)!=h:raise ValueError('Frozen model changed: '+f)
    if sha(root/'inputs.json')!=m['inputs_sha256']:raise ValueError('Input selection changed')
    if (root/'STOP').exists():raise ValueError('STOP is present')
    torch.set_num_threads(1);torch.manual_seed(m['seed']);torch.cuda.set_device(a.device);wp.init()
    t=lambda x:torch.as_tensor(x,device=a.device,dtype=torch.float32)
    start=time.perf_counter();inputs=[]
    for row in rows:
        if sha(row['input'])!=row['sha256']:raise ValueError('Input changed')
        inputs.append(np.load(row['input'],allow_pickle=True).item())
    cpu,frozen,meta=load_model(root/'assets','opt');batch=len(rows)
    hand_bodies=torch.tensor([cpu.body(meta['hand_prefix']+b).id for b in inputs[0]['hand_cbody']],device=a.device)
    object_mask=np.array([cpu.body(b).name.startswith('obj-') for b in cpu.geom_bodyid])
    physics=NativePhysics(cpu,batch,device=a.device,qpos_to_ctrl=frozen['native_qpos_to_ctrl'],
        object_geom_mask=object_mask,frozen_mesh_arrays=frozen)
    q=t(np.stack([v['grasp_qpos'].reshape(-1) for v in inputs]));physics.reset(q)
    # Input positions are decoded on device, not by a CPU pose evaluator.
    world=t(np.stack([v['hand_cpn_w'] for v in inputs]));local=physics.local_points(hand_bodies,world[...,:3])
    rotation=physics.tensor('xmat')[:,hand_bodies].reshape(batch,-1,3,3)
    normal=(rotation.transpose(-1,-2)@world[...,3:,None]).squeeze(-1)
    local=torch.cat((local,normal),-1)
    obj=dict(np.load(root/'assets/object.npz'))
    surfaces=ObjectSurfaces([obj['vertices']]*batch,[obj['faces']]*batch,a.device)
    cp=t(np.stack([v['obj_cpn_w'] for v in inputs]));external=t(np.stack([v['ext_wrench'] for v in inputs]))
    center=t(np.stack([v['ext_center'] for v in inputs]))
    required=[[cpu.body(meta['hand_prefix']+b).id for b in g] for g in inputs[0]['required_cbody']]
    if a.capture_physics:physics.prepare_graph(10)
    torch.cuda.synchronize();preparation=time.perf_counter()-start
    def status(iteration,steps,fallback,failed):
        if (root/'STOP').exists():raise InterruptedError('STOP requested')
        if iteration==1 or iteration%10==0:
            value=dict(state='solving',iteration=iteration,completed=0,total=batch,
                finite_active=int((~failed).sum()),fallback=int(fallback.sum()),wall_seconds=time.perf_counter()-start)
            write(root/'status.json',value);print(json.dumps(value),flush=True)
    qp=NativeBatchedQP(len(hand_bodies),a.device)
    torch.cuda.synchronize();begin=time.perf_counter()
    admm=solve(GPUProblem(physics,q,hand_bodies,local,cp,surfaces,qp,external,center,meta['n_hand_body']),progress=status)
    torch.cuda.synchronize();admm_seconds=time.perf_counter()-begin
    # Commit only the final ADMM state before the independent filter stage.
    final={('admm_'+k):v.detach().cpu().numpy() for k,v in admm.items() if isinstance(v,torch.Tensor)}
    final.update({('physics_'+k):v.detach().cpu().numpy() for k,v in physics.snapshot().items()})
    checkpoint=root/'admm_final_state.npz';temporary=root/'admm_final_state.tmp.npz'
    np.savez_compressed(temporary,**final);temporary.replace(checkpoint)
    write(root/'admm_receipt.json',dict(completed=batch,iterations=50,admm_seconds=admm_seconds,
        checkpoint_sha256=sha(checkpoint),source_manifest_sha256=sha(root/'MANIFEST.json'),
        finite=int(torch.isfinite(admm['qpos']).all(-1).sum()),fallback=int(admm['fallback_used'].sum())))
    nominal_margin=wp.to_torch(physics.model.geom_margin).clone()
    begin=time.perf_counter();native=native_filter_squeeze_pregrasp(physics,admm,required,external,center,
        hand_body_count=meta['n_hand_body'])
    torch.cuda.synchronize();filter_seconds=time.perf_counter()-begin
    # Restore normal grasp margins and final pose before posthoc certification.
    wp.to_torch(physics.model.geom_margin).copy_(nominal_margin);physics.reset(admm['qpos'])
    begin=time.perf_counter();slots=physics.contacts(hand_bodies,admm['hand_cpn_b'])
    projected,_,distance,faces,_=surfaces.query(slots[...,:3],torch.arange(batch,device=a.device))
    triangles,normals=surfaces.triangle_data(faces)
    _,outward=plane_projection(projected,triangles,normals)
    angle=torch.rad2deg(torch.acos((-(slots[...,3:]*outward).sum(-1)).clamp(-1,1)))
    coverage=((distance.abs()<=.002)&(angle<=30.)).float().mean(-1)
    packet=physics.native_contacts(meta['n_hand_body'],.002)
    collision=physics.collision_pass(packet,-.002,-.002)
    torch.cuda.synchronize();diagnostic_seconds=time.perf_counter()-begin
    # Persist actual solve states before the separate dynamic rollout stage.
    values=dict(initial_qpos=q,final_qpos=admm['qpos'],hand_cpn_b=admm['hand_cpn_b'],cp_o=admm['cp_o'],
        hand_slots=slots,projected_points=projected,object_normals=outward,signed_distance_m=distance,
        coverage=coverage,collision_pass=collision,
        native_output=native['native_output'],fallback=admm['fallback_used'],numerical_failure=admm['numerical_failure'],
        squeeze_qpos=native['squeeze_qpos'],pregrasp_qpos=native['pregrasp_qpos'],pregrasp_count=native['pregrasp_count'])
    np.savez_compressed(root/'poses.npz',**{k:v.detach().cpu().numpy() for k,v in values.items()})
    write(root/'solve_receipt.json',dict(completed=batch,total=batch,finite=int(torch.isfinite(admm['qpos']).all(-1).sum()),
        native_output=int(native['native_output'].sum()),coverage_pass=int((coverage>=.8).sum()),
        collision_pass=int(collision.sum()),native_geometry_joint_pass=int((native['native_output']&(coverage>=.8)&collision).sum()),
        load_and_preparation_seconds=preparation,admm_seconds=admm_seconds,filter_seconds=filter_seconds,
        diagnostic_seconds=diagnostic_seconds,admm_amortized_seconds=admm_seconds/batch,
        physics_launch='CUDA graph' if a.capture_physics else 'eager CUDA mjwarp; no CPU fallback',
        coverage_normal_source='Transported input slot normals; final surface-normal validation is pending',
        native_gsr_measured=False,scope='Finite restart preflight, not the full dataset'))
    write(root/'status.json',dict(state='solve_complete_pending_dynamic_validation',completed=batch,total=batch,
        wall_seconds=time.perf_counter()-start,production_launch_allowed=False))
    print((root/'solve_receipt.json').read_text(),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path)
    p.add_argument('--samples',type=int,default=16);p.add_argument('--device',default='cuda:0')
    p.add_argument('--capture-physics',action='store_true');a=p.parse_args()
    a.root=a.root.resolve();a.root.mkdir(parents=True,exist_ok=True)
    with (a.root/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.mode=='run' and (a.root/'solve_receipt.json').exists():raise ValueError('Already solved; do not replay completed updates')
        write(a.root/'process.json',dict(pid=os.getpid(),mode=a.mode,started_unix=time.time(),argv=sys.argv))
        try:prepare(a) if a.mode=='prepare' else run(a)
        except BaseException:
            write(a.root/'failure.json',dict(exception=traceback.format_exc(),optimizer_state_may_be_inflight=a.mode=='run'))
            write(a.root/'status.json',dict(state='failed',mode=a.mode));raise


if __name__=='__main__':main()
