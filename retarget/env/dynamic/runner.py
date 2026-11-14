"""GPU IK and MJWarp-only physical initialization, optimization and replay."""
from dataclasses import asdict
from pathlib import Path
import json,time,hashlib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from retarget.solver.dial_mpc import DIALConfig,DIALMPC
from .scene import build_scene,fit_kinematic_reference
from .execution import GPUExecution


def find_object_mesh(root,name):
    root=Path(root)
    if root.is_file():return root
    for p in (root/name/'textured_simple.obj',root/name/'model.obj',root/name/'textured.obj',root/name/'google_16k/textured.obj',root/name/'textured_simple.ply'):
        if p.is_file():return p
    raise FileNotFoundError(f'Object-frame YCB mesh not found: {root}/{name}')


def metrics(model,qpos,forces,penetration,reference,phase=None):
    error=np.linalg.norm(qpos[:,26:29]-reference[:,26:29],axis=1)
    dq=Rotation.from_quat(qpos[:,29:33],scalar_first=True).inv()*Rotation.from_quat(reference[:,29:33],scalar_first=True)
    angles=dq.magnitude();squeeze=np.sort(forces,axis=1)[:,-2]  # at least two loaded fingers
    active=np.ones(len(squeeze),bool) if phase is None else np.asarray(phase)>0
    grip=float(np.median(squeeze[active])) if active.any() else None
    all_finite=bool(np.isfinite(qpos).all())
    return dict(object_position_rmse_mm=float(np.sqrt(np.mean(error**2))*1000),object_position_median_mm=float(np.median(error)*1000),
                object_rotation_median_deg=float(np.rad2deg(np.median(angles))),finger_reference_rmse_rad=float(np.sqrt(np.mean((qpos[:,6:26]-reference[:,6:26])**2))),
                maximum_penetration_mm=float(np.max(penetration)*1000),mean_maximum_penetration_mm=float(np.mean(penetration)*1000),
                finger_force_p95_N=np.percentile(forces,95,axis=0).tolist(),median_two_finger_squeeze_N=grip,grasp_scored_frames=int(active.sum()),
                valid_finite=all_finite,
                accepted=bool(all_finite and np.max(error)<.15 and np.max(angles)<np.deg2rad(40) and np.mean(penetration)<=.005 and grip is not None and grip>=1 and np.percentile(forces,95,axis=0).max()<=100),
                acceptance_scope='GNR-inspired conservative tracking/contact checks; MuJoCo convex mesh contacts, not SuperDex SDF or hardware validation')


def replay(model,initial,controls,substeps,device='cuda:0'):
    return GPUExecution(model,initial,substeps=substeps,device=device).advance(controls)


def solve(motion_path,models,output,*,config=DIALConfig(),device='cuda:0',frames=None,mass=.1,friction=.8,substeps=10,grasp_frames=12,capture=True,mano_path=None,cost_config=None,floor_z=None,export=True):
    from .rollout import Evaluator,CostConfig
    import warp as wp
    import torch
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    with np.load(motion_path,allow_pickle=False) as f:motion={k:f[k] for k in f.files}
    n=min(len(motion['frame_ids']),frames or len(motion['frame_ids']))
    if n<2:raise ValueError('Need at least two motion frames')
    fps=float(motion['fps']);dt=1/fps
    mesh=find_object_mesh(models,str(motion['object_name']))
    model,xml=build_scene(mesh,mass=mass,friction=friction,timestep=dt/substeps,floor_z=floor_z)
    (out/'scene.xml').write_text(xml)
    ik,ik_info=fit_kinematic_reference(model,motion['mano_joints'][:n],device)
    reference=np.c_[ik,motion['object_pose'][:n]]
    np.save(out/'reference_qpos.npy',reference)
    np.save(out/'ik_controls.npy',ik)
    execution=GPUExecution(model,substeps=substeps,device=device)
    initial=execution.establish(ik[0],reference[0,26:],frames=grasp_frames)
    np.savez(out/'initial_state.npz',qpos=initial.qpos,qvel=initial.qvel,ctrl=initial.ctrl,qacc_warmstart=initial.qacc_warmstart,time=np.array(initial.time))
    current=initial
    # Hold terminal targets only to pad the final horizon, never fabricate frames.
    padded=np.concatenate([reference,np.repeat(reference[-1:],config.horizon+1,axis=0)])
    costs=CostConfig() if cost_config is None else cost_config
    phase_source='prepared_labels' if 'grasp_phase' in motion else 'unknown_disabled'
    if 'grasp_phase' not in motion and mano_path is not None:
        from retarget.task.ho3d.source import vertex_finger_owners
        owner=vertex_finger_owners(str(mano_path))
        contact=motion['vertex_contact']
        active=np.stack([contact[:,owner==f].sum(1)>=3 for f in range(5)],axis=1)
        motion['grasp_phase']=((active.sum(1)>=2)&motion['vertex_contact_valid']).astype(np.float32)
        phase_source='source_contact_with_explicit_MANO_weights'
    phase=np.asarray(motion.get('grasp_phase',np.zeros(len(motion['frame_ids']))))[:n]
    padded_phase=np.r_[phase,np.repeat(phase[-1],config.horizon+1)]
    started=time.perf_counter()
    evaluator=Evaluator(model,initial,padded,config.horizon,config.samples+1,substeps=substeps,device=device,cost=costs,capture=capture,contact_phase=padded_phase)
    warmup=time.perf_counter()-started
    solver=DIALMPC(evaluator,model.actuator_ctrlrange,config,device=device)
    controls=[];qpos=[];qvel=[];forces=[];penetration=[];plans=[]
    sigma=np.r_[np.full(3,.1),np.full(3,.5),np.full(20,1.)]
    start=time.perf_counter()
    for frame in range(0,n-1,config.execute_steps):
        before=time.perf_counter();evaluator.set_state(current,frame)
        plan,info=solver.solve(padded[frame:frame+config.horizon,:26],sigma)
        count=min(config.execute_steps,n-1-frame)
        command=plan[:count].detach().cpu().numpy()
        qs,vs,fs,ps=execution.advance(command)
        current=execution.snapshot()
        controls.extend(command);qpos.extend(qs);qvel.extend(vs);forces.extend(fs);penetration.extend(ps)
        solver.shift(count)
        info.update(frame=frame,seconds=time.perf_counter()-before,executed=count)
        plans.append(info);(out/'progress.json').write_text(json.dumps(dict(frame=frame+count,total=n-1,plans=len(plans),latest=info),indent=2))
        print(json.dumps(dict(frame=frame+count,total=n-1,cost=info['cost'],seconds=info['seconds'])),flush=True)
    qpos,qvel,forces,penetration=map(np.asarray,(qpos,qvel,forces,penetration))
    controls=np.asarray(controls)
    base=replay(model,initial,ik[:-1],substeps,device)
    target=reference[1:len(qpos)+1]
    result=dict(optimized=metrics(model,qpos,forces,penetration,target,phase[1:len(qpos)+1]),ik_baseline=metrics(model,base[0],base[2],base[3],target,phase[1:len(qpos)+1]),
                source=str(Path(motion_path).resolve()),source_sha256=hashlib.sha256(Path(motion_path).read_bytes()).hexdigest(),mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),frames=n,control_fps=fps,physics_fps=fps*substeps,grasp_initialization_frames=grasp_frames,
                mesh=str(mesh.resolve()),grasp_phase_source=phase_source,floor_z=floor_z,object_mass_kg=mass,object_friction=friction,collision_geometry='MuJoCo convex hull of original object mesh; no recentering',
                solver=asdict(config),cost=asdict(costs),warmup_seconds=warmup,optimization_seconds=time.perf_counter()-start,
                warp_cache=str(wp.config.kernel_cache_dir),device=device,physics_backend='robokit.MjWarpSim',cpu_mujoco_stepping=False,ik=ik_info,plans=plans)
    np.savez_compressed(out/'rollout.npz',qpos=qpos,qvel=qvel,controls=controls,reference_qpos=target,finger_force_N=forces,maximum_penetration_m=penetration,
                        baseline_qpos=base[0],baseline_finger_force_N=base[2],frame_ids=motion['frame_ids'][1:n],time_s=motion['time_s'][1:n])
    (out/'result.json').write_text(json.dumps(result,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x)))
    if export:
        from retarget.viewer.dynamic import export_viewer
        export_viewer(model,motion,qpos,base[0],target,out,device=device)
    return result
