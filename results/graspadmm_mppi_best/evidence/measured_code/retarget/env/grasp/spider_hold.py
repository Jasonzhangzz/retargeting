"""Independent six-group loaded-contact and force-closure hold verification."""
from pathlib import Path
import mujoco
import numpy as np
from retarget.common.results import write_json
from .spider_seed import load_seed,GROUPS
from .spider_geometry import ContactRequirements,exact_closure
from .spider_validation import object_errors
from retarget.env.geometry import ObjectMeshSurface


def audit_hold(model_path,trajectory,reference,gpu_observations,destination):
    model=mujoco.MjModel.from_xml_path(str(model_path));_,_,seed,_=load_seed(reference.seed_source)
    requirements=ContactRequirements(model,seed)
    radius=max(float(np.linalg.norm(ObjectMeshSurface.from_model(model,model.body('grasp_object').id).triangles.reshape(-1,3),axis=1).max()),.01)
    with np.load(trajectory) as raw:qpos=raw['qpos'].copy();qvel=raw['qvel'].copy();controls=raw['ctrl'].copy()
    duration=len(controls)*model.opt.timestep
    duration_complete=bool(duration>=.4-1e-12)
    zero_initial_velocity=bool(np.isfinite(qvel[0]).all() and np.max(np.abs(qvel[0]))<=1e-12)
    required=reference.required_groups
    gpu_data=mujoco.MjData(model);cpu_data=mujoco.MjData(model)
    cpu_data.qpos[:]=qpos[0];cpu_data.qvel[:]=qvel[0]
    cpu_q=[cpu_data.qpos.copy()];loads={'gpu':[],'cpu':[]};closure={'gpu':[],'cpu':[]};middle={'gpu':[],'cpu':[]}
    for step,command in enumerate(controls):
        gpu_data.qpos[:]=qpos[step];mujoco.mj_kinematics(model,gpu_data)
        obs=gpu_observations[step];indices=[];groups=[]
        for i,(body,point,normal,force) in enumerate(zip(obs['bodies'],obs['points'],obs['normals'],obs['forces'])):
            group=requirements.classify(gpu_data,int(body),point,normal)
            if group>=0 and force>1e-6:indices.append(i);groups.append(group)
        observed=dict(points=obs['points'][indices],normals=obs['normals'][indices],friction=obs['friction'])
        load=np.zeros(6)
        if indices:np.add.at(load,groups,obs['forces'][indices])
        loads['gpu'].append(load)
        obj=model.body('grasp_object').id
        closure['gpu'].append(exact_closure(observed,center=gpu_data.xpos[obj],radius=radius))
        if 'cylinder_middle_half_band' in seed:
            z=(observed['points']-gpu_data.xpos[obj])@gpu_data.xmat[obj].reshape(3,3)
            middle['gpu'].append(bool(np.any((np.asarray(groups)==2)&(abs(z[:,2])<=float(seed['cylinder_middle_half_band'])+.00001))))
        cpu_data.ctrl[:]=command;mujoco.mj_step(model,cpu_data);cpu_q.append(cpu_data.qpos.copy())
        obs_cpu=requirements.observe(cpu_data,loaded_only=True);loads['cpu'].append(obs_cpu['loads'])
        closure['cpu'].append(exact_closure(obs_cpu,center=cpu_data.xpos[obj],radius=radius))
        if 'cylinder_middle_half_band' in seed:
            z=(obs_cpu['points']-cpu_data.xpos[obj])@cpu_data.xmat[obj].reshape(3,3)
            middle['cpu'].append(bool(np.any((obs_cpu['groups']==2)&(abs(z[:,2])<=float(seed['cylinder_middle_half_band'])+.00001))))
    result={}
    for backend,states in [('gpu',qpos),('cpu',np.asarray(cpu_q))]:
        errors=(object_errors(states,reference.qpos) if np.isfinite(states).all() else
                dict(position_mm=np.full(len(states),np.inf),rotation_deg=np.full(len(states),np.inf)))
        force=np.asarray(loads[backend]);checks=closure[backend]
        finite=bool(np.isfinite(states).all() and np.isfinite(force).all())
        tracking=bool(finite and errors['position_mm'].max()<=2. and errors['rotation_deg'].max()<=5.)
        contact=bool(finite and (force[:,required]>=.05).all())
        fc=bool(all(c['success'] for c in checks))
        centered=all(middle[backend])
        result[backend]=dict(success=tracking and contact and fc and centered and duration_complete and zero_initial_velocity,
            object_hold=tracking,middle_center_contact=centered,
            required_contact_loads=contact,force_closure=fc,
            max_object_position_mm=float(errors['position_mm'].max()),max_object_rotation_deg=float(errors['rotation_deg'].max()),
            group_min_force_n=force.min(0),group_mean_force_n=force.mean(0),
            below_target_steps=(force<.05).sum(0),force_closure_passed_steps=sum(c['success'] for c in checks),
            force_closure_checks=checks)
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination/'loads.npz',gpu=loads['gpu'],cpu=loads['cpu'],cpu_qpos=cpu_q,
        required_groups=required,time=np.arange(len(controls))*model.opt.timestep)
    report=dict(success=result['gpu']['success'] and result['cpu']['success'],group_names=GROUPS,
        required_groups=required,min_force_n=.05,hold_seconds=duration,
        duration_complete=duration_complete,zero_initial_velocity=zero_initial_velocity,
        **result)
    write_json(destination/'summary.json',report)
    return report
