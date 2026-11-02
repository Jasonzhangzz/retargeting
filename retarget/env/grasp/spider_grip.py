"""Short native MPPI load trim while the acquisition object remains fixed."""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import torch
import mujoco
import warp as wp
from retarget.solver.spider_mppi import optimize
from retarget.common.results import write_json
from .spider_physics import BatchedPhysics
from .spider_geometry import NativeContactGeometry,ContactRequirements,exact_closure
from .spider_seed import load_seed
from .spider_reference import save_acquired_reference,read_input
from retarget.env.geometry import ObjectMeshSurface


def trim_grip(reference,model_path,command,destination,solver,device='cuda:0'):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    # The fixed model was already serialized by the physical grip establishment.
    fixed_path=destination.parent/'fixed_object.xml'
    model=mujoco.MjModel.from_xml_path(str(fixed_path));model.opt.iterations=20
    W=128;steps=20
    physics=BatchedPhysics(model,W,device,capture=True)
    _,_,seed,_=load_seed(reference.seed_source)
    geometry=NativeContactGeometry(physics,seed)
    tensor=lambda x,dtype=torch.float32:torch.as_tensor(x,device=device,dtype=dtype)
    qinitial=tensor(reference.qpos[:26]);initial=tensor(command)
    required=tensor(reference.required_groups,torch.bool)
    mass=float(model.body_mass[model.body('grasp_object').id]);gravity=np.asarray(model.opt.gravity)
    goal=tensor(np.r_[-mass*gravity,np.zeros(3)])
    scale=tensor([max(mass*np.linalg.norm(gravity),.1)]*3+[max(mass*np.linalg.norm(gravity)*geometry.radius,.001)]*3)
    physics.reset(qinitial,torch.zeros(26,device=device),control=initial);physics.warmup();state=physics.snapshot()
    elite=None;elite_cost=None
    def evaluate(controls):
        nonlocal elite,elite_cost
        result=[]
        for start in range(0,len(controls),W):
            u=physics.project_ctrl(controls[start:start+W,0]);n=len(u)
            if n<W:u=torch.cat((u,u[:1].expand(W-n,-1)))
            physics.restore(state)
            total=torch.zeros(W,device=device);worst=torch.zeros_like(total)
            for step in range(steps):
                physics.step(u)
                load=geometry.sample()
                shortage=torch.relu(1-load[:,required]/.3).square().sum(1)
                worst=torch.maximum(worst,shortage)
                balance=((wp.to_torch(geometry.wrench)-goal)/scale).square().sum(1)
                penetration=(torch.relu(wp.to_torch(geometry.depth)-.0005)/.001).square()
                total+=shortage+.5*balance+5*penetration
            total=total/steps+2*worst+.005*((u-initial)/tensor([.002]*3+[.02]*3+[.1]*20)).square().mean(1)
            total+=.2*geometry.wrench_cost()
            total=torch.where(torch.isfinite(total)&(wp.to_torch(physics.data.overflow)==0),total,torch.inf)
            result.append(total[:n].clone())
            candidates=u[:n].clone();costs=total[:n].clone()
            if elite is not None:candidates=torch.cat((elite,candidates));costs=torch.cat((elite_cost,costs))
            ids=torch.argsort(costs)[:8];elite=candidates[ids];elite_cost=costs[ids]
        return torch.cat(result)
    noise=tensor([.0005]*3+[.005]*3+[.025]*20)
    params=replace(solver,population=128,worlds=128,iterations=4,improvement_check_steps=3)
    with torch.no_grad():
        mean,report=optimize(initial[None],evaluate,params,knots=1,knot_steps=1,
            channel_noise=noise,seed=17,first_noise=1.,last_noise=1.,return_best=True)
    free=mujoco.MjModel.from_xml_path(str(model_path));free_data=mujoco.MjData(free)
    _,metadata,seed,_=load_seed(reference.seed_source);requirements=ContactRequirements(free,seed)
    obj=free.body('grasp_object').id
    radius=max(float(np.linalg.norm(ObjectMeshSurface.from_model(free,obj).triangles.reshape(-1,3),axis=1).max()),.01)
    candidates=torch.cat((mean,elite)).cpu().numpy()
    best=None;audits=[]
    free_data.qpos[:]=reference.qpos;free_data.qvel[:]=0.;free_data.ctrl[:]=command;mujoco.mj_forward(free,free_data)
    baseline=requirements.observe(free_data,loaded_only=True)
    baseline_fc=exact_closure(baseline,center=free_data.xpos[obj],radius=radius)
    if (baseline['loads'][reference.required_groups]>=.05).all() and baseline_fc['success']:
        score=float(np.linalg.norm(free_data.qacc[26:29])+radius*np.linalg.norm(free_data.qacc[29:32]))
        best=(score,np.asarray(command).copy(),reference.qpos[:26].copy(),baseline)
        audits.append(dict(index=-1,loaded=True,loads_n=baseline['loads'],force_closure=baseline_fc,free_acceleration=score))
    for index,u in enumerate(candidates):
        data=mujoco.MjData(model);data.qpos[:]=reference.qpos[:26];data.qvel[:]=0.;data.ctrl[:]=u
        for _ in range(steps):mujoco.mj_step(model,data)
        free_data.qpos[:]=np.r_[data.qpos,reference.qpos[26:]];free_data.qvel[:]=0.;free_data.ctrl[:]=u
        free_data.qacc_warmstart[:]=0.;mujoco.mj_forward(free,free_data)
        observed=requirements.observe(free_data,loaded_only=True)
        fc=exact_closure(observed,center=free_data.xpos[obj],radius=radius)
        loaded=bool((observed['loads'][reference.required_groups]>=.05).all())
        score=float(np.linalg.norm(free_data.qacc[26:29])+radius*np.linalg.norm(free_data.qacc[29:32]))
        audits.append(dict(index=index,loaded=loaded,loads_n=observed['loads'],force_closure=fc,free_acceleration=score))
        if loaded and fc['success'] and (best is None or score<best[0]):
            best=(score,u.copy(),data.qpos.copy(),observed)
    outcome=dict(optimizer=report,object_fixed=True,horizon_steps=steps,dt=model.opt.timestep,
        cost='native group load deficits, gravity wrench balance, penetration, physical wrench support',candidates=audits,accepted=best is not None)
    if best is not None:
        score,u,q,observed=best
        free_data.qpos[:]=np.r_[q,reference.qpos[26:]];free_data.qvel[:]=0.;free_data.ctrl[:]=u;mujoco.mj_forward(free,free_data)
        objective=SimpleNamespace(cpu=free,source=reference.seed_source,seed=seed,metadata=metadata)
        out=save_acquired_reference(destination/'reference',objective,q,observed,free_data)
        outcome.update(reference=str(out),free_acceleration=score)
        write_json(destination/'trim.json',outcome)
        return read_input(out),u.astype(np.float32),outcome
    write_json(destination/'trim.json',outcome)
    return reference,np.asarray(command,np.float32),outcome
