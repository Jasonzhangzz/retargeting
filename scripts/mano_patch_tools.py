"""Utilities for fixed-input current patch-CEM diagnostics."""
import hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from retarget.env.grasp.mano_solver import (ManoDecoder,FixedContacts,SurfaceBatch,MeshCache,ShadowSkeleton,Settings,initialize,lm_ik,pose_problem)
from retarget.solver.cem import SamplingSettings as CemSettings
from retarget.solver.cem import solve_patch_cem

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,list):return [clean(x) for x in v]
    if isinstance(v,float) and not np.isfinite(v):return None
    return v


def write(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def now():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())


def sync():torch.cuda.synchronize();return time.perf_counter()


def competition():
    try:return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],text=True,timeout=5).strip()
    except Exception as e:return str(e)


def lm_anchors(problem,adapter,steps=64):
    """Pure point-to-fixed-anchor kinematics, analytic MANO Jacobian, bounded LM.

    All slots contribute, with each anatomical group normalized separately.
    No surface, normal, collision, rollout or pose-prior residual is used.
    """
    q=problem.mean.clone();scale=problem.sigma;anchors=adapter.anchors
    weights=q.new_zeros(len(adapter.fixed.records))
    for ids in adapter.required_groups:weights[ids]=1/math_sqrt(len(ids))
    def residual(u,target):
        vertices,_,_=adapter.decoder((u*scale[0])[None])
        hand,_=adapter.fixed.evaluate(vertices)
        return ((hand[0]-target)*weights[:,None]/.002).flatten()
    jac=torch.func.vmap(torch.func.jacfwd(residual,argnums=0))
    fn=torch.func.vmap(residual)
    u=q/scale;d=q.shape[-1];damping=q.new_full((len(q),),.01)
    low=problem.project((q-1.e6)[:,None])[:,0]/scale
    high=problem.project((q+1.e6)[:,None])[:,0]/scale
    eye=torch.eye(d,device=q.device);accepted=torch.zeros(len(q),device=q.device,dtype=torch.long)
    t=sync()
    with torch.no_grad():cost=fn(u,anchors).square().sum(-1)
    for iteration in range(steps):
        with torch.enable_grad():j=jac(u,anchors).detach()
        with torch.no_grad():
            r=fn(u,anchors);jt=j.transpose(-1,-2);h=jt@j;g=(jt@r[...,None]).squeeze(-1)
            blocked=((u<=low+1.e-5)&(g>0))|((u>=high-1.e-5)&(g<0))|(low==high)
            free=(~blocked).to(q.dtype)
            h=h*free[:,:,None]*free[:,None,:]+torch.diag_embed(blocked.to(q.dtype))
            g=g*free
            original=u.clone();old=cost.clone();pending=torch.ones(len(q),device=q.device,dtype=torch.bool)
            for retry in range(4):
                delta,info=torch.linalg.solve_ex(h+damping[:,None,None]*torch.diag_embed(h.diagonal(dim1=-2,dim2=-1).clamp_min(1.e-4))+eye*1.e-7,-g)
                finite=(info==0)&torch.isfinite(delta).all(-1)
                delta=torch.nan_to_num(delta);delta*= (3/delta.norm(dim=-1).clamp_min(3))[:,None]
                candidate=problem.project(((original+delta)*scale)[:,None])[:,0]/scale
                value=fn(candidate,anchors).square().sum(-1)
                good=pending&finite&torch.isfinite(value)&(value<old)
                u=torch.where(good[:,None],candidate,u);cost=torch.where(good,value,cost)
                accepted+=good.long();damping=torch.where(good,damping*.5,torch.where(pending,damping*4,damping)).clamp(1.e-7,1.e8)
                pending &= ~good
    return dict(parameters=(u*scale).detach(),solve_seconds=sync()-t,accepted=accepted,kinematic_cost=cost,
                iterations=torch.full_like(accepted,steps),evaluations=torch.full_like(accepted,steps*4+1))


def math_sqrt(x):return float(x)**.5


def serialize_metrics(parts,adapter,solver=None):
    p={k:v.detach().cpu().numpy() for k,v in parts.items()};result=[]
    for i in range(len(p['valid'])):
        finger={}
        for j,(name,ids) in enumerate(zip(adapter.group_names,adapter.required_groups)):
            ix=ids.cpu().numpy()
            finger[name]=dict(passed=bool(p['patch_group_fitted'][i,j]),coverage=float(p['patch_fitted'][i,ix].mean()),
                min_patch_gap_mm=float(p['patch_gap_m'][i,ix].min()*1000),
                min_fixed_point_anchor_mm=float(p['anchor_distance_m'][i,ix].min()*1000))
        depth=p['penetration_depth_m'][i]*1000
        reasons=[]
        if not p['valid'][i]:reasons.append('nonfinite_or_collision_overflow')
        if not p['rollout_stable'][i]:reasons.append('five_step_velocity' if adapter.rollout_settings.cost_mode=='velocity' else 'five_step_motion')
        if p['patch_coverage'][i]+1.e-7<.8:reasons.append('patch_coverage_below_80_percent')
        reasons += ['missing_'+k for k,v in finger.items() if k!='palm' and not v['passed']]
        if depth[0]>5.00001:reasons.append('hand_object_penetration_above_5mm')
        if depth[1]>5.00001:reasons.append('self_penetration_above_5mm')
        m=dict(passed=bool(p['patch_joint_accepted'][i]),geometry=bool(p['patch_geometry_accepted'][i]),stable=bool(p['rollout_stable'][i]),
            valid=bool(p['valid'][i]),coverage=float(p['patch_coverage'][i]),all_fingers=bool(p['patch_all_required_fingers'][i]),
            fixed_anchor_rmse_mm=float(np.sqrt(np.mean(p['anchor_distance_m'][i]**2))*1000),
            anchor_cost=float(p['anchor_cost'][i]),total_cost=float(p['total_cost_v3'][i]),rollout_cost=float(p['rollout_cost'][i]),
            penetration_cost=float(p['penetration'][i]*1.e4),soft_penetration_cost=float(p['soft_penetration'][i]*.1),
            penetration_mm=depth.tolist(),displacement_mm=float(p['rollout_displacement_m'][i]*1000),
            rotation_deg=float(np.rad2deg(p['rollout_rotation_rad'][i])),native_contacts=int(p['collision_contact_count'][i]),
            end_linear_speed_m_s=float(np.linalg.norm(p['rollout_linear_velocity'][i])),
            end_angular_speed_rad_s=float(np.linalg.norm(p['rollout_angular_velocity'][i])),
            fingers=finger,failure_reasons=reasons)
        if solver:
            for key in ['iterations','evaluations','accepted','early_stopped']:
                if key in solver:m[key]=solver[key][i].item()
        result.append(m)
    return result


def save_result(folder,q,adapter,rows,solver=None,extra=None):
    folder.mkdir(parents=True,exist_ok=True);t=sync();_,parts=adapter.validate(q,rows)
    _,joints,_=adapter.decoder(q,full=True);acceptance=sync()-t
    m=serialize_metrics(parts,adapter,solver)
    arrays=dict(variables=q,mano_joints=joints,**parts)
    t=time.perf_counter();np.savez_compressed(folder/'final.npz',**{k:v.detach().cpu().numpy() for k,v in arrays.items()})
    write(folder/'metrics.json',m)
    report=dict(samples=len(q),passed=sum(x['passed'] for x in m),geometry=sum(x['geometry'] for x in m),
        stable=sum(x['stable'] for x in m),all_fingers=sum(x['all_fingers'] for x in m),acceptance_seconds=acceptance,
        median_anchor_rmse_mm=float(np.median([x['fixed_anchor_rmse_mm'] for x in m])),save_seconds=time.perf_counter()-t,**(extra or {}))
    write(folder/'report.json',report);write(folder/'complete.json',dict(final_sha256=sha(folder/'final.npz'),finished=now()))
    print(json.dumps(dict(folder=str(folder),**report)),flush=True)
    return report,m
