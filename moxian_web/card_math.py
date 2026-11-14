"""CARD manifold arithmetic and fixed-budget, best-seen CEM (no LM fallback)."""
from dataclasses import dataclass,asdict
import math
import torch
from retarget.initialization.dexonomy.util.torch_util import torch_matrix_to_quaternion

ALGORITHM='card_wuji2_v9_source_motion'

@dataclass(frozen=True)
class CardSettings:
    cem_backend:str='robokit'
    wc:float=40.;wfix:float=2.5;direction_price:float=2.;radius_mm:float=10.
    # Experimental bilateral closure weight; inactive in the default CARD objective.
    wpinch:float=320.
    experimental_self_contact:bool=False
    # Explicit force-cone variants are opt-in ablations; paper Eq. (3) remains default.
    contact_direction_model:str='card_alignment'
    friction_coefficient:float=1.
    normal_force_weight:float=0.
    wtip:float=3.;wn:float=60.;wb:float=20.
    # Online extension: discourage changes of CEM's joint correction relative
    # to current-source Robokit motion, without resisting observed gestures.
    wmotion_joint:float=300.
    motion_sample_fraction:float=.125
    contact_style_weight:float=1.
    wtheta:float=.1;wt:float=math.sqrt(125);womega:float=10.
    wself:float=1.5;margin_mm:float=6.;wobj_pseudo:float=3.5;wobj_mesh:float=14.
    allowance_mm:float=2.;hard_penetration_mm:float=3.
    self_penetration_mm:float=.5;wmesh:float=40.
    population:int=1024;iterations:int=16;elite_fraction:float=.05;temperature:float=.1
    translation_sigma_m:float=.002;rotation_sigma_rad:float=.016;joint_sigma_rad:float=.05
    gamma:float=.2;joint_step_rad:float=.25;translation_step_m:float=.020;seed:int=7

    def __post_init__(self):
        if self.cem_backend not in ('robokit','torch'):raise ValueError('Unknown CARD CEM backend')
        if self.population<2 or self.iterations<1 or not 0<self.elite_fraction<=1:raise ValueError('Invalid CEM budget')
        if self.temperature<=0 or not 0<self.gamma<=1:raise ValueError('Invalid annealing')
        if (not math.isfinite(self.wmotion_joint) or self.wmotion_joint<0 or not 0<=self.contact_style_weight<=1
                or not 0<=self.motion_sample_fraction<1):
            raise ValueError('Invalid online pose continuity settings')
        if self.contact_direction_model not in ('card_alignment','coulomb_force'):
            raise ValueError('Unknown contact direction model')
        if not math.isfinite(self.friction_coefficient) or self.friction_coefficient<0 or not math.isfinite(self.normal_force_weight) or self.normal_force_weight<0:
            raise ValueError('Invalid Coulomb force settings')

    def payload(self):return {**asdict(self),'elite_count':max(1,int(self.population*self.elite_fraction)),'length_cost_units':'mm','algorithm':ALGORITHM}


def exp_so3(v):
    x,y,z=v.unbind(-1);zero=torch.zeros_like(x)
    k=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(*v.shape[:-1],3,3)
    squared=v.square().sum(-1)
    angle=squared.clamp_min(1e-12).sqrt()
    # sinc's CUDA forward-AD tangent promotes to float64 on some PyTorch builds.
    # Polynomial identity branches keep JVPs finite, FP32 and accurate at zero.
    a=torch.where(squared<1e-4,1-squared/6+squared.square()/120,torch.sin(angle)/angle)[...,None,None]
    b=torch.where(squared<1e-4,.5-squared/24+squared.square()/720,(1-torch.cos(angle))/squared.clamp_min(1e-12))[...,None,None]
    return torch.eye(3,device=v.device,dtype=v.dtype)+a*k+b*(k@k)


def log_so3(r):
    # Fixed-shape quaternion candidates: no boolean indexing/nonzero in compiled
    # losses, and no acos derivative singularity at identity or pi.
    r00,r01,r02=r[...,0,0],r[...,0,1],r[...,0,2]
    r10,r11,r12=r[...,1,0],r[...,1,1],r[...,1,2]
    r20,r21,r22=r[...,2,0],r[...,2,1],r[...,2,2]
    a=torch.stack((1+r00+r11+r22,1+r00-r11-r22,1-r00+r11-r22,1-r00-r11+r22),-1).clamp_min(1e-12).sqrt()
    candidates=torch.stack((torch.stack((a[...,0].square(),r21-r12,r02-r20,r10-r01),-1),
                            torch.stack((r21-r12,a[...,1].square(),r10+r01,r02+r20),-1),
                            torch.stack((r02-r20,r10+r01,a[...,2].square(),r12+r21),-1),
                            torch.stack((r10-r01,r20+r02,r21+r12,a[...,3].square()),-1)),-2)/(2*a.clamp_min(.1)[..., :,None])
    index=a.argmax(-1)
    q=candidates.gather(-2,index[...,None,None].expand(*index.shape,1,4)).squeeze(-2)
    q=torch.where(q[...,:1]<0,-q,q)
    v=q[...,1:];length=torch.linalg.vector_norm(v,dim=-1,keepdim=True)
    angle=2*torch.atan2(length,q[...,:1].clamp_min(0))
    return v*torch.where(length>1e-7,angle/length.clamp_min(1e-12),2*torch.ones_like(length))


def rotation_mean(rotations,weights):
    mean=rotations[weights.argmax()]
    for _ in range(8):
        delta=(weights[:,None]*log_so3(mean.T@rotations)).sum(0)
        mean=mean@exp_so3(delta)
    return mean


def project(t,r,j,previous,low,high,settings):
    j=j.clamp(low,high)
    if previous is not None:
        pt,_,pj=previous
        j=torch.maximum(torch.minimum(j,pj+settings.joint_step_rad),pj-settings.joint_step_rad).clamp(low,high)
        d=t-pt;n=torch.linalg.vector_norm(d,dim=-1,keepdim=True)
        t=pt+d*torch.clamp(settings.translation_step_m/n.clamp_min(1e-12),max=1.)
    return t,r,j


def cem(initial,previous,low,high,evaluate,settings,generator,incumbents=(),verify=None):
    """Evaluate strict costs, or (objective, nonnegative constraint violation).

    Infeasible populations can reduce violation. Prefer a mesh verified
    feasible result; otherwise return this solve's least-violation finite
    candidate with feasible=False. Never substitute a historical pose.
    """
    def score(t,r,j):
        value=evaluate(t,r,j)
        if isinstance(value,tuple):
            objective,violation=value
            valid=torch.isfinite(objective)&torch.isfinite(violation)&(violation<=0)
            return torch.where(valid,objective,torch.inf),violation,objective
        return value,None,value
    fallback=None;fallback_violation=float('inf');fallback_cost=float('inf')
    def remember(t,r,j,objective,violation):
        nonlocal fallback,fallback_violation,fallback_cost
        v=violation if violation is not None else torch.zeros_like(objective)
        finite=torch.isfinite(objective)&torch.isfinite(v)
        if not finite.any():return
        minimum=v[finite].min();eligible=finite&(v==minimum)
        index=torch.where(eligible,objective,torch.inf).argmin()
        key=(float(v[index]),float(objective[index]))
        if key<(fallback_violation,fallback_cost):
            fallback_violation,fallback_cost=key
            fallback=tuple(a[index].clone() for a in (t,r,j))
    t,r,j=initial
    # Current-frame retargeting defines both the mean and refinement bounds.
    # Previous output never sets the mean or the projection reference.
    t,r,j=project(t,r,j,None,low,high,settings)
    centre=(t,r,j)
    # Re-evaluate the last accepted solution as a candidate inside the *new*
    # neighbourhood; it never sets the mean or the projection reference.
    seed_states=[centre]+([previous] if previous is not None else [])+list(incumbents)
    st=torch.stack([v[0] for v in seed_states]);sr=torch.stack([v[1] for v in seed_states]);sj=torch.stack([v[2] for v in seed_states])
    st,sr,sj=project(st,sr,sj,centre,low,high,settings)
    costs,violation,objective=score(st,sr,sj);remember(st,sr,sj,objective,violation)
    verified_count=0;verification_rejected=0
    if verify is not None:
        valid=verify(st,sr,sj);verified_count+=len(st);verification_rejected+=int((~valid).sum())
        costs=torch.where(valid,costs,torch.inf)
    initial_cost=costs[0];idx=costs.argmin();best_cost=costs[idx];best=(st[idx].clone(),sr[idx].clone(),sj[idx].clone())
    rejected=int((~torch.isfinite(costs)).sum());sigma0=j.new_tensor([settings.translation_sigma_m]*3+[settings.rotation_sigma_rad]*3+[settings.joint_sigma_rad]*len(j))
    elite_count=max(1,int(settings.population*settings.elite_fraction));evaluations=len(st);empty_restarts=0;recovery_iterations=0
    for i in range(settings.iterations):
        noise=torch.randn((settings.population,6+len(j)),device=j.device,dtype=j.dtype,generator=generator)*sigma0*settings.gamma**(i/settings.iterations)
        tt=t+noise[:,:3];rr=r@exp_so3(noise[:,3:6]);jj=j+noise[:,6:]
        tt,rr,jj=project(tt,rr,jj,centre,low,high,settings)
        cost,violation,objective=score(tt,rr,jj);remember(tt,rr,jj,objective,violation)
        evaluations+=len(cost);finite=torch.isfinite(cost);rejected+=int((~finite).sum())
        candidates=torch.argsort(cost,stable=True)
        candidates=candidates[torch.isfinite(cost[candidates])&(cost[candidates]<best_cost)][:4 if verify is not None else 1]
        if len(candidates):
            if verify is not None:
                valid=verify(tt[candidates],rr[candidates],jj[candidates]);verified_count+=len(candidates);verification_rejected+=int((~valid).sum())
                cost[candidates[~valid]]=torch.inf;candidates=candidates[valid]
            if len(candidates):
                candidate=candidates[0];best_cost=cost[candidate];best=(tt[candidate].clone(),rr[candidate].clone(),jj[candidate].clone())
        ranking=cost
        if not torch.isfinite(cost).any() and violation is not None and torch.isfinite(violation).any():
            ranking=violation;recovery_iterations+=1
        ids=torch.argsort(ranking,stable=True)[:elite_count];ids=ids[torch.isfinite(ranking[ids])]
        if not len(ids):
            if torch.isfinite(best_cost):
                t,r,j=(v.clone() for v in best);empty_restarts+=1
            continue
        scores=ranking[ids];std=scores.std(unbiased=False)
        weights=torch.softmax(-(scores-scores.mean())/std.clamp_min(1e-12)/settings.temperature,0)
        t=(weights[:,None]*tt[ids]).sum(0);j=(weights[:,None]*jj[ids]).sum(0);r=rotation_mean(rr[ids],weights)
    feasible=bool(torch.isfinite(best_cost))
    if not feasible and fallback is not None:best=fallback
    return best,{'feasible':feasible,'best_cost':float(best_cost) if feasible else None,
                 'finite_candidate_found':fallback is not None,'fallback_cost':fallback_cost if fallback is not None else None,
                 'fallback_violation':fallback_violation if fallback is not None else None,
                 'output_selection':'best_feasible_current_candidate' if feasible else 'least_violation_current_candidate' if fallback is not None else 'current_initializer_no_finite_objective',
                 'collision_scope':'candidate_endpoint_only_no_swept_path_check',
                 'initial_feasible':bool(torch.isfinite(initial_cost)),'initial_cost':float(initial_cost) if torch.isfinite(initial_cost) else None,
                 'search_center':'current_frame_retargeting','previous_used_as_search_center':False,
                 'previous_evaluated_as_incumbent':previous is not None,
                 'dense_verified_candidates':verified_count,'dense_verification_rejected':verification_rejected,
                 'empty_elite_restarts':empty_restarts,'initial_incumbents':len(incumbents),'evaluations':evaluations,'sampled_candidates':settings.population*settings.iterations,'iterations':settings.iterations,
                 'feasibility_recovery_iterations':recovery_iterations,
                 'elite_count':elite_count,'hard_rejected':rejected,'final_sigma_ratio':settings.gamma,'seed':settings.seed}


def continuous_euler_xyz(matrix,reference):
    """Nearest equivalent native RxRyRz chart; never used in optimization."""
    import warnings
    import numpy as np
    from scipy.spatial.transform import Rotation
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        angles=Rotation.from_matrix(matrix).as_euler('XYZ')
    reference=np.asarray(reference)
    choices=[angles,angles+np.array([np.pi,np.pi-2*angles[1],np.pi])]
    if abs(np.cos(angles[1]))<1e-6:
        adjusted=angles.copy();adjusted[2]=reference[2];adjusted[0]-=np.sign(angles[1])*reference[2];choices.append(adjusted)
    choices=[a+2*np.pi*np.round((reference-a)/(2*np.pi)) for a in choices]
    return min(choices,key=lambda a:np.sum((a-reference)**2))
