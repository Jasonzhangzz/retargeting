"""CARD manifold arithmetic and fixed-budget, best-seen CEM (no LM fallback)."""
from dataclasses import dataclass,asdict
import math
import torch

ALGORITHM='card_wuji2_v11_soft_collision_costs'

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
    # Retained configuration fields; nonzero values are explicit experiments.
    wmotion_joint:float=0.
    motion_sample_fraction:float=0.
    contact_style_weight:float=1.
    wtheta:float=.1;wt:float=math.sqrt(125);womega:float=10.
    wself:float=1.5;margin_mm:float=6.;wobj_pseudo:float=3.5;wobj_mesh:float=14.
    allowance_mm:float=2.;hard_penetration_mm:float=3.
    self_penetration_mm:float=.5;wmesh:float=0.
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

    @property
    def objective_profile(self):
        return ('paper_energy_soft_collision_costs' if not self.experimental_self_contact and self.contact_direction_model=='card_alignment'
                and self.wmotion_joint==0 and self.motion_sample_fraction==0 and self.contact_style_weight==1 else 'experimental')

    def payload(self):return {**asdict(self),'elite_count':max(1,int(self.population*self.elite_fraction)),'length_cost_units':'mm','algorithm':ALGORITHM,'objective_profile':self.objective_profile,'collision_hard_gates':False}


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


def axis_angle_near(rotation,reference=None):
    """Equivalent axis-angle vector nearest a reference (including full turns)."""
    vector=log_so3(rotation)
    if reference is None:return vector
    length=vector.norm(dim=-1,keepdim=True)
    ref_length=reference.norm(dim=-1,keepdim=True)
    axis=torch.where(length>1e-7,vector/length.clamp_min(1e-12),reference/ref_length.clamp_min(1e-12))
    turns=torch.round(((reference-vector)*axis).sum(-1,keepdim=True)/(2*math.pi))
    return vector+2*math.pi*turns*axis


def elite_weights(cost,elite_count,temperature):
    """Eq. (13), finite elites only; empty populations keep their mean."""
    ids=torch.argsort(torch.where(torch.isfinite(cost),cost,torch.inf),stable=True)[:elite_count]
    ids=ids[torch.isfinite(cost[ids])]
    scores=cost[ids]
    if not len(ids):return ids,scores
    scale=scores.std(unbiased=False).clamp_min(1e-12)*temperature
    return ids,torch.softmax(-(scores-scores.min())/scale,0)


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


def retained_elites(values, costs, count):
    """Stable best finite distinct candidates; never change the sampler/RNG."""
    values, costs = torch.cat(values), torch.cat(costs)
    finite = torch.isfinite(costs) & torch.isfinite(values).all(-1)
    ids = torch.nonzero(finite).flatten()
    ids = ids[torch.argsort(costs[ids], stable=True)]
    selected = []
    seen = set()
    for i in ids.tolist():
        key = tuple(torch.round(values[i].detach().cpu() * 1e6).long().tolist())
        if key not in seen:
            selected.append(i); seen.add(key)
            if len(selected) >= count: break
    indices = torch.as_tensor(selected, device=values.device, dtype=torch.long)
    return dict(values=values[indices].detach(), costs=costs[indices].detach())


@torch.no_grad()
def joint_elite_cem(centers, low, high, evaluate, settings, generator, *, check=lambda: None, constrain=None):
    """Fixed-wrist, multimodal local CEM for CART's offline tube.

    Unlike the paper CARD path, infeasible candidates may guide recovery but
    NEVER enter the returned archive. No full-domain random poses are drawn.
    ``constrain`` audits nominees inside each generation, before the mean update
    and archive insertion. It may change costs and remove infeasible candidates.
    """
    if centers.ndim != 2 or centers.shape[1] != len(low) or not len(centers):
        raise ValueError('Invalid local CEM centers')
    roots = centers.clamp(low, high).clone(); means = roots.clone()
    count = max(1, int(settings.population * settings.elite_fraction))
    values=[]; costs=[]; evaluated=0; generations=[]
    best_recovery=means[0]; best_cost=torch.tensor(torch.inf,device=means.device)
    def record(q):
        nonlocal evaluated,best_cost,best_recovery
        cost, feasible = evaluate(q); evaluated += len(q)
        cost=torch.where(torch.isfinite(cost),cost,torch.inf)
        if constrain is not None:
            cost,feasible=constrain(q,cost,feasible)
            cost=torch.where(torch.isfinite(cost),cost,torch.inf)
        feasible=feasible & torch.isfinite(cost)
        generations.append(int(feasible.sum()))
        values.append(q.clone()); costs.append(torch.where(feasible,cost,torch.inf))
        idx=cost.argmin()
        if cost[idx]<best_cost:best_cost=cost[idx].clone();best_recovery=q[idx].clone()
        return cost,feasible
    record(means)
    assignment=torch.arange(settings.population,device=means.device)%len(means)
    local_low=torch.maximum(low,roots-settings.joint_step_rad)
    local_high=torch.minimum(high,roots+settings.joint_step_rad)
    for iteration in range(settings.iterations):
        check()
        sigma=settings.joint_sigma_rad*settings.gamma**(iteration/settings.iterations)
        q=means[assignment]+torch.randn((settings.population,len(low)),device=means.device,generator=generator)*sigma
        q=q.clamp(local_low[assignment],local_high[assignment]); score,feasible=record(q)
        for branch in range(len(means)):
            ids=torch.nonzero(assignment==branch).flatten()
            branch_score=score[ids]
            if constrain is not None and feasible[ids].any():
                branch_score=torch.where(feasible[ids],branch_score,torch.inf)
            selected,weights=elite_weights(branch_score,max(1,int(len(ids)*settings.elite_fraction)),settings.temperature)
            if len(selected):means[branch]=(weights[:,None]*q[ids[selected]]).sum(0)
    record(means)
    archive=retained_elites(values,costs,count)
    archive.update(recovery=best_recovery, recovery_cost=float(best_cost), evaluations=evaluated,
                   local_only=True, centers=roots, joint_radius_rad=settings.joint_step_rad,
                   constraints_in_cem=constrain is not None,generation_feasible=generations,
                   device=str(means.device),gpu_parallel=means.device.type=='cuda')
    return archive


def cem(initial,previous,low,high,evaluate,settings,generator,incumbents=(),verify=None,*,elite_archive=None):
    """Paper CEM: fixed Gaussian schedule, strict constraints, best seen.

    ``verify`` is a diagnostic callback only. Tuple costs are accepted for
    compatibility, but constraint violation never becomes an optimization cost.
    """
    reference=axis_angle_near(previous[1]) if previous is not None else None
    def pack(state):
        t,r,j=state
        return torch.cat((t,axis_angle_near(r,reference),j),-1)
    def unpack(x):return x[...,:3],exp_so3(x[...,3:6]),x[...,6:]
    def projection(x):
        t,r,j=project(*unpack(x),previous,low,high,settings)
        return torch.cat((t,x[...,3:6],j),-1)
    def score(x):
        value=evaluate(*unpack(x))
        if isinstance(value,tuple):
            objective,violation=value
            value=torch.where(torch.isfinite(violation)&(violation<=0),objective,torch.inf)
        return torch.where(torch.isfinite(value),value,torch.inf)
    current=pack(initial)
    mean=projection((current+pack(previous))*.5 if previous is not None else current)
    seeds=[mean]+([pack(previous)] if previous is not None else [])
    # Explicit experimental seeds remain supported; production supplies none.
    seeds.extend(projection(pack(state)) for state in incumbents)
    xs=torch.stack(seeds);cost=score(xs);evaluations=len(xs)
    archive_values=[xs.clone()] if elite_archive is not None else []
    archive_costs=[cost.clone()] if elite_archive is not None else []
    index=cost.argmin();best_cost=cost[index].clone();best_x=xs[index].clone()
    previous_cost=cost[1].clone() if previous is not None else None
    initial_cost=cost[0].clone();rejected=int((~torch.isfinite(cost)).sum());empty=0
    sigma0=mean.new_tensor([settings.translation_sigma_m]*3+[settings.rotation_sigma_rad]*3+[settings.joint_sigma_rad]*len(initial[2]))
    elite_count=max(1,int(settings.population*settings.elite_fraction))
    for i in range(settings.iterations):
        noise=torch.randn((settings.population,len(mean)),device=mean.device,dtype=mean.dtype,generator=generator)
        xs=projection(mean+noise*sigma0*settings.gamma**(i/settings.iterations))
        cost=score(xs);evaluations+=len(xs);rejected+=int((~torch.isfinite(cost)).sum())
        if elite_archive is not None:
            archive_values.append(xs.clone());archive_costs.append(cost.clone())
        index=cost.argmin()
        if cost[index]<best_cost:best_cost=cost[index].clone();best_x=xs[index].clone()
        ids,weights=elite_weights(cost,elite_count,settings.temperature)
        if len(ids):mean=(weights[:,None]*xs[ids]).sum(0)
        else:empty+=1
    # The final updated mean is also evaluated, as in the population backend.
    final_cost=score(mean[None])[0];evaluations+=1;rejected+=int(not torch.isfinite(final_cost))
    if elite_archive is not None:
        archive_values.append(mean[None].clone());archive_costs.append(final_cost[None].clone())
        elite_archive.update(retained_elites(archive_values,archive_costs,elite_count))
    if final_cost<best_cost:best_cost=final_cost.clone();best_x=mean.clone()
    feasible=bool(torch.isfinite(best_cost))
    best=unpack(best_x if feasible else projection(current))
    diagnostic_valid=None
    if verify is not None:diagnostic_valid=bool(verify(*(v[None] for v in best))[0])
    return best,dict(feasible=feasible,best_cost=float(best_cost) if feasible else None,
        finite_candidate_found=feasible,fallback_cost=None,fallback_violation=None,
        output_selection='best_feasible_current_candidate' if feasible else 'current_initializer_no_finite_objective',
        output_selection_scope='all_scored_candidates',collision_scope='candidate_endpoint_only_no_swept_path_check',
        initial_feasible=bool(torch.isfinite(initial_cost)),initial_cost=float(initial_cost) if torch.isfinite(initial_cost) else None,
        search_center='half_previous_half_initializer' if previous is not None else 'current_frame_retargeting',
        previous_used_as_search_center=previous is not None,previous_evaluated_as_incumbent=previous is not None,
        previous_incumbent_cost=float(previous_cost) if previous_cost is not None and torch.isfinite(previous_cost) else None,
        dense_verified_candidates=0,dense_verification_rejected=0,mesh_diagnostic_satisfied=diagnostic_valid,
        empty_elite_restarts=0,empty_elite_iterations=empty,initial_incumbents=len(incumbents),evaluations=evaluations,
        sampled_candidates=settings.population*settings.iterations,iterations=settings.iterations,feasibility_recovery_iterations=0,
        elite_count=elite_count,hard_rejected=rejected,final_sigma_ratio=settings.gamma,
        last_sample_sigma_ratio=settings.gamma**((settings.iterations-1)/settings.iterations),seed=settings.seed)


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
