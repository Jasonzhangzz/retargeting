"""Bounded block LM with latent frame poses and train-only Schur diagnostics."""
from __future__ import annotations
import math,time
import numpy as np
import torch
from .qnbot_lsq_model import STAGES,SLICES

DEFAULTS=dict(max_seconds=600.,max_iterations=6,pose_iterations=2,shared_batch_size=96,
              chunk_size=48,relative_tolerance=1e-5,step_tolerance=1e-4,rank_rtol=1e-3,
              stages=list(STAGES))


def options(value=None):
    if value is None:return {**DEFAULTS,'stages':list(STAGES)}
    if not isinstance(value,dict) or set(value)-set(DEFAULTS):raise ValueError('无效的最小二乘求解选项')
    result={**DEFAULTS,**value}
    for name in ('max_seconds','relative_tolerance','step_tolerance','rank_rtol'):
        v=result[name]
        if type(v) not in (int,float) or not math.isfinite(v) or v<=0:raise ValueError('无效最小二乘选项 '+name)
    if not 1<=result['max_seconds']<=3600:raise ValueError('最小二乘优化预算须为1–3600秒')
    for name in ('max_iterations','pose_iterations','shared_batch_size','chunk_size'):
        if type(result[name]) is not int or not 1<=result[name]<=4096:raise ValueError('无效最小二乘选项 '+name)
    stages=result['stages']
    if not isinstance(stages,list) or len(set(stages))!=len(stages) or any(s not in STAGES for s in stages):raise ValueError('未知或重复最小二乘阶段')
    result['stages']=[s for s in STAGES if s in stages]
    return result


def check(cancel):
    if cancel.is_set():raise InterruptedError('最小二乘标定已取消')


def bounded_quadratic(h,gradient,basis,position,lower,upper,damping):
    """Small convex LM subproblem in the identifiable subspace.

    A single limiting coordinate must not scale every other coordinate to zero.
    SLSQP solves only this dense quadratic (at most 60 variables), never the
    hand model. All model derivatives still come from PyTorch forward AD.
    """
    from scipy.optimize import minimize,LinearConstraint
    matrix=(h+damping*torch.diag(h.diagonal().clamp_min(1.))).double().cpu().numpy()
    g=gradient.double().cpu().numpy();b=basis.double().cpu().numpy()
    lo=(lower-position).double().cpu().numpy();hi=(upper-position).double().cpu().numpy()
    result=minimize(lambda z:.5*z@matrix@z+g@z,np.zeros(len(g)),jac=lambda z:matrix@z+g,
        method='SLSQP',constraints=[LinearConstraint(b,lo,hi)],options=dict(ftol=1e-10,maxiter=150))
    feasible=np.isfinite(result.x).all() and np.all(b@result.x>=lo-1e-7) and np.all(b@result.x<=hi+1e-7)
    step=torch.tensor(result.x if feasible and result.success else np.zeros(len(g)),device=h.device,dtype=h.dtype)
    return step,dict(success=bool(feasible and result.success),status=str(result.message),iterations=int(result.nit),
                     stationary=bool(feasible and result.success and np.linalg.norm(result.x)<1e-6))


def coverage(rows,limit,iteration=0):
    """Deterministic rotating time coverage, stratified by action and window."""
    groups={}
    for i,row in enumerate(rows):groups.setdefault((row.get('stage'),row.get('window_id')),[]).append(i)
    selected=[];per=max(1,math.ceil(limit/len(groups)))
    for group in groups.values():
        n=min(len(group),per)
        positions=(np.linspace(0,len(group),n,endpoint=False).astype(int)+iteration*max(1,len(group)//(n*3)))%len(group)
        selected.extend(group[p] for p in positions)
    if len(selected)>limit:selected=[selected[i] for i in np.linspace(0,len(selected)-1,limit).astype(int)]
    return sorted(set(selected))


def schur_subspace(jg,jq,rtol=1e-3):
    """Eliminate independent latent poses WITHOUT including seed/parameter priors.

    Rows with no sensitivity to a shared coordinate cannot acquire confidence
    from a regularizer. Return only the data-supported right singular subspace.
    """
    jq=jq.double();jg=jg.double()
    u,s,_=torch.linalg.svd(jq,full_matrices=False)
    good=s>s.amax(-1,keepdim=True).clamp_min(1e-12)*rtol
    u=u*good[:,None,:]
    effective=jg-u@(u.transpose(-1,-2)@jg)
    a=effective.flatten(0,1)
    _,singular,vh=torch.linalg.svd(a,full_matrices=False)
    cutoff=max(float(singular.max())*rtol,1e-5) if singular.numel() else 1e-5
    supported=singular>cutoff
    basis=vh[supported].T.float()
    covariance=(vh[supported].T/(singular[supported].square()))@vh[supported] if bool(supported.any()) else torch.zeros((jg.shape[-1],jg.shape[-1]),device=jg.device,dtype=torch.float64)
    return basis,dict(rank=int(supported.sum()),coordinates=jg.shape[-1],singular_values=singular.cpu().tolist(),cutoff=cutoff,
        information_trace=float(a.square().sum()),conditional_std_supported_subspace=covariance.diagonal().clamp_min(0).sqrt().cpu().tolist(),
        uncertainty_scope='local_noise_normalized_linearization_after_eliminating_latent_pose; null_directions_have_no_finite_confidence',
        heldout_used=False,priors_used_for_rank=False)


@torch.no_grad()
def total_cost(o,x,q,chunk_size,contact=True):
    d=o.d;total=0.
    for start in range(0,len(q),chunk_size):
        ids=torch.arange(start,min(len(q),start+chunk_size),device=q.device)
        residual=o.frame(x,q[ids],ids,previous=q,contact=contact)
        total+=float((residual.square().sum(1)*d.weight[ids]).sum())
    if contact:total+=float(o.prior(x).square().sum()+o.collision(x,q).square().sum())
    return total


def pose_sweep(o,x,q,cancel,settings,deadline=None,contact=True,iterations=None,progress=None):
    """Parallel per-frame LM linearizations, with global backtracking.

    Temporal neighbors are frozen during a sweep. Acceptance re-evaluates the
    actual coupled objective over the entire sequence, including chunk borders.
    """
    q=q.detach().clone();d=o.d;damping=.1;steps=0
    for iteration in range(iterations or settings['pose_iterations']):
        check(cancel)
        if deadline is not None and time.monotonic()>=deadline:break
        previous=q.clone();proposal=q.clone()
        before=total_cost(o,x,q,settings['chunk_size'],contact)
        for start in range(0,len(q),settings['chunk_size']):
            check(cancel)
            if deadline is not None and time.monotonic()>=deadline:return q,dict(iterations=steps,interrupted_by_budget=True)
            ids=torch.arange(start,min(len(q),start+settings['chunk_size']),device=q.device)
            seed=previous[ids]
            def fn(delta):return o.frame(x,seed+delta,ids,previous=previous,contact=contact)
            delta=torch.zeros(25,device=q.device)
            residual=fn(delta).detach()
            jac=torch.func.jacfwd(fn)(delta).detach()
            h=jac.transpose(-1,-2)@jac;gradient=(jac.transpose(-1,-2)@residual[...,None]).squeeze(-1)
            diagonal=h.diagonal(dim1=-2,dim2=-1).clamp_min(1.)
            step=torch.linalg.solve(h+damping*torch.diag_embed(diagonal),-gradient)
            # Bound relative motion as well as anatomical limits. Existing SDK
            # out-of-range seeds may move inward but never farther outward.
            step=step.clamp(-.15,.15)
            proposal[ids]=(seed+step).maximum(d.low[ids]).minimum(d.high[ids])
            if progress and start%max(settings['chunk_size'],480)==0:progress(start+len(ids),len(q))
        accepted=False
        for fraction in (1.,.5,.25,.125):
            trial=previous+(proposal-previous)*fraction
            after=total_cost(o,x,trial,settings['chunk_size'],contact)
            if math.isfinite(after) and after<before-1e-7*max(1,abs(before)):
                q=trial.detach();accepted=True;steps+=1;damping=max(damping/2,1e-5);break
        if not accepted:damping*=10
        if float((q-previous).abs().max())<settings['step_tolerance']:break
    return q,dict(iterations=steps,interrupted_by_budget=False)


class Solver:
    def __init__(self,objective,cancel,progress,settings):
        self.o=objective;self.cancel=cancel;self.progress=progress;self.s=settings
        self.started=time.monotonic();self.deadline=self.started+settings['max_seconds'];self.trace=[];self.phases=[];self.iteration=0
        self.collision=None;self.collision_error=None

    def refresh_collision(self,x,q,selected):
        check(self.cancel)
        if self.o.m.device.type!='cuda':
            self.collision_error='训练碰撞查询需要CUDA；CPU仅用于数值测试'
            return
        from .qnbot_collision import QnbotCollision
        if self.collision is None:self.collision=QnbotCollision(self.o.m)
        self.o.witnesses=[]
        ids=[i for i in selected if bool((self.o.d.labels[i]==1).any())][:3]
        if not ids:ids=selected[:1]
        with torch.no_grad():
            for i in ids:
                v=self.o.m(q[i:i+1],geometry=self.o.c.geometry(x))[0][0]
                witness=self.collision.query(v)
                from .qnbot_surface import SOURCE_FINGERS
                from .contact import PATCH_IDS
                allowed=torch.zeros((20,20),device=q.device,dtype=torch.bool)
                for a,b,y in self.o.d.edges[i]:
                    if y!=1:continue
                    parts=[]
                    for name in (PATCH_IDS[a],PATCH_IDS[b]):
                        finger,part=name.split('_');source=SOURCE_FINGERS[('thumb','index','middle','ring','pinky').index(finger)]
                        parts.append(self.o.m.bone_names.index(self.o.m.side+'_'+source+'_'+part))
                    allowed[parts[0],parts[1]]=True;allowed[parts[1],parts[0]]=True
                witness['tolerance']=allowed[witness['source_part'],witness['target_part']].float()*.002
                self.o.witnesses.append((i,witness))

    def observable(self,x,q,indices,selected):
        ids=torch.tensor(selected,device=q.device);z=x[indices]
        jg=[];jq=[]
        for start in range(0,len(ids),self.s['chunk_size']):
            check(self.cancel);batch=ids[start:start+self.s['chunk_size']];pose=q[batch]
            fn=lambda value:self.o.measurement(x.index_copy(0,indices,value),pose,batch)
            jg.append(torch.func.jacfwd(fn)(z).detach()*self.o.d.weight[batch].sqrt()[:,None,None])
            fp=lambda delta:self.o.measurement(x,pose+delta,batch)
            jq.append(torch.func.jacfwd(fp)(torch.zeros(25,device=q.device)).detach()*self.o.d.weight[batch].sqrt()[:,None,None])
        basis,report=schur_subspace(torch.cat(jg),torch.cat(jq),self.s['rank_rtol'])
        report['linearization_frames']=selected
        report['coordinate_names']=[self.o.c.names[int(i)] for i in indices]
        leverage=basis.square().sum(1)
        report['frozen_coordinates']=[name for name,v in zip(report['coordinate_names'],leverage) if float(v)<1e-6]
        report['frozen_direction_count']=len(indices)-basis.shape[1]
        return basis,report

    def shared_step(self,x,q,indices,basis,selected,damping):
        o=self.o;ids=torch.tensor(selected,device=q.device);z=torch.zeros(basis.shape[1],device=q.device)
        def expand(value):return x.index_add(0,indices,basis@value)
        # Full training costs accept/reject steps; stratified rotating coverage
        # limits Jacobian memory. No validation or repeated sampled frame counts.
        def fn(value):
            trial=expand(value)
            measurement=o.frame(trial,q[ids],ids,previous=q)*o.d.weight[ids].sqrt()[:,None]
            # Importance correction for the unique linearization subset.
            measurement=measurement*math.sqrt(len(q)/len(ids))
            return torch.cat((measurement.flatten(),o.prior(trial),o.collision(trial,q)))
        residual=fn(z).detach();jac=torch.func.jacfwd(fn)(z).detach()
        h=jac.T@jac;g=jac.T@residual
        step,quadratic=bounded_quadratic(h,g,basis,x[indices],o.c.lower[indices],o.c.upper[indices],damping)
        direction=torch.zeros_like(x).index_add(0,indices,basis@step)
        # Line-search to the box preserves the identifiable subspace; clipping
        # individual coordinates could introduce an unsupported component.
        room=torch.where(direction>0,(o.c.upper-x)/direction.clamp_min(1e-20),(o.c.lower-x)/direction.clamp_max(-1e-20))
        fraction=min(1.,float(torch.where(direction.abs()>1e-7,room,torch.ones_like(room)).min().clamp_min(0)))
        before=total_cost(o,x,q,self.s['chunk_size'])
        for factor in (fraction,fraction*.5,fraction*.25,fraction*.125):
            proposal=x+factor*direction
            after=total_cost(o,proposal,q,self.s['chunk_size'])
            if math.isfinite(after) and after<before-1e-7*max(1,abs(before)):
                return proposal.detach(),dict(accepted=True,cost=after,previous_cost=before,step_norm=float((proposal-x).norm()),gradient_norm=float(g.norm()),damping=max(damping/2,1e-6),quadratic=quadratic)
        return x,dict(accepted=False,cost=before,previous_cost=before,step_norm=0.,gradient_norm=float(g.norm()),damping=min(damping*10,1e8),quadratic=quadratic)

    def emit(self,phase,details,q,ids):
        self.iteration+=1
        point=dict(sample_id=self.iteration,source='train_only',phase=phase,scope=phase,step=self.iteration,training_frames=len(q),linearized_frames=len(ids),
            seconds=time.monotonic()-self.started,solver='bounded_block_LM',loss=details['cost'],damping=details['damping'],accepted=details['accepted'],step_norm=details['step_norm'],gradient_norm=details['gradient_norm'],quadratic=details['quadratic'])
        self.trace.append(point)
        self.progress(dict(phase=phase,progress=.1+.6*min(1,(time.monotonic()-self.started)/self.s['max_seconds']),training_sample=point,
            message=f'最小二乘 · {phase} · 第{self.iteration}次迭代 · 阻尼 {details["damping"]:.3g}',estimated_remaining_s=max(0,self.deadline-time.monotonic())))

    def solve(self):
        o=self.o;c=o.c;d=o.d;x=torch.zeros(c.size,device=c.device);q=d.q0.clone()
        self.progress(dict(phase='pose',progress=.08,message='离线逐帧姿态初始化：有界LM，无网络',estimated_remaining_s=self.s['max_seconds']))
        q,pose_report=pose_sweep(o,x,q,self.cancel,self.s,self.deadline,contact=False,
            progress=lambda done,total:self.progress(dict(phase='pose',completed=done,total=total,message=f'离线姿态初始化 {done}/{total} 帧')))
        budget_exhausted=False
        for stage_index,stage in enumerate(self.s['stages']):
            check(self.cancel)
            if time.monotonic()>=self.deadline:budget_exhausted=True;break
            began=time.monotonic();sl=SLICES[stage];indices=torch.arange(sl.start,sl.stop,device=c.device)
            selected=coverage(d.rows,self.s['shared_batch_size'])
            self.progress(dict(phase=stage,progress=.1+.6*stage_index/max(1,len(self.s['stages'])),message='检查阶段可辨识性：'+stage,estimated_remaining_s=max(0,self.deadline-time.monotonic())))
            if stage in ('tactile','surface') and d.summary()['positive_contact_frames']<20:
                self.phases.append(dict(phase=stage,enabled=False,reason='insufficient_measured_bilateral_contact',heldout_used=False));continue
            if stage=='surface':
                with torch.no_grad():cost=float(o.measurement(x,q[torch.tensor(selected,device=q.device)],torch.tensor(selected,device=q.device))[:,30:].square().sum())
                if cost<1e-6:
                    self.phases.append(dict(phase=stage,enabled=False,reason='no_remaining_training_contact_residual',heldout_used=False));continue
                c.build_surface_basis(x)
            basis,diagnostic=self.observable(x,q,indices,selected)
            phase=dict(phase=stage,enabled=bool(basis.shape[1]),observability=diagnostic,iterations=0,converged=False,heldout_used=False)
            if not basis.shape[1]:
                phase.update(reason='no_measured_direction_after_latent_pose_elimination',seconds=time.monotonic()-began);self.phases.append(phase);continue
            damping=.1
            # Reserve optimization time for later parameter groups. Reaching a
            # stage allocation is reported, never mislabeled as convergence.
            phase_deadline=time.monotonic()+(self.deadline-time.monotonic())/(len(self.s['stages'])-stage_index)
            for iteration in range(self.s['max_iterations']):
                check(self.cancel)
                if time.monotonic()>=phase_deadline:phase['stage_budget_exhausted']=True;break
                selected=coverage(d.rows,self.s['shared_batch_size'],iteration)
                self.refresh_collision(x,q,selected)
                x,details=self.shared_step(x,q,indices,basis,selected,damping);damping=details['damping']
                self.emit(stage,details,q,selected);phase['iterations']+=1
                relative=abs(details['previous_cost']-details['cost'])/max(1,abs(details['previous_cost']))
                if (details['accepted'] and relative<self.s['relative_tolerance'] and details['step_norm']<self.s['step_tolerance']) or details['gradient_norm']<self.s['step_tolerance'] or details['quadratic']['stationary']:
                    phase['converged']=True;break
                q,_=pose_sweep(o,x,q,self.cancel,self.s,min(phase_deadline,self.deadline))
                if not details['accepted'] and damping>=1e6:phase['stalled']=True;break
            phase.update(seconds=time.monotonic()-began,parameter_vector=x[indices].cpu().tolist());self.phases.append(phase)
        budget_exhausted=budget_exhausted or time.monotonic()>=self.deadline or any(p.get('stage_budget_exhausted') for p in self.phases)
        c.install(x)
        report=dict(optimizer='bounded_block_Gauss_Newton_LM',parameterization='qnbot_lsq_coordinates_v1',
            steps=self.iteration,seconds=time.monotonic()-self.started,budget_exhausted=budget_exhausted,converged=all(p.get('converged',True) for p in self.phases if p['enabled']) and not budget_exhausted,
            phases=self.phases,initial_pose=pose_report,training_trace=self.trace,collision_error=self.collision_error,
            surface_basis=c.surface_basis_report,shared_parameter_count=c.size,trainable_vertex_parameters=False,latent_pose_count=int(q.numel()),
            latent_poses_deployed=False,heldout_used=False,parameter_snapshot=dict(coordinates=x.cpu().tolist(),coordinate_names=c.names))
        return x,q,report
