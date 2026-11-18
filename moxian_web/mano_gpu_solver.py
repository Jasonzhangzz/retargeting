"""CUDA Adam and matrix-free bounded LM, with train-only convergence checks."""
import time
import torch


def bounded_lm_gpu(objective, initial, cancel, deadline, iterations=8):
    from torch.func import jvp
    x=initial.detach().clone();damping=.05;accepted=0
    # One shared block and all independent frame blocks solved concurrently.
    blocks=[torch.arange(len(x),device=x.device)<objective.shared_size,
            torch.arange(len(x),device=x.device)>=objective.shared_size] if objective.shared else [torch.ones_like(x,dtype=torch.bool)]
    for _ in range(iterations):
        if cancel.is_set():raise InterruptedError('GPU LM 已取消')
        if time.monotonic()>=deadline:break
        objective.refresh(x)
        for mask in blocks:
            if hasattr(objective,'frozen_shared'):mask=mask & ~objective.frozen_shared
            if not bool(mask.any()):continue
            z=x.detach().requires_grad_(True);r=objective.residual(z)
            cost=r.detach().square().sum()
            g=torch.autograd.grad(r,z,r.detach(),retain_graph=True)[0]*mask
            def normal(p):
                _,jp=jvp(lambda v:objective.residual(v),(z.detach(),),(p*mask,))
                return torch.autograd.grad(r,z,jp.detach(),retain_graph=True)[0]*mask+damping*p
            direction=torch.zeros_like(x);res=-g;search=res.clone();rr=res@res
            for _cg in range(6):
                ap=normal(search);alpha=rr/(search@ap).clamp_min(1e-12)
                direction=direction+alpha*search;res=res-alpha*ap
                nxt=res@res;search=res+nxt/rr.clamp_min(1e-12)*search;rr=nxt
            proposal=(x+direction*mask).clamp(objective.lower,objective.upper)
            if getattr(objective,'fixed_forward',False):
                from .mano_shape_prior import project
                proposal=project(proposal,objective.lower,objective.upper,objective.shared_size)
            with torch.no_grad():new=objective.residual(proposal).square().sum()
            if bool(torch.isfinite(new)&(new<cost)):
                x=proposal.detach();accepted+=1;damping=max(damping*.5,1e-5)
            else:damping=min(damping*4,1e5)
            del r,z
    return x,dict(accepted_steps=accepted,method='matrix_free_bounded_block_LM_CG',iterations=iterations)


def solve_calibration_gpu(objective,cancel,progress,max_seconds=30,resume_phases=None,compile_loss=True):
    if getattr(objective,'mixed_training',False):
        from .mano_mixed_training import solve_mixed
        return solve_mixed(objective,cancel,progress,max_seconds,compile_loss,flash=False)
    from .mano_objective import SOLVER_REVISION
    from .mano_local_geometry import PairLocalRefinement,phase_stages,BLOCK_STEPS
    if any(row.get('partition','train')!='train' for row in objective.rows):
        raise ValueError('GPU 几何辨识只能使用训练分区，留出数据不能参与参数更新')
    began=time.monotonic();deadline=began+max_seconds
    x=torch.nn.Parameter(objective.initial.clone());optimizer=torch.optim.Adam([x],lr=.01)
    original=objective.weights.clone();reports=[];best=x.detach().clone()
    from .mano_training_metrics import TrainingMetrics
    metrics=TrainingMetrics(objective,progress)
    loss_fn=lambda z:objective.residual(z).square().sum()
    # Refresh BVHs outside captured graphs; witnesses have fixed padded shapes.
    compiled=torch.compile(loss_fn,options={'triton.cudagraphs':False},dynamic=False) if compile_loss else loss_fn
    frozen_geometry=[];observability={};exhausted=False;compile_seconds=0.;local=None
    localized=getattr(objective,'fixed_forward',False) and objective.personalized
    for phase_index,phase in enumerate(('reference','geometry','pinches')):
        if cancel.is_set():raise InterruptedError('GPU 标定已取消')
        if localized:
            included=phase_stages(objective.rows,phase)
            gate=objective.t([r['stage'] in included for r in objective.rows])
        elif phase_index==0:
            included={'open','together','mcp90','hook'}
            gate=objective.t([r['stage'] in included for r in objective.rows])
        else:included={r['stage'] for r in objective.rows};gate=torch.ones(len(objective.rows),device=x.device)
        objective.phase_stages=included;objective.weights=original*gate[:,None];objective.weights/=objective.weights.sum().clamp_min(1e-12)
        frozen=torch.zeros_like(x,dtype=torch.bool)
        if objective.personalized:
            if phase_index==0:frozen[46:objective.shared_size]=True
            else:
                frozen[:10]=True
                if not observability:
                    from .mano_calibrate_flashsac import _geometry_observability
                    frozen_geometry,observability=_geometry_observability(objective,x)
                if frozen_geometry:frozen[torch.tensor(frozen_geometry,device=x.device)+46]=True
        fixed=x.detach().clone();best_cost=float('inf');stale=0;previous=None;phase_start=time.monotonic();ema=fixed.clone()
        share=(.20,.3125,1.)[phase_index] if localized else 1/(3-phase_index)
        phase_deadline=time.monotonic()+max(0.,deadline-time.monotonic())*share
        if localized and phase=='pinches':
            local=PairLocalRefinement(objective,x,frozen,original)
            frozen=local.frozen;fixed=local.fixed
        completed=0
        # Each phase has distinct static gates and compiled graph.
        cycle=BLOCK_STEPS*len(local.stages) if local else 1
        max_steps=(objective.settings.adam_steps//cycle)*cycle if local and objective.settings.adam_steps>=cycle else objective.settings.adam_steps
        for step in range(max_steps):
            if cancel.is_set():raise InterruptedError('GPU 标定已取消')
            # Finish a balanced cycle before checking the budget or convergence.
            boundary=not local or step%(BLOCK_STEPS*len(local.stages))==0
            if step and boundary and time.monotonic()>=phase_deadline:exhausted=True;break
            if local and step%BLOCK_STEPS==0:
                local.select(step//BLOCK_STEPS,x,ema)
                optimizer.state.clear()
            if step%objective.settings.collision_refresh==0:objective.refresh(x.detach())
            if step==0:metrics.emit(x.detach(),ema,phase,0,local,force=True,advance=0,full_replay=bool(local))
            optimizer.zero_grad(set_to_none=True)
            before=time.monotonic()
            if step==0:progress(dict(state='running',phase='compile',progress=.08+.21*phase_index,message=f'GPU 编译／预热阶段 {phase_index+1}',estimated_remaining_s=None))
            loss=compiled(x);loss.backward()
            if cancel.is_set():raise InterruptedError('GPU 标定已取消')
            if step==0 and compile_loss:
                torch.cuda.synchronize();overhead=time.monotonic()-before;compile_seconds+=overhead;deadline+=overhead;phase_deadline+=overhead
            if not bool(torch.isfinite(loss)) or x.grad is None or not bool(torch.isfinite(x.grad).all()):raise ValueError('GPU 标定损失／梯度非有限')
            x.grad[frozen]=0;torch.nn.utils.clip_grad_norm_([x],100.)
            value=float(loss.detach())
            if value<best_cost:best_cost=value;best=x.detach().clone()
            optimizer.step()
            with torch.no_grad():
                x.clamp_(objective.lower,objective.upper);x[frozen]=fixed[frozen]
                if getattr(objective,'fixed_forward',False):
                    from .mano_shape_prior import project,EMA_DECAY
                    x.copy_(project(x,objective.lower,objective.upper,objective.shared_size))
                    x[frozen]=fixed[frozen]
                    ema.copy_(local.update_ema(ema,x) if local else objective.shape_prior.ema_update(ema,x));ema[frozen]=fixed[frozen]
                    if local:x.copy_(ema)
            completed+=1
            metrics.emit(x.detach(),ema,phase,completed,local)
            if local and (step+1)%BLOCK_STEPS==0:local.record_block(ema,BLOCK_STEPS)
            if step%20==0:
                if not local:
                    relative=abs(previous-value)/max(abs(previous),1e-8) if previous is not None else float('inf')
                    stale=stale+1 if relative<1e-4 else 0;previous=value
                progress(dict(state='running',phase=phase,progress=.08+.63*(phase_index+(step+1)/max_steps)/3,message=f'GPU 参数辨识 {phase_index+1}/3 · {local.message()+" · " if local else ""}{step+1} 步',estimated_remaining_s=max(0.,deadline-time.monotonic()),eta_scope='optimization_budget'))
                if not local and step>=80 and stale>=3:break
            if local and (step+1)%(BLOCK_STEPS*len(local.stages))==0:
                # Compare the same full training replay objective each cycle,
                # never costs of different pairs and never held-out rows.
                cycle_cost=local.cycle_cost(ema)
                metrics.emit(x.detach(),ema,phase,completed,local,force=True,advance=0,full_replay=True)
                relative=abs(previous-cycle_cost)/max(abs(previous),1e-8) if previous is not None else float('inf')
                stale=stale+1 if relative<1e-4 else 0;previous=cycle_cost
                if stale>=3:break
        metrics.emit(x.detach(),ema,phase,completed,local,force=True,advance=0,full_replay=bool(local))
        with torch.no_grad():x.copy_(ema if getattr(objective,'fixed_forward',False) else best)
        optimizer.state.clear();reports.append(dict(phase=phase,steps=completed,state='budget_exhausted' if exhausted else 'converged' if stale>=3 else 'completed',elapsed_s=time.monotonic()-phase_start))
        # Every stage receives a share, even when an earlier stage reaches its budget.
    objective.weights=original
    local_report=local.report() if local else None
    if local:local.close()
    if hasattr(objective,'phase_stages'):del objective.phase_stages
    objective.refresh(x.detach())
    progress(dict(state='running',progress=.74,message='冻结参数与局部表面 EMA，准备前向验收' if getattr(objective,'fixed_forward',False) else 'GPU 有界分块 LM 精修',estimated_remaining_s=None))
    objective.frozen_shared=torch.zeros_like(x,dtype=torch.bool)
    if objective.personalized:
        objective.frozen_shared[:10]=True
        if frozen_geometry:objective.frozen_shared[torch.tensor(frozen_geometry,device=x.device)+46]=True
    refined,lm=bounded_lm_gpu(objective,x.detach(),cancel,deadline,objective.settings.lm_steps) if not exhausted and not getattr(objective,'fixed_forward',False) else (x.detach(),dict(accepted_steps=0,reason='frozen_EMA_artifact' if getattr(objective,'fixed_forward',False) else 'budget'))
    objective.refresh(refined);objective.install(refined);objective.residual(refined,record=True)
    from .mano_contact_loss import loss_spec,gradient_report
    gradients=gradient_report(objective,refined)
    return dict(revision=SOLVER_REVISION,phases=reports,lm=lm,budget_exhausted=exhausted or time.monotonic()>=deadline,
                collision_witness_complete=not bool(objective.gpu_witness['witness_overflow'].any()),compilation_s=compile_seconds,compute_device=str(x.device),observability=observability,cost_terms=objective.last_terms,parameter_count=len(refined),
                parameter_snapshot=dict(vector=refined.cpu().tolist(),shared_size=objective.shared_size,frame_timestamps_ms=[r['timestamp_ms'] for r in objective.rows],free_coordinates=objective.free.cpu().tolist()),
                fixed_forward=getattr(objective,'fixed_forward',False),per_frame_trainable=not getattr(objective,'fixed_forward',False),
                ema_decay=.95 if getattr(objective,'fixed_forward',False) else None,
                ema_scope='global_stages_only_critical_fingers_direct_noncritical_surface_ema' if local_report else 'all_parameters',
                local_geometry=local_report,
                training_trace=metrics.report(),
                contact_loss=loss_spec(objective.settings),contact_gradient_diagnostics=gradients,
                loss_weights=dict(contact_position=320,contact_normal=40,pinch_observation=2,contact_tip_emf_orientation=objective.settings.calibration_tip_emf_orientation_weight))
