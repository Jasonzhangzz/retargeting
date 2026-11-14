"""All training actions in every update; sample-local geometry derivatives.

The complete train split is a resident, full-batch replay buffer. There are no
calibration phases, stage gates, pair blocks, padding or resampled duplicates.
"""
import time
import numpy as np
import torch
from .mano_gpu_objective import GPUManoObjective
from .mano_local_geometry import finger_coordinates,spatial_ema_step
from .contact import FINGERS
from .glove_geometry import CHAINS

REVISION='mano_mixed_unbounded_geometry_joint_delta_v3'


def geometry_gradient_mask(rows,device=None):
    if not rows or any(r.get('partition')!='train' for r in rows):
        raise ValueError('混训 buffer 只能包含训练分区')
    masks=[]
    for row in rows:
        mask=torch.ones(89,dtype=torch.bool)
        from .contact_protocol import active_fingers
        active=active_fingers(row)
        if active is not None:
            mask=torch.zeros(89,dtype=torch.bool)
            for finger in active:mask|=finger_coordinates(FINGERS.index(finger))
            from .glove_geometry import ROOT_FINGERS
            from .calibration_targets import target
            for region in target(row['stage'])['patches']:
                if region.startswith('palm_mcp_'):
                    finger=FINGERS.index(region[9:])
                    if finger in ROOT_FINGERS:
                        start=71+3*ROOT_FINGERS.index(finger);mask[start:start+3]=True
            if len(active)==5:mask[:10]=True
        elif row['stage'].startswith('pinch_'):
            partner=FINGERS.index(row['stage'][6:])
            mask=finger_coordinates(0)|finger_coordinates(partner)
        masks.append(torch.cat((mask[:10],mask[46:89])))
    return torch.stack(masks).to(device=device)


def masked_geometry(vector,mask):
    shared=torch.cat((vector[:10],vector[46:89]))
    # Numerically identical shared geometry for all samples. Only the backward
    # path differs: one pinch cannot reshape an unrelated finger or global beta.
    per_frame=shared.detach()[None]+mask*(shared-shared.detach())[None]
    return per_frame[:,:10],per_frame[:,10:]


class MixedGPUManoObjective(GPUManoObjective):
    mixed_training=True
    unbounded_shape=True

    def __init__(self,model,rows,poses,roots,**kwargs):
        if not kwargs.get('fixed_forward') or not kwargs.get('shared'):
            raise ValueError('完全混训使用共享几何和固定 FK')
        mask=geometry_gradient_mask(rows,model.device)
        super().__init__(model,rows,poses,roots,**kwargs)
        self.frame_geometry_mask=mask
        self.accepted_parameters_direct=True
        self.shape_prior.soft_constraints=True

    def surface(self,x):
        if self._surface_cache is not None:return self._surface_cache
        beta,mp,mr,align,trans,angles=self.unpack(x)
        betas,geometry=masked_geometry(x,self.frame_geometry_mask)
        from .card_math import exp_so3
        v,j,r,rest,rest_j=self.model.model.official_batched_matrix_forward(
            betas,exp_so3(self.seed_pose),geometry)
        v=(v-rest_j[:,:1])@self.root.transpose(-1,-2)
        j=(j-rest_j[:,:1])@self.root.transpose(-1,-2)
        r=self.root[:,None]@r
        self._mixed_rest=(rest,rest_j)
        return self.seed_pose,v,j,r,(beta,mp,mr,align,trans,angles)

    def contact_rest(self,beta,geometry):
        return self._mixed_rest


def solve_mixed(objective,cancel,progress,max_seconds=45,compile_loss=True,flash=False):
    from .mano_objective import SOLVER_REVISION
    from .mano_shape_prior import project
    from .mano_training_metrics import TrainingMetrics
    from .mano_contact_loss import loss_spec,gradient_report
    from .mano_calibrate_flashsac import CalibrationReplayBuffer,make_update
    if cancel.is_set():raise InterruptedError('全量混训已取消')
    if any(r.get('partition')!='train' for r in objective.rows):
        raise ValueError('混训不能读取留出帧')
    if hasattr(objective,'phase_stages') or hasattr(objective,'phase_active'):
        raise ValueError('完全混训不能带有动作阶段掩码')
    began=time.monotonic();deadline=began+max_seconds
    resident=CalibrationReplayBuffer(objective.rows,{k:getattr(objective,k) for k in
        ('seed_pose','seed_angles','root','rawp','rawr','weights','frame_geometry_mask')})
    for key,value in resident.tensors.items():setattr(objective,key,value)
    x=torch.nn.Parameter(objective.initial.clone());average=x.detach().clone()
    joint=getattr(objective,'joint_training',False)
    policy_parameters=list(objective.joint_policy.parameters()) if joint else []
    if joint and flash:raise ValueError('联合 Delta 使用同一次联合更新几何与网络')
    selected_optimizer=getattr(objective.settings,'optimizer','adam')
    if selected_optimizer=='muon' and (flash or not joint):raise ValueError('Muon 需要 MANO＋Delta 隐藏层；当前模式请选 Adam')
    frozen=torch.zeros_like(x,dtype=torch.bool);frozen[83:89]=True
    fixed=x.detach().clone()
    # These six axes are absent from the existing fixed-FK forward model. Do
    # not claim that mixing more data makes structurally unused axes observable.
    observability=dict(method='structural_zero_derivative_fixed_FK',rank=None,
        parameters=43,frozen_coordinates=list(range(37,43)),heldout_used=False)
    critical=torch.zeros(len(objective.model.rest),device=x.device,dtype=torch.bool)
    weights=objective.model.model.backend.lbs_weights.detach()
    partners=sorted({FINGERS.index(r['stage'][6:]) for r in objective.rows if r['stage'].startswith('pinch_')})
    from .contact_protocol import active_fingers
    active_indices={FINGERS.index(f) for row in objective.rows for f in (active_fingers(row) or ())}
    for partner in partners:active_indices.update((0,partner))
    for finger in sorted(active_indices):critical|=weights[:,list(CHAINS[finger])].sum(-1)>=.5
    templates=getattr(objective.model,'contact_template_artifact',None)
    if templates:
        from .calibration_targets import target
        for row in objective.rows:
            for region in target(row['stage']).get('patches',[]):
                if region in templates['regions']:critical[objective.model.face_t[templates['regions'][region]].flatten()]=True
    objective.model.model.surface_residual_enabled=True
    from .training_optimizer import TrainingOptimizer
    optimizer=None if flash else TrainingOptimizer(x,objective.joint_policy if joint else None,selected_optimizer)
    optimizer_name='compiled_Adam' if flash else optimizer.name
    loss_fn=lambda z:objective.residual(z).square().sum()
    update=make_update(objective,x,None,frozen,fixed,compile_loss) if flash else None
    compiled=torch.compile(loss_fn,options={'triton.cudagraphs':False},dynamic=False) if compile_loss and not flash else loss_fn
    metrics=TrainingMetrics(objective,progress)
    compilation=0.;completed=0;previous=None;stale=0;exhausted=False
    accepted=torch.zeros((),device=x.device,dtype=torch.long);rejected=torch.zeros_like(accepted)
    metrics.emit(x.detach(),average,'mixed',0,force=True,advance=0)
    for step in range(objective.settings.adam_steps):
        if cancel.is_set():raise InterruptedError('全量混训已取消')
        if step and time.monotonic()>=deadline:exhausted=True;break
        if step%objective.settings.collision_refresh==0:objective.refresh(x.detach())
        if step==0:progress(dict(state='running',phase='compile',progress=.08,message=f'全量混训编译 · {resident.size} 帧 · {optimizer_name}',estimated_remaining_s=None))
        start=time.monotonic()
        if flash:
            loss,finite,_=update()
            objective.optimizer_diagnostics=dict(geometry_gradient_norm=float(objective.latest_gradient_norm),policy_gradient_norm=0.,geometry_learning_rate=.01,policy_learning_rate=0.)
        else:
            optimizer.zero_grad(set_to_none=True);loss=compiled(x);loss.backward()
            finite=torch.isfinite(loss)&torch.isfinite(x.grad).all()
            for p in policy_parameters:
                if p.grad is not None:finite=finite&torch.isfinite(p.grad).all()
            x.grad[frozen]=0
            objective.optimizer_diagnostics=dict(geometry_gradient_norm=float(x.grad[:objective.shared_size].norm()),policy_gradient_norm=float(torch.stack([p.grad.square().sum() for p in policy_parameters if p.grad is not None]).sum().sqrt()) if policy_parameters else 0.,geometry_learning_rate=optimizer.geometry_lr,policy_learning_rate=optimizer.policy_lr,policy_muon_learning_rate=optimizer.muon_lr,optimizer_name=optimizer_name)
            torch.nn.utils.clip_grad_norm_([x]+policy_parameters,100.)
            optimizer.step()
        if step==0 and compile_loss:
            torch.cuda.synchronize();compilation=time.monotonic()-start;deadline+=compilation
        if not bool(finite):raise ValueError('全量混训损失或梯度非有限')
        if cancel.is_set():raise InterruptedError('全量混训已取消')
        with torch.no_grad():
            proposal=project(x,objective.lower,objective.upper,objective.shared_size,unbounded_shape=True)
            proposal[frozen]=fixed[frozen]
            updated,valid=spatial_ema_step(objective.model.model,objective.shape_prior,
                                          average,proposal,frozen,critical,soft_constraints=True)
            accepted.add_(valid.long());rejected.add_((~valid).long())
            average.copy_(updated);x.copy_(average)
        completed+=1
        metrics.emit(x.detach(),average,'mixed',completed)
        if step%20==0:
            value=float(loss.detach())
            relative=abs(previous-value)/max(abs(previous),1e-8) if previous is not None else float('inf')
            stale=stale+1 if relative<1e-4 else 0;previous=value
            progress(dict(state='running',phase='mixed',progress=.08+.63*completed/objective.settings.adam_steps,
                message=('几何＋Delta 联合混训' if joint else '全量混训')+f' · {resident.size} 帧／步 · {completed} 步',
                estimated_remaining_s=max(0.,deadline-time.monotonic()),eta_scope='optimization_budget'))
            if step>=100 and stale>=5:break
    metrics.emit(x.detach(),average,'mixed',completed,force=True,advance=0)
    objective.refresh(average);objective.install(average);objective.residual(average,record=True)
    if joint:
        objective.model.delta_policy=objective.joint_policy.eval()
        objective.model.delta_artifact=objective.artifact()
    gradients=gradient_report(objective,average)
    policy=dict(revision='mixed_sample_local_spatial_ema_v2',critical_parameter_decay=0.,
        critical_surface_decay=0.,noncritical_surface_decay=.95,critical_vertex_rule='union_of_observed_active_fingers_LBS_weight_at_least_0.5',
        space='wrist_relative_rest_vertices_before_pose_correctives_and_LBS',runtime_updates=False,
        trainable_vertex_parameters=False,accepted_surface_steps=int(accepted),rejected_surface_steps=int(rejected),
        fixed_surface_residual_max_mm=float(objective.model.model.rest_vertex_residual.norm(dim=-1).amax())*1000,
        pair_blocks=False,global_shape_frozen=False,pinch_geometry_gradient='thumb_and_partner_only; beta_detached',heldout_used=False)
    from .mano_shape_prior import SOFT_SHAPE_WEIGHT,SOFT_SHAPE_MARGIN,SOFT_SHAPE_SCALE
    soft_shape=dict(training_step_rejection=False,weight=SOFT_SHAPE_WEIGHT,margin_ratio=SOFT_SHAPE_MARGIN,
        scale_ratio=SOFT_SHAPE_SCALE,pooling='stable_p8_softplus_constraint_slacks',
        final_validation_unchanged=True,hard_shape_bounds=False,
        numerical_parameter_bounds_retained='mounts_and_alignment_only')
    elapsed=time.monotonic()-began
    return dict(revision=SOLVER_REVISION,algorithm=REVISION,optimizer=optimizer_name,
        optimizer_assignment=optimizer.assignment if optimizer is not None else dict(geometry='compiled_Adam',policy_muon=[],selection='adam'),
        phases=[dict(phase='mixed',steps=completed,state='budget_exhausted' if exhausted else 'converged' if stale>=5 else 'completed',elapsed_s=elapsed)],
        training_schedule='all_train_frames_every_update',stage_gating=False,pair_rotation=False,
        sac_actor_critic=False,reward_used=False,rollout_used=False,
        replay=dict(frames=resident.size,device=str(resident.device),update_batch='full_train_buffer',heldout_frames=0,
            stages=sorted({r['stage'] for r in objective.rows}),all_frames_used_every_step=True),
        lm=dict(accepted_steps=0,reason='fixed_forward_mixed_artifact'),budget_exhausted=exhausted,
        compilation_s=compilation,compute_device=str(x.device),observability=observability,
        collision_witness_complete=not bool(objective.gpu_witness['witness_overflow'].any()),
        cost_terms=dict(objective.last_terms),parameter_count=len(average),
        parameter_snapshot=dict(vector=average.cpu().tolist(),shared_size=objective.shared_size,
            shared_reference=dict(sensor_joint_position=objective.mp0.cpu().tolist(),mount_rotation=objective.mr0.cpu().tolist(),raw_rotation=objective.align0.cpu().tolist(),raw_translation=objective.trans0.cpu().tolist()),
            frame_timestamps_ms=[r['timestamp_ms'] for r in objective.rows],free_coordinates=objective.free.cpu().tolist()),
        continuation=getattr(objective,'resume_info',None),
        total_training_steps=completed+getattr(objective,'resume_info',{}).get('previous_steps',0),
        fixed_forward=True,per_frame_trainable=False,ema_decay=.95,ema_scope='noncritical_rest_surface_only',
        local_geometry=policy,training_trace=metrics.report(),contact_loss=loss_spec(objective.settings),
        shape_soft_constraints=soft_shape,
        contact_gradient_diagnostics=gradients,
        joint_geometry_policy=joint,geometry_frozen_during_policy_training=False if joint else None,
        policy_parameter_count=sum(p.numel() for p in policy_parameters),
        loss_weights=dict(contact_position=320,contact_normal=40,pinch_observation=2,
                          contact_tip_emf_orientation=objective.settings.calibration_tip_emf_orientation_weight))
