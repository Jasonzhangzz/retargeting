"""FlashSAC-inspired replay/compiled-update experiment for inverse calibration.

This is direct differentiable identification, NOT the SAC actor/critic algorithm.
No reward, rollout, Bellman target or learned dynamics is invented for EMF data.
The device-buffer/update organization is adapted from Holiday-Robot/FlashSAC,
MIT, revision recorded in licenses/FlashSAC-SOURCE.json. All geometry is FP32.
"""
import time
import torch

REVISION = 'mano_flash_replay_pair_local_v3'
UPSTREAM = 'https://github.com/Holiday-Robot/FlashSAC'


class CalibrationReplayBuffer:
    """Immutable device tensors; held-out records never enter this buffer.

    Calibration uses full-buffer replay to preserve action-normalized losses and
    the exact geometry Jacobian. Uniform sampling is available for experiments;
    it is not used to shuffle causal runtime histories.
    """
    def __init__(self, rows, tensors):
        if not rows or any(row.get('partition') != 'train' for row in rows):
            raise ValueError('Flash replay buffer 只能包含训练分区')
        if not tensors or any(len(value) != len(rows) for value in tensors.values()):
            raise ValueError('Replay buffer 帧数不一致')
        devices = {value.device for value in tensors.values()}
        if len(devices) != 1:
            raise ValueError('Replay buffer 必须位于同一设备')
        self.device = next(iter(devices))
        self.size = len(rows)
        self.tensors = {key: value.detach().clone() for key, value in tensors.items()}

    def sample(self, batch_size, generator=None):
        if batch_size <= 0:
            raise ValueError('Replay batch size 必须为正')
        indices = torch.randint(self.size, (batch_size,), device=self.device, generator=generator)
        return indices, {key: value.index_select(0, indices) for key, value in self.tensors.items()}


def chunked_jacobian(function,vector,chunk_size=8):
    """Forward AD without materializing all 43 tangent batches concurrently."""
    from torch.func import jvp,vmap
    basis=torch.eye(len(vector),device=vector.device,dtype=vector.dtype)
    columns=[]
    for start in range(0,len(vector),chunk_size):
        columns.append(vmap(lambda tangent:jvp(function,(vector,),(tangent,))[1])(basis[start:start+chunk_size]))
    return torch.cat(columns,0).T


def _geometry_observability(objective, vector):
    ids = torch.arange(46, objective.shared_size, device=vector.device)
    def measurement(z):
        return objective.residual(vector.detach().scatter(0, ids, z), measurement_only=True)
    # Observability includes every training action, including contacts which
    # will only be optimized later in local blocks. No validation row is used.
    stages=objective.phase_stages;weights=objective.weights
    from .mano_objective import stage_weights
    objective.phase_stages={r['stage'] for r in objective.rows}
    objective.weights=objective.t(stage_weights(objective.rows,objective.mask_np,True,
        objective.settings.pinch_observation_multiplier))
    try:jacobian = chunked_jacobian(measurement,vector[ids].detach())
    finally:objective.phase_stages=stages;objective.weights=weights
    threshold = jacobian.norm(dim=0).max().clamp_min(1e-6) * 1e-4
    remaining = jacobian.clone()
    basis = []
    for _ in range(len(ids)):
        norms = remaining.square().sum(0)
        column = int(norms.argmax())
        if float(norms[column].sqrt()) <= float(threshold):
            break
        unit = remaining[:, column] / norms[column].sqrt()
        basis.append(column)
        remaining -= unit[:, None] * (unit @ remaining)[None]
    frozen = sorted(set(range(len(ids))) - set(basis))
    return frozen, dict(method='GPU_conditional_pivoted_modified_Gram_Schmidt',
                        rank=len(basis), parameters=len(ids), frozen_coordinates=frozen,
                        threshold=float(threshold))


def make_update(objective, vector, optimizer, frozen, fixed, compile_step=True):
    """Compile forward/backward/Adam/projection; synchronize only at block ends.

    torch.compile permits graph breaks at autograd/optimizer boundaries. We do
    not claim one CUDA graph: Warp BVH refresh remains an external graph.
    """
    # Capturing a newly allocated torch.optim object guards its Python identity
    # and invalidates every fit's compiled graph. Keep only device tensor state.
    group=optimizer.param_groups[0] if optimizer is not None else {}
    lr=group.get('lr',.01);beta1,beta2=group.get('betas',(.9,.999));eps=group.get('eps',1e-8)
    if group.get('weight_decay',0) or group.get('amsgrad',False):raise ValueError('Flash replay 仅支持无衰减的标准 Adam')
    moment=torch.zeros_like(vector);variance=torch.zeros_like(vector)
    count=torch.zeros_like(vector)
    def update():
        vector.grad=None
        before = vector.detach().clone()
        loss = objective.residual(vector).square().sum()
        loss.backward()
        finite = torch.isfinite(loss) & torch.isfinite(vector.grad).all()
        vector.grad.masked_fill_(frozen, 0.)
        objective.latest_gradient_norm=vector.grad.detach().norm()
        with torch.no_grad():
            gradient=vector.grad
            gradient.mul_((100./(gradient.norm()+1e-6)).clamp_max(1.))
            # A newly activated pair must not inherit inactive Adam momentum
            # or a bias-correction clock advanced by the other three pairs.
            moment.masked_fill_(frozen,0.);variance.masked_fill_(frozen,0.)
            count.masked_fill_(frozen,0.);count.add_((~frozen).to(count.dtype))
            moment.lerp_(gradient,1.-beta1)
            variance.mul_(beta2).addcmul_(gradient,gradient,value=1.-beta2)
            corrected_moment=moment/(1.-beta1**count.clamp_min(1.))
            denominator=(variance/(1.-beta2**count.clamp_min(1.))).sqrt()+eps
            vector.add_(-lr*corrected_moment/denominator)
            vector.copy_(torch.where(frozen, fixed, vector.clamp(objective.lower, objective.upper)))
            if getattr(objective,'fixed_forward',False):
                from .mano_shape_prior import project
                vector.copy_(project(vector,objective.lower,objective.upper,objective.shared_size,unbounded_shape=getattr(objective,'unbounded_shape',False)))
            vector.copy_(torch.where(frozen,fixed,vector))
        return loss.detach(), finite.detach(), before
    return torch.compile(update, options={'triton.cudagraphs': False}, dynamic=False) if compile_step else update


def solve_calibration_gpu(objective, cancel, progress, max_seconds=30,
                          resume_phases=None, compile_loss=True):
    if getattr(objective,'mixed_training',False):
        from .mano_mixed_training import solve_mixed
        return solve_mixed(objective,cancel,progress,max_seconds,compile_loss,flash=True)
    from .mano_objective import SOLVER_REVISION
    from .mano_gpu_solver import bounded_lm_gpu
    from .mano_local_geometry import PairLocalRefinement,phase_stages,BLOCK_STEPS
    if resume_phases:
        raise ValueError('Flash replay 使用独立完整训练，不能复用其他优化器快照')
    buffer = CalibrationReplayBuffer(objective.rows, dict(root=objective.root,
        seed_pose=objective.seed_pose, seed_angles=objective.seed_angles,
        rawp=objective.rawp, rawr=objective.rawr))
    # The objective reads the immutable resident buffer directly on every update.
    for name, value in buffer.tensors.items():
        setattr(objective, name, value)
    began = time.monotonic()
    deadline = began + max_seconds
    x = torch.nn.Parameter(objective.initial.clone())
    original = objective.weights.clone()
    from .mano_training_metrics import TrainingMetrics
    metrics=TrainingMetrics(objective,progress)
    phases = []; observability = {}; weak = []; compilation = 0.; exhausted = False;local=None
    localized=getattr(objective,'fixed_forward',False) and objective.personalized
    for phase_index, phase in enumerate(('reference', 'geometry', 'pinches')):
        if cancel.is_set():
            raise InterruptedError('Flash replay 标定已取消')
        allowed = phase_stages(objective.rows,phase) if localized else ({'open', 'together', 'mcp90', 'hook'} if phase_index == 0 else {r['stage'] for r in objective.rows})
        objective.phase_stages = allowed
        objective.weights = original * objective.t([r['stage'] in allowed for r in objective.rows])[:, None]
        objective.weights /= objective.weights.sum().clamp_min(1e-12)
        frozen = torch.zeros_like(x, dtype=torch.bool)
        if objective.personalized:
            if phase_index == 0:
                frozen[46:objective.shared_size] = True
            else:
                frozen[:10] = True
                if not observability:
                    weak, observability = _geometry_observability(objective, x)
                if weak:
                    frozen[torch.tensor(weak, device=x.device) + 46] = True
        fixed = x.detach().clone()
        # Reserve compute time for every phase, rather than starving pinches.
        share=(.20,.3125,1.)[phase_index] if localized else 1/(3-phase_index)
        phase_deadline = time.monotonic() + max(0., deadline-time.monotonic())*share
        if localized and phase=='pinches':
            local=PairLocalRefinement(objective,x,frozen,original)
            frozen=local.frozen;fixed=local.fixed
        update = make_update(objective, x, None, frozen, fixed, compile_loss)
        best = fixed.clone(); best_cost = float('inf'); previous = None; stale = 0
        phase_start = time.monotonic(); completed = 0; limited = False;ema=fixed.clone()
        cycle=BLOCK_STEPS*len(local.stages) if local else 1
        max_steps=(objective.settings.adam_steps//cycle)*cycle if local and objective.settings.adam_steps>=cycle else objective.settings.adam_steps
        while completed < max_steps:
            if cancel.is_set():
                raise InterruptedError('Flash replay 标定已取消')
            if completed and completed%cycle==0 and time.monotonic() >= phase_deadline:
                limited = True
                break
            if local and completed%BLOCK_STEPS==0:local.select(completed//BLOCK_STEPS,x,ema)
            objective.refresh(x.detach())
            first = completed == 0
            if first:metrics.emit(x.detach(),ema,phase,0,local,force=True,advance=0,full_replay=bool(local))
            if first and compile_loss:
                progress(dict(state='running', phase='compile', progress=.08+.21*phase_index,
                    message=f'Flash replay 编译更新阶段 {phase_index+1}/3', estimated_remaining_s=None))
            start = time.monotonic()
            finite_block = torch.ones((), dtype=torch.bool, device=x.device)
            # No Python reads of costs or gradients inside the update block.
            block_steps=min(objective.settings.collision_refresh, max_steps-completed)
            for _ in range(block_steps):
                value, finite, snapshot = update()
                if getattr(objective,'fixed_forward',False):
                    from .mano_shape_prior import EMA_DECAY
                    with torch.no_grad():ema.copy_(local.update_ema(ema,x) if local else objective.shape_prior.ema_update(ema,x));ema[frozen]=fixed[frozen]
                    if local:
                        with torch.no_grad():x.copy_(ema)
                finite_block &= finite
                completed += 1
            metrics.emit(x.detach(),ema,phase,completed,local,advance=block_steps)
            if local and completed%BLOCK_STEPS==0:local.record_block(ema,BLOCK_STEPS)
            torch.cuda.synchronize()
            if first and compile_loss:
                seconds = time.monotonic()-start
                compilation += seconds; deadline += seconds; phase_deadline += seconds
            if cancel.is_set():
                raise InterruptedError('Flash replay 标定已取消')
            if not bool(finite_block):
                raise ValueError('Flash replay 损失或梯度非有限')
            cost = float(value)
            if cost < best_cost:
                best_cost = cost; best = snapshot.clone()
            if not local:
                relative = abs(previous-cost)/max(abs(previous), 1e-8) if previous is not None else float('inf')
                stale = stale+1 if relative < 1e-4 else 0
                previous = cost
            elif completed%cycle==0:
                cost=local.cycle_cost(ema)
                metrics.emit(x.detach(),ema,phase,completed,local,force=True,advance=0,full_replay=True)
                relative=abs(previous-cost)/max(abs(previous),1e-8) if previous is not None else float('inf')
                stale=stale+1 if relative<1e-4 else 0;previous=cost
            progress(dict(state='running', phase=phase, progress=.08+.21*(phase_index+completed/max_steps),
                message=f'Flash replay 参数更新 {phase_index+1}/3 · {local.message()+" · " if local else ""}{completed} 步',
                estimated_remaining_s=max(0., deadline-time.monotonic()), eta_scope='optimization_budget'))
            if completed >= 80 and stale >= (3 if local else 6):
                break
        metrics.emit(x.detach(),ema,phase,completed,local,force=True,advance=0,full_replay=bool(local))
        with torch.no_grad():
            x.copy_(ema if getattr(objective,'fixed_forward',False) else best)
        exhausted |= limited
        phases.append(dict(phase=phase, steps=completed, state='budget_exhausted' if limited else 'converged' if stale>=6 else 'completed',
                           elapsed_s=time.monotonic()-phase_start))
    objective.weights = original
    local_report=local.report() if local else None
    if local:local.close()
    del objective.phase_stages
    objective.frozen_shared = torch.zeros_like(x, dtype=torch.bool)
    if objective.personalized:
        objective.frozen_shared[:10] = True
        if weak:
            objective.frozen_shared[torch.tensor(weak, device=x.device)+46] = True
    objective.refresh(x.detach())
    progress(dict(state='running', progress=.74, message='Flash replay 冻结参数与局部表面 EMA，准备前向验收' if getattr(objective,'fixed_forward',False) else 'Flash replay GPU 有界 LM 精修', estimated_remaining_s=None))
    vector,lm=(x.detach(),dict(accepted_steps=0,reason='frozen_EMA_artifact')) if getattr(objective,'fixed_forward',False) else bounded_lm_gpu(objective, x.detach(), cancel, deadline, objective.settings.lm_steps)
    objective.refresh(vector); objective.residual(vector, record=True); objective.install(vector)
    from .mano_contact_loss import loss_spec,gradient_report
    gradients=gradient_report(objective,vector)
    return dict(revision=SOLVER_REVISION, experiment_revision=REVISION,
        algorithm='direct_differentiable_identification', optimizer='compiled_Adam_with_EMA' if getattr(objective,'fixed_forward',False) else 'compiled_Adam_then_bounded_block_LM',
        framework_reference=UPSTREAM, sac_actor_critic=False, reward_used=False, rollout_used=False,
        replay=dict(frames=buffer.size, device=str(buffer.device), update_batch='full_train_buffer', heldout_frames=0),
        phases=phases, lm=lm, budget_exhausted=exhausted, compilation_s=compilation,
        collision_witness_complete=not bool(objective.gpu_witness['witness_overflow'].any()),
        compute_device=str(x.device), observability=observability, cost_terms=objective.last_terms,
        parameter_count=len(vector), parameter_snapshot=dict(vector=vector.cpu().tolist(), shared_size=objective.shared_size,
            frame_timestamps_ms=[r['timestamp_ms'] for r in objective.rows], free_coordinates=objective.free.cpu().tolist()),
        fixed_forward=getattr(objective,'fixed_forward',False),per_frame_trainable=not getattr(objective,'fixed_forward',False),ema_decay=.95 if getattr(objective,'fixed_forward',False) else None,
        ema_scope='global_stages_only_critical_fingers_direct_noncritical_surface_ema' if local_report else 'all_parameters',
        local_geometry=local_report,
        training_trace=metrics.report(),
        contact_loss=loss_spec(objective.settings),contact_gradient_diagnostics=gradients,
        loss_weights=dict(contact_position=320, contact_normal=40, pinch_observation=2,contact_tip_emf_orientation=objective.settings.calibration_tip_emf_orientation_weight))


def fit_capture(model_path, profile, mapping, rows, progress, cancel, **kwargs):
    from .mano_calibration import run_backend
    return run_backend('flashsac', model_path, profile, mapping, rows, progress, cancel, **kwargs)
