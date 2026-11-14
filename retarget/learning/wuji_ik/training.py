"""torchrun DDP, count-weighted accumulation, paired optimizers and exact-step resume."""
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from .config import ModelConfig
from .batching import FrameBudgetSampler
from .data import SegmentDataset, collate_segments, write_json
from .features import JointNormalizer
from .metrics import autocast_context, evaluate_model, move_batch
from .models import build_model, flow_training_pair
from .optim import make_optimizers, make_schedulers
from .losses import acceleration_mask, acceleration_sum


@dataclass
class TrainConfig:
    variant: str = 'temporal_causal'
    epochs: int = 20
    batch_size: int = 1
    accumulation: int = 8
    muon_lr: float = .02
    adamw_lr: float = .0003
    weight_decay: float = .01
    warmup_fraction: float = .02
    grad_clip: float = 1.
    seed: int = 42
    workers: int = 4
    bf16: bool = True
    device: str = 'auto'
    threads: int = 1
    checkpoint_every: int = 500
    max_steps: int = 0
    stop_after_steps: int = 0
    effective_frames_per_gpu: int = 0
    gpu_memory_limit_gib: float = 0.
    smoothness_weight: float = 0.

    def validate(self):
        if min(self.epochs, self.batch_size, self.accumulation, self.threads, self.checkpoint_every) < 1:
            raise ValueError('Invalid training budget/batch configuration')
        if self.workers < 0 or min(self.max_steps, self.stop_after_steps) < 0:
            raise ValueError('Invalid worker or step limit')
        if min(self.muon_lr, self.adamw_lr, self.grad_clip) <= 0 or self.weight_decay < 0 or not 0 <= self.warmup_fraction < 1:
            raise ValueError('Invalid learning rate, clipping, decay or warmup')
        if self.effective_frames_per_gpu < 0 or self.gpu_memory_limit_gib < 0:
            raise ValueError('Invalid effective frame budget or GPU memory limit')
        if self.effective_frames_per_gpu and self.batch_size != 1:
            raise ValueError('Exact frame-budget mode uses one full segment per microbatch')
        if not math.isfinite(self.smoothness_weight) or self.smoothness_weight < 0:
            raise ValueError('Smoothness weight must be finite and nonnegative')
        if self.smoothness_weight and self.variant not in ('temporal_causal', 'temporal_bidirectional'):
            raise ValueError('Acceleration smoothness is only supported for temporal regression')


def distributed_context(requested='auto'):
    world = int(os.environ.get('WORLD_SIZE', '1')); rank = int(os.environ.get('RANK', '0'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    use_cuda = requested != 'cpu' and (requested.startswith('cuda') or torch.cuda.is_available())
    device = torch.device(f'cuda:{local}' if world > 1 and use_cuda else ('cuda:0' if requested == 'auto' and use_cuda else
        ('cpu' if requested == 'auto' else requested)))
    if use_cuda: torch.cuda.set_device(device)
    if world > 1 and not dist.is_initialized(): dist.init_process_group('nccl' if use_cuda else 'gloo')
    return device, rank, world


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device).cpu() if device.type == 'cuda' else None)


def restore_rng(state, device):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'].cpu())
    if device.type == 'cuda': torch.cuda.set_rng_state(state['cuda'].cpu(), device)


def scaled_ddp_loss(local_sum, global_count, world_size):
    # DDP averages gradients; undo that averaging before dividing by all valid elements.
    return local_sum * (world_size / global_count)


def save_checkpoint(path, model, normalizer, optimizers, schedulers, config, model_config,
                    signature, step, epoch, next_batch, best, device, rank, world, total_steps, scheduler_continuation=None,
                    initialization=None):
    local_state = rng_state(device); local_state['next_batch'] = next_batch
    states = [local_state]
    if world > 1:
        gathered = [None]*world; dist.all_gather_object(gathered, states[0]); states = gathered
    if rank == 0:
        state = dict(format='wuji_ik_checkpoint_v1', variant=config.variant, model_config=model_config.to_dict(),
            train_config=asdict(config), data_signature=signature, model=model.state_dict(), normalizer=normalizer.state_dict(),
            optimizers=[o.state_dict() for o in optimizers], schedulers=[s.state_dict() for s in schedulers],
            step=step, epoch=epoch, next_batch=next_batch, best=best, world_size=world, rng_by_rank=states,
            total_steps=total_steps, scheduler_continuation=scheduler_continuation,
            initialization=initialization,
            next_batch_by_rank=[v['next_batch'] for v in states], attention_backends=model.core.backend_report())
        temp = Path(str(path)+'.tmp'); torch.save(state, temp); os.replace(temp, path)
    if world > 1: dist.barrier()


def load_inference(checkpoint, device='cpu', backend=None):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['format'] != 'wuji_ik_checkpoint_v1': raise ValueError('Unknown checkpoint format')
    config = dict(state['model_config'])
    if backend is not None: config['backend'] = backend
    model = build_model(state['variant'], ModelConfig(**config)).to(device)
    model.load_state_dict(state['model']); model.eval()
    normalizer = JointNormalizer(state['normalizer']['limits']).to(device)
    return model, normalizer, state


def train(prepared, output, config=None, model_config=None, resume=None, return_model=False, extend_epochs=False,
          initialize_from=None, initialize_optimizer=False):
    config = config or TrainConfig(); config.validate(); model_config = model_config or ModelConfig()
    torch.set_num_threads(config.threads)
    device, rank, world = distributed_context(config.device)
    if device.type == 'cuda' and config.gpu_memory_limit_gib:
        torch.cuda.set_per_process_memory_fraction(config.gpu_memory_limit_gib*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
    seed_all(config.seed)
    root = Path(output)
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        if (root/'latest.pt').exists() and resume is None:
            raise ValueError('Output already has a checkpoint; explicitly supply --resume')
    if world > 1: dist.barrier()
    training = SegmentDataset(prepared, 'train'); validation = SegmentDataset(prepared, 'val')
    if len(training) < world or not len(validation): raise ValueError('Not enough train/validation segments')
    loader_generator = torch.Generator()
    if config.effective_frames_per_gpu:
        sampler = FrameBudgetSampler(training,config.effective_frames_per_gpu,rank,world,config.seed)
        loader = DataLoader(training,batch_sampler=sampler,num_workers=config.workers,collate_fn=collate_segments,
                            pin_memory=device.type=='cuda',generator=loader_generator)
        per_epoch = sampler.optimizer_steps
    else:
        sampler = DistributedSampler(training, num_replicas=world, rank=rank, shuffle=True, seed=config.seed, drop_last=True)
        loader = DataLoader(training, batch_size=config.batch_size, sampler=sampler, num_workers=config.workers,
                            collate_fn=collate_segments, pin_memory=device.type == 'cuda', generator=loader_generator)
        per_epoch = math.ceil(len(loader)/config.accumulation)
    total_steps = per_epoch*config.epochs
    if config.max_steps: total_steps = min(total_steps, config.max_steps)
    model = build_model(config.variant, model_config).to(device)
    normalizer = JointNormalizer(training.metadata['limits']).to(device)
    optimizers, parameter_groups = make_optimizers(model, config.muon_lr, config.adamw_lr, config.weight_decay)
    schedulers = make_schedulers(optimizers, total_steps, config.warmup_fraction)
    start_epoch = 0; start_batch = 0; step = 0; best = float('inf')
    signature = training.metadata['signature']
    scheduler_continuation = None
    initialization = None
    if initialize_optimizer and not initialize_from:
        raise ValueError('--initialize-optimizer requires --initialize-from')
    if initialize_from:
        if resume: raise ValueError('Choose exact resume or weight-only initialization')
        initial = torch.load(initialize_from, map_location='cpu', weights_only=False)
        if (initial['format'] != 'wuji_ik_checkpoint_v1' or initial['data_signature'] != signature
                or initial['variant'] != config.variant or initial['model_config'] != model_config.to_dict()):
            raise ValueError('Initialization requires identical model, variant and data signature')
        model.load_state_dict(initial['model'])
        normalizer.load_state_dict(initial['normalizer'])
        initialization = dict(checkpoint=str(initialize_from), step=initial['step'],
                              policy='weights only; fresh optimizer, scheduler, RNG and training position')
        if initialize_optimizer:
            if initial['train_config'].get('smoothness_weight', 0.) != config.smoothness_weight:
                raise ValueError('Optimizer initialization requires the same smoothness objective')
            # Keep moments, while explicitly retaining the NEW scheduler's initial
            # learning rates. Loading old groups verbatim would restore the high LR.
            for optimizer, saved in zip(optimizers, initial['optimizers'], strict=True):
                configured = [{key: group[key] for key in ('lr', 'initial_lr', 'weight_decay')}
                              for group in optimizer.param_groups]
                optimizer.load_state_dict(saved)
                for group, values in zip(optimizer.param_groups, configured, strict=True):
                    group.update(values)
            initialization['policy'] = 'weights and optimizer moments; new capped LR schedule, RNG and training position'
        del initial
    if extend_epochs and not resume: raise ValueError('--extend-epochs requires --resume')
    if resume:
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if state['format'] != 'wuji_ik_checkpoint_v1' or state['data_signature'] != signature:
            raise ValueError('Checkpoint/data signature mismatch')
        if state['variant'] != config.variant or state['model_config'] != model_config.to_dict() or state['world_size'] != world:
            raise ValueError('Resume requires identical model, variant, attention backend and world size')
        old_config = state['train_config']; new_config = asdict(config)
        old_config = {'smoothness_weight': 0., **old_config}
        ignored = {'stop_after_steps', 'checkpoint_every', 'workers', 'device', 'threads'}
        extending = extend_epochs and config.epochs > old_config['epochs'] and total_steps > state['total_steps']
        if extend_epochs and (config.epochs < old_config['epochs'] or total_steps < state['total_steps']):
            raise ValueError('--extend-epochs cannot reduce the epoch or scheduler horizon')
        if extending: ignored.add('epochs')
        if any(old_config[k] != new_config[k] for k in old_config if k not in ignored) or (not extending and total_steps != state['total_steps']):
            raise ValueError('Resume requires the same sampling, optimizer and scheduler horizon')
        model.load_state_dict(state['model']); normalizer.load_state_dict(state['normalizer'])
        for optimizer, value in zip(optimizers, state['optimizers'], strict=True): optimizer.load_state_dict(value)
        scheduler_continuation = state.get('scheduler_continuation')
        if extending:
            scheduler_continuation = dict(step=state['step'], total_steps=total_steps,
                factors=[[g['lr']/base for g, base in zip(o.param_groups, s['base_lrs'], strict=True)]
                         for o, s in zip(optimizers, state['schedulers'], strict=True)])
        if scheduler_continuation is not None:
            saved_lrs = [[g['lr'] for g in o.param_groups] for o in optimizers]
            schedulers = make_schedulers(optimizers, total_steps, config.warmup_fraction, scheduler_continuation)
            for o, values in zip(optimizers, saved_lrs, strict=True):
                for group, lr in zip(o.param_groups, values, strict=True): group['lr'] = lr
        for scheduler, value in zip(schedulers, state['schedulers'], strict=True): scheduler.load_state_dict(value)
        step, start_epoch, best = state['step'], state['epoch'], state['best']
        start_batch = state.get('next_batch_by_rank',[state['next_batch']]*world)[rank]
        restore_rng(state['rng_by_rank'][rank], device)
        initialization = state.get('initialization')
    ddp = DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None,
                                  broadcast_buffers=False) if world > 1 else model
    if rank == 0:
        write_json(root/'configuration.json', dict(train=asdict(config), model=model_config.to_dict(),
            data_signature=signature, world_size=world, total_steps=total_steps, parameter_groups=parameter_groups,
            parameters=sum(p.numel() for p in model.parameters()), scheduler_continuation=scheduler_continuation,
            resumed_from=str(resume) if resume else None, resumed_step=step,
            initialization=initialization,
            initialization_optimizer_moments=initialize_optimizer or bool(initialization and 'optimizer moments' in initialization['policy']),
            smoothness_definition='mean squared second difference of normalized q; contiguous triples owned by final frame',
            training_drop_last_segments=0 if config.effective_frames_per_gpu else len(training)%world,
            accumulation_policy='exact loss-frame budget with complete-segment context' if config.effective_frames_per_gpu else 'fixed microbatch count'))
    started = time.perf_counter(); next_epoch = start_epoch; next_batch = start_batch

    def commit(epoch, batch, is_best=False):
        save_checkpoint(root/'latest.pt', model, normalizer, optimizers, schedulers, config, model_config,
            signature, step, epoch, batch, best, device, rank, world, total_steps, scheduler_continuation, initialization)
        if rank == 0 and is_best:
            import shutil
            temp = root/'best.pt.tmp'; shutil.copyfile(root/'latest.pt', temp); os.replace(temp, root/'best.pt')

    for epoch in range(start_epoch, config.epochs):
        if step >= total_steps: break
        sampler.set_epoch(epoch); loader_generator.manual_seed(config.seed+epoch)
        iterator = iter(loader); consumed = start_batch if epoch == start_epoch else 0
        for _ in range(consumed): next(iterator)
        budget_group = sampler.group_at_microbatch(consumed) if config.effective_frames_per_gpu else None
        while consumed < len(loader) and step < total_steps:
            if config.stop_after_steps and step >= config.stop_after_steps: break
            microbatches = len(sampler.groups[budget_group]) if config.effective_frames_per_gpu else min(config.accumulation,len(loader)-consumed)
            frame_counts = sampler.frames_by_rank[budget_group] if config.effective_frames_per_gpu else None
            group = [next(iterator) for _ in range(microbatches)]
            count = torch.tensor(sum(int(b['valid'].sum())*20 for b in group), dtype=torch.float64, device=device)
            if world > 1: dist.all_reduce(count)
            smooth_count = torch.tensor(sum(int(acceleration_mask(b).sum())*20 for b in group)
                if config.smoothness_weight else 0, dtype=torch.float64, device=device)
            if world > 1 and config.smoothness_weight: dist.all_reduce(smooth_count)
            for optimizer in optimizers: optimizer.zero_grad(set_to_none=True)
            model.train(); local_sum = torch.zeros((), dtype=torch.float64, device=device)
            local_smooth = torch.zeros((), dtype=torch.float64, device=device)
            for micro, cpu_batch in enumerate(group):
                batch = move_batch(cpu_batch, device)
                sync = ddp.no_sync() if world > 1 and micro < len(group)-1 else nullcontext()
                with sync, autocast_context(device, config.bf16):
                    target = batch['target']
                    if config.variant == 'frame_flow':
                        noisy, times, velocity = flow_training_pair(target)
                        prediction = ddp(batch['features'], batch['side'], batch['lengths'], noisy_q=noisy, flow_time=times)
                        residual = prediction.float()-velocity
                    else:
                        prediction = ddp(batch['features'], batch['side'], batch['lengths'])
                        residual = prediction.float()-target
                    loss_sum = (residual.square()*batch['valid'][..., None]).sum()
                    loss = scaled_ddp_loss(loss_sum, count, world)
                    if config.smoothness_weight:
                        smooth_sum = acceleration_sum(prediction, acceleration_mask(batch))
                        loss = loss + config.smoothness_weight*scaled_ddp_loss(smooth_sum, smooth_count.clamp_min(1), world)
                        local_smooth += smooth_sum.detach().double()
                loss.backward(); local_sum += loss_sum.detach().double()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
            for optimizer in optimizers: optimizer.step()
            for scheduler in schedulers: scheduler.step()
            step += 1; consumed += len(group)
            if config.effective_frames_per_gpu: budget_group += 1
            next_epoch, next_batch = (epoch+1, 0) if consumed == len(loader) else (epoch, consumed)
            if world > 1: dist.all_reduce(local_sum)
            if world > 1 and config.smoothness_weight: dist.all_reduce(local_smooth)
            if rank == 0:
                regression_loss = float(local_sum/count)
                smoothness_loss = float(local_smooth/smooth_count.clamp_min(1))
                objective = ('velocity_mse' if config.variant == 'frame_flow' else 'normalized_q_mse')
                record = dict(step=step, epoch=epoch, objective='normalized_q_mse_plus_acceleration' if config.smoothness_weight else objective,
                    loss=regression_loss+config.smoothness_weight*smoothness_loss,
                    regression_loss=regression_loss, smoothness_loss=smoothness_loss,
                    smoothness_weight=config.smoothness_weight, weighted_smoothness_loss=config.smoothness_weight*smoothness_loss,
                    valid_smooth_joint_elements=int(smooth_count),
                    valid_joint_elements=int(count), grad_norm=float(norm),
                    muon_lr=optimizers[0].param_groups[0]['lr'], adamw_lr=optimizers[1].param_groups[0]['lr'],
                    attention_backends=model.core.backend_report(), elapsed_seconds=time.perf_counter()-started)
                if frame_counts is not None: record['effective_frames_by_rank']=frame_counts
                with (root/'train.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
                if step == 1 or step%10 == 0: print(json.dumps(record), flush=True)
            if step%config.checkpoint_every == 0: commit(next_epoch, next_batch)
        # Evaluate even after a deliberate smoke stop, so both checkpoint paths are exercised.
        report = evaluate_model(model, validation, normalizer, config.variant, device,
            batch_size=config.batch_size, workers=config.workers, bf16=config.bf16, rank=rank, world_size=world, seed=config.seed)
        score = report['metrics']['overall']['normalized_mse']; improved = score < best
        best = min(best, score)
        if rank == 0:
            validation_record = dict(step=step, epoch=epoch, **report)
            write_json(root/'validation.json', validation_record)
            with (root/'validation.jsonl').open('a') as f: f.write(json.dumps(validation_record)+'\n')
        commit(next_epoch, next_batch, improved)
        if config.stop_after_steps and step >= config.stop_after_steps: break
    if rank == 0:
        status = dict(complete=step >= total_steps, step=step, total_steps=total_steps,
            best_validation_normalized_mse=best, data_signature=signature, elapsed_seconds=time.perf_counter()-started)
        write_json(root/'status.json', status)
    if world > 1: dist.barrier()
    digest = hashlib.sha256()
    for value in model.state_dict().values(): digest.update(value.detach().cpu().numpy().tobytes())
    result = dict(step=step, total_steps=total_steps, best=best, model_checksum=digest.hexdigest())
    if return_model: result['model'] = model
    return result
