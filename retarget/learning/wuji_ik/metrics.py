"""Frame-weighted metrics, trajectory deltas, and deterministic flow evaluation."""
from contextlib import nullcontext
import time
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from .data import collate_segments
from .features import SIDES
from .models import sample_flow


def autocast_context(device, bf16=True):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' and bf16 else nullcontext()


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def flow_noise(features, identities, seed, shared=True):
    values = []
    t = features.shape[1]
    for identity in identities:
        generator = torch.Generator(device=features.device).manual_seed((seed + 1000003*identity) % (2**63-1))
        x = torch.randn((1 if shared else t, 20), generator=generator, device=features.device, dtype=features.dtype)
        values.append(x.expand(t, 20))
    return torch.stack(values)


class MetricSums:
    def __init__(self):
        self.groups = {}
        self.latencies = []
        self.frames = 0

    @staticmethod
    def empty():
        return dict(frames=0, pairs=0, absolute=torch.zeros(20, dtype=torch.float64),
            squared=torch.zeros(20, dtype=torch.float64), delta_absolute=torch.zeros(20, dtype=torch.float64),
            delta_squared=torch.zeros(20, dtype=torch.float64), normalized_squared=0.)

    def add(self, prediction, batch, normalizer, elapsed):
        pred = normalizer.denormalize(prediction.float(), batch['side'])
        target = normalizer.denormalize(batch['target'].float(), batch['side'])
        for i, n in enumerate(batch['lengths'].tolist()):
            valid = batch['valid'][i, :n]
            full_error = (pred[i, :n]-target[i, :n]).double()
            error = full_error[valid].cpu()
            normalized = (prediction[i, :n].float()-batch['target'][i, :n])[valid].double().square().sum().item()
            frames = batch['frames'][i, :n]
            pairs = valid[1:] & valid[:-1] & (frames[:-1]>=0) & (frames[1:]-frames[:-1]==1)
            delta = (full_error[1:]-full_error[:-1])[pairs].cpu()
            n = len(error)
            if not n: continue
            for key in ('overall', 'side/'+SIDES[int(batch['side'][i])], 'dataset/'+batch['dataset'][i]):
                g = self.groups.setdefault(key, self.empty())
                g['frames'] += n; g['pairs'] += len(delta)
                g['absolute'] += error.abs().sum(0); g['squared'] += error.square().sum(0)
                g['delta_absolute'] += delta.abs().sum(0); g['delta_squared'] += delta.square().sum(0)
                g['normalized_squared'] += normalized
            self.frames += n
        self.latencies.append(dict(sequences=len(batch['lengths']), frames=int(batch['valid'].sum()), seconds=elapsed))

    def merged(self, others):
        result = MetricSums()
        for source in others:
            result.frames += source.frames; result.latencies.extend(source.latencies)
            for key, value in source.groups.items():
                g = result.groups.setdefault(key, self.empty())
                for name, number in value.items(): g[name] += number
        return result

    def report(self):
        groups = {}
        for key, g in self.groups.items():
            n = max(g['frames'], 1); pairs = g['pairs']
            groups[key] = dict(frames=g['frames'], consecutive_pairs=pairs,
                mae_rad=float(g['absolute'].sum()/(n*20)), rmse_rad=float((g['squared'].sum()/(n*20)).sqrt()),
                per_joint_mae_rad=(g['absolute']/n).tolist(), per_joint_rmse_rad=(g['squared']/n).sqrt().tolist(),
                normalized_mse=g['normalized_squared']/(n*20),
                delta_mae_rad=float(g['delta_absolute'].sum()/(pairs*20)) if pairs else None,
                delta_rmse_rad=float((g['delta_squared'].sum()/(pairs*20)).sqrt()) if pairs else None)
        return groups


@torch.no_grad()
def evaluate_model(model, dataset, normalizer, variant, device, batch_size=1, workers=0, bf16=True,
                   flow_steps=8, seed=42, shared_noise=True, rank=0, world_size=1):
    model.eval(); accumulator = MetricSums()
    # No DistributedSampler padding: every evaluation segment appears exactly once.
    subset = Subset(dataset, list(range(rank, len(dataset), world_size)))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        collate_fn=collate_segments, pin_memory=device.type == 'cuda',
                        generator=torch.Generator().manual_seed(seed))
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        if device.type == 'cuda': torch.cuda.synchronize(device)
        start = time.perf_counter()
        with autocast_context(device, bf16):
            if variant == 'frame_flow':
                noise = flow_noise(batch['features'], batch['segment_id'], seed, shared_noise)
                prediction = sample_flow(model, batch['features'], batch['side'], batch['lengths'],
                    steps=flow_steps, initial_noise=noise)
            else:
                prediction = model(batch['features'], batch['side'], batch['lengths'])
        if device.type == 'cuda': torch.cuda.synchronize(device)
        accumulator.add(prediction, batch, normalizer, time.perf_counter()-start)
    memory = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
    payload = dict(metrics=accumulator, memory=memory, backends=model.core.backend_report())
    gathered = [payload]
    if world_size > 1:
        gathered = [None]*world_size; dist.all_gather_object(gathered, payload)
    combined = accumulator.merged([v['metrics'] for v in gathered])
    groups = combined.report()
    if 'overall' not in groups:
        raise ValueError('No evaluation frames')
    rank_time = [sum(v['seconds'] for v in p['metrics'].latencies) for p in gathered]
    times = [v['seconds']/v['sequences'] for v in combined.latencies]
    timing = torch.tensor(times, dtype=torch.float64)
    return dict(variant=variant, information='past_and_present' if variant == 'temporal_causal' else
        ('past_present_and_future' if variant == 'temporal_bidirectional' else 'current_frame_only'),
        metrics=groups, frames=combined.frames, model_seconds_per_rank=rank_time,
        frames_per_second=combined.frames/max(max(rank_time), 1e-12),
        mean_sequence_latency_ms=1000*sum(v['seconds'] for v in combined.latencies)/sum(v['sequences'] for v in combined.latencies),
        batch_amortized_sequence_p50_ms=float(timing.quantile(.5)*1000),
        batch_amortized_sequence_p95_ms=float(timing.quantile(.95)*1000),
        peak_allocated_bytes=max(p['memory'] for p in gathered),
        attention_backends=sorted({b for p in gathered for b in p['backends']}),
        flow_steps=flow_steps if variant == 'frame_flow' else None, seed=seed,
        timing_scope='model inference including flow integration; excludes disk I/O; GPU synchronized; includes first-call compilation')
