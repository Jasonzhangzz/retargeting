"""Native Muon for hidden matrices, AdamW for heads/embeddings/1D parameters."""
import math
import torch


def make_optimizers(model, muon_lr=.02, adamw_lr=.0003, weight_decay=.01):
    if not hasattr(torch.optim, 'Muon'):
        raise RuntimeError('Native torch.optim.Muon requires the configured PyTorch 2.10+ environment')
    hidden, decay, no_decay = [], [], []
    names = {'muon': [], 'adamw_decay': [], 'adamw_no_decay': []}
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if '.blocks.' in name and p.ndim == 2:
            hidden.append(p); names['muon'].append(name)
        elif p.ndim < 2 or 'norm' in name or 'embedding' in name or 'readout_token' in name:
            no_decay.append(p); names['adamw_no_decay'].append(name)
        else:
            decay.append(p); names['adamw_decay'].append(name)
    assigned = hidden + decay + no_decay
    if len({id(p) for p in assigned}) != len(assigned):
        raise ValueError('Optimizer groups overlap')
    muon = torch.optim.Muon(hidden, lr=muon_lr, momentum=.95, nesterov=True, weight_decay=weight_decay)
    adamw = torch.optim.AdamW([dict(params=decay, weight_decay=weight_decay), dict(params=no_decay, weight_decay=0.)], lr=adamw_lr)
    return [muon, adamw], names


def make_schedulers(optimizers, total_steps, warmup_fraction=.02, continuation=None):
    warmup = max(1, int(total_steps*warmup_fraction))
    def factor(step):
        if step < warmup: return (step+1)/warmup
        progress = min(1., (step-warmup)/max(1, total_steps-warmup))
        return .5*(1+math.cos(math.pi*progress))
    if continuation is None:
        return [torch.optim.lr_scheduler.LambdaLR(o, factor) for o in optimizers]
    start = continuation['step']; end = continuation['total_steps']
    if end != total_steps or not 0 <= start < end:
        raise ValueError('Invalid extended scheduler horizon')
    def extended_factor(step, initial):
        progress = min(1., max(0., (step-start)/(end-start)))
        return initial*.5*(1+math.cos(math.pi*progress))
    return [torch.optim.lr_scheduler.LambdaLR(o, [lambda step, initial=f: extended_factor(step, initial)
            for f in factors]) for o, factors in zip(optimizers, continuation['factors'], strict=True)]
