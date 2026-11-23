"""Muon for dense matrices; momentum SGD for embeddings and scalar vectors."""
import math
import torch
from torch import nn


class MuonSGD:
    """Checkpointable composition with no Adam moments or Adam fallback."""
    def __init__(self,model,lr=5.e-4,*,pretrained_lr=None,contact_lr=None,weight_decay=.01,vector_lr_scale=1.):
        if not hasattr(torch.optim,'Muon'):raise RuntimeError('This run requires torch.optim.Muon')
        embeddings={id(p) for module in model.modules() if isinstance(module,nn.Embedding) for p in module.parameters(recurse=False)}
        groups={'muon':{},'sgd':{}};self.assignment={}
        for name,p in model.named_parameters():
            if not p.requires_grad:continue
            rate=pretrained_lr if pretrained_lr is not None and name.startswith('pretrained.') else contact_lr if contact_lr is not None else lr
            kind='muon' if p.ndim==2 and id(p) not in embeddings else 'sgd'
            if kind=='sgd':rate*=vector_lr_scale
            groups[kind].setdefault(float(rate),[]).append(p);self.assignment[name]=kind
        self.optimizers={}
        for kind,by_lr in groups.items():
            if not by_lr:continue
            params=[dict(params=params,lr=rate) for rate,params in by_lr.items()]
            if kind=='muon':
                self.optimizers[kind]=torch.optim.Muon(params,lr=lr,momentum=.95,nesterov=True,
                    weight_decay=weight_decay,adjust_lr_fn='match_rms_adamw')
            else:
                self.optimizers[kind]=torch.optim.SGD(params,lr=lr,momentum=.95,nesterov=True,weight_decay=weight_decay)

    @property
    def param_groups(self):return [g for optimizer in self.optimizers.values() for g in optimizer.param_groups]

    def zero_grad(self,set_to_none=True):
        for optimizer in self.optimizers.values():optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        for optimizer in self.optimizers.values():optimizer.step()

    def state_dict(self):
        return dict(kind='muon_sgd_v1',assignment=self.assignment,
            optimizers={kind:optimizer.state_dict() for kind,optimizer in self.optimizers.items()})

    def load_state_dict(self,state):
        if state.get('kind')!='muon_sgd_v1' or state['assignment']!=self.assignment or set(state['optimizers'])!=set(self.optimizers):
            raise ValueError('Muon/SGD optimizer layout differs from checkpoint; Adam checkpoints cannot be resumed')
        for kind,optimizer in self.optimizers.items():optimizer.load_state_dict(state['optimizers'][kind])


def policy_optimizer(model,kind='adamw'):
    if kind=='muon_sgd':return MuonSGD(model,lr=5.e-4)
    if kind=='adamw':return torch.optim.AdamW(model.parameters(),lr=1.e-4)
    raise ValueError('Unknown policy optimizer')


def cosine_multiplier(completed,total,warmup=5,minimum=.1):
    """Update-count schedule: warm up, then decay to a nonzero final LR."""
    if total<1 or completed<0:raise ValueError('Invalid optimizer schedule cursor')
    warmup=min(warmup,total)
    step=min(completed+1,total)
    if step<=warmup:return step/max(1,warmup)
    progress=(step-warmup)/max(1,total-warmup)
    return minimum+(1-minimum)*.5*(1+math.cos(math.pi*progress))


def scale_learning_rates(optimizer,multiplier):
    for group in optimizer.param_groups:
        group.setdefault('initial_lr',group['lr'])
        group['lr']=group['initial_lr']*multiplier
