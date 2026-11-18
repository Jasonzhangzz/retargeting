"""Muon for Delta hidden matrices; Adam for physical parameters and heads.

The flat physical vector retains its original parameterization. Muon is never
silently selected for vectors or emulated by reshaping unrelated coordinates.
"""
import torch


class TrainingOptimizer:
    def __init__(self,geometry,policy=None,selection='adam'):
        if selection not in ('adam','muon'):raise ValueError('未知训练优化器')
        parameters=list(policy.named_parameters()) if policy is not None else []
        hidden=[]
        if selection=='muon':
            if policy is None:raise ValueError('Muon 需要 MANO＋Delta 的二维隐藏层；个体几何向量使用 Adam')
            if not hasattr(torch.optim,'Muon'):raise ValueError('当前 PyTorch 不提供 Muon')
            layers=[m for m in policy.modules() if isinstance(m,torch.nn.Linear)]
            hidden=[m.weight for m in layers[:-1] if m.weight.requires_grad]
            if not hidden:raise ValueError('Delta 网络缺少可用的二维隐藏层')
        hidden_ids={id(p) for p in hidden};rest=[p for _,p in parameters if id(p) not in hidden_ids and p.requires_grad]
        groups=[dict(params=[geometry],lr=.01)]
        if rest:groups.append(dict(params=rest,lr=.001))
        self.adam=torch.optim.Adam(groups)
        self.muon=torch.optim.Muon(hidden,lr=.001,weight_decay=0.,momentum=.95,adjust_lr_fn='match_rms_adamw') if hidden else None
        self.name='Muon+Adam' if hidden else 'Adam';self.selection=selection
        self.geometry_lr=.01;self.policy_lr=.001 if parameters else 0.;self.muon_lr=.001 if hidden else 0.
        self.assignment=dict(geometry='Adam',geometry_parameters=geometry.numel(),
            policy_muon=[name for name,p in parameters if id(p) in hidden_ids],
            policy_adam=[name for name,p in parameters if id(p) not in hidden_ids and p.requires_grad],
            muon_lr=self.muon_lr,adam_geometry_lr=self.geometry_lr,adam_policy_lr=self.policy_lr,
            muon_weight_decay=0.,muon_momentum=.95,muon_adjust_lr_fn='match_rms_adamw',geometry_reshaped=False)

    def zero_grad(self,set_to_none=True):
        self.adam.zero_grad(set_to_none=set_to_none)
        if self.muon is not None:self.muon.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.adam.step()
        if self.muon is not None:self.muon.step()
