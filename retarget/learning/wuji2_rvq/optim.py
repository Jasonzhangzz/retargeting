"""Native PyTorch Muon for hidden matrices with AdamW for remaining parameters."""
import torch


class MuonAdamW(torch.optim.Optimizer):
    """One scheduler/checkpoint interface over two disjoint native optimizers."""
    def __init__(self, model, lr=3e-4, weight_decay=1e-4):
        if not hasattr(torch.optim, 'Muon'):
            raise RuntimeError('Muon requires a PyTorch environment providing torch.optim.Muon')
        if model.config.get('joint'):
            output_weights={id(model.decoder.pose_head.weight),id(model.decoder.class_head.weight)}
        else:output_weights={id((model.classifier if hasattr(model,'classifier') else model.decoder)[-1].weight)}
        matrices, other, matrix_names, other_names = [], [], [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim == 2 and id(parameter) not in output_weights:
                matrices.append(parameter); matrix_names.append(name)
            else:
                other.append(parameter); other_names.append(name)
        super().__init__([{'params': matrices}, {'params': other}], dict(lr=lr, weight_decay=weight_decay))
        self.muon = torch.optim.Muon(self.param_groups[:1], lr=lr, weight_decay=weight_decay,
                                    momentum=.95, nesterov=True, ns_steps=5,
                                    adjust_lr_fn='match_rms_adamw')
        self.adamw = torch.optim.AdamW(self.param_groups[1:], lr=lr, weight_decay=weight_decay)
        self.param_groups = self.muon.param_groups + self.adamw.param_groups
        self.assignment = dict(muon=matrix_names, adamw=other_names)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():loss = closure()
        self.muon.step(); self.adamw.step()
        return loss

    def state_dict(self):
        return dict(kind='muon_adamw', version=1, assignment=self.assignment,
                    muon=self.muon.state_dict(), adamw=self.adamw.state_dict())

    def load_state_dict(self, value):
        if value.get('kind') != 'muon_adamw' or value.get('assignment') != self.assignment:
            raise ValueError('Optimizer kind or parameter assignment changed')
        self.muon.load_state_dict(value['muon']); self.adamw.load_state_dict(value['adamw'])
        self.param_groups = self.muon.param_groups + self.adamw.param_groups


def make_optimizer(model, kind):
    if kind == 'muon':return MuonAdamW(model)
    if kind == 'adamw':return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    raise ValueError(f'Unknown optimizer: {kind}')
