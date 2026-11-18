"""Masked reconstruction/SupCon and the released HACO loss formulation."""
import torch
from torch import nn
import torch.nn.functional as F


def reconstruction_l1(prediction, target):
    # The feature converter already expresses translation in units of 0.1 m.
    return (prediction[..., :3]-target[..., :3]).abs().mean() + (
        prediction[..., 3:]-target[..., 3:]).abs().reshape(*target.shape[:2], 16, 6).mean()


def supervised_contrastive(features, labels, valid, temperature=.1):
    x, y = features[valid], labels[valid]
    if len(x) < 2:
        return features.sum()*0
    x = F.normalize(x, dim=-1)
    scores = x @ x.T / temperature
    diagonal = torch.eye(len(x), dtype=torch.bool, device=x.device)
    positive = y[:, None].eq(y[None]) & ~diagonal
    usable = positive.any(-1)
    if not usable.any():
        return features.sum()*0
    log_prob = scores - scores.masked_fill(diagonal, -torch.inf).logsumexp(-1, keepdim=True)
    return -(log_prob.masked_fill(~positive, 0).sum(-1) / positive.sum(-1).clamp_min(1))[usable].mean()


class HACOLoss(nn.Module):
    """Same scales, effective-number VCB, FP penalty, prior and smooth loss.

    Statistics are supplied by the train split, rather than loading HACO's
    original image-training statistics. Unknown labels never become negatives.
    """
    def __init__(self, decoder, counts, beta=.9999):
        super().__init__()
        self.beta = beta
        self.register_buffer('counts', torch.as_tensor(counts, dtype=torch.float32))
        if self.counts.shape != (778, 2) or not self.counts.sum():
            raise ValueError('Expected nonempty TRAIN counts [778,2]')
        self.register_buffer('R336', decoder.V_regressor_336.detach().clone())
        self.register_buffer('R84', decoder.V_regressor_84.detach().clone())
        self.register_buffer('RJ', decoder.J_regressor.detach().clone())
        self.register_buffer('means', self.counts[:, 1]/self.counts.sum(-1).clamp_min(1))

    def configure_faces(self, faces):
        faces = torch.as_tensor(faces, dtype=torch.long, device=self.counts.device)
        row = torch.cat([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
        col = torch.cat([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
        adjacency = torch.sparse_coo_tensor(torch.stack((row, col)), torch.ones_like(row, dtype=torch.float32), (778, 778),check_invariants=True).to_dense()
        adjacency += torch.eye(778, device=faces.device)
        self.register_buffer('adjacency', adjacency)
        return self

    def class_weights(self, progress, regressor=None):
        vertex = self.counts/self.counts.sum(-1, keepdim=True).clamp_min(1)*1000
        global_counts = self.counts.sum(0)
        global_counts = global_counts/global_counts.sum()*1000
        mix = .3 * min(max(float(progress), 0), 1)
        counts = (mix*vertex+(1-mix)*global_counts).clamp_min(20)
        counts = counts/counts.sum(-1, keepdim=True)*1000
        if regressor is not None:
            counts = regressor @ counts
        weights = (1-self.beta)/(1-self.beta**counts).clamp_min(1e-12)
        return weights/weights.sum(-1, keepdim=True)

    def vcb(self, logits, target, progress, regressor=None):
        weights = self.class_weights(progress, regressor)
        selected = weights[None].expand(len(target), -1, -1).gather(2, target.long()[..., None]).squeeze(-1)
        # BCEWithLogits is the numerically stable equivalent of sigmoid+BCELoss.
        return (F.binary_cross_entropy_with_logits(logits, target, reduction='none')*selected).mean() + .5*((1-target)*logits.sigmoid()).mean()

    def forward(self, prediction, target, valid, progress=0.):
        if not valid.any():
            raise ValueError('Contact training batch has no known labels')
        p = {k: v[valid] for k, v in prediction.items()}
        y = target[valid].float()
        terms = {'vcb_778': self.vcb(p['contact_out'], y, progress)}
        for name, matrix in [('336', self.R336), ('84', self.R84), ('joint', self.RJ)]:
            coarse = (y @ matrix.T > 0).float()
            terms['vcb_'+name] = self.vcb(p['contact_'+name+'_out'], coarse, progress, matrix)
        prob = p['contact_out'].sigmoid()
        terms['prior'] = (prob-self.means).abs().mean()
        contact = prob @ self.adjacency.T
        noncontact = (1-prob) @ self.adjacency.T
        isolated = (prob-contact/contact.amax(-1, keepdim=True).clamp_min(1e-6)).abs().sum(-1)
        isolated += (1-prob-noncontact/noncontact.amax(-1, keepdim=True).clamp_min(1e-6)).abs().sum(-1)
        terms['smooth'] = torch.log1p(isolated/(contact.sum(-1)+noncontact.sum(-1)+1e-3)).mean()
        terms['frame'] = F.binary_cross_entropy_with_logits(p['frame_contact_logit'].flatten(), y.any(-1).float())
        terms['total'] = sum(terms[k] for k in ['vcb_778','vcb_336','vcb_84','vcb_joint','smooth','frame']) + .1*terms['prior']
        return terms
