"""An MLP of (wrist-space x, joint configuration q, link ID), without FK.

Only ancestors of a target link condition its rigid geometry. This mask is a
kinematic dependency, not an analytic coordinate transform or a static SDF.
All transforms, positions and distances are learned from articulated samples.
"""
import torch
from torch import nn


class ConfigurationSDF(nn.Module):
    def __init__(self, lower, upper, joint_mask, width=256, depth=4, embedding=16):
        super().__init__()
        self.config = dict(lower=list(lower), upper=list(upper), joint_mask=joint_mask,
                           width=width, depth=depth, embedding=embedding)
        self.register_buffer('lower', torch.tensor(lower, dtype=torch.float32))
        self.register_buffer('upper', torch.tensor(upper, dtype=torch.float32))
        self.register_buffer('joint_mask', torch.tensor(joint_mask, dtype=torch.float32))
        if self.joint_mask.shape != (21, 20) or not torch.all(self.upper > self.lower):
            raise ValueError('Expected 21 links and 20 nondegenerate joint ranges')
        if width < 8 or depth < 2:
            raise ValueError('MLP width >= 8 and depth >= 2 required')
        self.links = nn.Embedding(21, embedding)
        layers = [nn.Linear(3 + 60 + embedding, width), nn.Softplus(beta=10)]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.Softplus(beta=10)])
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, q, points, links):
        """q[...,20], points[...,3], links[...], broadcastable; returns metres.

        For B poses and N queries use q[:,None,:], points[B,N,3], links[B,N].
        Negative is inside. There is no output clipping, so x/q gradients survive.
        """
        shape = torch.broadcast_shapes(q.shape[:-1], points.shape[:-1], links.shape)
        q = q.expand(*shape, 20)
        points = points.expand(*shape, 3)
        links = links.long().expand(shape)
        normalized = 2 * (q - self.lower) / (self.upper - self.lower) - 1
        mask = self.joint_mask[links]
        pose = torch.cat((normalized * mask, q.sin() * mask, q.cos() * mask), -1)
        features = torch.cat((points / .1, pose, self.links(links)), -1)
        return .1 * self.network(features).squeeze(-1)

    def per_link(self, q, points):
        """q[B,20], wrist points[B,N,3] -> distances[B,N,21]."""
        links = torch.arange(21, device=points.device)
        return self(q[:, None, None, :], points[:, :, None, :], links)

    def union(self, q, points):
        """Minimum link field: correct union sign; overlap interiors aren't exact SDF."""
        return self.per_link(q, points).amin(-1)


def load_model(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model = ConfigurationSDF(**saved['model_config']).to(device)
    model.load_state_dict(saved['model'])
    model.eval()
    return model, saved
