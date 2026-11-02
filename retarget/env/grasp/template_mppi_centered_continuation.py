"""Centered axial contacts with bounded adjacent-size motion."""
import numpy as np
import torch
from .template_mppi_centered import CenteredCylinderObjective
from .template_mppi_continuation import ContinuationObjective


class CenteredAdjacentObjective(CenteredCylinderObjective, ContinuationObjective):
    """Keep centering while bounding adjacent wrist, joints and contact motion."""
    def __init__(self, *args, parent_points, parent_primitive, **kwargs):
        self.parent_points = np.asarray(parent_points, float)
        self.parent_primitive = parent_primitive
        super().__init__(*args, **kwargs)

    def configure_object_regions(self, name, primitive):
        super().configure_object_regions(name, primitive)
        target = self.parent_points.copy()
        target[:, :2] *= primitive['radius_m']/self.parent_primitive['radius_m']
        target[:, 2] *= primitive['height_m']/self.parent_primitive['height_m']
        self.transported_points = torch.as_tensor(target, device=self.device, dtype=torch.float32)

    def evaluate(self, controls):
        cost = super().evaluate(controls)
        points = self.parts['projected_points']
        residual = torch.linalg.vector_norm(points-self.transported_points, dim=-1)
        valid = residual.amax(-1) <= .002 + 1.e-7
        penalty = (residual/.001).square().mean(-1)
        self.parts['contact_continuity'] = penalty
        self.parts['valid'] &= valid
        return torch.where(valid, cost+penalty, torch.inf)

    def diagnostics(self, q):
        d = super().diagnostics(q)
        error = np.linalg.norm(np.asarray(d['projected_points'])-self.transported_points.cpu().numpy(), axis=-1)
        d['contact_continuity'] = dict(max_transport_residual_m=float(error.max()),
            limit_m=.002, passed=bool(error.max() <= .0020001),
            parent_points=self.parent_points.tolist(), transported_points=self.transported_points.cpu().tolist())
        d['cost_parts']['contact_continuity'] = float(self.parts['contact_continuity'][0])
        return d
