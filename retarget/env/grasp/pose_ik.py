"""Bounded GPU LM for pose initialization and shared surface-region queries."""

from dataclasses import dataclass


import math


import numpy as np

import torch


from .audit import GraspAudit

from ..geometry import ObjectMeshSurface

POLICY = "solved_canonical_contacts_cuda_lm_v2"


@dataclass(frozen=True)
class SizeIKSettings:
    enabled: bool = True
    iterations: int = 32
    starts: int = 4
    damping: float = 0.01
    bootstrap_canonical: bool = True
    canonical_grasp_directory: str | None = None

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("size_ik.enabled must be boolean")
        if not isinstance(self.bootstrap_canonical, bool):
            raise ValueError("size_ik.bootstrap_canonical must be boolean")
        if self.canonical_grasp_directory is not None and not isinstance(
            self.canonical_grasp_directory, str
        ):
            raise ValueError("size_ik.canonical_grasp_directory must be a path string")
        for name in ("iterations", "starts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"size_ik.{name} must be a positive integer")
        if (
            isinstance(self.damping, bool)
            or not math.isfinite(self.damping)
            or self.damping <= 0
        ):
            raise ValueError("size_ik.damping must be finite and positive")


class CylinderSideProjection:
    """Keep specified finger slots on the actual lateral mesh triangles."""

    def __init__(self, surface, device, slots):
        triangles = surface.triangles[np.abs(surface.normals[:, 2]) < 0.25]
        if not len(triangles):
            raise ValueError("Cylinder side constraint has no lateral triangles")
        self.audit = GraspAudit(
            ObjectMeshSurface(
                triangles.reshape(-1, 3), np.arange(len(triangles) * 3).reshape(-1, 3)
            ),
            str(device),
        )
        self.slots = torch.as_tensor(slots, device=device, dtype=torch.long)

    def apply(self, points, projected, normals, gap):
        p, n, _ = self.audit.query(points[:, self.slots].flatten(0, 1))
        p = p.reshape(len(points), len(self.slots), 3)
        n = n.reshape_as(p)
        distance = torch.linalg.vector_norm(points[:, self.slots] - p, dim=-1)
        signed = torch.where(gap[:, self.slots] < 0, -distance, distance)
        projected, normals, gap = projected.clone(), normals.clone(), gap.clone()
        projected[:, self.slots], normals[:, self.slots], gap[:, self.slots] = (
            p,
            n,
            signed,
        )
        return projected, normals, gap


def bounded_lm(residual, seeds, low, high, settings, oracle=None):
    """Independent bounded LM acceptance for a batch of hand seeds."""
    q = seeds.clamp(low, high)
    evaluate = torch.vmap(residual)
    jacobian = torch.vmap(torch.func.jacfwd(residual))
    frozen = oracle(q) if oracle else None
    r = evaluate(q, frozen) if oracle else evaluate(q)
    cost = r.square().sum(-1)
    damping = q.new_full((len(q),), settings.damping)
    history = [cost.detach().clone()]
    eye = torch.eye(q.shape[-1], device=q.device, dtype=q.dtype)
    for _ in range(settings.iterations):
        j = jacobian(q, frozen) if oracle else jacobian(q)
        h = j.transpose(-1, -2) @ j
        gradient = (j.transpose(-1, -2) @ r[..., None])[..., 0]
        diagonal = h.diagonal(dim1=-2, dim2=-1).clamp_min(1.0)
        matrix = h + torch.diag_embed(damping[:, None] * diagonal) + 1.0e-9 * eye
        step, info = torch.linalg.solve_ex(matrix, -gradient)
        finite = (info == 0) & torch.isfinite(step).all(-1)
        candidate = (q + torch.where(finite[:, None], step, 0.0)).clamp(low, high)
        candidate_frozen = oracle(candidate) if oracle else None
        candidate_r = (
            evaluate(candidate, candidate_frozen) if oracle else evaluate(candidate)
        )
        candidate_cost = candidate_r.square().sum(-1)
        accepted = finite & torch.isfinite(candidate_cost) & (candidate_cost < cost)
        q = torch.where(accepted[:, None], candidate, q)
        r = torch.where(accepted[:, None], candidate_r, r)
        cost = torch.where(accepted, candidate_cost, cost)
        if oracle:
            frozen = {
                key: torch.where(
                    accepted.reshape(len(q), *([1] * (value.ndim - 1))),
                    candidate_frozen[key],
                    value,
                )
                for key, value in frozen.items()
            }
        damping = torch.where(accepted, damping * 0.5, damping * 4).clamp(1.0e-8, 1.0e8)
        history.append(cost.detach().clone())
    return q.detach(), cost.detach(), torch.stack(history).detach()
