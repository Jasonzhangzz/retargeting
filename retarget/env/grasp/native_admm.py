"""Original GraspADMM contact updates and GPU contact-QP interface."""

from __future__ import annotations

from dataclasses import dataclass

import importlib

import math

import torch

import warp as wp

from .graspadmm_source import require_upstream


from retarget.solver.graspadmm import Settings, DEFAULTS


class ContactQP:
    def __init__(self, contacts, device, settings=Settings()):
        self.source = require_upstream()
        self.settings = settings
        reference = self.source.qp.ContactQPTorch(list(settings.opt_miu_coef))
        reference.set_num_contact(contacts)
        self.G = torch.as_tensor(reference.G_matrix, device=device, dtype=torch.float32)
        self.h = torch.as_tensor(reference.h_matrix, device=device, dtype=torch.float32)
        self.E = torch.as_tensor(reference.E_matrix, device=device, dtype=torch.float32)
        self.backend = importlib.import_module("dexonomy.qp.qp_batched")
        self.solvers = {}

    def matrices(self, points, inward, external, center):
        b, n, _ = points.shape
        rot = self.source.qp.torch_normal_to_rot(
            inward.reshape(-1, 3).double()
        ).reshape(b, n, 3, 3)
        grasp = points.new_zeros((b, n, 6, 6))
        grasp[..., :3, :3] = rot
        grasp[..., 3:, 3:] = rot
        grasp[..., 3:, :3] = torch.linalg.cross(
            (points - center[:, None]).double()[..., None].expand_as(rot), rot, dim=-2
        )
        transform = grasp @ self.E
        matrix = transform.transpose(1, 2).reshape(b, 6, -1)
        return (
            matrix.transpose(-1, -2) @ matrix,
            (external[:, None] @ matrix).squeeze(1),
            matrix,
        )

    def solve(self, points, inward, external, center):
        with torch.no_grad():
            H, g, matrix = self.matrices(points, inward, external, center)
            batch, nx = g.shape
            nc = len(self.h)
            if batch not in self.solvers:
                solver = self.backend._RELUQP(batch, nx, nc, points.device)
                solver.eps_abs = self.settings.qp_tolerance / math.sqrt(max(nx, nc))
                solver.max_step = self.settings.qp_max_steps
                self.solvers[batch] = solver
            solver = self.solvers[batch]
            solver.rho_ind.copy_((solver.rhos - 0.1).abs().argmin().expand(batch))
            G = self.G.expand(batch, -1, -1)
            h = self.h.expand(batch, -1)
            lower = torch.full_like(h, -torch.inf)
            alpha = solver.solve(H, g, G, lower, h).clone()
            dual = solver.output[:, nx + nc :, 0].clone()
            slack = h - (G @ alpha[..., None]).squeeze(-1)
            primal = (-slack).clamp_min(0).amax(-1)
            stationarity = (
                (
                    (H @ alpha[..., None]).squeeze(-1)
                    + g
                    + (G.transpose(-1, -2) @ dual[..., None]).squeeze(-1)
                )
                .abs()
                .amax(-1)
            )
            complementarity = (dual * slack).abs().amax(-1)
            dual_violation = (-dual).clamp_min(0).amax(-1)
            finite = torch.isfinite(alpha).all(-1) & torch.isfinite(dual).all(-1)
            valid = (
                finite
                & (primal <= 1.0e-5)
                & (stationarity <= 1.0e-5)
                & (complementarity <= 1.0e-5)
                & (dual_violation <= 1.0e-5)
            )
        return dict(
            alpha=alpha,
            valid=valid,
            primal=primal,
            stationarity=stationarity,
            complementarity=complementarity,
            dual_violation=dual_violation,
        )

    def error(self, points, inward, external, center, alpha):
        _, _, matrix = self.matrices(points, inward, external, center)
        return ((matrix @ alpha.detach()[..., None]).squeeze(-1) + external).norm(
            dim=-1
        )


def plane_projection(points, triangles, normals):
    a, b, c = triangles.unbind(-2)
    ab, ac, ap = b - a, c - a, points - a
    dot00 = (ab * ab).sum(-1)
    dot01 = (ab * ac).sum(-1)
    dot02 = (ab * ap).sum(-1)
    dot11 = (ac * ac).sum(-1)
    dot12 = (ac * ap).sum(-1)
    inverse = 1 / (dot00 * dot11 - dot01 * dot01 + 1.0e-20)
    u = (dot11 * dot02 - dot01 * dot12) * inverse
    v = (dot00 * dot12 - dot01 * dot02) * inverse
    projected = a + u[..., None] * ab + v[..., None] * ac
    na, nb, nc = normals.unbind(-2)
    normal = (1 - u - v)[..., None] * na + u[..., None] * nb + v[..., None] * nc
    normal = normal / (normal.norm(dim=-1, keepdim=True) + 1.0e-20)
    return projected, normal


def update_dual(lam, hand_points, object_points):
    old_large = lam.norm(dim=-1, keepdim=True) > 0.02
    return torch.where(old_large, 0.0, lam + hand_points - object_points)


def object_step(points, hand, lam, surfaces, qp, external, center, settings=Settings()):
    cp = points.detach().clone().requires_grad_(True)
    _, _, _, faces, _ = surfaces.query(
        cp.detach(), torch.arange(len(cp), device=cp.device)
    )
    triangles, normals = surfaces.triangle_data(faces)
    projected, outward = plane_projection(cp, triangles, normals)
    solved = qp.solve(projected.detach(), -outward.detach(), external, center)
    error = qp.error(projected, -outward, external, center, solved["alpha"])
    loss = error.square() if settings.square_res else error
    gradient = torch.autograd.grad(loss.sum(), cp)[0]
    gradient = gradient + settings.rho_admm * (cp.detach() - hand - lam)
    inward = -outward.detach()
    gradient = gradient - (gradient * inward).sum(-1, keepdim=True) * inward
    candidate = cp.detach() - settings.lr_obj * gradient
    updated = surfaces.query(candidate, torch.arange(len(cp), device=cp.device))[0]
    finite = torch.isfinite(updated).all((1, 2)) & torch.isfinite(gradient).all((1, 2))
    return updated.detach(), solved, gradient.detach(), finite


class ObjectSurfaces:
    def __init__(self, vertices, faces, device):
        import trimesh
        from .mano_solver import SurfaceBatch

        self.vertices = [
            torch.as_tensor(v, device=device, dtype=torch.float32) for v in vertices
        ]
        self.faces = [
            torch.as_tensor(f, device=device, dtype=torch.long) for f in faces
        ]
        # Match trimesh's original vertex-normal construction during static asset setup.
        self.normals = [
            torch.as_tensor(
                trimesh.Trimesh(v, f, process=False).vertex_normals.copy(),
                device=device,
                dtype=torch.float32,
            )
            for v, f in zip(vertices, faces)
        ]
        geometries = [
            dict(vertices=v, faces=f, watertight=True) for v, f in zip(vertices, faces)
        ]
        pose = torch.zeros((len(vertices), 7), device=device)
        pose[:, 3] = 1
        self.surface = SurfaceBatch(
            geometries, list(range(len(vertices))), pose, [1.0] * len(vertices), device
        )
        max_faces = max(len(f) for f in faces)
        self.triangles = torch.zeros((len(vertices), max_faces, 3, 3), device=device)
        self.triangle_normals = torch.zeros_like(self.triangles)
        centers = []
        radii = []
        for i, (v, f, n) in enumerate(zip(self.vertices, self.faces, self.normals)):
            t = v[f]
            self.triangles[i, : len(f)] = t
            self.triangle_normals[i, : len(f)] = n[f]
            volume = (t[:, 0] * torch.linalg.cross(t[:, 1], t[:, 2])).sum(-1) / 6
            c = (volume[:, None] * t.sum(1) / 4).sum(0) / volume.sum()
            centers.append(c)
            radii.append((v - c).norm(dim=-1).max().clamp_min(0.005))
        self.centers = torch.stack(centers)
        self.radii = torch.stack(radii)

    def query(self, points, rows):
        return self.surface.query(points, rows)

    def triangle_data(self, faces):
        rows = torch.arange(len(faces), device=faces.device)[:, None]
        safe = faces.clamp_min(0)
        return self.triangles[rows, safe], self.triangle_normals[rows, safe]


@wp.func
def triangle_barycentric(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3):
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return wp.vec3(1.0, 0.0, 0.0)
    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return wp.vec3(0.0, 1.0, 0.0)
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return wp.vec3(1.0 - v, v, 0.0)
    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return wp.vec3(0.0, 0.0, 1.0)
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return wp.vec3(1.0 - w, 0.0, w)
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return wp.vec3(0.0, 1.0 - w, w)
    denominator = 1.0 / (va + vb + vc)
    v = vb * denominator
    w = vc * denominator
    return wp.vec3(1.0 - v - w, v, w)
