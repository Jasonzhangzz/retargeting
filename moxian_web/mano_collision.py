"""Full-surface collision witnesses, independent of tactile/native contact gates.

Closed anatomical volumes establish the sign. Only original triangles supply
distances and gradients; artificial caps are never contact surfaces. Discrete
queries are refreshed outside the differentiable least-squares step.
"""
from __future__ import annotations

import numpy as np
import torch
import warp as wp

from .hand_anatomy import closed_surface_parts
from .contact import FINGERS
from scripts.calibrate_wuji_mano import CHAINS

HARD_PENETRATION_M = .002
COLLISION_MARGIN_M = .001


@wp.kernel(enable_backward=False)
def surface_queries(solids: wp.array(dtype=wp.uint64), meshes: wp.array(dtype=wp.uint64),
                    points: wp.array(dtype=wp.vec3),
                    face: wp.array2d(dtype=int), uv: wp.array2d(dtype=wp.vec2),
                    distance: wp.array2d(dtype=float)):
    sample, part = wp.tid()
    q = wp.mesh_query_point_sign_winding_number(solids[part], points[sample], 1.)
    hit = wp.mesh_query_point_no_sign(meshes[part], points[sample], 1.)
    face[sample, part] = -1
    distance[sample, part] = 1.
    if q.result and hit.result:
        p = wp.mesh_eval_position(meshes[part], hit.face, hit.u, hit.v)
        closed = wp.mesh_eval_position(solids[part], q.face, q.u, q.v)
        d = wp.length(points[sample] - p)
        # A point behind a virtual seam cap is not penetration of real skin.
        inside = q.sign < 0. and d <= wp.length(points[sample] - closed) + .0005
        distance[sample, part] = wp.where(inside, -d, d)
        face[sample, part] = hit.face
        uv[sample, part] = wp.vec2(hit.u, hit.v)


from .warp_runtime import serialized


class SurfaceCollision:
    """Vertices and every face centroid against all 16 anatomical segments.

    Topological two-ring neighbours are excluded locally, rather than removing
    entire palm/proximal or adjacent-finger pairs. Same-finger distant segments
    are included. Dense queries have no native-contact candidate buffer.
    """
    @serialized
    def __init__(self, model):
        self.model = model
        self.faces = np.asarray(model.faces, np.int32)
        self.parts = np.array([np.bincount(model.labels[f], minlength=16).argmax()
                               for f in self.faces], np.int32)
        self.part_faces = [np.flatnonzero(self.parts == p) for p in range(16)]
        n = len(model.rest_np)
        self.source_ids = np.concatenate((np.repeat(np.arange(n)[:, None], 3, axis=1), self.faces))
        self.source_bc = np.concatenate((np.tile([1., 0., 0.], (n, 1)), np.full((len(self.faces), 3), 1/3)))
        self.source_part = np.concatenate((model.labels, self.parts))
        neighbours = [set([i]) for i in range(n)]
        for f in self.faces:
            for i in f: neighbours[i].update(map(int, f))
        rings = [set().union(*(neighbours[j] for j in ns)) for ns in neighbours]
        self.excluded = np.zeros((len(self.source_ids), len(self.faces)), bool)
        for i, ids in enumerate(self.source_ids):
            near = set().union(*(rings[j] for j in ids))
            self.excluded[i] = np.isin(self.faces, list(near)).any(1)
        self.meshes = []
        self.solids = []
        self.buffers = []
        self.loops = []
        self.device = str(model.device)
        with wp.ScopedDevice(self.device):
            self.vertices = wp.array(model.rest_np, dtype=wp.vec3)
            for p, (points, closed, loops) in enumerate(closed_surface_parts(
                    model.rest_np, self.faces, self.parts, 16)):
                buf = wp.array(points, dtype=wp.vec3)
                self.buffers.append(buf); self.loops.append(loops)
                self.solids.append(wp.Mesh(points=buf, indices=wp.array(closed.ravel(), dtype=int), support_winding_number=True))
                self.meshes.append(wp.Mesh(points=self.vertices, indices=wp.array(self.faces[self.part_faces[p]].ravel(), dtype=int)))
            self.solid_ids = wp.array([x.id for x in self.solids], dtype=wp.uint64)
            self.mesh_ids = wp.array([x.id for x in self.meshes], dtype=wp.uint64)
            count = len(self.source_ids)
            self.points = wp.zeros(count, dtype=wp.vec3)
            self.face = wp.zeros((count, 16), dtype=int)
            self.uv = wp.zeros((count, 16), dtype=wp.vec2)
            self.distance = wp.zeros((count, 16), dtype=float)

    @serialized
    def query(self, vertices, margin=COLLISION_MARGIN_M, tensor_device=None):
        v = np.asarray(vertices, np.float32)
        if not np.isfinite(v).all(): raise ValueError('Non-finite MANO collision surface')
        wp.to_torch(self.vertices).copy_(torch.as_tensor(v, device=self.device))
        for mesh in self.meshes: mesh.refit()
        for mesh, buf, loops in zip(self.solids, self.buffers, self.loops):
            p = np.concatenate((v, np.array([v[loop].mean(0) for loop in loops]).reshape(-1, 3)))
            wp.to_torch(buf).copy_(torch.as_tensor(p, device=self.device)); mesh.refit()
        points = (v[self.source_ids] * self.source_bc[..., None]).sum(1)
        wp.to_torch(self.points).copy_(torch.as_tensor(points, device=self.device, dtype=torch.float32))
        wp.launch(surface_queries, (len(points), 16), inputs=[self.solid_ids, self.mesh_ids, self.points,
                  self.face, self.uv, self.distance], device=self.device)
        face, uv, distance = self.face.numpy(), self.uv.numpy(), self.distance.numpy()
        if not np.isfinite(distance).all(): raise ValueError('Invalid MANO collision query')
        if not np.all(face>=0): raise ValueError('Incomplete MANO collision surface query')
        samples, parts = np.where((face >= 0) & (distance < margin))
        faces = np.array([self.part_faces[p][face[s, p]] for s, p in zip(samples, parts)], np.int64)
        keep = (self.source_part[samples] != parts) & ~self.excluded[samples, faces]
        samples, parts, faces = samples[keep], parts[keep], faces[keep]
        bc = np.column_stack((uv[samples, parts], 1 - uv[samples, parts].sum(1)))
        target = (v[self.faces[faces]] * bc[..., None]).sum(1)
        delta = points[samples] - target
        normal = delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-9)
        gaps = distance[samples, parts]
        normal *= np.where(gaps < 0, -1., 1.)[:, None]
        # At exact contact use the triangle normal, not a zero derivative.
        tri = v[self.faces[faces]]
        nn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-9)
        normal[np.linalg.norm(delta, axis=1) < 1e-8] = nn[np.linalg.norm(delta, axis=1) < 1e-8]
        depths = {}
        owner = {0: 'palm', **{j: FINGERS[f] for f, c in enumerate(CHAINS) for j in c}}
        for s, p, gap in zip(samples, parts, gaps):
            pair = '_'.join(sorted((owner[int(self.source_part[s])], owner[int(p)])))
            depths[pair] = max(depths.get(pair, 0.), max(0., -float(gap)))
        maximum = max(depths.values(), default=0.)
        report = dict(max_penetration_m=maximum, pair_depths_mm={k: d*1000 for k, d in depths.items()},
                      constraints_satisfied=maximum <= HARD_PENETRATION_M,
                      query_valid=True, samples=len(points), anatomical_segments=16,
                      method='closed_segment_winding_vertices_and_face_centroids',
                      continuous_intersection_proof=False)
        device=self.model.device if tensor_device is None else tensor_device
        t = lambda x, dtype=torch.float32: torch.as_tensor(x, device=device, dtype=dtype)
        witness = dict(source_ids=t(self.source_ids[samples], torch.long), source_bc=t(self.source_bc[samples]),
                       target_ids=t(self.faces[faces], torch.long), target_bc=t(bc), normal=t(normal))
        return witness, report


def collision_residual(vertices, witness):
    """Signed separation with frozen correspondences and live MANO vertices."""
    a = (vertices[witness['source_ids']] * witness['source_bc'][..., None]).sum(-2)
    b = (vertices[witness['target_ids']] * witness['target_bc'][..., None]).sum(-2)
    gap = ((a - b) * witness['normal']).sum(-1)
    return torch.relu(COLLISION_MARGIN_M - gap) / .001
