"""Input-only mesh surface queries."""

import numpy as np
import warp as wp


@wp.kernel(enable_backward=False)
def query_surface(
    mesh: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    projected: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    distance: wp.array(dtype=float),
):
    i = wp.tid()
    p = points[i]
    hit = wp.mesh_query_point(mesh, p, 1.0e6)
    q = wp.mesh_eval_position(mesh, hit.face, hit.u, hit.v)
    # Warp's mesh_get_point takes a face-vertex index, not a vertex ID.
    a = wp.mesh_get_point(mesh, 3 * hit.face)
    b = wp.mesh_get_point(mesh, 3 * hit.face + 1)
    c = wp.mesh_get_point(mesh, 3 * hit.face + 2)
    normal = wp.normalize(wp.cross(b - a, c - a))
    projected[i] = q
    normals[i] = normal
    # The nearest face's plane does not classify points outside a mesh edge.
    # Use the mesh-wide inside/outside query (ray majority) at caps/vertices.
    distance[i] = hit.sign * wp.length(p - q)


class GraspAudit:
    def __init__(self, surface, device):
        self.device = device
        vertices, faces = np.unique(
            surface.triangles.reshape(-1, 3), axis=0, return_inverse=True
        )
        self.vertices = wp.array(vertices, dtype=wp.vec3, device=device)
        self.faces = wp.array(faces, dtype=int, device=device)
        self.mesh = wp.Mesh(points=self.vertices, indices=self.faces)
        wp.synchronize_device(device)
        self.radius = float(np.linalg.norm(vertices, axis=1).max())

    def query(self, points):
        import torch

        source = wp.from_torch(points.float().contiguous(), dtype=wp.vec3)
        output = wp.empty(len(points), dtype=wp.vec3, device=self.device)
        normals = wp.empty_like(output)
        distance = wp.empty(len(points), dtype=float, device=self.device)
        wp.launch(
            query_surface,
            len(points),
            inputs=[self.mesh.id, source],
            outputs=[output, normals, distance],
            device=self.device,
            stream=wp.stream_from_torch(points.device),
        )
        return wp.to_torch(output), wp.to_torch(normals), wp.to_torch(distance)
