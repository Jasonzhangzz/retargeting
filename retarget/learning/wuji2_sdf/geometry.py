"""Native mesh teacher. Exact FK is used only for labels and independent audits."""
import numpy as np
import torch
import warp as wp

from moxian_web.wuji_surface import WujiSurface
from retarget.baseline.card.collision import WujiMeshCollision
from retarget.baseline.geort.pose_domain import extension_domain
from retarget.env.grasp.pose_gpu import ModelFK


@wp.kernel(enable_backward=False)
def _label(meshes: wp.array(dtype=wp.uint64), points: wp.array2d(dtype=wp.vec3),
           links: wp.array2d(dtype=int), positions: wp.array2d(dtype=wp.vec3),
           rotations: wp.array2d(dtype=wp.mat33), distance: wp.array2d(dtype=float),
           valid: wp.array2d(dtype=int)):
    b, i = wp.tid()
    link = links[b, i]
    local = wp.transpose(rotations[b, link]) * (points[b, i] - positions[b, link])
    query = wp.mesh_query_point_sign_winding_number(meshes[link], local, 2.)
    if query.result:
        hit = wp.mesh_eval_position(meshes[link], query.face, query.u, query.v)
        distance[b, i] = query.sign * wp.length(local - hit)
        valid[b, i] = 1


class Geometry:
    def __init__(self, device='cpu'):
        self.device = str(device)
        self.surface = WujiSurface()
        self.model = self.surface.model
        self.fk = ModelFK(self.model, device)
        self.collision = WujiMeshCollision(self.surface, device)
        self.domain = extension_domain(self.model.jnt_range[6:, 0], self.model.jnt_range[6:, 1])
        self.mask = np.zeros((21, 20), np.float32)
        for owner, body in enumerate(self.surface.body_ids):
            while body:
                for j in range(int(self.model.body_jntnum[body])):
                    address = int(self.model.jnt_qposadr[self.model.body_jntadr[body] + j]) - 6
                    if address >= 0:
                        self.mask[owner, address] = 1
                body = int(self.model.body_parentid[body])
        self.joint_names = [self.model.joint(i).name for i in range(6, self.model.njnt)]
        # Conservative fixed-wrist box covering all articulated link surfaces.
        reach = 0.
        for owner, body in enumerate(self.surface.body_ids):
            radius = np.linalg.norm(self.surface.meshes[owner].vertices, axis=1).max()
            while body:
                radius += np.linalg.norm(self.model.body_pos[body])
                body = int(self.model.body_parentid[body])
            reach = max(reach, radius)
        self.extent = float(reach + .03)
        self.allowed = self.collision.allowed.numpy().astype(bool)

    def spec(self):
        return dict(asset_sha256=self.surface.asset_hash, joint_names=self.joint_names,
                    link_names=self.collision.names, body_names=[self.model.body(b).name for b in self.surface.body_ids],
                    joint_mask=self.mask.tolist(), domain=self.domain,
                    frame='native_model_world_with_six_wrist_dofs_zero', units='m', q_units='rad',
                    sign='negative_inside_positive_outside',
                    label_method='native_triangles_warp_winding_number',
                    watertight=[bool(m.is_watertight) for m in self.surface.meshes],
                    collision_method='native_vertices_face_centres_bidirectional_with_mechanical_exclusions',
                    continuous_collision_proof=False, global_extent_m=self.extent)

    @torch.no_grad()
    def frames(self, q):
        q = torch.as_tensor(q, dtype=torch.float32, device=self.device)
        return self.fk.frames(torch.cat((q.new_zeros((len(q), 6)), q), -1))

    @torch.no_grad()
    def label(self, points, links, frames):
        points = torch.as_tensor(points, dtype=torch.float32, device=self.device).contiguous()
        links = torch.as_tensor(links, dtype=torch.int32, device=self.device).contiguous()
        p, r = (x[:, self.surface.body_ids].contiguous() for x in frames)
        shape = links.shape
        distance = wp.zeros(shape, dtype=float, device=self.device)
        valid = wp.zeros(shape, dtype=int, device=self.device)
        wp.launch(_label, shape, inputs=[self.collision.mesh_ids, wp.from_torch(points, dtype=wp.vec3),
                  wp.from_torch(links), wp.from_torch(p, dtype=wp.vec3), wp.from_torch(r, dtype=wp.mat33),
                  distance, valid], device=self.device)
        if not bool(wp.to_torch(valid).all()):
            raise ValueError('Mesh SDF label query failed; refusing invalid training labels')
        return wp.to_torch(distance).cpu().numpy()

    @staticmethod
    def surface_points(mesh, count, rng):
        faces = rng.choice(len(mesh.faces), count, p=mesh.area_faces / mesh.area)
        uv = rng.random((count, 2))
        uv[uv.sum(-1) > 1] = 1 - uv[uv.sum(-1) > 1]
        return (mesh.vertices[mesh.faces[faces]] * np.c_[uv, 1 - uv.sum(-1)][:, :, None]).sum(1)

    @torch.no_grad()
    def queries(self, frames, per_link, rng):
        """Per link: 50% near surface, 25% bbox, 12.5% global, rest cross-link.

        Fresh area-uniform triangle samples at every pose. Cross-link queries
        have physical positions on another nonconnected moving link.
        """
        p, r = (x[:, self.surface.body_ids].cpu().numpy() for x in frames)
        b = len(p)
        points = np.empty((b, 21, per_link, 3), np.float32)
        near = max(1, per_link // 2)
        box = max(1, per_link // 4)
        glob = max(1, per_link // 8)
        for owner, mesh in enumerate(self.surface.meshes):
            local = self.surface_points(mesh, b * per_link, rng).reshape(b, per_link, 3)
            # Multiple near-surface scales supervise both sign and the distance slope.
            sigma = np.where(np.arange(near) % 2 == 0, .001, .006)
            local[:, :near] += rng.normal(size=(b, near, 3)) * sigma[None, :, None]
            lo, hi = mesh.bounds
            local[:, near:near + box] = rng.uniform(lo - .015, hi + .015, (b, box, 3))
            world = np.einsum('bij,bkj->bki', r[:, owner], local) + p[:, owner, None]
            world[:, near + box:near + box + glob] = rng.uniform(-self.extent, self.extent, (b, glob, 3))
            for j in range(near + box + glob, per_link):
                sources = np.flatnonzero(self.allowed[owner])
                source = int(rng.choice(sources))
                material = self.surface_points(self.surface.meshes[source], b, rng)
                world[:, j] = np.einsum('bij,bj->bi', r[:, source], material) + p[:, source]
            points[:, owner] = world
        links = np.broadcast_to(np.arange(21, dtype=np.int32)[None, :, None], (b, 21, per_link)).copy()
        return points.reshape(b, -1, 3), links.reshape(b, -1)
