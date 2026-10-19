"""Bounded mesh contact search for alternating contact/action optimization.

The contact block uses a geometric wrench-support surrogate. Actual contact
activation and wrench support remain the simulator/adjoint hand block's job.
All coordinates here are in the object's body frame, in meters.
"""
from dataclasses import dataclass
import numpy as np
import mujoco
from forceaware.mpc_loss import grasp_wrench_directions, GRASP_WRENCH_MARGIN


def closest_triangles(points, triangles):
    """Closest point on every triangle, including edges and vertices."""
    p = np.asarray(points, float)[:, None, :]
    a, b, c = np.asarray(triangles, float).transpose(1, 0, 2)
    ab, ac = b-a, c-a
    normal = np.cross(ab, ac)
    n2 = np.sum(normal*normal, axis=-1)
    projected = p - (np.sum((p-a)*normal, axis=-1) / n2)[..., None]*normal
    ap = projected-a
    d00, d01, d11 = (np.sum(ab*ab, axis=-1), np.sum(ab*ac, axis=-1),
                       np.sum(ac*ac, axis=-1))
    d20, d21 = np.sum(ap*ab, axis=-1), np.sum(ap*ac, axis=-1)
    denominator = n2
    v = (d11*d20-d01*d21)/denominator
    w = (d00*d21-d01*d20)/denominator
    inside = (v >= 0) & (w >= 0) & (v+w <= 1)
    choices = [projected]
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end-start
        t = np.clip(np.sum((p-start)*edge, axis=-1)/np.sum(edge*edge, axis=-1), 0, 1)
        choices.append(start + t[..., None]*edge)
    choices = np.stack(choices, axis=2)
    distance = np.sum((choices-p[:, :, None, :])**2, axis=-1)
    distance[:, :, 0] = np.where(inside, distance[:, :, 0], np.inf)
    return np.take_along_axis(choices, np.argmin(distance, axis=2)[..., None, None], axis=2)[:, :, 0]


@dataclass
class SurfacePatch:
    triangles: np.ndarray
    normals: np.ndarray
    template: np.ndarray
    anchor: np.ndarray
    anchor_normal: np.ndarray
    radius: float

    def project(self, proposals):
        closest = closest_triangles(np.asarray(proposals).reshape(-1, 3), self.triangles)
        # Projection and the trust region are enforced together: clipping a
        # projected point in Euclidean space could move it off the mesh again.
        valid = np.linalg.norm(closest-self.template, axis=-1) <= self.radius+1e-10
        distance = np.sum((closest-np.asarray(proposals)[:, None, :])**2, axis=-1)
        distance[~valid] = np.inf
        ids = np.argmin(distance, axis=1)
        rows = np.arange(len(ids))
        points = closest[rows, ids]
        missing = ~np.isfinite(distance[rows, ids])
        points[missing] = self.anchor
        normals = self.normals[ids].copy()
        normals[missing] = self.anchor_normal
        return points, normals


class ObjectMeshSurface:
    def __init__(self, vertices, faces):
        vertices, faces = np.asarray(vertices, float), np.asarray(faces, int)
        # Weld duplicated OBJ seam vertices before building edge connectivity.
        vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
        faces = inverse[faces]
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        keep = lengths > 1e-14
        self.triangles = triangles[keep]
        self.normals = normals[keep]/lengths[keep, None]
        if not len(self.triangles):
            raise ValueError('Object contact exploration requires a nondegenerate triangle mesh')
        from scipy.spatial import cKDTree
        centers = self.triangles.mean(axis=1)
        self.triangle_tree = cKDTree(centers)
        self.max_triangle_radius = float(np.linalg.norm(self.triangles-centers[:,None],axis=-1).max())
        self.adjacency = [set() for _ in self.triangles]
        edges = {}
        for i, face in enumerate(faces[keep]):
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                key = tuple(sorted((int(a), int(b))))
                for other in edges.setdefault(key, []):
                    self.adjacency[i].add(other); self.adjacency[other].add(i)
                edges[key].append(i)
        self.cache = {}

    @classmethod
    def from_model(cls, model, object_body):
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        rotation = data.xmat[object_body].reshape(3, 3)
        origin = data.xpos[object_body]
        vertices, faces, seen = [], [], set()
        offset = 0
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom] != object_body or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = int(model.geom_dataid[geom])
            key = (mesh, tuple(model.geom_pos[geom]), tuple(model.geom_quat[geom]))
            if key in seen:
                continue
            seen.add(key)
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            local = model.mesh_vert[start:start+count]
            world = local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            vertices.append((world-origin) @ rotation)
            start, count = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
            faces.append(model.mesh_face[start:start+count]+offset)
            offset += len(local)
        if not vertices:
            raise ValueError('Object contact exploration requires an object mesh in the planner model')
        return cls(np.concatenate(vertices), np.concatenate(faces))

    def patch(self, template, radius):
        template = np.asarray(template, float)
        key = (template.tobytes(), float(radius))
        if key in self.cache:
            return self.cache[key]
        # Every face intersecting the trust ball has its centroid within
        # radius + its bounding radius. This is an exact conservative cull.
        candidates = np.array(sorted(self.triangle_tree.query_ball_point(
            template,radius+self.max_triangle_radius+1e-10)),int)
        if not len(candidates):
            if len(self.cache) >= 2048:self.cache.clear()
            self.cache[key] = None
            return None
        closest = closest_triangles(template[None], self.triangles[candidates])[0]
        distances = np.linalg.norm(closest-template, axis=1)
        nearest = int(np.argmin(distances))
        seed = int(candidates[nearest])
        if distances[nearest] > radius+1e-10:
            # Invalid guidance cannot be repaired while respecting this radius.
            # Keep it fixed and report it, rather than silently making a jump.
            result = None
        else:
            allowed = np.zeros(len(self.triangles),bool)
            allowed[candidates] = distances <= radius+1e-10
            visited, pending = {seed}, [seed]
            while pending:
                for neighbor in self.adjacency[pending.pop()]:
                    if allowed[neighbor] and neighbor not in visited:
                        visited.add(neighbor); pending.append(neighbor)
            ids = np.array(sorted(visited))
            result = SurfacePatch(self.triangles[ids], self.normals[ids],
                                  template.copy(), closest[nearest], self.normals[seed], radius)
        # Bound cache across long sequences; patches are reused by all worlds
        # and repeated frames within a window.
        if len(self.cache) >= 2048:
            self.cache.clear()
        self.cache[key] = result
        return result


def friction_wrench_support(points, normals, friction, center, radius, directions):
    """Per-point support of four inward Coulomb-cone primitive wrenches."""
    normals = np.asarray(normals, float)
    axis = np.eye(3)[np.argmin(np.abs(normals), axis=-1)]
    tangent = np.cross(normals, axis)
    tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)
    second = np.cross(normals, tangent)
    tangents = np.stack((tangent, -tangent, second, -second), axis=-2)
    mu = np.repeat(np.asarray(friction, float), 2, axis=-1)
    force = (-normals[..., None, :] + mu[..., None]*tangents)/np.sqrt(1+mu[..., None]**2)
    arms = (np.asarray(points)-center)/radius
    wrench = np.concatenate((force, np.cross(arms[..., None, :], force)), axis=-1)
    return np.maximum(np.max(wrench @ directions.T, axis=-2), 0.)


def update_scaled_dual(dual, hand, surface, active, limit):
    """Dexonomy sign convention: lam += cp_hand - cp_object, in object frame."""
    residual = np.asarray(hand)-np.asarray(surface)
    updated = np.asarray(dual)+residual
    norm = np.linalg.norm(updated, axis=-1, keepdims=True)
    updated *= np.minimum(1., limit/np.maximum(norm, 1e-12))
    updated[..., ~np.asarray(active, bool), :] = 0.
    updated[~np.isfinite(updated)] = 0.
    return updated, residual


class LocalContactExplorer:
    def __init__(self, surface, *, radius=.005, wrench_weight=1., sweeps=1, rho=.1):
        if not np.isfinite([radius, wrench_weight, rho]).all() or radius <= 0 or wrench_weight < 0 or sweeps < 1 or rho <= 0:
            raise ValueError('Contact search needs a positive finite radius/sweep count and nonnegative wrench weight')
        self.surface, self.radius, self.wrench_weight, self.sweeps = surface, radius, wrench_weight, sweeps
        self.rho = rho
        self.directions = grasp_wrench_directions().astype(float)

    def solve(self, initial, template, hand, weights, friction, *, center, object_radius):
        points = np.asarray(initial, float).copy()
        template, hand, weights = np.asarray(template), np.asarray(hand), np.asarray(weights)
        active = np.flatnonzero(weights > 0)
        patches = {i: self.surface.patch(template[i], self.radius) for i in active}
        supports = np.zeros((len(points), len(self.directions)))
        movable = [i for i in active if patches[i] is not None]
        for i in movable:
            point, normal = patches[i].project(points[i:i+1])
            points[i] = point[0]
            supports[i] = friction_wrench_support(point, normal, friction[i:i+1], center, object_radius, self.directions)[0]
        total_weight = max(float(weights[active].sum()), 1e-12)
        def geometry(p, i):
            return weights[i]/total_weight * (.5*self.rho*np.sum((p-hand[i])**2, axis=-1)
                + .05*np.sum((p-template[i])**2, axis=-1))/self.radius**2
        def deficit(s):
            return np.mean((np.maximum(GRASP_WRENCH_MARGIN-s, 0)/GRASP_WRENCH_MARGIN)**2, axis=-1)
        def merit():
            return float(sum(geometry(points[i], i) for i in active)
                         + self.wrench_weight*deficit(supports.max(axis=0)))
        before, wrench_before = merit(), float(deficit(supports.max(axis=0)))
        accepted = 0
        for sweep in range(self.sweeps):
            step = self.radius * .5**(sweep+1)
            for i in movable:
                patch = patches[i]
                _, n = patch.project(points[i:i+1]); n = n[0]
                axis = np.eye(3)[np.argmin(np.abs(n))]
                u = np.cross(n, axis); u /= np.linalg.norm(u)
                v = np.cross(n, u)
                angles = np.arange(8)*np.pi/4
                ring = np.cos(angles)[:, None]*u + np.sin(angles)[:, None]*v
                proposals = np.vstack((points[i], patch.anchor, hand[i],
                                        points[i]+step*ring, points[i]+.5*step*ring))
                candidates, normals = patch.project(proposals)
                candidate_support = friction_wrench_support(candidates, normals,
                    np.broadcast_to(friction[i], (len(candidates), 2)), center, object_radius, self.directions)
                others = supports.copy(); others[i] = 0.
                value = geometry(candidates, i) + self.wrench_weight*deficit(
                    np.maximum(candidate_support, others.max(axis=0)))
                best = int(np.argmin(value))
                if value[best] < value[0]-1e-12:
                    points[i], supports[i] = candidates[best], candidate_support[best]
                    accepted += 1
        shift = np.linalg.norm(points[active]-template[active], axis=-1)
        return points.astype(np.float32), dict(method='bounded_mesh_wrench_search',
            active_contacts=len(active), movable_contacts=len(movable),
            frozen_off_surface_contacts=len(active)-len(movable), sweeps=self.sweeps,
            accepted_moves=accepted, initial_value=before, final_value=merit(),
            wrench_deficit_before=wrench_before, wrench_deficit_after=float(deficit(supports.max(axis=0))),
            max_shift_mm=float(shift.max(initial=0)*1000), radius_mm=self.radius*1000,
            wrench_is_geometric_surrogate=True)
