"""Triangle-mesh input geometry and signed surface queries."""
from dataclasses import dataclass
import numpy as np
import mujoco
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

