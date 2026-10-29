"""Geometry helpers for humanoid contact exploration."""
import itertools
import numpy as np
from scipy.spatial import ConvexHull

def box_mesh(size):
    vertices = np.array(list(itertools.product([-1., 1.], repeat=3))) * size
    hull = ConvexHull(vertices)
    faces = hull.simplices.copy()
    for face, equation in zip(faces, hull.equations):
        a, b, c = vertices[face]
        if np.dot(np.cross(b-a, c-a), equation[:3]) < 0:
            face[1], face[2] = face[2], face[1]
    return vertices.astype(np.float32), faces.astype(np.int32)


