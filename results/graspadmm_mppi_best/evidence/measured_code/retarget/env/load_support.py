import numpy as np
def skew(v):
    v = np.asarray(v)
    out = np.zeros(v.shape[:-1]+(3, 3))
    out[..., 0, 1], out[..., 0, 2] = -v[..., 2], v[..., 1]
    out[..., 1, 0], out[..., 1, 2] = v[..., 2], -v[..., 0]
    out[..., 2, 0], out[..., 2, 1] = -v[..., 1], v[..., 0]
    return out

def contact_basis(normals):
    inward = -np.asarray(normals)
    inward = inward/np.maximum(np.linalg.norm(inward,axis=-1,keepdims=True),1e-15)
    axis = np.eye(3)[np.argmin(np.abs(inward), axis=-1)]
    t1 = np.cross(inward, axis)
    t1 /= np.maximum(np.linalg.norm(t1, axis=-1, keepdims=True), 1e-15)
    return np.stack((inward, t1, np.cross(inward, t1)), axis=-1)

def grasp_matrix(points, basis, center, radius=1.):
    torque = skew((points-center)/radius) @ basis
    # (batch, contact, wrench coordinate, force coordinate) -> (batch, 6, 3*C)
    both = np.concatenate((basis, torque), axis=-2)
    return np.swapaxes(both, -3, -2).reshape(both.shape[:-3]+(6, -1))
