"""Unmodified force QP functions from git 4cc156c3^:retarget/contact_solvers.py."""
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

def _capped_simplex(value, cap):
    nonnegative = np.maximum(value, 0.)
    ordered = np.sort(value, axis=-1)[..., ::-1]
    threshold = (np.cumsum(ordered, axis=-1)-cap[..., None])/np.arange(1, value.shape[-1]+1)
    count = np.maximum(np.sum(ordered > threshold, axis=-1), 1)
    theta = np.take_along_axis(threshold, (count-1)[..., None], axis=-1)[..., 0]
    projected = np.maximum(value-theta[..., None], 0.)
    return np.where((nonnegative.sum(axis=-1) <= cap)[..., None], nonnegative, projected)

def project_pressures(value, caps, minimum):
    """Exact projection onto nonnegative, per-contact-capped pressures and a total lower bound."""
    projected = _capped_simplex(value, caps)
    missing = projected.sum(axis=(-2, -1)) < minimum
    if np.any(missing):
        values, limits, target = value[missing], caps[missing], minimum[missing]
        low = np.zeros(len(values))
        high = np.max(np.abs(values), axis=(-2, -1))+np.max(limits, axis=-1)+target+1.
        for _ in range(32):
            mid = .5*(low+high)
            mass = _capped_simplex(values+mid[:, None, None], limits).sum(axis=(-2, -1))
            low, high = np.where(mass < target, mid, low), np.where(mass >= target, mid, high)
        projected[missing] = _capped_simplex(values+high[:, None, None], limits)
    return projected

def wrench_qp(points, normals, friction, active, target, *, center, radius,
              force_scale=1., force_limit=10., minimum_pressure=1., iterations=96):
    """Convex force QP using four friction-pyramid edges per contact.

    min .5 ||G alpha - target||^2 + eps/2 ||alpha||^2,
    alpha >= 0, sum_edges alpha_i <= Fmax, sum_all alpha >= Pmin.
    Projected accelerated gradient solves all independent QPs as one batch.
    """
    batch, contacts = points.shape[:2]
    basis = contact_basis(normals)
    n, t1, t2 = basis[..., 0], basis[..., 1], basis[..., 2]
    primitive = np.stack((n+friction[..., :1]*t1, n-friction[..., :1]*t1,
                          n+friction[..., 1:]*t2, n-friction[..., 1:]*t2), axis=-1)
    matrix = grasp_matrix(points, primitive, center, radius)/force_scale
    target = target/force_scale
    caps = active.astype(float)*force_limit
    minimum = np.minimum(minimum_pressure, caps.sum(axis=-1))
    alpha = np.broadcast_to((minimum/np.maximum(4*active.sum(axis=1), 1))[:, None, None], (batch, contacts, 4)).copy()
    alpha *= active[..., None]
    y, t = alpha.copy(), 1.
    regularization = 1e-7
    lipschitz = np.sum(matrix**2, axis=(1, 2))+regularization
    for _ in range(iterations):
        residual = np.einsum('bri,bi->br', matrix, y.reshape(batch, -1))-target
        gradient = np.einsum('bri,br->bi', matrix, residual).reshape(alpha.shape)+regularization*y
        proposal = project_pressures(y-gradient/lipschitz[:, None, None], caps, minimum)
        next_t = .5*(1+np.sqrt(1+4*t*t))
        y = proposal+(t-1)/next_t*(proposal-alpha)
        alpha, t = proposal, next_t
    residual = np.einsum('bri,bi->br', matrix, alpha.reshape(batch, -1))-target
    force = np.einsum('bcij,bcj->bci', primitive, alpha)
    # Envelope derivative: on a mesh face the primitive directions are fixed.
    gradient_points = np.cross(force, residual[:, None, 3:])/(radius*force_scale)
    gradient = np.einsum('bri,br->bi', matrix, residual).reshape(alpha.shape)+regularization*alpha
    projected = project_pressures(alpha-gradient/lipschitz[:, None, None], caps, minimum)
    stationarity = np.linalg.norm((alpha-projected).reshape(batch, -1), axis=1)*lipschitz
    cost = .5*np.sum(residual**2, axis=1)+.5*regularization*np.sum(alpha**2, axis=(1, 2))
    return cost, gradient_points, dict(pressures=alpha, forces=force, residual=residual,
                                      projected_gradient_norm=stationarity)
