"""CPU HiGHS test oracle; production grasp audits use grasp_audit's CUDA LP."""
import numpy as np
from scipy.optimize import linprog
from retarget.contact_solvers import contact_basis, grasp_matrix


def wrench_generators(points, normals, friction, center, radius, edges=16):
    basis = contact_basis(np.asarray(normals))
    angle = np.arange(edges) * (2*np.pi/edges)
    rays = (basis[..., 0, None] + friction *
            (basis[..., 1, None]*np.cos(angle) + basis[..., 2, None]*np.sin(angle)))
    return grasp_matrix(np.asarray(points)[None], rays[None], np.asarray(center), radius)[0]


def force_closure(points, normals, *, friction=.6, center=None, radius=1., edges=16, tolerance=1e-7):
    """Full wrench rank AND strictly positive normalized zero-wrench preload.

    An inscribed polygon approximates each Coulomb cone conservatively.
    No gravity, task direction, torsional contact moment or actuator bound is
    part of this geometric definition.
    """
    points, normals = np.asarray(points, float), np.asarray(normals, float)
    if points.shape != normals.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points and normals must have shape (contacts, 3)')
    if not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise ValueError('Force-closure inputs must be finite')
    if len(normals) and np.any(np.linalg.norm(normals, axis=1) < 1e-12):
        raise ValueError('Contact normals must be nonzero')
    if not np.isfinite([friction, radius, tolerance]).all() or friction < 0 or radius <= 0 or tolerance <= 0 or edges < 4:
        raise ValueError('Invalid friction cone or closure tolerance')
    center = np.zeros(3) if center is None else np.asarray(center)
    if len(points) == 0:
        return dict(force_closure=False, rank=0, positive_margin=0., normalized_margin=0.,
                    equilibrium_residual=None, coefficients=[], contacts=0)
    matrix = wrench_generators(points, normals, friction, center, radius, edges)
    count = matrix.shape[1]
    rank = int(np.linalg.matrix_rank(matrix, tol=1e-8))
    result = linprog(np.r_[np.zeros(count), -1.],
                     A_ub=np.c_[-np.eye(count), np.ones(count)], b_ub=np.zeros(count),
                     A_eq=np.r_[np.c_[matrix, np.zeros(6)], np.r_[np.ones(count), 0.][None]],
                     b_eq=np.r_[np.zeros(6), 1.], bounds=[(0., None)]*(count+1),
                     method='highs', options={'primal_feasibility_tolerance': 1e-9,
                                              'dual_feasibility_tolerance': 1e-9})
    alpha = result.x[:-1] if result.success else np.zeros(count)
    margin = float(result.x[-1]) if result.success else 0.
    residual = float(np.linalg.norm(matrix @ alpha)) if result.success else None
    closed = bool(rank == 6 and margin > tolerance and residual < 1e-7)
    return dict(force_closure=closed, rank=rank, positive_margin=margin,
                normalized_margin=count*margin, equilibrium_residual=residual,
                coefficients=alpha.tolist(), contacts=len(points))
