"""Local contact-set selection: batched ComFree force LM or wrench QP.

Coordinates, velocities and spatial inertia use the object body frame. These
small force subproblems predict prospective contacts; they do not replace the
full Warp hand rollout or assert that a prospective contact is already loaded.
"""
import numpy as np
from retarget.contact_exploration import LocalContactExplorer


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


def project_contact_forces(forces, friction, active, limit):
    out = np.asarray(forces).copy().reshape(*active.shape, 3)
    mu = np.asarray(friction)
    tangent = np.where(mu > 0, out[..., 1:], 0.)
    cone_radius = np.linalg.norm(tangent/np.maximum(mu, 1e-15), axis=-1)
    # Project in friction-scaled Lorentz-cone coordinates. Merely clipping
    # tangent force at the existing normal force prevents LM from building
    # the extra squeeze needed to support tangential loads.
    original_normal = out[...,0]
    normal = np.where(cone_radius <= original_normal, original_normal,
                      np.maximum(.5*(cone_radius+original_normal),0.))
    tangent *= np.minimum(1.,normal/np.maximum(cone_radius,1e-15))[...,None]
    fn = np.minimum(normal,limit/np.sqrt(1+np.max(mu,axis=-1)**2))
    tangent *= np.minimum(1.,fn/np.maximum(normal,1e-15))[...,None]
    out[..., 0], out[..., 1:] = fn, tangent
    out *= active[..., None]
    return out.reshape(forces.shape)


def comfree_velocity(forces, matrix, inverse, velocity, external, dt,
                     support_row, support_gap, stiffness=.2, damping=.001):
    """Lambda's applied-force ComFree step, including unilateral floor response.

    Free motion + h M^-1 G f; evaluate the floor reaction at that predicted
    velocity, then apply its impulse once. This is the reduced Lambda model,
    with the same .2/.001 support coefficients as the reference implementation.
    """
    applied = np.einsum('bij,bj->bi', matrix, forces)
    free = velocity+dt[:, None]*np.einsum('bij,bj->bi', inverse, external+applied)
    mobility_row = np.einsum('bij,bj->bi', inverse, support_row)
    mobility = np.maximum(np.einsum('bi,bi->b', mobility_row, support_row), 1e-15)
    speed = np.einsum('bi,bi->b', support_row, free)
    correction = -(stiffness*(dt*speed+support_gap)+damping*speed)/mobility
    loaded = (support_gap < 0) & (correction > 0)
    result = free+mobility_row*np.where(loaded, correction, 0.)[:, None]
    jacobian = dt[:, None, None]*(inverse @ matrix)
    row_jacobian = np.einsum('bi,bij->bj', support_row, jacobian)
    factor = np.where(loaded, (stiffness*dt+damping)/mobility, 0.)
    jacobian -= mobility_row[:, :, None]*(factor[:, None]*row_jacobian)[:, None, :]
    return result, jacobian


def motion_residual(velocity, derivative, goal, dt, radius):
    """Position and rotation-vector residual of a normalized quaternion step."""
    omega = velocity[:, 3:]
    length = np.linalg.norm(omega, axis=1)
    a = .5*dt
    safe = np.maximum(length, 1e-10)
    factor = 2*np.arctan(a*safe)/safe
    df = (2*a/(1+(a*safe)**2)-factor)/safe**2
    df = np.where(length > 1e-5, df, -dt**3/6)
    rot_jac = factor[:, None, None]*np.eye(3)+df[:, None, None]*omega[:, :, None]*omega[:, None, :]
    displacement = np.concatenate((dt[:, None]*velocity[:, :3], factor[:, None]*omega), axis=1)
    jac = np.concatenate((dt[:, None, None]*derivative[:, :3], rot_jac @ derivative[:, 3:]), axis=1)
    scale = np.array([1/radius]*3+[np.sqrt(.3)]*3)
    return (displacement-goal)*scale, jac*scale[None, :, None]


def batched_force_lm(points, normals, friction, active, *, inverse, velocity,
                     external, goal, dt, center, radius, iterations=3,
                     force_limit=10., weight=1., support_row=None, support_gap=None):
    """Independent LM force solves; only batched 6x6 systems are factorized.

    A Woodbury solve handles 3*C force variables without a (3*C)^3 solve.
    Every candidate retains its best evaluated feasible force iterate.
    """
    batch, contacts = points.shape[:2]
    basis = contact_basis(normals)
    # The spatial inertia and velocity are about the body origin. Contact
    # moments must use that same origin, including a nonzero COM offset.
    matrix = grasp_matrix(points, basis, np.zeros(3))
    if support_row is None:
        support_row, support_gap = np.zeros((batch, 6)), np.zeros(batch)
    def evaluate(f):
        v, jac = comfree_velocity(f, matrix, inverse, velocity, external, dt, support_row, support_gap)
        r, jac = motion_residual(v, jac, goal, dt, radius)
        return np.sqrt(weight)*r, np.sqrt(weight)*jac
    initial = np.zeros((batch, contacts, 3))
    initial[..., 0] = np.linalg.norm(external[:, :3], axis=1)[:, None]/np.maximum(active.sum(axis=1), 1)[:, None]
    force = project_contact_forces(initial.reshape(batch, -1), friction, active, force_limit)
    regularization = 1e-6
    damping = np.full(batch, 1e-3)
    residual, jacobian = evaluate(force)
    cost = np.sum(residual**2, axis=1)+regularization*np.sum(force**2, axis=1)
    initial_cost = cost.copy()
    for _ in range(iterations):
        gradient = np.einsum('bri,br->bi', jacobian, residual)+regularization*force
        mu = damping+regularization
        system = jacobian @ jacobian.swapaxes(1, 2)+mu[:, None, None]*np.eye(6)
        jg = np.einsum('bri,bi->br', jacobian, gradient)
        correction = np.linalg.solve(system, jg[..., None])[..., 0]
        delta = -(gradient-np.einsum('bri,br->bi', jacobian, correction))/mu[:, None]
        delta *= np.minimum(1., 2./np.maximum(np.linalg.norm(delta, axis=1), 1e-15))[:, None]
        proposal = project_contact_forces(force+delta, friction, active, force_limit)
        next_r, next_j = evaluate(proposal)
        next_cost = np.sum(next_r**2, axis=1)+regularization*np.sum(proposal**2, axis=1)
        good = np.isfinite(next_cost) & (next_cost < cost)
        force[good], residual[good], jacobian[good], cost[good] = proposal[good], next_r[good], next_j[good], next_cost[good]
        damping = np.clip(damping*np.where(good, .5, 4.), 1e-8, 1e6)
    return force.reshape(batch, contacts, 3), cost, initial_cost


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


class ContactSetSolver(LocalContactExplorer):
    """Shared bounded surface geometry, with batch and single-group interfaces."""
    def __init__(self, *args, samples=8, lm_iterations=3, qp_iterations=96,
                 seed=0, mass=.2, dt=.01, device=None, motion_tolerance=.0001,
                 wrench_tolerance=.05, min_improvement=.01, **kwargs):
        super().__init__(*args, **kwargs)
        if min(samples, lm_iterations, qp_iterations) < 1 or not np.isfinite([mass, dt]).all() or min(mass, dt) <= 0:
            raise ValueError('Contact sample/iteration counts, mass and dt must be positive')
        if (not np.isfinite([motion_tolerance,wrench_tolerance,min_improvement]).all()
                or min(motion_tolerance,wrench_tolerance,min_improvement) < 0 or min_improvement >= 1):
            raise ValueError('Contact tolerances must be nonnegative and min_improvement below one')
        self.motion_tolerance,self.wrench_tolerance,self.min_improvement = motion_tolerance,wrench_tolerance,min_improvement
        self.samples, self.lm_iterations, self.qp_iterations = samples, lm_iterations, qp_iterations
        self.seed, self.mass, self.dt = seed, mass, dt
        self.projector = None
        if device is not None:
            from retarget.contact_projection import WarpPatchProjector
            self.projector = WarpPatchProjector(self.surface,self.radius,device)

    def solve(self, initial, template, hand, weights, friction, *, center, object_radius, context=None):
        result, diagnostics = self.solve_batch(np.asarray(initial)[None], np.asarray(template)[None],
            np.asarray(hand)[None], np.asarray(weights)[None], np.asarray(friction)[None],
            center=center, object_radius=object_radius, context=context)
        return result[0], diagnostics[0]

    def _project(self, proposals, template, weights):
        if self.projector is not None:
            return self.projector.project(proposals,template,weights)
        batch, samples, contacts = proposals.shape[:3]
        points = np.where((weights > 0)[:, None, :, None], proposals, template[:, None]).copy()
        normals = np.zeros_like(points); normals[..., 2] = 1.
        movable = np.zeros((batch, contacts), bool)
        groups = {}
        for b, c in zip(*np.nonzero(weights > 0)):
            key = np.asarray(template[b, c], float).tobytes()
            groups.setdefault(key, []).append((b, c))
        for rows in groups.values():
            b, c = rows[0]
            patch = self.surface.patch(template[b, c], self.radius)
            if patch is None:
                for b, c in rows:
                    points[b, :, c] = template[b, c]
                continue
            bs, cs = np.array(rows).T
            flat = proposals[bs, :, cs].reshape(-1, 3)
            projected, direction = [], []
            for start in range(0, len(flat), 128):
                p, n = patch.project(flat[start:start+128])
                projected.append(p); direction.append(n)
            points[bs, :, cs] = np.concatenate(projected).reshape(len(rows), samples, 3)
            normals[bs, :, cs] = np.concatenate(direction).reshape(len(rows), samples, 3)
            movable[bs, cs] = True
        return points, normals, movable

    def _physics(self, batch, object_radius, context):
        source = {} if context is None else context
        def get(name, default, shape):
            return np.broadcast_to(np.asarray(source.get(name, default), float), (batch,)+shape).copy()
        inverse = np.diag([1/self.mass]*3+[1/(.4*self.mass*object_radius**2)]*3)
        return dict(inverse=get('inverse', getattr(self, 'inverse', inverse), (6, 6)), velocity=get('velocity', np.zeros(6), (6,)),
                    external=get('external', [0, 0, -9.81*self.mass, 0, 0, 0], (6,)),
                    goal=get('goal', np.zeros(6), (6,)), dt=get('dt', self.dt, ()),
                    support_row=get('support_row', np.zeros(6), (6,)), support_gap=get('support_gap', 0., ()))

    def _geometry_cost(self, points, template, hand, weights):
        # points is (groups, samples, contacts, xyz).
        normalizer = np.maximum(weights.sum(axis=1), 1e-12)
        error = .5*self.rho*np.sum((points-hand[:, None])**2, axis=-1)
        error += .05*np.sum((points-template[:, None])**2, axis=-1)
        return np.sum(weights[:, None]*error, axis=-1)/(normalizer[:, None]*self.radius**2)

    def _diagnostics(self, points, template, weights, movable, initial_cost, cost, **extra):
        shift = np.linalg.norm(points-template, axis=-1)*(weights > 0)
        return [dict(method=self.method, active_contacts=int(np.sum(weights[b] > 0)),
            movable_contacts=int(movable[b].sum()),
            frozen_off_surface_contacts=int(np.sum(weights[b] > 0)-movable[b].sum()),
            initial_value=float(initial_cost[b]), final_value=float(cost[b]),
            sweeps=self.sweeps, force_witness_is_prospective=True,
            radius_mm=self.radius*1000, max_shift_mm=float(shift[b].max(initial=0)*1000),
            **{key: value[b].item() if isinstance(value, np.ndarray) else value for key, value in extra.items()})
            for b in range(len(points))]


class SampledComFreeLM(ContactSetSolver):
    method = 'sampled_comfree_batched_lm'

    def solve_batch(self, initial, template, hand, weights, friction, *, center, object_radius, context=None):
        points, template, hand, weights, friction = map(lambda x: np.asarray(x, float),
                                                       (initial, template, hand, weights, friction))
        batch, contacts = points.shape[:2]
        physics = self._physics(batch, object_radius, context)
        rng = np.random.default_rng(self.seed)
        points, normals, movable = self._project(points[:, None], template, weights)
        points, normals = points[:, 0], normals[:, 0]
        projection_shift = np.max(np.linalg.norm(points-initial,axis=-1),axis=1)*1000
        active = (weights > 0) & movable
        def evaluate(p,n,ids):
            local = {k:v[ids] for k,v in physics.items()}
            force,cost,_ = batched_force_lm(p,n,friction[ids],active[ids],**local,
                center=center,radius=object_radius,iterations=self.lm_iterations,
                weight=self.wrench_weight*(object_radius/self.radius)**2)
            matrix = grasp_matrix(p,contact_basis(n),np.zeros(3))
            velocity,derivative = comfree_velocity(force.reshape(len(ids),-1),matrix,
                local['inverse'],local['velocity'],local['external'],local['dt'],
                local['support_row'],local['support_gap'])
            residual,_ = motion_residual(velocity,derivative,local['goal'],local['dt'],object_radius)
            error = np.linalg.norm(residual,axis=1)*object_radius
            geometry = self._geometry_cost(p[:,None],template[ids],hand[ids],weights[ids])[:,0]
            return cost+geometry,error
        value,error = evaluate(points,normals,np.arange(batch))
        initial_cost,initial_error = value.copy(),error.copy()
        searched = np.zeros(batch,bool); selected = np.zeros(batch,int)
        for _ in range(self.sweeps):
            ids = np.flatnonzero((error > self.motion_tolerance) & active.any(axis=1))
            if not len(ids) or self.samples == 1:
                break
            searched[ids] = True
            proposals = np.broadcast_to(points[ids,None],(len(ids),self.samples-1,contacts,3)).copy()
            proposals[:,0] = (self.rho*hand[ids]+.1*template[ids])/(self.rho+.1)
            if self.samples > 2:
                offset = rng.normal(size=(self.samples-2,contacts,3))
                offset *= rng.uniform(0.,.5*self.radius,size=offset.shape[:-1]+(1,))/np.maximum(np.linalg.norm(offset,axis=-1,keepdims=True),1e-15)
                proposals[:,1:] += offset[None]
            candidates,directions,_ = self._project(proposals,template[ids],weights[ids])
            repeated = np.repeat(ids,self.samples-1)
            cost,motion = evaluate(candidates.reshape(-1,contacts,3),directions.reshape(-1,contacts,3),repeated)
            cost,motion = cost.reshape(len(ids),-1),motion.reshape(len(ids),-1)
            for row,index in enumerate(ids):
                feasible = motion[row] <= self.motion_tolerance
                # Task feasibility takes priority over the soft ADMM/anchor
                # penalty; that penalty breaks ties among feasible candidates.
                eligible = feasible | (motion[row] < error[index]-max(1e-9,self.min_improvement*error[index]))
                rank_value = motion[row]
                if feasible.any():
                    eligible = feasible
                    rank_value = cost[row]
                rank = np.where(eligible & np.isfinite(cost[row]),rank_value,np.inf)
                winner = int(np.argmin(rank))
                if np.isfinite(rank[winner]):
                    points[index],normals[index] = candidates[row,winner],directions[row,winner]
                    value[index],error[index] = cost[row,winner],motion[row,winner]
                    selected[index] = winner+1
        diagnostics = self._diagnostics(points,template,weights,movable,initial_cost,value,
            candidate_count=self.samples,selected_candidate=selected,lm_iterations=self.lm_iterations,
            contact_model='lambda_comfree_applied_force',force_solver='batched_lm_woodbury_6x6',
            includes_incumbent=True,incumbent_projection_mm=projection_shift,finite_difference_gradients=False,
            searched=searched,held_feasible=initial_error<=self.motion_tolerance,
            feasible=error<=self.motion_tolerance,motion_error_mm=error*1000,
            initial_motion_error_mm=initial_error*1000,selection_priority='task_feasibility_then_geometry')
        return points.astype(np.float32),diagnostics


class QPContactSolver(ContactSetSolver):
    method = 'wrench_qp_projected_contact_update'

    def solve_batch(self, initial, template, hand, weights, friction, *, center, object_radius, context=None):
        points, template, hand, weights, friction = map(lambda x: np.asarray(x, float),
                                                       (initial, template, hand, weights, friction))
        batch = len(points)
        physics = self._physics(batch, object_radius, context)
        # Required contact wrench includes external load and desired motion.
        acceleration = (physics['goal']/physics['dt'][:, None]-physics['velocity'])/physics['dt'][:, None]
        target = np.linalg.solve(physics['inverse'], acceleration[..., None])[..., 0]-physics['external']
        # G in the QP is about the COM; transform the required origin wrench.
        target[:, 3:] -= np.cross(center,target[:, :3])
        target[:, 3:] /= object_radius
        force_scale = max(self.mass*9.81, 1e-6)
        points, normals, movable = self._project(points[:, None], template, weights)
        points, normals = points[:, 0], normals[:, 0]
        active = (weights > 0) & movable
        def evaluate(p, n, ids=None):
            ids = np.arange(batch) if ids is None else ids
            qpcost, gradient, info = wrench_qp(p, n, friction[ids], active[ids], target[ids],
                center=center, radius=object_radius, force_scale=force_scale, iterations=self.qp_iterations)
            cost = self.wrench_weight*qpcost+self._geometry_cost(p[:, None], template[ids], hand[ids], weights[ids])[:, 0]
            return cost, self.wrench_weight*gradient, info
        value, gradient, info = evaluate(points, normals)
        initial_cost = value.copy()
        initial_residual = np.linalg.norm(info['residual'],axis=1)
        accepted = np.zeros(batch, int)
        searched = np.zeros(batch,bool)
        for _ in range(self.sweeps):
            # A trust-region step on the task residual. A geometry-weighted
            # tiny gradient step can get stuck outside the feasible set.
            tangent=gradient-np.sum(gradient*normals,axis=-1,keepdims=True)*normals
            tangent*=active[...,None]
            norm=np.max(np.linalg.norm(tangent,axis=-1),axis=1)
            delta=-.5*self.radius*tangent/np.maximum(norm[:,None,None],1e-15)
            pending = (np.linalg.norm(info['residual'],axis=1) > self.wrench_tolerance) & active.any(axis=1)
            searched |= pending
            if not pending.any(): break
            for step in (1., .5, .25):
                ids = np.flatnonzero(pending)
                proposal, next_normal, _ = self._project((points[ids]+step*delta[ids])[:,None],template[ids],weights[ids])
                proposal, next_normal = proposal[:,0], next_normal[:,0]
                next_value, next_gradient, next_info = evaluate(proposal,next_normal,ids)
                residual = np.linalg.norm(next_info['residual'],axis=1)
                feasible = residual <= self.wrench_tolerance
                previous=np.linalg.norm(info['residual'][ids],axis=1)
                improved = np.isfinite(next_value) & (feasible | (
                    residual < previous-np.maximum(1e-9,self.min_improvement*previous)))
                good = ids[improved]
                points[good], normals[good], value[good] = proposal[improved], next_normal[improved], next_value[improved]
                gradient[good] = next_gradient[improved]
                for name in info:
                    info[name][good] = next_info[name][improved]
                accepted[good] += 1; pending[good] = False
                if not pending.any():
                    break
        diagnostic = self._diagnostics(points, template, weights, movable, initial_cost, value,
            qp_iterations=self.qp_iterations, accepted_moves=accepted,
            searched=searched,held_feasible=initial_residual<=self.wrench_tolerance,
            feasible=np.linalg.norm(info['residual'],axis=1)<=self.wrench_tolerance,
            qp_projected_gradient_norm=info['projected_gradient_norm'],
            wrench_residual=np.linalg.norm(info['residual'], axis=1),
            initial_wrench_residual=initial_residual,selection_priority='task_feasibility_then_geometry',
            force_solver='batched_convex_pressure_qp', finite_difference_gradients=False)
        return points.astype(np.float32), diagnostic
