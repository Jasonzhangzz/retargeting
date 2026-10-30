"""Batched forward compliant contact prediction with independent servo offsets."""
from dataclasses import dataclass
import math
import mujoco
import numpy as np
import warp as wp
from .explicit_parameters import Parameters
from retarget.task.grasp.cost import step_pose_cost,average_pose_cost
UPSTREAM = 'https://github.com/asu-iris/Complementarity-Free-Dexterous-Manipulation'
COMMIT = '87b320ffb4b40ab4f3aa344371f6e5e8d619aa7e'
DEFAULTS = dict(horizon=4, dt=.1, sigma=.5, pose_weight=1.e6,
    translation_metric=50., rotation_metric=.1, contact_cutoff_mm=1.,
    contact_iterations=64, preload_limit=.5, motion_tolerance_mm=2., rotation_tolerance_deg=5.)


@wp.func
def local_rotation(v: wp.vec3):
    # Nonzero and finite derivative at the identity (unlike norm/axis at zero).
    theta2 = wp.dot(v, v)
    a, b = float(1.), float(.5)
    if theta2 > 1.e-8:
        theta = wp.sqrt(theta2)
        a = wp.sin(theta)/theta
        b = (1.-wp.cos(theta))/theta2
    else:
        a = 1.-theta2/6.+theta2*theta2/120.
        b = .5-theta2/24.+theta2*theta2/720.
    k = wp.mat33(0., -v[2], v[1], v[2], 0., -v[0], -v[1], v[0], 0.)
    return wp.identity(n=3, dtype=float)+a*k+b*(k@k)

@wp.func
def closest_triangle(point: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3):
    """Piecewise-exact projection with explicit edge/vertex derivatives."""
    ab, ac, ap = b-a, c-a, point-a
    d1, d2 = wp.dot(ab, ap), wp.dot(ac, ap)
    if d1 <= 0. and d2 <= 0.:
        return a
    bp = point-b
    d3, d4 = wp.dot(ab, bp), wp.dot(ac, bp)
    if d3 >= 0. and d4 <= d3:
        return b
    vc = d1*d4-d3*d2
    if vc <= 0. and d1 >= 0. and d3 <= 0.:
        return a+(d1/(d1-d3))*ab
    cp = point-c
    d5, d6 = wp.dot(ab, cp), wp.dot(ac, cp)
    if d6 >= 0. and d5 <= d6:
        return c
    vb = d5*d2-d1*d6
    if vb <= 0. and d2 >= 0. and d6 <= 0.:
        return a+(d2/(d2-d6))*ac
    va = d3*d6-d5*d4
    if va <= 0. and d4-d3 >= 0. and d5-d6 >= 0.:
        return b+((d4-d3)/((d4-d3)+(d5-d6)))*(c-b)
    denominator = 1./(va+vb+vc)
    return a+(vb*denominator)*ab+(vc*denominator)*ac

@wp.kernel
def initial_state(p: Parameters, x: wp.array2d(dtype=float), out: wp.array2d(dtype=float)):
    w = wp.tid()
    for j in range(p.seed.shape[0]):
        out[w, j] = p.seed[j]
    r = p.root@local_rotation(wp.vec3(x[w, 3], x[w, 4], x[w, 5]))
    y = wp.asin(wp.clamp(r[0, 2], -1., 1.))
    ax, az = wp.atan2(-r[1, 2], r[2, 2]), wp.atan2(-r[0, 1], r[0, 0])
    if wp.abs(wp.cos(y)) < 1.e-6:
        ax, az = wp.atan2(r[2, 1], r[1, 1]), 0.
    angles = wp.vec3(ax, y, az)
    for j in range(26):
        value = x[w, j]
        if j >= 3 and j < 6:
            value = angles[j-3]
        out[w, p.hand_qpos[j]] = value

@wp.kernel(enable_backward=False)
def select_patches(p: Parameters, position: wp.array2d(dtype=wp.vec3),
                   rotation: wp.array2d(dtype=wp.mat33), selected: wp.array2d(dtype=int)):
    w, c = wp.tid()
    best, index = float(wp.inf), int(-1)
    for i in range(p.starts[c], p.ends[c]):
        world = position[w, p.bodies[c]]+rotation[w, p.bodies[c]]@p.samples[i]
        local = wp.transpose(rotation[w, p.object_body])@(world-position[w, p.object_body])
        hit = wp.mesh_query_point(p.mesh, local, 1.e6)
        if hit.result and p.forbidden[i] == 0:
            nearest = wp.mesh_eval_position(p.mesh, hit.face, hit.u, hit.v)
            distance = hit.sign*wp.length(local-nearest)
            normal = hit.sign*wp.normalize(local-nearest)
            if wp.length(local-nearest) < 1.e-7:
                a = wp.mesh_get_point(p.mesh, 3*hit.face)
                b = wp.mesh_get_point(p.mesh, 3*hit.face+1)
                v = wp.mesh_get_point(p.mesh, 3*hit.face+2)
                normal = wp.normalize(wp.cross(b-a, v-a))
            hand_normal = wp.transpose(rotation[w, p.object_body])@(rotation[w, p.bodies[c]]@p.sample_normals[i])
            if distance < best and wp.dot(hand_normal, -normal) > .25:
                best, index = distance, i
    selected[w, c] = index

@wp.kernel
def contact_rows(p: Parameters, selected: wp.array2d(dtype=int),
                 position: wp.array2d(dtype=wp.vec3), rotation: wp.array2d(dtype=wp.mat33),
                 joint_anchor: wp.array2d(dtype=wp.vec3), joint_axis: wp.array2d(dtype=wp.vec3),
                 phi: wp.array2d(dtype=float), jac: wp.array3d(dtype=float),
                 rays: wp.array2d(dtype=wp.vec3)):
    w, c, edge = wp.tid()
    row = c*4+edge
    body, obj = p.bodies[c], p.object_body
    index = wp.max(selected[w, c], 0)
    point = position[w, body]+rotation[w, body]@p.samples[index]
    r = rotation[w, obj]
    local = wp.transpose(r)@(point-position[w, obj])
    hit = wp.mesh_query_point(p.mesh, local, 1.e6)
    gap = float(1.)
    ray, contact = wp.vec3(0.), wp.vec3(0.)
    if hit.result:
        a = wp.mesh_get_point(p.mesh, 3*hit.face)
        b = wp.mesh_get_point(p.mesh, 3*hit.face+1)
        d = wp.mesh_get_point(p.mesh, 3*hit.face+2)
        projected = closest_triangle(local, a, b, d)
        distance = hit.sign*wp.length(local-projected)
        if selected[w, c] >= 0 and distance <= p.cutoff and p.forbidden[index] == 0:
            outward = wp.normalize(wp.cross(b-a, d-a))
            separation = local-projected
            if wp.length(separation) > 1.e-7:
                # At mesh edges/vertices the SDF normal is the separating
                # direction, not an arbitrarily selected adjacent face normal.
                outward = hit.sign*wp.normalize(separation)
            normal = -(r@outward)
            # Keep this distinct from the later per-joint axis: reusing a
            # mutable local across that dynamic loop corrupts Warp AD replay.
            basis_axis = wp.vec3(1., 0., 0.)
            if wp.abs(normal[1]) < wp.abs(normal[0]) and wp.abs(normal[1]) <= wp.abs(normal[2]):
                basis_axis = wp.vec3(0., 1., 0.)
            elif wp.abs(normal[2]) < wp.abs(normal[0]) and wp.abs(normal[2]) < wp.abs(normal[1]):
                basis_axis = wp.vec3(0., 0., 1.)
            t1 = wp.normalize(wp.cross(normal, basis_axis))
            t2 = wp.cross(normal, t1)
            tangent = t1
            if edge == 1:
                tangent = t2
            elif edge == 2:
                tangent = -t1
            elif edge == 3:
                tangent = -t2
            ray = normal+p.friction*tangent
            # Real signed separation, not half-gap or a centimetre soft tail.
            gap = distance
            contact = .5*(point+position[w, obj]+r@projected)
    phi[w, row] = gap
    rays[w, row] = ray
    arm_local = wp.transpose(r)@(contact-position[w, obj])
    torque_local = wp.cross(arm_local, wp.transpose(r)@ray)
    for j in range(3):
        jac[w, row, j] = ray[j]
        jac[w, row, j+3] = torque_local[j]
    for j in range(26):
        value = float(0.)
        if p.ancestor[c, j] != 0:
            hinge_axis = joint_axis[w, p.joints[j]]
            velocity = hinge_axis
            if p.joint_slide[j] == 0:
                velocity = wp.cross(hinge_axis, contact-joint_anchor[w, p.joints[j]])
            value = -wp.dot(ray, velocity)
        jac[w, row, j+6] = value

@wp.kernel
def free_displacement(p: Parameters, qpos: wp.array2d(dtype=float),
                      hold: wp.array2d(dtype=float), gravity: wp.array2d(dtype=float), free: wp.array2d(dtype=float)):
    w, j = wp.tid()
    b = float(0.)
    if j < 3:
        b = p.gravity_force[j]
    elif j >= 6:
        address = p.hand_qpos[j-6]
        target = wp.clamp(hold[w, address], p.command_low[j-6], p.command_high[j-6])
        servo = wp.clamp(p.stiffness[j-6]*(target-qpos[w, address]), p.force_low[j-6], p.force_high[j-6])
        b = servo+gravity[w, j-6]
    free[w, j] = p.inverse[j]*b

@wp.kernel
def hold_command(p: Parameters, initial: wp.array2d(dtype=float),
                 x: wp.array2d(dtype=float), gravity: wp.array2d(dtype=float), hold: wp.array2d(dtype=float)):
    w, j = wp.tid()
    address = p.hand_qpos[j]
    offset = float(0.)
    if j >= 6:
        offset = x[w, j+20]
    hold[w, address] = wp.clamp(initial[w, address]+offset-gravity[w, j]/p.stiffness[j], p.command_low[j], p.command_high[j])

@wp.kernel
def hand_gravity(p: Parameters, position: wp.array2d(dtype=wp.vec3), rotation: wp.array2d(dtype=wp.mat33),
                 anchor: wp.array2d(dtype=wp.vec3), axis: wp.array2d(dtype=wp.vec3), out: wp.array2d(dtype=float)):
    w, j = wp.tid()
    value = float(0.)
    for b in range(p.body_mass.shape[0]):
        if p.body_ancestor[b, j] != 0:
            com = position[w, b]+rotation[w, b]@p.body_com[b]
            velocity = axis[w, p.joints[j]]
            if p.joint_slide[j] == 0:
                velocity = wp.cross(velocity, com-anchor[w, p.joints[j]])
            value += p.body_mass[b]*wp.dot(p.gravity, velocity)
    out[w, j] = value

@wp.kernel
def contact_trace(p: Parameters, jac: wp.array3d(dtype=float), out: wp.array(dtype=float)):
    w = wp.tid()
    trace = float(0.)
    for c in range(jac.shape[1]//4):
        for j in range(32):
            n = .25*(jac[w, c*4, j]+jac[w, c*4+1, j]+jac[w, c*4+2, j]+jac[w, c*4+3, j])
            t1 = (jac[w, c*4, j]-jac[w, c*4+2, j])/(2.*p.friction)
            t2 = (jac[w, c*4+1, j]-jac[w, c*4+3, j])/(2.*p.friction)
            trace += (n*n+t1*t1+t2*t2)*p.inverse[j]
    # trace bounds the largest eigenvalue of J Q^-1 J.T. Fixed relaxation
    # gives a convergent coupled solve without per-contact softplus forces.
    out[w] = trace

@wp.kernel
def contact_step_size(p: Parameters, trace: wp.array(dtype=float), rate: wp.array(dtype=float)):
    w = wp.tid()
    rate[w] = p.sigma/wp.max(trace[w], 1.e-8)

@wp.kernel
def projected_response(p: Parameters, jac: wp.array3d(dtype=float),
                       force: wp.array2d(dtype=float), response: wp.array2d(dtype=float)):
    w, j = wp.tid()
    total = float(0.)
    for row in range(jac.shape[1]):
        total += jac[w, row, j]*force[w, row]
    response[w, j] = p.inverse[j]*total

@wp.kernel
def projected_force(jac: wp.array3d(dtype=float), free: wp.array2d(dtype=float),
                    phi: wp.array2d(dtype=float), response: wp.array2d(dtype=float),
                    out: wp.array2d(dtype=float)):
    w, row = wp.tid()
    gradient = phi[w, row]
    for j in range(32):
        gradient += jac[w, row, j]*(free[w, j]+response[w, j])
    out[w, row] = gradient

@wp.kernel
def projected_update(p: Parameters, rate: wp.array(dtype=float), previous: wp.array2d(dtype=float),
                     gradient: wp.array2d(dtype=float), out: wp.array2d(dtype=float)):
    w, c = wp.tid()
    row = c*4
    normal = previous[w, row]+previous[w, row+1]+previous[w, row+2]+previous[w, row+3]
    gn = .25*(gradient[w, row]+gradient[w, row+1]+gradient[w, row+2]+gradient[w, row+3])
    normal = wp.max(normal-rate[w]*gn, 0.)
    # Normal complementarity must not contain tangential slip. Projecting the
    # entire force onto a cone couples slip into normal separation (dilatancy).
    tx = p.friction*(previous[w, row]-previous[w, row+2])-rate[w]*(gradient[w, row]-gradient[w, row+2])/(2.*p.friction)
    ty = p.friction*(previous[w, row+1]-previous[w, row+3])-rate[w]*(gradient[w, row+1]-gradient[w, row+3])/(2.*p.friction)
    radius = p.friction*normal
    ax, ay = wp.abs(tx), wp.abs(ty)
    if ax+ay > radius:
        if ax-ay >= radius:
            tx, ty = wp.sign(tx)*radius, 0.
        elif ay-ax >= radius:
            tx, ty = 0., wp.sign(ty)*radius
        else:
            threshold = .5*(ax+ay-radius)
            tx, ty = wp.sign(tx)*(ax-threshold), wp.sign(ty)*(ay-threshold)
    u, v = tx/p.friction, ty/p.friction
    shared = wp.max(normal-wp.abs(u)-wp.abs(v), 0.)*.25
    out[w, row] = shared+wp.max(u, 0.)
    out[w, row+1] = shared+wp.max(v, 0.)
    out[w, row+2] = shared+wp.max(-u, 0.)
    out[w, row+3] = shared+wp.max(-v, 0.)

@wp.kernel
def contact_velocity(p: Parameters, free: wp.array2d(dtype=float), jac: wp.array3d(dtype=float),
                     force: wp.array2d(dtype=float), velocity: wp.array2d(dtype=float)):
    w, j = wp.tid()
    response = float(0.)
    for row in range(force.shape[1]):
        response += jac[w, row, j]*force[w, row]
    velocity[w, j] = (free[w, j]+p.inverse[j]*response)/p.dt

@wp.kernel
def integrate(p: Parameters, qpos: wp.array2d(dtype=float), velocity: wp.array2d(dtype=float),
              out: wp.array2d(dtype=float)):
    w = wp.tid()
    address = p.object_qpos
    for j in range(3):
        out[w, address+j] = qpos[w, address+j]+p.dt*velocity[w, j]
    q = wp.vec4(qpos[w, address+3], qpos[w, address+4], qpos[w, address+5], qpos[w, address+6])
    v = wp.vec3(q[1], q[2], q[3])
    omega = wp.vec3(velocity[w, 3], velocity[w, 4], velocity[w, 5])
    dv = q[0]*omega+wp.cross(v, omega)
    next_q = wp.normalize(q+.5*p.dt*wp.vec4(-wp.dot(v, omega), dv[0], dv[1], dv[2]))
    for j in range(4):
        out[w, address+3+j] = next_q[j]
    for j in range(26):
        address_hand = p.hand_qpos[j]
        out[w, address_hand] = qpos[w, address_hand]+p.dt*velocity[w, j+6]

@dataclass
class Stage:
    pose: object
    phi: object
    jac: object
    rays: object
    free: object
    projected: object
    force: object
    velocity: object
    selected: object
    rate: object
    forces: object
    responses: object
    trace: object
    projections: object
    gravity: object

class ExplicitPoseModel:
    def __init__(self, node):
        self.node, self.runtime = node, node.runtime
        r, d, s = self.runtime, node.cost, node.state
        self.device, self.kinematics = node.device, node.backend.kinematics
        if r.model.nq != 33:
            raise ValueError('Explicit grasp adapter requires 26 hand coordinates and one free object')
        self.settings = {k: getattr(r.args, 'explicit_'+k, v) for k, v in DEFAULTS.items()}
        self.horizon=int(self.settings['horizon'])
        if self.horizon<1 or self.settings['contact_iterations']<1:raise ValueError('Invalid explicit dynamics budget')
        arr = node.array
        p = self.parameters = Parameters()
        p.mesh, p.object_body, p.object_qpos = d.object_mesh, d.object_body, r.index.object_qpos
        p.seed, p.hand_qpos, p.root = s.seed_qpos, s.hand_ids, s.root_rotation
        bodies = np.unique(d.sample_body.numpy())
        source_body = d.sample_body.numpy()
        p.bodies = arr(bodies, int)
        p.starts = arr([np.flatnonzero(source_body == b)[0] for b in bodies], int)
        p.ends = arr([np.flatnonzero(source_body == b)[-1]+1 for b in bodies], int)
        p.samples = d.sample_local
        p.sample_normals, p.forbidden = d.sample_normal, d.sample_forbidden
        joints = np.array([np.flatnonzero(r.model.jnt_qposadr == address)[0] for address in r.index.hand_qpos])
        p.joints = arr(joints, int)
        p.joint_slide = arr(r.model.jnt_type[joints] == mujoco.mjtJoint.mjJNT_SLIDE, int)
        mass = r.model.body_mass*(1-r.model.body_gravcomp)
        mass[p.object_body] = 0.
        p.body_mass, p.body_com = arr(mass), arr(r.model.body_ipos, wp.vec3)
        ancestors = np.zeros((r.model.nbody, 26), np.int32)
        for body in range(r.model.nbody):
            chain, b = set(), body
            while b > 0:
                chain.add(b); b = r.model.body_parentid[b]
            ancestors[body] = np.isin(r.model.jnt_bodyid[joints], list(chain))
        p.body_ancestor, p.gravity = arr(ancestors, int), wp.vec3(r.model.opt.gravity)
        mask = np.zeros((len(bodies), 26), np.int32)
        for c, body in enumerate(bodies):
            ancestors = set()
            while body > 0:
                ancestors.add(int(body)); body = r.model.body_parentid[body]
            mask[c] = np.isin(r.model.jnt_bodyid[joints], list(ancestors))
        p.ancestor = arr(mask, int)
        kp = r.model.actuator_gainprm[np.asarray(r.index.hand_ctrl), 0]
        if not np.all(np.isfinite(kp)&(kp > 0)):
            raise ValueError('Explicit model requires positive position-actuator stiffness')
        p.stiffness = arr(kp)
        controls = np.asarray(r.index.hand_ctrl)
        limited = r.model.actuator_ctrllimited[controls].astype(bool)
        p.command_low = arr(np.where(limited, r.model.actuator_ctrlrange[controls, 0], -np.inf))
        p.command_high = arr(np.where(limited, r.model.actuator_ctrlrange[controls, 1], np.inf))
        force_limited = r.model.actuator_forcelimited[controls].astype(bool)
        p.force_low = arr(np.where(force_limited, r.model.actuator_forcerange[controls, 0], -np.inf))
        p.force_high = arr(np.where(force_limited, r.model.actuator_forcerange[controls, 1], np.inf))
        p.inverse = arr(np.r_[np.full(3, 1/self.settings['translation_metric']),
            np.full(3, 1/self.settings['rotation_metric']), 1/kp])
        gravity = np.array(r.model.opt.gravity)*float(r.model.body_mass[p.object_body])
        p.gravity_force = wp.vec3(gravity)
        p.friction, p.dt, p.sigma = r.args.friction, self.settings['dt'], self.settings['sigma']
        p.cutoff, p.weight = min(self.settings['contact_cutoff_mm']/1000, .001), self.settings['pose_weight']
        n, rows = node.state.population, len(bodies)*4
        options = dict(dtype=float, device=self.device, requires_grad=False)
        self.qpos = [wp.zeros((n, r.model.nq), **options) for _ in range(self.horizon+1)]
        self.stages = []
        for _ in range(self.horizon):
            forces = [wp.zeros((n, rows), **options) for _ in range(2)]
            responses = [wp.zeros((n, 32), **options)]
            projections = [wp.zeros((n, rows), **options)]
            self.stages.append(Stage(self.kinematics.allocate(n),
                wp.zeros((n, rows), **options), wp.zeros((n, rows, 32), **options),
                wp.zeros((n, rows), dtype=wp.vec3, device=self.device),
                wp.zeros((n, 32), **options), wp.zeros((n, rows), **options),
                forces[0], wp.zeros((n, 32), **options),
                wp.zeros((n, len(bodies)), dtype=int, device=self.device),
                wp.zeros(n, **options), forces, responses, wp.zeros(n, **options), projections,
                wp.zeros((n, 26), **options)))
        self.selected = self.stages[0].selected
        self.hold = wp.zeros((n, r.model.nq), **options)
        self.cost = wp.zeros(n, **options)
        self.step_costs = wp.zeros((n, self.horizon), **options)


    def forward(self, worlds, *, select=True):
        p, contacts = self.parameters, len(self.parameters.bodies)
        launch = lambda kernel, dim, args: wp.launch(kernel, dim, inputs=args, device=self.device)
        launch(initial_state, worlds, [p, self.node.state.x, self.qpos[0]])
        for k, stage in enumerate(self.stages):
            view = self.qpos[k] if worlds == self.node.state.population else self.qpos[k][:worlds]
            self.kinematics.evaluate(view, stage.pose)
            if select:
                # Discrete patch identities have no derivative. The selected
                # points' distances, projections, normals and J remain in AD.
                wp.launch(select_patches, (worlds, contacts), inputs=[p, stage.pose.body_position,
                    stage.pose.body_matrix, stage.selected], device=self.device, record_tape=False)
            launch(hand_gravity, (worlds, 26), [p, stage.pose.body_position, stage.pose.body_matrix,
                stage.pose.joint_anchor, stage.pose.joint_axis, stage.gravity])
            if k == 0:
                launch(hold_command, (worlds, 26), [p, self.qpos[0], self.node.state.x, stage.gravity, self.hold])
            launch(contact_rows, (worlds, contacts, 4), [p, stage.selected, stage.pose.body_position,
                stage.pose.body_matrix, stage.pose.joint_anchor, stage.pose.joint_axis, stage.phi, stage.jac, stage.rays])
            launch(free_displacement, (worlds, 32), [p, self.qpos[k], self.hold, stage.gravity, stage.free])
            launch(contact_trace, worlds, [p, stage.jac, stage.trace])
            launch(contact_step_size, worlds, [p, stage.trace, stage.rate])
            stage.forces[0].zero_()
            for i in range(self.settings['contact_iterations']):
                launch(projected_response, (worlds, 32), [p, stage.jac, stage.forces[i%2], stage.responses[0]])
                launch(projected_force, (worlds, contacts*4), [stage.jac, stage.free, stage.phi,
                    stage.responses[0], stage.projections[0]])
                launch(projected_update, (worlds, contacts), [p, stage.rate, stage.forces[i%2],
                    stage.projections[0], stage.forces[(i+1)%2]])
            launch(contact_velocity, (worlds, 32), [p, stage.free, stage.jac, stage.forces[self.settings['contact_iterations']%2], stage.velocity])
            launch(integrate, worlds, [p, self.qpos[k], stage.velocity, self.qpos[k+1]])
            launch(step_pose_cost, worlds, [p, self.qpos[k+1], k, self.step_costs])
        launch(average_pose_cost, worlds, [self.step_costs, self.cost])

