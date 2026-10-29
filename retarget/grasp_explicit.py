"""Differentiable compliant contact prediction with independent servo preload.

Gravity and actuator forces drive a coupled projected contact solve. No
force-balance objective is added. Q remains a quasi-static displacement metric;
this is a lightweight model, not a full inertial MuJoCo rollout.
"""
from dataclasses import dataclass
import math

import mujoco
import numpy as np
import warp as wp
from retarget.grasp_cem import CostData

UPSTREAM = 'https://github.com/asu-iris/Complementarity-Free-Dexterous-Manipulation'
COMMIT = '87b320ffb4b40ab4f3aa344371f6e5e8d619aa7e'
DEFAULTS = dict(horizon=4, dt=.1, sigma=.5, pose_weight=1.e6,
    translation_metric=50., rotation_metric=.1, contact_cutoff_mm=1.,
    refinement_steps=32, refinement_step_size=1., line_search_steps=8,
    contact_iterations=64, preload_limit=.5, motion_tolerance_mm=2., rotation_tolerance_deg=5.)


@wp.struct
class Parameters:
    mesh: wp.uint64
    object_body: int
    object_qpos: int
    hand_qpos: wp.array(dtype=int)
    joints: wp.array(dtype=int)
    joint_slide: wp.array(dtype=int)
    ancestor: wp.array2d(dtype=int)
    bodies: wp.array(dtype=int)
    starts: wp.array(dtype=int)
    ends: wp.array(dtype=int)
    samples: wp.array(dtype=wp.vec3)
    sample_normals: wp.array(dtype=wp.vec3)
    forbidden: wp.array(dtype=int)
    body_mass: wp.array(dtype=float)
    body_com: wp.array(dtype=wp.vec3)
    body_ancestor: wp.array2d(dtype=int)
    gravity: wp.vec3
    force_low: wp.array(dtype=float)
    force_high: wp.array(dtype=float)
    seed: wp.array(dtype=float)
    inverse: wp.array(dtype=float)
    stiffness: wp.array(dtype=float)
    command_low: wp.array(dtype=float)
    command_high: wp.array(dtype=float)
    root: wp.mat33
    gravity_force: wp.vec3
    friction: float
    dt: float
    sigma: float
    cutoff: float
    weight: float


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


@wp.kernel
def terminal_pose_cost(p: Parameters, qpos: wp.array2d(dtype=float), cost: wp.array(dtype=float)):
    w = wp.tid()
    address = p.object_qpos
    error = wp.vec3(qpos[w, address]-p.seed[address], qpos[w, address+1]-p.seed[address+1],
                    qpos[w, address+2]-p.seed[address+2])
    q = wp.normalize(wp.vec4(qpos[w, address+3], qpos[w, address+4], qpos[w, address+5], qpos[w, address+6]))
    goal = wp.normalize(wp.vec4(p.seed[address+3], p.seed[address+4], p.seed[address+5], p.seed[address+6]))
    dot = wp.clamp(wp.dot(q, goal), -1., 1.)
    cost[w] = p.weight*(5000.*wp.length_sq(error)+50.*wp.max(1.-dot*dot, 0.))


@wp.kernel
def step_pose_cost(p: Parameters, qpos: wp.array2d(dtype=float), step: int,
                   costs: wp.array2d(dtype=float)):
    w = wp.tid()
    address = p.object_qpos
    error = wp.vec3(qpos[w, address]-p.seed[address], qpos[w, address+1]-p.seed[address+1],
                    qpos[w, address+2]-p.seed[address+2])
    q = wp.normalize(wp.vec4(qpos[w, address+3], qpos[w, address+4], qpos[w, address+5], qpos[w, address+6]))
    goal = wp.normalize(wp.vec4(p.seed[address+3], p.seed[address+4], p.seed[address+5], p.seed[address+6]))
    dot = wp.clamp(wp.dot(q, goal), -1., 1.)
    costs[w, step] = p.weight*(5000.*wp.length_sq(error)+50.*wp.max(1.-dot*dot, 0.))


@wp.kernel
def average_pose_cost(costs: wp.array2d(dtype=float), out: wp.array(dtype=float)):
    w = wp.tid()
    value = float(0.)
    for step in range(costs.shape[1]):
        value += costs[w, step]/float(costs.shape[1])
    out[w] = value


@wp.kernel
def scalar_loss(cost: wp.array(dtype=float), loss: wp.array(dtype=float)):
    loss[0] = cost[0]


@wp.kernel(enable_backward=False)
def attraction_selection(d: CostData, position: wp.array2d(dtype=wp.vec3),
                         rotation: wp.array2d(dtype=wp.mat33), selected: wp.array2d(dtype=int)):
    w, group = wp.tid()
    body, obj = d.group_body[group], d.object_body
    anchor = position[w, obj]+rotation[w, obj]@d.anchor[group]
    pref = position[w, body]+rotation[w, body]@d.preference[group]
    best, index = float(wp.inf), int(d.group_start[group])
    for i in range(d.group_start[group], d.group_end[group]):
        if d.sample_pad[i] != 0:
            point = position[w, body]+rotation[w, body]@d.sample_local[i]
            score = wp.length_sq(point-anchor)+d.alpha[group]*wp.length_sq(point-pref)
            if score < best:
                best, index = score, i
    selected[w, group] = index


@wp.kernel
def attraction_style(d: CostData, position: wp.array2d(dtype=wp.vec3),
                     rotation: wp.array2d(dtype=wp.mat33), selected: wp.array2d(dtype=int),
                     x: wp.array2d(dtype=float), seed: wp.array(dtype=float),
                     pose_cost: wp.array(dtype=float), objective: wp.array(dtype=float)):
    """Same contact, style and smoothness values as CARD, now in reverse AD."""
    w = wp.tid()
    value = pose_cost[w]
    for group in range(d.group_body.shape[0]):
        body, obj = int(d.group_body[group]), int(d.object_body)
        point = position[w, body]+rotation[w, body]@d.sample_local[selected[w, group]]
        anchor = position[w, obj]+rotation[w, obj]@d.anchor[group]
        normal = rotation[w, body]@d.sample_normal[selected[w, group]]
        push = rotation[w, obj]@d.push[group]
        previous = position[w, body]+rotation[w, body]@d.previous_local[group]
        reach = wp.length_sq((point-anchor)*1000.)
        cone = 200.*(1.-wp.clamp(wp.dot(normal, push), -1., 1.))
        fix = wp.length_sq((point-previous)*1000.)
        value += d.rho[group]*d.rho[group]*(1600.*(reach+cone)+6.25*fix)
    for i in range(15):
        style_body, axis = int(d.style_body[i]), wp.vec3(d.style_axis[i])
        value += 400.*(1.-wp.clamp(wp.dot(rotation[w, style_body]@axis, d.seed_rotation[style_body]@axis), -1., 1.))
        if i % 3 == 2:
            apex = position[w, style_body]+rotation[w, style_body]@d.style_apex[i]
            seed_apex = d.seed_position[style_body]+d.seed_rotation[style_body]@d.style_apex[i]
            value += 9.*wp.length_sq((apex-seed_apex)*1000.)
            value += 3600.*(1.-wp.clamp(wp.dot(rotation[w, style_body]@d.style_normal[i], d.seed_rotation[style_body]@d.style_normal[i]), -1., 1.))
    for j in range(seed.shape[0]):
        delta = x[w, j]-seed[j]
        if j < 3:
            value += 125.e6*delta*delta
        elif j < 6:
            value += 100.*delta*delta
        else:
            value += .01*delta*delta
    objective[w] = value


@wp.kernel(enable_backward=False)
def gradient_proposal(x: wp.array2d(dtype=float), gradient: wp.array2d(dtype=float),
                      best: wp.array(dtype=float), low: wp.array(dtype=float), high: wp.array(dtype=float),
                      seed: wp.array(dtype=float), sigma: wp.array(dtype=float), step: float, block_start: int):
    norm = float(0.)
    for j in range(block_start, best.shape[0]):
        norm += wp.pow(gradient[0, j]*sigma[j], 2.)
    norm = wp.max(wp.sqrt(norm), 1.e-20)
    for j in range(best.shape[0]):
        increment = step*sigma[j]*sigma[j]*gradient[0, j]/norm
        if j < block_start:
            increment = 0.
        value = best[j]-increment
        if not wp.isfinite(value):
            value = best[j]
        x[0, j] = wp.clamp(value, low[j], high[j])
    delta = wp.vec3(x[0, 0]-seed[0], x[0, 1]-seed[1], x[0, 2]-seed[2])
    if wp.length(delta) > .02:
        delta = .02*wp.normalize(delta)
        for j in range(3):
            x[0, j] = seed[j]+delta[j]


@wp.kernel(enable_backward=False)
def accept_proposal(x: wp.array2d(dtype=float), score: wp.array(dtype=float),
                    best: wp.array(dtype=float), best_score: wp.array(dtype=float),
                    accepted: wp.array(dtype=int), history: wp.array(dtype=float), iteration: int):
    if wp.isfinite(score[0]) and score[0] < best_score[0]:
        for j in range(best.shape[0]):
            best[j] = x[0, j]
        best_score[0] = score[0]
        accepted[0] += 1
    history[iteration] = best_score[0]


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
        from comfree_warp.native_adjoint.kinematics import allocate_output
        self.node, self.runtime = node, node.runtime
        r, d, s = self.runtime, node.cost, node.state
        self.device, self.kinematics = node.device, node.backend.compiled.base.kinematics
        self.kinematics_vjp = node.backend.compiled.base.kinematics_vjp
        self.recording = False
        if r.model.nq != 33:
            raise ValueError('Explicit grasp adapter requires 26 hand coordinates and one free object')
        self.settings = {k: getattr(r.args, 'explicit_'+k, v) for k, v in DEFAULTS.items()}
        if (self.settings['horizon'] < 1 or self.settings['refinement_steps'] < 0
                or self.settings['contact_iterations'] < 1 or not 0 < self.settings['sigma'] <= 1):
            raise ValueError('Explicit horizon must be positive and refinement steps nonnegative')
        if not all(math.isfinite(v) and v > 0 for k, v in self.settings.items() if k not in ('horizon', 'refinement_steps')):
            raise ValueError('Explicit model settings must be finite and positive')
        self.horizon = int(self.settings['horizon'])
        self.refinement_steps = int(self.settings['refinement_steps'])
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
        # Dynamics refines the one geometric incumbent. Keep its trajectory,
        # contact history and adjoints independent of the CEM population size.
        n, rows = 1, len(bodies)*4
        options = dict(dtype=float, device=self.device, requires_grad=True)
        self.qpos = [wp.zeros((n, r.model.nq), **options) for _ in range(self.horizon+1)]
        self.stages = []
        for _ in range(self.horizon):
            forces = [wp.zeros((n, rows), **options) for _ in range(self.settings['contact_iterations']+1)]
            responses = [wp.zeros((n, 32), **options) for _ in range(self.settings['contact_iterations'])]
            projections = [wp.zeros((n, rows), **options) for _ in range(self.settings['contact_iterations'])]
            self.stages.append(Stage(allocate_output(self.kinematics, n, requires_grad=True),
                wp.zeros((n, rows), **options), wp.zeros((n, rows, 32), **options),
                wp.zeros((n, rows), dtype=wp.vec3, device=self.device, requires_grad=True),
                wp.zeros((n, 32), **options), wp.zeros((n, rows), **options),
                forces[-1], wp.zeros((n, 32), **options),
                wp.zeros((n, len(bodies)), dtype=int, device=self.device),
                wp.zeros(n, **options), forces, responses, wp.zeros(n, **options), projections,
                wp.zeros((n, 26), **options)))
        self.selected = self.stages[0].selected
        self.hold = wp.zeros((n, r.model.nq), **options)
        self.attraction_selected = wp.zeros((n, len(d.group_body)), dtype=int, device=self.device)
        self.cost, self.objective, self.loss = wp.zeros(n, **options), wp.zeros(n, **options), wp.zeros(1, **options)
        self.step_costs = wp.zeros((n, self.horizon), **options)
        self.accepted = wp.zeros(1, dtype=int, device=self.device)
        self.base = wp.zeros(len(s.seed), dtype=float, device=self.device)
        self.initial_x = wp.zeros(len(s.seed), dtype=float, device=self.device)
        self.before = wp.zeros(1, dtype=float, device=self.device)
        self.pose_before = wp.zeros(1, dtype=float, device=self.device)
        self.history = wp.zeros(self.refinement_steps*self.settings['line_search_steps'], dtype=float, device=self.device)
        s.x.requires_grad = True
        self.tape = wp.Tape()

    def forward(self, worlds, *, select=True):
        from comfree_warp.native_adjoint.kinematics import kinematics
        if worlds != 1:
            raise ValueError('Contact dynamics refines one geometric incumbent')
        p, contacts = self.parameters, len(self.parameters.bodies)
        launch = lambda kernel, dim, args: wp.launch(kernel, dim, inputs=args, device=self.device)
        launch(initial_state, worlds, [p, self.node.state.x, self.qpos[0]])
        for k, stage in enumerate(self.stages):
            view = self.qpos[k] if worlds == self.node.state.population else self.qpos[k][:worlds]
            kinematics(self.kinematics, view, stage.pose, record_tape=False)
            if self.recording:
                from comfree_warp.native_adjoint.kinematics_vjp import add_kinematics_vjp
                qpos, pose = self.qpos[k], stage.pose
                self.tape.record_func(lambda qpos=qpos, pose=pose, view=view: add_kinematics_vjp(
                    self.kinematics_vjp, pose, view, output=qpos.grad),
                    arrays=[qpos, pose.body_position, pose.body_quaternion, pose.body_matrix,
                            pose.joint_anchor, pose.joint_axis])
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
            for i in range(self.settings['contact_iterations']):
                launch(projected_response, (worlds, 32), [p, stage.jac, stage.forces[i], stage.responses[i]])
                launch(projected_force, (worlds, contacts*4), [stage.jac, stage.free, stage.phi,
                    stage.responses[i], stage.projections[i]])
                launch(projected_update, (worlds, contacts), [p, stage.rate, stage.forces[i],
                    stage.projections[i], stage.forces[i+1]])
            launch(contact_velocity, (worlds, 32), [p, stage.free, stage.jac, stage.force, stage.velocity])
            launch(integrate, worlds, [p, self.qpos[k], stage.velocity, self.qpos[k+1]])
            launch(step_pose_cost, worlds, [p, self.qpos[k+1], k, self.step_costs])
        launch(average_pose_cost, worlds, [self.step_costs, self.cost])
        pose, d = self.stages[0].pose, self.node.cost
        wp.launch(attraction_selection, (worlds, len(d.group_body)), inputs=[d, pose.body_position,
            pose.body_matrix, self.attraction_selected], device=self.device, record_tape=False)
        launch(attraction_style, worlds, [d, pose.body_position, pose.body_matrix, self.attraction_selected,
            self.node.state.x, self.node.state.seed, self.cost, self.objective])

    def backward(self):
        # Fresh tapes keep stage dependencies distinct; zero prior adjoints.
        self.tape.zero()
        self.tape.reset()
        self.recording = True
        try:
            with self.tape:
                self.forward(1)
                wp.launch(scalar_loss, 1, inputs=[self.objective, self.loss], device=self.device)
        finally:
            self.recording = False
        self.tape.backward(self.loss)

    def export(self):
        p = self.parameters
        qpos = self.qpos[-1].numpy()[0].copy()
        initial = self.qpos[0].numpy()[0]
        forces = self.stages[0].force.numpy()[0]
        rays = self.stages[0].rays.numpy()[0]
        q, goal = qpos[p.object_qpos+3:p.object_qpos+7], initial[p.object_qpos+3:p.object_qpos+7]
        angle = 2*np.arccos(np.clip(abs(np.dot(q, goal)/(np.linalg.norm(q)*np.linalg.norm(goal))), 0., 1.))
        gradient = self.node.state.x.grad.numpy()[0]
        finite_gradient = bool(np.isfinite(gradient).all())
        active = np.linalg.norm(rays.reshape(-1, 4, 3), axis=-1).max(axis=1) > 0
        phi = self.stages[0].phi.numpy()[0][::4]
        force_world = (forces[:, None]*rays).reshape(-1, 4, 3).sum(axis=1)
        trajectory = np.stack([state.numpy()[0] for state in self.qpos])
        maximum_shift = float(np.linalg.norm(trajectory[:, p.object_qpos:p.object_qpos+3]-initial[p.object_qpos:p.object_qpos+3], axis=1).max()*1000)
        quaternions = trajectory[:, p.object_qpos+3:p.object_qpos+7]
        dots = np.abs(quaternions@goal)/(np.linalg.norm(quaternions, axis=1)*np.linalg.norm(goal))
        maximum_angle = float(np.degrees(2*np.arccos(np.clip(dots, 0., 1.))).max())
        stable = bool(maximum_shift <= self.settings['motion_tolerance_mm'] and maximum_angle <= self.settings['rotation_tolerance_deg'])
        return qpos, dict(enabled=True, upstream=f'{UPSTREAM}/tree/{COMMIT}',
            settings=self.settings, target='initial_object_pose', target_pose=initial[p.object_qpos:p.object_qpos+7].tolist(),
            terminal_pose=qpos[p.object_qpos:p.object_qpos+7].tolist(),
            predicted_displacement_mm=float(np.linalg.norm(qpos[p.object_qpos:p.object_qpos+3]-initial[p.object_qpos:p.object_qpos+3])*1000),
            predicted_rotation_deg=float(np.degrees(angle)), cost=float(self.cost.numpy()[0]),
            predicted_max_displacement_mm=maximum_shift, predicted_max_rotation_deg=maximum_angle,
            predicted_hold_feasible=stable,
            motion_objective='mean pose tracking at every prediction step',
            quasi_static_metric_not_physical_inertia=True, quaternion_normalized=True,
            contact_solver='coupled_projected_nonnegative_friction_rays',
            effective_contact_cutoff_mm=p.cutoff*1000,
            preload_offsets=self.node.state.x.numpy()[0, 26:].tolist(),
            hold_command=self.hold.numpy()[0, p.hand_qpos.numpy()].tolist(),
            hand_gravity_compensated_at_candidate=True,
            gravity_force=list(p.gravity_force), hand_stiffness=p.stiffness.numpy().tolist(),
            contact_body_ids=p.bodies.numpy().tolist(), contact_sample_indices=self.selected.numpy()[0].tolist(),
            contact_local_points=p.samples.numpy()[np.maximum(self.selected.numpy()[0], 0)].tolist(),
            initial_phi_m=phi.tolist(), initial_contact_active=active.tolist(),
            initial_gap_m=[float(value) if enabled else None for value, enabled in zip(phi, active)],
            initial_row_force=forces.tolist(), initial_force_world=force_world.tolist(),
            initial_net_object_force_world=(force_world.sum(axis=0)+np.array(list(p.gravity_force))).tolist(),
            force_source='projected_compliant_contact_prediction_not_measurement',
            gradient='Warp reverse AD through FK, mesh gap, Jacobian, coupled projected contact solve and servo preload',
            gradient_scope='object_pose, IK contact attraction, style and smoothness; full objective acceptance',
            refinement_accepted=int(self.accepted.numpy()[0]),
            refinement_total_cost_before=float(self.before.numpy()[0]) if np.isfinite(self.before.numpy()[0]) else None,
            refinement_total_cost_after=float(self.node.state.best_score.numpy()[0]) if np.isfinite(self.node.state.best_score.numpy()[0]) else None,
            refinement_pose_cost_before=float(self.pose_before.numpy()[0]),
            pose_gradient_finite=finite_gradient,
            pose_gradient_norm=float(np.linalg.norm(gradient)) if finite_gradient else None,
            pose_gradient=gradient.tolist() if finite_gradient and self.refinement_steps > 0 else None,
            gradient_at='optimized_static_hand_pose' if self.refinement_steps > 0 else 'not_computed',
            refinement_best_cost_history=[float(v) if np.isfinite(v) else None for v in self.history.numpy()])
