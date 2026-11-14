"""Additional contact objectives as a Robokit Task; source task costs stay source-owned."""
import numpy as np
import warp as wp
from mujoco_warp._src.support import contact_force_fn
from mujoco_warp._src.types import vec5
from robokit.terms.task import Task
from robokit.terms.simulation import SimContext
from .objective import force_penalty_parameters
from retarget.solver.contact_defaults import DEFAULTS


@wp.kernel(enable_backward=False)
def _state_cost(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
                qacc: wp.array2d(dtype=float), actuator_force: wp.array2d(dtype=float),
                ctrl: wp.array2d(dtype=float), previous: wp.array2d(dtype=float),
                joint_adr: wp.array(dtype=int), limits: wp.array2d(dtype=float), torque: wp.array(dtype=float),
                qref: wp.array2d(dtype=float), vref: wp.array2d(dtype=float),
                clock: wp.array(dtype=int), tracking: int, root_weight: wp.vec3,
                rotation_weight: float, joint_weight: float, root_velocity_weight: float,
                joint_velocity_weight: float, cost: wp.array(dtype=float)):
    w, c = wp.tid(), float(0.)
    t = wp.min(clock[1], qref.shape[0] - 1)
    for j in range(joint_adr.shape[0]):
        q = qpos[w, joint_adr[j]]
        error = wp.max(wp.max(limits[j, 0] - q, q - limits[j, 1]), 0.)
        c += 10. * error * error
    for j in range(actuator_force.shape[1]):
        error = actuator_force[w, j] / torque[j]
        c += 1.e-5 * error * error
        change = ctrl[w, j] - previous[w, j]
        c += .001 * change * change
        previous[w, j] = ctrl[w, j]
    for j in range(qacc.shape[1]):
        c += 1.e-8 * qacc[w, j] * qacc[w, j]
    if tracking == 1:
        for j in range(qpos.shape[1]):
            if j < 3 or j >= 7:
                e = qpos[w, j] - qref[t, j]
                weight = wp.where(j < 3, root_weight[wp.min(j, 2)], joint_weight)
                c += weight * e * e
        dot = float(0.)
        for j in range(3, 7):
            dot += qpos[w, j] * qref[t, j]
        angle = 2. * wp.acos(wp.clamp(wp.abs(dot), 0., 1.))
        c += rotation_weight * angle * angle
        for j in range(qvel.shape[1]):
            e = qvel[w, j] - vref[t, j]
            c += wp.where(j < 6, root_velocity_weight, joint_velocity_weight) * e * e
    cost[w] += c


@wp.kernel(enable_backward=False)
def _contact_cost(nacon: wp.array(dtype=int), worldid: wp.array(dtype=int),
                  geoms: wp.array(dtype=wp.vec2i), dist: wp.array(dtype=float),
                  frame: wp.array(dtype=wp.mat33), friction: wp.array(dtype=vec5),
                  dim: wp.array(dtype=int), address: wp.array2d(dtype=int),
                  force: wp.array2d(dtype=float), njmax: int, cone: int,
                  cvel: wp.array2d(dtype=wp.spatial_vector), subtree_com: wp.array2d(dtype=wp.vec3),
                  pos: wp.array(dtype=wp.vec3), geom_body: wp.array(dtype=int), root_body: wp.array(dtype=int),
                  owner: wp.array(dtype=int), targets: wp.array(dtype=int), support_obj: int,
                  support_margin: float, extra: int, stride: int, force_scale: float, force_weight: float,
                  clock: wp.array(dtype=int), horizon: int,
                  touch: wp.array2d(dtype=int), cost: wp.array(dtype=float)):
    i = wp.tid()
    if i >= nacon[0]:
        return
    w = worldid[i]
    g1, g2 = geoms[i][0], geoms[i][1]
    if support_obj >= 0 and dist[i] <= support_margin:
        other = wp.where(g1 == support_obj, g2, wp.where(g2 == support_obj, g1, -1))
        if other >= 0:
            wp.atomic_max(touch, w, owner[other], 1)
    if extra == 1:
        f = contact_force_fn(cone, frame, friction, dim, address, force, njmax, nacon, w, i, False)
        fn = wp.max(f[0], 0.0)
        violation = wp.max(wp.sqrt(f[1] * f[1] + f[2] * f[2]) - wp.min(friction[i][0], friction[i][1]) * fn, 0.0)
        c = force_weight * wp.pow(wp.max(fn / force_scale - 1.0, 0.0), 2.0) + 1.0e-4 * violation * violation
        if targets[g1] == 1 or targets[g2] == 1:
            b1, b2 = geom_body[g1], geom_body[g2]
            v1, v2 = cvel[w, b1], cvel[w, b2]
            p = pos[i]
            vel = wp.spatial_bottom(v2) + wp.cross(wp.spatial_top(v2), p - subtree_com[w, root_body[b2]])
            vel -= wp.spatial_bottom(v1) + wp.cross(wp.spatial_top(v1), p - subtree_com[w, root_body[b1]])
            n = frame[i][0]
            slip = vel - wp.dot(vel, n) * n
            c += .001 * wp.dot(slip, slip)
        else:
            c += 1000.0 * wp.pow(wp.max(-dist[i] - .001, 0.0), 2.0)
        wp.atomic_add(cost, w, wp.where(clock[1] + stride >= horizon, 1.0, float(stride)) * c)


@wp.kernel(enable_backward=False)
def _events(geom_pos: wp.array2d(dtype=wp.vec3), geom_mat: wp.array2d(dtype=wp.mat33),
            values: wp.array3d(dtype=float), pairs: wp.array3d(dtype=int),
            cvel: wp.array2d(dtype=wp.spatial_vector), subtree_com: wp.array2d(dtype=wp.vec3),
            geom_body: wp.array(dtype=int), root_body: wp.array(dtype=int),
            clock: wp.array(dtype=int), dt: float, stride: int, cost: wp.array(dtype=float)):
    w, i = wp.tid()
    g1, g2 = pairs[w, i, 0], pairs[w, i, 1]
    time = float(clock[1]) * dt
    if g1 < 0 or time < values[w, i, 12] or time > values[w, i, 12] + values[w, i, 13]:
        return
    lp = wp.vec3(values[w, i, 0], values[w, i, 1], values[w, i, 2])
    op = wp.vec3(values[w, i, 3], values[w, i, 4], values[w, i, 5])
    ln = wp.vec3(values[w, i, 6], values[w, i, 7], values[w, i, 8])
    on = wp.vec3(values[w, i, 9], values[w, i, 10], values[w, i, 11])
    p1 = geom_pos[w, g1] + geom_mat[w, g1] * lp
    p2 = geom_pos[w, g2] + geom_mat[w, g2] * op
    n1, n2 = geom_mat[w, g1] * ln, geom_mat[w, g2] * on
    delta = p1 - p2
    mode = values[w, i, 14]
    gap = wp.max(wp.where(mode == 0.0, wp.length(delta), wp.abs(wp.dot(delta, n2))) - .002, 0.0)
    c = 100.0 * gap * gap + .01 * wp.pow(1.0 + wp.dot(n1, n2), 2.0)
    if mode != 0.0:
        tangent_error = delta - wp.dot(delta, n2) * n2
        c += .1 * wp.dot(tangent_error, tangent_error)
    b1, b2 = geom_body[g1], geom_body[g2]
    v1, v2 = cvel[w, b1], cvel[w, b2]
    relative = wp.spatial_bottom(v1) + wp.cross(wp.spatial_top(v1), p1 - subtree_com[w, root_body[b1]])
    relative -= wp.spatial_bottom(v2) + wp.cross(wp.spatial_top(v2), p2 - subtree_com[w, root_body[b2]])
    vn = wp.dot(relative, n2)
    vt = wp.vec3(0.0, 0.0, 0.0)
    if mode == 1.0:
        vt = wp.vec3(values[w, i, 16], values[w, i, 17], values[w, i, 18])
        vt -= wp.dot(vt, n2) * n2
    tangent = relative - vn * n2 - vt
    c += .01 * (wp.pow(vn - values[w, i, 15], 2.0) + wp.dot(tangent, tangent))
    wp.atomic_add(cost, w, float(stride) * c)


class ContactDynamicsTask(Task):
    """Contact-event extension using Robokit's Task/SimContext lifecycle."""
    SUPPORTS_AUTODIFF = False

    def __init__(self, model, horizon: int, tracking: bool = False):
        self.cpu, self.horizon, self.tracking = model, horizon, int(tracking)
        self.models, self.bindings = {}, []

    def register(self, sim) -> None:
        self.models[sim.data.qpos.ptr] = sim
        if not hasattr(sim, '_contact_pre_state'):
            pre_q, pre_v = wp.empty_like(sim.data.qpos), wp.empty_like(sim.data.qvel)
            original_step = sim.step
            def step():
                # Captured inside the original SimMPC step graph; no dynamics
                # implementation is replaced. Force/geometry remain pre-step.
                wp.copy(pre_q, sim.data.qpos)
                wp.copy(pre_v, sim.data.qvel)
                original_step()
            sim._contact_pre_state = pre_q, pre_v
            sim.step = step

    def resolve(self, var_values, context) -> None:
        self.sim = self.models[context[SimContext.QPOS].ptr]
        self.clock = context[SimContext.CLOCK]
        sim, m, H = self.sim, self.cpu, self.horizon
        self.pre_q, self.pre_v = sim._contact_pre_state
        self.force_scale, self.force_weight = force_penalty_parameters(m)
        device, worlds = sim.device, sim.nworld
        self.targets = wp.zeros(m.ngeom, dtype=int, device=device)
        self.values = wp.zeros((worlds, 3, 19), dtype=float, device=device)
        self.pairs = wp.full((worlds, 3, 2), -1, dtype=int, device=device)
        self.geom_body = wp.array(m.geom_bodyid, dtype=int, device=device)
        self.root_body = wp.array(m.body_rootid, dtype=int, device=device)
        self.owner = wp.zeros(m.ngeom, dtype=int, device=device)
        self.touch = wp.zeros((worlds, 6), dtype=int, device=device)
        joints = np.flatnonzero(m.jnt_limited & np.isin(m.jnt_type, [2, 3]))
        self.joint_adr = wp.array(m.jnt_qposadr[joints], dtype=int, device=device)
        self.limits = wp.array(m.jnt_range[joints], dtype=float, device=device)
        self.torque = wp.array(np.maximum(np.max(np.abs(m.actuator_forcerange), axis=1), 1.), dtype=float, device=device)
        self.qref = wp.zeros((H + 1, m.nq), dtype=float, device=device)
        self.vref = wp.zeros((H + 1, m.nv), dtype=float, device=device)
        self.previous = wp.zeros_like(sim.data.ctrl)
        self.initial_ctrl = wp.zeros_like(sim.data.ctrl)
        self.bindings.append(self)
        # SimMPC creates its one-world validation graph lazily. It must inherit
        # the current window's references, rather than score against zeros.
        if hasattr(self, 'pending_update'):
            self.update(*self.pending_update)

    def update(self, packet: dict, events: list, count: int) -> None:
        self.pending_update = packet, events, count
        for built in self.bindings:
            worlds = built.sim.nworld
            values = np.zeros((worlds, 3, 19), np.float32)
            pairs = np.full((worlds, 3, 2), -1, np.int32)
            for row in range(min(count, worlds)):
                for i, event in enumerate(events[row][:3]):
                    pairs[row, i] = [event['geom1'], event['geom2']]
                    values[row, i] = [*event['link_local_point'], *event['object_local_point'],
                        *event['link_local_normal'], *event['object_local_normal'],
                        event['start_time'], event['duration'],
                        float(('stick', 'slide', 'roll').index(event['mode'])),
                        event['normal_velocity'], *event['tangential_velocity']]
            built.values.assign(values); built.pairs.assign(pairs)
            target = np.zeros(self.cpu.ngeom, np.int32)
            target[packet.get('target_geoms', [])] = 1
            built.targets.assign(target)
            built.initial_ctrl.assign(np.broadcast_to(packet['initial_ctrl'], (worlds, self.cpu.nu)).copy())
            if 'qpos_ref' in packet:
                built.qref.assign(packet['qpos_ref']); built.vref.assign(packet['qvel_ref'])

    def reset(self, sim) -> None:
        for built in self.bindings:
            if built.sim is sim:
                wp.copy(built.previous, built.initial_ctrl)

    def compute_weighted_cost(self, var_values, *, out_cost: wp.array) -> None:
        sim, d, m = self.sim, self.sim._d, self.sim._m
        wp.launch(_state_cost, dim=sim.nworld, inputs=[self.pre_q, self.pre_v, d.qacc, d.actuator_force,
            d.ctrl, self.previous, self.joint_adr, self.limits, self.torque, self.qref, self.vref,
            self.clock, self.tracking, wp.vec3(*DEFAULTS.tracking_root_position_weight),
            DEFAULTS.tracking_root_rotation_weight, DEFAULTS.tracking_joint_position_weight,
            DEFAULTS.tracking_root_velocity_weight, DEFAULTS.tracking_joint_velocity_weight, out_cost], device=sim.device)
        wp.launch(_contact_cost, dim=d.naconmax, inputs=[d.nacon, d.contact.worldid, d.contact.geom,
            d.contact.dist, d.contact.frame, d.contact.friction, d.contact.dim, d.contact.efc_address,
            d.efc.force, d.njmax, m.opt.cone, d.cvel, d.subtree_com, d.contact.pos,
            self.geom_body, self.root_body, self.owner, self.targets, -1, 0., 1, 1, self.force_scale, self.force_weight,
            self.clock, self.horizon, self.touch, out_cost], device=sim.device)
        wp.launch(_events, dim=(sim.nworld, 3), inputs=[d.geom_xpos, d.geom_xmat,
            self.values, self.pairs, d.cvel, d.subtree_com, self.geom_body, self.root_body,
            self.clock, float(self.cpu.opt.timestep), 1, out_cost], device=sim.device)
