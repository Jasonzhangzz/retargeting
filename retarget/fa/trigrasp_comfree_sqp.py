"""Measured-geometry symbolic MPC; servo increments are executed as predicted.

All guidance slots are retained, including the thumb. Contact geometry and
inertia are measured once per window. CasADi differentiates the reduced
prediction; no Warp forward trajectory is used by this planner. This reduced
contact model is not numerically identical to the complete Warp executor.
"""
from __future__ import annotations
from pathlib import Path
from time import perf_counter
import casadi as cs
import mujoco
import numpy as np
from scipy.spatial import ConvexHull


def _quat_matrix(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return cs.vertcat(
        cs.horzcat(1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
        cs.horzcat(2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
        cs.horzcat(2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))


def _quat_step(q, omega, h):
    out = q + .5*h*cs.vertcat(-cs.dot(q[1:], omega), q[0]*omega+cs.cross(q[1:], omega))
    return out / cs.sqrt(cs.sumsqr(out)+1e-20)


def _softplus(x, beta):
    # Unlike clipping beta*x to 60, this preserves the large-force derivative.
    y = beta*x
    return (cs.fmax(y, 0)+cs.log(1+cs.exp(-cs.fabs(y))))/beta


def servo_commands(previous, increments, lower, upper):
    """The command accumulated inside MPC is the actual executor position target."""
    values = np.asarray(increments, float)
    previous = np.asarray(previous, float).copy()
    result = []
    for delta in values:
        previous = np.clip(previous+delta, lower, upper)
        result.append(previous.copy())
    return np.asarray(result)


class TrigraspComFreeSQP:
    def __init__(self, backend, *, radius=1., threads=1, build_solver=True):
        self.backend, self.model, self.index, self.cfg = backend, backend.cpu_model, backend.index, backend.cfg
        self.N = len(backend.steps)
        self.nu, self.n_contacts = self.index.action_dim, backend.guidance_count
        self.h = float(self.cfg.simulator.time_grid.mpc_dt)
        self.nq, self.nv = 7+self.nu, 6+self.nu
        self.nx = self.nq+self.nv+self.nu
        self.beta = max(float(self.cfg.optimizer.contact_softness), 1.)
        self.radius = float(radius)
        self.hand_dofs = np.asarray([self.model.jnt_dofadr[int(np.flatnonzero(self.model.jnt_qposadr == q)[0])]
                                     for q in self.index.hand_qpos], int)
        self.object_dofs = np.arange(self.index.object_qpos, self.index.object_qpos+6)
        joints = self.model.actuator_trnid[self.index.hand_ctrl, 0]
        limited = self.model.jnt_limited[joints].astype(bool)
        self.lower = np.where(limited, self.model.jnt_range[joints, 0], -1e6)
        self.upper = np.where(limited, self.model.jnt_range[joints, 1], 1e6)
        self.kp = self.model.actuator_gainprm[self.index.hand_ctrl, 0].copy()
        self.kd = -self.model.actuator_biasprm[self.index.hand_ctrl, 2].copy()
        self.physical_limit = self.h*np.r_[np.full(3, .25), np.ones(3), np.full(self.nu-6, 1.5)]
        self.physical_limit *= min(self.radius, 1.)
        self.data = mujoco.MjData(self.model)
        self.hull_planes = self._object_hull()
        planes = np.flatnonzero((self.model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
                                & (self.model.geom_bodyid == 0))
        self.floor = None
        if len(planes):
            g = int(planes[0]); rot = np.empty(9)
            mujoco.mju_quat2Mat(rot, self.model.geom_quat[g])
            self.floor = (self.model.geom_pos[g].copy(), rot.reshape(3, 3)[:, 2])
        n = self.n_contacts
        self.pdim = 7+6+self.nu+self.nq+6*n+3*n*self.nu+n+4*n+4*n*self.nv+36+2*self.nu+6
        self._build_model()
        self._warm_controls = None
        self._advance = self.cfg.simulator.time_grid.action_substeps
        if not build_solver:
            return
        from retarget.support.acados import _load_acados, _set_acados_code_export_directory
        from retarget.support.acados_env import acados_cache_directory, build_cached_solver
        Model, Ocp, _ = _load_acados()
        m = Model(); m.name = f'trigrasp_contact_v5_x{self.nx}_u{self.nu}_c{n}_n{self.N}'
        m.x, m.u, m.p = self.x, self.u, self.p
        m.disc_dyn_expr = self.next_x
        m.cost_expr_ext_cost = m.cost_expr_ext_cost_0 = self.path_cost
        m.cost_expr_ext_cost_e = self.final_cost
        ocp = Ocp(); ocp.model = m; ocp.parameter_values = np.zeros(self.pdim)
        ocp.cost.cost_type = ocp.cost.cost_type_0 = ocp.cost.cost_type_e = 'EXTERNAL'
        ocp.constraints.x0 = np.zeros(self.nx)
        ocp.constraints.idxbu = np.arange(self.nu)
        ocp.constraints.lbu, ocp.constraints.ubu = -self.physical_limit, self.physical_limit
        opts = ocp.solver_options
        opts.N_horizon, opts.tf = self.N, self.N*self.h
        opts.integrator_type, opts.nlp_solver_type = 'DISCRETE', 'SQP_RTI'
        opts.qp_solver, opts.qp_solver_cond_N = 'PARTIAL_CONDENSING_HPIPM', min(5, self.N)
        opts.hessian_approx, opts.regularize_method = 'GAUSS_NEWTON', 'PROJECT'
        # The residual form supplies the positive-semidefinite contact curvature.
        m.cost_expr_ext_cost_custom_hess = self.cost_hessian
        m.cost_expr_ext_cost_custom_hess_0 = self.cost_hessian
        m.cost_expr_ext_cost_custom_hess_e = self.terminal_hessian
        opts.levenberg_marquardt, opts.qp_solver_iter_max, opts.print_level = .01, 100, 0
        opts.cost_scaling = np.ones(self.N+1)
        directory = Path(acados_cache_directory(m.name)); directory.mkdir(parents=True, exist_ok=True)
        _set_acados_code_export_directory(ocp, str(directory))
        self.solver = build_cached_solver(ocp, str(directory/(m.name+'.json')),
                                         str(directory/('libacados_ocp_solver_'+m.name+'.so')),
                                         prune_generated=False)

    def _object_hull(self):
        geoms = np.flatnonzero((self.model.geom_bodyid == self.index.object_body)
                              & (self.model.geom_type == mujoco.mjtGeom.mjGEOM_MESH))
        if not len(geoms):
            raise ValueError('SQP contact geometry requires an object mesh')
        g = int(geoms[0]); mesh = self.model.geom_dataid[g]
        start, count = self.model.mesh_vertadr[mesh], self.model.mesh_vertnum[mesh]
        rotation = np.empty(9); mujoco.mju_quat2Mat(rotation, self.model.geom_quat[g])
        vertices = (self.model.mesh_vert[start:start+count] @ rotation.reshape(3, 3).T
                    + self.model.geom_pos[g])
        hull = ConvexHull(vertices)
        self.support_vertices = vertices[hull.vertices].copy()
        return hull.equations

    def _build_model(self):
        n, nu, nv, h = self.n_contacts, self.nu, self.nv, self.h
        self.x = x = cs.MX.sym('state', self.nx)
        self.u = u = cs.MX.sym('servo_increment', nu)
        self.p = p = cs.MX.sym('measurements_and_targets', self.pdim)
        c = 0
        def take(size):
            nonlocal c
            out = p[c:c+size]; c += size
            return out
        goal, goal_velocity, posture, reference = take(7), take(6), take(nu), take(self.nq)
        anchors, tips = cs.reshape(take(3*n), 3, n), take(3*n)
        tip_jac = cs.reshape(take(3*n*nu), 3*n, nu)
        mask, phi = take(n), take(4*n)
        jac = cs.reshape(take(4*n*nv), 4*n, nv)
        object_inverse = cs.reshape(take(36), 6, 6)
        hand_mass, external, hand_external = take(nu), take(6), take(nu)
        assert c == self.pdim
        q, hand = x[3:7], x[7:self.nq]
        velocity, servo = x[self.nq:self.nq+nv], x[-nu:]
        next_servo = cs.fmin(cs.DM(self.upper), cs.fmax(cs.DM(self.lower), servo+u))
        response = h*cs.DM(self.kd)+h*h*cs.DM(self.kp)
        hand_inverse = 1/(hand_mass+response)
        free_hand = velocity[6:] + hand_inverse*(h*(cs.DM(self.kp)*(next_servo-hand)+hand_external)
                                               -response*velocity[6:])
        free = cs.vertcat(velocity[:6]+h*object_inverse@external, free_hand)
        inverse = cs.diagcat(object_inverse, cs.diag(hand_inverse))
        qr = reference[3:7]
        angular = 2*(qr[0]*q[1:]-q[0]*qr[1:]-cs.cross(qr[1:], q[1:]))
        displacement = cs.vertcat(x[:3]-reference[:3], angular, hand-reference[7:])
        # A friction row's tangent component must not transport the normal gap.
        transported = cs.vertcat(*[cs.repmat(cs.sum1(jac[4*i:4*i+4, :])@displacement/4, 4, 1)
                                    for i in range(n)])+phi
        row_mask = cs.vertcat(*[cs.repmat(cs.fmin(mask[i], 1.), 4, 1) for i in range(n)])
        effective_inverse = cs.sum2((jac@inverse)*jac)+1e-8
        speed = jac@free
        # Reduced complementarity-free friction-pyramid velocity response.
        # Effective row mobility prevents a mass-independent force injection.
        impulse = row_mask*_softplus(-speed-.2*transported/h, self.beta)/effective_inverse/4
        next_velocity = free+inverse@jac.T@impulse
        if self.floor is not None:
            floor_point, floor_normal = map(cs.DM, self.floor)
            local_normal = _quat_matrix(q).T@floor_normal
            support_point = cs.DM(self.support_vertices[0])
            support_gap = cs.dot(local_normal, support_point)+cs.dot(floor_normal, x[:3]-floor_point)
            for vertex in self.support_vertices[1:]:
                gap = cs.dot(local_normal, cs.DM(vertex))+cs.dot(floor_normal, x[:3]-floor_point)
                support_point = cs.if_else(gap < support_gap, cs.DM(vertex), support_point)
                support_gap = cs.fmin(gap, support_gap)
            floor_row = cs.vertcat(floor_normal, cs.cross(support_point, local_normal))
            mobility = cs.mtimes([floor_row.T, object_inverse, floor_row])+1e-8
            floor_impulse = cs.fmax(-(cs.dot(floor_row, next_velocity[:6])+support_gap/h)/mobility, 0.)
            next_velocity = cs.vertcat(next_velocity[:6]+object_inverse@floor_row*floor_impulse,
                                       next_velocity[6:])
        next_q = _quat_step(q, next_velocity[3:6], h)
        next_pose = cs.vertcat(x[:3]+h*next_velocity[:3], next_q, hand+h*next_velocity[6:])
        next_velocity = cs.vertcat(next_velocity[:3], _quat_matrix(next_q).T@_quat_matrix(q)@next_velocity[3:6],
                                   next_velocity[6:])
        self.next_x = cs.vertcat(next_pose, next_velocity, next_servo)
        predicted_tips = tips+tip_jac@(hand-reference[7:])
        target_tips = cs.vec(cs.repmat(x[:3], 1, n)+_quat_matrix(q)@anchors)
        self.tip_error = predicted_tips-target_tips
        weights = self.cfg.loss
        residuals = []
        marked_mm = self.cfg.contact.marked_query_dist_mm
        temperature = self.cfg.contact.marked_query_temp_mm
        for i in range(n):
            error = self.tip_error[3*i:3*i+3]
            distance_mm = 1000*cs.sqrt(cs.sumsqr(error)+1e-12)
            gate = 1/(1+cs.exp(-(distance_mm-marked_mm)/max(temperature, 1e-6)))
            residuals.append(cs.sqrt(weights.marked_query_pos*mask[i]*gate+1e-20)*error)
        contact_residual = cs.vertcat(*residuals)
        qref = goal[3:7]
        rot_error = 2*(qref[0]*q[1:]-q[0]*qref[1:]-cs.cross(qref[1:], q[1:]))
        object_residual = cs.vertcat(weights.obj_pos*(x[:3]-goal[:3]), weights.obj_rot*rot_error,
            weights.obj_vel*(velocity[:3]-goal_velocity[:3]),
            weights.obj_ang_vel*(_quat_matrix(q)@velocity[3:6]-goal_velocity[3:6]))
        hand_weights = cs.DM(np.r_[np.full(3, weights.action_ref_base),
                                    np.full(3, weights.action_ref_base_rot),
                                    np.full(nu-6, weights.action_ref_finger)])
        # Scale the mm^2 objective by 1e-6 as on the AdamW evaluator; geometric
        # weights and reference-pose weights retain their relative units.
        residual = cs.vertcat(contact_residual, cs.sqrt(weights.reference_pose_scale)*object_residual,
                              hand_weights*(hand-posture), .01*u/self.physical_limit)
        final_residual = np.sqrt(weights.terminal)*cs.vertcat(contact_residual,
                                cs.sqrt(weights.reference_pose_scale)*object_residual)
        self.path_cost, self.final_cost = cs.sumsqr(residual), cs.sumsqr(final_residual)
        jac_r = cs.jacobian(residual, cs.vertcat(u, x))
        jac_e = cs.jacobian(final_residual, x)
        self.cost_hessian = 2*jac_r.T@jac_r
        self.terminal_hessian = 2*jac_e.T@jac_e
        self.prediction_fn = cs.Function('contact_servo_step', [x, u, p], [self.next_x])
        self.tip_error_fn = cs.Function('contact_tip_error', [x, p], [self.tip_error])
        self.tip_jacobian_fn = cs.Function('contact_tip_derivative', [x, p], [cs.jacobian(self.tip_error, x)])
        self.path_cost_fn = cs.Function('contact_stage_cost', [x, u, p], [self.path_cost])
        self.final_cost_fn = cs.Function('contact_terminal_cost', [x, p], [self.final_cost])

    def _geometry(self, qpos, targets, contact_points, stage=0):
        m, data = self.model, self.data
        slot_stage = max(stage-1, 0) if stage < len(targets.dense_actions) else len(targets.dense_actions)-1
        body = np.asarray(targets.query_body[slot_stage], int)
        local = np.asarray(targets.query_local[slot_stage], float)
        mask = np.asarray(targets.contact_mask[slot_stage]*targets.contact_weight[slot_stage], float)
        anchors = np.asarray(contact_points, float)
        rotation = data.xmat[self.index.object_body].reshape(3, 3)
        position = data.xpos[self.index.object_body]
        tips, tip_jac, gaps, jac = [], [], [], []
        for i in range(self.n_contacts):
            b = int(body[i]); point = data.xpos[b]+data.xmat[b].reshape(3, 3)@local[i]
            jp = np.zeros((3, m.nv)); jo = np.zeros((3, m.nv))
            mujoco.mj_jac(m, data, jp, None, point, b)
            anchor_world = position+rotation@anchors[i]
            mujoco.mj_jac(m, data, jo, None, anchor_world, self.index.object_body)
            relative = np.c_[(jp-jo)[:, self.object_dofs], jp[:, self.hand_dofs]]
            plane = self.hull_planes[np.argmax(self.hull_planes[:, :3]@anchors[i]+self.hull_planes[:, 3])]
            normal = rotation@plane[:3]
            axis = np.eye(3)[np.argmin(np.abs(normal))]
            tangent = np.cross(normal, axis); tangent /= np.linalg.norm(tangent)
            other = np.cross(normal, tangent)
            friction = float(m.geom_friction[np.flatnonzero(m.geom_bodyid == self.index.object_body)[-1], 0])
            directions = np.asarray([normal+friction*tangent, normal+friction*other,
                                     normal-friction*tangent, normal-friction*other])
            tips.append(point); tip_jac.append(jp[:, self.hand_dofs])
            gaps.extend([normal@(point-anchor_world)]*4); jac.append(directions@relative)
        return dict(anchors=anchors, tips=np.asarray(tips), tip_jac=np.vstack(tip_jac),
                    mask=mask, phi=np.asarray(gaps), jac=np.vstack(jac), body=body,
                    local=local)

    def parameters(self, qpos, qvel, targets, contact_points, stage=0):
        data = self.data
        data.qpos[:], data.qvel[:] = qpos, qvel
        mujoco.mj_forward(self.model, data)
        geo = self._geometry(qpos, targets, contact_points, stage)
        full_mass = np.empty((self.model.nv, self.model.nv))
        try:
            mujoco.mj_fullM(self.model, data, full_mass)
        except TypeError:
            mujoco.mj_fullM(self.model, full_mass, data.qM)
        inv_mass = np.linalg.inv(full_mass[np.ix_(self.object_dofs, self.object_dofs)])
        external = (data.qfrc_passive-data.qfrc_bias)[self.object_dofs]
        reference = np.r_[qpos[self.index.object_qpos:self.index.object_qpos+7], qpos[self.index.hand_qpos]]
        k = min(stage, len(targets.dense_actions)-1)
        reference_velocity=getattr(targets,'object_velocity_world',None)
        target_velocity=np.zeros(6) if reference_velocity is None else reference_velocity[k]
        values = np.r_[targets.object_qpos[k], target_velocity, targets.dense_actions[k], reference,
                       geo['anchors'].ravel(), geo['tips'].ravel(), geo['tip_jac'].ravel(order='F'),
                       geo['mask'], geo['phi'], geo['jac'].ravel(order='F'),
                       inv_mass.ravel(order='F'), np.diag(full_mass)[self.hand_dofs], external,
                       (data.qfrc_passive-data.qfrc_bias)[self.hand_dofs]]
        assert values.size == self.pdim
        return values, geo

    def update(self, variables, targets, *, qpos, qvel, previous_action=None, contact_points=None):
        started = perf_counter()
        previous = np.asarray(qpos[self.index.hand_qpos] if previous_action is None else previous_action, float)
        state = np.r_[qpos[self.index.object_qpos:self.index.object_qpos+7], qpos[self.index.hand_qpos],
                      qvel[self.object_dofs], qvel[self.hand_dofs], previous]
        parameters, geometries = [], []
        for k in range(self.N+1):
            params, geo = self.parameters(qpos, qvel, targets, contact_points[k], k)
            parameters.append(params); geometries.append(geo)
        prepared = perf_counter()
        self.solver.set(0, 'lbx', state); self.solver.set(0, 'ubx', state)
        current = state.copy()
        if self._warm_controls is None:
            # Feasible servo-reference initialization using the same rate limits
            # as execution; this is a symbolic-model prediction, not Warp rollout.
            goal_commands = np.asarray(targets.dense_actions[:self.N], float)
            controls = np.empty_like(goal_commands); servo = previous.copy()
            for k, goal in enumerate(goal_commands):
                controls[k] = np.clip(goal-servo, -self.physical_limit, self.physical_limit)
                servo = np.clip(servo+controls[k], self.lower, self.upper)
        else:
            advance = min(self._advance, self.N)
            controls = np.vstack((self._warm_controls[advance:], np.repeat(self._warm_controls[-1:], advance, axis=0)))
        for k in range(self.N+1):
            self.solver.set(k, 'p', parameters[k]); self.solver.set(k, 'x', current)
            if k < self.N:
                self.solver.set(k, 'u', controls[k])
                current = np.asarray(self.prediction_fn(current, controls[k], parameters[k])).ravel()
        init_done = perf_counter()
        status = int(self.solver.solve())
        if status == 0:
            controls = np.vstack([np.asarray(self.solver.get(k, 'u')).ravel() for k in range(self.N)])
        else:
            controls = np.zeros((self.N, self.nu))
        controls = np.clip(controls, -self.physical_limit, self.physical_limit)
        commands = servo_commands(previous, controls, self.lower, self.upper)
        self._warm_controls = controls.copy()
        cost = float(self.path_cost_fn(state, controls[0], parameters[0]))
        active = geometries[0]['mask'] > 0
        body_names = [self.model.body(int(b)).name for b in geometries[0]['body'][active]]
        self.last_geometry, self.last_parameters, self.last_state = geometries, parameters, state
        return np.zeros_like(variables), int(status == 0), dict(
            method='trigrasp_comfree_symbolic_sqp_rti', status=status,
            forward_comfree_rollout=False, analytic_dynamics=True, analytic_cost=True,
            contact_model='reduced_comfree_friction_pyramid', full_adamw_warp_objective=False,
            controls=commands, controls_are_absolute=True, cost=cost,
            physical_action_limits=self.physical_limit.tolist(),
            geometry_seconds=prepared-started, symbolic_prediction_seconds=init_done-prepared,
            solver_wall_seconds=float(self.solver.get_stats('time_tot')),
            total_seconds=perf_counter()-started, active_contact_bodies=body_names,
            contact_capacity=self.n_contacts, horizon_seconds=self.N*self.h)
