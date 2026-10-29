"""MJWarp humanoid rollouts and central-difference action gradients.

Prediction and execution share the same MuJoCo Warp step. Contact geometry is
observed in a separate Data object, so observation cannot alter simulation.
"""
from types import SimpleNamespace

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

from forceaware.control import action_knot_interpolation_map
from retarget.fa.adamw import AdamW
from retarget.humanoid_terms import (
    Model, Targets, Stage, TERMS, SPIDER_TERMS, build_controls,
    build_controls_from_dense_base, compute_com, whole_body_loss, add_losses,
    diagnose_terms, pack_snapshot,
)


@wp.kernel(enable_backward=False)
def contact_distances(dist: wp.array(dtype=float), world: wp.array(dtype=int),
                      total: wp.array(dtype=int), counts: wp.array(dtype=int),
                      output: wp.array2d(dtype=float), active: wp.array2d(dtype=int)):
    i = wp.tid()
    if i < total[0]:
        w = world[i]
        if w >= 0 and w < output.shape[0]:
            slot = wp.atomic_add(counts, w, 1)
            if slot < output.shape[1]:
                output[w, slot] = dist[i]
                active[w, slot] = 1


def data_arrays(value):
    """Include newer solver/island buffers absent from older state copiers."""
    for field in vars(value).values():
        if isinstance(field, wp.array):
            if field.size:
                yield field
        elif hasattr(field, '__dataclass_fields__'):
            yield from data_arrays(field)


class MJWarpHumanoid:
    def __init__(self, problem):
        self.problem = problem
        self.settings = s = problem.settings
        wp.init()
        self.device = wp.get_device(s.device)
        if not self.device.is_cuda:
            raise ValueError('Humanoid MJWarp optimization requires CUDA')
        if problem.model.na:
            raise ValueError('Humanoid actuator activation states are not supported')
        self.torch_stream = torch.cuda.ExternalStream(wp.get_stream(self.device).cuda_stream,
                                                     device=str(self.device))
        self.cpu_model = problem.model
        self.worlds, self.knots = s.multistarts, s.horizon
        self.knot_substeps = s.knot_stride*s.substeps
        self.dense_steps = self.knots*self.knot_substeps
        self.maps = action_knot_interpolation_map(self.knots, self.knot_substeps)
        self.batch = max(self.worlds, s.mjwarp_fd_batch)
        self.graphs = {}
        self.forward_calls = self.backward_calls = self.simulated_trajectories = 0
        self.dense_control_base = None
        self._cached = self._snapshot_cache = self._value = None
        self._external_rollout = False
        self.dirty = True
        self.contact_qp = None
        if s.cost_mode == 'legacy':
            from retarget.humanoid_qp import ContactQP
            self.contact_qp = ContactQP(self.device)
        alloc = lambda shape, dtype=float: wp.zeros(shape, dtype=dtype, device=self.device)
        w, n, p = self.worlds, self.dense_steps+1, problem
        self.variables = alloc((w, self.knots*p.model.nu))
        self.gradient = alloc(self.variables.shape)
        self.q0, self.v0 = alloc((w, p.model.nq)), alloc((w, p.model.nv))
        self.warmstart0 = alloc((w, p.model.nv))
        self.initial_time = alloc(w)
        self.previous, self.previous_controls = alloc(p.model.nu), alloc((w, p.model.nu))
        self.dense_control_boundary = alloc(p.model.nu)
        self.lower = wp.array(p.lower.astype(np.float32), device=self.device)
        self.upper = wp.array(p.upper.astype(np.float32), device=self.device)
        self.knot_reference = alloc((self.knots, p.model.nu))
        self.controls = [alloc((w, p.model.nu)) for _ in self.maps]
        self.total, self.ones, self.no_failure = alloc(w), wp.ones(w, device=self.device), alloc(w, int)
        self.model = m = Model()
        self._configure_geometry()
        self.targets = self._targets(alloc)
        self.term_names = SPIDER_TERMS if m.spider_cost else TERMS
        self.stages = self._make_stages(w, self.targets, self.controls, alloc)
        self.warmstart_history = [alloc((w, p.model.nv)) for _ in range(n)]
        # A small trajectory record preserves the solver/export interface;
        # these are stored MJWarp values, with no alternate dynamics runtime.
        self.steps = [SimpleNamespace(
            inputs=SimpleNamespace(time=self.initial_time),
            recorded=SimpleNamespace(result=SimpleNamespace(
                qpos=self.stages[i+1].job.qpos, qvel=self.stages[i+1].job.qvel)))
            for i in range(self.dense_steps)]
        self.loss = SimpleNamespace(immediate=self.stages[:-1], terminal=self.stages[-1])
        self._snapshot_buffer = alloc((n, w, 51))
        with wp.ScopedDevice(self.device):
            self.mj_model = mjwarp.put_model(p.model)
            initial = p.initial_state(0)
            self.data = mjwarp.put_data(p.model, initial, nworld=self.batch,
                                       nconmax=s.mjwarp_contact_capacity, njmax=s.mjwarp_constraint_capacity)
            self.observer = mjwarp.make_data(p.model, nworld=self.batch,
                                            nconmax=1, njmax=1)
            arrays = {a.ptr: a for a in data_arrays(self.data)}
            self.initial_arrays = [(a, wp.clone(a)) for a in arrays.values()]
            mjwarp.step(self.mj_model, self.data)
            with wp.ScopedCapture(device=self.device) as capture:
                mjwarp.step(self.mj_model, self.data)
            self.step_graph = capture.graph
        self._probe_x = alloc((self.batch, self.variables.shape[1]))
        self._probe_controls = [alloc((self.batch, p.model.nu)) for _ in self.maps]
        self._probe_target = self._batch_targets(alloc)
        self._probe_stages = self._make_stages(self.batch, self._probe_target,
                                              self._probe_controls, alloc)
        self._probe_warmstarts = [alloc((self.batch, p.model.nv)) for _ in range(n)]
        self._probe_total = alloc(self.batch)
        self._probe_failure = alloc(self.batch, int)
        self._contact_counts = alloc(self.batch, int)
        self.adamw = AdamW(learning_rate=s.adamw_lr, weight_decay=s.adamw_weight_decay,
                           persist=s.adamw_persist, device=self.device)
        from retarget.humanoid_cache import RolloutCache
        self.rollout_cache = RolloutCache(self) if s.trajectory_cache else None

    def _configure_geometry(self):
        problem, m, s = self.problem, self.model, self.settings
        m.bodies = wp.array(problem.bodies, dtype=int, device=self.device)
        ids = np.array([b for b in range(problem.model.nbody) if problem._robot_body(b)])
        m.robot_bodies = wp.array(ids, dtype=int, device=self.device)
        m.mass = wp.array(problem.model.body_mass, dtype=float, device=self.device)
        m.ipos = wp.array(problem.model.body_ipos, dtype=wp.vec3, device=self.device)
        m.robot_mass = float(problem.model.body_mass[ids].sum())
        m.joint_qpos = wp.array(problem.joint_qpos, dtype=int, device=self.device)
        m.robot_dofs = wp.array(problem.robot_dofs, dtype=int, device=self.device)
        self.role_bodies = np.array([problem.model.body(f'{side}_wrist_yaw_link').id for side in ('left', 'right')]
                                    +[problem.model.body(f'{side}_ankle_roll_link').id for side in ('left', 'right')])
        m.role_bodies = wp.array(self.role_bodies, dtype=int, device=self.device)
        centers, radii = [], []
        for ids in problem.foot_geoms:
            cloud, sizes = [], []
            for geom in ids:
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, problem.model.geom_quat[geom])
                offsets = [np.zeros(3)]
                if problem.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    offsets = [sign*problem.model.geom_size[geom, 1]*rotation.reshape(3, 3)[:, 2] for sign in (-1, 1)]
                cloud.extend([problem.model.geom_pos[geom]+offset for offset in offsets])
                sizes.extend([problem.model.geom_size[geom, 0]]*len(offsets))
            centers.append(cloud); radii.append(sizes)
        count = max(map(len, centers))
        cloud, sizes = np.zeros((2, count, 3)), np.zeros((2, count))
        for foot in range(2):
            cloud[foot, :len(centers[foot])] = centers[foot]
            sizes[foot, :len(radii[foot])] = radii[foot]
        m.foot_centers = wp.array(cloud, dtype=wp.vec3, device=self.device)
        m.foot_radii = wp.array(sizes, dtype=float, device=self.device)
        m.foot_counts = wp.array([len(c) for c in centers], dtype=int, device=self.device)
        m.hand_centers = wp.array(problem.model.geom_pos[problem.hand_geoms] if problem.surface else np.zeros((2, 3)), dtype=wp.vec3, device=self.device)
        m.hand_radii = wp.array(problem.model.geom_size[problem.hand_geoms, 0] if problem.surface else np.zeros(2), dtype=float, device=self.device)
        m.weights = wp.array([getattr(problem.weights, k) for k in TERMS[:14]], dtype=float, device=self.device)
        m.spider_cost = int(s.cost_mode == 'spider')
        m.root, m.torso = problem.root, problem.torso
        m.object_body = int(problem.model.geom_bodyid[problem.object_geom]) if problem.surface else -1
        m.object_qpos = int(problem.model.jnt_qposadr[problem.model.body_jntadr[m.object_body]]) if m.object_body >= 0 else -1
        m.object_dof = int(problem.model.jnt_dofadr[problem.model.body_jntadr[m.object_body]]) if m.object_body >= 0 else -1
        m.realization_weight, m.wrench_robust_scale = s.contact_realization_weight, s.wrench_robust_scale
        m.object_mass = float(problem.model.body_mass[m.object_body]) if m.object_body >= 0 else 0.
        m.box_object = int(bool(problem.surface) and problem.model.geom_type[problem.object_geom] == mujoco.mjtGeom.mjGEOM_BOX)
        if problem.surface:
            geom = problem.object_geom
            rotation = np.empty(9); mujoco.mju_quat2Mat(rotation, problem.model.geom_quat[geom])
            m.object_geom_position = wp.vec3(*problem.model.geom_pos[geom])
            m.object_geom_rotation = wp.mat33(*rotation)
            m.object_halfsize = wp.vec3(*problem.model.geom_size[geom])
        m.scale, m.dt, m.ground = s.contact_scale, problem.ref_dt/s.substeps, problem.ground_height
        for name in ('balance_xy_scale', 'balance_z_scale', 'balance_velocity_scale', 'balance_height_margin', 'balance_height_scale', 'upright_scale'):
            setattr(m, name, float(getattr(s, name)))

    def _make_stages(self, worlds, targets, controls, alloc):
        stages = []
        p, s = self.problem, self.settings
        for k in range(self.dense_steps+1):
            job = Stage(); job.model, job.target, job.index = self.model, targets, k
            job.normalizer = (s.terminal_weight if k == self.dense_steps else 1.)/self.dense_steps
            if self.model.spider_cost and k == 0:
                job.normalizer = 0.
            job.qpos, job.qvel = alloc((worlds, p.model.nq)), alloc((worlds, p.model.nv))
            job.position, job.rotation = alloc((worlds, p.model.nbody), wp.vec3), alloc((worlds, p.model.nbody), wp.mat33)
            job.distance, job.active = alloc((worlds, s.mjwarp_contact_capacity)), alloc((worlds, s.mjwarp_contact_capacity), int)
            job.constraint_force = alloc((worlds, p.model.nv))
            job.physical_contact = int(k > 0)
            job.ctrl = controls[min(k, self.dense_steps-1)]
            prior = stages[-1].job if stages else job
            job.previous_position, job.previous_rotation = prior.position, prior.rotation
            job.previous_ctrl = controls[k-1] if k else alloc((worlds, p.model.nu))
            job.com = alloc(worlds, wp.vec3)
            job.previous_com = prior.com if stages else job.com
            job.terms, job.loss = alloc((worlds, len(TERMS))), alloc(worlds)
            job.query_points, job.balance_parts = alloc((worlds, 4), wp.vec3), alloc((worlds, 5))
            stages.append(SimpleNamespace(job=job))
        return stages

    def _batch_targets(self, alloc):
        target = Targets()
        for name in Targets.vars:
            setattr(target, name, getattr(self.targets, name))
        for name in ('z', 'dual', 'forces', 'floor_point', 'floor_force', 'floor_torque'):
            source = getattr(self.targets, name)
            shape = (source.shape[0], self.batch)+source.shape[2:]
            setattr(target, name, alloc(shape, source.dtype))
        return target

    def _record(self, k):
        job, d = self._probe_stages[k].job, self.data
        wp.copy(self.observer.qpos, d.qpos)
        mjwarp.kinematics(self.mj_model, self.observer)
        for target, source in ((job.qpos, d.qpos), (job.qvel, d.qvel),
                               (job.position, self.observer.xpos), (job.rotation, self.observer.xmat),
                               (self._probe_warmstarts[k], d.qacc_warmstart)):
            wp.copy(target, source)
        job.active.zero_(); job.distance.zero_()
        if k:
            wp.copy(job.constraint_force, d.qfrc_constraint)
            self._contact_counts.zero_()
            wp.launch(contact_distances, d.contact.dist.shape[0], inputs=[
                d.contact.dist, d.contact.worldid, d.nacon, self._contact_counts,
                job.distance, job.active], device=self.device)
        else:
            job.constraint_force.zero_()
        # Identical physical inputs must share one nominal trajectory. ADMM
        # targets are deliberately excluded: they change scoring, not physics.
        for array in (job.qpos, job.qvel, job.position, job.rotation, job.distance,
                      job.active, job.constraint_force, self._probe_warmstarts[k]):
            tensor = wp.to_torch(array)
            tensor.copy_(tensor[self._canonical])
        wp.launch(compute_com, self.batch, inputs=[job], device=self.device)
        wp.launch(whole_body_loss, self.batch, inputs=[job], device=self.device)
        wp.launch(add_losses, self.batch, inputs=[job, self._probe_failure, self._probe_total, int(k == self.dense_steps)], device=self.device)

    def _rollout(self, x, ids, export=False):
        count = len(x)
        padded = x[:1].expand(self.batch, -1).clone(); padded[:count] = x
        source = torch.zeros(self.batch, dtype=torch.long, device=x.device); source[:count] = ids.long()
        keys = torch.cat([padded]+[wp.to_torch(a)[source].double().reshape(self.batch, -1)
                                  for a in (self.q0, self.v0, self.warmstart0, self.initial_time,
                                            self.previous_controls)], dim=1)
        _, inverse = torch.unique(keys, dim=0, return_inverse=True)
        first = torch.full((self.batch,), self.batch, dtype=torch.long, device=x.device)
        first.scatter_reduce_(0, inverse, torch.arange(self.batch, device=x.device), reduce='amin')
        self._canonical = first[inverse]
        wp.to_torch(self._probe_x).copy_(padded)
        for name in ('z', 'dual', 'forces', 'floor_point', 'floor_force', 'floor_torque'):
            wp.to_torch(getattr(self._probe_target, name)).copy_(wp.to_torch(getattr(self.targets, name))[:, source])
        for current, saved in self.initial_arrays:
            wp.copy(current, saved)
        for target, initial in ((self.data.qpos, self.q0), (self.data.qvel, self.v0),
                                 (self.data.qacc_warmstart, self.warmstart0),
                                 (self.data.ctrl, self.previous_controls), (self.data.time, self.initial_time)):
            wp.to_torch(target).copy_(wp.to_torch(initial)[source])
        wp.to_torch(self._probe_stages[0].job.previous_ctrl).copy_(wp.to_torch(self.previous_controls)[source])
        for i, (ctrl, (left, right, fraction)) in enumerate(zip(self._probe_controls, self.maps)):
            if self.dense_control_base is None:
                wp.launch(build_controls, (self.batch, self.problem.model.nu), inputs=[
                    self._probe_x, self.knot_reference, self.settings.control_scale,
                    self.previous, self.lower, self.upper, left, right, fraction, ctrl], device=self.device)
            else:
                wp.launch(build_controls_from_dense_base, (self.batch, self.problem.model.nu), inputs=[
                    self._probe_x, self.dense_control_base, self.dense_control_boundary,
                    self.settings.control_scale, self.lower, self.upper, left, right, fraction, i, ctrl], device=self.device)
        self._probe_total.zero_()
        self._record(0)
        for i, ctrl in enumerate(self._probe_controls):
            wp.copy(self.data.ctrl, ctrl)
            wp.capture_launch(self.step_graph)
            self._record(i+1)
        value = wp.to_torch(self._probe_total)[:count].double().clone()
        if export:
            names = ('qpos', 'qvel', 'position', 'rotation', 'distance', 'active', 'constraint_force', 'com')
            for target, probe, warm, probe_warm in zip(self.stages, self._probe_stages, self.warmstart_history, self._probe_warmstarts):
                for name in names:
                    wp.to_torch(getattr(target.job, name)).copy_(wp.to_torch(getattr(probe.job, name))[:self.worlds])
                wp.to_torch(warm).copy_(wp.to_torch(probe_warm)[:self.worlds])
            for target, probe in zip(self.controls, self._probe_controls):
                wp.to_torch(target).copy_(wp.to_torch(probe)[:self.worlds])
            wp.to_torch(self.stages[0].job.previous_ctrl).copy_(wp.to_torch(self.previous_controls))
            wp.to_torch(self.variables).copy_(x)
            self._snapshot_cache = None
            self._objective()
        self.forward_calls += 1
        self.simulated_trajectories += count
        return value

    def _objective(self):
        self.total.zero_()
        for k, stage in enumerate(self.stages):
            wp.launch(compute_com, self.worlds, inputs=[stage.job], device=self.device)
            wp.launch(whole_body_loss, self.worlds, inputs=[stage.job], device=self.device)
            wp.launch(add_losses, self.worlds, inputs=[stage.job, self.no_failure, self.total, int(k == self.dense_steps)], device=self.device)

    def evaluate(self, variables, gradient=False):
        on_device = isinstance(variables, torch.Tensor)
        caller = torch.cuda.current_stream(str(self.device))
        self.torch_stream.wait_stream(caller)
        with torch.no_grad(), torch.cuda.stream(self.torch_stream):
            x = torch.as_tensor(variables, dtype=torch.float64, device=str(self.device))
            same = self._cached is not None and bool((x == self._cached).all())
            restored = False
            if not same and self.rollout_cache is not None:
                restored = self.rollout_cache.restore(x)
            if not same and not restored:
                self._rollout(x, torch.arange(self.worlds, device=x.device), export=True)
                if self.rollout_cache is not None:
                    self.rollout_cache.record(x)
            self._objective()
            value = wp.to_torch(self.total).double().clone()
            g = None
            if gradient:
                d, w, h = x.shape[1], self.worlds, self.settings.mjwarp_fd_step
                scores = torch.empty(2*w*d, dtype=torch.float64, device=x.device)
                for start in range(0, len(scores), self.batch):
                    indices = torch.arange(start, min(start+self.batch, len(scores)), device=x.device)
                    worlds = indices % w
                    dimension = indices // (2*w)
                    sign = torch.where((indices//w) % 2 == 0, 1., -1.)
                    probe = x[worlds].clone()
                    probe[torch.arange(len(indices), device=x.device), dimension] += h*sign
                    scores[start:start+len(indices)] = self._rollout(probe, worlds)
                pairs = scores.reshape(d, 2, w)
                g = ((pairs[:, 0]-pairs[:, 1])/(2*h)).T.contiguous()
                wp.to_torch(self.gradient).copy_(g)
                self.backward_calls += 1
            valid = torch.isfinite(value)
            if g is not None:
                valid &= torch.isfinite(g).all(dim=1)
                g[~valid] = 0.
            value[~valid] = torch.inf
            self._cached, self._value = x.clone(), value.clone()
            self.dirty = False
        caller.wait_stream(self.torch_stream)
        if on_device:
            return value, g
        return value.cpu().numpy(), None if g is None else g.cpu().numpy()

    def accept_execution(self, selected):
        self.execution_warmstart = self.warmstart_history[self.settings.substeps].numpy()[selected].copy()

    def observable_arrays(self):
        arrays = []
        for stage in self.stages:
            arrays.extend(getattr(stage.job, name) for name in
                          ('qpos', 'qvel', 'position', 'rotation', 'ctrl', 'distance', 'active', 'constraint_force'))
        arrays.extend(self.warmstart_history)
        return list({(a.ptr, a.shape, str(a.dtype)): a for a in arrays}.values())

    def _targets(self,alloc):
        p,n = self.problem,self.dense_steps+1
        alloc = lambda shape,dtype=float: wp.zeros(shape,dtype=dtype,device=self.device)
        t = Targets()
        t.qpos,t.qvel,t.ctrl = alloc((n,p.model.nq)),alloc((n,p.model.nv)),alloc((n,p.model.nu))
        t.position,t.rotation = alloc((n,p.model.nbody),wp.vec3),alloc((n,p.model.nbody),wp.mat33)
        t.com,t.wrench = alloc(n,wp.vec3),alloc((n,2),wp.vec3)
        t.object_wrench=alloc((n,2),wp.vec3)
        t.hand_position=alloc((n,2),wp.vec3)
        t.mask,t.local = alloc((n,4)),alloc((n,4),wp.vec3)
        t.anchor,t.stance = alloc((n,4),wp.vec3),alloc((n,4),wp.vec3)
        t.z,t.dual,t.forces = alloc((n,self.worlds,4),wp.vec3),alloc((n,self.worlds,4),wp.vec3),alloc((n,self.worlds,4),wp.vec3)
        t.floor_point,t.floor_force = alloc((n,self.worlds),wp.vec3),alloc((n,self.worlds),wp.vec3)
        t.floor_torque = alloc((n,self.worlds),wp.vec3)
        t.penalty = wp.array([self.settings.rho,self.settings.anchor_weight],device=self.device,dtype=float)
        return t


    def prepare(self,state,first,previous):
        p,s,n = self.problem,self.settings,self.dense_steps+1
        self.frame_times = first-1+np.arange(n)/s.substeps
        self.reference_indices = np.clip(np.rint(self.frame_times).astype(int),0,len(p.qref)-1)
        from scipy.spatial.transform import Rotation,Slerp
        frames = np.arange(len(p.qref))
        query = np.clip(self.frame_times,0,frames[-1])
        q = np.stack([np.interp(query,frames,p.qref[:,j]) for j in range(p.model.nq)],axis=1)
        for joint in np.flatnonzero(p.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
            adr = p.model.jnt_qposadr[joint]+3
            quats = p.qref[:,adr:adr+4][:,[1,2,3,0]]
            q[:,adr:adr+4] = Slerp(frames,Rotation.from_quat(quats))(query).as_quat()[:,[3,0,1,2]]
        v = np.stack([np.interp(query,frames,p.tracking_vref[:,j]) for j in range(p.model.nv)],axis=1)
        a = np.stack([np.interp(query,frames,p.aref[:,j]) for j in range(p.model.nu)],axis=1)
        positions,rotations,coms,locals_,wrenches = [],[],[],[],[]
        d = mujoco.MjData(p.model)
        for k in range(n):
            d.qpos[:],d.qvel[:] = q[k],v[k]
            mujoco.mj_forward(p.model,d)
            positions.append(d.xpos.copy());rotations.append(d.xmat.reshape(-1,3,3).copy())
            coms.append(d.subtree_com[p.root].copy())
            # Active hand/foot queries are computed from each predicted pose
            # on CUDA. Only inactive hand roles use this zero wrist offset.
            locals_.append(np.zeros((4,3)))
            wrenches.append(p.reference_wrench[self.reference_indices[k]])
        for name,value in [('qpos',q),('qvel',v),('ctrl',a),('position',positions),('rotation',rotations),('com',coms),('local',locals_),('wrench',wrenches),('mask',p.mask[self.reference_indices].astype(float))]:
            getattr(self.targets,name).assign(np.asarray(value,np.float32))
        self.positions_ref,self.rotations_ref,self.com_ref = np.array(positions),np.array(rotations),np.array(coms)
        self.wrench_ref = np.array(wrenches)
        self.object_wrench_ref=p.reference_object_wrench[self.reference_indices]
        self.targets.object_wrench.assign(self.object_wrench_ref.astype(np.float32))
        self.reference_qvel=v
        anchors = p.anchors[self.reference_indices].copy()
        if p.surface:
            geom=p.object_geom
            rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,p.model.geom_quat[geom])
            anchors[:,:2] = anchors[:,:2]@rotation.reshape(3,3).T+p.model.geom_pos[geom]
        self.anchors=anchors
        self.targets.anchor.assign(anchors.astype(np.float32))
        hand_position=np.zeros((n,2,3))
        if p.surface:
            # The released hand spheres can be several cm off the box. Use
            # the SAME reference surface anchor as consensus, plus the real
            # sphere radius, so the two task targets are geometrically compatible.
            geom=p.object_geom;body=p.model.geom_bodyid[geom]
            normal=p.surface.normal(p.anchors[self.reference_indices,:2],p.anchors[self.reference_indices,:2])
            local=anchors[:,:2]+p.model.geom_size[p.hand_geoms,:1]*(normal@rotation.reshape(3,3).T)
            hand_position=np.einsum('nij,nsj->nsi',self.rotations_ref[:,body],local)+self.positions_ref[:,body,None]
        self.hand_position_ref=hand_position
        self.targets.hand_position.assign(hand_position.astype(np.float32))
        self.targets.stance.assign(p.stance_anchors[self.reference_indices].astype(np.float32))
        knot_ids = np.arange(1,self.knots+1)*self.knot_substeps
        self.knot_reference.assign(a[knot_ids].astype(np.float32))
        self.previous.assign(np.asarray(previous,np.float32))
        self.previous_controls.assign(np.broadcast_to(previous,self.previous_controls.shape).astype(np.float32))
        self.q0.assign(np.broadcast_to(state.qpos,self.q0.shape).astype(np.float32))
        self.v0.assign(np.broadcast_to(state.qvel,self.v0.shape).astype(np.float32))
        self.initial_time.fill_(float(state.time))
        self.warmstart0.assign(np.broadcast_to(state.qacc_warmstart,self.warmstart0.shape).astype(np.float32))
        self._cached=None;self._snapshot_cache=None;self.dirty=True
        if self.rollout_cache is not None:self.rollout_cache.clear()


    def use_dense_control_base(self,controls=None,previous_delta=None):
        """Fixed dense nominal plus knot increments for tests or feedforward.

        The standard reference/previous-control parametrization remains the
        default. Its interpolation VJP also applies to these knot increments.
        """
        old=self.dense_control_base is not None
        if controls is None:self.dense_control_base=None
        else:
            controls=np.asarray(controls,np.float32)
            if controls.shape!=(self.dense_steps,self.problem.model.nu) or not np.isfinite(controls).all():
                raise ValueError('Dense control base must be finite and have shape (dense_steps, nu)')
            if self.dense_control_base is None:
                self.dense_control_base=wp.zeros(controls.shape,device=self.device)
            self.dense_control_base.assign(controls)
        if previous_delta is None:self.dense_control_boundary.zero_()
        else:
            previous_delta=np.asarray(previous_delta,np.float32)
            if previous_delta.shape!=(self.problem.model.nu,) or not np.isfinite(previous_delta).all():
                raise ValueError('Previous control correction must be a finite actuator vector')
            self.dense_control_boundary.assign(previous_delta/self.settings.control_scale)
        if old!=(self.dense_control_base is not None):self.graphs.clear()
        self._cached=None;self._snapshot_cache=None;self.dirty=True
        if self.rollout_cache is not None:self.rollout_cache.clear()


    def diagnose(self):
        # Diagnostics are not part of AdamW's repeated value/gradient callback.
        key='diagnostics'
        if key not in self.graphs:
            def launch():
                for stage in self.stages:
                    wp.launch(diagnose_terms,self.worlds,inputs=[stage.job],device=self.device)
            launch()
            with wp.ScopedCapture(device=self.device) as capture:launch()
            self.graphs[key]=capture.graph
        wp.capture_launch(self.graphs[key])


    def snapshots(self):
        """Transfer only the contact block's geometry, once per physical rollout."""
        if self._snapshot_cache is not None:return self._snapshot_cache
        key='snapshot'
        if key not in self.graphs:
            def launch():
                for stage in self.stages:
                    wp.launch(pack_snapshot,self.worlds,inputs=[stage.job,self._snapshot_buffer],device=self.device)
            launch()
            with wp.ScopedCapture(device=self.device) as capture:launch()
            self.graphs[key]=capture.graph
        wp.capture_launch(self.graphs[key])
        packed=self._snapshot_buffer.numpy()
        n,w=packed.shape[:2]
        points=packed[:,:,3:15].reshape(n,w,4,3)
        object_pos=packed[:,:,24:27]
        object_rot=packed[:,:,27:36].reshape(n,w,3,3)
        local=points.copy()
        if self.problem.surface:
            local[:,:,:2]=np.einsum('nwij,nwcj->nwci',object_rot.swapaxes(-1,-2),points[:,:,:2]-object_pos[:,:,None])
        result=dict(points=points,x=local,com=packed[:,:,:3],qvel=packed[:,:,36:39],
                    torso_rotation=packed[:,:,15:24].reshape(n,w,3,3),
                    object_position=object_pos,object_rotation=object_rot,realized_wrench=packed[:,:,39:51])
        self._snapshot_cache=result
        return result
