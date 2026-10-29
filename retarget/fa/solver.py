"""Fixed-template FA: L-BFGS, AdamW and acados SQP-RTI."""
from dataclasses import dataclass, replace
from time import perf_counter
import mujoco
import numpy as np
import warp as wp
from forceaware import mpc_loss as fa
from forceaware.servo_prior import ServoPriorOptimizer
from forceaware.optimizer_contract import finalize_candidate_evaluation
from retarget.trigrasp import (TrigraspMPC, TrigraspMPCParams,
                               quat_wxyz_to_matrix)

@dataclass(frozen=True)
class SolverSettings:
    history: int = 5
    line_search: int = 8
    control_scale: float = .001
    balanced_coordinates: bool = True
    reuse_forward: bool = True
    fuse_adjoint_zero: bool = True
    hand_solver: str = 'lbfgs'
    trust_radius: float = 1.
    compiled_threads: int = 8
    adamw_lr: float = .3
    adamw_weight_decay: float = .01
    adamw_persist: bool = False
    contact_rounds: int = 1
    contact_radius: float = .005
    contact_admm: bool = True
    contact_exploration: bool = True
    contact_sweeps: int = 1
    contact_wrench_weight: float = 1.
    contact_rho: float = .1
    contact_solver: str = 'sampled-lm'
    contact_samples: int = 8
    contact_lm_iterations: int = 3
    contact_qp_iterations: int = 96
    contact_seed: int = 0
    contact_motion_tolerance: float = .0001
    contact_wrench_tolerance: float = .05
    contact_min_improvement: float = .01
    rti_validate: bool = False


def settings_from_args(args):
    """One settings adapter for both run.py and Dexonomy-built grasp inputs."""
    from dataclasses import fields
    defaults = SolverSettings()
    values = {field.name: getattr(args, field.name, getattr(defaults, field.name))
              for field in fields(defaults)}
    values.update(compiled_threads=getattr(args, 'threads', defaults.compiled_threads),
                  contact_radius=args.contact_radius_mm/1000,
                  contact_seed=args.seed,
                  contact_admm=not getattr(args, 'no_admm', False),
                  contact_exploration=not getattr(args, 'no_contact_exploration', False),
                  contact_motion_tolerance=args.contact_motion_tolerance_mm/1000)
    return SolverSettings(**values)

@wp.struct
class PointState:
    c: wp.array2d(dtype=wp.vec3)
    z: wp.array2d(dtype=wp.vec3)
    last_c: wp.array2d(dtype=wp.vec3)
    dual: wp.array2d(dtype=wp.vec3)
    consensus_scale: wp.array(dtype=float)
    physical_wrench: int

@wp.struct
class ContactBatch:
    x: wp.array3d(dtype=wp.vec3)
    c: wp.array3d(dtype=wp.vec3)
    dual: wp.array3d(dtype=wp.vec3)
    anchor: wp.array3d(dtype=wp.vec3)
    weights: wp.array2d(dtype=float)
    friction: wp.array3d(dtype=wp.vec2)
    rotation: wp.array2d(dtype=wp.mat33)
    position: wp.array2d(dtype=wp.vec3)
    velocity: wp.array2d(dtype=wp.spatial_vector)
    goal: wp.array2d(dtype=wp.spatial_vector)

@wp.kernel(enable_backward=False)
def gather_contact_batch(job: fa.LossJob, points: PointState, stage: int,
                         qvel: wp.array2d(dtype=float), object_dof: int, batch: ContactBatch):
    world, slot = wp.tid()
    obj = job.model.object_body
    rotation = job.state.body_matrix[world, obj]
    position = job.state.body_position[world, obj]
    body = job.target.query_body[slot]
    hand = job.state.body_position[world, body]+job.state.body_matrix[world, body]@job.target.query_local[slot]
    batch.x[stage, world, slot] = wp.transpose(rotation)@(hand-position)
    batch.c[stage, world, slot] = points.c[world, slot]
    batch.dual[stage, world, slot] = points.dual[world, slot]
    batch.anchor[stage, world, slot] = job.target.query_position_object[slot]
    batch.friction[stage, world, slot] = wp.vec2(0.)
    output = job.target.contact_output[slot]
    if output >= 0 and output < job.model.output_count:
        batch.friction[stage, world, slot] = job.model.output_friction[output]
    if world == 0:
        batch.weights[stage, slot] = float(job.query_enabled)*job.target.query_mask[slot]*job.target.query_weight[slot]
    if slot == 0:
        batch.rotation[stage, world] = rotation
        batch.position[stage, world] = position
        linear = wp.transpose(rotation)@wp.vec3(qvel[world, object_dof], qvel[world, object_dof+1], qvel[world, object_dof+2])
        batch.velocity[stage, world] = wp.spatial_vector(linear[0], linear[1], linear[2],
            qvel[world, object_dof+3], qvel[world, object_dof+4], qvel[world, object_dof+5])
        delta = wp.transpose(rotation)@(wp.vec3(job.target.object_qpos[0], job.target.object_qpos[1], job.target.object_qpos[2])-position)
        target_q = wp.quat(job.target.object_qpos[4], job.target.object_qpos[5], job.target.object_qpos[6], job.target.object_qpos[3])
        relative = wp.quat_inverse(wp.quat_from_matrix(rotation))*target_q
        sign = wp.where(relative[3] >= 0., 2., -2.)
        batch.goal[stage, world] = wp.spatial_vector(delta[0], delta[1], delta[2],
            sign*relative[0], sign*relative[1], sign*relative[2])

@wp.kernel(enable_backward=False)
def scatter_contact_batch(updated: wp.array3d(dtype=wp.vec3), stage: int, points: PointState):
    world, slot = wp.tid()
    points.last_c[world, slot] = points.c[world, slot]
    points.c[world, slot] = updated[stage, world, slot]
    points.z[world, slot] = updated[stage, world, slot]

@wp.kernel(enable_backward=False)
def update_contact_dual(job: fa.LossJob, points: PointState, limit: float,
                        reduction: wp.array(dtype=float)):
    world, slot = wp.tid()
    weight = job.target.query_mask[slot]*job.target.query_weight[slot]
    if job.query_enabled == 1 and weight > 0.:
        obj, body = job.model.object_body, job.target.query_body[slot]
        hand = job.state.body_position[world,body]+job.state.body_matrix[world,body]@job.target.query_local[slot]
        x = wp.transpose(job.state.body_matrix[world,obj])@(hand-job.state.body_position[world,obj])
        residual = x-points.c[world,slot]
        dual = points.dual[world,slot]+residual
        if wp.isfinite(wp.length(dual)):
            points.dual[world,slot] = dual*wp.min(1.,limit/wp.max(wp.length(dual),1.e-12))
            wp.atomic_add(reduction,0,wp.length(residual))
            wp.atomic_add(reduction,1,wp.length(points.c[world,slot]-points.last_c[world,slot]))
            wp.atomic_add(reduction,2,1.)
        else:
            points.dual[world,slot] = wp.vec3(0.)
    else:
        points.dual[world,slot] = wp.vec3(0.)

@wp.kernel(enable_backward=False)
def initialize_points(job: fa.LossJob, points: PointState):
    world, slot = wp.tid()
    initial = job.target.query_position_object[slot]
    points.c[world, slot] = initial
    points.z[world, slot] = initial
    points.last_c[world, slot] = initial
    points.dual[world, slot] = wp.vec3(0.)

@wp.kernel(enable_backward=False)
def freeze_contact_points(job: fa.LossJob, points: PointState):
    world, slot = wp.tid()
    initial = job.target.query_position_object[slot]
    points.c[world, slot] = initial
    points.z[world, slot] = initial
    points.last_c[world, slot] = initial

@wp.func
def _hand_point(job: fa.LossJob, world: int, slot: int):
    body=job.target.query_body[slot]
    return job.state.body_position[world,body]+job.state.body_matrix[world,body]@job.target.query_local[slot]

@wp.kernel(enable_backward=False)
def variable_auxiliary(job: fa.LossJob, points: PointState):
    world,slot=wp.tid()
    obj=job.model.object_body
    target=job.state.body_position[world,obj]+job.state.body_matrix[world,obj]@points.c[world,slot]
    error=(_hand_point(job,world,slot)-target)*fa.MILLIMETERS
    distance=wp.sqrt(wp.dot(error,error)+fa.LOSS_EPSILON)
    argument=(distance-job.weights.marked_distance_mm)/wp.max(job.weights.marked_temperature_mm,fa.TEMPERATURE_EPSILON)
    job.auxiliary.marked_query_weight[world,slot]=fa._sigmoid(argument)

@wp.kernel
def variable_FA_loss(job: fa.LossJob, points: PointState, output: wp.array(dtype=float)):
    world=wp.tid()
    value=job.reference_scale*fa._reference_pose_loss(job,world)
    obj=job.model.object_body
    for slot in range(job.model.guidance_count):
        if job.guidance_enabled==1:
            value+=fa._physical_contact_loss(job,world,slot)
        if job.query_enabled==1:
            target=job.state.body_position[world,obj]+job.state.body_matrix[world,obj]@points.c[world,slot]
            error=(_hand_point(job,world,slot)-target)*fa.MILLIMETERS
            mask=job.target.query_mask[slot]*job.target.query_weight[slot]
            marked=job.auxiliary.marked_query_weight[world,slot]
            value+=job.weights.marked_query*mask*marked*wp.dot(error,error)
            consensus = error/fa.MILLIMETERS + job.state.body_matrix[world,obj]@points.dual[world,slot]
            value += points.consensus_scale[0]*mask*wp.dot(consensus,consensus)
    if job.penetration_enabled==1:
        value+=fa._penetration_loss(job,world)
    if points.physical_wrench == 1:
        value += fa._grasp_wrench_loss(job,world)
    output[world]=value+fa._finger_joint_collision_loss(job,world)

class FixedTemplateObjective(fa.MpcObjective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.points=[]; workspaces=[]
        for workspace in (*self.training.immediate, self.training.terminal):
            points=PointState(); shape=(self.worlds,workspace.job.model.guidance_count)
            points.consensus_scale = wp.zeros(1, dtype=float, device=self.device)
            for name in ('c','z','last_c','dual'):
                setattr(points,name,wp.zeros(shape,dtype=wp.vec3,device=self.device,requires_grad=name=='c'))
            self.points.append(points)
            tape=wp.Tape()
            tape.record_launch(variable_FA_loss,self.worlds,0,[workspace.job,points],[workspace.loss],wp.get_device(self.device),block_dim=256)
            workspaces.append(replace(workspace,tape=tape))
        self.training=replace(self.training,immediate=tuple(workspaces[:-1]),terminal=workspaces[-1])

    def prepare_window(self,*args,**kwargs):
        super().prepare_window(*args,**kwargs)
        for workspace,points in zip((*self.training.immediate,self.training.terminal),self.points):
            wp.launch(initialize_points,dim=points.c.shape,inputs=[workspace.job,points],device=self.device)

    def _evaluate_trajectory(self,runtime):
        for column,(workspace,points) in enumerate(zip((*runtime.immediate,runtime.terminal),self.points)):
            wp.launch(variable_auxiliary,dim=points.c.shape,inputs=[workspace.job,points],device=self.device)
            wp.launch(variable_FA_loss,dim=self.worlds,inputs=[workspace.job,points],outputs=[workspace.loss],device=self.device)
            wp.launch(fa._store_trajectory_loss,dim=self.worlds,inputs=[workspace.loss,column],outputs=[runtime.trajectory_terms],device=self.device)

    def snapshot(self,selected):
        rows=[]
        for workspace,points in zip((*self.training.immediate,self.training.terminal),self.points):
            if not workspace.job.query_enabled:continue
            job=workspace.job;bp=job.state.body_position.numpy()[selected];bm=job.state.body_matrix.numpy()[selected]
            body=job.target.query_body.numpy();local=job.target.query_local.numpy()
            h=bp[body]+np.einsum('nij,nj->ni',bm[body],local)
            x=(h-bp[self.index.object_body])@bm[self.index.object_body]
            rows.append(dict(c=points.c.numpy()[selected],z=points.z.numpy()[selected],x=x,
                c_before_surface_polish=points.last_c.numpy()[selected],
                dual=points.dual.numpy()[selected],body=body,hand_local=local,
                template=job.target.query_position_object.numpy(),template_hand_local=local,
                mask=job.target.query_mask.numpy()*job.target.query_weight.numpy()))
        result={key:np.stack([r[key] for r in rows]) for key in rows[0]}
        return result


def _lbfgs_direction(gradient,history):
    q=gradient.copy();alphas=[]
    for s,y in reversed(history):
        sy=np.sum(s*y,axis=1);valid=sy>1e-12
        inverse=np.divide(1.,sy,out=np.zeros_like(sy),where=valid)
        alpha=inverse*np.sum(s*q,axis=1);q-=alpha[:,None]*y;alphas.append((alpha,inverse))
    if history:
        s,y=history[-1];yy=np.sum(y*y,axis=1);sy=np.sum(s*y,axis=1)
        gamma=np.clip(np.divide(sy,yy,out=np.ones_like(sy),where=(yy>1e-12)&(sy>1e-12)),1e-8,1e4)
        q*=gamma[:,None]
    else:q/=np.maximum(np.max(np.abs(q),axis=1),1.)[:,None]
    for (s,y),(alpha,inverse) in zip(history,reversed(alphas)):
        beta=inverse*np.sum(y*q,axis=1);q+=s*(alpha-beta)[:,None]
    direction=-q
    bad=np.sum(direction*gradient,axis=1)>=0
    direction[bad]=-gradient[bad]/np.maximum(np.max(np.abs(gradient[bad]),axis=1),1.)[:,None]
    return direction

def optimizer_class(settings):
    class EvaluationBackend(ServoPriorOptimizer):
        def _make_objective(self, cpu_model, index, *, cfg, guidance_count):
            return FixedTemplateObjective(cpu_model,self.compiled,index,self.steps,self.terminal_observation,
                cfg=cfg,guidance_count=guidance_count,worlds=self.settings.worlds)
        def solve(self, *args, **kwargs):
            raise RuntimeError('Use FixedTemplateSolver')
        def _iteration(self, *args, **kwargs):
            raise RuntimeError('Base optimizer updates are disabled')

    class FixedTemplateSolver:
        def __init__(self, *args, **kwargs):
            if settings.hand_solver not in ('lbfgs','adamw','acados-rti'):
                raise ValueError('Unknown solver')
            if (settings.contact_rounds < 1 or settings.contact_sweeps < 1
                    or min(settings.contact_samples, settings.contact_lm_iterations, settings.contact_qp_iterations) < 1
                    or settings.contact_solver not in ('sampled-lm', 'qp')
                    or not np.isfinite([settings.contact_radius, settings.contact_wrench_weight, settings.contact_rho]).all()
                    or settings.contact_radius < 0 or settings.contact_wrench_weight < 0 or settings.contact_rho <= 0):
                raise ValueError('Invalid contact exploration settings')
            self.search_worlds=kwargs['cfg'].optimizer.multistart_count
            self.backend=EvaluationBackend(*args,**kwargs); self.graphs={}; self.last_solver_diag={}
            if settings.fuse_adjoint_zero and settings.hand_solver != 'acados-rti':
                from retarget.fa.adjoint import install
                install(self.backend)
            scale=np.ones(self.backend.index.action_dim)
            if settings.balanced_coordinates: scale[3:6]=10.; scale[6:]=50.
            self.variable_scale=np.tile(scale*settings.control_scale,self.backend.settings.horizon)
            self.adamw=None; self.trigrasp_sqp=None
            self.contact_explorer = None
            if settings.hand_solver in ('adamw', 'acados-rti'):
                if settings.contact_radius > 0 and settings.contact_admm:
                    from retarget.contact_exploration import ObjectMeshSurface
                    from retarget.contact_solvers import SampledComFreeLM, QPContactSolver, skew
                    contact_type = SampledComFreeLM if settings.contact_solver == 'sampled-lm' else QPContactSolver
                    self.contact_explorer = contact_type(
                        ObjectMeshSurface.from_model(self.backend.cpu_model, self.backend.index.object_body),
                        radius=settings.contact_radius, wrench_weight=settings.contact_wrench_weight,
                        sweeps=settings.contact_sweeps, rho=settings.contact_rho,
                        samples=settings.contact_samples, lm_iterations=settings.contact_lm_iterations,
                        qp_iterations=settings.contact_qp_iterations, seed=settings.contact_seed,
                        motion_tolerance=settings.contact_motion_tolerance,
                        wrench_tolerance=settings.contact_wrench_tolerance,
                        min_improvement=settings.contact_min_improvement,
                        device=self.backend.device,
                        mass=float(self.backend.cpu_model.body_mass[self.backend.index.object_body]),
                        dt=float(self.backend.cfg.simulator.time_grid.mpc_dt))
                    m, obj = self.backend.cpu_model, self.backend.index.object_body
                    com = np.asarray(m.body_ipos[obj])
                    inertia_rotation = quat_wxyz_to_matrix(m.body_iquat[obj])
                    inertia = inertia_rotation@np.diag(m.body_inertia[obj])@inertia_rotation.T
                    mass = float(m.body_mass[obj]); cross = skew(com)
                    spatial = np.block([[mass*np.eye(3), -mass*cross],
                                        [mass*cross, inertia-mass*cross@cross]])
                    self.contact_inverse = np.linalg.inv(spatial)
                    self.contact_explorer.inverse = self.contact_inverse
                    self.contact_object_dof = int(m.jnt_dofadr[m.body_jntadr[obj]])
                    from scipy.spatial import ConvexHull
                    vertices = np.unique(self.contact_explorer.surface.triangles.reshape(-1,3), axis=0)
                    self.contact_support_vertices = vertices[ConvexHull(vertices).vertices]
                    floors = np.flatnonzero((m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE) & (m.geom_bodyid == 0))
                    self.contact_floor = None
                    if len(floors):
                        geom = floors[0]
                        self.contact_floor = (m.geom_pos[geom].copy(), quat_wxyz_to_matrix(m.geom_quat[geom])[:,2])
                if settings.hand_solver == 'adamw':
                    for points in self.backend.objective.points:
                        points.physical_wrench = 1
            self.trigrasp_mpc = None
            if settings.hand_solver == 'adamw':
                self.trigrasp_mpc = TrigraspMPC(TrigraspMPCParams(
                    h=float(kwargs['cfg'].simulator.time_grid.mpc_dt),
                    horizon=self.backend.settings.horizon,
                    n_cmd=self.backend.index.action_dim,
                    num_contacts=4,
                ))
            if settings.hand_solver=='adamw':
                from retarget.fa.adamw import AdamW
                self.adamw=AdamW(learning_rate=settings.adamw_lr,
                    weight_decay=settings.adamw_weight_decay,persist=settings.adamw_persist,
                    device=self.backend.device if self.backend.device.is_cuda else None)
            if settings.hand_solver=='acados-rti':
                from retarget.fa.trigrasp_comfree_sqp import TrigraspComFreeSQP
                self.trigrasp_sqp=TrigraspComFreeSQP(
                    self.backend, radius=settings.trust_radius,
                    threads=settings.compiled_threads)

        def __getattr__(self,key):return getattr(self.backend,key)

        def _shared_contact_plan(self, qpos, targets, *, hand_points_object=None):
            from retarget.contact_plan import optimize_contact_plan
            objective = self.backend.objective
            jobs = [w.job for w in (*objective.training.immediate, objective.training.terminal)]
            friction = [job.model.output_friction.numpy()[job.target.contact_output.numpy()]
                        for job in jobs]
            return optimize_contact_plan(
                self.backend.cpu_model, self.backend.index, self.contact_explorer,
                qpos, targets, friction=friction,
                center=np.asarray(jobs[0].model.object_com_local),
                object_radius=jobs[0].model.object_radius,
                hand_points_object=hand_points_object,
                query_enabled=[job.query_enabled for job in jobs])

        def _contact_context(self, positions, rotations, velocity, goal):
            b = self.backend
            obj = b.index.object_body
            gravity = np.asarray(b.cpu_model.opt.gravity)
            linear = float(b.cpu_model.body_mass[obj])*np.einsum('bij,j->bi', rotations.transpose(0,2,1), gravity)
            com = np.asarray(b.cpu_model.body_ipos[obj])
            external = np.concatenate((linear, np.cross(com, linear)), axis=1)
            context = dict(inverse=self.contact_inverse, velocity=velocity,
                           external=external, goal=goal)
            if self.contact_floor is not None:
                floor_point, floor_normal = self.contact_floor
                local = np.einsum('bij,j->bi', rotations.transpose(0,2,1), floor_normal)
                support = self.contact_support_vertices
                heights = local@support.T
                ids = np.argmin(heights, axis=1)
                context['support_row'] = np.concatenate((local,np.cross(support[ids],local)),axis=1)
                context['support_gap'] = heights[np.arange(len(ids)),ids]+(positions-floor_point)@floor_normal
            return context

        def _contact_block(self, variables):
            """Pack the horizon once; solve all stages, starts and contact sets in batches."""
            rollout_started=perf_counter()
            if not self._same_variables(variables):
                self._evaluate(variables, False)
            self._contact_rollout_seconds.append(perf_counter()-rollout_started)
            b = self.backend
            workspaces = (*b.loss.immediate,b.loss.terminal)
            if not settings.contact_exploration:
                if not hasattr(self, '_contact_reduction'):
                    self._contact_reduction=wp.zeros(3,dtype=float,device=b.device)
                for workspace,points in zip(workspaces,b.objective.points):
                    wp.launch(freeze_contact_points,points.c.shape,inputs=[workspace.job,points],device=b.device)
                self._objective_dirty=True
                return []
            stages, worlds, contacts = len(workspaces), b.settings.worlds, b.guidance_count
            if not hasattr(self, '_contact_batch'):
                batch = ContactBatch()
                for name in ('x','c','dual','anchor'):
                    setattr(batch,name,wp.empty((stages,worlds,contacts),dtype=wp.vec3,device=b.device))
                batch.weights = wp.empty((stages,contacts),dtype=float,device=b.device)
                batch.friction = wp.empty((stages,worlds,contacts),dtype=wp.vec2,device=b.device)
                batch.rotation = wp.empty((stages,worlds),dtype=wp.mat33,device=b.device)
                batch.position = wp.empty((stages,worlds),dtype=wp.vec3,device=b.device)
                for name in ('velocity','goal'):
                    setattr(batch,name,wp.empty((stages,worlds),dtype=wp.spatial_vector,device=b.device))
                self._contact_batch = batch
                self._contact_updated = wp.empty((stages,worlds,contacts),dtype=wp.vec3,device=b.device)
                self._contact_reduction = wp.zeros(3,dtype=float,device=b.device)
            batch = self._contact_batch
            def gather():
                for stage,(workspace,points) in enumerate(zip(workspaces,b.objective.points)):
                    velocity = (b.steps[stage].inputs.dynamics.qvel if stage < stages-1
                                else b.steps[-1].recorded.result.qvel)
                    wp.launch(gather_contact_batch,dim=(worlds,contacts),inputs=[workspace.job,points,stage,
                        velocity,self.contact_object_dof,batch],device=b.device)
            if settings.contact_solver == 'sampled-lm':
                from retarget.contact_lm_warp import DeviceLambdaLM
                if not hasattr(self,'_lambda_device'):
                    m,obj=b.cpu_model,b.index.object_body
                    physics=dict(inverse=self.contact_inverse,com=m.body_ipos[obj],gravity=m.opt.gravity,
                                 floor=self.contact_floor,support_vertices=self.contact_support_vertices)
                    self._lambda_device=DeviceLambdaLM(self.contact_explorer,groups=stages*worlds,
                        contacts=contacts,worlds=worlds,physics=physics,device=b.device)
                k=np.r_[0,np.arange(stages-1)]
                templates=np.broadcast_to(self._current_targets.contact_position_object[k,None],
                    (stages,worlds,contacts,3)).reshape(-1,contacts,3)
                weight=self._current_targets.contact_mask[k]*self._current_targets.contact_weight[k]
                weight=weight*np.array([w.job.query_enabled for w in workspaces])[:,None]
                weights=np.broadcast_to(weight[:,None],(stages,worlds,contacts)).reshape(-1,contacts)
                geometry_started=perf_counter()
                changed=self._lambda_device.bind(batch,templates,weights,workspaces[0].job.model.object_radius)
                self._contact_geometry_seconds.append(perf_counter()-geometry_started)
                started=perf_counter()
                updated=self._lambda_device.job.selected.reshape((stages,worlds,contacts))
                def contact_launch():
                    gather()
                    self._lambda_device._launch()
                    for stage,points in enumerate(b.objective.points):
                        wp.launch(scatter_contact_batch,dim=(worlds,contacts),inputs=[updated,stage,points],device=b.device)
                if b.device.is_cuda:
                    if not hasattr(self,'_lambda_kernels_ready'):
                        gather();self._lambda_device.solve();wp.synchronize_device(b.device)
                        self._lambda_kernels_ready=True
                    if changed or not hasattr(self,'_lambda_contact_graph'):
                        with wp.ScopedCapture(device=b.device) as capture:contact_launch()
                        self._lambda_contact_graph=capture.graph
                    wp.capture_launch(self._lambda_contact_graph)
                else:contact_launch()
                statistics=self._lambda_device.job.statistics.numpy()
                elapsed=perf_counter()-started
                diagnostics=[]
                for group,row in enumerate(statistics):
                    if row[4]<=0:continue
                    diagnostics.append(dict(stage=int(group//worlds),world=int(group%worlds),
                        method='sampled_comfree_batched_lm',solver_backend='warp_device_lambda_lm_6x6',
                        initial_value=float(row[0]),final_value=float(row[1]),selected_candidate=int(row[2]),
                        max_shift_mm=float(row[3]),active_contacts=int(row[4]),
                        frozen_off_surface_contacts=int(row[5]),incumbent_projection_mm=float(row[6]),
                        candidate_count=settings.contact_samples,lm_iterations=settings.contact_lm_iterations,
                        contact_model='lambda_comfree_applied_force',finite_difference_gradients=False,
                        includes_incumbent=True,selection_priority='task_feasibility_then_geometry',held_feasible=bool(row[7]),searched=bool(row[8]),
                        motion_error_mm=float(row[9]),feasible=bool(row[10])))
                self._objective_dirty=True
                self._contact_device_seconds.append(elapsed)
                return diagnostics
            gather()
            # All arrays are small packed contact data, not full body-state copies.
            current = batch.c.numpy().reshape(-1,contacts,3)
            template = batch.anchor.numpy().reshape(-1,contacts,3)
            hand = (batch.x.numpy()+batch.dual.numpy()).reshape(-1,contacts,3)
            weights = np.broadcast_to(batch.weights.numpy()[:,None],(stages,worlds,contacts)).reshape(-1,contacts)
            friction = batch.friction.numpy().reshape(-1,contacts,2)
            enabled = np.any(weights > 0,axis=1) & np.isfinite(hand).all(axis=(1,2))
            diagnostics = []
            if enabled.any():
                context = self._contact_context(batch.position.numpy().reshape(-1,3)[enabled],
                    batch.rotation.numpy().reshape(-1,3,3)[enabled],
                    batch.velocity.numpy().reshape(-1,6)[enabled],batch.goal.numpy().reshape(-1,6)[enabled])
                job = workspaces[0].job
                current[enabled], rows = self.contact_explorer.solve_batch(current[enabled],template[enabled],
                    hand[enabled],weights[enabled],friction[enabled],center=np.asarray(job.model.object_com_local),
                    object_radius=job.model.object_radius,context=context)
                for group,row in zip(np.flatnonzero(enabled),rows):
                    diagnostics.append(dict(stage=int(group//worlds),world=int(group%worlds),**row))
            self._contact_updated.assign(current.astype(np.float32).reshape(stages,worlds,contacts,3))
            for stage,points in enumerate(b.objective.points):
                wp.launch(scatter_contact_batch,dim=(worlds,contacts),inputs=[self._contact_updated,stage,points],device=b.device)
            self._objective_dirty = True
            return diagnostics

        def _dual_block(self):
            b = self.backend
            self._contact_reduction.zero_()
            for workspace,points in zip((*b.loss.immediate,b.loss.terminal),b.objective.points):
                wp.launch(update_contact_dual,dim=points.c.shape,inputs=[workspace.job,points,
                    4*settings.contact_radius,self._contact_reduction],device=b.device)
            values = self._contact_reduction.numpy()
            self._objective_dirty = True
            return dict(primal_mean_mm=float(values[0]/max(values[2],1)*1000),
                        surface_step_mean_mm=float(values[1]/max(values[2],1)*1000))

        def _trigrasp_seed(self, qpos, qvel, targets, execution_steps):
            """Legacy lightweight AdamW seed; never used by the SQP branch."""
            if self.trigrasp_mpc is None:
                return None
            try:
                b = self.backend
                index = b.index
                model = b.cpu_model
                data = mujoco.MjData(model)
                data.qpos[:] = np.asarray(qpos, float)
                data.qvel[:] = np.asarray(qvel, float)
                mujoco.mj_forward(model, data)
                qobj = np.asarray(qpos[index.object_qpos:index.object_qpos + 7], float)
                hand = np.asarray(qpos[index.hand_qpos], float)
                hand_qvel = np.asarray([
                    model.jnt_dofadr[int(np.flatnonzero(model.jnt_qposadr == q)[0])]
                    for q in index.hand_qpos
                ], np.int32)
                velocity = np.r_[qvel[index.object_qpos:index.object_qpos + 6], qvel[hand_qvel]]
                state = np.r_[qobj, hand, velocity, hand]
                target_pose = np.asarray(targets.object_qpos[min(len(targets.object_qpos) - 1, execution_steps - 1)], float)
                target_hand = np.asarray(targets.actions[min(len(targets.actions) - 1, execution_steps - 1)], float)
                frame = min(len(targets.contact_position_object) - 1, max(execution_steps - 1, 0))
                mask = np.asarray(targets.contact_mask[frame], float) > 0
                ids = np.flatnonzero(mask)[:self.trigrasp_mpc.params.num_contacts]
                if ids.size == 0:
                    return {"status": "no_active_contact"}
                anchors = np.asarray(targets.contact_position_object[frame, ids], float)
                tips, jac_rows, gaps = [], [], []
                for point, body_index in zip(np.asarray(targets.query_local[frame, ids]),
                                             np.asarray(targets.query_body[frame, ids], int)):
                    body_index = int(body_index)
                    tips.append(data.xpos[body_index] + data.xmat[body_index].reshape(3, 3) @ point)
                    jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
                    mujoco.mj_jac(model, data, jacp, jacr, point, body_index)
                    selected = np.c_[jacp[:, index.object_qpos:index.object_qpos + 6], jacp[:, hand_qvel]]
                    target_world = qobj[:3] + quat_wxyz_to_matrix(qobj[3:7]) @ anchors[len(tips) - 1]
                    gap = float(np.linalg.norm(tips[-1] - target_world) - .01)
                    gaps.extend([gap] * 4); jac_rows.extend([selected] * 4)
                result = self.trigrasp_mpc.plan_once(
                    state=state, target_object_pose=target_pose,
                    target_hand_q=target_hand, phi=np.asarray(gaps),
                    jac=np.vstack(jac_rows), anchors=anchors,
                    tip_positions=np.asarray(tips),
                    tip_jacobian=np.vstack([j[:3, 6:] for j in jac_rows[::4]]),
                )
                return {"status": result["solve_status"], "cost": result["cost"],
                        "controls": result["u_traj"]}
            except Exception as exc:
                return {"status": "disabled", "error": str(exc)}

        def _optimize_direct_contacts(self, qpos, targets):
            plan = self._shared_contact_plan(qpos, targets)
            self.contact_snapshot = plan.snapshot
            return plan.points, plan.diagnostics

        def _evaluate(self,variables,gradient):
            if settings.hand_solver == 'adamw' and self.backend.device.is_cuda:
                return self._evaluate_cuda(variables,gradient)
            b=self.backend
            same=settings.reuse_forward and np.array_equal(variables,getattr(self,'_forward_variables',None))
            dirty=getattr(self,'_objective_dirty',False)
            if same and not gradient and not dirty:
                self.evaluations+=1
                if hasattr(self,'value_reuses'):self.value_reuses+=1
                return self._forward_value.copy(),None
            reuse=gradient and same
            key=('objective',gradient) if same and dirty else ('backward' if reuse else gradient)
            if not same:
                raw=b.control.raw.numpy();raw[:,b.index.action_dim:]=variables*self.variable_scale
                b.control.raw.assign(raw);b.control.valid.fill_(1)
            def launch():
                if same:
                    if dirty:b.objective.evaluate(b.control,allow_init=False)
                    if gradient:b._backward(allow_init=False)
                else:self._launch(gradient)
            if key not in self.graphs:
                launch();wp.synchronize()
                with wp.ScopedCapture(device=b.device) as capture:launch()
                self.graphs[key]=capture.graph
            wp.capture_launch(self.graphs[key]);wp.synchronize()
            self._forward_variables=variables.copy()
            self._objective_dirty=False
            if hasattr(self,'forward_calls'):
                self.forward_calls+=int(not same);self.backward_calls+=int(gradient);self.backward_reuses+=int(reuse)
                self.objective_only_calls+=int(same and dirty)
            value=b.loss.total.numpy().astype(float)*1e-6
            valid=b.control.valid.numpy().astype(bool)&np.isfinite(value)
            value[~valid]=np.inf
            self.evaluations+=1
            if not gradient:
                self._forward_value=value.copy()
                return value,None
            g=b.control.gradient.numpy()[:,b.index.action_dim:].astype(float)*self.variable_scale
            valid&=np.isfinite(g).all(axis=1);value[~valid]=np.inf;g[~valid]=0.
            self._forward_value=value.copy()
            return value,g

        def _same_variables(self,variables):
            previous=getattr(self,'_forward_variables',None)
            if previous is None:return False
            if settings.hand_solver == 'adamw' and self.backend.device.is_cuda:
                import torch
                current=torch.as_tensor(variables,device=str(self.backend.device),dtype=torch.float64)
                return torch.equal(current,previous)
            return np.array_equal(variables,previous)

        def _evaluate_cuda(self,variables,gradient):
            """Common AdamW callback without control/value/gradient host copies."""
            import torch
            b=self.backend
            on_device=isinstance(variables,torch.Tensor)
            current=torch.as_tensor(variables,device=str(b.device),dtype=torch.float64)
            same=settings.reuse_forward and self._same_variables(current)
            dirty=getattr(self,'_objective_dirty',False)
            if same and not gradient and not dirty:
                self.evaluations+=1
                if hasattr(self,'value_reuses'):self.value_reuses+=1
                value=self._forward_value.clone()
                return (value if on_device else value.cpu().numpy()),None
            reuse=gradient and same
            key=('objective',gradient) if same and dirty else ('backward' if reuse else gradient)
            # Capture on Warp's own non-default stream. Torch's legacy
            # default stream cannot be captured by Warp on all versions.
            stream=wp.get_stream(b.device)
            caller_stream=torch.cuda.current_stream(current.device)
            if not hasattr(self,'_adamw_torch_stream'):
                self._adamw_torch_stream=torch.cuda.ExternalStream(stream.cuda_stream,device=current.device)
            torch_stream=self._adamw_torch_stream
            if not hasattr(self,'_device_variable_scale'):
                self._device_variable_scale=torch.as_tensor(self.variable_scale,device=current.device)
            torch_stream.wait_stream(caller_stream)
            with torch.no_grad(),torch.cuda.stream(torch_stream):
                if not same:
                    wp.to_torch(b.control.raw)[:,b.index.action_dim:].copy_(current*self._device_variable_scale)
                    b.control.valid.fill_(1)
                def launch():
                    if same:
                        if dirty:b.objective.evaluate(b.control,allow_init=False)
                        if gradient:b._backward(allow_init=False)
                    else:self._launch(gradient)
                if key not in self.graphs:
                    launch();wp.synchronize_device(b.device)
                    with wp.ScopedCapture(device=b.device) as capture:launch()
                    self.graphs[key]=capture.graph
                wp.capture_launch(self.graphs[key])
                wp.synchronize_device(b.device)
                value=wp.to_torch(b.loss.total).double()*1e-6
                valid=wp.to_torch(b.control.valid).bool()&torch.isfinite(value)
                g=None
                if gradient:
                    g=wp.to_torch(b.control.gradient)[:,b.index.action_dim:].double()*self._device_variable_scale
                    valid=valid&torch.isfinite(g).all(dim=1)
                    g[~valid]=0.
                value[~valid]=torch.inf
                self._forward_value=value.clone()
                self._forward_variables=current.clone()
            caller_stream.wait_stream(torch_stream)
            self._objective_dirty=False;self.evaluations+=1
            if hasattr(self,'forward_calls'):
                self.forward_calls+=int(not same);self.backward_calls+=int(gradient);self.backward_reuses+=int(reuse)
                self.objective_only_calls+=int(same and dirty)
            if on_device:return value,g
            return value.cpu().numpy(),None if g is None else g.cpu().numpy()

        def _launch(self,gradient):
            b=self.backend;b._clear_contact_history();b._set_contact_dynamics();b._forward(allow_init=False)
            b._validate_simulator_candidates();b.objective.evaluate(b.control,allow_init=False)
            if gradient:b._backward(allow_init=False)

        def _hand_block(self,variables,iterations,*,continue_window=False):
            if settings.hand_solver=='adamw':
                variables,accepted,diagnostic=self.adamw.update(variables,self._evaluate,
                    iterations=iterations,horizon=self.backend.settings.horizon,
                    advance=self._current_execution_steps/self.backend.cfg.simulator.time_grid.knot_substeps,
                    continue_window=continue_window,evaluate_on_device=self.backend.device.is_cuda)
                self.hand_diagnostics.append(diagnostic)
                return variables,accepted
            self._forward_variables=None
            value,gradient=self._evaluate(variables,True);history=[];accepted_steps=0
            for _ in range(iterations):
                direction=_lbfgs_direction(gradient,history)
                slope=np.sum(gradient*direction,axis=1)
                pending=np.isfinite(value)&(slope<0);accepted=np.zeros(len(value),bool)
                alpha=np.ones(len(value));trial=variables.copy();new=variables.copy()
                for _ in range(settings.line_search):
                    if not pending.any():break
                    trial[pending]=variables[pending]+alpha[pending,None]*direction[pending]
                    candidate,_=self._evaluate(trial,False)
                    good=pending&(candidate<=value+1e-4*alpha*slope)
                    new[good]=trial[good];accepted|=good;pending&=~good
                    alpha[pending]*=.5
                next_value,next_gradient=self._evaluate(new,True)
                s=new-variables;y=next_gradient-gradient
                sy=np.sum(s*y,axis=1);keep=(sy>1e-12)&accepted
                s[~keep]=0.;y[~keep]=0.;history.append((s,y))
                if len(history)>settings.history:history.pop(0)
                variables,value,gradient=new,next_value,next_gradient
                accepted_steps+=int(accepted.sum())
                if not accepted.any():break
            return variables,accepted_steps

        def _direct_sqp_plan(self, raw, qpos, qvel, targets, controls,
                             diagnostic, execution_steps, started, previous_action):
            from forceaware.optimizer import Plan
            b = self.backend
            controls = np.asarray(controls, float).reshape(-1, b.index.action_dim)
            if not diagnostic.get('controls_are_absolute') or len(controls) < execution_steps:
                raise ValueError('SQP must supply the predicted absolute servo commands')
            control_sequence = controls[:execution_steps].copy()
            limits = np.asarray(diagnostic.get('physical_action_limits', np.full(b.index.action_dim, .01)), float)
            previous = np.asarray(previous_action, float).copy()
            safe_sequence = np.empty_like(control_sequence)
            for step in range(len(control_sequence)):
                safe_sequence[step] = previous + np.clip(
                    control_sequence[step] - previous, -limits, limits)
                previous = safe_sequence[step]
            control_sequence = safe_sequence
            worlds = raw.shape[0]
            scores = np.full(worlds, float(diagnostic.get('cost', 0.)), np.float32)
            valid = np.ones(worlds, dtype=bool)
            diag = dict(
                selected=0, scores=scores, drifts=np.zeros(worlds, np.float32),
                has_finite_best=valid, valid=valid,
                training_failure_iteration=np.full(worlds, -1, np.int32),
                training_failure_code=np.zeros(worlds, np.int32),
                rejection_code=np.zeros(worlds, np.int32),
                solver=diagnostic,
            )
            self.last_solver_diag = diagnostic
            qpos_sequence = np.repeat(np.asarray(qpos, float)[None], execution_steps, axis=0)
            qvel_sequence = np.repeat(np.asarray(qvel, float)[None], execution_steps, axis=0)
            return Plan(
                raw=np.asarray(raw[0], np.float32), initial_qpos=np.asarray(qpos, np.float64).copy(),
                next_qpos=np.asarray(qpos, np.float64).copy(), next_qvel=np.asarray(qvel, np.float64).copy(),
                action=control_sequence[-1].copy(), qpos_sequence=qpos_sequence,
                qvel_sequence=qvel_sequence, control_sequence=control_sequence,
                loss=float(scores[0]), drift=0., grad_norm=0.,
                seconds=perf_counter() - started, diag=diag)

        def solve(self,qpos,qvel,targets,*,warm,previous_action,allow_init,seed,execution_steps,start_time):
            started=perf_counter(); b=self.backend
            if allow_init: raise ValueError('The prepared initial pose is fixed')
            if b.settings.iterations<1: raise ValueError('Positive iteration budget required')
            self._current_targets=targets; self._current_execution_steps=execution_steps
            b._prepare_window(qpos,qvel,targets,warm=warm,previous_action=previous_action,seed=seed,start_time=start_time)
            if settings.hand_solver == 'adamw' and self.contact_explorer is not None:
                from retarget.contact_memory import restore
                if hasattr(self,'_contact_memory'):
                    for stage,(workspace,points) in enumerate(zip((*b.loss.immediate,b.loss.terminal),b.objective.points)):
                        job=workspace.job;target=job.target
                        wp.launch(restore,points.c.shape,inputs=[points.c,points.z,points.last_c,points.dual,
                            target.query_position_object,target.query_local,target.query_body,target.query_mask,
                            target.query_weight,job.query_enabled,stage,self._contact_memory_shift,
                            settings.contact_radius,self._contact_memory],device=b.device)
                for workspace, points in zip((*b.loss.immediate, b.loss.terminal), b.objective.points):
                    weight = workspace.job.target.query_mask.numpy()*workspace.job.target.query_weight.numpy()
                    # Normalize on the host: reverse replay of consecutive
                    # dynamic Warp loops cannot reuse the first reduction.
                    points.consensus_scale.fill_(.5*settings.contact_rho*1.e6
                        /(settings.contact_radius**2*max(float(weight.sum()), 1.e-12)))
            if settings.hand_solver=='acados-rti':
                raw = b.control.raw.numpy(); raw[:,:b.index.action_dim] = 0.; b.control.raw.assign(raw)
                variables = raw[:,b.index.action_dim:].astype(float) / self.variable_scale
                contact_points, contact_diagnostic = self._optimize_direct_contacts(qpos, targets)
                variables, accepted, diagnostic = self.trigrasp_sqp.update(
                    variables, targets, qpos=qpos, qvel=qvel,
                    previous_action=previous_action, contact_points=contact_points)
                diagnostic['contact_optimizer'] = contact_diagnostic
                raw[:,b.index.action_dim:] = variables * self.variable_scale
                b.control.raw.assign(raw)
                diagnostic['accepted_candidates'] = int(accepted)
                return self._direct_sqp_plan(
                    raw, qpos, qvel, targets, diagnostic.pop('controls'),
                    diagnostic, execution_steps, started, previous_action)
            contact_diagnostics = []
            trigrasp_seed = self._trigrasp_seed(qpos, qvel, targets, execution_steps)
            trigrasp_diagnostic = None if trigrasp_seed is None else {
                key: value for key, value in trigrasp_seed.items() if key != 'controls'
            }
            raw=b.control.raw.numpy(); raw[:,:b.index.action_dim]=0.; b.control.raw.assign(raw)
            if trigrasp_seed is not None and 'controls' in trigrasp_seed:
                raw = b.control.raw.numpy()
                controls = np.asarray(trigrasp_seed['controls'], float)
                width = b.index.action_dim
                count = min(controls.shape[0], b.settings.horizon)
                scale_slice = self.variable_scale[:count * width]
                controls = np.clip(controls[:count].reshape(-1), -scale_slice, scale_slice)
                raw[0, width:width + count * width] = (
                    controls / scale_slice
                )
                b.control.raw.assign(raw)
            variables=raw[:,b.index.action_dim:].astype(float)/self.variable_scale
            prepared=perf_counter(); self.evaluations=0; accepted=0; hand_seconds=0.; self.hand_diagnostics=[]
            self._forward_variables=None; self._objective_dirty=False
            self.forward_calls=0; self.backward_calls=0; self.backward_reuses=0; self.value_reuses=0
            self.objective_only_calls=0; self._contact_device_seconds=[]
            self._contact_rollout_seconds=[]; self._contact_geometry_seconds=[]
            # Preserve the successful FA baseline: five L-BFGS blocks, each
            # with its own curvature history, splitting the total 40 updates.
            blocks=5 if settings.hand_solver=='lbfgs' else (settings.contact_rounds if settings.contact_admm else 1)
            counts=(np.full(blocks,b.settings.iterations,int) if settings.hand_solver=='adamw'
                    else np.full(blocks,b.settings.iterations//blocks,int))
            if settings.hand_solver=='lbfgs': counts[:b.settings.iterations%blocks]+=1
            if settings.hand_solver=='lbfgs': self._evaluate(variables,False)
            alternating_diagnostics = []
            contact_seconds = 0.
            for block, count in enumerate(counts):
                if count==0: continue
                if self.contact_explorer is not None:
                    contact_started = perf_counter()
                    contact_diagnostics.extend(self._contact_block(variables))
                    contact_seconds += perf_counter()-contact_started
                hand_started=perf_counter()
                variables,n=self._hand_block(variables,int(count),continue_window=block > 0); accepted+=n
                hand_seconds+=perf_counter()-hand_started
                if self.contact_explorer is not None:
                    # The last hand evaluation restores the best controls and
                    # their rollout. Use exactly that rollout for the dual.
                    alternating_diagnostics.append(dict(round=block, **self._dual_block()))
            # Multipliers belong to the augmented subproblems, not final
            # candidate ranking. Re-evaluate the final contact/action pair
            # with physical FA terms only, retaining multipliers in snapshots.
            for points in b.objective.points:
                points.consensus_scale.zero_()
            if self.contact_explorer is not None:
                self._objective_dirty = True
            values,_=self._evaluate(variables,False)
            valid=np.isfinite(values); selected=int(np.argmin(values))
            if not valid.any(): raise FloatingPointError('All solver candidates are invalid')
            if self.adamw is not None and self.adamw.persist:
                self.adamw.retain_candidate_memory(selected)
            if settings.hand_solver == 'adamw' and self.contact_explorer is not None:
                from retarget.contact_memory import allocate,remember
                workspaces=(*b.loss.immediate,b.loss.terminal)
                if not hasattr(self,'_contact_memory'):
                    self._contact_memory=allocate(len(workspaces),b.guidance_count,b.device)
                for stage,(workspace,points) in enumerate(zip(workspaces,b.objective.points)):
                    job=workspace.job;target=job.target
                    wp.launch(remember,b.guidance_count,inputs=[points.c,points.dual,
                        target.query_local,target.query_body,target.query_mask,target.query_weight,
                        job.query_enabled,stage,selected,self._contact_memory],device=b.device)
                self._contact_memory_shift=int(execution_steps)
            scores=(values*1e6).astype(np.float32); drifts=b._candidate_drifts(targets,execution_steps)
            b.control.best_raw.assign(b.control.raw); b.control.best_loss.assign(scores)
            b.control.best_available.assign(valid.astype(np.int32))
            b.control.best_grad_norm.assign(np.linalg.norm(b.control.gradient.numpy(),axis=1).astype(np.float32))
            evaluation=finalize_candidate_evaluation(scores,drifts,has_finite_best=valid,hard_simulator_valid=valid,
                training_failure_iteration=np.full(len(valid),-1,np.int32),training_failure_code=np.zeros(len(valid),np.int32))
            plan=b._plan(selected,started,evaluation=evaluation,execution_steps=execution_steps)
            b._commit_regularization_history()
            self.last_solver_diag=dict(solver=settings.hand_solver,hand_iteration_budget=int(counts.sum()),
                evaluations=self.evaluations,accepted_candidate_steps=accepted,hand_subproblem_iteration_budgets=counts.tolist(),
                final_ranking=('physical_FA_with_updated_contacts' if settings.hand_solver == 'adamw' else 'pure_FA'),
                variable_scales=self.variable_scale[:b.index.action_dim].tolist(),
                MANO_role=('local_contact_anchor' if self.contact_explorer is not None and settings.contact_exploration else 'fixed_template'),
                contact_admm_enabled=self.contact_explorer is not None,
                contact_exploration_enabled=self.contact_explorer is not None and settings.contact_exploration,
                prepare_seconds=prepared-started,hand_seconds=hand_seconds,hand_solver=settings.hand_solver,
                hand_subproblems=self.hand_diagnostics,search_starts=self.search_worlds,evaluation_worlds=b.settings.worlds,
                contact_surface_optimizer=(self.contact_explorer.method if self.contact_explorer is not None and settings.contact_exploration else None),
                contact_surface_diagnostics=contact_diagnostics,
                contact_seconds=contact_seconds, alternating_subproblems=alternating_diagnostics,
                contact_device_solve_seconds=self._contact_device_seconds,
                contact_rollout_seconds=self._contact_rollout_seconds,
                contact_geometry_seconds=self._contact_geometry_seconds,
                objective_only_calls=self.objective_only_calls,
                contact_radius_mm=settings.contact_radius*1000, contact_rho=settings.contact_rho,
                trigrasp_mpc=trigrasp_diagnostic,
                forward_calls=self.forward_calls,backward_calls=self.backward_calls,backward_reuses=self.backward_reuses,
                value_reuses=self.value_reuses,fused_adjoint_zero=settings.fuse_adjoint_zero,evaluation_backend='native',
                gradient_source='analytic_adjoint',
                finite_difference_gradients=False,grad_norm_at_final_controls=settings.hand_solver=='lbfgs')
            return plan
    return FixedTemplateSolver
