"""Four-pose CARD rollout using main's energy, CEM and adjacent-pose costs.

The current robot output is a fixed continuity reference. Current and three true
future observations are optimized jointly, publishing only the first pose.
Collision costs stay soft. Joint continuity bridges consecutive published poses;
overlapping predictions are warm starts, never treated as executed poses.
"""
import copy
import hashlib
import json
from dataclasses import dataclass,asdict,replace
import time
import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation
from robokit.opt import PopulationSolverConfig,Stage,VarValues,VectorVar
from robokit.opt.var_values import VarKey
from robokit.terms.task import Task
from robokit.utils.warp_utils import wp_vec7
from robokit.lie.se3_kernels import se3_apply_func,se3_inverse_func
from .retarget import TactileRetarget
from .math import CardSettings
from .robokit import (CardCostTask,CardData,CardSampler,CardSamplerConfig,CardPopulationSolver,
    RobokitCardSolver,card_cost,pack,advance,rotvec_near,squared)
from moxian_web.wuji_window import identity,FUTURE,HORIZON
BASELINE='b84fa0a9c57187ce5131fcbc1f6c129ae217a32c'
ALGORITHM='card_wuji2_v21_main_future3_rtc_prefix'
# q is in radians, whereas fingertip/contact residuals are in millimetres.
# The former 0.15 squared was negligible even for a multi-degree joint jump.
LIVE_JOINT_SMOOTH_WEIGHT=30.
BOUNDARY_JOINT_MULTIPLIER=2.


def aligned_warm_start(stamps,old_stamps,old_values,fallback):
    """Reuse only the timestamp overlap; do not freeze/extrapolate the old tail."""
    mask=np.zeros(len(stamps),dtype=bool)
    if old_stamps is None or old_values is None:return fallback.copy(),mask
    mask=(stamps>=old_stamps[0])&(stamps<=old_stamps[-1])
    result=fallback.copy()
    for j in range(result.shape[1]):
        result[mask,j]=np.interp(stamps[mask],old_stamps,old_values[:,j])
    return result,mask


def chunk_prefix_weights(stamps,overlap,inference_delay_ms):
    """RTC exponential prefix schedule, clipped to real timestamp coverage.

    Adapted from Physical-Intelligence/real-time-chunking-kinetix/src/model.py
    get_prefix_weights. Here the guidance is a CEM cost, not a flow-model VJP.
    """
    end=int(np.count_nonzero(overlap))
    if not end:return np.zeros(len(stamps),np.float32)
    start=min(int(np.searchsorted(stamps,stamps[0]+max(0.,inference_delay_ms))),end)
    weights=np.clip((start-1-np.arange(len(stamps)))/(end-start+1)+1.,0.,1.)
    weights*=np.expm1(weights)/np.expm1(1.)
    return np.where(overlap,weights,0.).astype(np.float32)


def valid_window(frames):
    if len(frames)!=HORIZON:return False
    stamps=np.asarray([f['timestamp_ms'] for f in frames],float)
    return (np.isfinite(stamps).all() and (np.diff(stamps)>0).all() and
            (np.diff(stamps)<=250).all() and all(identity(f)==identity(frames[0]) for f in frames))


@wp.kernel(enable_backward=False)
def adjacent_main_cost(x:wp.array2d(dtype=float),before:wp.array2d(dtype=float),
        poses:wp.array2d(dtype=wp_vec7),prior_poses:wp.array2d(dtype=wp_vec7),
        data:CardData,prior:CardData,parts:wp.array2d(dtype=float),costs:wp.array(dtype=float)):
    """Main's smooth and material-fix terms with candidate h-1 as previous."""
    i=wp.tid();w=data.weights;extra=float(0.0)
    for j in range(20):parts[i,6]+=w[6]*(x[i,6+j]-before[i,6+j])**2.0
    for j in range(3):parts[i,7]+=w[7]*((x[i,j]-before[i,j])*1000.0)**2.0
    rv=wp.vec3(x[i,3],x[i,4],x[i,5]);old=wp.vec3(before[i,3],before[i,4],before[i,5])
    parts[i,8]+=w[8]*squared(rotvec_near(rv,old)-old)
    for region in range(16):
        if data.active[region]!=0 and prior.active[region]!=0:
            tf=poses[i,data.links[region]];ptf=prior_poses[i,prior.links[region]]
            alpha=float(0.0)
            if region>0 and (region-1)%3>0:alpha=0.15
            target=se3_apply_func(se3_inverse_func(tf),data.target[region])
            previous_target=se3_apply_func(se3_inverse_func(ptf),prior.target[region])
            hit=wp.mesh_query_point_no_sign(data.meshes[region],(target+alpha*data.preference[region])/(1.0+alpha),10.0)
            old_hit=wp.mesh_query_point_no_sign(prior.meshes[region],(previous_target+alpha*prior.preference[region])/(1.0+alpha),10.0)
            if hit.result and old_hit.result:
                point=wp.mesh_eval_position(data.meshes[region],hit.face,hit.u,hit.v)
                old_point=wp.mesh_eval_position(prior.meshes[region],old_hit.face,old_hit.u,old_hit.v)
                parts[i,2]+=data.rho[region]**2.0*w[2]*squared((se3_apply_func(tf,point)-se3_apply_func(tf,old_point))*1000.0)
    extra=parts[i,2]+parts[i,6]+parts[i,7]+parts[i,8]
    costs[i]+=extra


@wp.kernel(enable_backward=False)
def reduce_cost(rows:wp.array2d(dtype=float),boundary_parts:wp.array2d(dtype=float),
        boundary_joint_multiplier:float,costs:wp.array(dtype=float)):
    i=wp.tid();value=float(0.0)
    for h in range(rows.shape[1]):value+=rows[i,h]
    horizon=float(rows.shape[1])
    # The previous-output -> current-output edge occurs once per solve. It
    # must not weaken when extra future observations are added to the window.
    # Only joint continuity is strengthened; retain main's wrist/contact costs.
    costs[i]+=value/horizon+(boundary_joint_multiplier-1.0/horizon)*boundary_parts[i,6]


@wp.kernel(enable_backward=False)
def reduce_chunk_cost(rows:wp.array2d(dtype=float),boundary_parts:wp.array2d(dtype=float),
        boundary_joint_multiplier:float,x:wp.array2d(dtype=float),prefix:wp.array2d(dtype=float),
        prefix_weights:wp.array(dtype=float),prefix_coefficient:float,costs:wp.array(dtype=float)):
    # Fused into the existing reduction: no extra FK, mesh query, candidate,
    # optimization pass, backward pass or per-iteration GPU launch.
    i=wp.tid();value=float(0.0);continuity=float(0.0)
    for h in range(rows.shape[1]):
        value+=rows[i,h]
        if prefix_weights[h]>0.0:
            for j in range(20):
                delta=x[i,h*26+6+j]-prefix[h,j]
                continuity+=prefix_weights[h]*delta*delta
    horizon=float(rows.shape[1])
    costs[i]+=value/horizon+(boundary_joint_multiplier-1.0/horizon)*boundary_parts[i,6]+prefix_coefficient*continuity


@wp.kernel(enable_backward=False)
def save_trajectory(x:wp.array2d(dtype=float),costs:wp.array(dtype=float),counter:wp.array(dtype=int),
        values:wp.array2d(dtype=float),saved:wp.array(dtype=float),widths:wp.array(dtype=int),population:int):
    i=wp.tid();slot=counter[0]%widths.shape[0];offset=slot*population+i
    for j in range(x.shape[1]):values[offset,j]=x[i,j]
    saved[offset]=costs[i]
    if i==0:widths[slot]=x.shape[0]


@wp.kernel(enable_backward=False)
def project_future(x:wp.array2d(dtype=float),noise:wp.array2d(dtype=float),horizon:int,
        lower:wp.array2d(dtype=float),upper:wp.array2d(dtype=float),joint_step:float,radius:float):
    i=wp.tid()
    for h in range(1,horizon):
        offset=h*26;previous=(h-1)*26
        center=wp.vec3(x[i,previous],x[i,previous+1],x[i,previous+2])
        delta=wp.vec3(x[i,offset],x[i,offset+1],x[i,offset+2])-center
        if wp.length(delta)>radius:
            change=delta*(radius/wp.length(delta)-1.0)
            for j in range(3):x[i,offset+j]+=change[j];noise[i,offset+j]+=change[j]
        for j in range(6,26):
            old=x[i,offset+j]
            low=wp.max(lower[0,offset+j],x[i,previous+j]-joint_step)
            high=wp.min(upper[0,offset+j],x[i,previous+j]+joint_step)
            x[i,offset+j]=wp.clamp(old,low,high);noise[i,offset+j]+=x[i,offset+j]-old


class WindowSampler(CardSampler):
    def __init__(self,horizon,**kwargs):
        self.horizon=horizon;super().__init__(**kwargs)

    def sample(self,var,scores):
        result=super().sample(var,scores)
        if self.horizon>1 and self.lower is not None:
            wp.launch(project_future,result.batch_size,[result.get(self.var_key).x,self.noise,self.horizon,
                self.lower,self.upper,self.joint_step,self.radius],device=self._device)
        return result


@dataclass(frozen=True)
class WindowSamplerConfig(CardSamplerConfig):
    horizon:int=HORIZON

    def build(self,var,num_candidates,active_dof_mask=None):
        return WindowSampler(**asdict(self),tangent_dim=26*self.horizon,instances=var.batch_size,
            device=var.device,active_dof_mask=active_dof_mask)


class WindowCost(Task):
    def __init__(self,owner,horizon):
        self.owner=owner;self.horizon=horizon;self.device=str(owner.device)
        self.frames=[CardCostTask(owner) for _ in range(horizon)];self.flat_task=self.frames[0]
        self.views={};s=owner.settings;slots=s.iterations+8
        self.counter=wp.zeros(1,dtype=int,device=self.device)
        self.archive_x=wp.zeros((slots*s.population,horizon*26),dtype=float,device=self.device)
        self.archive_cost=wp.full(slots*s.population,wp.inf,dtype=float,device=self.device)
        self.archive_width=wp.zeros(slots,dtype=int,device=self.device)
        self.prefix=wp.zeros((horizon,20),dtype=float,device=self.device)
        self.prefix_weights=wp.zeros(horizon,dtype=float,device=self.device)

    def update(self,contexts):
        for h,(task,c) in enumerate(zip(self.frames,contexts)):
            task.update(c)
            if h:task.data.previous_valid.zero_();task.data.old_valid.zero_()

    def precompute(self,values,need_gradient=False):
        b=values.batch_size;h=self.horizon;n=b*h
        flat=values.get(VarKey.ROBOT).x.reshape((n,26))
        flat_values=VarValues.from_var(VectorVar(flat),VarKey.ROBOT)
        self.flat_task.precompute(flat_values)
        key=(b,flat.ptr)
        if key not in self.views:
            # Strided views keep each frame's variables/poses in one FK batch.
            x=wp.to_torch(flat).reshape(b,h,26)
            poses=wp.to_torch(self.flat_task.states[n].T_world_link).reshape(b,h,-1,7)
            parts,raw,depths=self.flat_task.buffers[n]
            p=wp.to_torch(parts).reshape(b,h,-1);r=wp.to_torch(raw).reshape(b,h);d=wp.to_torch(depths).reshape(b,h,2)
            rows=wp.zeros((b,h),dtype=float,device=self.device);row=wp.to_torch(rows)
            self.views[key]=(rows,[(wp.from_torch(x[:,k]),wp.from_torch(poses[:,k],dtype=wp_vec7),
                wp.from_torch(p[:,k]),wp.from_torch(r[:,k]),wp.from_torch(d[:,k]),wp.from_torch(row[:,k])) for k in range(h)])

    def compute_weighted_cost(self,values,*,out_cost):
        b=values.batch_size;rows,views=self.views[(b,values.get(VarKey.ROBOT).x.ptr)];rows.zero_()
        for h,(x,poses,parts,raw,depths,cost) in enumerate(views):
            wp.launch(card_cost,b,[x,poses,self.frames[h].data,parts,raw,depths,cost],device=self.device)
            if h:
                wp.launch(adjacent_main_cost,b,[x,views[h-1][0],poses,views[h-1][1],
                    self.frames[h].data,self.frames[h-1].data,parts,cost],device=self.device)
        wp.launch(reduce_chunk_cost,b,[rows,views[0][2],BOUNDARY_JOINT_MULTIPLIER,
            values.get(VarKey.ROBOT).x,self.prefix,self.prefix_weights,self.owner.settings.wtheta**2,out_cost],device=self.device)
        wp.launch(save_trajectory,b,[values.get(VarKey.ROBOT).x,out_cost,self.counter,self.archive_x,
            self.archive_cost,self.archive_width,self.owner.settings.population],device=self.device)
        wp.launch(advance,1,[self.counter],device=self.device)


class TrajectoryCEM(RobokitCardSolver):
    """Main's PopulationSolver and Gaussian elite update over a wider variable."""
    def __init__(self,owner,horizon=HORIZON):
        self.owner=owner;self.settings=owner.settings;self.horizon=horizon;s=self.settings;device=str(owner.device)
        self.window_task=WindowCost(owner,horizon);self.task=self.window_task.frames[0]
        self.var=VarValues.from_var(VectorVar(wp.zeros((1,horizon*26),dtype=float,device=device)),VarKey.ROBOT)
        config=WindowSamplerConfig(horizon=horizon,num_draws=s.population,elite_ratio=s.elite_fraction,
            mean_elites=True,temperature=s.temperature,init_std=1.,min_std=1.e-9,gamma=0.,
            noise_decay=s.gamma**(1/s.iterations),sample_mode='mean',seed=s.seed)
        self.solver=CardPopulationSolver([self.window_task],PopulationSolverConfig(stages=[Stage(s.iterations)],
            samplers=[config],cuda_graph_mode='full',use_early_stopping=False),device)
        self.lower=wp.zeros((1,horizon*26),dtype=float,device=device);self.upper=wp.zeros_like(self.lower)
        self.seed_var=VarValues.from_var(VectorVar(wp.zeros((1,horizon*26),dtype=float,device=device)),VarKey.ROBOT)
        self.rotation_reference=None;self.initialized=False;self.warm_stamps=None;self.warm_values=None
        self.inference_delay_ms=0.

    def solve(self,initials,contexts,stamps):
        began=time.perf_counter();o=self.owner;s=self.settings;h=self.horizon;device=str(o.device)
        previous=pack(contexts[0]['previous']) if contexts[0]['previous'] is not None else None
        self.rotation_reference=previous[3:6] if previous is not None else None
        current=np.stack([pack(p,self.rotation_reference) for p in initials])
        center=current.copy()
        for k in range(h):
            prior=previous if k==0 else current[k-1]
            if prior is not None:center[k]=(center[k]+prior)*.5
        warm,warm_mask=aligned_warm_start(stamps,self.warm_stamps,self.warm_values,center)
        prefix_weights=chunk_prefix_weights(stamps,warm_mask,self.inference_delay_ms)
        # Begin the search at the previously optimized predictions at the SAME
        # timestamps, rather than merely scoring them after unrelated draws.
        center[warm_mask]=warm[warm_mask]
        lo=np.tile(np.r_[[-np.inf]*6,o.lower_np[6:]],(h,1)).astype(np.float32)
        hi=np.tile(np.r_[[np.inf]*6,o.upper_np[6:]],(h,1)).astype(np.float32)
        if previous is not None:
            lo[0,:3]=previous[:3]-s.translation_step_m;hi[0,:3]=previous[:3]+s.translation_step_m
            lo[0,6:]=np.maximum(lo[0,6:],previous[6:]-s.joint_step_rad)
            hi[0,6:]=np.minimum(hi[0,6:],previous[6:]+s.joint_step_rad)
        def project(value):
            value=np.clip(value.copy(),lo,hi)
            for k in range(h):
                before=previous if k==0 else value[k-1]
                if before is None:continue
                delta=value[k,:3]-before[:3];length=np.linalg.norm(delta)
                if length>s.translation_step_m:value[k,:3]=before[:3]+delta*(s.translation_step_m/length)
                value[k,6:]=np.clip(value[k,6:],np.maximum(lo[k,6:],before[6:]-s.joint_step_rad),
                    np.minimum(hi[k,6:],before[6:]+s.joint_step_rad))
            return value
        center=project(center);self.window_task.update(contexts)
        wp.copy(self.window_task.prefix,wp.array(np.ascontiguousarray(warm[:,6:]),dtype=float,device=device))
        wp.copy(self.window_task.prefix_weights,wp.array(prefix_weights,dtype=float,device=device))
        wp.copy(self.var.get(VarKey.ROBOT).x,wp.array(center.reshape(1,-1),dtype=float,device=device))
        wp.copy(self.lower,wp.array(lo.reshape(1,-1),dtype=float,device=device));wp.copy(self.upper,wp.array(hi.reshape(1,-1),dtype=float,device=device))
        if not self.initialized:
            mode=self.solver.config.cuda_graph_mode;self.solver.config.cuda_graph_mode='none';self.solver.setup(self.var)
            sampler=self.solver.samplers[0];sampler.lower=self.lower;sampler.upper=self.upper
            sampler.radius=s.translation_step_m;sampler.joint_step=s.joint_step_rad
            self.solver.config.cuda_graph_mode=mode
            std=np.tile([s.translation_sigma_m]*3+[s.rotation_sigma_rad]*3+[s.joint_sigma_rad]*20,h).astype(np.float32)
            wp.copy(sampler.std_reset,wp.array(std[None],dtype=float,device=device));self.solver._warmup_and_capture();self.initialized=True
        task=self.window_task;task.counter.zero_();task.archive_width.zero_();task.archive_cost.fill_(wp.inf)
        wp.synchronize_device(device);prepared=time.perf_counter()
        self.solver.solve(self.var);wp.synchronize_device(device);sampled=time.perf_counter()
        widths=task.archive_width.numpy();costs=task.archive_cost.numpy();xs=task.archive_x.numpy()
        depth=task.flat_task.buffers[s.population*h][2].numpy().reshape(s.population,h,2)
        exceedances=dict(palm_capsule=int((depth[:,:,1]<s.margin_mm).any(1).sum()),
            object_penetration=int((depth[:,:,0]>s.hard_penetration_mm).any(1).sum()),
            population=s.population,scope='last_population_any_window_frame')
        ids=np.concatenate([np.arange(k*s.population,k*s.population+int(n)) for k,n in enumerate(widths) if n])
        seeds=[center]+([np.tile(previous,(h,1))] if previous is not None else [])
        # The projected warm start is already center. Do not evaluate the same
        # trajectory again (each seed otherwise repeats FK and mesh costs).
        seed_costs=[]
        for value in seeds:
            variable=self.seed_var
            wp.copy(variable.get(VarKey.ROBOT).x,wp.array(value.reshape(1,-1),dtype=float,device=device))
            task.precompute(variable);out=wp.zeros(1,dtype=float,device=device)
            task.compute_weighted_cost(variable,out_cost=out);cost=float(out.numpy()[0]);seed_costs.append(cost)
            ids=np.r_[ids,len(costs)];xs=np.vstack((xs,value.reshape(1,-1)));costs=np.r_[costs,cost]
        finite=ids[np.isfinite(costs[ids])&np.isfinite(xs[ids]).all(1)];feasible=bool(len(finite))
        chosen=int(finite[np.argmin(costs[finite])]) if feasible else None
        value=xs[chosen].reshape(h,26) if feasible else project(current)
        if feasible:self.warm_values=value.copy();self.warm_stamps=stamps.copy()
        first=value[0];best=(o.tensor(first[:3]),o.tensor(Rotation.from_rotvec(first[3:6]).as_matrix()),o.tensor(first[6:]))
        elapsed=(time.perf_counter()-began)*1000
        return best,dict(backend='main_robokit_population_solver_window',baseline_commit=BASELINE,
            feasible=feasible,best_cost=float(costs[chosen]) if feasible else None,
            finite_candidate_found=feasible,initial_cost=seed_costs[0] if np.isfinite(seed_costs[0]) else None,
            output_selection='oldest_pose_of_main_card_trajectory' if feasible else 'current_initializer_no_finite_objective',
            initial_feasible=bool(np.isfinite(seed_costs[0])),horizon=h,future_frames=h-1,
            trajectory_timestamps_ms=stamps.tolist(),sampled_candidates=s.population*s.iterations,
            evaluations=int(sum(widths))+len(seeds),iterations=s.iterations,population=s.population,
            elite_count=max(1,int(s.population*s.elite_fraction)),hard_rejected=int((~np.isfinite(costs[ids])).sum()),
            cuda_graph=True,search_center=('same_timestamp_previous_trajectory' if warm_mask.any() else 'main_half_previous_half_initializer'),
            warm_start_frames=int(warm_mask.sum()),
            chunk_continuity=dict(method='rtc_inspired_cem_soft_prefix',prefix_weights=prefix_weights.tolist(),
                overlap_frames=int(warm_mask.sum()),inference_delay_ms=self.inference_delay_ms,
                prefix_joint_cost=float(s.wtheta**2*np.sum(prefix_weights[:,None]*(value[:,6:]-warm[:,6:])**2)),
                extra_fk_batches=0,extra_mesh_queries=0,extra_cem_iterations=0),
            boundary_reference_timestamp_ms=o.previous_timestamp if previous is not None else None,
            boundary_joint_multiplier=BOUNDARY_JOINT_MULTIPLIER,
            boundary_joint_cost=(float(s.wtheta**2*BOUNDARY_JOINT_MULTIPLIER*np.sum((first[6:]-previous[6:])**2)) if previous is not None else 0.),
            previous_evaluated_as_incumbent=previous is not None,previous_incumbent_selected=bool(previous is not None and chosen==ids[-len(seeds)+1]),
            empty_elite_iterations=int(self.solver.samplers[0].empty_iterations.numpy()[0]),
            objective_extension='main_adjacent_pose_costs_with_published_boundary_and_rtc_soft_prefix',extra_velocity_acceleration_cost=False,
            constraint_rejections=dict(palm_capsule=0,object_penetration=0,population=s.population,
                scope='disabled_collision_hard_gates'),collision_threshold_exceedances=exceedances,
            collision_search='robokit_native_mesh_fitted_capsules',collision_output='capsule_and_object_soft_costs_only',
            backend_timing_ms=dict(prepare=(prepared-began)*1000,sample=(sampled-prepared)*1000,total=elapsed))


class FixedLagCardRetarget(TactileRetarget):
    def __init__(self,*args,realtime=True,settings=None,**kwargs):
        self.window=[];self.engine=None;self.live_frame=None
        # Only the live window changes; offline CARD/CART defaults stay intact.
        settings=settings or replace(CardSettings(),wtheta=LIVE_JOINT_SMOOTH_WEIGHT)
        super().__init__(*args,settings=settings,**kwargs)

    def topology(self):
        model=super().topology()
        model.update(algorithm=ALGORITHM,baseline_commit=BASELINE,trajectory=dict(future_frames=FUTURE,
            horizon=HORIZON,observation_policy='actual_contiguous_source_frames',output='oldest_optimized_pose',
            temporal_objective='origin_main_smoothness_and_material_fix',state_reference='current_robot_output',
            joint_smooth_weight=self.settings.wtheta,joint_smooth_coefficient=self.settings.wtheta**2,
            boundary_joint_multiplier=BOUNDARY_JOINT_MULTIPLIER,boundary_joint_averaged_over_horizon=False,
            warm_start_policy='same_timestamp_overlap_only',chunk_continuity='rtc_inspired_cem_soft_prefix',
            prefix_schedule='rtc_exp',inference_delay_source='previous_worker_step_wall_time',post_filter=False))
        model['topology_version']=hashlib.sha256(json.dumps([model['topology_version'],ALGORITHM,model['trajectory']],sort_keys=True).encode()).hexdigest()[:24]
        return model

    def reset(self):
        super().reset();self.window=[];self.live_frame=None
        if self.engine is not None:
            self.engine.warm_values=None;self.engine.warm_stamps=None;self.engine.inference_delay_ms=0.

    def _solve_frame(self,initial,context,incumbents):
        cold=self.engine is None
        if cold:self.engine=TrajectoryCEM(self)
        if context['previous'] is None:self.engine.warm_stamps=None
        contexts=[context];initials=[initial];future_sdf=self.sdf
        # Future IK uses main's unchanged initializer. Restore committed history
        # afterwards, so lookahead does not overwrite the current-frame state.
        saved={k:copy.deepcopy(getattr(self.initializer,k)) for k in
            ('previous_q','previous_points','previous_result','last_report')}
        try:
            for frame in self.window[1:]:
                (v,j,_,_),_,anchors=self.extractor.extract(frame,self.contact_enabled)
                initials.append(self.initializer(j));c=self.context(v,j,anchors);c['previous']=None
                if self.external_sdf is None:
                    future_sdf=copy.copy(future_sdf);future_sdf.range=future_sdf.range.copy()
                    future_sdf.add(anchors,j,float(frame['timestamp_ms']));c['sdf']=future_sdf
                elif frame.get('object_transform') is not None:
                    c['sdf']=copy.copy(self.external_sdf);c['sdf'].transform=np.asarray(frame['object_transform'],float)
                contexts.append(c)
        finally:
            for k,value in saved.items():setattr(self.initializer,k,value)
        best,diagnostic=self.engine.solve(initials,contexts,np.asarray([f['timestamp_ms'] for f in self.window]))
        diagnostic['state_reference_q_rad']=context['previous'][2].cpu().tolist() if context['previous'] is not None else None
        self.robokit_cem=self.engine
        return best,diagnostic,True,cold

    def step(self,frame,contact_enabled=True):
        window=[frame]+list(frame.get('card_future_frames') or [])
        if not valid_window(window):
            self.reset()
            return dict(status='buffering',ready=False,display_available=False,
                buffering_frames=min(len(window),HORIZON),required_frames=HORIZON,solution_feasible=False,
                message=f'CARD 等待当前帧和连续未来{FUTURE}帧；不补造未来观测',algorithm=ALGORITHM)
        calibration=(frame.get('skeleton_runtime_revision'),frame.get('pose_calibration_version'))
        if self.source_calibration is not None and calibration!=self.source_calibration:self.reset()
        if self.last_seen_timestamp is not None and (frame['timestamp_ms']<=self.last_seen_timestamp or
                identity(window[0])[0]!=self.previous_input_stream_epoch):self.reset()
        self.window=window;self.live_frame=window[-1];self.contact_enabled=contact_enabled
        result=super().step(frame,contact_enabled)
        if self.engine is not None:
            self.engine.inference_delay_ms=float(result.get('quality',{}).get('processing_ms',0.))
        result.update(algorithm=ALGORITHM,baseline_commit=BASELINE,postprocessing='main_card_window_no_output_filter',
            live_source_timestamp_ms=window[-1]['timestamp_ms'],buffer_delay_frames=FUTURE,
            buffer_delay_ms=window[-1]['timestamp_ms']-frame['timestamp_ms'])
        return result
