"""Causal CARD continuity from developer commit 98df8ec5 (2026-11-14).

The original Robokit sampler, local source-motion draws, refinement bounds and
history proposals are retained. Current CARD geometry/energy is reused, with
collision soft costs and a diagnostic-only endpoint mesh audit.
"""
from dataclasses import dataclass,replace
import hashlib,json,time
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation
from robokit.opt import PopulationSolver,PopulationSolverConfig,Stage,VarValues,VectorVar
from robokit.opt.var_values import VarKey
from robokit.opt.sampler import GaussianSampler,GaussianSamplerConfig
from .geometry import source_wrist
from .math import CardSettings
from .retarget import TactileRetarget
from .robokit import CardCostTask,RobokitCardSolver as BaseCardSolver,pack,card_cost,PARTS

BASELINE='98df8ec5'
ALGORITHM='card_wuji2_v17_developer_causal_soft_collisions'

@wp.kernel(enable_backward=False)
def multiply(values: wp.array(dtype=float), scale: float):
    i=wp.tid();values[i]=wp.max(values[i]*scale,1.0e-12)


@wp.kernel(enable_backward=False)
def ball_projection(x: wp.array2d(dtype=float), noise: wp.array2d(dtype=float), lower: wp.array2d(dtype=float), upper: wp.array2d(dtype=float), radius: float):
    i=wp.tid();center=wp.vec3((lower[0,0]+upper[0,0])*0.5,(lower[0,1]+upper[0,1])*0.5,(lower[0,2]+upper[0,2])*0.5)
    delta=wp.vec3(x[i,0],x[i,1],x[i,2])-center
    if wp.length(delta)>radius:
        change=delta*(radius/wp.length(delta)-1.0)
        for j in range(3):x[i,j]+=change[j];noise[i,j]+=change[j]


@wp.kernel(enable_backward=False)
def motion_draws(x: wp.array2d(dtype=float),noise: wp.array2d(dtype=float),centers: wp.array2d(dtype=float),
                 motion: wp.array2d(dtype=float),valid: wp.array(dtype=int),lower: wp.array2d(dtype=float),
                 upper: wp.array2d(dtype=float),count: int,scale: float,std: wp.array2d(dtype=float),
                 active: wp.array(dtype=float),seed: int,epoch: wp.array(dtype=int),salt: int):
    i,j=wp.tid()
    if valid[0]!=0 and i>=x.shape[0]-count:
        # Read the same raw normal as Robokit, before global-centre clamping.
        # Clipped global noise would wrongly truncate an interior local cloud
        # whenever the global CEM mean sits on a native joint limit.
        state=wp.rand_init(seed+epoch[0]*7919,(salt*1000003+i)*x.shape[1]+j)
        value=centers[0,j]
        if active[j]!=0.0:value=wp.clamp(motion[0,j]+scale*std[0,j]*wp.randn(state),lower[0,j],upper[0,j])
        x[i,j]=value;noise[i,j]=value-centers[0,j]


class CardSampler(GaussianSampler):
    def _move(self, scores, out_delta):
        # Reuse Robokit's sorting, elite masking, annealing, weights and update
        # kernels, adding paper eta after its elite-standard-deviation estimate.
        super()._move(scores,out_delta)
        if self.temperature!=1.0:
            wp.launch(multiply,self.temperatures.shape,[self.temperatures,self.temperature],device=self._device)
            block=1 if self._device.is_cpu else 256
            wp.launch(self._weights_kernel,(self.instances,block),[self._elite_costs,self.temperatures,self.epsilon,block],[self.weights],block_dim=block,device=self._device)
            wp.launch(self._update_kernel,(self.instances,self.tangent_dim,block),
                [self.weights,self.noise,self._elite_costs,self.best_indices,0.0,self.min_std,self.learning_rate,self.epsilon,block],
                [self.std,out_delta],block_dim=block,device=self._device)

    def sample(self, var, scores):
        result=super().sample(var,scores)
        if self.lower is not None:
            if getattr(self,'motion_center',None) is not None:
                wp.launch(motion_draws,result.get(self.var_key).x.shape,
                    [result.get(self.var_key).x,self.noise,self._centers.get(self.var_key).x,
                     self.motion_center,self.motion_valid,self.lower,self.upper,self.motion_count,.15,
                     self.std,self.active_dof_mask,self.seed,self.epoch,self._round-1],device=self._device)
            wp.launch(ball_projection,result.batch_size,[result.get(self.var_key).x,self.noise,self.lower,self.upper,self.radius],device=self._device)
        return result


@dataclass(frozen=True)
class CardSamplerConfig(GaussianSamplerConfig):
    def build(self,var,num_candidates,active_dof_mask=None):
        from dataclasses import asdict
        return CardSampler(**asdict(self),tangent_dim=26,instances=var.batch_size,device=var.device,active_dof_mask=active_dof_mask)



class OnlineCardSolver(BaseCardSolver):
    def __init__(self,owner):
        self.owner=owner;self.settings=owner.settings;s=self.settings;device=str(owner.device)
        self.task=CardCostTask(owner);self.var=VarValues.from_var(VectorVar(wp.zeros((1,26),dtype=float,device=device)),VarKey.ROBOT)
        config=CardSamplerConfig(num_draws=s.population,elite_ratio=s.elite_fraction,mean_elites=True,temperature=s.temperature,
            init_std=1.,min_std=1.0e-9,gamma=0.,noise_decay=s.gamma**(1/s.iterations),sample_mode='mean',seed=s.seed)
        self.solver=PopulationSolver([self.task],PopulationSolverConfig(stages=[Stage(s.iterations)],samplers=[config],cuda_graph_mode='full',use_early_stopping=False),device)
        # Preallocate both widths before capture. Context buffers are filled first.
        self.lower=wp.zeros((1,26),dtype=float,device=device);self.upper=wp.zeros((1,26),dtype=float,device=device)
        self.motion_center=wp.zeros((1,26),dtype=float,device=device);self.motion_valid=wp.zeros(1,dtype=int,device=device)
        self.initialized=False;self.rotation_reference=None


    def solve(self,initial,c):
        began=time.perf_counter();o=self.owner;s=self.settings;center=pack(initial);device=str(o.device)
        self.task.update(c)
        wp.copy(self.var.get(VarKey.ROBOT).x,wp.array(center[None],dtype=float,device=device))
        lo=np.r_[center[:3]-s.translation_step_m,[-1.0e6]*3,np.maximum(o.lower_np[6:],center[6:]-s.joint_step_rad)]
        hi=np.r_[center[:3]+s.translation_step_m,[1.0e6]*3,np.minimum(o.upper_np[6:],center[6:]+s.joint_step_rad)]
        wp.copy(self.lower,wp.array(lo[None],dtype=float,device=device));wp.copy(self.upper,wp.array(hi[None],dtype=float,device=device))
        motion=None
        if c.get('motion_joint') is not None and s.wmotion_joint>0:
            t,r,_=c['previous']
            if c.get('previous_source_wrist') is not None:
                origin,rotation_now=source_wrist(c['joints']);old_origin,old_rotation=c['previous_source_wrist'];delta=o.tensor(rotation_now@old_rotation.T)
                t=o.tensor(origin)+delta@(t-o.tensor(old_origin));r=delta@r
            motion=(t,r,c['motion_joint'])
            wp.copy(self.motion_center,wp.array(pack(motion,center[3:6])[None],dtype=float,device=device))
        self.motion_valid.fill_(int(motion is not None))
        if not self.initialized:
            # Setup must see bounds on the very first draw, before graph capture.
            original=self.solver.config.cuda_graph_mode;self.solver.config.cuda_graph_mode='none';self.solver.setup(self.var)
            sampler=self.solver.samplers[0];sampler.lower=self.lower;sampler.upper=self.upper;sampler.radius=s.translation_step_m
            sampler.motion_center=self.motion_center;sampler.motion_valid=self.motion_valid
            sampler.motion_count=min(s.population-1,int(s.population*s.motion_sample_fraction))
            self.solver.config.cuda_graph_mode=original
            std=np.array([s.translation_sigma_m]*3+[s.rotation_sigma_rad]*3+[s.joint_sigma_rad]*20,np.float32)
            wp.copy(sampler.std_reset,wp.array(std[None],dtype=float,device=device))
            self.solver._warmup_and_capture();self.initialized=True
        self.task.counter.zero_();self.task.archive_width.zero_();self.task.archive_cost.fill_(wp.inf)
        wp.synchronize_device(device);prepared_at=time.perf_counter()
        self.solver.solve(self.var);wp.synchronize_device(device)
        sampled_at=time.perf_counter()
        widths=self.task.archive_width.numpy();costs=self.task.archive_cost.numpy();xs=self.task.archive_x.numpy()
        ids=np.concatenate([np.arange(i*s.population,i*s.population+int(n)) for i,n in enumerate(widths) if n])
        # The previous solution is a candidate, never the sampling centre or an
        # unconditional display fallback. Re-score it with today's anchors,
        # field and temporal reference, within today's Robokit refinement box.
        # Without this incumbent, tiny changes to smoothness can send identical
        # source frames into different CEM basins and return a worse pose.
        incumbent_ids={};incumbent_costs={};proposals=[]
        if c['previous'] is not None:
            proposals.append(('previous',c['previous']))
            if c.get('previous_source_wrist') is not None:
                origin,rotation_now=source_wrist(c['joints']);old_origin,old_rotation=c['previous_source_wrist']
                delta=rotation_now@old_rotation.T;t,r,j=c['previous']
                transported=(o.tensor(origin)+o.tensor(delta)@(t-o.tensor(old_origin)),o.tensor(delta)@r,j)
                if not np.allclose(pack(transported),pack(c['previous']),atol=1e-8,rtol=0):
                    proposals.append(('source_rigid_motion',transported))
            if c.get('motion_joint') is not None:
                t,r,_=proposals[-1][1]
                proposal=(t,r,c['motion_joint'])
                if not any(np.allclose(pack(proposal),pack(p),atol=1e-8,rtol=0) for _,p in proposals):
                    proposals.append(('source_joint_motion',proposal))
        for kind,proposal in proposals:
            old=pack(proposal,center[3:6])
            if (np.isfinite(old).all() and np.all(old[6:]>=lo[6:]) and np.all(old[6:]<=hi[6:])
                    and np.linalg.norm(old[:3]-center[:3])<=s.translation_step_m+1e-7):
                idx=len(costs);cost=self.parts(proposal,total=True)
                incumbent_ids[idx]=kind;incumbent_costs[kind]=cost
                xs=np.vstack((xs,old));costs=np.r_[costs,cost];ids=np.r_[ids,idx]
        finite=ids[np.isfinite(xs[ids]).all(1)&np.isfinite(costs[ids])]
        if not len(finite):raise ValueError('Robokit CARD produced no finite current candidate')
        chosen=int(finite[np.argmin(costs[finite])]);valid_found=True;fallback_violation=0.0
        value=xs[chosen];best=(o.tensor(value[:3]),o.tensor(Rotation.from_rotvec(value[3:6]).as_matrix()),o.tensor(value[6:]))
        wp.synchronize_device(device);verified_at=time.perf_counter()
        diagnostic=dict(baseline_commit=BASELINE,continuity_policy='developer_v9_source_motion',backend='robokit_population_solver_warp_card',feasible=valid_found,best_cost=float(costs[chosen]) if valid_found else None,
            finite_candidate_found=True,fallback_cost=float(costs[chosen]),fallback_violation=fallback_violation,
            output_selection='best_feasible_current_candidate' if valid_found else 'least_violation_current_candidate',
            search_center='current_frame_retargeting',previous_used_as_search_center=False,previous_evaluated_as_incumbent=bool(incumbent_ids),
            previous_incumbent_selected=chosen in incumbent_ids,previous_incumbent_kind=incumbent_ids.get(chosen),
            previous_incumbent_cost=incumbent_costs.get('previous'),temporal_candidate_costs=incumbent_costs,
            sampled_candidates=s.population*s.iterations,evaluations=int(sum(widths))+len(incumbent_ids),iterations=s.iterations,elite_count=int(s.population*s.elite_fraction),
            source_motion_samples_per_round=min(s.population-1,int(s.population*s.motion_sample_fraction)) if motion is not None else 0,
            coarse_verified_candidates=0,coarse_verification_rejected=0,
            dense_verified_candidates=0,dense_verification_rejected=0,hard_rejected=int((~np.isfinite(costs[ids])).sum()),
            seed=s.seed,final_sigma_ratio=s.gamma,cuda_graph=True,collision_search='robokit_native_mesh_fitted_capsules',
            collision_output='soft_collision_costs_only',validation_limit=None,
            output_selection_scope='all_scored_candidates',infeasibility_proven=False,
            fallback_violation_units='native_self_depth_excess_mm',
            initial_cost=float(costs[0]) if np.isfinite(costs[0]) else None,initial_feasible=bool(np.isfinite(costs[0])),initial_incumbents=0,empty_elite_restarts=0,feasibility_recovery_iterations=0,
            backend_timing_ms=dict(prepare=(prepared_at-began)*1000,sample=(sampled_at-prepared_at)*1000,
                endpoint_verification=(verified_at-sampled_at)*1000,total=(verified_at-began)*1000))
        return best,diagnostic


    def parts(self,initial,*,total=False):
        """Evaluate the same search cost at one output, without changing context."""
        device=str(self.owner.device)
        var=VarValues.from_var(VectorVar(wp.array(pack(initial)[None],dtype=float,device=device)),VarKey.ROBOT)
        self.task.precompute(var);parts,raw,depths=self.task.buffers[1]
        cost=wp.zeros(1,dtype=float,device=device)
        wp.launch(card_cost,1,[var.get(VarKey.ROBOT).x,self.task.states[1].T_world_link.reshape((1,self.task.robot.spec.num_links)),self.task.data,parts,raw,depths,cost],device=device)
        return float(cost.numpy()[0]) if total else dict(zip(PARTS,parts.numpy()[0].tolist()))



class OnlineCardRetarget(TactileRetarget):
    def __init__(self,*args,settings=None,**kwargs):
        settings=settings or replace(CardSettings(),wmotion_joint=300.,motion_sample_fraction=.125)
        super().__init__(*args,settings=settings,**kwargs)

    def topology(self):
        model=super().topology()
        online=dict(baseline_commit=BASELINE,future_frames=0,
            state_reference='previous_solution_plus_current_IK_motion',
            source_motion_samples_per_round=int(self.settings.population*self.settings.motion_sample_fraction),
            post_filter=False)
        model.update(algorithm=ALGORITHM,online=online)
        model['topology_version']=hashlib.sha256(json.dumps([model['topology_version'],ALGORITHM,online],sort_keys=True).encode()).hexdigest()[:24]
        return model

    def _solve_frame(self,initial,context,incumbents):
        if self.settings.cem_backend!='robokit' or self.settings.experimental_self_contact or self.settings.contact_direction_model!='card_alignment':
            return super()._solve_frame(initial,context,incumbents)
        cold=self.robokit_cem is None
        if cold:self.robokit_cem=OnlineCardSolver(self)
        best,diagnostic=self.robokit_cem.solve(initial,context)
        return best,diagnostic,True,cold

    def step(self,frame,contact_enabled=True):
        result=super().step(frame,contact_enabled)
        result.update(algorithm=ALGORITHM,baseline_commit=BASELINE,buffer_delay_frames=0,buffer_delay_ms=0.,
            postprocessing='causal_source_motion_no_output_filter')
        return result
