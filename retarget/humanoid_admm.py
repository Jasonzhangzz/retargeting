"""Contact/action ADMM with MJWarp prediction and execution."""
from types import SimpleNamespace
import time
import mujoco
import numpy as np
import warp as wp
from retarget.humanoid_mjwarp import MJWarpHumanoid
from retarget.humanoid_terms import BALANCE_PARTS, stack_host
from retarget.fa.adamw import shift_knots

class ProjectionADMM:
    """Joint hand/foot projection followed by the selected action optimizer."""
    def __init__(self,problem):
        self.problem=problem
        s=problem.settings
        if s.cost_mode=='spider' and s.action_solver!='adamw':
            raise ValueError('SPIDER contact/action mode uses the shared AdamW solver')
        if s.action_solver not in ('adamw','adamw-ls','mppi','acados-gn','sample-lm'):
            raise ValueError('Unknown humanoid action solver: '+s.action_solver)
        if s.action_solver in ('mppi','acados-gn','sample-lm') and s.adamw_persist:
            raise ValueError('AdamW moment persistence applies only to AdamW solvers')
        self.backend=MJWarpHumanoid(problem)
        if s.action_solver=='adamw':
            self.action_optimizer=self.backend.adamw
            self.action_solver_name='retarget.fa.adamw.AdamW'
        elif s.action_solver=='adamw-ls':
            from retarget.humanoid_optimizers import LineSearchAdamW
            self.action_optimizer=LineSearchAdamW(learning_rate=s.adamw_lr,weight_decay=s.adamw_weight_decay,
                persist=s.adamw_persist,device=self.backend.device,max_trials=s.line_search_trials,
                contraction=s.line_search_contraction,armijo=s.line_search_armijo)
            self.action_solver_name='retarget.humanoid_optimizers.LineSearchAdamW'
        elif s.action_solver=='mppi':
            from retarget.humanoid_optimizers import MPPI
            self.action_optimizer=MPPI(samples=s.mppi_samples,noise_std=s.mppi_noise_std,
                temperature=s.mppi_temperature,control_scale=s.control_scale,seed=s.seed,device=self.backend.device,
                update_rule=s.mppi_update)
            self.action_solver_name='retarget.humanoid_optimizers.MPPI'
        self.gradient_source='sampling_no_derivatives' if s.action_solver=='mppi' else 'MJWarp_central_finite_differences'
        if s.action_solver=='acados-gn':
            from retarget.humanoid_acados import AcadosGN
            self.action_optimizer=AcadosGN(self.backend)
            self.action_solver_name='retarget.humanoid_acados.AcadosGN'
            self.gradient_source='MJWarp_batched_finite_difference_residual_Jacobian'
        self.common_rollout=None
        if s.action_solver=='sample-lm':
            from retarget.humanoid_lm_oracle import MJWarpSampleLM
            self.action_optimizer=MJWarpSampleLM(self.backend)
            self.action_solver_name='retarget.humanoid_lm_oracle.MJWarpSampleLM'
            self.gradient_source='sampled_residual_LM_model' if s.sample_lm_reuse_samples else 'MJWarp_residual_subspace_finite_differences'
            self.common_rollout=self.action_optimizer.common_rollout
        if s.action_solver=='mppi' and s.mppi_single_rollout:
            from retarget.humanoid_common_rollout import CommonRollout
            self.common_rollout=CommonRollout(self.backend)
            self.action_optimizer.common_rollout=self.common_rollout
        self.window=0
        self.warm=None
        self.z_dense=self.dual_dense=self.force_dense=None
        self.floor_force=self.floor_torque=None
        self.previous_first=None
        self.spider_contacts=None
        if s.cost_mode=='spider':
            from retarget.humanoid_spider_contacts import SpiderContacts
            self.spider_contacts=SpiderContacts(self.backend)

    def _assign_contacts(self,z,dual,force,floor_force,floor_torque):
        b=self.backend
        for name,value in [('z',z),('dual',dual),('forces',force),('floor_force',floor_force),('floor_torque',floor_torque)]:
            getattr(b.targets,name).assign(np.asarray(value,np.float32))
        b.dirty=True

    def solve(self,state,indices,previous_action):
        from retarget.humanoid_contacts import update,project
        p,b,s=self.problem,self.backend,self.problem.settings
        first=int(indices[0])
        started=time.perf_counter()
        b.prepare(state,first,previous_action)
        shape=(b.dense_steps+1,b.worlds,4,3)
        z=np.broadcast_to(b.anchors[:,None],shape).copy()
        dual=np.zeros(shape);force=np.zeros(shape)
        floor_force=np.zeros(shape[:2]+(3,));floor_torque=np.zeros_like(floor_force)
        if self.z_dense is not None:
            shift=(first-self.previous_first)*s.substeps
            if 0 < shift < len(z):
                for new,old in [(z,self.z_dense),(dual,self.dual_dense),(force,self.force_dense),(floor_force,self.floor_force),(floor_torque,self.floor_torque)]:
                    new[:-shift]=old[shift:]
        mask=np.broadcast_to(p.mask[b.reference_indices][:,None,:,None],shape)
        dual*=mask;force*=mask;dual[0]=0.
        self._assign_contacts(z,dual,force,floor_force,floor_torque)
        b.targets.penalty.assign(np.array([s.rho,s.anchor_weight],np.float32))
        variables=np.zeros(b.variables.shape)
        if self.warm is not None:
            variables[:]=shift_knots(self.warm,b.knots,(first-self.previous_first)/s.knot_stride)
        rng=np.random.default_rng(s.seed+self.window)
        if b.worlds > 1 and s.action_solver not in ('mppi','acados-gn','sample-lm'):
            variables[1:]+=rng.normal(0.,s.multistart_noise_std/s.control_scale,variables[1:].shape)
        b.forward_calls=b.backward_calls=0
        if self.common_rollout is not None:
            import torch
            self.common_rollout.forward_calls=0
            self.common_rollout.begin()
            self.common_rollout.export(torch.as_tensor(variables,device=str(b.device),dtype=torch.float64))
        history=[]
        for iteration in range(s.admm_iterations):
            tick=time.perf_counter()
            before=b.evaluate(variables,False)[0]
            snapshot=b.snapshots()
            contact_started=time.perf_counter()
            if self.spider_contacts is not None:
                candidate,force_candidate,floor_candidate,torque_candidate,contact_diag=self.spider_contacts.update(snapshot,z,dual)
            else:
                candidate,force_candidate,floor_candidate,torque_candidate,contact_diag=update(b,snapshot,z,dual,rng)
            old_z=z.copy()
            self._assign_contacts(candidate,dual,force_candidate,floor_candidate,torque_candidate)
            after=b.evaluate(variables,False)[0]
            # Full global objective safeguard, including all tracking/balance
            # terms, not only the reduced force-QP's local merit.
            accepted=np.isfinite(after)&(after<=before+1e-6)
            if self.spider_contacts is not None:
                # Same as run.py: contact LM selects by prospective task
                # feasibility, then action solves the new augmented problem.
                accepted=np.isfinite(after)
            z=np.where(accepted[None,:,None,None],candidate,z)
            force=np.where(accepted[None,:,None,None],force_candidate,force)
            floor_force=np.where(accepted[None,:,None],floor_candidate,floor_force)
            floor_torque=np.where(accepted[None,:,None],torque_candidate,floor_torque)
            contact_after=np.where(accepted,after,before)
            if s.action_solver in ('mppi','acados-gn','sample-lm'):
                # Weighted controls must share ONE fixed ADMM objective.
                contact_selected=int(np.argmin(np.where(accepted,after,before)))
                for value in (z,dual,force,floor_force,floor_torque):
                    value[:]=value[:,contact_selected,None].copy()
                contact_after=np.full(b.worlds,contact_after[contact_selected])
            self._assign_contacts(z,dual,force,floor_force,floor_torque)
            contact_seconds=time.perf_counter()-contact_started
            action_before=b.evaluate(variables,False)[0]
            action_started=time.perf_counter()
            variables,_,adam_diag=self.action_optimizer.update(variables,b.evaluate,
                iterations=s.action_iterations,horizon=b.knots,
                advance=(first-self.previous_first)/s.knot_stride if self.previous_first is not None else 0.,
                continue_window=iteration>0,evaluate_on_device=True)
            if s.action_solver in ('adamw','adamw-ls'):
                adam_diag.update(finite_difference_gradients=True,gradient_source=self.gradient_source)
            action_seconds=time.perf_counter()-action_started
            snapshot=b.snapshots()
            errors=(snapshot['x']-z)*mask
            errors[0]=0.
            delta=(z-old_z)*mask;delta[0]=0.
            primal=np.linalg.norm(errors,axis=-1).max(axis=(0,2))
            dual_residual=s.rho*np.linalg.norm(delta,axis=-1).max(axis=(0,2))
            action_after=b.evaluate(variables,False)[0]
            b.diagnose()
            costs=stack_host([stage.job.terms for stage in b.stages]).sum(axis=0)
            balance_costs=stack_host([stage.job.balance_parts for stage in b.stages]).sum(axis=0)
            dual+=errors
            # Same bounded scaled-dual safeguard as run.py's _dual_block.
            limit=4.*np.maximum(np.array([s.contact_radius]*2+[s.foot_contact_radius]*2),1e-6)[:,None]
            dual*=np.minimum(1.,limit/np.maximum(np.linalg.norm(dual,axis=-1,keepdims=True),1e-12))
            self._assign_contacts(z,dual,force,floor_force,floor_torque)
            history.append(dict(iteration=iteration,update_order=['hands_and_feet_z','action','dual'],
                contact_cost_before=before.tolist(),contact_cost_after=contact_after.tolist(),
                contact_update_accepted=accepted.tolist(),contact=contact_diag,
                action_cost_before=action_before.tolist(),action_cost_after=action_after.tolist(),
                costs_by_world=costs.tolist(),primal_by_world=primal.tolist(),dual_by_world=dual_residual.tolist(),
                balance_parts_by_world=balance_costs.tolist(),
                executed_support_ratio_by_world=snapshot['realized_wrench'][:s.substeps,:,2].mean(axis=0).tolist(),
                contact_seconds=contact_seconds,action_seconds=action_seconds,
                action_optimizer=adam_diag,
                **({'adamw':adam_diag} if s.action_solver in ('adamw','adamw-ls') else {s.action_solver:adam_diag}),
                action_solver=self.action_solver_name,
                gradient_source=self.gradient_source,finite_difference_gradients=(s.action_solver in ('adamw','adamw-ls','acados-gn') or
                    (s.action_solver=='sample-lm' and (not s.sample_lm_reuse_samples or s.action_iterations>1))),
                seconds=time.perf_counter()-tick))
        # run.py ranks the SPIDER mode by its physical task after removing
        # ADMM penalties. Legacy mode retains its primal-feasibility safeguard.
        b.targets.dual.zero_()
        b.targets.penalty.assign(np.array([0.,0.] if s.cost_mode=='spider' else [s.rho,s.anchor_weight],np.float32));b.dirty=True
        selection_cost=b.evaluate(variables,False)[0]
        if not np.isfinite(selection_cost).any():
            raise FloatingPointError('All MJWarp humanoid action candidates are invalid')
        snapshot=b.snapshots()
        residual=(snapshot['x']-z)*mask/s.contact_scale;residual[0]=0.
        normalizer=np.ones(b.dense_steps+1)/b.dense_steps;normalizer[-1]*=s.terminal_weight
        primal_cost=(0. if s.cost_mode=='spider' else s.rho)*np.einsum('n,nwcr->w',normalizer,residual**2)
        physical=selection_cost-primal_cost
        selected=int(np.argmin(selection_cost))
        b.targets.dual.assign(dual.astype(np.float32));b.dirty=True
        sample=np.arange(s.substeps,b.dense_steps+1,s.substeps)
        qs=stack_host([b.steps[i-1].recorded.result.qpos for i in sample])[:,selected]
        vs=stack_host([b.steps[i-1].recorded.result.qvel for i in sample])[:,selected]
        controls=stack_host(b.controls)[:,selected]
        next_state=mujoco.MjData(p.model)
        next_state.qpos[:],next_state.qvel[:]=qs[0],vs[0]
        next_state.ctrl[:]=controls[s.substeps-1]
        next_state.time=state.time+p.ref_dt
        mujoco.mj_forward(p.model,next_state)
        self.actions=controls[sample-1]
        self.z,self.dual=z[sample,selected],dual[sample,selected]
        self.contact_forces=force[sample,selected]
        self.warm=variables[selected].copy()
        # Each new window starts from the selected physical candidate, as in
        # run.py, rather than inheriting unrelated starts' duals/forces.
        self.z_dense=np.broadcast_to(z[:,selected,None],z.shape).copy()
        self.dual_dense=np.broadcast_to(dual[:,selected,None],dual.shape).copy()
        self.force_dense=np.broadcast_to(force[:,selected,None],force.shape).copy()
        self.floor_force=np.broadcast_to(floor_force[:,selected,None],floor_force.shape).copy()
        self.floor_torque=np.broadcast_to(floor_torque[:,selected,None],floor_torque.shape).copy()
        if s.adamw_persist:
            self.action_optimizer.retain_candidate_memory(selected)
        self.backend.accept_execution(selected)
        next_state.qacc_warmstart[:]=self.backend.execution_warmstart
        self.previous_first=first;self.window+=1
        for row in history:
            row['primal_max_m']=row['primal_by_world'][selected]
            row['dual_max_rho_m']=row['dual_by_world'][selected]
            row['converged']=bool(row['primal_max_m']<s.primal_tolerance and row['dual_max_rho_m']<s.dual_tolerance)
            row['action_cost_before']=row['action_cost_before'][selected]
            row['action_cost_after']=row['action_cost_after'][selected]
            row['contact_cost_before']=row['contact_cost_before'][selected]
            row['contact_cost_after']=row['contact_cost_after'][selected]
            row['costs']=dict(zip(b.term_names,row['costs_by_world'][selected]))
            row['balance_parts']=dict(zip(BALANCE_PARTS,row['balance_parts_by_world'][selected]))
            row['executed_support_ratio']=row['executed_support_ratio_by_world'][selected]
            row['executed_root_height_m']=float(qs[0,2])
            row['executed_torso_tilt_deg']=float(np.degrees(np.arccos(np.clip(next_state.xmat[p.torso].reshape(3,3)[:,2]@b.rotations_ref[s.substeps,p.torso,:,2],-1.,1.))))
            row.update(selected=selected,physical_selection_cost=float(physical[selected]),
                forward_calls=b.forward_calls,backward_calls=b.backward_calls,
                selection_policy='spider_pose_without_ADMM' if s.cost_mode=='spider' else 'physical_plus_primal_contact_penalty_without_dual',
                selection_cost=float(selection_cost[selected]),
                physical_cost_by_world=physical.tolist(),primal_contact_cost_by_world=primal_cost.tolist(),selection_cost_by_world=selection_cost.tolist(),
                rollout_cache_hits=b.rollout_cache.hits if b.rollout_cache else 0,
                common_forward_calls=self.common_rollout.forward_calls if self.common_rollout else 0,
                action_variables=int(variables.shape[1]),horizon_seconds=b.dense_steps*b.model.dt)
        return dict(qpos=qs,qvel=vs,contacts=snapshot['x'][sample,selected],
                    next_state=next_state,controls=self.actions,controls_dense=controls[:s.substeps],
                    realized_wrench_dense=snapshot['realized_wrench'][:s.substeps,selected],
                    solve_seconds=time.perf_counter()-started),history
