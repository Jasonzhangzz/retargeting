"""Condensed, globalized MJWarp Gauss--Newton SQP with an acados QP solve.

This is intentionally NOT a sparse stage-dynamics RTI implementation. The
MJWarp rollout is condensed into knot controls. Batched finite differences
of its elementwise residuals form a PSD GN model; acados SQP_RTI solves that
quadratic model, and exact MJWarp rollouts globalize the proposed step.
No simplified robot dynamics, state resets, or objective terms are substituted.
"""
from copy import copy
from dataclasses import replace
from functools import cache
import time

import numpy as np
import torch
import warp as wp

from retarget.warp_arrays import device_copy
from retarget.humanoid_common_rollout import copy_kernel
from retarget.humanoid_residuals import Residuals


def knot_step_bounds(nominal,control_scale,trust_radius):
    """Trust bounds on raw knots; physical limits act AFTER interpolation.

    Bounding raw endpoints by actuator limits would change the original
    feasible control trajectories (e.g. reaching saturation before the end
    of an interval). Keep precisely the original MJWarp control clamp.
    """
    bound=np.full_like(nominal,trust_radius/control_scale,dtype=float)
    return -bound,bound


@cache
def target_kernel(dtype):
    @wp.kernel(enable_backward=False,module='unique')
    def expand(source:wp.array(dtype=dtype),source_worlds:int,worlds:int,width:int,target:wp.array(dtype=dtype)):
        i=wp.tid();stage=i//(worlds*width)
        target[i]=source[stage*source_worlds*width+i%width]
    return expand


@cache
def selected_kernel(dtype):
    @wp.kernel(enable_backward=False,module='unique')
    def broadcast(values:wp.array(dtype=dtype),selected:wp.array(dtype=int),width:int):
        i=wp.tid()
        # The source row is never written, so the in-place gather is safe.
        if i//width!=selected[0]:values[i]=values[selected[0]*width+i%width]
    return broadcast


class AcadosQuadratic:
    """Use acados/HPIPM for a bounded SPD condensed control QP."""
    def __init__(self,dimension):
        from retarget.support.acados import _load_acados,_set_acados_code_export_directory
        from retarget.support.acados_env import acados_cache_directory,build_cached_solver
        import casadi as cs
        Model,Ocp,_=_load_acados();d=dimension
        m=Model();m.name=f'humanoid_condensed_gn_v1_u{d}'
        m.x=cs.SX.sym('x',1);m.u=cs.SX.sym('u',d)
        m.disc_dyn_expr=0*m.x
        ocp=Ocp();ocp.model=m
        ocp.cost.cost_type=ocp.cost.cost_type_e='LINEAR_LS'
        ocp.cost.Vx=np.zeros((d,1));ocp.cost.Vu=np.eye(d)
        ocp.cost.W=np.eye(d);ocp.cost.yref=np.zeros(d)
        ocp.cost.Vx_e=np.ones((1,1));ocp.cost.W_e=np.ones((1,1));ocp.cost.yref_e=np.zeros(1)
        ocp.constraints.x0=np.zeros(1)
        ocp.constraints.idxbu=np.arange(d);ocp.constraints.lbu=-np.ones(d);ocp.constraints.ubu=np.ones(d)
        o=ocp.solver_options;o.N_horizon=1;o.tf=1.
        o.integrator_type='DISCRETE';o.nlp_solver_type='SQP_RTI'
        o.qp_solver='FULL_CONDENSING_HPIPM';o.hessian_approx='GAUSS_NEWTON'
        o.qp_solver_iter_max=100;o.print_level=0
        o.qp_solver_tol_stat=o.qp_solver_tol_eq=o.qp_solver_tol_ineq=o.qp_solver_tol_comp=1e-7
        o.cost_scaling=np.ones(2)
        directory=acados_cache_directory(m.name);directory.mkdir(parents=True,exist_ok=True)
        _set_acados_code_export_directory(ocp,str(directory))
        self.solver=build_cached_solver(ocp,str(directory/(m.name+'.json')),
                                       str(directory/('libacados_ocp_solver_'+m.name+'.so')))
        self.dimension=d

    def solve(self,H,g,lower,upper):
        from scipy.linalg import cho_factor,cho_solve
        s=self.solver
        yref=-cho_solve(cho_factor(H,check_finite=False),g,check_finite=False)
        s.cost_set(0,'W',H,api='new');s.cost_set(0,'yref',yref)
        s.constraints_set(0,'lbu',lower);s.constraints_set(0,'ubu',upper)
        s.set(0,'u',np.zeros(self.dimension));s.set(0,'x',np.zeros(1));s.set(1,'x',np.zeros(1))
        status=s.solve()
        return s.get(0,'u'),dict(status=int(status),qp_status=int(s.get_stats('qp_stat')[-1]),
                               qp_iterations=int(s.get_stats('qp_iter')[-1]),
                               acados_seconds=float(s.get_stats('time_tot')))


class MJWarpLinearization:
    def __init__(self,parent,difference):
        from retarget.humanoid_mjwarp import MJWarpHumanoid
        self.parent=parent;self.d=int(parent.variables.shape[1]);self.central=difference=='central'
        problem=copy(parent.problem)
        problem.settings=replace(parent.settings,multistarts=1+self.d*(2 if self.central else 1),trajectory_cache=False)
        self.backend=MJWarpHumanoid(problem);self.residuals=Residuals(self.backend);self.sync_graph=None
        self.primary_residuals=Residuals(parent)

    def sync(self):
        p,c=self.parent,self.backend
        if (p.dense_control_base is not None)!=(c.dense_control_base is not None):
            c.use_dense_control_base(None if p.dense_control_base is None else np.zeros(p.dense_control_base.shape,np.float32))
            self.sync_graph=None
        def launch():
            if p.dense_control_base is not None:
                device_copy(c.dense_control_base,p.dense_control_base)
                device_copy(c.dense_control_boundary,p.dense_control_boundary)
            for name in ('qpos','qvel','ctrl','position','rotation','com','wrench','object_wrench','hand_position','mask','local','anchor','stance','penalty'):
                device_copy(getattr(c.targets,name),getattr(p.targets,name))
            for name in ('z','dual','forces','floor_point','floor_force','floor_torque'):
                src,dst=getattr(p.targets,name),getattr(c.targets,name)
                width=dst.size//(dst.shape[0]*c.worlds)
                wp.launch(target_kernel(src.dtype),dst.size,inputs=[src.flatten(),p.worlds,c.worlds,width,dst.flatten()],device=p.device)
            for name in ('q0','v0','warmstart0','previous_controls'):
                src,dst=getattr(p,name),getattr(c,name)
                wp.launch(copy_kernel(src.dtype)[0],dst.size,inputs=[src.flatten(),src.size,dst.flatten()],device=p.device)
            for name in ('previous','knot_reference'):device_copy(getattr(c,name),getattr(p,name))
            device_copy(c.model.weights,p.model.weights)
            src,dst=p.steps[0].inputs.time,c.steps[0].inputs.time
            wp.launch(copy_kernel(src.dtype)[0],dst.size,inputs=[src.flatten(),1,dst.flatten()],device=p.device)
        if self.sync_graph is None:
            launch()
            with wp.ScopedCapture(device=p.device) as capture:launch()
            self.sync_graph=capture.graph
        wp.capture_launch(self.sync_graph)
        c._cached=None;c._snapshot_cache=None;c.dirty=True

    def linearize(self,x,epsilon):
        p,c=self.parent,self.backend
        caller=torch.cuda.current_stream(str(p.device))
        stream=torch.cuda.ExternalStream(wp.get_stream(p.device).cuda_stream,device=str(p.device))
        stream.wait_stream(caller)
        with torch.cuda.stream(stream):self.sync()
        caller.wait_stream(stream)
        batch=x[0].expand(c.worlds,-1).clone()
        index=torch.arange(self.d,device=x.device)
        batch[1+index,index]+=epsilon
        if self.central:batch[1+self.d+index,index]-=epsilon
        costs,_=c.evaluate(batch,False)
        values=self.residuals.evaluate().permute(1,0,2).reshape(c.worlds,-1)
        if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(costs).all()) or bool((costs>1e10).any()):
            raise FloatingPointError('Invalid MJWarp finite-difference trajectory in acados linearization')
        # Center the local model at the exact nominal used by globalization.
        # MJWarp contact reductions can differ between batch sizes.
        nominal_cost,_=p.evaluate(x,False)
        residual=self.primary_residuals.evaluate().permute(1,0,2).reshape(p.worlds,-1)[0].double()
        if self.central:J=((values[1:1+self.d]-values[1+self.d:])/(2*epsilon)).double()
        else:J=((values[1:]-values[:1])/epsilon).double()
        H=2*(J@J.T);g=2*(J@residual)
        return H.cpu().numpy(),g.cpu().numpy(),float(nominal_cost[0]),dict(
            derivative_trajectories=c.worlds,residual_dimensions=values.shape[1],
            residual_cost_error=float(values[0].double().square().sum()-costs[0]))


class AcadosGN:
    def __init__(self,backend):
        self.backend=backend;self.settings=s=backend.settings
        if backend.worlds<2:raise ValueError('acados-gn needs at least two rollout worlds for step safeguarding')
        if s.acados_difference not in ('forward','central') or s.acados_globalization not in ('linesearch','cauchy'):
            raise ValueError('Unknown acados difference or globalization rule')
        if any(not np.isfinite(v) or v<=0 for v in (s.acados_fd_step,s.acados_trust_radius,s.acados_damping)):
            raise ValueError('acados difference step, trust radius and damping must be finite and positive')
        if s.acados_initial_iterations<0:raise ValueError('acados initial iterations must be nonnegative')
        self.linearization=MJWarpLinearization(backend,s.acados_difference)
        self.qp=AcadosQuadratic(backend.variables.shape[1]);self.damping=s.acados_damping
        self.action_blocks=0
        self.persist=False;self.selected=wp.zeros(1,dtype=int,device=backend.device);self.export_graph=None
        # Observable arrays only; differentiation workspace must be rebuilt.
        self.observables=backend.observable_arrays()

    def export(self,x,selected):
        b=self.backend;self.selected.assign(np.array([selected],np.int32))
        caller=torch.cuda.current_stream(str(b.device));b.torch_stream.wait_stream(caller)
        with torch.no_grad(),torch.cuda.stream(b.torch_stream):
            def launch():
                for a in self.observables:
                    wp.launch(selected_kernel(a.dtype),a.size,inputs=[a.flatten(),self.selected,a.size//b.worlds],device=b.device)
                b._objective()
            if self.export_graph is None:
                launch()
                with wp.ScopedCapture(device=b.device) as capture:launch()
                self.export_graph=capture.graph
            wp.capture_launch(self.export_graph)
            wp.to_torch(b.variables).copy_(x)
            b._cached=x.clone();b._value=wp.to_torch(b.total).double().clone()
        caller.wait_stream(b.torch_stream)
        b._external_rollout=True;b._snapshot_cache=None;b.dirty=False
        if b.rollout_cache is not None:b.rollout_cache.clear();b.rollout_cache.workspace_valid=False
        return b._value.clone()

    def update(self,variables,evaluate,*,iterations,horizon,advance,continue_window=False,evaluate_on_device=False):
        b,s=self.backend,self.settings
        if iterations<1 or not evaluate_on_device:raise ValueError('acados-gn requires positive updates and device evaluation')
        x=torch.as_tensor(variables,dtype=torch.float64,device=str(b.device)).clone()
        if not bool((x==x[:1]).all()):raise ValueError('acados-gn requires one shared nominal trajectory')
        requested_iterations=iterations
        if self.action_blocks==0:iterations=max(iterations,s.acados_initial_iterations)
        self.action_blocks+=1
        history=[]
        with torch.no_grad():
            values,_=evaluate(x,False)
            if not torch.allclose(values,values[:1].expand_as(values),rtol=1e-5,atol=1e-5):
                raise ValueError('acados-gn candidate worlds must share one fixed ADMM objective')
            for _ in range(iterations):
                started=time.perf_counter();H,g,nominal,fd=self.linearization.linearize(x,s.acados_fd_step)
                linearization_seconds=time.perf_counter()-started
                scale=1/np.sqrt(np.maximum(np.diag(H),1e-8))
                scaled_H=H*scale[:,None]*scale[None,:]+self.damping*np.eye(len(g))
                scaled_g=g*scale
                trust=s.acados_trust_radius/s.control_scale
                lower,upper=knot_step_bounds(x[0].cpu().numpy(),s.control_scale,s.acados_trust_radius)
                qp_started=time.perf_counter()
                step,qp=self.qp.solve(scaled_H,scaled_g,lower/scale,upper/scale)
                qp_seconds=time.perf_counter()-qp_started
                if qp['status']!=0 or qp['qp_status']!=0 or not np.isfinite(step).all():
                    # Record failure and retain the real incumbent, never
                    # silently route to another optimizer or unsafe action.
                    self.damping=min(self.damping*10,1e8)
                    history.append(dict(**qp,**fd,value_before=float(values[0]),value_after=float(values[0]),
                                        accepted=False,alpha=0.,linearization_seconds=linearization_seconds,qp_seconds=qp_seconds))
                    continue
                step=scale*step
                n_gn=(b.worlds//2 if s.acados_globalization=='cauchy' and b.worlds>2 else b.worlds-1)
                n_cauchy=b.worlds-1-n_gn
                alpha_np=np.r_[.5**np.arange(n_gn),.5**np.arange(n_cauchy),0.]
                direction=np.broadcast_to(step,(b.worlds,len(step))).copy()
                if n_cauchy:
                    # Cauchy point in the same diagonal variable metric as
                    # the GN system. This is a residual-model descent step,
                    # not an AdamW fallback. All candidates use exact cost.
                    length=(scaled_g@scaled_g)/max(float(scaled_g@scaled_H@scaled_g),1e-20)
                    cauchy=-scale*scaled_g*length
                    cauchy*=min(1.,trust/max(float(np.abs(cauchy).max()),1e-20))
                    direction[n_gn:-1]=np.clip(cauchy,lower,upper)
                displacements=alpha_np[:,None]*direction
                trial=x+torch.as_tensor(displacements,device=x.device)
                before=float(values[0]);costs,_=evaluate(trial,False)
                selected=int(costs.argmin());chosen_alpha=float(alpha_np[selected])
                x=trial[selected:selected+1].expand_as(x).clone();values=self.export(x,selected)
                delta=displacements[selected]
                actual=before-float(values[0]);predicted=-(g@delta+.5*delta@H@delta)
                ratio=actual/max(predicted,1e-12)
                if ratio>.75:self.damping=max(self.damping*.5,s.acados_damping)
                elif ratio<.25:self.damping=min(self.damping*4,1e8)
                history.append(dict(**qp,**fd,value_before=before,value_after=float(values[0]),accepted=actual>0.,
                                    alpha=chosen_alpha,model_ratio=float(ratio),damping=self.damping,
                                    gradient_l2=float(np.linalg.norm(g)),predicted_reduction=float(predicted),actual_reduction=float(actual),
                                    chosen_direction=('incumbent' if selected==b.worlds-1 else 'Gauss_Newton' if selected<n_gn else 'Cauchy'),
                                    step_max_rad=float(np.max(np.abs(delta))*s.control_scale),
                                    linearization_seconds=linearization_seconds,qp_seconds=qp_seconds))
        return x.cpu().numpy(),sum(row['accepted'] for row in history),dict(
            method='acados_condensed_MJWarp_Gauss_Newton',iterations=iterations,
            requested_iterations=requested_iterations,
            qp_solver='FULL_CONDENSING_HPIPM',nlp_solver_type='SQP_RTI',sparse_stage_dynamics=False,
            globalization=s.acados_globalization,damping_floor=s.acados_damping,
            control_constraints='original_MJWarp_clamp_after_interpolation; QP_raw_knot_trust_box_only',
            finite_difference_gradients=True,difference=s.acados_difference,fd_step_control_radians=s.acados_fd_step*s.control_scale,
            steps=history,final_values=values.cpu().tolist(),selected_candidate=0)
