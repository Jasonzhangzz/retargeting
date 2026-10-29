"""Sampling starts followed by safeguarded residual LM in action space.

This module optimizes the action knots, not the separate six-dimensional
contact-force LM. An oracle supplies the unchanged physical cost and residuals.
All proposals are accepted only by their actual nonlinear rollout cost.
"""
import math
import torch


def lm_step(jacobian, residual, damping):
    """Jacobian rows are parameter directions; solve the damped GN system."""
    matrix=jacobian@jacobian.T
    gradient=jacobian@residual
    diagonal=torch.diag(matrix).clamp(min=1.e-8)
    step=torch.linalg.solve(matrix+damping*torch.diag(diagonal),-gradient)
    return step,matrix,gradient


class SampleLM:
    def __init__(self, *, samples=0, rank=32, iterations=1, noise_std=.05,
                 control_scale=.05, difference_step=.02, trust_radius=.03,
                 damping=.01, seed=0, reuse_sample_residuals=False):
        if min(rank,iterations)<1 or samples<0:
            raise ValueError('Positive LM rank/iterations and nonnegative sample count required')
        if not all(math.isfinite(v) and v>0 for v in
                   (noise_std,control_scale,difference_step,trust_radius,damping)):
            raise ValueError('LM scales must be finite and positive')
        if reuse_sample_residuals and samples<rank+1:
            raise ValueError('Sample-fit LM needs more samples than subspace dimensions')
        self.samples,self.rank,self.iterations=samples,rank,iterations
        self.sigma,self.scale=noise_std/control_scale,control_scale
        self.epsilon,self.trust,self.damping=difference_step,trust_radius/control_scale,damping
        self.seed,self.reuse=seed,reuse_sample_residuals

    def solve(self, initial, oracle):
        x=initial.detach().clone().reshape(1,-1)
        d=x.shape[1];rank=min(d,self.rank)
        generator=torch.Generator(device=x.device).manual_seed(self.seed)
        cost,residual=oracle(x,True)
        if not bool(torch.isfinite(cost[0]) and torch.isfinite(residual[0]).all()):
            raise FloatingPointError('LM incumbent must have finite cost and residuals')
        initial_cost=float(cost[0]);current_cost=cost[0];r=residual[0].double()
        sample_x=sample_r=None
        sampling_improved=False
        if self.samples:
            noise=self.sigma*torch.randn((self.samples,d),dtype=x.dtype,device=x.device,generator=generator)
            sample_x=torch.cat((x,x+noise),dim=0)
            sample_cost,sample_r=oracle(sample_x,True)
            valid=torch.isfinite(sample_cost)&torch.isfinite(sample_r).all(dim=1)
            sample_cost=torch.where(valid,sample_cost,torch.inf)
            chosen=int(sample_cost.argmin())
            sampling_improved=bool(sample_cost[chosen]<current_cost)
            x=sample_x[chosen:chosen+1].clone();current_cost=sample_cost[chosen];r=sample_r[chosen].double()
            sample_x,sample_r=sample_x[valid],sample_r[valid]
        sample_cost_value=float(current_cost)
        basis_generator=torch.Generator(device=x.device).manual_seed(self.seed+100003)
        if rank==d:
            directions=torch.eye(d,dtype=x.dtype,device=x.device)
        else:
            raw=torch.randn((d,rank),dtype=x.dtype,device=x.device,generator=basis_generator)
            directions=torch.linalg.qr(raw,mode='reduced').Q.T.contiguous()
        history=[];damping=self.damping
        for iteration in range(self.iterations):
            if self.reuse and iteration==0 and len(sample_x)>=rank+1:
                coefficients=(sample_x-x)@directions.T
                differences=sample_r.double()-r
                # Reuse sampled secants; this is a fitted local model, not an
                # exact Jacobian. Nonlinear rollout validation remains required.
                ridge=1.e-4*torch.diag(coefficients.T@coefficients).mean().clamp(min=1.e-8)
                J=torch.linalg.solve(coefficients.T@coefficients+ridge*torch.eye(rank,device=x.device,dtype=x.dtype),
                                     coefficients.T@differences)
                derivative_trajectories=0;model_kind='sampled_residual_secants'
            else:
                probes=torch.cat((x+self.epsilon*directions,x-self.epsilon*directions),dim=0)
                _,values=oracle(probes,True)
                J=(values[:rank].double()-values[rank:].double())/(2*self.epsilon)
                derivative_trajectories=2*rank;model_kind='central_difference_subspace_Jacobian'
            if not bool(torch.isfinite(J).all()):
                history.append(dict(iteration=iteration,rank=rank,model=model_kind,
                    derivative_trajectories=derivative_trajectories,value_before=float(current_cost),
                    value_after=float(current_cost),accepted=False,invalid_linearization=True,
                    damping=damping,proposal_max_displacement_rad=0.))
                damping=min(damping*10.,1.e8)
                continue
            candidate_x=[];models=[]
            for multiplier in (.1,1.,10.,100.):
                alpha,H,g=lm_step(J,r,damping*multiplier)
                step=alpha@directions
                factor=min(1.,self.trust/max(float(step.abs().max()),1.e-20))
                for fraction in (1.,.5):
                    a=alpha*(factor*fraction)
                    candidate_x.append(x[0]+a@directions)
                    predicted=-2*torch.dot(g,a)-torch.dot(a,H@a)
                    models.append((damping*multiplier,float(predicted)))
            candidates=torch.stack(candidate_x)
            values,trial_residuals=oracle(candidates,True)
            chosen=int(values.argmin());best=values[chosen]
            accepted=bool(torch.isfinite(best) and best<current_cost)
            improvement=float(current_cost-best) if accepted else 0.
            selected_damping,predicted=models[chosen]
            ratio=improvement/predicted if accepted and predicted>0 else 0.
            history.append(dict(iteration=iteration,rank=rank,model=model_kind,
                derivative_trajectories=derivative_trajectories,value_before=float(current_cost),
                value_after=float(best if accepted else current_cost),accepted=accepted,
                predicted_reduction=predicted,actual_to_predicted_ratio=ratio,
                damping=selected_damping,proposal_max_displacement_rad=float((candidates[chosen]-x[0]).abs().max())*self.scale))
            if accepted:
                x=candidates[chosen:chosen+1].clone();current_cost=best;r=trial_residuals[chosen].double()
                damping=selected_damping*(.5 if ratio>.75 else (4. if ratio<.25 else 1.))
            else:
                damping=min(damping*10.,1.e8)
        return x,dict(method='sampling_plus_LM' if self.samples else 'local_LM',samples=self.samples,rank=rank,
            value_before=initial_cost,value_after_sampling=sample_cost_value,value_after=float(current_cost),
            sampling_improved=sampling_improved,steps=history,
            finite_difference_model=any(s['model']=='central_difference_subspace_Jacobian' for s in history),
            sample_fit_model=any(s['model']=='sampled_residual_secants' for s in history),analytic_backward_calls=0)
