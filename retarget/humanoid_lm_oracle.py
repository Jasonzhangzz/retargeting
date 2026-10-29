"""Shared exact MJWarp residual oracle for checkpoint tests and sampled LM."""
from copy import copy
from dataclasses import replace
import torch
import warp as wp

from retarget.humanoid_acados import MJWarpLinearization
from retarget.humanoid_mjwarp import MJWarpHumanoid
from retarget.humanoid_residuals import Residuals


class MJWarpResidualOracle(MJWarpLinearization):
    def __init__(self,parent,worlds=65):
        if parent.worlds!=1:
            raise ValueError('Residual oracle primary must have one common nominal world')
        self.parent=parent;self.d=parent.variables.shape[1];self.central=True
        p=copy(parent.problem);p.settings=replace(parent.settings,multistarts=worlds,trajectory_cache=False)
        self.backend=MJWarpHumanoid(p);self.residuals=Residuals(self.backend)
        self.primary_residuals=Residuals(parent);self.sync_graph=None
        self.reset_counts()

    def reset_counts(self):
        self.evaluated_trajectories=0;self.padded_trajectories=0;self.batch_calls=0

    def __call__(self,x,residuals=False):
        x=x.reshape(-1,self.d);results=[];rr=[]
        if len(x)==1:
            value,_=self.parent.evaluate(x,False)
            self.evaluated_trajectories+=1;self.batch_calls+=1
            if residuals:r=self.primary_residuals.evaluate().permute(1,0,2).reshape(1,-1).clone()
            return value.clone(),r if residuals else None
        c=self.backend
        for start in range(0,len(x),c.worlds):
            n=min(c.worlds,len(x)-start)
            batch=x[start:start+1].expand(c.worlds,-1).clone();batch[:n]=x[start:start+n]
            value,_=c.evaluate(batch,False);results.append(value[:n].clone())
            if residuals:rr.append(self.residuals.evaluate().permute(1,0,2).reshape(c.worlds,-1)[:n].clone())
            self.evaluated_trajectories+=n;self.padded_trajectories+=c.worlds-n;self.batch_calls+=1
        return torch.cat(results),torch.cat(rr) if residuals else None


class MJWarpSampleLM:
    """MPC adapter: one common nominal, frozen ADMM block, exact export."""
    def __init__(self,parent):
        from retarget.humanoid_common_rollout import CommonRollout
        from retarget.humanoid_sample_lm import SampleLM
        self.parent=parent;self.settings=parent.settings
        s=self.settings
        SampleLM(samples=s.sample_lm_samples,rank=s.sample_lm_rank,noise_std=s.sample_lm_noise_std,
            control_scale=s.control_scale,difference_step=s.sample_lm_fd_step,
            trust_radius=s.sample_lm_trust_radius,damping=s.sample_lm_damping,
            reuse_sample_residuals=s.sample_lm_reuse_samples)
        self.common_rollout=CommonRollout(parent)
        slots=min(65,max(parent.settings.sample_lm_samples+1,2*parent.settings.sample_lm_rank,8))
        self.oracle=MJWarpResidualOracle(self.common_rollout.backend,worlds=slots)
        self.blocks=0;self.persist=False

    def update(self,variables,evaluate,*,iterations,horizon,advance,continue_window=False,evaluate_on_device=False):
        from retarget.humanoid_sample_lm import SampleLM
        if iterations<1 or not evaluate_on_device:
            raise ValueError('Sampled LM requires positive iterations and a device objective')
        b,s=self.parent,self.settings
        x=torch.as_tensor(variables,dtype=torch.float64,device=str(b.device))
        if not bool((x==x[:1]).all()):raise ValueError('Sampled LM requires a shared nominal')
        common=[]
        for name in ('z','dual','forces','floor_force','floor_torque'):
            from_source=wp.to_torch(getattr(b.targets,name))
            common.append((from_source==from_source[:,:1]).all())
        if not bool(torch.stack(common).all()):
            raise ValueError('Sampled LM requires one fixed common ADMM objective')
        self.common_rollout.begin();self.oracle.sync();self.oracle.reset_counts()
        primary=self.common_rollout.backend;primary.forward_calls=primary.backward_calls=0
        self.oracle.backend.forward_calls=self.oracle.backend.backward_calls=0
        solver=SampleLM(samples=s.sample_lm_samples,rank=s.sample_lm_rank,iterations=iterations,
            noise_std=s.sample_lm_noise_std,control_scale=s.control_scale,
            difference_step=s.sample_lm_fd_step,trust_radius=s.sample_lm_trust_radius,
            damping=s.sample_lm_damping,seed=s.seed+self.blocks,
            reuse_sample_residuals=s.sample_lm_reuse_samples)
        answer,diag=solver.solve(x[:1],self.oracle)
        result=answer.expand_as(x).clone();self.common_rollout.export(result)
        self.blocks+=1
        diag.update(iterations=iterations,gradient_evaluations=0,parameter_update_backend='torch_cuda',
            device_objective_callback=True,continued_within_window=continue_window,
            finite_difference_gradients=diag['finite_difference_model'],
            primary_forward_calls=primary.forward_calls,batch_forward_calls=self.oracle.backend.forward_calls,
            requested_trajectories=self.oracle.evaluated_trajectories,padded_trajectories=self.oracle.padded_trajectories,
            export='best_actual_rollout_broadcast_to_shared_nominal')
        return result.cpu().numpy().copy(),int(diag['value_after']<diag['value_before']),diag
