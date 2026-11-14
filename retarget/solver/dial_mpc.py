"""Diffusion-style annealed MPPI over control knots; no learned diffusion model.

Adapted mathematically from LeCAR-Lab/dial-mpc/core/dial_core.py (Apache-2.0):
horizon-dependent Gaussian noise, trajectory annealing, full-population cost
standardization and softmax mean. Physics is supplied by a Robokit evaluator.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
import torch


@dataclass(frozen=True)
class DIALConfig:
    samples: int = 128
    iterations: int = 6
    horizon: int = 16
    knot_stride: int = 4
    execute_steps: int = 2
    temperature: float = .03
    trajectory_decay: float = .7
    horizon_factor: float = .8
    noise_scale: float = .10
    seed: int = 42
    hold_first: bool = False

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.samples,self.iterations,self.horizon,self.knot_stride,self.execute_steps)):
            raise ValueError('Counts must be positive integers')
        if self.execute_steps>self.horizon or not 0<self.temperature or not 0<self.trajectory_decay<=1 or not 0<self.horizon_factor<=1 or self.noise_scale<=0:
            raise ValueError('Invalid DIAL schedule')
        if not np.isfinite([self.temperature,self.trajectory_decay,self.horizon_factor,self.noise_scale]).all(): raise ValueError('Nonfinite schedule')


def softmax_mean(candidates, costs, temperature):
    """Ignore invalid rollouts; equal costs give a finite uniform mean."""
    finite=torch.isfinite(costs)
    if not bool(finite.any()): raise RuntimeError('All physical rollouts are invalid')
    valid=costs[finite]
    scale=valid.std(unbiased=False).clamp_min(1e-6)*temperature
    scores=torch.where(finite,(valid.min()-costs)/scale,torch.full_like(costs,-torch.inf))
    weights=torch.softmax(scores,dim=0)
    return torch.einsum('n,nkd->kd',weights,candidates),weights


class DIALMPC:
    def __init__(self, evaluator, bounds, config=DIALConfig(), *, device='cpu'):
        self.evaluator,self.config,self.device=evaluator,config,torch.device(device)
        self.bounds=torch.as_tensor(bounds,dtype=torch.float32,device=self.device)
        if self.bounds.ndim!=2 or self.bounds.shape[1]!=2 or not torch.isfinite(self.bounds).all() or not (self.bounds[:,1]>self.bounds[:,0]).all(): raise ValueError('Finite ordered bounds required')
        self.generator=torch.Generator(device=self.device).manual_seed(config.seed)
        self.nodes=np.unique(np.r_[np.arange(0,config.horizon,config.knot_stride),config.horizon-1]).astype(int)
        if len(self.nodes)<2: self.nodes=np.array([0])
        x=np.arange(config.horizon)
        basis=np.stack([np.interp(x,self.nodes,np.eye(len(self.nodes))[i]) for i in range(len(self.nodes))],axis=1)
        self.basis=torch.tensor(basis,dtype=torch.float32,device=self.device)
        self.correction=None

    def shift(self, steps):
        if self.correction is None:return
        dense=self.basis@self.correction
        dense=torch.cat([dense[steps:],torch.zeros_like(dense[:steps])])
        self.correction=dense[self.nodes].clone()

    def solve(self, reference, action_sigma=None):
        cfg=self.config
        ref=torch.as_tensor(reference,dtype=torch.float32,device=self.device)
        if ref.shape!=(cfg.horizon,len(self.bounds)):raise ValueError('Reference must cover the whole horizon')
        mean=torch.zeros((len(self.nodes),ref.shape[1]),device=self.device) if self.correction is None else self.correction.clone()
        # Noise is in actual command units, avoiding a 4 m wrist range dominating.
        sigma=torch.as_tensor(action_sigma if action_sigma is not None else .5*(self.bounds[:,1]-self.bounds[:,0]),device=self.device,dtype=torch.float32)
        schedule=cfg.horizon_factor**torch.arange(len(self.nodes)-1,-1,-1,device=self.device)
        best_cost=math.inf;best=None;history=[];evaluations=0
        def controls(nodes):return (ref[None]+torch.einsum('hk,nkd->nhd',self.basis,nodes)).clamp(self.bounds[:,0],self.bounds[:,1])
        for i in range(cfg.iterations):
            eps=torch.randn((cfg.samples,len(self.nodes),ref.shape[1]),device=self.device,generator=self.generator)
            noise=eps*schedule[None,:,None]*sigma*cfg.noise_scale*cfg.trajectory_decay**i
            if cfg.hold_first:noise[:,0]=0
            proposed=torch.cat([mean[None],mean[None]+noise],dim=0)
            dense=controls(proposed)
            cost=self.evaluator(dense)
            if cost.shape!=(len(dense),):raise ValueError('Evaluator must return one cost per rollout')
            k=int(torch.nan_to_num(cost,nan=torch.inf,posinf=torch.inf,neginf=torch.inf).argmin())
            value=float(cost[k]);evaluations+=len(dense)
            if np.isfinite(value) and value<best_cost:best_cost=value;best=dense[k].clone()
            # Clip knot proposals consistently with the actually simulated bounds.
            knot_controls=(ref[self.nodes][None]+proposed).clamp(self.bounds[:,0],self.bounds[:,1])
            mean,weights=softmax_mean(knot_controls-ref[self.nodes][None],cost,cfg.temperature)
            history.append(dict(iteration=i,best_cost=best_cost,spread=float(cost[torch.isfinite(cost)].std(unbiased=False)),effective_samples=float(1/(weights.square().sum()))))
        moved=controls(mean[None])
        moved_cost=float(self.evaluator(moved)[0]);evaluations+=1
        if np.isfinite(moved_cost) and moved_cost<=best_cost:best,best_cost=moved[0],moved_cost
        if best is None:raise RuntimeError('No valid control sequence')
        self.correction=(best-ref)[self.nodes].clone()
        return best,dict(cost=best_cost,evaluations=evaluations,history=history,config=asdict(cfg),backend='DIAL-MPC/Robokit')
