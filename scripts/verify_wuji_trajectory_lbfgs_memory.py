#!/usr/bin/env python3
"""Bounded long-vector global-history L-BFGS correctness checks."""
from pathlib import Path
import argparse,json
import numpy as np,torch,warp as wp
from scipy.optimize import minimize
from retarget.learning.wuji_ik.trajectory import TrajectoryObjective
from retarget.learning.wuji_ik.trajectory_lbfgs import RobokitTrajectoryLBFGS,ProjectedLBFGS


class Quadratic:
 def __init__(self,target):self.target=target
 def __call__(self,q):return q-self.target,torch.eye(q.shape[1],device=q.device,dtype=q.dtype).expand(len(q),-1,-1)


p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;torch.set_grad_enabled(False);torch.set_num_threads(2);torch._inductor.config.compile_threads=2;torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory);wp.config.kernel_cache_dir='/dev/shm/wuji_lm_runtime_2207_20261111/anchor_warp_cache';torch.manual_seed(42);checks=[]
for n,offsets in [(13,[0,5,13]),(240,[0,120,240])]:
 target=torch.randn(n,3,dtype=torch.float64)*3;obj_cpu=TrajectoryObjective(Quadratic(target),offsets,torch.zeros(2,3,dtype=torch.float64),.3)
 def fn(x):
  cost,g,_,_=obj_cpu.evaluate(torch.from_numpy(x.reshape(n,3)));return float(cost.sum()),g.numpy().ravel()
 expected=minimize(fn,np.zeros(n*3),jac=True,method='L-BFGS-B',bounds=[(-1,1)]*(n*3),options=dict(gtol=1e-12,ftol=1e-15,maxiter=1000)).x.reshape(n,3)
 initial=torch.zeros(n,3,device='cuda',dtype=torch.float64);obj=TrajectoryObjective(Quadratic(target.cuda()),offsets,initial[[0,offsets[1]]],.3);lo=initial.new_full((3,),-1);hi=-lo
 modes=[False,True] if n==13 else [True];solutions=[]
 for force in modes:
  descriptor=ProjectedLBFGS.__dict__['_build_kernel']
  if force:ProjectedLBFGS._build_kernel=classmethod(lambda cls,*args:None)
  try:
   engine=RobokitTrajectoryLBFGS(obj,128);result=engine(initial,lo,hi);torch.cuda.synchronize();error=float(np.max(abs(result.q.cpu().numpy()-expected)));assert error<3e-5,(n,force,error);assert torch.all(result.cost<=result.initial_cost+1e-9);assert (engine.solver._search_direction_kernel is None)==force;solutions.append(result.q.clone());row=dict(frames=n,forced_global=force,max_error_vs_scipy=error,backend=result.backend);checks.append(row);print('LBFGS_MEMORY_PASS',row,flush=True)
  finally:ProjectedLBFGS._build_kernel=descriptor
 if len(solutions)==2:assert float((solutions[0]-solutions[1]).abs().max())<3e-5
(r/'lbfgs_global_history_validation.json').write_text(json.dumps(dict(passed=True,checks=checks,history_length=10,iterations=128,neural_optimizer_updates=0),indent=2))
