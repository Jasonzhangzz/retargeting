#!/usr/bin/env python3
"""Bounded mathematical, memory, FK and causal-information validation."""
from pathlib import Path
import argparse,json,gc
import numpy as np
import torch,warp as wp
from scipy.optimize import minimize
from retarget.learning.wuji_ik.trajectory import TrajectoryObjective,CUDABandedLM
from retarget.learning.wuji_ik.trajectory_lbfgs import RobokitTrajectoryLBFGS
from retarget.learning.wuji_ik.trajectory_ilqr import AnalyticILQR
from retarget.learning.wuji_ik.trajectory_pipeline import EMFTrajectorySolver


class Quadratic:
 def __init__(self,target):self.target=target
 def __call__(self,q):return q-self.target,torch.eye(q.shape[-1],device=q.device,dtype=q.dtype).expand(len(q),-1,-1)


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;torch.set_grad_enabled(False);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory);wp.config.kernel_cache_dir='/dev/shm/wuji_lm_runtime_2207_20261111/anchor_warp_cache';wp.init();torch.manual_seed(42)
 target=torch.randn(11,3,dtype=torch.float64)*3;cpu=TrajectoryObjective(Quadratic(target),[0,4,11],torch.zeros(2,3,dtype=target.dtype),.3)
 def fn(x):
  q=torch.from_numpy(x.reshape(11,3));cost,g,_,_=cpu.evaluate(q);return float(cost.sum()),g.numpy().ravel()
 optimum=minimize(fn,np.zeros(33),jac=True,method='L-BFGS-B',bounds=[(-1,1)]*33,options=dict(gtol=1e-12,ftol=1e-15,maxiter=500)).x.reshape(11,3)
 q=target.cuda()*0;obj=TrajectoryObjective(Quadratic(target.cuda()),[0,4,11],q[[0,4]],.3);lo=q.new_full((3,),-1);hi=-lo;checks=[]
 for backend in ['banded_lm','trajectory_lbfgs','gpu_ilqr']:
  if backend=='banded_lm':result=CUDABandedLM(obj,q,lo,hi)(q,128)
  elif backend=='trajectory_lbfgs':result=RobokitTrajectoryLBFGS(obj,128)(q,lo,hi)
  else:result=AnalyticILQR(obj,q,lo,hi)(q,32)
  error=float(np.max(abs(result.q.cpu().numpy()-optimum)));assert error<3e-4,(backend,error);assert torch.all(result.q>=lo-1e-7) and torch.all(result.q<=hi+1e-7);checks.append(dict(backend=backend,max_bounded_q_error=error));print('BOUNDED_PASS',checks[-1],flush=True)
 memory=[]
 for length in [64,128,256,512,1024,2048]:
  gc.collect();torch.cuda.empty_cache();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();t=torch.randn(length,20,device='cuda',dtype=torch.float64)*.1;initial=torch.zeros_like(t);o=TrajectoryObjective(Quadratic(t),[0,length],initial[:1],.01);e=CUDABandedLM(o,initial,t.new_full((20,),-1),t.new_full((20,),1));res=e(initial,2);torch.cuda.synchronize();memory.append(dict(length=length,peak_increment_bytes=torch.cuda.max_memory_allocated()-baseline,normal_shape=[length,20,20]));del e,res,o,t,initial;print('MEMORY',memory[-1],flush=True)
 assert memory[-1]['peak_increment_bytes']<memory[0]['peak_increment_bytes']*(2048/64)*2
 data=np.load(r/'inputs/trajectory_0_right.npz');position=data['position'][:8].copy();rotation=data['quaternion'][:8].copy();pipeline=EMFTrajectorySolver(r/'reference',r/'assets/builtin_right.urdf','right',device='cuda:0');cpu_fk=pipeline.human.forward(pipeline.seed.cpu().numpy())['skeleton'][:,:3,3];gpu_fk=pipeline.sfk.forward(pipeline.seed[None])[0][0].cpu().numpy();assert abs(cpu_fk-gpu_fk).max()<1e-10
 causal=[]
 for backend in ['causal_lm','causal_lbfgs']:
  original=pipeline.solve_causal(position,rotation,[0,8],backend=backend);changed=position.copy();changed[4:]+=.003;altered=pipeline.solve_causal(changed,rotation,[0,8],backend=backend);drift=float((original.q_raw[:4]-altered.q_raw[:4]).abs().max());assert drift==0.;causal.append(dict(backend=backend,past_difference_after_future_change=drift));print('CAUSAL_PASS',causal[-1],flush=True)
 (r/'structural_validation.json').write_text(json.dumps(dict(passed=True,bounded=checks,memory=memory,causal=causal,fk_max_m=float(abs(cpu_fk-gpu_fk).max()),neural_optimizer_updates=0,numerical_budgets=dict(bounded=[128,128,32],memory_iterations=2,causal_frames=8)),indent=2));print('STRUCTURAL_VALIDATION_COMPLETE',flush=True)


if __name__=='__main__':main()
