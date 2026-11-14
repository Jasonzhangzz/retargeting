#!/usr/bin/env python3
"""Numerical equivalence and steady timing of two CUDA Riccati executions."""
import argparse,json,time
from pathlib import Path
import torch,warp as wp
from retarget.learning.wuji_ik.trajectory import TrajectoryObjective
from retarget.learning.wuji_ik.trajectory_ilqr import AnalyticILQR


class DenseQuadratic:
 def __init__(self,target,matrix):self.target=target;self.matrix=matrix
 def __call__(self,q):return torch.einsum('ij,nj->ni',self.matrix,q-self.target),self.matrix.expand(len(q),-1,-1)


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_grad_enabled(False);torch.set_num_threads(2);torch._inductor.config.compile_threads=2;torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory);wp.config.kernel_cache_dir='/dev/shm/wuji_lm_runtime_2207_20261111/anchor_warp_cache';wp.init();torch.manual_seed(42);rows=[]
 for d,w in [(3,.3),(20,.01),(21,20.)]:
  n=17;q=torch.randn(n,d,device='cuda',dtype=torch.float64)*.03;target=q+torch.randn_like(q)*.8;matrix=torch.eye(d,device=q.device,dtype=q.dtype)+torch.randn(d,d,device=q.device,dtype=q.dtype)*.03
  obj=TrajectoryObjective(DenseQuadratic(target,matrix),[0,7,17],q[[0,7]],w);lo=q.new_full((d,),-.3);hi=q.new_full((d,),.3);answers=[];measure=[]
  for fused in [False,True]:
   solver=AnalyticILQR(obj,q,lo,hi,fused=fused);solver(q,4);torch.cuda.synchronize();times=[]
   for repeat in range(3):
    begin=time.perf_counter();out=solver(q,4);torch.cuda.synchronize();times.append(time.perf_counter()-begin)
   answers.append((out.q.clone(),wp.to_torch(solver._feedforward).clone(),wp.to_torch(solver._feedback).clone()));measure.append(times)
  diffs=[float((x-y).abs().max()) for x,y in zip(answers[0],answers[1])];assert diffs[0]<1e-4 and max(diffs[1:])<3e-4,(d,w,diffs)
  row=dict(d=d,history=w,max_q_difference=diffs[0],max_ff_difference=diffs[1],max_feedback_difference=diffs[2],old_seconds=measure[0],fused_seconds=measure[1]);rows.append(row);print('PASS',row,flush=True)
 (a.root/'ilqr_fusion_validation.json').write_text(json.dumps(dict(passed=True,checks=rows,neural_optimizer_updates=0,scope='Dense SPD spatial blocks, active bounds, two different-length independent segments, robot/human history weights; FP64 values and FP32 stored gains'),indent=2))


if __name__=='__main__':main()
