#!/usr/bin/env python3
"""End-to-end API/future and reset isolation preflight on CUDA, no training."""
import argparse,json,os
from pathlib import Path
import numpy as np
import torch,warp as wp
from retarget.learning.wuji_ik.trajectory_pipeline import EMFTrajectorySolver


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;w=r/'oneway_comparison';assert json.loads((w/'STATUS.json').read_text()).get('complete');torch.set_grad_enabled(False);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory);torch._inductor.config.compile_threads=2;wp.config.kernel_cache_dir='/dev/shm/wuji_lm_runtime_2207_20261111/anchor_warp_cache'
    data=np.load(r/'inputs/trajectory_0_left.npz');pos=data['position'][:8].copy();quat=data['quaternion'][:8].copy();model=EMFTrajectorySolver(r/'reference',r/'assets/builtin_left.urdf','left',device='cuda:0');initial=model.tensor(model.robot.bounds.mean(1)).expand(8,-1).clone();rows=[]
    for backend in ['causal_prefix','frozen_parallel','frozen_history']:
        base=model.solve(pos,quat,[0,4,8],backend=backend,initial_q=initial,iterations=4,stage1_iterations=4)
        changed=pos.copy();changed[6:]+=.001
        future=model.solve(changed,quat,[0,4,8],backend=backend,initial_q=initial,iterations=4,stage1_iterations=4)
        changed=pos.copy();changed[4:]+=.001
        segment=model.solve(changed,quat,[0,4,8],backend=backend,initial_q=initial,iterations=4,stage1_iterations=4)
        for v in [base,future,segment]:assert torch.isfinite(v.q_raw).all() and torch.isfinite(v.human_q).all()
        row=dict(backend=backend,raw_future_past_max_rad=float((base.q_raw[:6]-future.q_raw[:6]).abs().max()),filtered_future_past_max_rad=float((base.q_filtered[:6]-future.q_filtered[:6]).abs().max()),segment_isolation_max_rad=float((base.q_raw[:4]-segment.q_raw[:4]).abs().max()),human_segment_isolation_max_rad=float((base.human_q[:4]-segment.human_q[:4]).abs().max()),information=base.information)
        assert max(v for k,v in row.items() if k.endswith('_rad'))==0.,row
        rows.append(row);print('ONEWAY_API_CHECK',row,flush=True)
    (w/'pipeline_validation.json').write_text(json.dumps(dict(passed=True,rows=rows,neural_optimizer_updates=0,actual_pipeline_solves=9),indent=2))

if __name__=='__main__':main()
