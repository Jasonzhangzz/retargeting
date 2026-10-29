#!/usr/bin/env python3
"""Profile MJWarp humanoid MPC including opt-in condensed acados GN."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import subprocess

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
from retarget.humanoid import HumanoidProblem,ProjectionADMM,Settings


def profile(settings,repeats=5,task_name='humanoid-object'):
    from run_humanoid import task_paths
    scene,reference=task_paths(ROOT/'data/spider',task_name)
    start=time.perf_counter()
    p=HumanoidProblem(scene,reference,settings=settings)
    solver=ProjectionADMM(p)
    setup=time.perf_counter()-start
    try:
        result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        other_processes=[line.strip() for line in result.stdout.splitlines()
                         if line.strip() and int(line.split(',')[0])!=os.getpid()]
    except (OSError,ValueError,subprocess.SubprocessError):
        other_processes=None
    state=p.initial_state(2);previous=p.aref[2]
    runs=[]
    # First solve includes graph capture and module loading. Warm timings use
    # the same allocated solver, just as the real receding-horizon entry point.
    for repeat in range(repeats+1):
        start=time.perf_counter()
        rollout,history=solver.solve(state,[3+repeat],previous)
        elapsed=time.perf_counter()-start
        runs.append(dict(repeat=repeat,cold=repeat==0,seconds=elapsed,
                         contact_seconds=sum(row['contact_seconds'] for row in history),
                         action_seconds=sum(row['action_seconds'] for row in history),
                         forward_calls=history[-1]['forward_calls'],
                         backward_calls=history[-1]['backward_calls'],
                         common_forward_calls=history[-1].get('common_forward_calls',0),
                         rollout_cache_hits=history[-1].get('rollout_cache_hits',0),
                         action_optimizer=history[-1]['action_optimizer'],
                         **({'adamw':history[-1]['adamw']} if 'adamw' in history[-1] else {})))
        state=rollout['next_state'];previous=solver.actions[0]
    warm=[row['seconds'] for row in runs[1:]]
    return dict(task=task_name,scene=str(scene),reference=str(reference),
                nbody=p.model.nbody,nv=p.model.nv,mjwarp_contact_capacity=settings.mjwarp_contact_capacity,
                weights=asdict(p.weights),settings=asdict(p.settings),backend='MJWarp_implicitfast',gradient=solver.gradient_source,
                action_solver=solver.action_solver_name,finite_difference_gradients=(settings.action_solver in ('adamw','adamw-ls','acados-gn') or
                    (settings.action_solver=='sample-lm' and (not settings.sample_lm_reuse_samples or settings.action_iterations>1))),
                setup_seconds=setup,cold_solve_seconds=runs[0]['seconds'],
                median_solve_seconds=float(np.median(warm)),solve_hz=1/float(np.median(warm)),
                reference_dt=p.ref_dt,horizon_seconds=settings.horizon*settings.knot_stride*p.ref_dt,
                physical_steps_per_forward=settings.horizon*settings.knot_stride*settings.substeps,
                control_dimensions=p.model.nu,action_variables=settings.horizon*p.model.nu,
                worlds=settings.multistarts,runs=runs,
                other_cuda_processes_at_start=other_processes,
                gpu_exclusive=other_processes==[],
                note='Warm end-to-end MPC solve on one allocated solver. No viewer or disk logging. Other CUDA processes may affect these observed timings; a resident context alone does not prove activity. Timings are not a tracking success claim.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__,epilog='Default profiling uses 1x1 updates for speed comparison; run_humanoid.py defaults to 1x2 AdamW updates with lr=0.3.')
    for name,default in [('horizon',8),('knot-stride',3),('substeps',2),('multistarts',8),
                         ('admm-iterations',1),('action-iterations',1),('repeats',5)]:
        parser.add_argument('--'+name,type=int,default=default)
    parser.add_argument('--adamw-lr',type=float,default=.3)
    parser.add_argument('--action-solver',choices=['adamw','adamw-ls','mppi','acados-gn','sample-lm'],default='adamw')
    parser.add_argument('--acados-difference',choices=['forward','central'],default='forward')
    parser.add_argument('--acados-fd-step',type=float,default=.01)
    parser.add_argument('--acados-trust-radius',type=float,default=.15)
    parser.add_argument('--acados-damping',type=float,default=.01)
    parser.add_argument('--acados-globalization',choices=['linesearch','cauchy'],default='cauchy')
    parser.add_argument('--acados-initial-iterations',type=int,default=0)
    parser.add_argument('--line-search-trials',type=int,default=6)
    parser.add_argument('--mppi-samples',type=int,default=64)
    parser.add_argument('--mppi-noise-std',type=float,default=.05)
    parser.add_argument('--mppi-temperature',type=float,default=1.)
    parser.add_argument('--mppi-update',choices=['safeguarded','weighted-mean'],default='safeguarded')
    parser.add_argument('--mppi-single-rollout',action='store_true')
    parser.add_argument('--sample-lm-samples',type=int,default=64)
    parser.add_argument('--sample-lm-rank',type=int,default=32)
    parser.add_argument('--sample-lm-noise-std',type=float,default=.05)
    parser.add_argument('--sample-lm-fd-step',type=float,default=.02)
    parser.add_argument('--sample-lm-trust-radius',type=float,default=.03)
    parser.add_argument('--sample-lm-damping',type=float,default=.01)
    parser.add_argument('--sample-lm-reuse-samples',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--task',choices=['humanoid','humanoid-object'],default='humanoid-object')
    parser.add_argument('--trajectory-cache',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if min(getattr(args,name) for name in ['horizon','knot_stride','substeps','multistarts','admm_iterations','action_iterations','repeats'])<1:
        parser.error('Counts must be positive')
    settings=Settings(adamw_lr=args.adamw_lr,trajectory_cache=args.trajectory_cache,
        **{name:getattr(args,name) for name in ['sample_lm_samples','sample_lm_rank','sample_lm_noise_std','sample_lm_fd_step','sample_lm_trust_radius','sample_lm_damping','sample_lm_reuse_samples']},
        **{name:getattr(args,name) for name in ['horizon','knot_stride','substeps','multistarts','admm_iterations','action_iterations','action_solver','line_search_trials','mppi_samples','mppi_noise_std','mppi_temperature','mppi_update','mppi_single_rollout','acados_difference','acados_fd_step','acados_trust_radius','acados_damping','acados_globalization','acados_initial_iterations']})
    result=profile(settings,args.repeats,args.task)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(f'{result["action_variables"]} variables, {result["horizon_seconds"]:.3f}s horizon, '
          f'warm {result["median_solve_seconds"]:.4f}s ({result["solve_hz"]:.2f} Hz), '
          f'{result["runs"][-1]["forward_calls"]}/{result["runs"][-1]["backward_calls"]} forward/backward')
