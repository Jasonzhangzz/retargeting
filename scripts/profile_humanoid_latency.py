#!/usr/bin/env python3
"""Separate physics, optimizer bookkeeping and CPU/GPU latency tradeoffs.

The CPU rollout benchmark deliberately excludes the whole-body cost and uses
CPU MuJoCo implicitfast for a replay timing reference. It is not an alternative controller
or a claim of identical dynamics/closed-loop performance.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import mujoco
import mujoco.rollout
import numpy as np
import torch
import warp as wp
from retarget.fa.adamw import AdamW
from retarget.humanoid import HumanoidProblem,ProjectionADMM,Settings
from run_humanoid import task_paths


def profile(output,task='humanoid',repeats=7):
    output.mkdir(parents=True,exist_ok=True)
    scene,reference=task_paths(ROOT/'data/spider',task)
    settings=Settings(admm_iterations=1,action_iterations=8)
    p=HumanoidProblem(scene,reference,settings=settings)
    solver=ProjectionADMM(p);b=solver.backend
    state=p.initial_state(2)
    solver.solve(state,[3],p.aref[2])
    x=np.broadcast_to(solver.warm,b.variables.shape).copy()
    b.evaluate(x,True)
    phases={}
    x=np.zeros(b.variables.shape)
    for label,gradient in [('forward_and_cost',False),('central_difference_gradient_and_cost',True)]:
        values=[]
        for repeat in range(3):
            trial=x+(repeat+1)*.0001
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            begin.record();b.evaluate(trial,gradient);end.record();end.synchronize()
            values.append(begin.elapsed_time(end))
        phases[label]=dict(median_cuda_ms=float(np.median(values)),samples_cuda_ms=values)
    kernel_profile={}
    try:
        # Capture is already warmed up. Trace one forward+cost+backward replay.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            b.evaluate(x+.001,True)
            torch.cuda.synchronize()
        grouped=defaultdict(lambda:[0,0.])
        for event in prof.events():
            if event.device_type==torch.autograd.DeviceType.CUDA:
                grouped[event.name][0]+=1
                grouped[event.name][1]+=event.device_time_total
        entries=[dict(name=k,calls=v[0],total_cuda_ms=v[1]/1000) for k,v in grouped.items()]
        entries.sort(key=lambda x:x['total_cuda_ms'],reverse=True)
        kernel_profile={'total_device_events':sum(x['calls'] for x in entries),
                        'total_profiled_cuda_ms':sum(x['total_cuda_ms'] for x in entries),
                        'kernels':entries,
                        'note':'Profiler instrumentation changes runtime; use separate CUDA-event/whole-solve timings for latency.'}
        prof.export_chrome_trace(str(output/'mjwarp_trace.json'))
    except (RuntimeError,AttributeError) as error:
        kernel_profile={'unavailable':str(error)}
    optimizer=AdamW(learning_rate=1.,device='cuda:0')
    variables=np.ones(b.variables.shape)
    def cheap_objective(x,gradient):
        return x.square().sum(dim=1),2*x if gradient else None
    optimizer_times=[]
    for repeat in range(repeats+1):
        torch.cuda.synchronize();started=time.perf_counter()
        optimizer.update(variables,cheap_objective,iterations=8,horizon=settings.horizon,
                         advance=0.,evaluate_on_device=True)
        torch.cuda.synchronize()
        if repeat:optimizer_times.append((time.perf_counter()-started)*1000)
    # Same CPU model and nominal controls, but a different physics integrator.
    # Sensor/cost evaluation needed by a complete controller is not included.
    controls=np.stack([control.numpy()[0] for control in b.controls])
    initial=np.empty(mujoco.mj_stateSize(p.model,mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(p.model,state,initial,mujoco.mjtState.mjSTATE_FULLPHYSICS)
    cpu=[]
    for worlds in [8,128,512]:
        u=np.broadcast_to(controls,(worlds,*controls.shape)).copy()
        q=np.broadcast_to(initial,(worlds,len(initial))).copy()
        for threads in [1,8]:
            data=[mujoco.MjData(p.model) for _ in range(threads)]
            with mujoco.rollout.Rollout(nthread=threads) as rollout:
                rollout.rollout(p.model,data,q,u)
                samples=[]
                for _ in range(repeats):
                    started=time.perf_counter();states,sensors=rollout.rollout(p.model,data,q,u)
                    samples.append((time.perf_counter()-started)*1000)
            cpu.append(dict(worlds=worlds,threads=threads,physics_steps=len(controls),
                            median_ms=float(np.median(samples)),samples_ms=samples,
                            finite=bool(np.isfinite(states).all())))
    result=dict(task=task,scene=str(scene),settings=asdict(settings),
                torch=torch.__version__,warp=wp.__version__,gpu=torch.cuda.get_device_name(),
                phases=phases,kernel_profile=kernel_profile,
                cheap_objective_adamw_8_updates_ms=float(np.median(optimizer_times)),
                cheap_objective_samples_ms=optimizer_times,
                cheap_objective_note='Includes optimizer bookkeeping, best-iterate retention and a cheap quadratic GPU callback. Not a directly subtractable component of the MJWarp solve.',
                cpu_physics_only=cpu,
                cpu_note='Different MuJoCo implicitfast physics; no humanoid cost, ADMM, search, sensor extensions or closed-loop validation. Not a measured controller frequency.')
    (output/'latency.json').write_text(json.dumps(result,indent=2)+'\n')
    print('CUDA phases',phases)
    print('AdamW 8 updates with cheap GPU objective:',result['cheap_objective_adamw_8_updates_ms'],'ms')
    print('Top device events:',kernel_profile.get('kernels',[])[:8])
    print('CPU physics only:',[{k:v for k,v in row.items() if k!='samples_ms'} for row in cpu])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--task',choices=['humanoid','humanoid-object'],default='humanoid')
    parser.add_argument('--repeats',type=int,default=7)
    args=parser.parse_args()
    if args.repeats<1:parser.error('Positive repeats required')
    profile(args.output,args.task,args.repeats)
