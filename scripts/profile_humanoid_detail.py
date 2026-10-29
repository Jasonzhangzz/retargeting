#!/usr/bin/env python3
"""CPU dispatch / CUDA graph phase timing and objective-gradient attribution."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import argparse,json,time,cProfile,pstats,io,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,warp as wp,mujoco
from retarget.humanoid import HumanoidProblem,ProjectionADMM,Settings,Weights
from retarget.humanoid_terms import TERMS


def profile(output,case=None,frame=2,substeps=2):
    task=ROOT/'data/spider/processed/omomo/unitree_g1/humanoid_object/move_largebox'
    scene,reference=task/'scene.xml',task/'0/trajectory_kinematic.npz'
    weights=Weights();state_data=None
    if case:
        cfg=json.loads((case/'config.json').read_text());scene,reference=cfg['scene'],cfg['reference'];weights=Weights(**cfg['weights'])
        state_data=np.load(case/'rollout.npz')
    p=HumanoidProblem(scene,reference,settings=Settings(substeps=substeps,admm_iterations=1,action_iterations=1,adamw_lr=.3),weights=weights)
    solver=ProjectionADMM(p);b=solver.backend
    state=p.initial_state(frame);previous=p.aref[frame]
    if state_data is not None:
        state.qpos[:]=state_data['qpos'][frame];state.qvel[:]=state_data['qvel'][frame]
        if frame:previous=state_data['ctrl'][frame-1]
        mujoco.mj_forward(p.model,state)
    solver.solve(state,[frame+1],previous)
    meters={}
    def instrument(owner,name,label):
        original=getattr(owner,name)
        def wrapped(*args,**kwargs):
            start=time.perf_counter()
            try:return original(*args,**kwargs)
            finally:
                m=meters.setdefault(label,dict(calls=0,seconds=0.));m['calls']+=1;m['seconds']+=time.perf_counter()-start
        setattr(owner,name,wrapped)
    for name in ('prepare','snapshots','evaluate'):instrument(b,name,name)
    if b.contact_qp is not None:instrument(b.contact_qp,'solve','force_qp')
    profiler=cProfile.Profile();profiler.enable();solver.solve(state,[frame+1],previous);profiler.disable()
    solve_meters={k:dict(v) for k,v in meters.items()}
    stream=io.StringIO();pstats.Stats(profiler,stream=stream).sort_stats('cumtime').print_stats(35)
    output.mkdir(parents=True,exist_ok=True);(output/'cpu_profile.txt').write_text(stream.getvalue())
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
    original=b.model.weights.numpy();penalty=b.targets.penalty.numpy()
    x=np.zeros(b.variables.shape);components={}
    for name in TERMS[:-2]:
        w=np.zeros_like(original);w[TERMS.index(name)]=original[TERMS.index(name)]
        b.model.weights.assign(w);b.targets.penalty.zero_();b.dirty=True
        value,g=b.evaluate(x,True)
        components[name]=dict(value=value.tolist(),gradient_norm=np.linalg.norm(g,axis=1).tolist(),first_knot_norm=np.linalg.norm(g[:,:p.model.nu],axis=1).tolist())
    b.model.weights.assign(original);b.targets.penalty.assign(penalty);b.dirty=True
    result=dict(frame=frame,settings=vars(p.settings),phases=phases,meters=solve_meters,gradient_components=components,
                note='Nested CPU meters overlap. CUDA phases time actual MJWarp callbacks. Component values are predicted horizon costs at the same fixed action/contact inputs.')
    (output/'profile.json').write_text(json.dumps(result,indent=2)+'\n')
    print('CUDA phases',phases);print('CPU meters',solve_meters)
    for name in ('balance','upright','keypoint_position','contact_wrench'):
        c=components[name];print(name,'cost',c['value'][0],'gradient',c['gradient_norm'][0],'first knot',c['first_knot_norm'][0])
    print(stream.getvalue())

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--case',type=Path);parser.add_argument('--frame',type=int,default=2);parser.add_argument('--substeps',type=int,default=2);args=parser.parse_args();profile(args.output,args.case,args.frame,args.substeps)
