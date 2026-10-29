#!/usr/bin/env python3
"""Track an offline SPIDER control plan with the same mjwarp local objective.

Only the initial state is read from SPIDER. Later states come from actual mjwarp
execution; the saved plan supplies feedforward controls, never future states.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from retarget.humanoid import HumanoidProblem,Settings,ProjectionADMM,run
from run_humanoid import task_paths


def execute(reference,output,method,frames=None,multistarts=8):
    meta=json.loads((reference/'summary.json').read_text());data=np.load(reference/'trajectory.npz')
    if not meta['feasible_under_balance_audit']:raise ValueError('Source plan failed its balance audit')
    settings=Settings(action_solver='sample-lm',admm_iterations=1,action_iterations=1,
                      sample_lm_reuse_samples=method=='sample-fit',multistarts=multistarts)
    if method.startswith('adamw'):
        settings=replace(settings,action_solver='adamw',adamw_lr=.3,action_iterations=int(method[5:]))
    p=HumanoidProblem(*task_paths(ROOT/'data/spider',meta['task']),settings=settings,ref_dt=2*meta['config']['sim_dt'])
    # The executed plan plus the unexecuted tail of its final optimized horizon.
    dense=np.concatenate((data['ctrl'],data['planned_controls'][-1,meta['config']['ctrl_steps']:]),axis=0)
    state=mujoco.MjData(p.model)
    state.qpos[:]=data['checkpoint_qpos'][0];state.qvel[:]=data['checkpoint_qvel'][0]
    state.ctrl[:]=data['checkpoint_previous_ctrl'][0];mujoco.mj_forward(p.model,state)
    def factory(problem):
        solver=ProjectionADMM(problem);b=solver.backend;prepare=b.prepare
        def with_feedforward(measured,first,previous):
            prepare(measured,first,previous)
            step=(first-1)*settings.substeps
            ids=np.minimum(step+np.arange(b.dense_steps),len(dense)-1)
            prior=(data['checkpoint_previous_ctrl'][0] if step==0 else dense[min(step-1,len(dense)-1)])
            prior=np.clip(prior,p.lower,p.upper)
            b.use_dense_control_base(dense[ids],previous_delta=np.asarray(previous)-prior)
        b.prepare=with_feedforward
        return solver
    run(p,output,frames=frames,initial_state=state,solver_factory=factory)
    path=output/'config.json';config=json.loads(path.read_text())
    config.update(offline_spider_reference=str(reference),feedforward_method=method,
        initial_state_source='First freshly simulated SPIDER checkpoint; copied only once',
        future_spider_states_injected=False,
        previous_control_correction_retained_at_window_boundary=True,
        control_parameterization='Saved executed SPIDER dense plan plus shifted local control-knot increments',
        tracking_target='Original kinematic reference and unchanged complete mjwarp objective')
    path.write_text(json.dumps(config,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('reference',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--method',choices=['sample-fit','sample-fd','adamw1','adamw2','adamw3','adamw8'],default='sample-fd')
    p.add_argument('--multistarts',type=int,default=8)
    p.add_argument('--frames',type=int);a=p.parse_args();execute(a.reference,a.output,a.method,a.frames,a.multistarts)
