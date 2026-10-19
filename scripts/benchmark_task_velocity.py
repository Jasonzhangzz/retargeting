#!/usr/bin/env python3
"""Matched terminal-hold comparison and a native control-gradient check."""
import argparse
from dataclasses import replace,fields
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def native_gradient_check(source):
    from forceaware.config import load_retarget_config
    from forceaware.runtime import setup_runtime
    from forceaware.workflow import _load_runtime_models,_collision_output_layout
    from forceaware.targets import build_index,load_sequence,make_window
    from retarget.fa.solver import SolverSettings,optimizer_class
    import comfree_warp
    root=ROOT/'data/example'
    cfg=load_retarget_config(root/'sequence',root/'forceaware.yaml',contactaware_dir=root/'contactaware')
    weights={field.name:0. for field in fields(cfg.loss)}
    weights.update(obj_vel=.1,obj_ang_vel=.03,reference_pose_scale=100.,terminal=2.)
    cfg=replace(cfg,loss=replace(cfg.loss,**weights),smoothing=replace(cfg.smoothing,enabled=False),
                optimizer=replace(cfg.optimizer,multistart_count=1,n_iter=1))
    setup_runtime(cfg);models=_load_runtime_models(cfg,comfree_warp);index=build_index(models.planner_cpu)
    body,enabled=_collision_output_layout(models.planner_device)
    sequence=load_sequence(cfg,output_body_ids=body,enabled_output_indices=enabled)
    solver=optimizer_class(SolverSettings(hand_solver='adamw',contact_radius=0.))(
        models.planner_cpu,models.planner_device,index,guidance_count=sequence.contact.mask.shape[1],cfg=cfg)
    targets=make_window(sequence,executed_steps=0,horizon=cfg.optimizer.horizon,time_grid=cfg.simulator.time_grid,
                        start_frame=71,end_frame=71)
    targets=replace(targets,object_velocity_world=np.tile([.02,-.01,.03,.1,-.2,.05],(len(targets.frames),1)))
    saved=np.load(source/'rollout.npz');step=237
    solver.backend._prepare_window(saved['qpos_traj'][step],saved['qvel_traj'][step],targets,
        warm=saved['raw_warm_history'][step],previous_action=saved['ctrl_traj'][step-1],seed=0,start_time=step*.01)
    raw=solver.backend.control.raw.numpy();variables=raw[:,index.action_dim:].astype(float)/solver.variable_scale
    solver._forward_variables=None;solver._objective_dirty=False;solver.evaluations=0
    value,gradient=solver._evaluate(variables,True)
    rows=[]
    for coordinate in (2,3,6,4*26+2):
        numeric=[]
        for eps in (.02,.01):
            plus,minus=variables.copy(),variables.copy();plus[0,coordinate]+=eps;minus[0,coordinate]-=eps
            vp,_=solver._evaluate(plus,False);vm,_=solver._evaluate(minus,False)
            numeric.append(float((vp[0]-vm[0])/(2*eps)))
        analytic=float(gradient[0,coordinate])
        rows.append(dict(coordinate=coordinate,analytic=analytic,finite_difference=numeric,
            relative_error=abs(analytic-numeric[-1])/max(abs(analytic),abs(numeric[-1]),1e-8)))
    assert max(row['relative_error'] for row in rows)<.03
    return dict(objective=float(value[0]),only_velocity_tracking=True,rows=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'outputs/comfree/adamw_20261019_145931_413461')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/task_velocity_benchmark')
    parser.add_argument('--duration',type=float,default=1.2)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    report=dict(hold_cases={})
    environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    with tempfile.TemporaryDirectory(prefix='retarget_velocity_benchmark_') as tmp:
        prepared=Path(tmp);root=ROOT/'data/example'
        for name in ('sequence','contactaware','demonstration_visual.npz'):(prepared/name).symlink_to(root/name)
        for enabled in (False,True):
            label='with_velocity' if enabled else 'without_velocity'
            config=yaml.safe_load((root/'forceaware.yaml').read_text())
            config['sequence'].update(start=70,hold_frame=71,hold_seconds=args.duration)
            if not enabled:config['loss'].update(obj_vel=0.,obj_ang_vel=0.)
            (prepared/'forceaware.yaml').write_text(yaml.safe_dump(config))
            command=[sys.executable,str(ROOT/'scripts/audit_contact_hold.py'),'--worker','baseline',
                '--experiment',str(prepared),'--source',str(args.source),'--output',str(args.output/label),
                '--duration',str(args.duration)]
            with (args.output/(label+'.log')).open('w') as log:
                subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
            saved=np.load(args.output/label/'runs/comfree/baseline/rollout.npz')
            v=saved['qvel_traj']
            report['hold_cases'][label]=dict(position_error_end_mm=float(saved['drift'][-1]*1000),
                orientation_error_end_deg=float(saved['rot_err_deg'][-1]),
                linear_speed_rms_m_s=float(np.sqrt(np.mean(np.sum(v[:,26:29]**2,axis=1)))),
                angular_speed_rms_rad_s=float(np.sqrt(np.mean(np.sum(v[:,29:32]**2,axis=1)))))
            if enabled:
                assert np.count_nonzero(saved['object_reference_velocity_world'])==0
            print(label,json.dumps(report['hold_cases'][label]),flush=True)
            (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    report['native_gradient']=native_gradient_check(args.source)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('native_gradient',json.dumps(report['native_gradient']),flush=True)


if __name__=='__main__':main()
