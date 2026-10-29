#!/usr/bin/env python3
"""Run the pinned SPIDER sampling optimizer headlessly and save test windows.

Unlike replaying released controls, this optimizes closed loop in the installed
MJWarp. Missing public contact labels are explicitly excluded from the reward.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'data/spider/source'))
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
import torch
import warp as wp
import yaml
from spider.config import Config, process_config
from spider.io import load_data
from spider.optimizers.sampling import make_rollout_fn, make_optimize_once_fn, make_optimize_fn
from spider.simulators import mjwp


def compatible_mjwp():
    """Use the same dtype-only Warp API adapter as the strict ADMM runner."""
    from retarget.spider_source import modules
    return modules().mjwp


def run(task, output, samples=2048, iterations=16, windows=None):
    mjwp = compatible_mjwp()
    override = 'humanoid' if task=='humanoid' else 'humanoid_object'
    values = yaml.safe_load((ROOT/f'data/spider/source/examples/config/override/{override}.yaml').read_text())
    values.update(dataset_dir=str(ROOT/'data/spider'), contact_rew_scale=0., use_torch_compile=False,
                  num_samples=samples, max_num_iterations=iterations, show_viewer=False,
                  save_video=False, save_config=False, sanity_check_seconds=0.,
                  num_trace_uniform_samples=0, num_trace_topk_samples=0)
    config = process_config(Config(**values)); config.contact_site_ids=[]
    torch.set_num_threads(1)
    refs = load_data(config, config.data_path)
    env = mjwp.setup_env(config, refs)
    # Same one-DR-member schedule as examples/run_mjwp.py (linspace(...,1)).
    config.env_params_list = [[dict(pair_margin=config.pair_margin_range[0], xy_offset=config.xy_offset_range[0])]
                             for _ in range(config.max_num_iterations)]
    rollout = make_rollout_fn(mjwp.step_env,mjwp.save_state,mjwp.load_state,mjwp.get_reward,
        mjwp.get_terminal_reward,mjwp.get_terminate,mjwp.get_trace,mjwp.save_env_params,
        mjwp.load_env_params,mjwp.copy_sample_state)
    optimize = make_optimize_fn(make_optimize_once_fn(rollout))
    count = int(np.ceil((len(refs[0])-config.horizon_steps-config.ctrl_steps)/config.ctrl_steps))
    count = count if windows is None else min(windows,count)
    output.mkdir(parents=True, exist_ok=False)
    controls = refs[2][:config.horizon_steps].clone()
    host = lambda a: a.numpy()[0].copy()
    qs, vs = [host(env.data_wp.qpos)], [host(env.data_wp.qvel)]
    applied, states, velocities, warmstarts, previous, nominal, planned, times, logs = [], [], [], [], [], [], [], [], []
    started = time.perf_counter()
    for window in range(count):
        sim_step=window*config.ctrl_steps
        states.append(host(env.data_wp.qpos));velocities.append(host(env.data_wp.qvel))
        warmstarts.append(host(env.data_wp.qacc_warmstart));previous.append(host(env.data_wp.ctrl))
        nominal.append(controls.cpu().numpy().copy());times.append(float(env.data_wp.time.numpy()[0]))
        ref_slice=tuple(a[sim_step+1:sim_step+config.horizon_steps+1] for a in refs)
        torch.cuda.synchronize();tick=time.perf_counter()
        controls,info=optimize(config,env,controls,ref_slice)
        torch.cuda.synchronize();seconds=time.perf_counter()-tick
        planned.append(controls.cpu().numpy().copy())
        for i in range(config.ctrl_steps):
            mjwp.step_env(config,env,controls[i])
            qs.append(host(env.data_wp.qpos));vs.append(host(env.data_wp.qvel))
            applied.append(controls[i].cpu().numpy().copy())
        mjwp.sync_env(config,env,None)
        next_step=sim_step+config.ctrl_steps
        retained=controls[config.ctrl_steps:]
        controls=torch.cat((retained,refs[2][next_step+len(retained):next_step+config.horizon_steps]),dim=0)
        logs.append(dict(window=window,sim_step=sim_step,seconds=seconds,opt_steps=int(info['opt_steps'][0]),
                         reward_before=float(info['rew_mean'][0]),reward_best=float(np.max(info['rew_max'])),
                         root_height=float(qs[-1][2])))
        print(task,window+1,'/',count,'height',qs[-1][2],'seconds',seconds,flush=True)
        # Keep completed checkpoints if a later window or inspection fails.
        np.savez_compressed(output/'trajectory.npz',qpos=qs,qvel=vs,ctrl=applied,
                            checkpoint_qpos=states,checkpoint_qvel=velocities,
                            checkpoint_qacc_warmstart=warmstarts,checkpoint_previous_ctrl=previous,
                            checkpoint_time=times,nominal_controls=nominal,planned_controls=planned,
                            qpos_ref=refs[0].cpu().numpy(),qvel_ref=refs[1].cpu().numpy(),ctrl_ref=refs[2].cpu().numpy())
        (output/'windows.json').write_text(json.dumps(logs,indent=2)+'\n')
    # Main loop uses ref_slice starting at sim_step+1, as in the upstream runner.
    aligned=refs[0][:len(qs)].cpu().numpy()
    d=mujoco.MjData(env.model_cpu);dr=mujoco.MjData(env.model_cpu);torso=env.model_cpu.body('torso_link').id
    tilt=[]
    for q,r in zip(qs,aligned):
        d.qpos[:]=q;dr.qpos[:]=r;mujoco.mj_forward(env.model_cpu,d);mujoco.mj_forward(env.model_cpu,dr)
        tilt.append(float(np.degrees(np.arccos(np.clip(d.xmat[torso].reshape(3,3)[:,2]@dr.xmat[torso].reshape(3,3)[:,2],-1,1)))))
    deficit=aligned[:,2]-np.array(qs)[:,2]
    config_dict={k:v for k,v in asdict(config).items() if k not in ('noise_scale','env_params_list')}
    result=dict(task=task,source_commit='44717007de41cbef7565dff7ff9f4453557a2d3d',config=config_dict,
        contact_reward_disabled_reason='Public NPZ lacks contact/contact_pos labels.',
        source_sampling_and_physics_unmodified=True,torch_compile=False,
        api_adapter='Reinterpret from_torch scalar arrays as the destination vec/mat dtype when copying; identical numerical values.',
        completed_windows=count,executed_physics_steps=len(applied),seconds=time.perf_counter()-started,
        root_height_min_m=float(np.min(np.array(qs)[:,2])),torso_tilt_max_deg=max(tilt),
        root_position_rms_m=float(np.sqrt(np.mean(np.sum((np.array(qs)[:,:3]-aligned[:,:3])**2,axis=1)))),
        balance_failure_frames=int(np.sum((deficit>.18)|(np.array(tilt)>60))),
        feasible_under_balance_audit=bool(np.isfinite(qs).all() and not np.any((deficit>.18)|(np.array(tilt)>60))),
        versions=dict(mujoco=mujoco.__version__,warp=wp.__version__,torch=torch.__version__),
        note='Balance audit is an operational filter; not proof of hard contact/torque feasibility. No reference state is injected after initialization.')
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=['humanoid','humanoid-object'],default='humanoid')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=2048)
    p.add_argument('--iterations',type=int,default=16)
    p.add_argument('--windows',type=int)
    a=p.parse_args();run(a.task,a.output,a.samples,a.iterations,a.windows)
