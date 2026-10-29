#!/usr/bin/env python3
"""Validate fresh SPIDER playback and independently replay solver controls."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp

from run_spider_reference import compatible_mjwp


def source_replay(reference):
    meta=json.loads((reference/'summary.json').read_text());a=np.load(reference/'trajectory.npz')
    cfg=SimpleNamespace(**meta['config']);source=compatible_mjwp()
    model=source.setup_mj_model(cfg);data=mujoco.MjData(model)
    data.qpos[:]=a['checkpoint_qpos'][0];data.qvel[:]=a['checkpoint_qvel'][0]
    data.qacc_warmstart[:]=a['checkpoint_qacc_warmstart'][0];data.time=float(a['checkpoint_time'][0])
    wp.init();wp.set_device('cuda:0')
    m=mjwarp.put_model(model);d=mjwarp.put_data(model,data,nworld=1,nconmax=100,njmax=350)
    mjwarp.step(m,d)
    with wp.ScopedCapture(device='cuda:0') as cap:mjwarp.step(m,d)
    d.qpos.assign(a['checkpoint_qpos'][:1].astype(np.float32));d.qvel.assign(a['checkpoint_qvel'][:1].astype(np.float32))
    d.qacc_warmstart.assign(a['checkpoint_qacc_warmstart'][:1].astype(np.float32));d.time.fill_(float(a['checkpoint_time'][0]))
    qs=[d.qpos.numpy()[0]];vs=[d.qvel.numpy()[0]]
    for control in a['ctrl']:
        d.ctrl.assign(control[None].astype(np.float32));wp.capture_launch(cap.graph)
        qs.append(d.qpos.numpy()[0]);vs.append(d.qvel.numpy()[0])
    q,v=np.asarray(qs),np.asarray(vs)
    root=np.linalg.norm(q[:,:3]-a['qpos'][:,:3],axis=1)
    result=dict(steps=len(a['ctrl']),root_error_max_m=float(root.max()),
                qpos_abs_max=float(np.max(abs(q-a['qpos']))),qvel_abs_max=float(np.max(abs(v-a['qvel']))),
                note='Independent one-world MJWarp continuous playback from saved initial state and qacc_warmstart; reference used 2048 sampling worlds.')
    local_errors=[];stride=meta['config']['ctrl_steps']
    for i in range(len(a['checkpoint_qpos'])):
        d.qpos.assign(a['checkpoint_qpos'][i:i+1].astype(np.float32));d.qvel.assign(a['checkpoint_qvel'][i:i+1].astype(np.float32))
        d.qacc_warmstart.assign(a['checkpoint_qacc_warmstart'][i:i+1].astype(np.float32));d.time.fill_(float(a['checkpoint_time'][i]))
        maximum=0.
        for j in range(stride):
            d.ctrl.assign(a['ctrl'][i*stride+j:i*stride+j+1].astype(np.float32));wp.capture_launch(cap.graph)
            maximum=max(maximum,float(np.linalg.norm(d.qpos.numpy()[0,:3]-a['qpos'][i*stride+j+1,:3])))
        local_errors.append(maximum)
    result.update(checkpoint_windows=len(local_errors),checkpoint_window_root_error_median_m=float(np.median(local_errors)),
                  checkpoint_window_root_error_max_m=max(local_errors),checkpoint_window_root_errors_m=local_errors)
    np.savez_compressed(reference/'independent_replay.npz',qpos=q,qvel=v)
    (reference/'replay_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(reference,result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference',type=Path)
    a=p.parse_args()
    source_replay(a.reference)
