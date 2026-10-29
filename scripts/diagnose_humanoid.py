#!/usr/bin/env python3
"""Audit balance on executed states, separately from future optimizer costs."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse
from pathlib import Path
import sys,json
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from retarget.humanoid import HumanoidProblem,Weights,Settings


def audit(case,output,render=False):
    config=json.loads((case/'config.json').read_text())
    p=HumanoidProblem(config['scene'],config['reference'],weights=Weights(**config['weights']),settings=Settings(**{k:v for k,v in config['settings'].items() if k in Settings.__dataclass_fields__}))
    rows=[json.loads(line) for line in (case/'iterations.jsonl').read_text().splitlines()]
    result=np.load(case/'rollout.npz')
    frames=np.r_[result['reference_frames'][0]-1,result['reference_frames']]
    d=mujoco.MjData(p.model);ref=mujoco.MjData(p.model);records=[]
    for j,frame in enumerate(frames):
        d.qpos[:]=result['qpos'][j];d.qvel[:]=result['qvel'][j]
        ref.qpos[:]=p.qref[frame];ref.qvel[:]=p.tracking_vref[frame]
        for data in (d,ref):mujoco.mj_forward(p.model,data);mujoco.mj_subtreeVel(p.model,data)
        dc=d.subtree_com[p.root]-ref.subtree_com[p.root]
        dv=d.subtree_linvel[p.root]-ref.subtree_linvel[p.root]
        up=d.xmat[p.torso].reshape(3,3)[:,2];upr=ref.xmat[p.torso].reshape(3,3)[:,2]
        features=p.features(d)
        record=dict(frame=int(frame),time=float(j*p.ref_dt),root_height=float(d.qpos[2]),reference_height=float(ref.qpos[2]),
                    com_error=dc.tolist(),com_velocity_error=dv.tolist(),com_velocity=d.subtree_linvel[p.root].tolist(),
                    torso_tilt_rad=float(np.arccos(np.clip(up@upr,-1,1))),
                    instant_balance_cost=float(p.weights.balance*(dc@dc)),
                    instant_upright_cost=float(p.weights.upright*np.sum((up-upr)**2)),
                    instant_keypoint_cost=float(p.weights.keypoint_position*np.sum((d.xpos[p.bodies]-ref.xpos[p.bodies])**2)),
                    reference_stance=p.mask[frame,2:].tolist(),actual_foot_gap=(features['contacts'][2:,2]-p.ground_height).tolist())
        if config.get('objective_version') in ('normalized_com_velocity_height_v2','normalized_balance_realized_contact_v3','relative_keypoints_realized_balance_v4','relative_keypoints_realized_balance_v5','relative_keypoints_realized_balance_v6','world_hand_dynamic_wrench_v7'):
            s=p.settings
            dcm=dc+dv/np.sqrt(9.81/max(ref.subtree_com[p.root,2]-p.ground_height,.2))
            parts=np.array([np.sum(dc[:2]**2)/s.balance_xy_scale**2,dc[2]**2/s.balance_z_scale**2,
                            .5*np.sum(dcm[:2]**2)/s.balance_xy_scale**2,
                            .25*dv[2]**2/s.balance_velocity_scale**2,
                            (max(ref.qpos[2]-d.qpos[2]-s.balance_height_margin,0)/s.balance_height_scale)**2])
            record['instant_balance_cost']=float(p.weights.balance*parts.sum())
            record['balance_parts']=dict(zip(('com_xy','com_height','dcm','vertical_velocity','collapse'),(p.weights.balance*parts).tolist()))
            record['instant_upright_cost']/=s.upright_scale**2
            record['velocity_audit']='exact subtree kinematic velocity; optimizer uses adjacent predicted COM differences'
        if config.get('objective_version') in ('relative_keypoints_realized_balance_v4','relative_keypoints_realized_balance_v5','relative_keypoints_realized_balance_v6','world_hand_dynamic_wrench_v7'):
            actual=d.xpos[p.bodies]-d.xpos[p.root]
            desired=ref.xpos[p.bodies]-ref.xpos[p.root]
            record['instant_global_keypoint_cost']=record['instant_keypoint_cost']
            record['instant_keypoint_cost']=float(p.weights.keypoint_position*np.sum((actual-desired)**2))
        if j:record['predicted_horizon_costs']=rows[j-1]['iterations'][-1]['costs']
        records.append(record)
    output.mkdir(parents=True,exist_ok=True)
    (output/'balance_audit.json').write_text(json.dumps(records,indent=2)+'\n')
    if render:
        import matplotlib.pyplot as plt
        selected=np.unique(np.linspace(0,len(frames)-1,6).astype(int))
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        renderer=mujoco.Renderer(p.model,height=360,width=480)
        for ax,i in zip(axes.flat,selected):
            d.qpos[:]=result['qpos'][i];mujoco.mj_forward(p.model,d)
            camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:]=[p.qref[frames[i],0],p.qref[frames[i],1],.5]
            camera.distance=2.6;camera.azimuth=135;camera.elevation=-15
            renderer.update_scene(d,camera=camera)
            ax.imshow(renderer.render());ax.axis('off')
            r=records[i];ax.set_title(f'frame {frames[i]}: h={r["root_height"]:.2f} m, tilt={np.degrees(r["torso_tilt_rad"]):.0f} deg\nbalance={r["instant_balance_cost"]:.2f}, keypoints={r["instant_keypoint_cost"]:.1f}')
        fig.tight_layout();fig.savefig(output/'executed_states.png',dpi=130);plt.close(fig);renderer.close()
        fig,axs=plt.subplots(3,1,figsize=(10,9),sharex=True)
        t=np.array([r['time'] for r in records])
        for key in ('instant_balance_cost','instant_upright_cost','instant_keypoint_cost'):axs[0].plot(t,[r[key] for r in records],label=key)
        axs[0].legend();axs[0].set_ylabel('executed cost')
        axs[1].plot(t,[r['root_height'] for r in records],label='root height');axs[1].plot(t,[r['reference_height'] for r in records],label='reference');axs[1].legend();axs[1].set_ylabel('m')
        axs[2].plot(t,[np.linalg.norm(r['com_error']) for r in records],label='COM error [m]');axs[2].plot(t,[np.linalg.norm(r['com_velocity_error']) for r in records],label='COM velocity error [m/s]');axs[2].legend();axs[2].set_xlabel('seconds')
        fig.tight_layout();fig.savefig(output/'balance_trace.png',dpi=130);plt.close(fig)
    for i in np.unique(np.linspace(0,len(records)-1,7).astype(int)):
        r=records[i];print({k:r[k] for k in ['frame','root_height','torso_tilt_rad','instant_balance_cost','instant_upright_cost','instant_keypoint_cost','com_velocity_error']})
    return records

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('case',type=Path);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--render',action='store_true');args=parser.parse_args();audit(args.case,args.output,args.render)
