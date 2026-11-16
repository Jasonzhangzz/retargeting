"""Paired source-noise, contact-change and motion replay; no labels enter CEM."""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from retarget.baseline.card.math import CardSettings
from retarget.baseline.card.retarget import TactileRetarget
from moxian_web.wuji_surface import CHAINS


def deform(frame,topology,rotations,translation=(0,0,0)):
    f=copy.deepcopy(frame);v=np.array(f['vertices_m']);j=np.array(f['joints_m'])
    skin=np.array(topology['skin_weights']);displacement=np.zeros_like(v)
    for finger,chain in enumerate(CHAINS):
        ids=list(range(1+4*finger,5+4*finger));pivot=j[ids[0]];r=Rotation.from_rotvec(rotations[finger]).as_matrix()
        displacement+=((v-pivot)@r.T+pivot-v)*skin[:,chain].sum(-1)[:,None]
        j[ids]=(j[ids]-pivot)@r.T+pivot
    f['vertices_m']=(v+displacement+translation).tolist();f['joints_m']=(j+translation).tolist()
    return f


def sequence(source):
    original=source['frames'][len(source['frames'])//2];rng=np.random.default_rng(71);frames=[]
    for _ in range(18):frames.append(('static',copy.deepcopy(original)))
    for _ in range(20):frames.append(('finger_noise',deform(original,source['topology'],rng.normal(0,.0008,(5,3)))))
    # A measurable current-source gesture, followed immediately by release.
    for amount in [.04,.08,.12,.12,.12,0,0,0]:
        rotations=np.zeros((5,3));rotations[3,0]=amount
        frames.append(('gesture' if amount else 'gesture_return',deform(original,source['topology'],rotations)))
    for _ in range(4):
        f=copy.deepcopy(original);f['source_contact']['cell_contact']=[False]*460;frames.append(('contact_off',f))
    for _ in range(4):frames.append(('contact_on',copy.deepcopy(original)))
    for _ in range(3):frames.append(('translate_50mm',deform(original,source['topology'],np.zeros((5,3)),(.05,0,0))))
    for index,(_,f) in enumerate(frames):
        stamp=original['timestamp_ms']+index*100;f.update(timestamp_ms=stamp,tactile_timestamp_ms=stamp)
        f['source_contact']['timestamp_ms']=stamp
    return frames


def motion(points):
    points=np.array(points);segments=np.diff(points[:,1:].reshape(-1,5,4,3),axis=2)
    directions=segments/np.linalg.norm(segments,axis=-1,keepdims=True)
    angular=np.rad2deg(np.arccos(np.clip((directions[1:]*directions[:-1]).sum(-1),-1,1))).max((1,2))
    position=np.linalg.norm(np.diff(points[:,[4,8,12,16,20]],axis=0),axis=-1).max(-1)*1000
    return angular,position


def summarize(rows):
    q=np.array([r['qpos'] for r in rows]);step=np.rad2deg(np.abs(np.diff(q[:,6:],axis=0))).max(-1)
    source_angle,source_position=motion([r['source_joints_m'] for r in rows])
    robot_angle,robot_position=motion([r['robot_keypoints_m'] for r in rows]);summary={}
    for stage in dict.fromkeys(r['stage'] for r in rows):
        ids=np.flatnonzero([r['stage']==stage for r in rows]);d=ids[ids>0]-1;selected=[rows[i] for i in ids]
        result={'frames':len(ids),'feasible':sum(r['solution_feasible'] for r in selected),
            'max_penetration_mm':max(r['collision']['self_mesh']['max_penetration_mm'] for r in selected)}
        for name,values in [('joint_step_deg',step[d]),('source_bone_step_deg',source_angle[d]),
                            ('robot_bone_step_deg',robot_angle[d]),('source_tip_step_mm',source_position[d]),
                            ('robot_tip_step_mm',robot_position[d])]:
            result[name]={'max':float(values.max()),'p95':float(np.percentile(values,95)),'median':float(np.median(values))} if len(values) else None
        error=[r['quality']['contact_rmse_mm'] for r in selected if r['quality']['contact_rmse_mm'] is not None]
        result['contact_rmse_mean_mm']=float(np.mean(error)) if error else None
        result['wrist_displacement_mm']=(np.array(selected[-1]['qpos'][:3])-np.array(rows[max(0,ids[0]-1)]['qpos'][:3])).tolist()
        result['wrist_displacement_mm']=[1000*x for x in result['wrist_displacement_mm']]
        summary[stage]=result
    return summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,nargs='+',required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--variants',nargs='+',choices=['baseline','style_only','motion_global','online'],default=['baseline','online'])
    ap.add_argument('--recorded',action='store_true',help='Replay every saved real frame, without synthetic perturbations')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    variants={'baseline':replace(CardSettings(),wmotion_joint=0,contact_style_weight=0,motion_sample_fraction=0),
              'style_only':replace(CardSettings(),wmotion_joint=0,motion_sample_fraction=0),
              'motion_global':replace(CardSettings(),wmotion_joint=300.,motion_sample_fraction=0),'online':CardSettings()}
    manifest={'sources':{},'variants':{k:variants[k].payload() for k in args.variants},'source_pose_noise_rad_per_axis':None if args.recorded else .0008,
              'labels_used_in_solver':False,'budget':[1024,16],'units':'m/rad'};summary={}
    for path in args.source:
        raw=path.read_bytes();source=json.loads(raw)
        frames=[('recorded',copy.deepcopy(f)) for f in source['frames']] if args.recorded else sequence(source)
        manifest['sources'][str(path)]=hashlib.sha256(raw).hexdigest();summary[path.stem]={}
        for variant in args.variants:
            o=TactileRetarget(source['topology'],args.output/'mappings',settings=variants[variant]);rows=[]
            for stage,f in frames:
                r=o.step(f)
                local=o.adapter.keypoints(o.tensor(np.r_[np.zeros(6),r['joint_positions_rad']])).cpu().numpy()
                world=local@np.array(r['wrist_rotation']).T+np.array(r['wrist_position_m'])
                rows.append({'stage':stage,'robot_keypoints_m':world.tolist(),**{k:r[k] for k in (
                    'qpos','robokit_qpos','source_joints_m','contact_anchors','solution_feasible','quality','collision','solver')}})
            (args.output/f'{path.stem}-{variant}.json').write_text(json.dumps(rows,allow_nan=False))
            summary[path.stem][variant]=summarize(rows)
            print(json.dumps({'source':path.stem,'variant':variant,'summary':summary[path.stem][variant]},ensure_ascii=False),flush=True)
            del o
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()
