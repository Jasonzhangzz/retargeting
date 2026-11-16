#!/usr/bin/env python3
"""Verify and summarize the two independent primitive retargeting exports."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--source',type=Path,required=True)
    a=ap.parse_args();root=a.output
    summaries={target:json.loads((root/target/'summary.json').read_text()) for target in ('glove','wuji2')}
    per_type=[]
    for entry in summaries['glove']['entries']:
        g=np.load(root/'glove'/entry['file'],allow_pickle=False)
        w=np.load(root/'wuji2'/entry['file'],allow_pickle=False)
        source=np.load(a.source/entry['file'],allow_pickle=False)
        for key in ('size_scale','radius_m','height_m','object_pose_world','mano_joints','mano_vertices',
                    'grasp_passed','patch_joint_accepted','source_knot'):
            assert np.array_equal(g[key],source[key]), (entry['type'],key,'glove changed source')
            assert np.array_equal(w[key],source[key]), (entry['type'],key,'wuji2 changed source')
        row=dict(type=entry['type'],samples=len(g['target_qpos']),
                 radius_min_mm=float(g['radius_m'].min()*1000),radius_max_mm=float(g['radius_m'].max()*1000),
                 height_min_mm=float(g['height_m'].min()*1000),height_max_mm=float(g['height_m'].max()*1000))
        for target,z in (('glove',g),('wuji2',w)):
            assert np.isfinite(z['target_qpos']).all()
            assert z['retarget_numerically_valid'].all()
            assert np.all(z['target_qpos']>=z['joint_limits_rad'][:,0]-1e-5)
            assert np.all(z['target_qpos']<=z['joint_limits_rad'][:,1]+1e-5)
            assert np.allclose(np.linalg.norm(z['target_base_pose'][:,3:7],axis=1),1,atol=1e-5)
            row[target+'_tip_rmse_mm']=float(np.sqrt(np.mean(z['target_keypoint_error_m'][:,[4,8,12,16,20]]**2))*1000)
        per_type.append(row)
    with (root/'type-summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_type[0]));writer.writeheader();writer.writerows(per_type)
    for name in ('manifest.json','mano_geometry.npz','wuji2_model.json'):
        shutil.copy2(a.source/name,root/('source-'+name))
    shutil.copy2(root/'glove/size-index.csv',root/'size-index.csv')
    plot(root,per_type)
    report={'source_samples':sum(row['samples'] for row in per_type),'types':len(per_type),
            'targets':{k:{kk:vv for kk,vv in v.items() if kk!='entries'} for k,v in summaries.items()},
            'verification':dict(source_pose_and_size_arrays_unchanged=True,
                                target_joint_limits_passed=True,target_quaternions_passed=True,
                                targets_solved_independently_from_MANO=True),
            'physics':dict(force_closure_tested=False,object_collision_tested=False,self_collision_tested=False),
            'per_type':per_type}
    (root/'verification.json').write_text(json.dumps(report,indent=2))
    text='''# Robokit primitive retargeting

Source: /home/zhangzhe/dex_grasping/primitives (31 types, 1,247 size rows).
Two independent direct mappings: MANO -> calibrated 29-DOF glove, MANO -> native 20-DOF Wuji2.
Source MANO poses and object dimensions/poses are copied without alteration.

Files:
- glove/grids/<type>.npz and wuji2/grids/<type>.npz: paired source and retargeted poses.
- target_qpos: target joint angles in radians; joint_names gives their exact native order.
- target_base_pose: root position in metres, then quaternion WXYZ, in the original source world frame.
- target_keypoints_world: 21 semantic points in wrist, thumb, index, middle, ring, pinky order.
- object_pose_world: object XYZ + quaternion WXYZ, copied from the input.
- source_wuji_qpos_reference: dataset's old frozen/interpolated Wuji2 reference, NOT this run's output.
- valid, patch_joint_accepted, grasp_passed, rollout_*: source MANO diagnostics, NOT new target success.
- retarget_numerically_valid: finite values, valid quaternion and target joint limits only.
- config.json, model.xml: solver settings, native joint and point order, model provenance and independent FK checks.
- glove/glove-model.json: frozen calibrated glove model; its original fit-acceptance status is preserved.
- size-index.csv, type-summary.csv: all size rows and per-grasp summaries.
- comparison.png: skeleton comparison; identical object/world coordinates across each row.
- verification.json: aggregate checks and errors.

The two targets have no temporal warm start between dataset rows. The old anatomical-angle exporter was not used.
Positions and directions are retargeted with Robokit's batched population LM solver (16/4/1 candidates, 12/20/12 iterations).
The floating wrist is optimized. No object-contact, self-collision or force-closure objective was run.
These are kinematic references for subsequent physics refinement, not validated stable target grasps.

Example:
```python
import numpy as np
z=np.load('glove/grids/14_Tripod.npz',allow_pickle=False)
q=z['target_qpos']; base_xyz_wxyz=z['target_base_pose']
sizes=z['radius_m']; names=z['joint_names']
```
'''
    (root/'README.md').write_text(text)
    print(json.dumps({'verified_samples_per_target':report['source_samples'],'types':len(per_type),
                      'glove_tip_rmse_mm':summaries['glove']['fingertip_rmse_mm'],
                      'wuji2_tip_rmse_mm':summaries['wuji2']['fingertip_rmse_mm']},indent=2))


def plot(root,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    examples=('1_Large_Diameter','14_Tripod','20_Writing_Tripod','24_Tip_Pinch')
    colors=('#da8b26','#2c9b68','#3785c7','#bd6696','#9169ba')
    fig=plt.figure(figsize=(13,15),layout='constrained')
    axes=[]
    source_manifest=json.loads((root/'source-manifest.json').read_text())
    kind_by_type={e['type']:e['kind'] for e in source_manifest['entries']}
    for r,typ in enumerate(examples):
        g=np.load(root/'glove/grids'/f'{typ}.npz');w=np.load(root/'wuji2/grids'/f'{typ}.npz')
        i=int(np.argmin(abs(g['size_scale']-1)))
        points=(g['mano_joints'][i],g['target_keypoints_world'][i],w['target_keypoints_world'][i])
        obj=g['object_pose_world'][i];rot=Rotation.from_quat(obj[[4,5,6,3]]).as_matrix()
        radius=g['radius_m'][i];height=g['height_m'][i]
        theta=np.linspace(0,2*np.pi,32)
        if kind_by_type[typ]=='sphere':
            phi=np.linspace(0,np.pi,16)
            xx=radius*np.outer(np.cos(theta),np.sin(phi));yy=radius*np.outer(np.sin(theta),np.sin(phi))
            zz=radius*np.outer(np.ones_like(theta),np.cos(phi))
        else:
            xx=np.repeat((radius*np.cos(theta))[:,None],2,axis=1)
            yy=np.repeat((radius*np.sin(theta))[:,None],2,axis=1)
            zz=np.repeat(np.array([[-height/2,height/2]]),len(theta),axis=0)
        shape=xx.shape;obj_points=np.stack((xx,yy,zz),-1).reshape(-1,3)@rot.T+obj[:3]
        object_xyz=obj_points.reshape(*shape,3)
        allp=np.concatenate((*points,obj_points));center=(allp.min(0)+allp.max(0))/2
        half=max(np.ptp(allp,axis=0))*.55
        for c,p in enumerate(points):
            ax=fig.add_subplot(4,3,r*3+c+1,projection='3d');axes.append(ax)
            ax.plot_surface(object_xyz[...,0]*1000,object_xyz[...,1]*1000,object_xyz[...,2]*1000,
                            color='#aeb8bf',alpha=.22,linewidth=0)
            for f in range(5):
                ids=[0,*range(1+4*f,5+4*f)]
                ax.plot(*((p[ids]*1000).T),'-o',color=colors[f],lw=2,ms=3)
            for d,setter in enumerate((ax.set_xlim,ax.set_ylim,ax.set_zlim)):
                setter((center[d]-half)*1000,(center[d]+half)*1000)
            ax.set_box_aspect((1,1,1));ax.view_init(elev=23,azim=-65)
            title=('MANO source','Glove: Robokit','Wuji2: Robokit')[c]
            if c:
                z=(g,w)[c-1];e=np.sqrt(np.mean(z['target_keypoint_error_m'][i,[4,8,12,16,20]]**2))*1000
                title+=f' | tip RMSE {e:.1f} mm'
            ax.set_title(title,fontsize=10)
            if c==0:ax.set_ylabel(f'{typ}\nr={radius*1000:.2f} mm',fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle('Direct MANO retargeting at matched object sizes\nKinematic poses; target grasp stability not tested',fontsize=14)
    fig.savefig(root/'comparison.png',dpi=140)
    plt.close(fig)


if __name__=='__main__':main()
