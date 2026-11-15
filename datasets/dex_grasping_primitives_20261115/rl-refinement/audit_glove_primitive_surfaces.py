#!/usr/bin/env python3
"""Inspect calibrated Moxian skin against frozen primitive objects."""
import argparse,importlib.util,json,sys,csv
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--poses',type=Path,required=True);a=ap.parse_args()
    root=a.poses.resolve();repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo))
    for name in ('bone_extrinsics','qnbot_surface'):
        path=root/f'model-code-snapshot/{name}.py'
        spec=importlib.util.spec_from_file_location(f'moxian_web.{name}',path)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    from moxian_web.qnbot_surface import QnbotSurface
    resources=list((repo/'.venv-moxian-ik/lib').glob('python*/site-packages/qnbot_sdk/glove/_private/algorithms/skeleton/*/share/qnbot_hand/hands/dynamic_openxr_hand'))
    assert len(resources)==1
    artifact=json.load(open(root/'glove/glove-model.json'))
    model=QnbotSurface('right','cuda:0',root=resources[0],finger_twist=True);model.load_artifact(artifact)
    manifest=json.load(open(root/'source-manifest.json'));rows=[]
    with torch.no_grad():
        for entry in manifest['entries']:
            if entry['kind']=='plate':continue
            z=np.load(root/'glove'/entry['file']);q=torch.tensor(z['target_qpos'],device='cuda:0')
            v,j,_=model(q);v=v.cpu().numpy()
            base=z['target_base_pose'];obj=z['object_pose_world']
            br=Rotation.from_quat(base[:,[4,5,6,3]]).as_matrix();orr=Rotation.from_quat(obj[:,[4,5,6,3]]).as_matrix()
            world=np.einsum('bvi,bji->bvj',v,br)+base[:,None,:3]
            local=np.einsum('bvi,bij->bvj',world-obj[:,None,:3],orr)
            rad=z['radius_m'][:,None];height=z['height_m'][:,None]
            if entry['kind']=='sphere':sdf=np.linalg.norm(local,axis=-1)-rad
            else:
                d=np.stack((np.linalg.norm(local[...,:2],axis=-1)-rad,np.abs(local[...,2])-height/2),-1)
                sdf=np.linalg.norm(np.maximum(d,0),axis=-1)+np.minimum(np.max(d,axis=-1),0)
            for i in range(len(q)):
                hull=ConvexHull(z['target_keypoints_world'][i]);inside=bool(np.max(hull.equations[:,:3]@obj[i,:3]+hull.equations[:,3])<=.005)
                rows.append(dict(type=entry['type'],row=i,center_in_keypoint_hull_5mm=inside,
                                 max_vertex_penetration_mm=float(max(0,-sdf[i].min())*1000),
                                 skin_vertices_within_2mm=int(np.sum(np.abs(sdf[i])<.002))))
            if entry['type']=='10_Power_Disk':
                np.savez_compressed(root/'glove-selected-surface.npz',vertices_world=world[27],faces=model.faces.cpu().numpy(),
                                    object_pose_world=obj[27],radius_m=z['radius_m'][27],height_m=z['height_m'][27])
    out=root/'physics-audit-final'
    with (out/'glove-surface-audit.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(audited=len(rows),center_in_keypoint_hull=sum(r['center_in_keypoint_hull_5mm'] for r in rows),
                 vertex_penetration_over_3mm=sum(r['max_vertex_penetration_mm']>3 for r in rows),
                 model_hash=artifact['model_hash'],region_revision=artifact['region_revision'],
                 note='Skin vertex SDF is a surface sampling diagnostic, not triangle intersection or grasp stability certification.')
    (out/'glove-surface-audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
