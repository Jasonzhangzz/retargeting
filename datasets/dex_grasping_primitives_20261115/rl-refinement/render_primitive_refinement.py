#!/usr/bin/env python3
"""Render frozen MANO, calibrated Moxian skin, and native Wuji2 mesh poses."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def hand_mesh(q,base):
    import mujoco
    from retarget.env.grasp.hand import hand_tree
    import xml.etree.ElementTree as ET
    model=mujoco.MjModel.from_xml_string(ET.tostring(hand_tree('wuji2'),encoding='unicode'));data=mujoco.MjData(model)
    data.qpos[:6]=np.r_[base[:3],Rotation.from_quat(base[[4,5,6,3]]).as_euler('XYZ')]
    data.qpos[6:26]=q;mujoco.mj_forward(model,data)
    vertices=[];faces=[];offset=0
    for g in range(model.ngeom):
        if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH or not (model.geom_contype[g] or model.geom_conaffinity[g]):continue
        mesh=model.geom_dataid[g];start=model.mesh_vertadr[mesh];n=model.mesh_vertnum[mesh]
        v=model.mesh_vert[start:start+n]@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
        fstart=model.mesh_faceadr[mesh];nf=model.mesh_facenum[mesh]
        vertices.append(v);faces.append(model.mesh_face[fstart:fstart+nf]+offset);offset+=n
    return np.concatenate(vertices),np.concatenate(faces)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import trimesh
    ap=argparse.ArgumentParser();ap.add_argument('--poses',type=Path,required=True);ap.add_argument('--experiment',type=Path,required=True);a=ap.parse_args()
    root=a.poses;g=np.load(root/'glove-selected-surface.npz');source=np.load(root/'wuji2/grids/10_Power_Disk.npz')
    faces=np.load(root/'source-mano_geometry.npz')['faces']
    verify=json.load(open(a.experiment/'independent-closure-check.json'))
    best=min(verify['learned']['rows'],key=lambda r:(not r['candidate'],r['native_mesh_penetration_mm']))
    index=best['env'];learned=np.load(a.experiment/'learned-evaluation.npz')
    task_names=[f'right_finger{f}_joint{j}' for f in range(1,6) for j in range(1,5)]
    ids=[learned['joint_names'].tolist().index(n) for n in task_names]
    initialmesh=hand_mesh(source['target_qpos'][27],source['target_base_pose'][27])
    refinedmesh=hand_mesh(learned['target_qpos'][index,ids],learned['target_base_pose'][index])
    panels=[('MANO source',source['mano_vertices'][27],faces,source['object_pose_world'][27]),
            ('Uploaded Moxian glove',g['vertices_world'],g['faces'],source['object_pose_world'][27]),
            ('Wuji2: kinematic reference',*initialmesh,source['object_pose_world'][27]),
            (f'Wuji2: RL refinement (env {index})',*refinedmesh,learned['object_pose_world'][index])]
    allpoints=np.concatenate([x[1] for x in panels]);center=(allpoints.min(0)+allpoints.max(0))/2;half=np.ptp(allpoints,axis=0).max()*.55
    fig=plt.figure(figsize=(12,11),layout='constrained')
    for i,(name,v,f,obj) in enumerate(panels):
        ax=fig.add_subplot(2,2,i+1,projection='3d')
        coll=Poly3DCollection(v[f]*1000,facecolor=('#c7a586' if i<2 else '#67829c'),edgecolor='none',alpha=.75,rasterized=True)
        ax.add_collection3d(coll)
        mesh=trimesh.creation.cylinder(radius=float(source['radius_m'][27]),height=float(source['height_m'][27]),sections=48)
        ov=mesh.vertices@Rotation.from_quat(obj[[4,5,6,3]]).as_matrix().T+obj[:3]
        ax.add_collection3d(Poly3DCollection(ov[mesh.faces]*1000,facecolor='#efb43c',edgecolor='none',alpha=.65,rasterized=True))
        for k,setter in enumerate((ax.set_xlim,ax.set_ylim,ax.set_zlim)):setter((center[k]-half)*1000,(center[k]+half)*1000)
        ax.set_box_aspect((1,1,1));ax.view_init(elev=25,azim=-62);ax.set_title(name)
        ax.set_xlabel('X (mm)');ax.set_ylabel('Y (mm)');ax.set_zlabel('Z (mm)');ax.tick_params(labelsize=7)
    fig.suptitle('Power disk: diameter 77.5 mm, thickness 15 mm\nActual hand meshes; individual pose geometry does not certify general grasp success',fontsize=13)
    fig.savefig(a.experiment/'mesh-comparison.png',dpi=130);plt.close(fig)


if __name__=='__main__':main()
