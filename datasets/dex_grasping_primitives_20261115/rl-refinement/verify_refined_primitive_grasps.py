#!/usr/bin/env python3
"""Independently check PPO-evaluated poses with native Wuji2 meshes and cone LP."""
import argparse,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_primitive_grasp_initials import closure


def main():
    import mujoco
    from retarget.env.grasp.hand import hand_tree
    ap=argparse.ArgumentParser();ap.add_argument('--experiment',type=Path,required=True);a=ap.parse_args()
    meta=json.load(open(a.experiment/'experiment.json'));tree=hand_tree('wuji2')
    body=ET.SubElement(tree.find('worldbody'),'body',name='verify_object');ET.SubElement(body,'freejoint',name='verify_object_joint')
    typ='sphere' if meta['kind']=='sphere' else 'cylinder';size=str(meta['radius']) if typ=='sphere' else f"{meta['radius']} {meta['height']/2}"
    ET.SubElement(body,'geom',name='verify_object_geom',type=typ,size=size,mass=str(meta['mass']),friction='.8 .005 .0001',condim='4')
    model=mujoco.MjModel.from_xml_string(ET.tostring(tree,encoding='unicode'));data=mujoco.MjData(model);gid=model.geom('verify_object_geom').id
    native_names=[model.joint(i).name for i in range(6,26)]
    # Explicit semantic name map, identical to the upstream task conversion.
    task_names=[f'right_finger{f}_joint{j}' for f in range(1,6) for j in range(1,5)]
    summary={};saved={}
    for label in ('baseline','learned'):
        z=np.load(a.experiment/f'{label}-evaluation.npz');ids=[z['joint_names'].tolist().index(n) for n in task_names]
        rows=[]
        for i in range(len(z['target_qpos'])):
            q=z['target_qpos'][i,ids];base=z['target_base_pose'][i];obj=z['object_pose_world'][i]
            data.qpos[:6]=np.r_[base[:3],Rotation.from_quat(base[[4,5,6,3]]).as_euler('XYZ')]
            data.qpos[6:26]=q;data.qpos[26:33]=obj;data.qvel[:]=0;mujoco.mj_forward(model,data)
            contacts=[];penetration=0.;self_pen=0.
            for c in data.contact:
                if gid in (c.geom1,c.geom2):
                    penetration=max(penetration,-float(c.dist));frame=c.frame.reshape(3,3)
                    if c.dist<=.001:contacts.append((c.pos.copy(),frame[0]*(1 if c.geom2==gid else -1),frame[1],frame[2]))
                else:self_pen=max(self_pen,-float(c.dist))
            fc,margin,rank=closure(contacts,obj[:3],meta['radius'])
            # Also check the exact contact state reported by the training model.
            traincontacts=[]
            for k,found in enumerate(z['contact_found'][i]):
                if found:
                    n=z['contact_normal'][i,k];t=z['contact_tangent'][i,k]
                    traincontacts.append((z['contact_pos'][i,k],n,t,np.cross(n,t)))
            simfc,_,_=closure(traincontacts,obj[:3],meta['radius'])
            rows.append(dict(env=i,native_mesh_closure=fc,training_geometry_closure=simfc,
                             native_mesh_penetration_mm=penetration*1000,native_self_penetration_mm=self_pen*1000,
                             closure_margin=margin,wrench_rank=rank,contact_count=len(contacts),
                             held_in_training_geometry=bool(z['hold_pass'][i]) if 'hold_pass' in z else None,
                             candidate=bool(fc and penetration<.003 and self_pen<.003)))
        accepted=[r['env'] for r in rows if r['candidate']]
        summary[label]=dict(evaluated=len(rows),native_mesh_closure=sum(r['native_mesh_closure'] for r in rows),
                            training_geometry_closure=sum(r['training_geometry_closure'] for r in rows),
                            native_penetration_under_3mm=sum(r['native_mesh_penetration_mm']<3 for r in rows),
                            native_candidate_count=len(accepted),rows=rows)
        if accepted:
            saved[label+'_qpos']=z['target_qpos'][accepted][:,ids]
            saved[label+'_base_pose']=z['target_base_pose'][accepted]
            saved[label+'_object_pose_world']=z['object_pose_world'][accepted]
            if 'control_target_qpos' in z:saved[label+'_control_target_qpos']=z['control_target_qpos'][accepted][:,ids]
            if 'actuator_force' in z:saved[label+'_actuator_force']=z['actuator_force'][accepted][:,ids]
    if saved:np.savez_compressed(a.experiment/'mesh-closure-candidates.npz',**saved,joint_names=np.array(native_names))
    summary['note']='Finite-cone mu=.8 LP uses normalized positive weights and rank=6. Native candidates additionally require object/self mesh penetration <3mm. This is not an unlimited-load or real-world guarantee.'
    (a.experiment/'independent-closure-check.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='rows'} if isinstance(v,dict) else v for k,v in summary.items()},indent=2))


if __name__=='__main__':main()
