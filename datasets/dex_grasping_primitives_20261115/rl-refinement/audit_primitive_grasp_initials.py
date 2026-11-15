#!/usr/bin/env python3
"""Check frozen Robokit poses against native mesh contacts, without optimization."""
import argparse,csv,json,sys,re,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from scipy.optimize import linprog

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def closure(contacts,center,scale,mu=.8):
    if len(contacts)<3:return False,0.,0
    w=[]
    for p,n,t1,t2 in contacts:
        for theta in np.linspace(0,2*np.pi,8,endpoint=False):
            force=n+mu*(np.cos(theta)*t1+np.sin(theta)*t2)
            w.append(np.r_[force,np.cross((p-center)/max(scale,.01),force)])
    G=np.array(w).T;n=G.shape[1];rank=int(np.linalg.matrix_rank(G,tol=1e-6))
    if rank<6:return False,0.,rank
    # Positive coefficients balancing all six wrench directions, not G*0=0.
    c=np.r_[np.zeros(n),-1.]
    eq=np.c_[np.vstack([G,np.ones(n)]),np.zeros(7)]
    sol=linprog(c,A_ub=np.c_[-np.eye(n),np.ones(n)],b_ub=np.zeros(n),
                A_eq=eq,b_eq=np.r_[np.zeros(6),1.],bounds=[(0,None)]*(n+1),method='highs')
    margin=float(sol.x[-1]) if sol.success else 0.
    return margin>1e-7,margin,rank


def main():
    import mujoco
    from retarget.env.grasp.hand import hand_tree
    ap=argparse.ArgumentParser();ap.add_argument('--poses',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    manifest=json.load(open(a.poses/'source-manifest.json'));models={};rows=[]
    for e in manifest['entries']:
        z=np.load(a.poses/'wuji2'/e['file']);kind=e['kind']
        if kind=='plate':continue # Frozen release does not state the box footprint.
        gt='sphere' if kind=='sphere' else 'cylinder'
        if gt not in models:
            tree=hand_tree('wuji2');b=ET.SubElement(tree.find('worldbody'),'body',name='audit_object')
            ET.SubElement(b,'freejoint',name='audit_object_joint')
            ET.SubElement(b,'geom',name='audit_object_geom',type=gt,size='.01' if gt=='sphere' else '.01 .02',
                          mass='.03',friction='.8 .005 .0001',condim='4')
            model=mujoco.MjModel.from_xml_string(ET.tostring(tree,encoding='unicode'));data=mujoco.MjData(model)
            models[gt]=(model,data,model.geom('audit_object_geom').id)
        model,data,gid=models[gt]
        for i in range(len(z['target_qpos'])):
            rad=float(z['radius_m'][i]);height=float(z['height_m'][i]);obj=z['object_pose_world'][i]
            model.geom_size[gid,:]=[rad,0,0] if gt=='sphere' else [rad,height/2,0]
            model.geom_rbound[gid]=rad if gt=='sphere' else np.hypot(rad,height/2)
            model.geom_aabb[gid,:3]=0
            model.geom_aabb[gid,3:]=[rad,rad,rad if gt=='sphere' else height/2]
            base=z['target_base_pose'][i];q=z['target_qpos'][i]
            data.qpos[:6]=np.r_[base[:3],Rotation.from_quat(base[[4,5,6,3]]).as_euler('XYZ')]
            data.qpos[6:26]=q;data.qpos[26:33]=obj;data.qvel[:]=0;mujoco.mj_forward(model,data)
            contacts=[];fingers=set();pen=0.;self_pen=0.;count=0
            for c in data.contact:
                if gid in (c.geom1,c.geom2):
                    count+=1;pen=max(pen,-float(c.dist))
                    hand_geom=c.geom2 if c.geom1==gid else c.geom1
                    body=model.body(model.geom_bodyid[hand_geom]).name
                    match=re.search(r'right_finger([1-5])_',body)
                    if match:fingers.add(match.group(1))
                    if c.dist<=.001:
                        frame=c.frame.reshape(3,3);sign=1 if c.geom2==gid else -1
                        contacts.append((c.pos.copy(),frame[0]*sign,frame[1],frame[2]))
                else:self_pen=max(self_pen,-float(c.dist))
            p=z['target_keypoints_world'][i]
            try:
                hull=ConvexHull(p);inside=bool(np.max(hull.equations[:,:3]@obj[:3]+hull.equations[:,3])<=.005)
            except Exception:inside=False
            fc,margin,rank=closure(contacts,obj[:3],rad)
            rows.append(dict(type=e['type'],row=i,kind=kind,size_scale=float(z['size_scale'][i]),radius_m=rad,height_m=height,
                             object_contact_count=count,contact_finger_count=len(fingers),
                             thumb_contact='1' in fingers,contact_fingers=''.join(sorted(fingers)),
                             object_penetration_mm=pen*1000,self_penetration_mm=self_pen*1000,
                             center_in_keypoint_hull_5mm=inside,friction_cone_closure=fc,closure_margin=margin,wrench_rank=rank,
                             source_grasp_passed=bool(z['grasp_passed'][i]),
                             candidate=bool(len(fingers)>=2 and '1' in fingers and inside and pen<.003 and self_pen<.003)))
    with (a.output/'wuji2-initial-audit.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    candidates=sorted([r for r in rows if r['candidate']],key=lambda r:(not r['friction_cone_closure'],-r['contact_finger_count'],r['object_penetration_mm']))
    summary=dict(total_input=1247,audited=len(rows),plate_rows_not_audited=1247-len(rows),
                 center_in_keypoint_hull=sum(r['center_in_keypoint_hull_5mm'] for r in rows),
                 object_contact=sum(r['object_contact_count']>0 for r in rows),
                 multiple_finger_contacts=sum(r['contact_finger_count']>=2 for r in rows),
                 object_penetration_over_3mm=sum(r['object_penetration_mm']>3 for r in rows),
                 friction_cone_closure=sum(r['friction_cone_closure'] for r in rows),
                 candidates=len(candidates),best_candidates=candidates[:30],
                 note='Native MuJoCo convex collision mesh contacts; hull is only an enclosure proxy. Closure uses mu=.8, an 8-edge friction cone and full-rank normalized positive wrench balance; not a finite-actuator stability proof.')
    (a.output/'initial-audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
