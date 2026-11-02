"""Opposed cylinder seeds for the original, fixed palmar-pinch anchors."""
import json,math
from pathlib import Path
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from retarget.common.results import write_json


def opposed_seeds(model,template):
    data=mujoco.MjData(model);data.qpos[:]=np.r_[np.zeros(6),template.finger_qpos]
    mujoco.mj_kinematics(model,data)
    ids=[model.body(n).id for n in template.body_names]
    rotation=data.xmat[ids].reshape(-1,3,3)
    points=data.xpos[ids]+np.einsum('nij,nj->ni',rotation,template.local_points)
    normals=np.einsum('nij,nj->ni',rotation,template.local_normals)
    x=-normals[0]+normals[1];x/=np.linalg.norm(x)
    # Preserve the actual pinch midpoint. Sample cylinder axes in the tangent
    # plane rather than matching two nearly coincident hand anchors to one rim.
    center=points.mean(0)
    long=rotation[0]@model.site('right_index_tip').pos
    z=long-x*(long@x);z/=np.linalg.norm(z);y=np.cross(z,x);y/=np.linalg.norm(y)
    seeds=[]
    for angle in (0.,math.pi/4,math.pi/2,3*math.pi/4):
        axis=z*math.cos(angle)+y*math.sin(angle)
        basis=np.column_stack((x,np.cross(axis,x),axis))
        r=basis.T;t=-r@center
        seeds.append(np.r_[t,Rotation.from_matrix(r).as_euler('XYZ'),template.finger_qpos])
    return seeds,dict(original_anchor_separation_m=float(np.linalg.norm(points[1]-points[0])),
        original_hand_normal_dot=float(normals[0]@normals[1]),midpoint_in_wrist=center.tolist(),
        hand_anchors_changed=False)


def bootstrap_palmar_pinch(path,template,args,device,seed):
    from .template_mppi import TemplatePoseObjective
    from dataclasses import replace
    path=Path(path);record=json.loads(path.read_text());primitive=record['canonical_primitive']
    first=TemplatePoseObjective(path.with_name('object_1x.obj'),args.settings.objective,device=device,object_mass=args.settings.object_mass)
    seeds,metadata=opposed_seeds(first.cpu,template);del first
    results=[]
    for i,initial in enumerate(seeds):
        objective=TemplatePoseObjective(path.with_name('object_1x.obj'),args.settings.objective,device=device,object_mass=args.settings.object_mass)
        objective.bind(template,initial);objective.configure_object_regions('9_Palmar_Pinch',primitive)
        radius=primitive['radius_m']
        reference=dict(record,canonical_grasp_qpos=initial.tolist(),object_points=[[radius,0.,0.],[-radius,0.,0.]],
            object_outward_normals=[[1.,0.,0.],[-1.,0.,0.]])
        reference_path=path.with_name(f'opposed_seed_{i}.json');write_json(reference_path,reference)
        q,ik=objective.initialize_size(reference_path,primitive,initial,replace(args.settings.size_ik,bootstrap_canonical=False),seed+i)
        q,optimization=objective.optimize(q,args.solver,seed+i)
        diagnostics=objective.final_diagnostics(q)
        results.append(dict(seed=i,qpos=q,ik=ik,optimization=optimization,diagnostics=diagnostics))
        del objective
    def rank(r):
        d=r['diagnostics'];return (not(d['valid'] and d['hand_hand_penetration_accepted'] and d['projected_closure']),d['cost'])
    selected=min(range(len(results)),key=lambda i:rank(results[i]))
    write_json(path.with_name('opposed_pinch_bootstrap.json'),dict(policy='fixed_anchor_opposed_cylinder_seed_v1',
        metadata=metadata,selected=selected,candidates=results))
    chosen=results[selected];write_json(path.with_name('canonical_grasp.json'),chosen)
    d=chosen['diagnostics']
    return np.array(chosen['qpos']),np.array(d['projected_points']),np.array(d['object_normals'])
