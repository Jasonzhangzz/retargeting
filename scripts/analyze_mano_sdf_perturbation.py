#!/usr/bin/env python3
"""Evaluate executed lift/actions; optional diagnostics inspect contact layouts."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from retarget.mano_noise import nearest_sdf,task_lp
from retarget.contact_exploration import ObjectMeshSurface
from retarget.contact_solvers import contact_basis


def local_force_relaxation(surface,points,active,force,radius=.005):
    """Outer friction cones for every available patch normal, ignoring torque.

    Splitting each contact among ANY of its face normals is also allowed.
    Infeasibility of this relaxation certifies failure of even force balance
    in this local patch; feasibility is only an inconclusive upper bound.
    """
    columns=[];owners=[];count=0
    for point in points[active]:
        patch=surface.patch(point,radius)
        if patch is None:continue
        normals=np.unique(patch.normals,axis=0)
        basis=contact_basis(normals);n,t1,t2=basis[...,0],basis[...,1],basis[...,2]
        # Circumscribed square contains the entire unit-friction circular cone.
        primitive=np.stack((n+t1+t2,n+t1-t2,n-t1+t2,n-t1-t2),axis=1).reshape(-1,3)
        columns.append(primitive);owners.extend([count]*len(primitive));count+=1
    if not count:return dict(status='infeasible',primitive_count=0)
    G=np.concatenate(columns).T;owners=np.asarray(owners)
    cap=csr_matrix((np.ones(len(owners)),(owners,np.arange(len(owners)))),shape=(count,len(owners)))
    result=linprog(np.zeros(len(owners)),A_eq=G,b_eq=force,A_ub=cap,b_ub=np.full(count,10.),bounds=(0,None),method='highs')
    return dict(status='infeasible' if result.status==2 else 'possible' if result.success else 'unknown',
                primitive_count=len(owners))


def local_wrench_relaxation(surface,points,active,force,torque,center,radius=.005):
    """Outer bound: circumscribed force cones at every corner of a 5 mm box.

    The box contains the spherical search region, and contact force may split
    across box corners and every patch face. Infeasibility is conclusive for
    this contact model; feasibility is only an inconclusive relaxation.
    """
    columns=[];owners=[];count=0
    offsets=np.array([[x,y,z] for x in (-radius,radius) for y in (-radius,radius) for z in (-radius,radius)])
    for point in points[active]:
        patch=surface.patch(point,radius)
        if patch is None:continue
        basis=contact_basis(np.unique(patch.normals,axis=0));n,t1,t2=basis[...,0],basis[...,1],basis[...,2]
        primitive=np.stack((n+t1+t2,n+t1-t2,n-t1+t2,n-t1-t2),axis=1).reshape(-1,3)
        directions=np.broadcast_to(primitive,(8,*primitive.shape))
        moments=np.cross((point+offsets-center)[:,None],directions)
        wrenches=np.concatenate((directions,moments),axis=-1).reshape(-1,6)
        columns.append(wrenches);owners.extend([count]*len(wrenches));count+=1
    if not count:return dict(status='infeasible',primitive_count=0)
    G=np.concatenate(columns).T;owners=np.asarray(owners)
    cap=csr_matrix((np.ones(len(owners)),(owners,np.arange(len(owners)))),shape=(count,len(owners)))
    result=linprog(np.zeros(len(owners)),A_eq=G,b_eq=np.r_[force,torque],A_ub=cap,b_ub=np.full(count,10.),bounds=(0,None),method='highs')
    return dict(status='infeasible' if result.status==2 else 'possible' if result.success else 'unknown',primitive_count=len(owners))


def operation_metrics(surface,archive):
    """Measure actual executed lift, independently of virtual contact layouts."""
    from scipy.spatial.transform import Rotation
    q=archive['qpos_traj'];reference=archive['object_reference_qpos']
    vertices=np.unique(surface.triangles.reshape(-1,3),axis=0)
    def clearance(poses):
        rotation=Rotation.from_quat(poses[:,3:],scalar_first=True).as_matrix()
        return np.min(np.einsum('vi,fji->fvj',vertices,rotation)[:,:,2],axis=1)+poses[:,2]
    actual_clearance=clearance(q[:,26:33]);desired_clearance=clearance(reference)
    # Errors/targets are post-step; qpos_traj also includes the initial state.
    moving=np.arange(len(archive['drift']))<237
    required_airborne=(desired_clearance[1:]>.005)&moving
    actual_airborne=(actual_clearance[1:]>.005)&moving
    endpoint=min(237,len(q)-1)
    return dict(peak_object_lift_mm=float(1000*(np.max(q[:endpoint+1,28])-q[0,28])),
        motion_endpoint_lift_mm=float(1000*(q[endpoint,28]-q[0,28])),
        motion_endpoint_clearance_mm=float(1000*actual_clearance[endpoint]),
        minimum_hold_clearance_mm=float(1000*np.min(actual_clearance[endpoint:])),
        final_object_clearance_mm=float(1000*actual_clearance[-1]),
        reference_endpoint_lift_mm=float(1000*(reference[endpoint,2]-reference[0,2])),
        actual_airborne_fraction_when_reference_airborne=float(np.mean(actual_clearance[1:][required_airborne]>.005)),
        lift_delay_to_5mm_ms=float(10*(np.flatnonzero(actual_airborne)[0]-np.flatnonzero(required_airborne)[0]))
            if actual_airborne.any() else None,
        airborne_reference_steps=int(np.count_nonzero(required_airborne)),
        minimum_clearance_when_reference_airborne_mm=float(1000*np.min(actual_clearance[1:][required_airborne])))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=ROOT/'reports/mano_sdf_perturbation')
    parser.add_argument('--local-only',action='store_true')
    parser.add_argument('--output-layouts',action='store_true')
    args=parser.parse_args();r=args.report
    manifest=json.loads((r/'manifest.json').read_text());reference=np.load(r/'reference_task.npz')
    import mujoco
    model=mujoco.MjModel.from_xml_path(str(ROOT/'data/example/sequence/scene/wuji_right/sugar_box_mesh.xml'))
    surface=ObjectMeshSurface.from_model(model,model.body('obj').id);center=model.body_ipos[model.body('obj').id]
    local_path=r/'local_bounds.json';local=json.loads(local_path.read_text()) if local_path.exists() else {}
    for condition in manifest['conditions'] if (args.local_only or args.output_layouts) else []:
        name=condition['name']
        if name in local:continue
        g=np.load(Path(condition['input'])/'contactaware/contact_guidance.npz')
        frames=[int(f) for f,feasible in condition['task_feasible_airborne_frames'].items() if not feasible]
        rows={str(f):local_force_relaxation(surface,g['contact_pos_obj'][f],g['contact_mask'][f]>0,reference['forces'][f]) for f in frames}
        local[name]=dict(input_infeasible_frames=len(frames),
            certified_no_local_force_balance=sum(row['status']=='infeasible' for row in rows.values()),frames=rows)
        local_path.write_text(json.dumps(local,indent=2)+'\n')
        print('bound',name,local[name]['input_infeasible_frames'],local[name]['certified_no_local_force_balance'],flush=True)
    if args.local_only:return
    result_path=r/'results.json'
    if not result_path.exists():return
    results=json.loads(result_path.read_text());conditions={c['name']:c for c in manifest['conditions']}
    for key,row in results.items():
        run=r/'runs/comfree'/key
        with np.load(run/'rollout.npz') as archive:row.update(operation_metrics(surface,archive))
        row['lift_and_motion_complete']=bool(row['tracking_pass'] and row['motion_endpoint_lift_mm']>=50.)
        row['operation_pass']=bool(row['lift_and_motion_complete']
            and row['actual_airborne_fraction_when_reference_airborne']>=.95)
        # Correct experiment labels also for runs completed before annotation
        # was added; fixed controls preserve the common physics and ADMM terms.
        audit_path=run/'audit.json';audit=json.loads(audit_path.read_text())
        audit.update(experiment_contact_method=row['method'],
            template_query_targets_optimized=row['method']!='fixed',
            contact_update_disabled=row['method']=='fixed',
            fixed_control_keeps_physical_wrench_and_consensus=True)
        audit_path.write_text(json.dumps(audit,indent=2)+'\n')
    # The matrix process appends results while it runs. Put enriched metrics in
    # a separate file so analysis never races with its incremental checkpoint.
    (r/'operation_results.json').write_text(json.dumps(results,indent=2)+'\n')
    if args.output_layouts:
        output_path=r/'output_task_feasibility.json';output=json.loads(output_path.read_text()) if output_path.exists() else {}
        for key,result in results.items():
            condition=conditions[result['condition']]
            if key in output or condition['task_feasible_airborne_fraction']==1.:continue
            run=r/'runs/comfree'/key;a=np.load(run/'rollout.npz');snap=np.load(run/'contacts.npz');rows=[]
            for window in snap['window_indices']:
                stage=int(window);frame=int(np.rint(a['target_frames'][stage*3]))
                if not reference['airborne'][frame]:continue
                points=snap['c'][stage,0];active=snap['mask'][stage,0]>0
                _,normal,_=nearest_sdf(surface,points)
                okay=task_lp(points,normal,active,reference['forces'][frame],reference['torques'][frame],center)
                rows.append(dict(window=stage,frame=frame,task_feasible=okay,
                    input_feasible=condition['task_feasible_airborne_frames'][str(frame)]))
            output[key]=dict(optimized_task_feasible_fraction=float(np.mean([row['task_feasible'] for row in rows])) if rows else None,
                repaired_infeasible_windows=sum(not row['input_feasible'] and row['task_feasible'] for row in rows),
                input_infeasible_windows=sum(not row['input_feasible'] for row in rows),windows=rows)
            output_path.write_text(json.dumps(output,indent=2)+'\n');print('output',key,output[key]['optimized_task_feasible_fraction'],flush=True)
    fields=['condition','method','lift_and_motion_complete','operation_pass','tracking_pass','moving_position_rmse_mm','moving_rotation_rmse_deg',
            'trajectory_endpoint_position_error_mm','trajectory_endpoint_rotation_error_deg',
            'motion_endpoint_lift_mm','actual_airborne_fraction_when_reference_airborne','lift_delay_to_5mm_ms',
            'hold_position_rmse_mm','hold_rotation_rmse_deg','max_contact_shift_mm']
    with (r/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:row[k] for k in fields} for row in results.values())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=list(conditions);methods=['fixed','sampled-lm','qp'];fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True,layout='constrained')
    x=np.arange(len(labels));width=.25
    for i,method in enumerate(methods):
        for ax,field in zip(axes,['moving_position_rmse_mm','moving_rotation_rmse_deg']):
            values=[results.get(name+'__'+method,{}).get(field,np.nan) for name in labels]
            ax.bar(x+(i-1)*width,values,width,label=method)
    axes[0].set_ylabel('Moving position RMSE (mm)');axes[1].set_ylabel('Moving orientation RMSE (deg)')
    axes[1].set_xticks(x,labels,rotation=35,ha='right');axes[0].legend()
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.savefig(r/'tracking_matrix.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
