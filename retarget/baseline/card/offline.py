"""Whole-segment CARD anchors, collision field and validated centered filtering."""
import copy
import numpy as np
import torch
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from retarget.baseline.card.geometry import PseudoObject
from retarget.baseline.card.math import project,log_so3,exp_so3,continuous_euler_xyz


def prepare_clip(solver,frames,contact=True):
    """Two-pass noncausal preprocessing. Never called by the live worker."""
    revisions={(f.get('skeleton_runtime_revision'),f.get('pose_calibration_version')) for f in frames}
    if len(revisions)>1:raise ValueError('IK／MANO 姿态标定在片段中改变，请按标定版本分段重算')
    samples=[];by_stamp={};maximum=0.
    for frame in frames:
        source,weights,anchors=solver.extractor.extract(frame,contact)
        samples.append((frame,source,anchors));by_stamp[float(frame['timestamp_ms'])]=anchors
        maximum=max(maximum,max([a['force'] for a in anchors],default=0.))
    for region in range(16):
        runs=[];run=[];last=None
        for i,(frame,source,anchors) in enumerate(samples):
            a=next((a for a in anchors if a['region']==region),None);stamp=float(frame['timestamp_ms'])
            if a is None or (last is not None and stamp-last>250):
                if run:runs.append(run)
                run=[]
            if a is not None:run.append((i,a))
            last=stamp
        if run:runs.append(run)
        for run in runs:
            uv=np.array([a['uv_m'] for _,a in run])
            for offset,(i,a) in enumerate(run):
                average=uv[max(0,offset-5):offset+6].mean(0)
                frame,source,anchors=samples[i];new=solver.extractor.relocate(a,average,source[0],source[1])
                anchors[anchors.index(a)]=new
    if solver.external_sdf is None:
        field=PseudoObject(solver.device);field.mode='whole_clip'
        for frame,source,anchors in samples:
            paired=solver.update_self_contacts(frame,anchors)
            field.add([a for a in anchors if a['region'] not in paired],source[1],float(frame['timestamp_ms']))
        field.frozen=True;solver.sdf=field
    solver.offline_anchors=by_stamp;solver.offline_force_max=maximum
    return dict(anchor_filter='centered_11_frame_chart_mean_within_contact_episodes',object_mode=solver.sdf.metadata()['mode'],force_max=maximum)


def filter_clip(solver,frames,results):
    """Centered SG(15,3), then motion projection + hard-collision verification."""
    output=copy.deepcopy(results);runs=[];run=[];last=None
    for i,(frame,result) in enumerate(zip(frames,results)):
        stamp=float(frame['timestamp_ms'])
        if not result['solution_feasible'] or (last is not None and stamp-last>250):
            if run:runs.append(run)
            run=[]
        if result['solution_feasible']:run.append(i)
        last=stamp
    if run:runs.append(run)
    solver.previous_contacts={}
    for run in runs:
        solver.previous_contacts={}
        raw=np.array([results[i]['raw_qpos'] for i in run]);translations=raw[:,:3];joints=raw[:,6:]
        matrices=Rotation.from_euler('XYZ',raw[:,3:6]).as_matrix()
        # Continuous local increments, avoiding the pi jump of global axis-angle.
        chart=np.zeros((len(run),3))
        for k in range(1,len(run)):chart[k]=chart[k-1]+Rotation.from_matrix(matrices[k-1].T@matrices[k]).as_rotvec()
        if len(run)>=15:
            target_t=savgol_filter(translations,15,3,axis=0,mode='interp');target_j=savgol_filter(joints,15,3,axis=0,mode='interp');filtered=savgol_filter(chart,15,3,axis=0,mode='interp')
            target_r=np.empty_like(matrices);target_r[0]=matrices[0]@Rotation.from_rotvec(filtered[0]-chart[0]).as_matrix()
            for k in range(1,len(run)):target_r[k]=target_r[k-1]@Rotation.from_rotvec(filtered[k]-filtered[k-1]).as_matrix()
        else:target_t=translations;target_j=joints;target_r=matrices
        previous=None;previous_euler=raw[0,3:6]
        for k,i in enumerate(run):
            solver.update_object_transform(frames[i])
            source,_,anchors=solver.extractor.extract(frames[i]);anchors=solver.offline_anchors[float(frames[i]['timestamp_ms'])]
            solver.update_self_contacts(frames[i],anchors)
            c=solver.context(source[0],source[1],anchors);c['previous']=previous
            accepted=None;reason='filtered' if len(run)>=15 else 'short_segment_raw'
            for attempt in range(2):
                t=solver.tensor(target_t[k] if attempt==0 else translations[k]);r=solver.tensor(target_r[k] if attempt==0 else matrices[k]);j=solver.tensor(target_j[k] if attempt==0 else joints[k])
                t,r,j=project(t,r,j,previous,solver.adapter.low[6:],solver.adapter.high[6:],solver.settings)
                with torch.no_grad():cost,_=solver.evaluate(t[None],r[None],j[None],c,True)
                if torch.isfinite(cost[0]):accepted=(t,r,j);reason=reason if attempt==0 else 'raw_fallback_after_projection';break
            result=output[i];result['postprocessing']='sg15_order3_validated';result['filter_status']=reason if accepted else 'infeasible_after_filter_and_raw_projection'
            if accepted is None:
                result.update(status='infeasible',ready=False,solution_feasible=False,contact_ready=False,contact_mask=[None]*460,filtered_qpos=None)
                continue
            t,r,j=accepted;previous=accepted;angles=continuous_euler_xyz(r.cpu().numpy(),previous_euler);previous_euler=angles
            q=np.r_[t.cpu().numpy(),angles,j.cpu().numpy()];p,rr=solver.frames(t,r,j);p=p.cpu().numpy()[solver.surface.body_ids];rr=rr.cpu().numpy()[solver.surface.body_ids]
            taxels=p[solver.owners]+np.einsum('nij,nj->ni',rr[solver.owners],solver.local);normals=np.einsum('nij,nj->ni',rr[solver.owners],solver.normal)
            result.update(filtered_qpos=q.tolist(),qpos=q.tolist(),joint_positions_rad=q[6:].tolist(),wrist_position_m=t.cpu().tolist(),wrist_rotation=r.cpu().tolist(),link_positions_m=p.tolist(),link_rotations=rr.tolist(),taxel_positions_m=taxels.tolist(),taxel_normals=normals.tolist())
            # Selected anchors and diagnostics must refer to the filtered pose too.
            with torch.no_grad():_,detail=solver.evaluate(t[None],r[None],j[None],c,True)
            solver.previous_contacts={region:local[0].clone() for region,local in detail['local_points'].items()}
            result['raw_quality']=result['quality'];result['quality']=copy.deepcopy(result['quality'])
            result['quality']['cost_terms']={key:float(v[0]) for key,v in detail['groups'].items()};result['quality']['cost_subterms']={key:float(v[0]) for key,v in detail['parts'].items()}
            errors=[];angles=[]
            for anchor in result['contact_anchors']:
                region=anchor['region'];point=detail['points'][region][0].cpu().numpy();normal=detail['normals'][region][0].cpu().numpy()
                error=float(np.linalg.norm(point-anchor['position_m'])*1000);angle=float(np.degrees(np.arccos(np.clip(normal@np.array(anchor['direction']),-1,1))))
                anchor.update(robot_position_m=point.tolist(),robot_normal=normal.tolist(),error_mm=error,normal_error_deg=angle);errors.append(error);angles.append(angle)
                material=detail['material_points'].get(region)
                anchor.update(robot_face_id=int(material[0][0]) if material is not None else None,
                              robot_barycentric=material[1][0].cpu().tolist() if material is not None else None,
                              force_support=solver.force_report(detail,region))
            result['quality'].update(contact_rmse_mm=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,contact_max_error_mm=max(errors) if errors else None,patch_center_rmse_mm=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,contact_normal_mean_deg=float(np.mean(angles)) if angles else None)
            result['collision'].update(palm_hard_rejected=bool(detail['hard_palm'][0]),object_hard_rejected=bool(detail['hard_object'][0]),min_palm_gap_mm=float(detail['min_palm_gap_mm'][0]),max_penetration_mm=float(detail['max_penetration_mm'][0]),
                self_mesh=solver.mesh_collision.report(detail['mesh_depth'][0],detail['mesh_valid'][0]),
                mesh_hard_rejected=False,mesh_diagnostic_exceeded=bool(detail['hard_mesh'][0]),object_max_penetration_mm=float(detail['object_max_penetration_mm'][0]))
            result['self_contact_pairs']=solver.contact_report(detail)
            result['contact_constraints_satisfied']=all(p['satisfied'] for p in result['self_contact_pairs']) if result['self_contact_pairs'] else None
    return output
