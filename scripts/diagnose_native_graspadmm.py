#!/usr/bin/env python3
"""Read-only native-state diagnostics; never used by upstream optimization."""
from pathlib import Path
import argparse,importlib,json,sys,time
import numpy as np
import native_graspadmm_retest as runner


def _geometry_job(job):
    import trimesh
    source=runner.SOURCE
    record,case_dir=job
    case_dir=Path(case_dir)
    result=dict(input=record['input'],valid_final_state=False,projected_targets=False,projected_hand=False,
                slot_distance_pass=False,slot_normal_pass=False,slot_joint_pass=False,required_groups_pass=False,
                native_operation_output=record['native_operation_output'])
    if not record['captured_final_state']:
        result['failure']='missing_final_state';return result,None
    data=np.load(record['input'],allow_pickle=True).item()
    arrays=np.load(case_dir/'final_state.npz')
    q=arrays['final_grasp_qpos'].reshape(-1)
    if not np.isfinite(q).all():result['failure']='nonfinite_final_state';return result,None
    cfg=runner.BASE
    env=source.sim.MuJoCo_OptEnv(hand_cfg=source.sim.HandCfg(cfg.hand.xml_path,freejoint=True),
        scene_cfg=source.files.load_scene_cfg(data['scene_path']),sim_cfg=source.sim.MuJoCo_OptCfg(obj_margin=.001))
    env.reset_qpos(q)
    # This is a separate reconstruction, after the original operation returned.
    ho,hh=env.get_contacts(.002)
    bodies=record['hand_cbody'];required=data['required_cbody']
    native_bodies=set(ho['bn1'])
    group_hits=[bool(set(group)&native_bodies) for group in required]
    result.update(valid_final_state=True,required_group_count=len(required),required_groups_met=sum(group_hits),
        required_groups_pass=bool(required) and all(group_hits),native_contact_links=len(set(ho['bn1'])),
        native_contact_count=len(ho['dist']),hand_object_depth_m=max(0.,-float(np.min(ho['dist']))) if len(ho['dist']) else 0.,
        self_depth_m=max(0.,-float(np.min(hh['dist']))) if len(hh['dist']) else 0.,
        joint_limits_pass=bool(env.check_qpos_limit(0.)),fallback_used=record['fallback_used'])
    stats=importlib.import_module('dexonomy.op.compute_stats')
    groups=source.files.load_yaml(cfg.hand_bgroup_path)['body_group']
    if len(ho['dist']):
        cln,cdc,pd=stats.compute_cln_and_cdc_pd(ho,groups)
        result.update(cln=cln,cdc_mm=cdc,native_penetration_m=pd)
    else:result.update(cln=0.,cdc_mm=None,native_penetration_m=0.,failure='no_native_contacts')
    if 'hand_cpn_b' not in arrays or not len(arrays['hand_cpn_b']):
        result['failure']='missing_updated_slots';return result,None
    slots=env.transform_cpn_b2w(bodies,arrays['hand_cpn_b'])
    mesh,_,_,_=source.core.load_obj_trimesh(data)
    def project(points):
        closest,dist,ids=trimesh.proximity.closest_point(mesh,points)
        bary=trimesh.triangles.points_to_barycentric(mesh.triangles[ids],closest)
        normal=(mesh.vertex_normals[mesh.faces[ids]]*bary[...,None]).sum(1)
        normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-20)
        return closest,dist,normal
    points,dist,normals=project(slots[:,:3])
    angles=np.rad2deg(np.arccos(np.clip(-(slots[:,3:]*normals).sum(-1),-1,1)))
    distance_ok=dist<=.002;normal_ok=angles<=30.;joint_ok=distance_ok&normal_ok
    result.update(slot_count=len(slots),slot_distance_coverage=float(distance_ok.mean()),
        slot_normal_coverage=float(normal_ok.mean()),slot_joint_coverage=float(joint_ok.mean()),
        slot_distance_count=int(distance_ok.sum()),slot_normal_count=int(normal_ok.sum()),slot_joint_count=int(joint_ok.sum()),
        slot_distance_pass=bool(distance_ok.mean()>=.8),slot_normal_pass=bool(normal_ok.mean()>=.8),
        slot_joint_pass=bool(joint_ok.mean()>=.8),slot_distances_m=dist,slot_angles_deg=angles,
        native_collision_pass=result['hand_object_depth_m']<=0. and result['self_depth_m']<=0.)
    target=arrays['cp_o'] if 'cp_o' in arrays else None
    target_points,target_normals=None,None
    if target is not None and len(target) and np.isfinite(target).all():
        target_points,target_dist,target_normals=project(target)
        gap=np.linalg.norm(slots[:,:3]-target,axis=-1)
        result.update(target_hand_gap_max_m=float(gap.max()),target_hand_gap_mean_m=float(gap.mean()),
            target_on_surface_error_max_m=float(target_dist.max()),targets_from_fallback=record['fallback_used'])
    center=mesh.center_mass;radius=max(float(np.linalg.norm(mesh.vertices-center,axis=-1).max()),.005)
    contact_arrays=dict(hand_slots=slots,hand_projections=points,hand_projected_normals=normals,
        object_center=center,object_radius=np.array(radius))
    if target_points is not None:contact_arrays.update(target_points=target_points,target_normals=target_normals)
    np.savez_compressed(case_dir/'diagnostic_contacts.npz',**contact_arrays)
    packet=dict(points=points,normals=normals,center=center,radius=radius,target=target_points,target_normals=target_normals)
    return result,packet


def geometry_job(job):
    try:return _geometry_job(job)
    except Exception:
        import traceback
        record,_=job
        return dict(input=record['input'],valid_final_state=False,projected_targets=False,projected_hand=False,
            slot_distance_pass=False,slot_normal_pass=False,slot_joint_pass=False,required_groups_pass=False,
            failure='diagnostic_exception',exception=traceback.format_exc(),native_operation_output=record['native_operation_output']),None


def closure_batches(rows,packets,production):
    # Import the identical production diagnostic, with no changes to its formulas.
    import torch
    sys.path.insert(0,str(production))
    from retarget.env.grasp.projected_qp import ProjectedQPClosure,ClosureSettings
    groups={}
    for i,p in enumerate(packets):
        if p is not None:groups.setdefault(len(p['points']),[]).append(i)
    for count,indices in groups.items():
        if not count:continue
        for start in range(0,len(indices),64):
            selected=indices[start:start+64]
            for name,pkey,nkey in [('hand','points','normals'),('targets','target','target_normals')]:
                ids=[i for i in selected if packets[i][pkey] is not None]
                if not ids:continue
                stack=lambda key:torch.tensor(np.stack([packets[i][key] for i in ids]),dtype=torch.float32)
                solver=ProjectedQPClosure(ClosureSettings(friction=.6,cuda_graph=False),center=stack('center')[0],
                    radius=packets[ids[0]]['radius'],object_mass=.1)
                solver.set_object_batch(stack('center'),stack('radius'))
                output=solver.solve(stack(pkey),stack(nkey))
                for k,i in enumerate(ids):
                    rows[i]['projected_'+name]=bool(output.certified[k])
                    rows[i]['projected_'+name+'_valid']=bool(output.valid[k])
                    rows[i]['projected_'+name+'_max_error']=float(output.scenario_rms[k,1:].max())
                    rows[i]['projected_'+name+'_margin']=float(output.certified_margin[k])
                    rows[i]['projected_'+name+'_cost']=float(output.cost[k])
    for row in rows:
        row['diagnostic_joint_pass']=row['projected_hand'] and row['slot_joint_pass'] and row.get('native_collision_pass',False) and row.get('native_contact_count',0)>0


def run(root,upstream,production):
    import multiprocessing as mp
    source=runner.load_upstream(upstream);runner.verify_assets(root,source)
    cfg=runner.configuration(source,root,'grasp_admm');cfg_dict=source.OmegaConf.to_container(cfg)
    quality=json.loads((root/'quality_operation.json').read_text())
    jobs=[(r,root/'quality/cases'/f'{i:05d}') for i,r in enumerate(quality['rows'])]
    start=time.perf_counter()
    with mp.Pool(32,initializer=runner.worker_init,initargs=(str(upstream),cfg_dict)) as pool:
        output=list(pool.imap(geometry_job,jobs,chunksize=1))
    rows=[x[0] for x in output];packets=[x[1] for x in output]
    geometry_seconds=time.perf_counter()-start;start=time.perf_counter()
    closure_batches(rows,packets,production)
    result=dict(denominator=len(rows),valid_final_states=sum(r['valid_final_state'] for r in rows),
        projected_hand_passed=sum(r['projected_hand'] for r in rows),projected_targets_passed=sum(r['projected_targets'] for r in rows),
        distance_passed=sum(r['slot_distance_pass'] for r in rows),normal_passed=sum(r['slot_normal_pass'] for r in rows),
        slot_joint_passed=sum(r['slot_joint_pass'] for r in rows),required_groups_passed=sum(r['required_groups_pass'] for r in rows),
        diagnostic_joint_passed=sum(r['diagnostic_joint_pass'] for r in rows),geometry_seconds=geometry_seconds,
        pooled_slots=sum(r.get('slot_count',0) for r in rows),pooled_distance=sum(r.get('slot_distance_count',0) for r in rows),
        pooled_normals=sum(r.get('slot_normal_count',0) for r in rows),pooled_joint=sum(r.get('slot_joint_count',0) for r in rows),
        closure_seconds=time.perf_counter()-start,rows=rows,
        thresholds=dict(distance_m=.002,angle_deg=30.,coverage=.8,friction=.6,qp_iterations=96,load_cases=13,certificate_tolerance=.05),
        caveat='Posthoc diagnostics; projected target closure is not actual contact or native six-direction GSR')
    errors=[r for r in rows if r.get('failure')=='diagnostic_exception']
    result['complete']=not errors
    runner.write(root/'diagnostics.json',result);runner.verify_assets(root,source)
    if errors:raise RuntimeError('Posthoc diagnostics failed, not a zero quality score: '+errors[0]['exception'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--production',type=Path,required=True);a=p.parse_args();run(a.root.resolve(),a.upstream.resolve(),a.production.resolve())
