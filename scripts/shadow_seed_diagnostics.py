"""Read-only before/after metrics with normals at the actual moved hand slots."""
from pathlib import Path
import json,multiprocessing as mp,time
import numpy as np
from scipy.spatial.transform import Rotation
import native_graspadmm_retest as runner
import diagnose_native_graspadmm as original
try:
    from shadow_mano_seed import RigidHand,MANO_CHAINS
except ImportError:
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from retarget.env.grasp.shadow_mano_seed import RigidHand,MANO_CHAINS

HAND=None


def job(job_data):
    global HAND
    record,folder=job_data;row,packet=original.geometry_job(job_data)
    if packet is None:return row,packet
    if HAND is None:HAND=RigidHand(runner.BASE.hand.xml_path,MANO_CHAINS)
    folder=Path(folder);state=np.load(folder/'final_state.npz');q=state['final_grasp_qpos'].reshape(-1)
    a=np.load(folder/'diagnostic_contacts.npz');arrays={k:a[k] for k in a.files}
    slots=arrays['hand_slots'].copy();HAND.set(q[7:]);root=Rotation.from_quat(q[3:7],scalar_first=True).as_matrix()
    normals=[];surface_gaps=[]
    for name,p in zip(record['hand_cbody'],state['hand_cpn_b'][:,:3]):
        _,gap,normal=HAND.project(name,np.asarray(p)[None]);body=HAND.model.body(name).id
        normals.append(normal[0]@HAND.data.xmat[body].reshape(3,3).T@root.T);surface_gaps.append(float(gap[0]))
    normals=np.array(normals);outward=arrays['hand_projected_normals']
    angle=np.rad2deg(np.arccos(np.clip(-(normals*outward).sum(-1),-1,1)))
    distance=np.array(row['slot_distances_m']);good_d=distance<=.002;good_n=angle<=30;good_both=good_d&good_n
    row['legacy_template_slot_angles_deg']=row['slot_angles_deg'];row['slot_angles_deg']=angle
    row.update(slot_normal_count=int(good_n.sum()),slot_joint_count=int(good_both.sum()),slot_normal_coverage=float(good_n.mean()),
        slot_joint_coverage=float(good_both.mean()),slot_normal_pass=bool(good_n.mean()>=.8),slot_joint_pass=bool(good_both.mean()>=.8),
        slot_to_hand_collision_surface_m=surface_gaps,normal_source='Nearest actual MANO collision triangle, barycentrically interpolated vertex normals')
    arrays['transported_template_hand_normals']=slots[:,3:].copy();slots[:,3:]=normals;arrays['hand_slots']=slots
    np.savez_compressed(folder/'diagnostic_contacts.npz',**arrays)
    return row,packet


def run(root,source,cfg,production,initial=False):
    quality=json.loads((root/'quality_operation.json').read_text()) if not initial else None
    paths=runner.list_inputs(root);records=quality['rows'] if quality else []
    if initial:
        HAND_INIT=RigidHand(cfg.hand.xml_path,MANO_CHAINS)
        for i,path in enumerate(paths):
            data=np.load(path,allow_pickle=True).item();folder=root/'initial_geometry/cases'/f'{i:05d}';folder.mkdir(parents=True,exist_ok=True)
            q=data['grasp_qpos'].reshape(-1);HAND_INIT.set(q[7:]);rot=Rotation.from_quat(q[3:7],scalar_first=True).as_matrix()
            local=[]
            for body,cp in zip(data['hand_cbody'],data['hand_cpn_w']):
                bid=HAND_INIT.model.body(body).id;br=HAND_INIT.data.xmat[bid].reshape(3,3)
                p=((cp[:3]-q[:3])@rot-HAND_INIT.data.xpos[bid])@br
                n=cp[3:]@rot@br;local.append(np.r_[p,n])
            np.savez_compressed(folder/'final_state.npz',final_grasp_qpos=data['grasp_qpos'],hand_cpn_b=local,cp_o=data['obj_cpn_w'][:,:3])
            records.append(dict(input=path,captured_final_state=True,hand_cbody=data['hand_cbody'],fallback_used=False,native_operation_output=False))
    folder_root=root/('initial_geometry' if initial else 'quality')
    jobs=[(r,folder_root/'cases'/f'{i:05d}') for i,r in enumerate(records)];start=time.perf_counter()
    with mp.Pool(32,initializer=runner.worker_init,initargs=(str(source.root),source.OmegaConf.to_container(cfg))) as pool:
        output=list(pool.imap(job,jobs,chunksize=1))
    rows=[r for r,_ in output];packets=[p for _,p in output];geometry_seconds=time.perf_counter()-start;start=time.perf_counter()
    original.closure_batches(rows,packets,production)
    for row in rows:
        row['tolerated_collision_pass']=row.get('valid_final_state',False) and row.get('hand_object_depth_m',np.inf)<=.002 and row.get('self_depth_m',np.inf)<=.002
        row['diagnostic_joint_pass']=row['projected_hand'] and row['slot_joint_pass'] and row['tolerated_collision_pass'] and row.get('native_contact_count',0)>0
    counts=dict(denominator=len(rows),valid_final_states=sum(r['valid_final_state'] for r in rows),
        projected_hand_passed=sum(r['projected_hand'] for r in rows),projected_targets_passed=sum(r['projected_targets'] for r in rows),
        distance_passed=sum(r['slot_distance_pass'] for r in rows),normal_passed=sum(r['slot_normal_pass'] for r in rows),
        slot_joint_passed=sum(r['slot_joint_pass'] for r in rows),tolerated_collision_passed=sum(r['tolerated_collision_pass'] for r in rows),
        diagnostic_joint_passed=sum(r['diagnostic_joint_pass'] for r in rows),required_groups_passed=sum(r['required_groups_pass'] for r in rows),
        pooled_slots=sum(r.get('slot_count',0) for r in rows),pooled_distance=sum(r.get('slot_distance_count',0) for r in rows),
        pooled_normals=sum(r.get('slot_normal_count',0) for r in rows),pooled_joint=sum(r.get('slot_joint_count',0) for r in rows),
        geometry_seconds=geometry_seconds,closure_seconds=time.perf_counter()-start,rows=rows,
        complete=not any(r.get('failure')=='diagnostic_exception' for r in rows),normal_version='actual_hand_surface_v2',
        penetration_tolerances_m=dict(hand_object=.002,self=.002),thresholds=dict(distance_m=.002,angle_deg=30,coverage=.8,friction=.6,load_cases=13,qp_iterations=96,certificate_tolerance=.05),
        optimizer_updates=0,stage='mapped_initial' if initial else 'ADMM_final')
    runner.write(root/('initial_diagnostics.json' if initial else 'diagnostics.json'),counts)
    if not counts['complete']:raise RuntimeError('Geometry diagnostic exception; not a zero quality score')
