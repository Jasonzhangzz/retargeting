"""Bounded, held-out MANO calibration. Runs on a private model in a worker thread."""
from __future__ import annotations
import copy,time,hashlib,json
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .mano_runtime import ManoSurface,CollisionSurface,skeleton_targets,fuse_contacts
from .contact import PATCH_IDS,FINGERS
from .hand_model import positive_gain,MANO_FIT_REVISION
from .hand_anatomy import GEOMETRY_REVISION
from .hand_dimensions import DIMENSION_MODE,DIMENSION_COUNT,PENETRATION_TOLERANCE_M
from scripts.calibrate_wuji_mano import TIPS,CHAINS,rotmat
from .proximal_shape import proximal_pair_gaps
from .official_mano import OFFICIAL_MANO_MODE, OFFICIAL_REFERENCE_POLICY, reference_from_frame


FIT_REVISION=MANO_FIT_REVISION
FLAT_STRAIGHT_WEIGHT=300.
FLAT_PLANE_WEIGHT=100.
FLAT_ANGLE_SCALE_RAD=.05
TOGETHER_DIRECTION_WEIGHT=100.


def flat_finger_terms(m, vertices, joints, rotations):
    """Physical bone collinearity and palm-plane alignment, not raw q=0."""
    directions=[]
    for f in range(1,5):
        c=CHAINS[f]
        points=torch.stack((joints[:,c[0]],joints[:,c[1]],joints[:,c[2]],vertices[:,TIPS[f]]),1)
        directions.append(torch.nn.functional.normalize(points[:,1:]-points[:,:-1],dim=-1))
    d=torch.stack(directions,1)
    normal=torch.as_tensor(m.anatomy.palmar_normal,device=d.device,dtype=d.dtype)
    normal=(rotations[:,0]@normal[:,None]).squeeze(-1)
    straight=((d[:,:,1:]-d[:,:,:1])/FLAT_ANGLE_SCALE_RAD).square().mean()
    plane=((d*normal[:,None,None]).sum(-1)/FLAT_ANGLE_SCALE_RAD).square().mean()
    return straight,plane,d,normal


def flat_pose_report(m,rows,poses,roots):
    ids=[i for i,r in enumerate(rows) if r['stage'] in ('open','together','open_repeat','together_repeat')]
    if not ids:return {'observations':0,'valid':False}
    with torch.no_grad():
        v,j,r,_=m.decode(poses[ids],roots[ids]);_,_,d,n=flat_finger_terms(m,v,j,r)
        bends=torch.rad2deg(torch.acos((d[:,:,1:]*d[:,:,:-1]).sum(-1).clamp(-1,1)))
        plane=torch.rad2deg(torch.asin((d*n[:,None,None]).sum(-1).abs().clamp(0,1)))
        together_ids=[k for k,i in enumerate(ids) if rows[i]['stage'] in ('together','together_repeat')]
        proximal=d[together_ids,:,0]
        spread=float(torch.rad2deg(torch.acos((proximal@proximal.transpose(-1,-2)).clamp(-1,1))).max()) if together_ids else None
        thumb_ids=[k for k,i in enumerate(ids) if rows[i].get('thumb_index_side_contact') is True]
        thumb_bend=None
        if thumb_ids:
            c=CHAINS[0];a=torch.nn.functional.normalize(j[thumb_ids,c[2]]-j[thumb_ids,c[1]],dim=-1)
            b=torch.nn.functional.normalize(v[thumb_ids,TIPS[0]]-j[thumb_ids,c[2]],dim=-1)
            thumb_bend=float(torch.rad2deg(torch.acos((a*b).sum(-1).clamp(-1,1))).max())
    maximum=float(bends.max());palmar=float(plane.max())
    return {'observations':len(ids),'mean_bend_deg':float(bends.mean()),'max_bend_deg':maximum,
            'max_plane_angle_deg':palmar,'per_finger_max_bend_deg':dict(zip(FINGERS[1:],bends.amax((0,2)).cpu().tolist())),
            'together_max_spread_deg':spread,
            'together_thumb_max_ip_bend_deg':thumb_bend,
            'valid':maximum<=8. and palmar<=12. and (spread is None or spread<=12.) and (thumb_bend is None or thumb_bend<=8.)}


def fit_flat_pose_calibration(model_path,profile,mapping,rows,cancel):
    """Retired: no partial pose artifact can bypass complete holdout acceptance."""
    if cancel.is_set():raise InterruptedError('MANO optimization cancelled')
    return {'revision':FIT_REVISION,'accepted':False,'calibration_pose_offsets':np.zeros((15,3)).tolist(),
            'reason':'独立平摊零位已停用；姿态、形状和安装参数统一通过八动作及2 mm碰撞验收'}


def sensor_sample_weights(rows):
    from .mano_objective import participation
    return participation(rows,training=True)


def calibration_phases(rows):
    """Capture order is independent of optimization order; fist is last only."""
    base=[r for r in rows if r['stage'] in ('open','together','mcp90','hook')]
    contacts=base+[r for r in rows if r['stage'].startswith('pinch_')]
    return [('flat_and_together',base),('tactile_pinches',contacts),('fist_refinement',contacts+[r for r in rows if r['stage']=='fist'])]


def fit_pose_zeros(m,rows,cancel,steps=60):
    """Compatibility entry point; shared free-axis pose offsets are retired."""
    if cancel.is_set():raise InterruptedError('MANO optimization cancelled')
    m.calibration_pose_offsets=np.zeros((15,3),np.float32)
    return {'parameters':'retired_shared_pose_offsets','applied':False}


def bilateral_contact(row):
    stage=row.get('stage','')
    if not stage.startswith('pinch_'):return False
    p={p['id']:p for p in row.get('contact',{}).get('patches',[])}
    return bool(row.get('contact',{}).get('ready') and p.get('thumb_distal',{}).get('contact') and p.get(stage[6:]+'_distal',{}).get('contact'))


def holdout_rows(rows):
    if rows and all(r.get('capture_protocol') in ('geort_motion_v2','contact_motion_v3') for r in rows):
        return sorted((r for r in rows if r.get('partition')=='validation' and r.get('training_eligible',True)),key=lambda r:r['timestamp_ms'])
    shared=bool(rows and all(r.get('capture_protocol')=='official_eight_v1' for r in rows))
    result=[]
    for stage in dict.fromkeys(r['stage'] for r in rows):
        group=sorted((r for r in rows if r['stage']==stage),key=lambda r:r['timestamp_ms'])
        result.extend(group[len(group)//2:] if shared else group if stage=='open_repeat' else [r for r in group if r.get('cycle',0)>=1] if stage.startswith('pinch_') else group[len(group)//2:])
    return sorted(result,key=lambda r:r['timestamp_ms'])


def choose_rows(rows):
    if rows and rows[0].get('capture_protocol')=='contact_motion_v3':
        return tuple(sorted((copy.deepcopy(r) for r in rows if r.get('partition')==part and r.get('training_eligible',True)),key=lambda r:r['timestamp_ms']) for part in ('train','validation'))
    if rows and all(r.get('capture_protocol') in ('geort_motion_v2','contact_motion_v3') for r in rows):
        groups={}
        for row in rows:
            key=(row['stage'],row.get('cycle',-1),row.get('motion_phase','static'),row['partition'])
            groups.setdefault(key,[]).append(row)
        train=[];valid=[]
        for key,group in groups.items():
            group=sorted(group,key=lambda r:r['timestamp_ms'])
            target=train if key[-1]=='train' else valid
            count=6 if key[0].startswith('range_') or key[0]=='thumb_opposition' else 3
            selected=set(np.linspace(0,len(group)-1,min(count,len(group)),dtype=int).tolist())
            contacts=[i for i,r in enumerate(group) if r.get('contact_confirmed')]
            if contacts:selected.update((contacts[0],contacts[len(contacts)//2],contacts[-1]))
            target.extend(copy.deepcopy(group[i]) for i in sorted(selected))
        return sorted(train,key=lambda r:r['timestamp_ms']),sorted(valid,key=lambda r:r['timestamp_ms'])
    train=[];valid=[]
    for stage in dict.fromkeys(r['stage'] for r in rows):
        group=sorted((r for r in rows if r['stage']==stage),key=lambda r:r['timestamp_ms'])
        if all(r.get('capture_protocol')=='official_eight_v1' for r in group):
            for target,part in ((train,group[:len(group)//2]),(valid,group[len(group)//2:])):
                if len(part)<3:raise ValueError('八动作每组需要独立的训练和留出帧')
                target.extend(copy.deepcopy(part[i]) for i in sorted(set([0,len(part)//2,len(part)-1])))
            continue
        if stage=='open_repeat':
            valid.extend(copy.deepcopy(group[i]) for i in sorted(set([0,len(group)//2,len(group)-1])))
            continue
        if stage.startswith('pinch_'):
            first=[r for r in group if r.get('cycle',0)==0];second=[r for r in group if r.get('cycle',0)>=1]
            if not first or not second:raise ValueError('每组对指需要两次完整接触与释放')
            def strength(row):
                p={p['id']:p for p in row['contact']['patches']}
                return p['thumb_distal']['response']+p[stage[6:]+'_distal']['response']
            for target,part in ((train,first),(valid,second)):
                both=[r for r in part if bilateral_contact(r)];selected={}
                if both:
                    onset=min(both,key=strength);peak=max(both,key=strength)
                    selected[onset['timestamp_ms']]=(onset,'onset')
                    if peak['timestamp_ms']!=onset['timestamp_ms']:selected[peak['timestamp_ms']]=(peak,'peak')
                # The entire unconfirmed approach/release range, including the
                # closest near miss, is part of the fit and held-out evaluation.
                approach=[r for r in part if not bilateral_contact(r)];finger=stage[6:]
                approach.sort(key=lambda r:np.linalg.norm(np.array(r['fingers']['thumb']['position'])-r['fingers'][finger]['position']))
                if approach:
                    for index in sorted(set([0,len(approach)//3,2*len(approach)//3,len(approach)-1])):
                        r=approach[index];selected[r['timestamp_ms']]=(r,'approach')
                for _,(r,role) in sorted(selected.items()):
                    saved=copy.deepcopy(r);saved['load_role']=role;target.append(saved)
        else:
            for target,part in ((train,group[:len(group)//2]),(valid,group[len(group)//2:])):
                target.extend(copy.deepcopy(part[i]) for i in sorted(set([0,len(part)//2,len(part)-1])))
    return train,valid


def constrain_capture_pose_(m,q,rows):
    """Training-only known-gesture bounds; never used by runtime/holdout IK."""
    angles=m.anatomy.angles(q.detach().cpu().numpy())
    for i,row in enumerate(rows):
        stage=row['stage']
        for f in range(1,5):
            ids=np.array(CHAINS[f])-1
            if stage in ('open','together'):
                angles[i,ids,0]=np.clip(angles[i,ids,0],np.deg2rad(-5),np.deg2rad(5))
            elif stage=='mcp90':
                angles[i,ids[0],0]=np.clip(angles[i,ids[0],0],np.deg2rad(75),np.deg2rad(100))
                angles[i,ids[1:],0]=np.clip(angles[i,ids[1:],0],np.deg2rad(-5),np.deg2rad(7))
            elif stage=='hook':
                angles[i,ids[0],0]=np.clip(angles[i,ids[0],0],np.deg2rad(-5),np.deg2rad(8))
    b=m.anatomy.basis
    local=Rotation.from_euler('XYZ',angles.reshape(-1,3)).as_matrix().reshape(len(rows),15,3,3)
    pose=Rotation.from_matrix((b@local@b.swapaxes(-2,-1)).reshape(-1,3,3)).as_rotvec().reshape(len(rows),15,3)
    with torch.no_grad():q.copy_(torch.tensor(pose,device=m.device,dtype=torch.float32))


def validation_retarget(m,selected,sequence,mounting_calibrated,progress=None,cancel=None):
    """Use the actual sensor-only renderer solver, without gesture/contact hints."""
    m.tracking_time=None;m.raw_aligned=True;m.mounting_calibrated=mounting_calibrated;m.contact_pose_enabled=False
    if m.flat_reference is not None:
        m.has_wrist_targets=True;m.reference_root=np.eye(3,dtype=np.float32)
        m.flat_seed=np.array(m.flat_reference.get('flat_seed',np.zeros((15,3))),dtype=np.float32)
    wanted={r['timestamp_ms'] for r in selected};poses={};roots={};errors=[]
    for index,row in enumerate(sequence):
        if cancel is not None and cancel.is_set():raise InterruptedError("MANO 回放已取消")
        _,_,_,_,error=m.track(row,steps=1)
        if progress is not None:progress(index+1,len(sequence))
        if m.flat_reference is not None:
            target=np.array([row['fingers'][f]['position'] for f in FINGERS])@m.raw_rotation.T+m.raw_translation
            weights=sensor_sample_weights([row])[0]
            error=float(np.sqrt(np.sum(np.sum((m.sensor_predictions-target)**2,axis=-1)*weights)/max(float(weights.sum()),1))*1000)
        if row['timestamp_ms'] in wanted:
            poses[row['timestamp_ms']]=m.last_pose.copy();roots[row['timestamp_ms']]=m.last_root.copy()
        errors.append({'stage':row['stage'],'timestamp_ms':row['timestamp_ms'],'rmse_mm':error,'bilateral_contact':bilateral_contact(row),'guided_contact':row.get('guided_contact',False),
                       'is_approach':row.get('motion_phase') in ('apart','approach','release')})
    if wanted-set(poses):raise ValueError('留出连续序列缺少拟合观测')
    t=lambda x:torch.tensor(np.array(x),device=m.device,dtype=torch.float32)
    return t([poses[r['timestamp_ms']] for r in selected]),t([roots[r['timestamp_ms']] for r in selected]),errors


def trajectory_summary(records):
    result={}
    for stage in dict.fromkeys(r['stage'] for r in records):
        group=[r for r in records if r['stage']==stage]
        approach=[r['rmse_mm'] for r in group if (stage.startswith('pinch_') or stage.startswith('contact_')) and (r.get('is_approach') or not r['bilateral_contact'] and not r.get('guided_contact'))]
        result[stage]={'frames':len(group),'sensor_rmse_mm':float(np.sqrt(np.mean([r['rmse_mm']**2 for r in group]))),
            'approach_frames':len(approach),'approach_sensor_rmse_mm':float(np.sqrt(np.mean(np.square(approach)))) if approach else None}
    return result


def geometry_report(m,rows,poses,roots):
    from .mano_collision import SurfaceCollision
    full_collision=SurfaceCollision(m);collision_frames=[]
    g=CollisionSurface(m);depth=0.;contact=[];records=[];contact_stages=[];together=[];together_pairs={};together_distances=[];contact_records=[];missing_sections=[];side_records=[]
    base_offsets=None;joint_centres=None
    with torch.no_grad():v,j,rr,ordered=m.decode(poses,roots,base_offsets,joint_centres)
    for i,row in enumerate(rows):
        geom=g.query(v[i].cpu().numpy(),j[i].cpu().numpy(),rr[i].cpu().numpy())
        if geom['overflow']:raise ValueError('碰撞缓冲区溢出，候选无效')
        assignments=np.array(m.mapping['assignments']);touch={p['id']:bool(p.get('contact')) for p in row['contact']['patches']}
        confirmed=set(m.mapping.get('confirmed',[]))
        _,collision=full_collision.query(v[i].cpu().numpy())
        depth=max(depth,collision['max_penetration_m'])
        collision_frames.append({'stage':row['stage'],'timestamp_ms':row.get('timestamp_ms'),**collision})
        if row['stage'].startswith('pinch_') and row.get('load_role')=='onset':
            a=0;b=PATCH_IDS.index(row['stage'][6:]+'_distal');ia=np.concatenate((np.flatnonzero(assignments==a),460+np.flatnonzero(m.vertex_patches==a)));ib=np.concatenate((np.flatnonzero(assignments==b),460+np.flatnonzero(m.vertex_patches==b)))
            ds=np.concatenate((geom['gap'][ia,b][geom['facing'][ia,b]<-.15],geom['gap'][ib,a][geom['facing'][ib,a]<-.15]))
            supported=touch.get('thumb_distal') and touch.get(row['stage'][6:]+'_distal') and 'thumb_distal' in confirmed and row['stage'][6:]+'_distal' in confirmed
            supported=supported or row.get('guided_contact',False)
            if supported:
                # A wrong initial pose can give same-facing pads. Guided tactile
                # contact must still measure actual surface distance, never a
                # fabricated fallback residual that can pass acceptance.
                if not len(ds):ds=np.concatenate((geom['gap'][ia,b],geom['gap'][ib,a]))
                ds=ds[np.isfinite(ds)&(np.abs(ds)<.5)]
                if len(ds):contact.append(float(np.min(np.abs(ds))));contact_stages.append(row['stage'])
        if row['stage'].startswith('pinch_'):
            from .tactile_pose import frame_constraints
            _,pairs=frame_constraints(m,row,v[i].cpu().numpy(),ordered[i].cpu().numpy(),guided=True)
            for pair in pairs:
                if pair['evidence']!='confirmed_calibration_gesture':continue
                gap=float(np.linalg.norm(np.array(pair['a']['position_m'])-pair['b']['position_m']))
                # The active material CoPs, not a lucky nearest point anywhere
                # on either patch, define the calibration contact residual.
                contact_records.append(dict(stage=row['stage'],gap_m=gap,normal_dot=float(np.dot(pair['a']['direction'],pair['b']['direction'])),evidence=[pair['a']['evidence'],pair['b']['evidence']],sids=[pair['a']['sids'],pair['b']['sids']]))
        if row['stage'] in ('together','together_repeat'):
            from .side_contact import side_material_pairs
            try:side_records.extend(dict(timestamp_ms=row['timestamp_ms'],**p) for p in side_material_pairs(m,v[i].cpu().numpy(),j[i].cpu().numpy(),include_thumb=row.get('thumb_index_side_contact') is True))
            except ValueError as exc:missing_sections.append({'timestamp_ms':row['timestamp_ms'],'reason':str(exc)})
            try:sections=proximal_pair_gaps(v[i].cpu().numpy(),j[i].cpu().numpy(),m.faces,m.model.weights.cpu().numpy(),CHAINS,True,include_thumb=row.get('thumb_index_side_contact') is True)
            except ValueError as exc:
                sections=[];missing_sections.append({'timestamp_ms':row['timestamp_ms'],'reason':str(exc)})
            together_distances.append([s['surface_distance_m']*1000 for s in sections])
            for section in sections:
                a,b=map(int,section['pair'].split('_'));gap=abs(section['gap_m'])
                together.append(gap);together_pairs.setdefault(FINGERS[a]+'_'+FINGERS[b],[]).append(gap)
        fusion=fuse_contacts(m,row,geom,row['stage']);records.append(fusion)
    if contact_records:contact=[r['gap_m'] for r in contact_records];contact_stages=[r['stage'] for r in contact_records]
    return {'thumb_index_side_contact':any(r.get('thumb_index_side_contact') is True for r in rows),'together_material_records':side_records,'max_penetration_m':depth,'collision_frames':collision_frames,'collision_query_valid':all(r['query_valid'] for r in collision_frames),'noncontact_penetration_m':depth,'contact_gap_m':float(np.mean(contact)) if contact else 0.,'contact_max_gap_m':max(contact,default=0.),'contact_material_records':contact_records,'contact_observations':len(contact),'contact_stages':sorted(set(contact_stages)),
            'together_gap_m':float(np.mean(together)) if together else 0.,'together_observations':len(together),'together_pair_gaps_mm':{k:float(np.mean(v))*1000 for k,v in together_pairs.items()},'together_surface_distances_mm':together_distances,'together_missing_sections':missing_sections,
            'fusion':records,'relative_response_cost':relative_response_cost(rows,records,m.mapping)}



def relative_response_cost(rows,fusions,mapping):
    costs=[]
    for f in FINGERS[1:]:
        predicted=[];observed=[]
        a=PATCH_IDS.index('thumb_distal');b=PATCH_IDS.index(f+'_distal')
        ids=[i for i,p in enumerate(mapping['assignments']) if p in (a,b)]
        for row,fusion in zip(rows,fusions):
            if row['stage']!='pinch_'+f:continue
            baseline=(row.get('tare_profile') or {}).get('baseline')
            if not baseline:continue
            usable=[sid for sid in ids if row['raw'][sid]<254]
            if not usable:continue
            observed.append(sum(max(row['raw'][sid]-baseline[sid],0) for sid in usable)/len(usable))
            predicted.append(sum(p['relative_model_load'] for p in fusion['pairs'] if {p['patch_a'],p['patch_b']}=={'thumb_distal',f+'_distal'}))
        if len(observed)<2 or max(observed)-min(observed)<1:continue
        gain=positive_gain(predicted,observed)
        costs.append(1. if gain is None else float(np.mean(((np.array(predicted)*gain-observed)/max(max(observed),1))**2)))
    return float(np.mean(costs)) if costs else 0.


def adam_articulation(m,rows,poses,roots,steps,cancel,mounting=True):
    """Retired compatibility shim; fit_capture owns the joint shared solve."""
    if cancel.is_set():raise InterruptedError('MANO optimization cancelled')
    return poses.detach().clone()


def position_rmse(m,rows,poses,roots):
    base_offsets=None;joint_centres=None
    with torch.no_grad():v,j,r,o=m.decode(poses,roots,base_offsets,joint_centres)
    dr=r[:,[c[-1] for c in CHAINS]].cpu().numpy();p=(j[:,[c[-1] for c in CHAINS]].cpu().numpy()+(dr@m.sensor_offsets()[None,:,:,None]).squeeze(-1)) if m.flat_reference is not None else v[:,list(TIPS)].cpu().numpy()+(dr@np.asarray(m.mount_pos)[None,:,:,None]).squeeze(-1)
    target=np.array([row['skeleton_targets']['positions_local'] for row in rows]) if m.flat_reference is None and all(row.get('skeleton_targets') for row in rows) else np.array([[row.get('fingers',{}).get(f,{}).get('position',[0,0,0]) for f in FINGERS] for row in rows])@m.raw_rotation.T+m.raw_translation
    weights=sensor_sample_weights(rows)
    return float(np.sqrt(np.sum(np.sum((p-target)**2,axis=-1)*weights)/weights.sum()))


def candidate_validation(baseline_error,error,geometry,tolerance):
    """Primary holdout EMF error, with explicit geometry guards; not training loss."""
    required={'pinch_'+f for f in FINGERS[1:]}
    if geometry.get('contact_protocol')=='contact_motion_v3':
        from .contact_protocol import STAGES as CONTACT_STAGES
        required={s['id'] for s in CONTACT_STAGES}
    gap=geometry.get('contact_gap_m',float('inf'));baseline_gap=geometry.get('baseline_contact_gap_m',0.)
    contact_valid=required<=set(geometry.get('contact_stages',[])) and np.isfinite(gap) and gap<=.003 and geometry.get('contact_max_gap_m',gap)<=.005 and all(r['normal_dot']<-.1 for r in geometry.get('contact_material_records',[]))
    if geometry.get('contact_protocol')=='contact_motion_v3':contact_valid=contact_valid and geometry.get('contact_clusters_valid') is True
    from .mano_collision import HARD_PENETRATION_M
    depth=geometry.get('max_penetration_m',geometry['noncontact_penetration_m'])
    accepted=bool(np.isfinite([baseline_error,error,depth]).all() and depth<=HARD_PENETRATION_M and geometry.get('collision_query_valid',True) and error<baseline_error*.999 and contact_valid)
    return accepted,contact_valid


def together_validation(geometry,baseline):
    if geometry.get('together_missing_sections'):return False
    pairs=geometry.get('together_pair_gaps_mm',{})
    required={'index_middle','middle_ring','ring_pinky'}
    if geometry.get('thumb_index_side_contact'):required.add('thumb_index')
    if set(pairs)!=required or not geometry.get('together_observations'):return False
    if not all(np.isfinite(gap) and gap<=2.5 for gap in pairs.values()):return False
    if any(p['gap_m']>.0025 or p['normal_dot']>=-.1 for p in geometry.get('together_material_records',[])):return False
    distances=np.asarray(geometry.get('together_surface_distances_mm',[]),dtype=float)
    if distances.size:
        before=np.zeros_like(distances) if baseline.get('together_missing_sections') else np.asarray(baseline.get('together_surface_distances_mm',np.zeros_like(distances)),dtype=float)
        if before.shape!=distances.shape or not np.isfinite(distances).all():return False
        # Official betas cannot independently move/rotate each finger base.
        # Check every held-out cross-section, without requiring forced contact.
        return bool(np.all(distances<=2.5))
    return True


def calibration_geometry_indices(rows):
    """Cover all eight gestures, including shared pinches without load_role."""
    subset=[]
    for stage in dict.fromkeys(r['stage'] for r in rows):
        indices=[i for i,r in enumerate(rows) if r['stage']==stage]
        if stage.startswith('pinch_'):
            onset=[i for i in indices if rows[i].get('load_role')=='onset']
            approach=[i for i in indices if rows[i].get('load_role')=='approach']
            subset.extend(onset or [indices[len(indices)//2]])
            if approach:subset.append(approach[0])
        else:subset.append(indices[len(indices)//2])
    return sorted(set(subset))


def geometry_score(position,geometry,tolerance):
    pairs=list(geometry.get('together_pair_gaps_mm',{}).values())
    together_cost=float(np.mean((np.array(pairs)/3.)**2)) if pairs else (geometry.get('together_gap_m',0.)/.003)**2
    normal_cost=np.mean([(1+r['normal_dot'])**2 for r in geometry.get('contact_material_records',[])]) if geometry.get('contact_material_records') else 0.
    invalid_geometry_penalty=1e6*len(geometry.get('together_missing_sections',[]))
    return invalid_geometry_penalty+(position/.003)**2+(geometry['contact_gap_m']/.003)**2+2*normal_cost+4*together_cost+3*(max(geometry['noncontact_penetration_m']-tolerance,0)/.003)**2


def restore_training_snapshot(model,candidate):
    """Restore the original parameter coordinates, bounds AND prior centres."""
    snapshot=candidate['report']['solver']['parameter_snapshot'];x=np.asarray(snapshot['vector'],dtype=float)
    if snapshot['shared_size'] not in (46,89) or len(x)<snapshot['shared_size'] or not np.isfinite(x).all():raise ValueError('无效的共享训练快照')
    model.set_rest_vertex_residual(candidate.get('rest_vertex_residual_m'))
    model.set_shape(candidate['shape']);model.flat_reference=copy.deepcopy(candidate['flat_reference'])
    if candidate.get('glove_parameters') is not None:model.set_glove_geometry(candidate['glove_parameters'])
    base=snapshot.get('shared_reference')
    if base:
        mp=np.asarray(base['sensor_joint_position']);mr=np.asarray(base['mount_rotation'])
        align=np.asarray(base['raw_rotation']);trans=np.asarray(base['raw_translation'])
    else:
        mp=np.asarray(candidate['sensor_joint_position'])-.01*x[10:25].reshape(5,3)
        mr=np.asarray(candidate['mount_rot'])@Rotation.from_rotvec(x[25:40].reshape(5,3)).as_matrix().transpose(0,2,1)
        align=Rotation.from_rotvec(x[40:43]).as_matrix().T@np.asarray(candidate['raw_rotation'])
        trans=np.asarray(candidate['raw_translation'])-.01*x[43:46]
    model.sensor_joint_position=mp;model.mount_rot=mr;model.raw_rotation=align;model.raw_translation=trans
    model.mount_pos=mp-(model.tip_anchors-model.rest_j_np[[c[-1] for c in CHAINS]])
    return x


def fit_capture(model_path,profile,mapping,rows,progress,cancel,max_seconds=300,frozen_candidate=None,initial_candidate=None,solver_settings=None,gpu_method=None,compile_loss=True,training_batch_size=0):
    """Joint calibration with a frozen, all-frame, gesture-blind holdout replay."""
    from .mano_objective import ManoObjective,solve_calibration,SOLVER_REVISION
    from .mano_collision import HARD_PENETRATION_M
    source_capture_hash=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if initial_candidate is not None and initial_candidate.get('report',{}).get('source_capture_hash')!=source_capture_hash:
        raise ValueError('续训必须使用 checkpoint 对应的完整采集与分区')
    started=time.monotonic()
    if rows and rows[0].get('capture_protocol')=='contact_motion_v3':
        from .contact_capture import validate_document
        validate_document(dict(protocol='contact_motion_v3',state=dict(state='complete'),rows=rows))
        if not gpu_method:raise ValueError('区域接触标定需要 GPU 标定入口')
    delta_artifact=None;stage_seconds={};stage_started=started
    def stage_progress(label,lower,upper):
        began=time.monotonic()
        def update(done,total):
            elapsed=time.monotonic()-began
            if done==1 or done%10==0 or done==total:
                progress(dict(state='running',progress=lower+(upper-lower)*done/total,message=f'{label} {done}/{total} 帧',
                              estimated_remaining_s=elapsed*(total-done)/done,eta_scope='stage'))
        return update
    if cancel.is_set():raise InterruptedError('MANO optimization cancelled')
    motion=all(r.get('capture_protocol') in ('geort_motion_v2','contact_motion_v3') for r in rows)
    if gpu_method=='delta' and not motion:raise ValueError('联合 Delta 训练需要完整连续标定记录及独立留出轮次')
    if not motion and profile.get('glove_parameters') is not None:
        raise ValueError('个体手套几何须使用完整连续标定；旧八动作用默认模型离线对照')
    if profile.get('tracking_source')!='official_qnbot_skeleton_fk_v1' or not (motion or all(r.get('capture_protocol')=='official_eight_v1' for r in rows)):
        raise ValueError('MANO 标定仅接受官方八动作记录及官方骨架旋转')
    if any(r.get('mapping',{}).get('assignments')!=mapping['assignments'] for r in rows):raise ValueError('采集期间 SID 分区改变，请重新采集')
    from .training_options import training_rows
    train=training_rows(rows,training_batch_size,full_buffer=gpu_method in ('parameters','flashsac','delta') and motion);valid=holdout_rows(rows)
    validation=validation_retarget;geometry=geometry_report
    if gpu_method:
        from .mano_gpu_runtime import replay as validation
        from .mano_gpu_validation import geometry_report as geometry
        if gpu_method in ('parameters','flashsac','delta'):
            from .mano_fixed import replay as validation
        torch.cuda.reset_peak_memory_stats()
    m=ManoSurface(model_path,side=profile.get('side','left'));m.enable_official_fk();m.contact_pose_enabled=False
    m.registration_serial=profile.get('serial');m.contact_registration=copy.deepcopy(profile.get('contact_registration'))
    m.material=copy.deepcopy(profile['material']);m.binding_overrides=copy.deepcopy(profile.get('binding_overrides',{}));m.build_binding(mapping)
    m.set_rest_vertex_residual(profile.get('rest_vertex_residual_m'))
    m.set_shape(profile['shape']);m.raw_rotation=np.array(profile['raw_rotation']);m.raw_translation=np.array(profile['raw_translation'])
    if profile.get('glove_parameters') is not None:m.set_glove_geometry(profile['glove_parameters'])
    m.mount_pos=np.array(profile['mount_pos']);m.mount_rot=np.array(profile['mount_rot'])
    m.flat_reference=copy.deepcopy(profile.get('flat_reference'));m.sensor_joint_position=np.array(profile['sensor_joint_position']) if profile.get('sensor_joint_position') is not None else None
    if not m.flat_reference or m.flat_reference.get('reference_policy')!=OFFICIAL_REFERENCE_POLICY or m.flat_reference.get('calibration_fingerprint')!=rows[0].get('calibration_fingerprint'):
        opened=next(r for r in rows if r['stage']=='open')
        m.install_flat_reference(reference_from_frame(opened,m.weights_hash,profile.get('serial'),'joint-calibration-training'),allow_coarse=True)
    contact_templates=None
    if rows[0].get('capture_protocol')=='contact_motion_v3':
        rows=copy.deepcopy(rows)
        from .contact_templates import prepare_templates,reuse_templates
        progress(dict(state='running',phase='templates',progress=.03,message='恢复 checkpoint 冻结接触模板' if initial_candidate else '标准 MANO：从训练事件构建冻结区域接触模板',estimated_remaining_s=None))
        templates_began=time.monotonic()
        contact_templates=reuse_templates(m,rows,initial_candidate['contact_templates']) if initial_candidate else prepare_templates(m,rows,cancel)
        stage_seconds['templates']=time.monotonic()-templates_began;stage_started=time.monotonic()
        train=training_rows(rows,training_batch_size,full_buffer=True);valid=holdout_rows(rows)
    auxiliary=[r for r in rows if r.get('initialization')] if contact_templates else []
    def initialization_checks():
        vals=[m.official_fk.parameters(m,row,CHAINS) for row in auxiliary]
        aq=torch.as_tensor(np.asarray([v[0] for v in vals]),device=m.device,dtype=torch.float32)
        ar=torch.as_tensor(np.asarray([v[1] for v in vals]),device=m.device,dtype=torch.float32)
        return geometry(m,[dict(r,capture_protocol='official_eight_v1') for r in auxiliary],aq,ar,check_collision=False),flat_pose_report(m,auxiliary,aq,ar)
    dimensions_before=m.dimension_summary()
    # Old offsets are evaluated for comparison only, never installed in the new fit.
    m.calibration_pose_offsets=np.asarray(profile.get('calibration_pose_offsets',np.zeros((15,3))),np.float32)
    m.solver_enabled=False
    progress({'state':'running','progress':.03 if gpu_method else .01,'message':'完整留出基线：逐指定位与并拢截面检查'})
    qb,rb,baseline_trajectory=validation(m,valid,valid,bool(profile.get('mounting_calibrated')),stage_progress('基线验证',.03,.08),cancel)
    baseline_pos=position_rmse(m,valid,qb,rb);basegeo=geometry(m,valid,qb,rb,check_collision=False) if gpu_method else geometry(m,valid,qb,rb);baseline_flat=flat_pose_report(m,valid,qb,rb)
    auxiliary_baseline=initialization_checks()[0] if auxiliary else None
    stage_seconds['baseline']=time.monotonic()-stage_started;stage_started=time.monotonic()
    m.calibration_pose_offsets=np.zeros((15,3),np.float32);m.solver_enabled=True
    resume_phases=None
    if initial_candidate is not None:
        c=initial_candidate
        if frozen_candidate is not None:raise ValueError('Cannot resume and freeze the same fit')
        if c.get('weights_hash')!=m.weights_hash or c.get('side')!=m.side or c.get('calibration_fingerprint')!=rows[0]['calibration_fingerprint'] or c.get('skeleton_runtime_revision')!=rows[0]['skeleton_runtime_revision']:
            raise ValueError('训练快照与采集、手侧或运行时不兼容')
        if c.get('report',{}).get('fit_revision')!=FIT_REVISION or c['report'].get('solver_revision')!=SOLVER_REVISION:
            raise ValueError('只能续算同一目标函数版本的训练快照')
        restore_training_snapshot(m,c)
    if frozen_candidate is None:
        values=[m.official_fk.parameters(m,row,CHAINS) for row in train]
        # The small differentiable MANO batch is launch-bound on CUDA. Keep
        # exact autograd on CPU and full-surface collision queries on CUDA.
        # Both views retain the same official weights and canonical chart.
        compute=m
        if m.device.type=='cuda' and gpu_method is None:
            from .mano_collision import SurfaceCollision
            compute=copy.copy(m);compute.device=torch.device('cpu');compute.model=copy.deepcopy(m.model).to('cpu')
            collision=SurfaceCollision(m)
        else:collision=None
        Objective=ManoObjective
        if gpu_method:
            from .mano_gpu_objective import GPUManoObjective
            Objective=GPUManoObjective
            if gpu_method in ('parameters','flashsac') and motion:
                from .mano_mixed_training import MixedGPUManoObjective
                Objective=MixedGPUManoObjective
            if gpu_method=='delta' and motion:
                from .mano_joint_delta import JointGPUManoObjective
                Objective=JointGPUManoObjective
        if gpu_method:progress(dict(state='running',phase='compile',progress=.07,message='GPU 数据缓冲区与并行 FK 编译／预热',estimated_remaining_s=None))
        resume_policy={}
        if initial_candidate is not None and gpu_method=='delta':
            if not initial_candidate.get('delta_policy') or initial_candidate['report'].get('method')!='delta':raise ValueError('联合续训缺少 Delta 权重')
            resume_policy['initial_policy']=initial_candidate['delta_policy']
        objective=Objective(compute,train,np.array([v[0] for v in values]),np.array([v[1] for v in values]),training=True,shared=True,tactile=True,collision=collision,settings=solver_settings,fixed_forward=gpu_method in ('parameters','flashsac','delta'),**resume_policy)
        if initial_candidate is not None:
            previous=initial_candidate['report']['solver'];snapshot=previous['parameter_snapshot']
            if snapshot['frame_timestamps_ms']!=[r['timestamp_ms'] for r in train] or snapshot['free_coordinates']!=objective.free.cpu().tolist():
                raise ValueError('训练快照的帧顺序或参与掩码不一致')
            saved=objective.t(snapshot['vector'])
            if len(saved)!=len(objective.initial) or not torch.isfinite(saved).all():raise ValueError('无效的完整训练快照')
            if (saved<objective.lower-1e-5).any() or (saved>objective.upper+1e-5).any():raise ValueError('训练快照超出原有参数限位')
            objective.initial=saved.clamp(objective.lower,objective.upper)
            objective.resume_info=dict(parent_run_id=initial_candidate['report'].get('run_id'),
                previous_steps=previous.get('total_training_steps',sum(p['steps'] for p in previous['phases'])),
                geometry_restored=True,delta_restored=bool(resume_policy),normalization_restored=bool(resume_policy),
                contact_templates_reused=bool(contact_templates),optimizer_state_restored=False,
                optimizer_state_reason='checkpoint_does_not_include_optimizer_moments')
            objective.refresh(objective.initial,contacts=True)
            resume_phases={p['phase']:p['steps'] for p in previous['phases']}
        solve=solve_calibration
        if gpu_method:
            if gpu_method=='flashsac':from .mano_calibrate_flashsac import solve_calibration_gpu
            else:from .mano_gpu_solver import solve_calibration_gpu
            solve=lambda *a,**kw:solve_calibration_gpu(*a,**kw,compile_loss=compile_loss)
        optimizer=solve(objective,cancel,progress,max_seconds=max(.01,max_seconds if gpu_method else max_seconds-(time.monotonic()-started)),resume_phases=resume_phases)
        optimizer.update(pose_compute_device=str(compute.device),collision_compute_device=str(objective.collision.device))
        if compute is not m:
            # Install the identical complete snapshot, not a mix of stages.
            m.set_rest_vertex_residual(compute.model.rest_vertex_residual.detach().cpu().numpy())
            objective.model=m;objective.install(objective.t(optimizer['parameter_snapshot']['vector']))
        if initial_candidate is not None:optimizer['resumed_training_snapshot']=True
        if gpu_method=='delta' and not getattr(objective,'joint_training',False):
            from .mano_calibrate_delta import train_policy
            values=[m.official_fk.parameters(m,row,CHAINS) for row in train]
            del objective
            objective=GPUManoObjective(m,train,np.array([v[0] for v in values]),np.array([v[1] for v in values]),training=True,shared=True,tactile=True,settings=solver_settings)
            shared=objective.initial.detach().clone()
            network,delta_artifact=train_policy(objective,shared,cancel,progress,max_seconds=15,compile_loss=compile_loss)
            delta_artifact.update(side=m.side,weights_hash=m.weights_hash,serial=profile['serial'],calibration_fingerprint=rows[0]['calibration_fingerprint'],binding_version=m.binding_version,mapping_version=mapping.get('version'))
            m.delta_policy=network;m.delta_artifact=delta_artifact
            optimizer['budget_exhausted']=optimizer['budget_exhausted'] or delta_artifact['budget_exhausted']
        elif gpu_method=='delta':delta_artifact=m.delta_artifact
        if gpu_method:
            optimizer['gesture_masks']={row['stage']:objective.mask_np[i].astype(int).tolist() for i,row in enumerate(train)}
            optimizer['shared_beta_affects_whole_hand']=True
            m.gpu_pose_enabled=True

    else:
        c=frozen_candidate
        if c.get('weights_hash')!=m.weights_hash or c.get('side')!=m.side or c.get('calibration_fingerprint')!=rows[0]['calibration_fingerprint'] or c.get('skeleton_runtime_revision')!=rows[0]['skeleton_runtime_revision']:
            raise ValueError('冻结候选与采集、手侧或模型不兼容')
        m.set_rest_vertex_residual(c.get('rest_vertex_residual_m'))
        m.set_shape(c['shape']);m.sensor_joint_position=np.array(c['sensor_joint_position']);m.flat_reference=copy.deepcopy(c['flat_reference'])
        for name in ('mount_pos','mount_rot','raw_rotation','raw_translation'):setattr(m,name,np.array(c[name]))
        optimizer=copy.deepcopy(c['report']['solver'])
        if c.get('glove_parameters') is not None:m.set_glove_geometry(c['glove_parameters'])
        if c.get('delta_policy') is not None:
            from .mano_calibrate_delta import load_policy
            delta_artifact=copy.deepcopy(c['delta_policy']);m.delta_policy=load_policy(delta_artifact,m.device);m.delta_artifact=delta_artifact
    if cancel.is_set():raise InterruptedError('MANO optimization cancelled')
    stage_seconds['identification']=time.monotonic()-stage_started;stage_started=time.monotonic()
    progress({'state':'running','progress':.80,'message':'冻结几何与策略权重：固定前向回放全部留出帧' if gpu_method in ('parameters','flashsac','delta') else '固定共享参数：MANO 求解器回放留出帧'})
    skeleton_payload=None;skeleton_validation=None
    if motion:
        from .glove_geometry import export_skeleton
        skeleton_payload=export_skeleton(m)
        if not gpu_method:valid,skeleton_validation=replay_personalized_skeleton(valid,profile,skeleton_payload,cancel,progress=stage_progress("骨架留出验证",.80,.84))
    m._mano_solver_previous=None
    q,r,trajectory=validation(m,valid,valid,True,stage_progress("MANO 留出验证",.84,.90),cancel)
    pos=position_rmse(m,valid,q,r);geo=geometry(m,valid,q,r);flat=flat_pose_report(m,valid,q,r)
    if motion and gpu_method:
        from .mano_gpu_validation import skeleton_report
        skeleton_validation=skeleton_report(valid,m._gpu_replay_reports)
    accepted,contact_valid=candidate_validation(baseline_pos,pos,geo,HARD_PENETRATION_M)
    if auxiliary:
        auxiliary_geo,flat=initialization_checks()
        together_valid=together_validation(auxiliary_geo,auxiliary_baseline)
        for key in ('thumb_index_side_contact','together_material_records','together_pair_gaps_mm','together_surface_distances_mm'):
            geo[key]=auxiliary_geo[key];basegeo[key]=auxiliary_baseline[key]
        baseline_flat=dict(baseline_flat,scope='SDK_initialization_diagnostic_not_independent_pass')
        flat.update(scope='SDK_initialization_diagnostic_not_independent_pass')
    else:together_valid=together_validation(geo,basegeo)
    before=trajectory_summary(baseline_trajectory);after=trajectory_summary(trajectory)
    per_stage_valid=all(after[s]['sensor_rmse_mm']<=before[s]['sensor_rmse_mm']*1.05+.5 for s in before)
    penetration_valid=bool(geo['collision_query_valid'] and geo['max_penetration_m']<=HARD_PENETRATION_M)
    accepted=bool(accepted and together_valid and flat['valid'] and per_stage_valid)
    # A separate replay measures the actual tactile-enabled path. Shared
    # parameters remain frozen and are never selected using these holdout data.
    stage_seconds['sensor_validation']=time.monotonic()-stage_started;stage_started=time.monotonic()
    m.contact_pose_enabled=True;m.tracking_time=None;m._mano_solver_previous=None
    runtime_rows=valid
    if gpu_method:
        replay_label='GPU 固定权重前向回放验证' if gpu_method in ('parameters','flashsac','delta') else 'GPU 触觉实时路径验证'
        rq,rr,_=validation(m,valid,valid,True,stage_progress(replay_label,.90,.99),cancel)
        actual=[{'stage':row['stage'],'timestamp_ms':row['timestamp_ms'],**r} for row,r in zip(valid,m._gpu_replay_reports)]
        runtime_poses=rq.cpu().numpy();runtime_roots=rr.cpu().numpy()
        if motion:
            skeleton_validation['tactile_path']=skeleton_report(valid,m._gpu_replay_reports)
            skeleton_validation['accepted'] &= skeleton_validation['tactile_path']['accepted']
    else:
        if motion:
            runtime_rows,skeleton_tactile=replay_personalized_skeleton(valid,profile,skeleton_payload,cancel,tactile=True,progress=stage_progress("触觉骨架验证",.90,.94))
            skeleton_validation['tactile_path']=skeleton_tactile
            skeleton_validation['accepted']=bool(skeleton_validation['accepted'] and skeleton_tactile['accepted'])
        actual=[];runtime_poses=[];runtime_roots=[]
        runtime_progress=stage_progress("实时路径验证",.94,.99)
        for index,row in enumerate(runtime_rows):
            if cancel.is_set():raise InterruptedError('MANO optimization cancelled')
            m.track(row);runtime_poses.append(m.last_pose.copy());runtime_roots.append(m.last_root.copy())
            runtime_progress(index+1,len(runtime_rows))
            actual.append({'stage':row['stage'],'timestamp_ms':row['timestamp_ms'],**copy.deepcopy(m.solver_report)})
    stage_seconds['tactile_validation']=time.monotonic()-stage_started
    mjwarp_check=None
    if gpu_method:
        from .mano_mjwarp_check import cross_check
        mjwarp_check=cross_check(m,rq,rr,cancel)
    runtime_penetration=max((a['collision']['max_penetration_m'] for a in actual),default=float('inf'))
    runtime_valid=runtime_penetration<=HARD_PENETRATION_M and all(a['collision']['query_valid'] and a['constraints_satisfied'] and a.get('source_tracking_satisfied',True) for a in actual)
    independent=not motion or independent_cycles_valid(rows)
    approach_valid=not motion or all(after[s]['approach_sensor_rmse_mm']<=before[s]['approach_sensor_rmse_mm']*1.05+.5
        for s in before if before[s]['approach_sensor_rmse_mm'] is not None and after[s]['approach_sensor_rmse_mm'] is not None)
    if contact_templates:approach_valid=bool(approach_valid and geo.get('approach_release_valid'))
    skeleton_valid=not motion or skeleton_validation['accepted']
    from .mano_shape_prior import ShapePrior
    from .calibration_targets import DEFINITION
    shape_validation=ShapePrior(m).report(torch.as_tensor(m.shape[:10],device=m.device,dtype=torch.float32),getattr(m.model,'glove_parameters',None))
    fixed_forward=gpu_method in ('parameters','flashsac','delta')
    mjwarp_valid=not gpu_method or bool((mjwarp_check or {}).get('query_valid'))
    accepted=bool(accepted and mjwarp_valid and runtime_valid and independent and approach_valid and skeleton_valid and not optimizer.get('budget_exhausted') and (shape_validation['accepted'] or not fixed_forward))
    failures=[]
    if not mjwarp_valid:failures.append('MJWarp 独立交叉检查失败；训练候选已保留但不应用')
    if fixed_forward and not shape_validation['accepted']:failures.append('标准 MANO 厚度／局部形变保护未通过')
    if pos>=baseline_pos*.999:failures.append('留出定位误差未改善')
    if not per_stage_valid:failures.append('逐动作定位误差退化')
    if not contact_valid:failures.append('指定接触区域间隙／法向／连通性未通过')
    if not together_valid:failures.append('并拢逐截面间隙未通过')
    if not flat['valid']:failures.append('平摊伸直／平面／并拢方向未通过')
    if not penetration_valid:failures.append('静态留出曲面穿透超过2 mm或查询无效')
    if not runtime_valid:failures.append('实时路径留出接触／2 mm碰撞检查未通过')
    if not independent:failures.append('完整独立复测缺失')
    if not approach_valid:failures.append('接近／释放阶段定位误差退化')
    if not skeleton_valid:failures.append('个体骨架留出 EMF 检查未通过')
    if gpu_method and optimizer.get('collision_witness_complete') is False:
        accepted=False;failures.append('GPU 训练碰撞见证缓冲区溢出')
    if optimizer.get('budget_exhausted'):failures.append('优化预算耗尽，请继续拟合后验收')
    report=dict(state='preview',accepted=accepted,fit_revision=FIT_REVISION,solver_revision=SOLVER_REVISION,
        optimizer=('differentiable_MANO_'+optimizer.get('optimizer','Adam')+'_with_EMA_frozen_forward') if fixed_forward else 'differentiable_MANO_Adam_then_bounded_block_LM',solver=optimizer,
        shared_parameters_refit=frozen_candidate is None,
        frozen_forward=fixed_forward,online_optimization=False if fixed_forward else True,shape_validation=shape_validation,
        expected_contact_regions=DEFINITION,contact_templates=contact_templates,
        baseline_position_rmse_mm=baseline_pos*1000,position_rmse_mm=pos*1000,
        baseline_contact_gap_mm=basegeo['contact_gap_m']*1000,contact_gap_mm=geo['contact_gap_m']*1000,
        baseline_contact_normal_error_deg=basegeo.get('contact_normal_error_deg'),baseline_contact_smooth_normal_error_deg=basegeo.get('contact_smooth_normal_error_deg'),
        contact_max_gap_mm=geo['contact_max_gap_m']*1000,contact_material_records=geo['contact_material_records'],
        contact_normal_error_deg=geo.get('contact_normal_error_deg'),contact_normal_max_error_deg=geo.get('contact_normal_max_error_deg'),
        contact_smooth_normal_error_deg=geo.get('contact_smooth_normal_error_deg'),
        approach_release_geometry=geo.get('approach_release_geometry'),
        contact_clusters=geo.get('contact_clusters'),preferred_thumb_contacts=geo.get('preferred_thumb_records'),
        auxiliary_initialization_checks=auxiliary_geo if auxiliary else None,
        contact_stages=geo['contact_stages'],contact_observations=geo['contact_observations'],contact_validation_passed=contact_valid,
        together_validation_passed=together_valid,together_pair_gaps_mm=geo['together_pair_gaps_mm'],
        thumb_index_side_contact=geo['thumb_index_side_contact'],together_material_records=geo['together_material_records'],
        side_contact_evidence='operator_confirmed_calibration_gesture' if geo['thumb_index_side_contact'] else 'official_four_finger_together_prior',
        baseline_together_pair_gaps_mm=basegeo['together_pair_gaps_mm'],together_surface_distances_mm=geo['together_surface_distances_mm'],
        flat_validation_passed=flat['valid'],baseline_flat_pose=baseline_flat,flat_pose=flat,
        penetration_validation_passed=penetration_valid,noncontact_penetration_mm=geo['max_penetration_m']*1000,
        max_penetration_mm=geo['max_penetration_m']*1000,collision_frames=geo['collision_frames'],
        penetration_tolerance_mm=2.,penetration_policy='hard_calibration_gate_runtime_best_finite_with_warning',
        runtime_validation_passed=runtime_valid,runtime_max_penetration_mm=runtime_penetration*1000,
        runtime_solver_frames=actual,runtime_latency_ms={k:float(np.percentile([a['elapsed_ms'] for a in actual],v)) for k,v in [('p50',50),('p95',95),('max',100)]},
        per_stage_validation_passed=per_stage_valid,trajectory_before=before,trajectory_after=after,
        capture_protocol=rows[0].get('capture_protocol','official_eight_v1'),validation_pose_source='runtime_mano_sensor_only',tactile_pose_feedback=False,
        validation_scope=('separate_complete_contact_pass' if rows[0].get('capture_protocol')=='contact_motion_v3' else 'complete_third_cycles_and_final_repeats') if motion else 'all_held_out_frames_within_eight_gestures; no_repeat_or_approach_capture',holdout_validation_passed=accepted,
        independent_cycles_valid=independent,skeleton_validation_passed=skeleton_valid,skeleton_validation=skeleton_validation,
        baseline_pose_source='legacy_official_fk_and_offsets_diagnostic_only',approach_validation_passed=approach_valid if motion else None,
        selected_training_frames=len(train),selected_validation_frames=len(valid),continuous_validation_frames=len(valid),
        gesture_masks=optimizer['gesture_masks'],shared_beta_affects_whole_hand=True,
        source_capture_hash=source_capture_hash,
        dimensions_before_mm=dimensions_before,dimensions_after_mm=m.dimension_summary(),shape_parameter_count=10,geometry_parameter_count=43 if motion else 0,
        force_calibrated=False,geometry_fixed_during_tracking=True,pose_correctives=True,
        seconds=time.monotonic()-started,rejection_reasons=failures,quality_warnings=[],
        reason='完整留出帧定位、动作及2 mm碰撞检查通过' if accepted else '；'.join(failures)+'；当前模型保留')
    result=copy.deepcopy(profile)
    result.update(calibrated=False,revision=None,applied_at_ms=None,dimension_mode=OFFICIAL_MANO_MODE,
        dimension_parameters=np.zeros(DIMENSION_COUNT).tolist(),shape=m.shape.tolist(),betas=m.shape[:10].tolist(),
        rest_vertex_residual_m=m.model.rest_vertex_residual.detach().cpu().tolist(),
        calibration_pose_offsets=np.zeros((15,3)).tolist(),calibration_model='smplx.MANO',mano_solver_revision=SOLVER_REVISION,
        penetration_tolerance_m=HARD_PENETRATION_M,material=m.material,geometry_revision=GEOMETRY_REVISION,
        flat_reference=copy.deepcopy(m.flat_reference),sensor_joint_position=m.sensor_offsets().tolist(),
        mounting_calibrated=accepted,mounting_estimated=True,observation_space='raw_emf_sensor',
        raw_rotation=m.raw_rotation.tolist(),raw_translation=m.raw_translation.tolist(),mount_pos=m.mount_pos.tolist(),mount_rot=m.mount_rot.tolist(),
        calibration_fingerprint=rows[0]['calibration_fingerprint'],skeleton_runtime_revision=rows[0]['skeleton_runtime_revision'],
        mapping=mapping,bone_reference=m.bone_reference.tolist() if m.bone_reference is not None else None,response_gains={})
    if gpu_method:
        duration=time.monotonic()-started
        compilation=optimizer.get('compilation_s',0.)+(delta_artifact or {}).get('compilation_s',0.)+getattr(m,'_gpu_runtime_compile_s',0.)+getattr(m,'_gpu_surface_compile_s',0.)+(mjwarp_check or {}).get('compilation_s',0.)
        report.update(method=gpu_method,quality_passed=bool(accepted),runtime_latency_scope='GPU_replay_batch_wall_time_not_single_frame',stage_seconds=stage_seconds,mjwarp_cross_check=mjwarp_check,gpu_backend='torch_warp_v1',end_to_end_s=duration,compilation_s=compilation,
                      performance_passed=duration-compilation<=60.,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
                      performance_target_s=60.,runtime_backend='gpu_mano_v1')
        if not report['performance_passed']:
            report.update(accepted=False,holdout_validation_passed=False,reason=report['reason']+'；端到端耗时超过60秒，当前模型保留')
            report['rejection_reasons'].append('端到端性能未达标')
        if delta_artifact is not None:
            delta_artifact.update(side=m.side,weights_hash=m.weights_hash,serial=profile['serial'],calibration_fingerprint=rows[0]['calibration_fingerprint'],geometry_hash=skeleton_payload['geometry_hash'],binding_version=m.binding_version,mapping_version=mapping.get('version'),surface_version=m.shape_version)
            result['delta_policy']=delta_artifact
        if gpu_method in ('parameters','flashsac'):result.pop('delta_policy',None)
        result['runtime_backend']='fixed_mano_v1' if fixed_forward else 'gpu_mano_v1'
        report['runtime_backend']=result['runtime_backend']
        if fixed_forward:report['runtime_latency_scope']='frozen_forward_replay_and_diagnostics_per_frame_amortized'
        result['skeleton_seed_model']=copy.deepcopy(rows[0].get('skeleton',{}).get('personalized_model'))
    if contact_templates:result['contact_templates']=contact_templates
    result.pop('artifact_home',None)
    if motion:result.update(glove_parameters=m.glove_parameters.tolist(),skeleton_model=skeleton_payload)
    return result,report


def independent_cycles_valid(rows):
    if rows and rows[0].get('capture_protocol')=='contact_motion_v3':
        from .contact_capture import validate_document
        try:validate_document(dict(protocol='contact_motion_v3',state=dict(state='complete'),rows=rows))
        except (ValueError,KeyError):return False
        return True
    from .motion_capture import STAGES,confirmed_touch
    for stage in STAGES:
        group=[r for r in rows if r['stage']==stage['id'] and not r.get('initialization')]
        if not group:return False
        if stage['cycles']==3:
            for c in range(3):
                cycle=[r for r in group if r.get('cycle')==c]
                if not cycle or any(r.get('partition')!=('validation' if c==2 else 'train') for r in cycle):return False
                if stage['id'].startswith('pinch_') and not any(confirmed_touch(r) for r in cycle):return False
        elif any(r.get('partition')!='validation' for r in group):return False
    return True


def replay_personalized_skeleton(rows,profile,payload,cancel,tactile=False,progress=None):
    """Frozen shared geometry is tested through the real SLSQP process."""
    import subprocess
    from pathlib import Path
    home=Path(profile['artifact_home']);folder=home/'joint-candidates'/payload['geometry_hash'];folder.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1];key=hashlib.sha256(profile['serial'].encode()).hexdigest()[:16]
    config=home.parents[1]/'ik'/key/'runtime'/('kinematic-'+rows[0]['calibration_fingerprint'][:16]+'.yaml')
    tag='tactile' if tactile else 'sensor'
    source=folder/(tag+'-source.json');target=folder/(tag+'-replayed.json');geometry=folder/'geometry.json'
    source.write_text(json.dumps({'rows':rows},allow_nan=False));geometry.write_text(json.dumps(payload,allow_nan=False))
    command=[str(root/'.venv-moxian-ik/bin/python'),'-I',str(root/'moxian_web/prepare_wrist_targets.py'),'--input',str(source),'--output',str(target),'--config',str(config),'--fingerprint',rows[0]['calibration_fingerprint'],'--runtime-revision',rows[0]['skeleton_runtime_revision'],'--personalized',str(geometry)]
    if tactile:command.append('--tactile')
    from .fit_process import run_replay
    run_replay(command,cancel,lambda value:progress(value['completed'],value['total']) if progress else None)
    replay=json.loads(target.read_text())['rows'];errors={}
    baseline={}
    for original,row in zip(rows,replay):
        for f,record in row['skeleton']['tracking_errors'].items():
            errors.setdefault(f,[]).append(record['position_error_mm'])
            previous=original['skeleton'].get('tracking_errors',{}).get(f,{})
            baseline.setdefault(f,[]).append(previous.get('position_error_mm',float('inf')))
    summary={f:dict(rmse_mm=float(np.sqrt(np.mean(np.square(e)))),max_mm=float(max(e)),baseline_rmse_mm=float(np.sqrt(np.mean(np.square(baseline[f]))))) for f,e in errors.items()}
    accepted=all(v['rmse_mm']<=v['baseline_rmse_mm']*1.05+.5 and v['rmse_mm']<=15. for v in summary.values())
    return replay,dict(accepted=accepted,per_finger=summary,frames=len(replay),source='frozen_geometry_tactile_SLSQP' if tactile else 'frozen_geometry_sensor_only_SLSQP')
