"""Bounded, held-out MANO calibration. Runs on a private model in a worker thread."""
from __future__ import annotations
import copy,math,time,hashlib,json
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from .mano_runtime import ManoSurface,CollisionSurface,skeleton_targets,fuse_contacts
from .contact import PATCH_IDS,FINGERS
from .hand_model import positive_gain
from .hand_anatomy import GEOMETRY_REVISION
from .hand_dimensions import DIMENSION_MODE,DIMENSION_COUNT,PENETRATION_TOLERANCE_M
from scripts.calibrate_wuji_mano import TIPS,CHAINS,rotmat
from .proximal_shape import proximal_pair_gaps,refine_proximal_widths


FIT_REVISION='official_full_se3_proximal_sections_v12'


def bilateral_contact(row):
    stage=row.get('stage','')
    if not stage.startswith('pinch_'):return False
    p={p['id']:p for p in row.get('contact',{}).get('patches',[])}
    return bool(row.get('contact',{}).get('ready') and p.get('thumb_distal',{}).get('contact') and p.get(stage[6:]+'_distal',{}).get('contact'))


def holdout_rows(rows):
    result=[]
    for stage in dict.fromkeys(r['stage'] for r in rows):
        group=sorted((r for r in rows if r['stage']==stage),key=lambda r:r['timestamp_ms'])
        result.extend(group if stage=='open_repeat' else [r for r in group if r.get('cycle',0)>=1] if stage.startswith('pinch_') else group[len(group)//2:])
    return sorted(result,key=lambda r:r['timestamp_ms'])


def choose_rows(rows):
    train=[];valid=[]
    for stage in dict.fromkeys(r['stage'] for r in rows):
        group=sorted((r for r in rows if r['stage']==stage),key=lambda r:r['timestamp_ms'])
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


def validation_retarget(m,selected,sequence,mounting_calibrated):
    """Use the actual sensor-only renderer solver, without gesture/contact hints."""
    m.tracking_time=None;m.raw_aligned=True;m.mounting_calibrated=mounting_calibrated
    if m.flat_reference is not None:
        m.has_wrist_targets=True;m.reference_root=np.eye(3,dtype=np.float32)
        m.flat_seed=np.array(m.flat_reference.get('flat_seed',np.zeros((15,3))),dtype=np.float32)
    wanted={r['timestamp_ms'] for r in selected};poses={};roots={};errors=[]
    for row in sequence:
        _,_,_,_,error=m.track(row,steps=1)
        if row['timestamp_ms'] in wanted:
            poses[row['timestamp_ms']]=m.last_pose.copy();roots[row['timestamp_ms']]=m.last_root.copy()
        errors.append({'stage':row['stage'],'timestamp_ms':row['timestamp_ms'],'rmse_mm':error,'bilateral_contact':bilateral_contact(row)})
    if wanted-set(poses):raise ValueError('留出连续序列缺少拟合观测')
    t=lambda x:torch.tensor(np.array(x),device=m.device,dtype=torch.float32)
    return t([poses[r['timestamp_ms']] for r in selected]),t([roots[r['timestamp_ms']] for r in selected]),errors


def trajectory_summary(records):
    result={}
    for stage in dict.fromkeys(r['stage'] for r in records):
        group=[r for r in records if r['stage']==stage]
        approach=[r['rmse_mm'] for r in group if stage.startswith('pinch_') and not r['bilateral_contact']]
        result[stage]={'frames':len(group),'sensor_rmse_mm':float(np.sqrt(np.mean([r['rmse_mm']**2 for r in group]))),
            'approach_frames':len(approach),'approach_sensor_rmse_mm':float(np.sqrt(np.mean(np.square(approach)))) if approach else None}
    return result


def geometry_report(m,rows,poses,roots):
    g=CollisionSurface(m);depth=0.;contact=[];records=[];contact_stages=[];together=[];together_pairs={}
    base_offsets=None;joint_centres=None
    if m.official_fk is not None:poses,roots,base_offsets,joint_centres=m.official_parameters(rows)
    with torch.no_grad():v,j,rr,ordered=m.decode(poses,roots,base_offsets,joint_centres)
    for i,row in enumerate(rows):
        geom=g.query(v[i].cpu().numpy(),j[i].cpu().numpy(),rr[i].cpu().numpy())
        if geom['overflow']:raise ValueError('碰撞缓冲区溢出，候选无效')
        assignments=np.array(m.mapping['assignments']);touch={p['id']:bool(p.get('contact')) for p in row['contact']['patches']}
        confirmed=set(m.mapping.get('confirmed',[]));worst=0.
        for a in range(20):
            for b in range(a+1,20):
                if m.patch_owners[a]==m.patch_owners[b]:continue
                owners=tuple(sorted((m.patch_owners[a],m.patch_owners[b])))
                if owners not in geom['native_pairs']:continue
                ia=np.concatenate((np.flatnonzero(assignments==a),460+np.flatnonzero(m.vertex_patches==a)));ib=np.concatenate((np.flatnonzero(assignments==b),460+np.flatnonzero(m.vertex_patches==b)))
                ds=np.concatenate((geom['gap'][ia,b][geom['facing'][ia,b]<-.15],geom['gap'][ib,a][geom['facing'][ib,a]<-.15]))
                if not len(ds):continue
                pair_touch=touch.get(PATCH_IDS[a]) and touch.get(PATCH_IDS[b]) and PATCH_IDS[a] in confirmed and PATCH_IDS[b] in confirmed
                fa=FINGERS.index(PATCH_IDS[a].split('_')[0]) if a<14 else -1
                fb=FINGERS.index(PATCH_IDS[b].split('_')[0]) if b<14 else -1
                if row['stage']=='together' and min(fa,fb)>=1 and abs(fa-fb)==1:pair_touch=True
                if not pair_touch:worst=max(worst,float(max(0,-ds.min())))
        depth=max(depth,worst)
        if row['stage'].startswith('pinch_') and row.get('load_role')=='onset':
            a=0;b=PATCH_IDS.index(row['stage'][6:]+'_distal');ia=np.concatenate((np.flatnonzero(assignments==a),460+np.flatnonzero(m.vertex_patches==a)));ib=np.concatenate((np.flatnonzero(assignments==b),460+np.flatnonzero(m.vertex_patches==b)))
            ds=np.concatenate((geom['gap'][ia,b][geom['facing'][ia,b]<-.15],geom['gap'][ib,a][geom['facing'][ib,a]<-.15]))
            supported=touch.get('thumb_distal') and touch.get(row['stage'][6:]+'_distal') and 'thumb_distal' in confirmed and row['stage'][6:]+'_distal' in confirmed
            if supported:
                # A wrong initial pose can give same-facing pads. Guided tactile
                # contact must still measure actual surface distance, never a
                # fabricated fallback residual that can pass acceptance.
                if not len(ds):ds=np.concatenate((geom['gap'][ia,b],geom['gap'][ib,a]))
                ds=ds[np.isfinite(ds)&(np.abs(ds)<.5)]
                if len(ds):contact.append(float(np.min(np.abs(ds))));contact_stages.append(row['stage'])
        if row['stage']=='together':
            sections=proximal_pair_gaps(v[i].cpu().numpy(),j[i].cpu().numpy(),m.faces,m.model.weights.cpu().numpy(),CHAINS)
            for section in sections:
                a,b=map(int,section['pair'].split('_'));gap=abs(section['gap_m'])
                together.append(gap);together_pairs.setdefault(FINGERS[a]+'_'+FINGERS[b],[]).append(gap)
        fusion=fuse_contacts(m,row,geom,row['stage']);records.append(fusion)
    return {'noncontact_penetration_m':depth,'contact_gap_m':float(np.mean(contact)) if contact else 0.,'contact_observations':len(contact),'contact_stages':sorted(set(contact_stages)),
            'together_gap_m':float(np.mean(together)) if together else 0.,'together_observations':len(together),'together_pair_gaps_mm':{k:float(np.mean(v))*1000 for k,v in together_pairs.items()},
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
    device=m.device;t=lambda x:torch.tensor(np.array(x),device=device,dtype=torch.float32)
    official=m.official_fk is not None;base_offsets=None;joint_centres=None
    if official:poses,roots,base_offsets,joint_centres=m.official_parameters(rows)
    raw_sensor=m.flat_reference is not None or official
    mapped=all(r.get('skeleton_targets') for r in rows) and not raw_sensor
    rawp=t([r['skeleton_targets']['positions_local'] for r in rows]) if mapped else t([[r['fingers'][f]['position'] for f in FINGERS] for r in rows])
    rawr=t([np.array(r['skeleton_targets']['rotations_local']).reshape(5,3,3) for r in rows]) if mapped else t([Rotation.from_quat([r['fingers'][f]['quaternion_xyzw'] for f in FINGERS]).as_matrix() for r in rows])
    baseR=t(m.raw_rotation);baseT=t(m.raw_translation)
    q=torch.nn.Parameter(poses.detach().clone(),requires_grad=not official)
    if not official:
        m.constrain_pose_(q);constrain_capture_pose_(m,q,rows)
    prior=q.detach().clone()
    mp=torch.nn.Parameter(t(m.sensor_offsets() if raw_sensor else m.mount_pos));mount_prior=mp.detach().clone();mr=torch.nn.Parameter(t(Rotation.from_matrix(m.mount_rot).as_rotvec()))
    da=torch.nn.Parameter(torch.zeros(3,device=device));dt=torch.nn.Parameter(torch.zeros(3,device=device))
    root_aa=torch.nn.Parameter(t(Rotation.from_matrix(roots.detach().cpu().numpy()).as_rotvec()))
    opt=torch.optim.Adam([{'params':[q,mr,da,root_aa],'lr':.015},{'params':[mp,dt],'lr':.0003}]) if mounting else torch.optim.Adam([q,root_aa],lr=.015)
    if official:
        root_aa.requires_grad_(False)
        if not mounting:return q.detach()
        opt=torch.optim.Adam([{'params':[mr,da],'lr':.015},{'params':[mp,dt],'lr':.0003}])
    if mapped:
        da.requires_grad_(False);dt.requires_grad_(False);baseR=t(np.eye(3));baseT=t(np.zeros(3))
    if not mounting:
        for parameter in (mp,mr,da,dt):parameter.requires_grad_(False)
    contact_indices=[]
    if m.dimension_parameters is not None and not official:
        confirmed=set(m.mapping.get('confirmed',[]))
        for i,row in enumerate(rows):
            stage=row.get('stage','');pp={p['id']:p for p in row.get('contact',{}).get('patches',[])}
            if stage.startswith('pinch_') and row.get('contact',{}).get('ready') and all(p in confirmed and pp.get(p,{}).get('contact') for p in ('thumb_distal',stage[6:]+'_distal')):
                contact_indices.append((i,FINGERS.index(stage[6:])))
    pad_offsets=t(m.distal_pad_landmarks()[0]) if contact_indices else None
    together_indices=[i for i,row in enumerate(rows) if row['stage']=='together'] if not official else []
    together_regions=m.together_regions() if together_indices else []
    for step in range(steps):
        if cancel.is_set():raise InterruptedError('手型拟合已取消，原配置保留')
        opt.zero_grad();v,j,r,ordered=m.decode(q,rotmat(root_aa) if mapped else roots,base_offsets,joint_centres);distal=r[:,[c[-1] for c in CHAINS]]
        offsets=mp if raw_sensor else t(m.tip_anchors-m.rest_j_np[[c[-1] for c in CHAINS]])+mp
        p=j[:,[c[-1] for c in CHAINS]]+(distal@offsets[None,:,:,None]).squeeze(-1) if raw_sensor else v[:,list(TIPS)]+(distal@mp[None,:,:,None]).squeeze(-1)
        trans=rotmat(da)@baseR;targets=rawp@trans.T+baseT+dt
        loss=torch.nn.functional.smooth_l1_loss((p-targets)/.003,torch.zeros_like(p))
        loss=loss+(1/.15**2 if raw_sensor else .04)*((distal@rotmat(mr)[None]-trans@rawr)**2).mean()+.005*((q-prior)**2).mean()+.02*((mp-mount_prior if raw_sensor else mp)/.012).square().mean()+.02*(dt/.01).square().mean()+.01*da.square().mean()
        if contact_indices:
            pads=j[:,[c[-1] for c in CHAINS]]+(distal@pad_offsets[None,:,:,None]).squeeze(-1)
            gaps=torch.stack([pads[i,0]-pads[i,f] for i,f in contact_indices])
            loss=loss+2*torch.nn.functional.smooth_l1_loss(gaps/.004,torch.zeros_like(gaps))
        if together_indices:
            side_gaps=torch.stack([torch.cdist(v[together_indices][:,region['ids_a']],v[together_indices][:,region['ids_b']]).amin(dim=(-2,-1)) for region in together_regions],dim=1)
            loss=loss+2*torch.nn.functional.smooth_l1_loss((side_gaps-.0015)/.003,torch.zeros_like(side_gaps))
            directions=j[together_indices][:,[CHAINS[f][1] for f in range(1,5)]]-j[together_indices][:,[CHAINS[f][0] for f in range(1,5)]]
            directions=torch.nn.functional.normalize(directions,dim=-1);common=torch.nn.functional.normalize(directions.mean(1,keepdim=True),dim=-1)
            loss=loss+((directions-common)/.087).square().mean()
        if not torch.isfinite(loss):raise ValueError('姿态拟合出现非有限值')
        loss.backward();opt.step()
        if not official:m.constrain_pose_(q);constrain_capture_pose_(m,q,rows)
        with torch.no_grad():
            if raw_sensor:m.project_sensor_joint_mount_(mp)
            else:mp.clamp_(-.025,.025)
            dt.clamp_(-.015,.015);da.clamp_(-.3,.3)
    if mounting:
        if raw_sensor:m.sensor_joint_position=mp.detach().cpu().numpy();m.mount_pos=m.sensor_joint_position-(m.tip_anchors-m.rest_j_np[[c[-1] for c in CHAINS]])
        else:m.mount_pos=mp.detach().cpu().numpy()
        m.mount_rot=Rotation.from_rotvec(mr.detach().cpu().numpy()).as_matrix()
        m.raw_rotation=(rotmat(da)@baseR).detach().cpu().numpy();m.raw_translation=(baseT+dt).detach().cpu().numpy()
    if mapped:
        with torch.no_grad():roots.copy_(rotmat(root_aa))
    return q.detach()


def position_rmse(m,rows,poses,roots):
    base_offsets=None;joint_centres=None
    if m.official_fk is not None:poses,roots,base_offsets,joint_centres=m.official_parameters(rows)
    with torch.no_grad():v,j,r,o=m.decode(poses,roots,base_offsets,joint_centres)
    dr=r[:,[c[-1] for c in CHAINS]].cpu().numpy();p=(j[:,[c[-1] for c in CHAINS]].cpu().numpy()+(dr@m.sensor_offsets()[None,:,:,None]).squeeze(-1)) if m.flat_reference is not None else v[:,list(TIPS)].cpu().numpy()+(dr@np.asarray(m.mount_pos)[None,:,:,None]).squeeze(-1)
    target=np.array([row['skeleton_targets']['positions_local'] for row in rows]) if m.flat_reference is None and all(row.get('skeleton_targets') for row in rows) else np.array([[row['fingers'][f]['position'] for f in FINGERS] for row in rows])@m.raw_rotation.T+m.raw_translation
    return float(np.sqrt(np.mean(np.sum((p-target)**2,axis=-1))))


def candidate_validation(baseline_error,error,geometry,tolerance):
    """Primary holdout EMF error, with explicit geometry guards; not training loss."""
    required={'pinch_'+f for f in FINGERS[1:]}
    gap=geometry.get('contact_gap_m',float('inf'));baseline_gap=geometry.get('baseline_contact_gap_m',0.)
    contact_valid=required<=set(geometry.get('contact_stages',[])) and np.isfinite(gap) and (gap<=.01 or gap<=baseline_gap*1.05+.001)
    # Penetration is a soft objective, never an absolute rejection threshold.
    accepted=bool(np.isfinite([baseline_error,error,geometry['noncontact_penetration_m']]).all() and error<baseline_error*.999 and contact_valid)
    return accepted,contact_valid


def together_validation(geometry,baseline):
    pairs=geometry.get('together_pair_gaps_mm',{})
    if set(pairs)!={'index_middle','middle_ring','ring_pinky'} or not geometry.get('together_observations'):return False
    return all(np.isfinite(gap) and (gap<=4. or gap<=baseline.get('together_pair_gaps_mm',{}).get(pair,0)*1.05+.5) for pair,gap in pairs.items())


def geometry_score(position,geometry,tolerance):
    pairs=list(geometry.get('together_pair_gaps_mm',{}).values())
    together_cost=float(np.mean((np.array(pairs)/3.)**2)) if pairs else (geometry.get('together_gap_m',0.)/.003)**2
    return (position/.003)**2+(geometry['contact_gap_m']/.003)**2+4*together_cost+3*(max(geometry['noncontact_penetration_m']-tolerance,0)/.003)**2


def fit_capture(model_path,profile,mapping,rows,progress,cancel,max_seconds=300):
    started=time.monotonic();required={'thumb_distal',*[f+'_distal' for f in FINGERS[1:]]}
    if not required<=set(mapping.get('confirmed',[])):raise ValueError('请先核对拇指与其余四指末端分区，再进行尺寸拟合')
    if any(r.get('mapping',{}).get('version')!=mapping['version'] for r in rows):
        # Confirmation changes version only; assignments must remain consistent.
        if any(r.get('mapping',{}).get('assignments')!=mapping['assignments'] for r in rows):raise ValueError('采集期间 SID 分区改变，请重新采集')
    train,valid=choose_rows(rows);allrows=train+valid
    m=ManoSurface(model_path)
    official=profile.get('tracking_source')=='official_qnbot_skeleton_fk_v1'
    if official:
        if not profile.get('flat_reference'):raise ValueError('尺寸拟合前请建立平摊参考；实时官方 FK 显示不需要额外参考')
        m.enable_official_fk()
    m.material=copy.deepcopy(profile['material']);m.binding_overrides=copy.deepcopy(profile.get('binding_overrides',{}));m.build_binding(mapping)
    bone_samples=[skeleton_targets(r.get('skeleton')) for r in train]
    bone_samples=[p for p in bone_samples if p is not None]
    if not bone_samples:raise ValueError('没有可用的骨架长度先验')
    m.bone_reference=np.median([[[np.linalg.norm(p[2+4*f+k]-p[1+4*f+k]) for k in range(3)] for f in range(5)] for p in bone_samples],axis=0)
    fixed_dimensions=profile.get('dimension_mode')==DIMENSION_MODE
    m.proximal_lateral_scales=np.array(profile.get('proximal_lateral_scales',[0.]*4),dtype=float)
    if fixed_dimensions:m.dimension_parameters=np.array(profile.get('dimension_parameters',np.zeros(DIMENSION_COUNT)),dtype=float)
    tolerance=float(profile.get('penetration_tolerance_m',PENETRATION_TOLERANCE_M))
    if not 0<=tolerance<=.01:raise ValueError('穿透容差必须在 0–10 mm')
    m.set_shape(profile['shape']);m.raw_rotation=np.array(profile['raw_rotation']);m.raw_translation=np.array(profile['raw_translation']);m.raw_aligned=True
    m.mount_pos=np.array(profile['mount_pos']);m.mount_rot=np.array(profile['mount_rot']);m.flat_reference=copy.deepcopy(profile.get('flat_reference'));m.sensor_joint_position=np.array(profile['sensor_joint_position']) if profile.get('sensor_joint_position') is not None else None
    m.mounting_estimated=bool(profile.get('mounting_estimated') or profile.get('report',{}).get('validation_pose_source')=='runtime_emf_only')
    dimensions_before=m.dimension_summary()
    sequence=holdout_rows(rows)
    progress({'state':'running','progress':.01,'message':'基线：连续留出动作使用实际官方骨架 FK 驱动；无第二次 IK' if official else '基线：连续留出动作使用实际 EMF 求解器；无触觉贴合或动作标签约束'})
    qbaseline,rbaseline,baseline_trajectory=validation_retarget(m,valid,sequence,bool(profile.get('mounting_calibrated')))
    baseline_pos=position_rmse(m,valid,qbaseline,rbaseline);basegeo=geometry_report(m,valid,qbaseline,rbaseline)
    baseline_score=geometry_score(baseline_pos,basegeo,tolerance)
    if m.flat_reference is not None:
        for f in FINGERS:
            rotations=Rotation.from_quat([r['fingers'][f]['quaternion_xyzw'] for r in rows])
            if np.max((rotations[0].inv()*rotations).magnitude())<np.deg2rad(15):raise ValueError(f'{f} 的 EMF 方向变化不足，不能辨识共享安装偏移，请补充多动作采集')
    q=[];roots=[]
    for row_index,row in enumerate(allrows):
        if official:
            pose,root,_,_,_=m.official_fk.parameters(m,row,CHAINS);q.append(pose);roots.append(root);continue
        target=skeleton_targets(row.get('skeleton'))
        if target is None:raise ValueError('采集缺少有效骨架初始化，请重试')
        pose,root=m.pose_from_bones(target)
        if m.flat_reference is not None:
            root=np.eye(3,dtype=np.float32)
            raw_positions=np.array([row['fingers'][f]['position'] for f in FINGERS])@m.raw_rotation.T+m.raw_translation
            raw_rotations=m.raw_rotation@Rotation.from_quat([row['fingers'][f]['quaternion_xyzw'] for f in FINGERS]).as_matrix()
            start=np.array(m.flat_reference.get('flat_seed',pose),dtype=np.float32)
            pose=m.fit_fingertip_positions(start,root,raw_positions,orientations=raw_rotations)
            if fixed_dimensions and row_index<len(train):pose=m.fit_sensor_supported_pinch(pose,root,raw_positions,raw_rotations,row)
        elif row.get('skeleton_targets'):
            observed=np.array(row['skeleton_targets']['positions_local'])
            with torch.no_grad():_,_,_,initial=m.decode(torch.tensor(pose[None],device=m.device),torch.eye(3,device=m.device)[None])
            local=initial[0,[4,8,12,16,20]].cpu().numpy();u,_,vt=np.linalg.svd(local.T@observed)
            root=(vt.T@np.diag([1.,1.,np.linalg.det(vt.T@u.T)])@u.T).astype(np.float32)
            pose=m.fit_thumb_directions(pose,root,target,tip_target=observed[0])
        q.append(pose);roots.append(root)
    q=torch.tensor(np.array(q),device=m.device,dtype=torch.float32);roots=torch.tensor(np.array(roots),device=m.device,dtype=torch.float32)
    n=len(train)
    progress({'state':'running','progress':.03,'message':'拟合原始定位与共享传感器安装偏移'})
    qtrain=adam_articulation(m,train,q[:n],roots[:n],100,cancel)
    original_shape=m.shape.copy();best_shape=original_shape.copy();original_dimensions=m.dimension_parameters.copy() if fixed_dimensions else None;best_dimensions=original_dimensions.copy() if fixed_dimensions else None;bestscore=float('inf');evaluations=0
    active=[k for f in range(5) for k in (3*f,3*f+1,3*f+2,15+f)] if fixed_dimensions else list(range(11,21))+[21+PATCH_IDS.index(p) for p in mapping.get('confirmed',[])]
    # Bone data are a strong but imperfect length prior. Surface fingertip
    # extension is distinct from inter-joint length. Betas/global scale freeze.
    bounds=[(-2.,2.)]*10+[(math.log(.8),math.log(1.2))]+[(math.log(.97),math.log(1.03))]*5+[(-.002,.002)]*5
    bounds += [(-.002,.002)]*20
    if fixed_dimensions:bounds=[(math.log(.8),math.log(1.2))]*15+[(math.log(.75),math.log(1.3))]*5
    # Geometry evaluation includes each guide type, not just a convenient pinch.
    subset=[]
    for stage in dict.fromkeys(r['stage'] for r in train):
        indices=[i for i,r in enumerate(train) if r['stage']==stage]
        if stage.startswith('pinch_'):
            subset.extend(i for i in indices if train[i].get('load_role')=='onset')
            near=[i for i in indices if train[i].get('load_role')=='approach']
            if near:subset.append(near[0])
        elif stage in ('open','fist','together'):subset.append(indices[len(indices)//2])
    visited_dimensions=set();previous_x=None
    def evaluate(x):
        nonlocal evaluations,bestscore,best_shape,best_dimensions,previous_x
        if cancel.is_set():raise InterruptedError('手型拟合已取消')
        if time.monotonic()-started>max_seconds:raise TimeoutError('尺寸搜索达到时间预算，保留最佳有效候选')
        if previous_x is not None:visited_dimensions.update(active[i] for i in np.flatnonzero(abs(np.asarray(x)-previous_x)>1e-5))
        previous_x=np.asarray(x).copy()
        shape=original_shape.copy()
        if fixed_dimensions:
            dimensions=original_dimensions.copy();dimensions[active]=x;m.set_dimensions(dimensions)
        else:shape[active]=x;m.set_shape(shape)
        pos=position_rmse(m,train,qtrain,roots[:n]);geo=geometry_report(m,[train[i] for i in subset],qtrain[subset],roots[:n][subset])
        prior=.1*np.mean((m.dimension_parameters[:15]/.07)**2)+.1*np.mean((m.dimension_parameters[15:]/.12)**2) if fixed_dimensions else .1*np.mean((shape[11:16]/.01)**2)+.1*np.mean((shape[16:21]/.001)**2)+.2*np.mean((shape[21:]/.001)**2)
        score=geometry_score(pos,geo,tolerance)+(0. if fixed_dimensions else .2*geo['relative_response_cost'])+prior
        evaluations+=1
        if math.isfinite(score) and score<bestscore:
            bestscore=score;best_shape=shape.copy()
            if fixed_dimensions:best_dimensions=m.dimension_parameters.copy()
        progress({'state':'running','progress':min(.85,.15+evaluations/(480 if fixed_dimensions else 160)*.7),'evaluations':evaluations,'message':'Powell 尺寸搜索；每个候选重建碰撞体并复核三角面'})
        return score
    # A Powell sweep over 20 variables needs more than 80 evaluations.
    # Budget full sweeps so late variables (finger radii) are actually visited.
    for outer in range(2):
        bestscore=float('inf')  # Scores with a newly fitted pose are not comparable to the previous round.
        try:
            minimize(evaluate,m.dimension_parameters[active] if fixed_dimensions else m.shape[active],method='Powell',bounds=[bounds[k] for k in active],options={'maxfev':240 if fixed_dimensions else 80,'maxiter':2,'xtol':.001,'ftol':.001})
        except TimeoutError:break
        if fixed_dimensions:m.set_dimensions(best_dimensions)
        else:m.set_shape(best_shape)
        qtrain=adam_articulation(m,train,qtrain,roots[:n],25,cancel)
    if fixed_dimensions:m.set_dimensions(best_dimensions)
    else:m.set_shape(best_shape)
    proximal_fit=refine_proximal_widths(m,train,cancel,max_nfev=15) if fixed_dimensions and official else {'state':'unobserved','applied':False}
    progress({'state':'running','progress':.9,'message':'留出第二轮接触与重复张手验证'})
    qvalid,rvalid,after_trajectory=validation_retarget(m,valid,sequence,True)
    after_pos=position_rmse(m,valid,qvalid,rvalid);aftergeo=geometry_report(m,valid,qvalid,rvalid)
    score=geometry_score(after_pos,aftergeo,tolerance)
    aftergeo['baseline_contact_gap_m']=basegeo['contact_gap_m']
    accepted,contact_valid=candidate_validation(baseline_pos,after_pos,aftergeo,tolerance)
    baseline_approach=[r['rmse_mm'] for r in baseline_trajectory if r['stage'].startswith('pinch_') and not r['bilateral_contact']]
    after_approach=[r['rmse_mm'] for r in after_trajectory if r['stage'].startswith('pinch_') and not r['bilateral_contact']]
    before_approach_rmse=float(np.sqrt(np.mean(np.square(baseline_approach)))) if baseline_approach else None
    after_approach_rmse=float(np.sqrt(np.mean(np.square(after_approach)))) if after_approach else None
    approach_valid=bool(before_approach_rmse is not None and after_approach_rmse<=before_approach_rmse*1.05+.5)
    together_valid=together_validation(aftergeo,basegeo)
    accepted=accepted and approach_valid and together_valid
    # Relative gain is fit only over a single pair's repeated observations. Its
    # arbitrary scale must not be interpreted as force calibration.
    gains={}
    for f in FINGERS[1:]:
        predicted=[];response=[]
        for i,row in enumerate(valid):
            if row['stage']!='pinch_'+f:continue
            for pair in aftergeo['fusion'][i]['pairs']:
                if {pair['patch_a'],pair['patch_b']}=={'thumb_distal',f+'_distal'}:
                    predicted.append(pair['relative_model_load'])
                    pp={p['id']:p for p in row['contact']['patches']};response.append((pp['thumb_distal']['response']+pp[f+'_distal']['response'])/2)
        gains[f]=positive_gain(predicted,response)
    result=copy.deepcopy(profile);result.update(calibrated=False,applied_at_ms=None,revision=None,dimension_mode=DIMENSION_MODE if fixed_dimensions else 'legacy_patch_offsets',dimension_parameters=best_dimensions.tolist() if fixed_dimensions else None,penetration_tolerance_m=tolerance,sensor_joint_position=m.sensor_offsets().tolist() if m.flat_reference is not None else None,mounting_calibrated=bool(accepted and m.flat_reference is not None),geometry_revision=GEOMETRY_REVISION,observation_space='raw_emf_sensor' if m.flat_reference is not None else 'sdk_wrist_targets' if all(r.get('skeleton_targets') for r in rows) else 'legacy_rigid_fit',calibration_fingerprint=rows[0].get('calibration_fingerprint'),shape=best_shape.tolist(),bone_reference=m.bone_reference.tolist(),raw_rotation=m.raw_rotation.tolist(),raw_translation=m.raw_translation.tolist(),mount_pos=m.mount_pos.tolist(),mount_rot=m.mount_rot.tolist(),response_gains=gains,mapping=mapping)
    result['proximal_lateral_scales']=m.proximal_lateral_scales.tolist()
    report={'state':'preview','accepted':accepted,'baseline_position_rmse_mm':baseline_pos*1000,'position_rmse_mm':after_pos*1000,
            'baseline_contact_gap_mm':basegeo['contact_gap_m']*1000,'contact_gap_mm':aftergeo['contact_gap_m']*1000,
            'noncontact_penetration_mm':aftergeo['noncontact_penetration_m']*1000,'baseline_score':baseline_score,'score':score,'evaluations':evaluations,
            'baseline_together_pair_gaps_mm':basegeo['together_pair_gaps_mm'],'evaluated_dimension_indices':sorted(visited_dimensions),'unevaluated_dimension_indices':sorted(set(active)-visited_dimensions),'baseline_together_gap_mm':basegeo['together_gap_m']*1000,'together_gap_mm':aftergeo['together_gap_m']*1000,'together_pair_gaps_mm':aftergeo['together_pair_gaps_mm'],'together_validation_passed':together_valid,
            'seconds':time.monotonic()-started,'unobserved_patches':[p for p in PATCH_IDS if p not in mapping.get('confirmed',[])],
            'contact_gap_definition':'nearest_patch_distance_with_opposing_normal_filter',
            'dimension_parameter_layout':{'segment_axial_scales':15,'finger_radial_scales':5,'proximal_lateral_scales':3,'pinky_local_shape_frozen':True,'patch_displacements':0},'proximal_width_fit':proximal_fit,
            'contact_observations':aftergeo['contact_observations'],'contact_stages':aftergeo['contact_stages'],'contact_validation_passed':contact_valid,
            'distance_query_radius_m':.4,'fit_revision':FIT_REVISION,'acceptance_metric':'held_out_emf_rmse_with_pairwise_geometry_guards','geometry_score_degraded':bool(score>baseline_score),'validation_pose_source':'official_skeleton_fk' if official else 'runtime_emf_only','tactile_pose_feedback':False,
            'source_capture_hash':hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'selected_training_frames':len(train),'selected_validation_frames':len(valid),'continuous_validation_frames':len(sequence),
            'approach_validation_passed':approach_valid,'baseline_approach_sensor_rmse_mm':before_approach_rmse,'approach_sensor_rmse_mm':after_approach_rmse,
            'trajectory_before':trajectory_summary(baseline_trajectory),'trajectory_after':trajectory_summary(after_trajectory),
            'dimensions_before_mm':dimensions_before,'dimensions_after_mm':m.dimension_summary(),
            'thickness_identification':'contact_and_prior_only; flat_EMF_observations_do_not_measure_thickness',
            'known_training_pose_constraints':['official IK poses fixed; guides score geometry only'] if official else ['open: straight ±5°','together: straight, parallel axes, adjacent side surfaces at 3 common levels','mcp90: 75–100° MCP / straight PIP,DIP','hook: MCP −5–8°'],
            'penetration_tolerance_mm':tolerance*1000,'penetration_policy':'soft_penalty_only','reason':'独立留出定位改善，逐对几何与接近轨迹检查通过；穿透仅作软惩罚' if accepted else '未通过传感器独立留出检查，当前尺寸保留',
            'sensor_correspondence':'distal_bone_SE3_times_fixed_EMF_mount' if m.flat_reference is not None else 'legacy_proxy','sensor_position_calibrated':bool(accepted and m.flat_reference is not None),'distal_to_emf_position_m':m.sensor_offsets().tolist(),'force_calibrated':False,'surface':'gloved_effective','geometry_version':m.shape_version,
            'optimizer':'Adam shared mounting/registration + Powell dimensions; official poses frozen' if official else 'Adam articulation/mounting + Powell fixed segment/radial scales' if fixed_dimensions else 'legacy Adam/Powell','geometry_fixed_during_tracking':True,'pose_correctives':not fixed_dimensions,'segment_scales':np.exp(best_dimensions[:15].reshape(5,3)).tolist() if fixed_dimensions else None,'radial_scales':np.exp(best_dimensions[15:]).tolist() if fixed_dimensions else None,
            'optimized_thickness_patches':[] if fixed_dimensions else [PATCH_IDS[k-21] for k in active if k>=21],
            'bone_reference_role':'strong_soft_prior','longitudinal_scale':np.exp(best_shape[11:16]).tolist(),
            'fingertip_extension_mm':(best_shape[16:21]*1000).tolist(),
            'bone_reference_m':m.bone_reference.tolist() if m.bone_reference is not None else None}
    if not accepted:
        failures=[]
        if after_pos>=baseline_pos*.999:failures.append('留出定位误差未改善')
        if not contact_valid:failures.append('四组有效对指／接触间隙检查未通过')
        if not approach_valid:failures.append('非双侧接触的接近／释放轨迹未通过')
        if not together_valid:failures.append('并拢平摊的相邻侧面贴合未改善')
        report['rejection_reasons']=failures;report['reason']='；'.join(failures)+'；当前尺寸保留'
    report['quality_warnings']=([f"对指法向筛选间隙仍为 {aftergeo['contact_gap_m']*1000:.1f} mm"] if aftergeo['contact_gap_m']>.01 else [])+[f"并拢 {pair} 侧面间隙仍为 {gap:.1f} mm" for pair,gap in aftergeo['together_pair_gaps_mm'].items() if gap>4.]+([f"搜索预算未覆盖尺寸变量 {sorted(set(active)-visited_dimensions)}"] if set(active)-visited_dimensions else [])
    if score>baseline_score:report['quality_warnings'].append('综合几何评分升高；已按留出定位误差及逐对几何守则分别验收，请核对曲面')
    return result,report
