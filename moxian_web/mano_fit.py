"""Bounded, held-out MANO calibration. Runs on a private model in a worker thread."""
from __future__ import annotations
import copy,math,time
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


def choose_rows(rows):
    train=[];valid=[]
    for stage in dict.fromkeys(r['stage'] for r in rows):
        group=[r for r in rows if r['stage']==stage]
        if stage=='open_repeat':valid.append(group[len(group)//2]);continue
        if stage.startswith('pinch_'):
            first=[r for r in group if r.get('cycle',0)==0];second=[r for r in group if r.get('cycle',0)>=1]
            # An onset is the first confirmed low-load contact, not the hard squeeze.
            def low_contact(values):
                both=[]
                for r in values:
                    p={p['id']:p for p in r['contact']['patches']};a=p['thumb_distal'];b=p[stage[6:]+'_distal']
                    if a.get('contact') and b.get('contact'):both.append((max(a['response'],b['response']),r))
                return min(both,key=lambda z:z[0])[1] if both else values[len(values)//2]
            if not first or not second:raise ValueError('每组对指需要两次完整接触与释放')
            for target,group in ((train,first),(valid,second)):
                onset=copy.deepcopy(low_contact(group));onset['load_role']='onset';target.append(onset)
                def strength(row):
                    pp={p['id']:p for p in row['contact']['patches']}
                    return pp['thumb_distal']['response']+pp[stage[6:]+'_distal']['response']
                peak=copy.deepcopy(max(group,key=strength));peak['load_role']='peak';target.append(peak)
        else:
            train.append(group[len(group)//3]);valid.append(group[2*len(group)//3])
    return train,valid


def geometry_report(m,rows,poses,roots):
    g=CollisionSurface(m);depth=0.;contact=[];records=[]
    with torch.no_grad():v,j,rr,ordered=m.decode(poses,roots)
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
                if not pair_touch:worst=max(worst,float(max(0,-ds.min())))
        depth=max(depth,worst)
        if row['stage'].startswith('pinch_') and row.get('load_role')!='peak':
            a=0;b=PATCH_IDS.index(row['stage'][6:]+'_distal');ia=np.concatenate((np.flatnonzero(assignments==a),460+np.flatnonzero(m.vertex_patches==a)));ib=np.concatenate((np.flatnonzero(assignments==b),460+np.flatnonzero(m.vertex_patches==b)))
            ds=np.concatenate((geom['gap'][ia,b][geom['facing'][ia,b]<-.15],geom['gap'][ib,a][geom['facing'][ib,a]<-.15]))
            contact.append(float(np.min(np.abs(ds))) if len(ds) else .04)
        fusion=fuse_contacts(m,row,geom,row['stage']);records.append(fusion)
    return {'noncontact_penetration_m':depth,'contact_gap_m':float(np.mean(contact)) if contact else 0.,'fusion':records,'relative_response_cost':relative_response_cost(rows,records,m.mapping)}



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
    raw_sensor=m.flat_reference is not None
    mapped=all(r.get('skeleton_targets') for r in rows) and not raw_sensor
    rawp=t([r['skeleton_targets']['positions_local'] for r in rows]) if mapped else t([[r['fingers'][f]['position'] for f in FINGERS] for r in rows])
    rawr=t([np.array(r['skeleton_targets']['rotations_local']).reshape(5,3,3) for r in rows]) if mapped else t([Rotation.from_quat([r['fingers'][f]['quaternion_xyzw'] for f in FINGERS]).as_matrix() for r in rows])
    baseR=t(m.raw_rotation);baseT=t(m.raw_translation)
    q=torch.nn.Parameter(poses.detach().clone());m.constrain_pose_(q);prior=q.detach().clone()
    mp=torch.nn.Parameter(t(m.sensor_offsets() if raw_sensor else m.mount_pos));mount_prior=mp.detach().clone();mr=torch.nn.Parameter(t(Rotation.from_matrix(m.mount_rot).as_rotvec()))
    da=torch.nn.Parameter(torch.zeros(3,device=device));dt=torch.nn.Parameter(torch.zeros(3,device=device))
    root_aa=torch.nn.Parameter(t(Rotation.from_matrix(roots.detach().cpu().numpy()).as_rotvec()))
    opt=torch.optim.Adam([{'params':[q,mr,da,root_aa],'lr':.015},{'params':[mp,dt],'lr':.0003}]) if mounting else torch.optim.Adam([q,root_aa],lr=.015)
    if mapped:
        da.requires_grad_(False);dt.requires_grad_(False);baseR=t(np.eye(3));baseT=t(np.zeros(3))
    if not mounting:
        for parameter in (mp,mr,da,dt):parameter.requires_grad_(False)
    for step in range(steps):
        if cancel.is_set():raise InterruptedError('手型拟合已取消，原配置保留')
        opt.zero_grad();v,j,r,ordered=m.decode(q,rotmat(root_aa) if mapped else roots);distal=r[:,[c[-1] for c in CHAINS]]
        offsets=mp if raw_sensor else t(m.tip_anchors-m.rest_j_np[[c[-1] for c in CHAINS]])+mp
        p=j[:,[c[-1] for c in CHAINS]]+(distal@offsets[None,:,:,None]).squeeze(-1) if raw_sensor else v[:,list(TIPS)]+(distal@mp[None,:,:,None]).squeeze(-1)
        trans=rotmat(da)@baseR;targets=rawp@trans.T+baseT+dt
        loss=torch.nn.functional.smooth_l1_loss((p-targets)/.003,torch.zeros_like(p))
        loss=loss+(1/.15**2 if raw_sensor else .04)*((distal@rotmat(mr)[None]-trans@rawr)**2).mean()+.005*((q-prior)**2).mean()+.02*((mp-mount_prior if raw_sensor else mp)/.012).square().mean()+.02*(dt/.01).square().mean()+.01*da.square().mean()
        if not torch.isfinite(loss):raise ValueError('姿态拟合出现非有限值')
        loss.backward();opt.step();m.constrain_pose_(q)
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
    with torch.no_grad():v,j,r,o=m.decode(poses,roots)
    dr=r[:,[c[-1] for c in CHAINS]].cpu().numpy();p=(j[:,[c[-1] for c in CHAINS]].cpu().numpy()+(dr@m.sensor_offsets()[None,:,:,None]).squeeze(-1)) if m.flat_reference is not None else v[:,list(TIPS)].cpu().numpy()+(dr@np.asarray(m.mount_pos)[None,:,:,None]).squeeze(-1)
    target=np.array([row['skeleton_targets']['positions_local'] for row in rows]) if m.flat_reference is None and all(row.get('skeleton_targets') for row in rows) else np.array([[row['fingers'][f]['position'] for f in FINGERS] for row in rows])@m.raw_rotation.T+m.raw_translation
    return float(np.sqrt(np.mean(np.sum((p-target)**2,axis=-1))))


def fit_capture(model_path,profile,mapping,rows,progress,cancel,max_seconds=180):
    started=time.monotonic();required={'thumb_distal',*[f+'_distal' for f in FINGERS[1:]]}
    if not required<=set(mapping.get('confirmed',[])):raise ValueError('请先核对拇指与其余四指末端分区，再进行尺寸拟合')
    if any(r.get('mapping',{}).get('version')!=mapping['version'] for r in rows):
        # Confirmation changes version only; assignments must remain consistent.
        if any(r.get('mapping',{}).get('assignments')!=mapping['assignments'] for r in rows):raise ValueError('采集期间 SID 分区改变，请重新采集')
    train,valid=choose_rows(rows);allrows=train+valid
    m=ManoSurface(model_path);m.material=copy.deepcopy(profile['material']);m.binding_overrides=copy.deepcopy(profile.get('binding_overrides',{}));m.build_binding(mapping)
    bone_samples=[skeleton_targets(r.get('skeleton')) for r in rows]
    bone_samples=[p for p in bone_samples if p is not None]
    if not bone_samples:raise ValueError('没有可用的骨架长度先验')
    m.bone_reference=np.median([[[np.linalg.norm(p[2+4*f+k]-p[1+4*f+k]) for k in range(3)] for f in range(5)] for p in bone_samples],axis=0)
    fixed_dimensions=profile.get('dimension_mode')==DIMENSION_MODE
    if fixed_dimensions:m.dimension_parameters=np.array(profile.get('dimension_parameters',np.zeros(DIMENSION_COUNT)),dtype=float)
    tolerance=float(profile.get('penetration_tolerance_m',PENETRATION_TOLERANCE_M))
    if not 0<=tolerance<=.01:raise ValueError('穿透容差必须在 0–10 mm')
    m.set_shape(profile['shape']);m.raw_rotation=np.array(profile['raw_rotation']);m.raw_translation=np.array(profile['raw_translation']);m.raw_aligned=True
    m.mount_pos=np.array(profile['mount_pos']);m.mount_rot=np.array(profile['mount_rot']);m.flat_reference=copy.deepcopy(profile.get('flat_reference'));m.sensor_joint_position=np.array(profile['sensor_joint_position']) if profile.get('sensor_joint_position') is not None else None
    if m.flat_reference is not None:
        for f in FINGERS:
            rotations=Rotation.from_quat([r['fingers'][f]['quaternion_xyzw'] for r in rows])
            if np.max((rotations[0].inv()*rotations).magnitude())<np.deg2rad(15):raise ValueError(f'{f} 的 EMF 方向变化不足，不能辨识共享安装偏移，请补充多动作采集')
    q=[];roots=[]
    for row in allrows:
        target=skeleton_targets(row.get('skeleton'))
        if target is None:raise ValueError('采集缺少有效骨架初始化，请重试')
        pose,root=m.pose_from_bones(target)
        if m.flat_reference is not None:
            root=np.eye(3,dtype=np.float32)
            raw_positions=np.array([row['fingers'][f]['position'] for f in FINGERS])@m.raw_rotation.T+m.raw_translation
            raw_rotations=m.raw_rotation@Rotation.from_quat([row['fingers'][f]['quaternion_xyzw'] for f in FINGERS]).as_matrix()
            start=np.array(m.flat_reference.get('flat_seed',pose),dtype=np.float32)
            pose=m.fit_fingertip_positions(start,root,raw_positions,orientations=raw_rotations)
        elif row.get('skeleton_targets'):
            observed=np.array(row['skeleton_targets']['positions_local'])
            with torch.no_grad():_,_,_,initial=m.decode(torch.tensor(pose[None],device=m.device),torch.eye(3,device=m.device)[None])
            local=initial[0,[4,8,12,16,20]].cpu().numpy();u,_,vt=np.linalg.svd(local.T@observed)
            root=(vt.T@np.diag([1.,1.,np.linalg.det(vt.T@u.T)])@u.T).astype(np.float32)
            pose=m.fit_thumb_directions(pose,root,target,tip_target=observed[0])
        q.append(pose);roots.append(root)
    q=torch.tensor(np.array(q),device=m.device,dtype=torch.float32);roots=torch.tensor(np.array(roots),device=m.device,dtype=torch.float32)
    n=len(train);validation_roots=roots[n:].clone()
    def adapt_valid():
        with torch.no_grad():roots[n:].copy_(validation_roots)
        return adam_articulation(m,valid,q[n:],roots[n:],40 if fixed_dimensions else 12,cancel,mounting=False)
    # Baseline and candidate both receive identical pose optimization budget.
    progress({'state':'running','progress':.03,'message':'拟合原始定位与共享传感器安装偏移'})
    qtrain=adam_articulation(m,train,q[:n],roots[:n],60,cancel);qvalid=adapt_valid()
    baseline_pos=position_rmse(m,valid,qvalid,roots[n:]);basegeo=geometry_report(m,valid,qvalid,roots[n:])
    baseline_score=(baseline_pos/.003)**2+(basegeo['contact_gap_m']/.003)**2+3*(max(basegeo['noncontact_penetration_m']-tolerance,0)/.003)**2
    original_shape=m.shape.copy();best_shape=original_shape.copy();original_dimensions=m.dimension_parameters.copy() if fixed_dimensions else None;best_dimensions=original_dimensions.copy() if fixed_dimensions else None;bestscore=float('inf');evaluations=0
    active=list(range(DIMENSION_COUNT)) if fixed_dimensions else list(range(11,21))+[21+PATCH_IDS.index(p) for p in mapping.get('confirmed',[])]
    # Bone data are a strong but imperfect length prior. Surface fingertip
    # extension is distinct from inter-joint length. Betas/global scale freeze.
    bounds=[(-2.,2.)]*10+[(math.log(.8),math.log(1.2))]+[(math.log(.97),math.log(1.03))]*5+[(-.002,.002)]*5
    bounds += [(-.002,.002)]*20
    if fixed_dimensions:bounds=[(math.log(.8),math.log(1.2))]*15+[(math.log(.75),math.log(1.3))]*5
    # Geometry evaluation includes each guide type, not just a convenient pinch.
    subset=[i for i,r in enumerate(train) if r['stage'].startswith('pinch_') or r['stage'] in ('open','fist','together')]
    def evaluate(x):
        nonlocal evaluations,bestscore,best_shape,best_dimensions
        if cancel.is_set():raise InterruptedError('手型拟合已取消')
        if time.monotonic()-started>max_seconds:raise TimeoutError('尺寸搜索达到时间预算，保留最佳有效候选')
        shape=original_shape.copy()
        if fixed_dimensions:m.set_dimensions(x)
        else:shape[active]=x;m.set_shape(shape)
        pos=position_rmse(m,train,qtrain,roots[:n]);geo=geometry_report(m,[train[i] for i in subset],qtrain[subset],roots[:n][subset])
        prior=.1*np.mean((m.dimension_parameters[:15]/.07)**2)+.1*np.mean((m.dimension_parameters[15:]/.12)**2) if fixed_dimensions else .1*np.mean((shape[11:16]/.01)**2)+.1*np.mean((shape[16:21]/.001)**2)+.2*np.mean((shape[21:]/.001)**2)
        score=(pos/.003)**2+(geo['contact_gap_m']/.003)**2+(0. if fixed_dimensions else .2*geo['relative_response_cost'])+3*(max(geo['noncontact_penetration_m']-tolerance,0)/.003)**2+prior
        evaluations+=1
        if math.isfinite(score) and score<bestscore:
            bestscore=score;best_shape=shape.copy()
            if fixed_dimensions:best_dimensions=m.dimension_parameters.copy()
        progress({'state':'running','progress':min(.85,.15+evaluations/160*.7),'evaluations':evaluations,'message':'Powell 尺寸搜索；每个候选重建碰撞体并复核三角面'})
        return score
    for outer in range(2):
        try:
            minimize(evaluate,m.dimension_parameters if fixed_dimensions else m.shape[active],method='Powell',bounds=[bounds[k] for k in active],options={'maxfev':80,'maxiter':2,'xtol':.001,'ftol':.001})
        except TimeoutError:break
        if fixed_dimensions:m.set_dimensions(best_dimensions)
        else:m.set_shape(best_shape)
        qtrain=adam_articulation(m,train,qtrain,roots[:n],25,cancel)
    if fixed_dimensions:m.set_dimensions(best_dimensions)
    else:m.set_shape(best_shape)
    progress({'state':'running','progress':.9,'message':'留出第二轮接触与重复张手验证'})
    qvalid=adapt_valid();after_pos=position_rmse(m,valid,qvalid,roots[n:]);aftergeo=geometry_report(m,valid,qvalid,roots[n:])
    score=(after_pos/.003)**2+(aftergeo['contact_gap_m']/.003)**2+3*(max(aftergeo['noncontact_penetration_m']-tolerance,0)/.003)**2
    accepted=bool(score<baseline_score*.999 and aftergeo['noncontact_penetration_m']<=tolerance)
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
    result=copy.deepcopy(profile);result.update(dimension_mode=DIMENSION_MODE if fixed_dimensions else 'legacy_patch_offsets',dimension_parameters=best_dimensions.tolist() if fixed_dimensions else None,penetration_tolerance_m=tolerance,sensor_joint_position=m.sensor_offsets().tolist() if m.flat_reference is not None else None,mounting_calibrated=bool(accepted and m.flat_reference is not None),geometry_revision=GEOMETRY_REVISION,observation_space='raw_emf_sensor' if m.flat_reference is not None else 'sdk_wrist_targets' if all(r.get('skeleton_targets') for r in rows) else 'legacy_rigid_fit',calibration_fingerprint=rows[0].get('calibration_fingerprint'),shape=best_shape.tolist(),bone_reference=m.bone_reference.tolist(),raw_rotation=m.raw_rotation.tolist(),raw_translation=m.raw_translation.tolist(),mount_pos=m.mount_pos.tolist(),mount_rot=m.mount_rot.tolist(),response_gains=gains,mapping=mapping)
    report={'state':'preview','accepted':accepted,'baseline_position_rmse_mm':baseline_pos*1000,'position_rmse_mm':after_pos*1000,
            'baseline_contact_gap_mm':basegeo['contact_gap_m']*1000,'contact_gap_mm':aftergeo['contact_gap_m']*1000,
            'noncontact_penetration_mm':aftergeo['noncontact_penetration_m']*1000,'baseline_score':baseline_score,'score':score,'evaluations':evaluations,
            'seconds':time.monotonic()-started,'unobserved_patches':[p for p in PATCH_IDS if p not in mapping.get('confirmed',[])],
            'penetration_tolerance_mm':tolerance*1000,'reason':f'留出误差改善且穿透 ≤{tolerance*1000:g} mm' if accepted else f'未通过留出改进／{tolerance*1000:g} mm 穿透检查，旧模型保留',
            'sensor_correspondence':'distal_bone_SE3_times_fixed_EMF_mount' if m.flat_reference is not None else 'legacy_proxy','sensor_position_calibrated':bool(accepted and m.flat_reference is not None),'distal_to_emf_position_m':m.sensor_offsets().tolist(),'force_calibrated':False,'surface':'gloved_effective','geometry_version':m.shape_version,
            'optimizer':'Adam articulation/mounting + Powell fixed segment/radial scales' if fixed_dimensions else 'legacy Adam/Powell','geometry_fixed_during_tracking':True,'pose_correctives':not fixed_dimensions,'segment_scales':np.exp(best_dimensions[:15].reshape(5,3)).tolist() if fixed_dimensions else None,'radial_scales':np.exp(best_dimensions[15:]).tolist() if fixed_dimensions else None,
            'optimized_thickness_patches':[] if fixed_dimensions else [PATCH_IDS[k-21] for k in active if k>=21],
            'bone_reference_role':'strong_soft_prior','longitudinal_scale':np.exp(best_shape[11:16]).tolist(),
            'fingertip_extension_mm':(best_shape[16:21]*1000).tolist(),
            'bone_reference_m':m.bone_reference.tolist() if m.bone_reference is not None else None}
    return result,report
