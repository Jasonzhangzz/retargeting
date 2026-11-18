"""Independent staged native-hand calibration. Never publishes current.json."""
from __future__ import annotations
import copy,hashlib,json,math,time,uuid
from pathlib import Path
import numpy as np
import torch
from .qnbot_surface import QnbotSurface,FINGERS
from .qnbot_reference import initialize_registration
from .qnbot_lsq_model import Coordinates,Measurements,Residuals,PARAMETER_REVISION,observed
from .qnbot_lsq_solver import Solver,options,pose_sweep,check
from .calibration_targets import target
from .contact import PATCH_IDS

FIT_REVISION='qnbot_least_squares_v1'


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')));tmp.replace(path)


def statistics(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    if not a.size:return dict(count=0,mean=None,median=None,p95=None,rmse=None,maximum=None)
    return dict(count=len(a),mean=float(a.mean()),median=float(np.median(a)),p95=float(np.percentile(a,95)),rmse=float(np.sqrt(np.mean(a*a))),maximum=float(a.max()))


@torch.no_grad()
def kinematic_asset(model):
    """Keep delivered body frames distinct from the 25 movable joint origins."""
    from .qnbot_surface import exp_so3
    _,body,rot=model(model.q0[None]);owners=torch.tensor(model.mj_model.jnt_bodyid,device=model.device,dtype=torch.long)
    origins=body[0,owners]+(rot[0,owners]@model.joint_p0[...,None]).squeeze(-1)
    g=model.geometry
    return dict(asset_hash=model.asset_hash,coordinate_frame='wrist_local',joint_names=model.joint_names,joint_origins_m=origins.cpu().tolist(),
        body_names=model.body_names,body_positions_m=body[0].cpu().tolist(),root_corrections_m=(.01*g['root']).cpu().tolist(),
        joint_axes_local=(exp_so3(g['axis'])@model.joint_axis0[...,None]).squeeze(-1).cpu().tolist(),axis_rotation_vectors=g['axis'].cpu().tolist(),
        joint_limits_rad=model.limits.cpu().tolist(),zero_offsets='fixed',coupling='fixed',joint_source='EMF_derived_seed')


def validate(o,x,q,cancel,progress,label,collision=True):
    """All validation frames, no inference from action or contact labels.

    Labels/evidence below select evaluation metrics only. Poses are passed in
    already frozen, either sensor-only reconstruction or exact streaming seeds.
    """
    m=o.m;d=o.d;c=o.c;frames=[];positions=[];rotations=[];per_finger={f:[] for f in FINGERS};per_stage={}
    gaps=[];angles=[];release=[];clusters=[];maximum=0.;checked=0;query_error=None
    collision_query=None
    if collision:
        try:
            from .qnbot_collision import QnbotBatchCollision
            if m.device.type!='cuda':raise ValueError('完整曲面验收需要CUDA')
            collision_query=QnbotBatchCollision(m,batch=16)
        except Exception as exc:query_error=str(exc)
    start_time=time.monotonic()
    with torch.no_grad():
        for start in range(0,len(q),16):
            check(cancel);ids=torch.arange(start,min(start+16,len(q)),device=q.device);qq=q[ids]
            g=c.geometry(x);pred,rotation=m.sensor_prediction(qq,geometry=g);align,trans=m.alignment(g)
            error=(pred-(d.p[ids]@align.T+trans)).norm(dim=-1)*1000
            relative=rotation.transpose(-1,-2)@(align@d.r[ids]);degrees=torch.rad2deg(torch.acos(((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)))
            sid,normal=c.skin(x,qq);patch=c.patch[None,:,None].expand(len(ids),-1,3);w=d.sid_weight[ids]
            pp=torch.zeros((len(ids),20,3),device=q.device).scatter_add(1,patch,sid*w[...,None])
            nn=torch.nn.functional.normalize(torch.zeros_like(pp).scatter_add(1,patch,normal*w[...,None]),dim=-1)
            depth=torch.zeros(len(ids),device=q.device)
            if collision_query:
                depth=collision_query.query(m(qq,geometry=g)[0]);maximum=max(maximum,float(depth.max()));checked+=len(ids)
            for local,i in enumerate(ids.cpu().tolist()):
                row=d.rows[i];mask=d.mask[i].bool();rmask=d.rotation_mask[i].bool();stage=row['stage']
                p=error[local][mask].cpu().tolist();r=degrees[local][rmask].cpu().tolist();positions.extend(p);rotations.extend(r)
                record=per_stage.setdefault(stage,dict(position=[],rotation=[],gaps=[],normals=[]));record['position'].extend(p);record['rotation'].extend(r)
                for f,name in enumerate(FINGERS):
                    if bool(mask[f]):per_finger[name].append(float(error[local,f]))
                contacts=[]
                for a,b,y in d.edges[i]:
                    gap=float((pp[local,a]-pp[local,b]).norm())*1000
                    angle=float(torch.rad2deg(torch.acos((-nn[local,a]@nn[local,b]).clamp(-1,1))))
                    if y==1:
                        gaps.append(gap);angles.append(angle);record['gaps'].append(gap);record['normals'].append(angle)
                        contacts.append(dict(regions=[PATCH_IDS[a],PATCH_IDS[b]],gap_mm=gap,normal_error_deg=angle,evidence='fresh_bilateral_taxels'))
                    else:release.append(gap)
                if target(stage).get('kind')=='distal_cluster' and contacts:
                    names={n for edge in contacts for n in edge['regions']};seen={next(iter(names))}
                    for _ in names:
                        for edge in contacts:
                            a,b=edge['regions']
                            if edge['gap_mm']<=5 and (a in seen or b in seen):seen.update((a,b))
                    clusters.append(len(seen)==len(names))
                frames.append(dict(timestamp_ms=row['timestamp_ms'],stage=stage,position_error_mm={f:float(error[local,k]) if bool(mask[k]) else None for k,f in enumerate(FINGERS)},
                    rotation_error_deg={f:float(degrees[local,k]) if bool(rmask[k]) else None for k,f in enumerate(FINGERS)},contacts=contacts,penetration_mm=float(depth[local]) if collision_query else None))
            done=start+len(ids)
            progress(dict(phase='validation',completed=done,total=len(q),message=label+f' {done}/{len(q)} 帧',estimated_remaining_s=(time.monotonic()-start_time)*(len(q)-done)/done))
    per_stage={s:dict(position_mm=statistics(v['position']),rotation_deg=statistics(v['rotation']),contact_gap_mm=statistics(v['gaps']),contact_normal_deg=statistics(v['normals'])) for s,v in per_stage.items()}
    missing=[s for s,v in per_stage.items() if target(s).get('patches') and not v['contact_gap_mm']['count']]
    return dict(position_mm=statistics(positions),rotation_deg=statistics(rotations),per_finger_mm={f:statistics(v) for f,v in per_finger.items()},per_stage=per_stage,
        contact_gap_mm=float(np.mean(gaps)) if gaps else None,contact_max_gap_mm=max(gaps) if gaps else None,
        contact_normal_error_deg=float(np.mean(angles)) if angles else None,contact_normal_max_deg=max(angles) if angles else None,
        max_penetration_mm=maximum if collision_query else None,validation_frames_checked=checked,validation_complete=checked==len(q),
        collision_error=query_error,collision_query='original_qnbot_triangles_all_vertices_and_face_centroids',
        release_min_gap_mm=min(release) if release else None,release_samples=len(release),contact_clusters_connected=all(clusters) if clusters else None,
        unverified_contact_stages=missing,contact_query='tactile_weighted_material_coordinates_and_outward_triangle_normals',
        joint_measurements_source='EMF_derived_seed_not_independent_supervision',seconds=time.monotonic()-start_time,frames=frames)


def quality(runtime,baseline,solver,shape):
    reasons=[]
    if not runtime['validation_complete']:reasons.append('未完成全部留出帧原始曲面检查')
    if runtime['collision_error']:reasons.append('碰撞查询无效：'+runtime['collision_error'])
    if runtime['contact_gap_mm'] is None:reasons.append('留出数据缺少有效双侧触觉接触')
    elif runtime['contact_gap_mm']>3 or runtime['contact_max_gap_mm']>5:reasons.append('固定前向接触间隙超过3/5mm')
    if runtime['unverified_contact_stages']:reasons.append('部分指定接触缺少可验证触觉证据')
    if runtime['contact_normal_max_deg'] is not None and runtime['contact_normal_max_deg']>15:reasons.append('接触法向超过15°')
    if runtime['max_penetration_mm'] is not None and runtime['max_penetration_mm']>2:reasons.append('原始曲面穿透超过2mm')
    if any(v['rmse'] is not None and v['rmse']>15 for v in runtime['per_finger_mm'].values()):reasons.append('逐指EMF定位RMSE超过15mm')
    if runtime['rotation_deg']['mean'] is not None and runtime['rotation_deg']['mean']>15:reasons.append('EMF平均朝向误差超过15°')
    if runtime['release_min_gap_mm'] is not None and runtime['release_min_gap_mm']<2:reasons.append('无触觉接触时材料点仍错误吸合')
    if runtime['contact_clusters_connected'] is False:reasons.append('多指接触簇未连通')
    for stage,v in runtime['per_stage'].items():
        before=baseline['per_stage'].get(stage,{}).get('position_mm',{}).get('rmse');after=v['position_mm']['rmse']
        if before is not None and after is not None and after>before*1.05+.5:
            reasons.append('逐动作定位相对基线退化超过5%+0.5mm');break
    if not shape['accepted']:reasons.append('局部尺寸或边长保护未通过')
    if solver.get('budget_exhausted'):reasons.append('优化预算耗尽，尚未收敛')
    return reasons


@torch.no_grad()
def shape_report(c,x):
    m=c.model;g=c.geometry(x);ratios=torch.cat([g[k].exp() for k in ('width','depth')]);rest=m.rest(geometry=g)[0];e=m.edges
    base=(c.reference_rest[e[:,1]]-c.reference_rest[e[:,0]]).norm(dim=-1).clamp_min(1e-8)
    edge=(rest[e[:,1]]-rest[e[:,0]]).norm(dim=-1)/base
    residual=g['residual'].norm(dim=-1)
    # Independent guards; they never manufacture observability.
    valid=bool((ratios>=.8).all()&(ratios<=1.25).all()&(edge>=.7).all()&(edge<=1.4).all()&(residual<=.003).all())
    return dict(accepted=valid,width_depth_range=[float(ratios.min()),float(ratios.max())],edge_length_ratio_range=[float(edge.min()),float(edge.max())],maximum_residual_mm=float(residual.max())*1000,
        topology_unchanged=True,skinning_weights_unchanged=True)


def fit_capture(model_path,profile,mapping,rows,progress,cancel,solver_options=None,initial_candidate=None,**kwargs):
    if initial_candidate is not None:raise ValueError('最小二乘分支从标准手型独立辨识，不支持神经网络续训')
    settings=options(solver_options);began=time.monotonic();run_id=profile.get('run_id') or uuid.uuid4().hex
    device=profile.get('device','cuda:0');side=profile['side']
    def update(value):progress(dict({'state':'running','method':'least_squares','surface_backend':'qnbot','run_id':run_id},**value))
    check(cancel)
    if not rows or any(r.get('hand_side')!=side for r in rows):raise ValueError('标定记录手侧缺失或混合')
    if any(r.get('partition') not in ('train','validation','initialization') for r in rows):raise ValueError('需要明确的独立训练、验证和初始化分区')
    if any(r.get('mapping',{}).get('assignments')!=mapping.get('assignments') for r in rows):raise ValueError('采集期间SID分区改变')
    train=[r for r in rows if r['partition']=='train' and r.get('training_eligible',True)]
    valid=[r for r in rows if r['partition']=='validation']
    init=[r for r in rows if r['partition']=='initialization']
    if not train or not valid:raise ValueError('缺少训练或独立验证分区')
    if {r['timestamp_ms'] for r in train}&{r['timestamp_ms'] for r in valid}:raise ValueError('训练与验证时间戳重叠')
    source_hash=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    model=QnbotSurface(side,device);model.bind_mapping(mapping)
    if profile.get('material_binding'):model.install_binding(profile['material_binding'])
    update(dict(phase='prepare',message='标准Qnbot手型：初始化外参，只读取初始化与训练记录',progress=.02))
    initialize_registration(model,init or train)
    baseline_artifact=model.artifact()
    coords=Coordinates(model);data=Measurements(model,train,training=True);objective=Residuals(coords,data)
    if not bool(data.valid.any(0).all()):raise ValueError('训练分区缺少部分手指的有效EMF位置观测')
    solver=Solver(objective,cancel,update,settings);x,qtrain,solver_report=solver.solve()
    optimization_s=time.monotonic()-began
    # Export frozen optimization before evaluating holdout. Validation cannot
    # choose a stage, coefficient, template, noise statistic or stopping point.
    valid_data=Measurements(model,valid,training=False,noise=data.noise);evaluation=Residuals(coords,valid_data)
    qfixed=valid_data.q0.clone()
    update(dict(phase='validation_pose',progress=.73,message='留出姿态仅用EMF求解；接触标签不进入求解'))
    qoffline,offline_pose=pose_sweep(evaluation,x,qfixed,cancel,settings,contact=False,iterations=4,
        progress=lambda done,total:update(dict(phase='validation_pose',completed=done,total=total,message=f'留出姿态仅用EMF求解 {done}/{total} 帧')))
    baseline_model=QnbotSurface(side,device);baseline_model.bind_mapping(mapping);baseline_model.load_artifact(baseline_artifact)
    bc=Coordinates(baseline_model);bd=Measurements(baseline_model,valid,training=False,noise=data.noise);bo=Residuals(bc,bd)
    baseline=validate(bo,torch.zeros(bc.size,device=model.device),bd.q0,cancel,update,'基线固定前向')
    fixed=validate(evaluation,x,qfixed,cancel,update,'候选实时固定前向')
    offline=validate(evaluation,x,qoffline,cancel,update,'候选离线EMF重建')
    shape=shape_report(coords,x);reasons=quality(fixed,baseline,solver_report,shape);duration=time.monotonic()-began
    report=dict(fit_revision=FIT_REVISION,method='least_squares',run_id=run_id,state='completed',surface_backend='qnbot',accepted=not reasons,
        automatic_application=False,preview_only=True,geometry_parameterization=PARAMETER_REVISION,frozen_forward=True,online_optimization=False,
        solver=solver_report,solver_options=settings,steps=solver_report['steps'],selected_training_frames=len(train),selected_validation_frames=len(valid),
        partition_counts={p:sum(r['partition']==p for r in rows) for p in ('train','validation','initialization')},initialization_frames=len(init),
        source_capture_hash=source_hash,training_evidence=data.summary(),validation_evidence=valid_data.summary(),noise_statistics=data.noise,
        baseline_fixed=baseline,runtime_validation=fixed,offline_validation=offline,offline_pose_solver=offline_pose,shape_validation=shape,
        validation_complete=fixed['validation_complete'] and offline['validation_complete'] and baseline['validation_complete'],
        validation_pose_source='upstream_qnbot25_fixed_forward',offline_validation_pose_source='EMF_only_LM_without_contact_or_action_labels',
        holdout_validation_passed=not reasons,contact_gap_mm=fixed['contact_gap_mm'],contact_max_gap_mm=fixed['contact_max_gap_mm'],
        max_penetration_mm=fixed['max_penetration_mm'],position_rmse_mm=fixed['position_mm']['rmse'],baseline_position_rmse_mm=baseline['position_mm']['rmse'],
        end_to_end_s=duration,optimization_s=optimization_s,validation_s=duration-optimization_s,time_target_met=duration<=60,performance_passed=duration<=60,
        performance_is_acceptance_gate=False,compilation_s=0.,rejection_reasons=reasons,reason='；'.join(reasons) if reasons else '独立复测通过；仅保存候选，未应用',
        result='not_converged' if solver_report['budget_exhausted'] else 'converged' if solver_report['converged'] else 'iteration_limit_or_stationary_subspace')
    candidate=dict(model.artifact(),serial=profile['serial'],mapping=copy.deepcopy(mapping),calibration_fingerprint=profile.get('calibration_fingerprint'),
        skeleton_runtime_revision=profile.get('skeleton_runtime_revision'),capture_session_id=profile.get('capture_session_id'),calibrated=False,
        geometry_parameterization=PARAMETER_REVISION,delta_policy=None,least_squares_parameters=dict(revision=PARAMETER_REVISION,coordinates=x.cpu().tolist(),names=coords.names),report=report)
    home=Path(profile['artifact_home']);folder=Path(profile.get('run_output') or home/'qnbot'/'runs'/'least_squares'/run_id)
    export(folder,model,coords,x,qtrain,qoffline,data,valid_data,candidate,report,cancel)
    report['artifact_directory']=str(folder.resolve());candidate['report']=report
    write_json(folder/'candidate.json',candidate);write_json(folder/'report.json',report)
    update(dict(state='completed',phase='completed',progress=1.,accepted=report['accepted'],message=report['reason']))
    return candidate,report


def export(folder,model,coords,x,qtrain,qoffline,train,valid,candidate,report,cancel):
    import trimesh
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);check(cancel)
    with torch.no_grad():
        g=coords.geometry(x);v,j,r=model(model.q0[None],geometry=g);v=v[0].cpu().numpy();rest=model.rest(geometry=g)[0].cpu().numpy()
        bc,offset=coords.binding(x);points,normal=coords.skin(x,model.q0[None])
    trimesh.Trimesh(v,model.faces.cpu().numpy(),process=False).export(folder/'hand_mesh.obj')
    trimesh.Trimesh(v,model.faces.cpu().numpy(),process=False).export(folder/'collision_mesh.obj')
    np.save(folder/'skin_weights.npy',model.weights.cpu().numpy())
    np.savez_compressed(folder/'offline_poses.npz',training_q=qtrain.cpu().numpy(),validation_q=qoffline.cpu().numpy(),
        training_timestamps_ms=[r['timestamp_ms'] for r in train.rows],validation_timestamps_ms=[r['timestamp_ms'] for r in valid.rows],joint_names=model.joint_names)
    np.savez_compressed(folder/'surface_basis.npz',basis=coords.surface_basis.cpu().numpy(),coefficients_m=.001*x[129:].cpu().numpy())
    write_json(folder/'hand_shape.json',dict(model='qnbot',parameterization='15_tied_finger_log_dimensions',
        dimensions={f:{k:float(g[k][torch.nonzero(coords.movable&(coords.owner==i))[0,0]].exp()) for k in ('length','width','depth')} for i,f in enumerate(FINGERS)},
        residual_basis=coords.surface_basis_report,coefficients_m=(.001*x[129:]).cpu().tolist(),topology_hash=model.topology_hash))
    write_json(folder/'kinematics.json',kinematic_asset(model))
    write_json(folder/'emf_extrinsics.json',dict(registration=candidate['registration'],mount_position_m=(model.mount_p0+.01*g['mount_pos']).cpu().tolist(),
        mount_rotation_vectors=g['mount_rot'].cpu().tolist(),alignment_rotation_vector=g['alignment'].cpu().tolist(),alignment_translation_m=(model.trans0+.01*g['translation']).cpu().tolist(),time_offset='not_identified'))
    write_json(folder/'tactile_map.json',dict(candidate['material_binding'],region_coordinates=x[69:129].reshape(20,3).cpu().tolist(),
        offset_interpretation='combined_glove_thickness_and_fit_gap_not_separately_identifiable',baseline='recorded_tare_profile',gain='not_force_calibrated'))
    write_json(folder/'contact_geometry.json',dict(surface='original_qnbot_triangles',sid_points_m=points[0].cpu().tolist(),sid_outward_normals=normal[0].cpu().tolist(),
        patch_ids=PATCH_IDS,patch_anchors=model.patch_anchors,tip_body_positions_m=j[0,model.tip_bodies].cpu().tolist()))
    write_json(folder/'validation.json',dict(baseline=report['baseline_fixed'],offline=report['offline_validation'],runtime=report['runtime_validation'],shape=report['shape_validation'],observability=report['solver']['phases']))
    (folder/'trace.jsonl').write_text(''.join(json.dumps(p,ensure_ascii=False,allow_nan=False)+'\n' for p in report['solver']['training_trace']))
    def fmt(value):return '无法验证' if value is None else f'{value:.3f}'
    text=f'''# Qnbot 分阶段最小二乘标定

设备 {candidate['serial']}，{model.side}；训练 {len(train.rows)} 帧，完整留出 {len(valid.rows)} 帧。

| 指标 | 基线固定前向 | 候选固定前向 | 候选离线 EMF 重建 |
|---|---:|---:|---:|
'''
    for title,key in [('位置 RMSE / mm','position_rmse'),('接触均值 / mm','contact_gap_mm'),('接触最大 / mm','contact_max_gap_mm'),('最大穿透 / mm','max_penetration_mm')]:
        values=[v['position_mm']['rmse'] if key=='position_rmse' else v[key] for v in (report['baseline_fixed'],report['runtime_validation'],report['offline_validation'])]
        text+='| '+title+' | '+' | '.join(fmt(v) for v in values)+' |\n'
    text+=f"\n验收通过：{report['accepted']}；优化 {report['optimization_s']:.1f} 秒；拟合与验证 {report['end_to_end_s']:.1f} 秒。60 秒目标仅作诊断。\n\n{report['reason']}。\n\n"
    for phase in report['solver']['phases']:
        diag=phase.get('observability',{});text+=f"- {phase['phase']}：支持秩 {diag.get('rank',0)}/{diag.get('coordinates',0)}；{phase.get('reason','已按证据求解')}。\n"
    text+='\n逐帧姿态只保存在 offline_poses.npz，不装入实时前向。原始数据与 current.json 未修改。\n'
    (folder/'REPORT.md').write_text(text)
