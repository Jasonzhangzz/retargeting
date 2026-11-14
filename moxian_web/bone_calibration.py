"""51 shared coordinates plus latent native poses, fitted by NLopt L-BFGS."""
from __future__ import annotations
import copy
import hashlib
import json
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .bone_extrinsics import (METHOD,FIT_REVISION,FINGERS,segment_indices,
    definition_from_surface,model_hash,validate_candidate)
from .static_capture import PROTOCOL,static_schedule,aggregate
from .static_pose import REVISION as POSE_REVISION
from .qnbot_surface import QnbotSurface,exp_so3
from .qnbot_reference import initialize_registration
from .qnbot_lsq_model import observed
from .card_math import log_so3
from .calibration_targets import target
from .wuji_calibration_solver import WujiCalibrationLbfgs
from .bone_objective_evaluator import ObjectiveEvaluator

DEFAULTS=dict(ftol_rel=1e-10,maxeval=20000)


def capture_hashes(rows):
    """Same hashes as json.dumps(sorted rows), encoding large raw rows once.

    Raw snapshots include tactile arrays and repeated mappings. Avoid two
    complete JSON strings while retaining the existing fingerprint contract.
    """
    source=hashlib.sha256();training=hashlib.sha256()
    source.update(b'[');training.update(b'[')
    seen=False;training_seen=False
    for row in rows:
        encoded=json.dumps(row,sort_keys=True).encode()
        if seen:source.update(b', ')
        source.update(encoded);seen=True
        if row.get('partition')!='validation':
            if training_seen:training.update(b', ')
            training.update(encoded);training_seen=True
    source.update(b']');training.update(b']')
    return source.hexdigest(),training.hexdigest()


def options(value=None):
    result={**DEFAULTS,**(value or {})}
    if set(result)!=set(DEFAULTS) or type(result['maxeval']) is not int or not 1<=result['maxeval']<=20000 or not 0<float(result['ftol_rel'])<1:
        raise ValueError('无效L-BFGS设置')
    return result


class Coordinates:
    def __init__(self,model):
        self.model=model;self.ids=torch.tensor(segment_indices(model),device=model.device)
        self.base={k:v.detach().clone()*0 for k,v in model.geometry.items()}

    def geometry(self,x):
        g=dict(self.base)
        g['length']=g['length'].index_copy(0,self.ids,x[:15])
        g['mount_pos']=x[15:30].reshape(5,3)
        g['mount_rot']=x[30:45].reshape(5,3)
        g['alignment']=x[45:48];g['translation']=x[48:51]
        return g

    def bounds(self):
        high=np.r_[np.full(15,.15),np.full(15,2.),np.full(15,.3),np.full(3,.2),np.full(3,2.)]
        return -high,high


def noise_from_training(rows):
    """Only accepted training collection intervals, never held-out samples."""
    groups={}
    for row in rows:
        if row.get('static_accepted') and row.get('partition')=='train':
            groups.setdefault(row['static_key'],[]).append(row)
    positions=[];rotations=[]
    for group in groups.values():
        for finger in FINGERS:
            p=np.array([r['fingers'][finger]['position'] for r in group])
            q=Rotation.from_quat([r['fingers'][finger]['quaternion_xyzw'] for r in group])
            positions.extend(np.std(p,axis=0).tolist())
            rotations.extend(np.std((q.mean().inv()*q).as_rotvec(),axis=0).tolist())
    return dict(position_sigma_m=float(np.clip(np.median(positions) if positions else .003,.001,.01)),
                rotation_sigma_rad=float(np.clip(np.median(rotations) if rotations else .15,.05,.3)))


def frozen_contacts(model,rows):
    """Choose material witnesses and cluster edges once from training seeds."""
    from .contact_protocol import cluster_tree
    from .qnbot_calibration import nearest_region_anchors
    actions={}
    with torch.no_grad():
        q=model.angles(rows).to(dtype=model.q0.dtype)
        vertices=model(q)[0]
        for i,row in enumerate(rows):
            stage=row['stage'];spec=target(stage)
            if not row.get('contact_confirmed') or stage in actions:continue
            edges=copy.deepcopy(spec.get('edges',[]))
            if spec['kind']=='distal_cluster':
                names=spec['patches'];pp,_=model.anchors(vertices[i:i+1],[model.patch_anchors[n] for n in names])
                points=pp[0].cpu().numpy();edges=cluster_tree(names,np.linalg.norm(points[:,None]-points[None],axis=-1))
            result=[]
            for edge in edges:
                a,b=edge['regions'];_,anchors=nearest_region_anchors(vertices[i].cpu().numpy(),model.faces.cpu().numpy(),model.regions,a,b)
                result.append(dict(edge,anchors=anchors))
            if not result:raise ValueError('缺少接触区域定义 '+stage)
            actions[stage]=result
    return actions


class Objective:
    def __init__(self,model,rows,contacts,noise):
        self.model=model;self.rows=rows;self.coordinates=Coordinates(model);self.contacts=contacts;self.noise=noise
        self.p,self.r,self.valid,_=observed(rows,model.device)
        self.p=self.p.double();self.r=self.r.double();self.valid=self.valid.double()
        self.mask=self.valid*torch.tensor([r.get('observation_weights',[1]*5) for r in rows],device=model.device)
        seed=model.angles(rows).double()
        if getattr(model,'finger_twist_enabled',False):
            seed=seed.clone();seed[:,25:]=torch.remainder(seed[:,25:]+np.pi,2*np.pi)-np.pi
        self.q0=seed.maximum(model.limits[:,0]).minimum(model.limits[:,1])
        self.rest=torch.tensor(definition_from_surface(model)['rest_angles_rad'],device=model.device,dtype=torch.float64)
        self.flat_ids=torch.tensor([i for i,r in enumerate(rows) if r.get('static_endpoint') in ('flat','spread','together')],device=model.device,dtype=torch.long)
        self.flat_mcps=torch.tensor([model.body_names.index(model.side+'_'+f+'_proximal') for f in ('index','middle','ring','little')],device=model.device,dtype=torch.long)
        with torch.no_grad():
            rr,pp=model.fk(self.rest[None]);wr=rr[0,model.wrist_body];pp=(pp[0]-pp[0,model.wrist_body])@wr
            mcps=[model.body_names.index(model.side+'_'+f+'_proximal') for f in ('index','middle','ring','little')]
            direction=(pp[model.tip_bodies[1:]]-pp[mcps]).mean(0)
            lateral=pp[mcps[-1]]-pp[mcps[0]]
            self.flat_normal=torch.nn.functional.normalize(torch.linalg.cross(direction,lateral),dim=0)
        self.anchors=[];self.edges=[]
        for i,row in enumerate(rows):
            if not row.get('contact_confirmed'):continue
            for e in contacts[row['stage']]:
                start=len(self.anchors);self.anchors.extend(e['anchors'])
                self.edges.append((i,start,start+1,e.get('weight',1.),e.get('required',True)))
        if self.anchors:
            faces=torch.tensor([a['face'] for a in self.anchors],device=model.device)
            self.support=torch.unique(model.faces[faces].flatten(),sorted=True)
            self.tri=torch.searchsorted(self.support,model.faces[faces])
            self.bc=torch.tensor([a['barycentric'] for a in self.anchors],device=model.device,dtype=torch.float64)

    def bounds(self):
        lo,hi=self.coordinates.bounds()
        qlo=self.model.limits[:,0].expand(len(self.rows),-1).clone();qhi=self.model.limits[:,1].expand(len(self.rows),-1).clone()
        active=self.mask[:,self.model.joint_finger_ids]>0
        qlo=torch.where(active,qlo,self.q0);qhi=torch.where(active,qhi,self.q0)
        return np.r_[lo,qlo.cpu().numpy().ravel()],np.r_[hi,qhi.cpu().numpy().ravel()]

    def loss(self,vector):
        m=self.model;x=vector[:51];q=vector[51:].reshape(len(self.rows),len(m.joint_names));g=self.coordinates.geometry(x)
        # All poses and the derived rest geometry share one FK traversal.
        all_r,all_p=m.fk(torch.cat((q,m.q0[None])),geometry=g)
        kinematics=(all_r[:-1],all_p[:-1]);rest_kinematics=(all_r[-1:],all_p[-1:])
        pred,rot=m.sensor_prediction(q,g,kinematics=kinematics);a,t=m.alignment(g)
        terms=dict(sensor_position=(((pred-(self.p@a.T+t))/self.noise['position_sigma_m']).square().sum(-1)*self.mask).sum(),
            sensor_orientation=((log_so3(rot.transpose(-1,-2)@(a@self.r))/self.noise['rotation_sigma_rad']).square().sum(-1)*self.mask).sum())
        terms['bone_prior']=(x[:15]/.1).square().sum()
        lengths=x[:15].reshape(5,3)
        terms['bone_ratio_prior']=((lengths[:,1:]-lengths[:,:-1])/.1).square().sum()
        terms['extrinsics_prior']=x[15:30].square().sum()+(x[30:45]/.15).square().sum()+x[48:51].square().sum()+(x[45:48]/.15).square().sum()
        # A weak seed prior fixes unobservable null directions; the vendor
        # angles are initialization, not measurement targets or online inputs.
        terms['pose_null_prior']=.01*(((q-self.q0)/.5).square()*self.mask[:,m.joint_finger_ids]).sum()
        if self.anchors:
            vertices,_,_=m(q,vertex_ids=self.support,geometry=g,kinematics=kinematics,rest_kinematics=rest_kinematics);tri=vertices[:,self.tri]
            p=(tri*self.bc[None,...,None]).sum(-2)
            n=torch.nn.functional.normalize(torch.linalg.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0],dim=-1),dim=-1)
            terms['contact_position']=sum(w*((p[i,a]-p[i,b])/.003).square().sum() for i,a,b,w,_ in self.edges)
            # ||na+nb|| is the angle near antiparallel contact, in radians.
            terms['contact_normal']=sum(w*((n[i,a]+n[i,b])/np.deg2rad(15)).square().sum() for i,a,b,w,_ in self.edges)
        if self.flat_ids.numel():
            r,p=kinematics
            j=((p-p[:,m.wrist_body,None])@r[:,m.wrist_body])[self.flat_ids]
            terms['flat_plane']=(((j[:,m.tip_ids[1:]]-j[:,self.flat_mcps])*self.flat_normal).sum(-1)/.003).square().sum()
            terms['flat_angles']=((q[self.flat_ids]-self.rest)/np.deg2rad(10)).square().sum()
        return sum(terms.values()),terms


def replay(model,candidate,rows,cancel,progress=None,label='个体模型'):
    from .emf_qp import QPSkeleton
    engine=QPSkeleton(candidate);results=[]
    for index,row in enumerate(rows):
        if cancel.is_set():raise InterruptedError('标定已取消')
        # Reset at independently held poses. No instructed target or fitted
        # latent angle enters the live solver, including its first pose.
        result=engine.step(row);results.append(result)
        if progress and index%20==0:progress(dict(state='running',phase='validation',completed=index,total=len(rows),progress=.78,message=label+'：逐帧原始EMF驱动QP'))
    return results


def validate(model,candidate,rows,contacts,baseline,cancel,progress):
    from .qnbot_collision import QnbotBatchCollision
    results=replay(model,candidate,rows,cancel,progress);seed=replay(model,baseline,rows,cancel,progress,'官方初始化基线')
    q=torch.tensor([r['pose']['joint_positions_rad'] for r in results],device=model.device,dtype=model.q0.dtype)
    seeds=np.array([[v['position_error_mm'] for v in r['ik_diagnostics']['fingers']] for r in seed])
    errors=np.array([[v['position_error_mm'] for v in r['ik_diagnostics']['fingers']] for r in results])
    angles=np.array([[v['rotation_error_deg'] for v in r['ik_diagnostics']['fingers']] for r in results])
    mask=np.array([r.get('observation_weights',[1]*5) for r in rows],bool)
    position=float(np.sqrt(np.mean(errors[mask]**2)));seed_error=float(np.sqrt(np.mean(seeds[mask]**2)))
    gaps=[];normal_errors=[]
    rejection=[];penetration=0.;complete=True;release=[];checked=0
    with torch.no_grad():
        for start in range(0,len(rows),8):
            batch=rows[start:start+8]
            objective=Objective(model,batch,contacts,candidate['bone_extrinsics']['solver_config'])
            if not objective.anchors:continue
            v=model(q[start:start+8],vertex_ids=objective.support)[0];tri=v[:,objective.tri]
            points=(tri*objective.bc[None,...,None]).sum(-2)
            normals=torch.nn.functional.normalize(torch.linalg.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0],dim=-1),dim=-1)
            for i,a,b,w,required in objective.edges:
                if not required:continue
                gaps.append(float((points[i,a]-points[i,b]).norm())*1000)
                normal_errors.append(float(torch.rad2deg(torch.acos((-normals[i,a]@normals[i,b]).clamp(-1,1)))))
        try:
            collision=QnbotBatchCollision(model,batch=min(len(rows),8))
            for start in range(0,len(rows),8):
                if cancel.is_set():raise InterruptedError('复测已取消')
                penetration=max(penetration,float(collision.query(model(q[start:start+8])[0]).max()));checked+=len(q[start:start+8])
                progress(dict(state='running',phase='validation',completed=min(start+8,len(rows)),total=len(rows),progress=.8+.19*min(start+8,len(rows))/len(rows),message='原始EMF驱动QP独立复测'))
        except (RuntimeError,ValueError) as exc:complete=False;rejection.append('完整曲面穿透验收不可用：'+str(exc))
        for i,row in enumerate(rows):
            if row.get('static_endpoint') not in ('spread','flat'):continue
            vertices=model(q[i:i+1])[0]
            for edges in contacts.values():
                for edge in edges:
                    if not edge.get('required',True):continue
                    p,_=model.anchors(vertices,edge['anchors']);release.append(float((p[0,0]-p[0,1]).norm())*1000)
    if not gaps:rejection.append('独立复测缺少接触检查')
    if gaps and (np.mean(gaps)>3 or max(gaps)>5):rejection.append('接触均值／最大间隙超过3／5mm')
    if normal_errors and max(normal_errors)>15:rejection.append('接触法向超过15度')
    if penetration>2:rejection.append('穿透超过2mm')
    if position>15 or position>seed_error*1.05+.5:rejection.append('EMF定位残差超标或较基线退化')
    if not release or min(release)<2:rejection.append('展平／张开防吸合检查未通过')
    per_finger={f:float(np.sqrt(np.mean(errors[mask[:,i],i]**2))) for i,f in enumerate(FINGERS) if mask[:,i].any()}
    seed_angles=np.array([[v['rotation_error_deg'] for v in r['ik_diagnostics']['fingers']] for r in seed])
    if any(v>15 for v in per_finger.values()):rejection.append('单指EMF定位残差超过15mm')
    if not np.isfinite(angles).all() or angles[mask].mean()>15 or angles[mask].mean()>seed_angles[mask].mean()+2:rejection.append('EMF朝向误差超过15度或较基线退化超过2度')
    return dict(validation_complete=complete and checked==len(rows),validation_frames_checked=checked,position_rmse_mm=position,baseline_position_rmse_mm=seed_error,sensor_rmse_mm=per_finger,
        contact_gap_mm=float(np.mean(gaps)) if gaps else None,contact_max_gap_mm=max(gaps) if gaps else None,
        contact_normal_error_deg=max(normal_errors) if normal_errors else None,noncontact_penetration_mm=penetration,
        emf_orientation_error_deg=float(angles[mask].mean()),release_min_gap_mm=min(release) if release else None,
        validation_source='raw_emf_qp_only',rejection_reasons=rejection)


def fit_capture(model_path,profile,mapping,rows,progress,cancel,method=METHOD,solver_options=None,**kwargs):
    if cancel.is_set():raise InterruptedError('标定已取消')
    began=time.monotonic();settings=options(solver_options)
    fingerprint=profile.get('calibration_fingerprint')
    if not fingerprint or any(r.get('calibration_fingerprint')!=fingerprint for r in rows):
        raise ValueError('原始记录与八动作初始化指纹不一致，请重新采集')
    aggregates=copy.deepcopy(profile.get('static_aggregates',[]))
    expected=[':'.join(x) for x in static_schedule()]
    keys=[r.get('static_key') for r in aggregates]
    if len(keys)<18 or len(keys)>len(expected) or keys!=expected[:len(keys)] or any(r.get('capture_protocol')!=PROTOCOL for r in aggregates):
        raise ValueError('需要完整的静态17动作拟合采集；旧连续轨迹仅供查看')
    validation_ready=len(keys)==len(expected)
    if any(r.get('static_pose_validation',{}).get('revision')!=POSE_REVISION or not r['static_pose_validation'].get('accepted') for r in aggregates):
        raise ValueError('静态记录缺少目标姿态检验，请重新采集17动作')
    initial=[r for r in rows if r.get('initialization')]
    if not initial:raise ValueError('缺少八动作初始化')
    train=[r for r in aggregates if r['partition']=='train'];valid=[r for r in aggregates if r['partition']=='validation']
    opened=[r for r in initial if r['stage']=='open']
    if opened:
        flat=aggregate(opened);flat.update(stage='open',partition='initialization',static_endpoint='flat',contact_confirmed=False,observation_weights=[1]*5)
        train.append(flat)
    train.sort(key=lambda r:r['timestamp_ms'])
    device=profile.get('device','cuda:0' if torch.cuda.is_available() else 'cpu')
    model=QnbotSurface(profile['side'],device);model.bind_mapping(mapping)
    if profile.get('material_binding'):model.install_binding(profile['material_binding'])
    initialize_registration(model,initial);model.double();model.fixed=False
    for p in model.parameters():p.requires_grad_(False)
    noise=noise_from_training(rows)
    solver_config=noise
    source_hash,training_hash=capture_hashes(rows)
    def candidate():
        value=model.artifact();value.update(serial=profile['serial'],side=profile['side'],calibration_fingerprint=profile['calibration_fingerprint'],
            source_calibration_fingerprint=profile['calibration_fingerprint'],skeleton_runtime_revision=profile.get('skeleton_runtime_revision'),
            mapping=copy.deepcopy(mapping),delta_policy=None,bone_extrinsics=definition_from_surface(model,solver_config),
            capture_session_id=profile.get('capture_session_id'),source_capture_hash=source_hash)
        value['model_hash']=model_hash(value);return value
    baseline=candidate()
    previous=kwargs.get('initial_candidate')
    if previous is not None and previous.get('report',{}).get('validation_pending') and validation_ready:
        previous_report=previous['report']
        if (previous_report.get('training_capture_hash')!=training_hash or
            previous.get('capture_session_id')!=profile.get('capture_session_id') or
            previous.get('source_calibration_fingerprint')!=fingerprint):
            raise ValueError('待复测候选与当前拟合采集不一致，请重新拟合')
        model.load_artifact(previous);model.fixed=True
        return finish_capture(model,copy.deepcopy(previous),rows,previous_report['contact_templates'],baseline,
            cancel,progress,began,train,valid,noise,previous_report['solver'],training_hash,True,
            reused_fitted_model=True)

    # Never inherit a previous personalized model's capture-time IK seeds.
    # Recompute training seeds from the immutable official initialization.
    seeds=replay(model,baseline,train,cancel)
    for row,result in zip(train,seeds):row['skeleton']=result['pose']
    contacts=frozen_contacts(model,train)
    progress(dict(state='running',phase='prepare',progress=.1,message='15段骨长＋36维外参；冻结材料接触与其余几何'))
    objective=Objective(model,train,contacts,noise);lo,hi=objective.bounds()
    start=np.r_[np.zeros(51),objective.q0.cpu().numpy().ravel()]
    progress(dict(state='running',phase='prepare',progress=None,message='准备固定形状的目标函数与梯度'))
    evaluator=ObjectiveEvaluator(objective,start,backend=kwargs.get('objective_backend','auto'))
    evaluations=0;last_emit=0.;trace=[];best=start.copy();best_loss=np.inf
    optimization_started=time.monotonic()
    optimizer=WujiCalibrationLbfgs(len(start));optimizer.set_lower_bounds(lo);optimizer.set_upper_bounds(hi)
    optimizer.set_ftol_rel(settings['ftol_rel']);optimizer.set_maxeval(settings['maxeval'])
    def callback(values,gradient):
        nonlocal evaluations,last_emit,best,best_loss
        if cancel.is_set():raise InterruptedError('标定已取消')
        scalar,derivative,terms=evaluator(values)
        if gradient.size:gradient[:]=derivative
        evaluations+=1
        if scalar<best_loss:best_loss=scalar;best=values.copy()
        if time.monotonic()-last_emit>.75 or evaluations==1:
            point=dict(sample_id=evaluations,step=evaluations,phase='mixed',scope='mixed',source='train_only',
                raw=dict(loss=scalar,terms=terms,optimizer_name='Wuji native NLopt LD_LBFGS',geometry_gradient_norm=float(np.linalg.norm(derivative[:51]))))
            trace.append(point);last_emit=time.monotonic()
            progress(dict(state='running',phase='joint_fit',progress=None,completed=evaluations,evaluation_limit=settings['maxeval'],total=None,
                objective_value=scalar,best_objective_value=best_loss,evaluations_per_s=evaluations/max(time.monotonic()-optimization_started,1e-9),
                objective_backend=evaluator.backend,message='联合优化共享骨长、EMF外参与各静态姿态',training_sample=point,estimated_remaining_s=None))
        return scalar
    optimizer.set_min_objective(callback)
    result=optimizer.optimize(start);status=optimizer.last_optimize_result()
    if status<0 and np.isfinite(best_loss):result=best
    if cancel.is_set():raise InterruptedError('标定已取消')
    with torch.no_grad():
        geometry=objective.coordinates.geometry(torch.tensor(result[:51],device=model.device,dtype=torch.float64))
        for key,value in geometry.items():model.geometry[key].copy_(value)
    model._rest_cache=None;model.fixed=True
    fitted=candidate();validate_candidate(fitted)
    solver_info=dict(optimizer_name='Wuji native NLopt LD_LBFGS',**optimizer.provenance,status=status,evaluations=evaluations,
        converged=status in (1,2,3,4),training_trace=dict(points=trace),objective_execution=evaluator.provenance(),
        best_objective_value=best_loss,final_objective_value=evaluator(result)[0],
        optimization_s=time.monotonic()-optimization_started,**settings)
    return finish_capture(model,fitted,rows,contacts,baseline,cancel,progress,began,train,valid,noise,
        solver_info,training_hash,validation_ready)


def finish_capture(model,fitted,rows,contacts,baseline,cancel,progress,began,train,valid,noise,
                   solver_info,training_hash,validation_ready,reused_fitted_model=False):
    # Every raw held-out frame is replayed; contact equalities are scored only
    # on accepted static collection intervals. Labels never enter QP.step.
    validation_rows=[]
    if validation_ready:
        for row in rows:
            if row.get('partition')!='validation' or not row.get('training_eligible',True):continue
            r=copy.deepcopy(row);r['contact_confirmed']=bool(r.get('static_accepted') and target(r['stage'])['kind']!='noncontact_motion')
            validation_rows.append(r)
        if not validation_rows:raise ValueError('独立复测缺少原始EMF保持片段')
        validation=validate(model,fitted,validation_rows,contacts,baseline,cancel,progress)
    else:
        validation=dict(validation_complete=False,validation_frames_checked=0,rejection_reasons=[],validation_source='pending_independent_capture')
    if cancel.is_set():raise InterruptedError('标定已取消')
    reasons=validation['rejection_reasons']
    if not solver_info['converged']:reasons.append('L-BFGS未收敛：'+str(solver_info['status']))
    accepted=validation['validation_complete'] and not reasons
    pending=not validation_ready
    report=dict(validation,fit_revision=FIT_REVISION,method=METHOD,surface_backend='qnbot',state='completed',accepted=accepted,
        validation_pending=pending,reused_fitted_model=reused_fitted_model,training_capture_hash=training_hash,
        capture_session_id=fitted.get('capture_session_id'),
        holdout_validation_passed=accepted,reason='EMF与骨架拟合完成，待独立复测' if pending else '验收通过' if accepted else '；'.join(reasons),
        independent_geometry_parameters=51,geometry_parameterization='15_segment_lengths_36_extrinsics',
        frozen_geometry=['width','depth','root','axis','residual','material_binding','skin_weights'],
        bone_lengths_m=fitted['bone_extrinsics']['bone_lengths_m'],T_wrist_TX=fitted['bone_extrinsics']['T_wrist_TX'],T_distal_RX=fitted['bone_extrinsics']['T_distal_RX'],
        capture_protocol=PROTOCOL,selected_training_frames=len(train),selected_validation_frames=len(validation_rows),validation_poses=len(valid),
        end_to_end_s=time.monotonic()-began,online_optimization=True,frozen_forward=False,
        source_capture_hash=fitted['source_capture_hash'],loss_spec=dict(noise,contact_sigma_m=.003,contact_normal_sigma_deg=15),
        solver=solver_info,contact_templates=contacts)
    fitted['report']=report;return fitted,report
