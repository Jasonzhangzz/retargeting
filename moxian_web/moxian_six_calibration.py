"""Moxian native25 plus root torsion bone/extrinsics fit with Wuji's six poses and native L-BFGS.

The FK, mesh and sensor frames are Moxian's. Wuji's built-in hand profiles and
21-to-25 display adapter are never used by this calibration method.
"""
from __future__ import annotations
import copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from .wuji_native_abi import SDK_POSES,FINGERS
from .wuji_native_capture import PROTOCOL
from .bone_calibration import Objective,noise_from_training,frozen_contacts,replay
from .bone_extrinsics import definition_from_surface,model_hash,validate_candidate
from .bone_objective_evaluator import ObjectiveEvaluator
from .qnbot_surface import QnbotSurface
from .qnbot_reference import initialize_registration
from .static_capture import aggregate
from .wuji_calibration_solver import WujiCalibrationLbfgs

METHOD='moxian_wuji'
REVISION='moxian_wuji_six_v3_bounded_stable'
STAGES=dict(zip(SDK_POSES,('contact_pair_index','contact_pair_middle','contact_pair_ring','contact_pair_pinky','mcp90','open')))


def initial_rows(profile,mapping,cancel,progress):
    if profile.get('initial_rows'):return copy.deepcopy(profile['initial_rows'])
    from .shared_capture import official_capture_document,STAGE_MAP
    from .mano_calibration import prepare_rows
    root=Path(__file__).resolve().parents[1];key=hashlib.sha256(profile['serial'].encode()).hexdigest()[:16]
    home=root/'output/moxian/ik'/key;_,document=official_capture_document(home,profile['calibration_fingerprint'])
    if document['payload']['side']!=profile['side']:raise ValueError('Moxian初始化手侧不一致')
    rows=[]
    for group in document['payload']['stage_samples']:
        if group['stage_id'] not in STAGE_MAP:continue
        for sample in group['frames']:
            p=sample['payload'];rows.append(dict(timestamp_ms=p['timestamp_ms'],hand_side=profile['side'],fingers=copy.deepcopy(p['fingertip_local']),
                stage=STAGE_MAP[group['stage_id']],initialization=True,partition='initialization',mapping=mapping,
                calibration_fingerprint=profile['calibration_fingerprint']))
    # The immutable official Moxian runtime is used for initialization, never
    # the currently displayed Wuji/Moxian candidate's inferred angles.
    settings=dict(profile,artifact_home=str(root/'output/moxian/hand_model'/key))
    rows,_=prepare_rows(settings,rows,progress,cancel)
    return rows


def training_rows(document):
    if document.get('protocol')!=PROTOCOL or document.get('completed_groups')!=list(SDK_POSES) or document.get('state',{}).get('state')!='complete':raise ValueError('请先完成六个静态动作')
    raw=[];train=[]
    for stage in SDK_POSES:
        frames=document.get('pose_frames',{}).get(stage,[])
        if not frames:raise ValueError('六动作原始观测缺失：'+stage)
        group=[]
        for i,nodes in enumerate(frames):
            if len(nodes)!=5:raise ValueError('需要五个原始EMF接收器')
            row=dict(timestamp_ms=1000+len(raw)*20,stage=STAGES[stage],native_pose=stage,fingers=dict(zip(FINGERS,copy.deepcopy(nodes))),
                partition='train',static_key='train:'+stage,static_accepted=True,contact_confirmed=stage.startswith('pinch_'),
                observation_weights=[1]*5,static_endpoint='flat' if stage=='flat_open' else 'tabletop' if stage=='four_finger_bend_90' else 'target')
            group.append(row);raw.append(row)
        row=aggregate(group);row.update(stage=STAGES[stage],native_pose=stage,partition='train',static_key='train:'+stage,
            contact_confirmed=stage.startswith('pinch_'),observation_weights=[1]*5,static_endpoint=group[0]['static_endpoint'])
        train.append(row)
    return raw,train


class SixObjective(Objective):
    def __init__(self,model,rows,contacts,noise):
        super().__init__(model,rows,contacts,noise)
        ids=[i for i,n in enumerate(model.joint_names) if any(n.endswith('_'+s) for s in ('mcp_flexion','pip_flexion','dip_flexion')) and '_thumb_' not in n]
        self.table_joints=torch.tensor(ids,device=model.device,dtype=torch.long)
        reference=self.rest.clone()
        for i in ids:
            if model.joint_names[i].endswith('_mcp_flexion'):reference[i]=np.pi/2
        self.table_reference=reference[self.table_joints].clamp(model.limits[self.table_joints,0],model.limits[self.table_joints,1])
        self.table_ids=torch.tensor([i for i,r in enumerate(rows) if r.get('static_endpoint')=='tabletop'],device=model.device,dtype=torch.long)
        self.thumb_palm_joints=torch.tensor([i for i,n in enumerate(model.joint_names) if n.endswith(('_thumb_cmc_flexion','_thumb_mcp_abduction'))],device=model.device,dtype=torch.long)
        self.flat_thumb_joints=torch.tensor([model.joint_names.index(n) for n in model.thumb_straight],device=model.device,dtype=torch.long)
    def bounds(self):
        lo,hi=super().bounds()
        # The explicitly collected flat endpoint anchors the RX mounting fit:
        # do not explain its orientation by folding the thumb backwards.
        qlo=lo[51:].reshape(len(self.rows),-1);qhi=hi[51:].reshape(len(self.rows),-1)
        for row in self.flat_ids.cpu().tolist():
            for j in self.flat_thumb_joints.cpu().tolist():
                qlo[row,j]=max(qlo[row,j],float(self.rest[j])-np.deg2rad(5))
                qhi[row,j]=min(qhi[row,j],float(self.rest[j])+np.deg2rad(5))
        return lo,hi
    def loss(self,vector):
        loss,terms=super().loss(vector)
        q=vector[51:].reshape(len(self.rows),len(self.model.joint_names))
        terms['tabletop_angles']=((q[self.table_ids][:,self.table_joints]-self.table_reference)/np.deg2rad(10)).square().sum()
        terms['flat_thumb_straight']=((q[self.flat_ids][:,self.flat_thumb_joints]-self.rest[self.flat_thumb_joints])/np.deg2rad(3)).square().sum()
        if self.flat_ids.numel():
            reference=q[self.flat_ids[-1],self.thumb_palm_joints]
            terms['thumb_palm_branch']=((q[:,self.thumb_palm_joints]-reference)/np.deg2rad(20)).square().sum()
        else:terms['thumb_palm_branch']=loss*0
        return loss+terms['tabletop_angles']+terms['flat_thumb_straight']+terms['thumb_palm_branch'],terms


def fit_capture(profile,mapping,document,progress,cancel):
    if cancel.is_set():raise InterruptedError('标定已取消')
    if profile['side']!=document['side'] or profile['calibration_fingerprint']!=document['calibration_fingerprint'] or mapping['assignments']!=document['mapping']['assignments']:raise ValueError('六动作记录与当前Moxian初始化不匹配')
    began=time.monotonic();raw,train=training_rows(document)
    initial=initial_rows(profile,mapping,cancel,progress)
    device=profile.get('device','cuda:0' if torch.cuda.is_available() else 'cpu')
    model=QnbotSurface(profile['side'],device,finger_twist=True);model.bind_mapping(mapping)
    if profile.get('material_binding'):model.install_binding(profile['material_binding'])
    initialize_registration(model,initial);model.double();model.fixed=False
    for p in model.parameters():p.requires_grad_(False)
    noise=noise_from_training(raw);source_hash=hashlib.sha256(json.dumps(document['pose_frames'],sort_keys=True).encode()).hexdigest()
    def artifact():
        value=model.artifact();value.update(serial=profile['serial'],side=profile['side'],mapping=copy.deepcopy(mapping),
            calibration_fingerprint=profile['calibration_fingerprint'],source_calibration_fingerprint=profile['calibration_fingerprint'],
            skeleton_runtime_revision=profile.get('skeleton_runtime_revision'),capture_session_id=profile.get('capture_session_id'),
            source_capture_hash=source_hash,delta_policy=None,bone_extrinsics=definition_from_surface(model,noise))
        value['model_hash']=model_hash(value);return value
    baseline=artifact()
    progress(dict(state='running',phase='prepare',progress=None,message='Moxian25＋四指侧旋：末节允许反折，固定模板、宽厚与蒙皮；仅拟合15骨长和36维外参'))
    for row,result in zip(train,replay(model,baseline,train,cancel)):row['skeleton']=result['pose']
    contacts=frozen_contacts(model,train);objective=SixObjective(model,train,contacts,noise)
    lo,hi=objective.bounds();start=np.clip(np.r_[np.zeros(51),objective.q0.cpu().numpy().ravel()],lo,hi)
    evaluator=ObjectiveEvaluator(objective,start,backend=profile.get('objective_backend','auto'))
    optimizer=WujiCalibrationLbfgs(len(start));optimizer.set_lower_bounds(lo);optimizer.set_upper_bounds(hi);optimizer.set_ftol_rel(1e-10);optimizer.set_maxeval(profile.get('maxeval',20000))
    evaluations=0;last_emit=0.;best=start.copy();best_loss=np.inf;optimization_start=time.monotonic()
    def callback(values,gradient):
        nonlocal evaluations,last_emit,best,best_loss
        if cancel.is_set():raise InterruptedError('标定已取消')
        loss,derivative,terms=evaluator(values)
        if gradient.size:gradient[:]=derivative
        evaluations+=1
        if loss<best_loss:best_loss=loss;best=values.copy()
        if evaluations==1 or time.monotonic()-last_emit>.75:
            last_emit=time.monotonic();progress(dict(state='running',phase='joint_fit',completed=evaluations,progress=None,objective_value=loss,
                message=f'Moxian骨长与EMF外参拟合 · {evaluations}次评估 · 目标值{loss:.6g}',
                training_sample=dict(sample_id=evaluations,step=evaluations,phase='mixed',scope='mixed',source='train_only',
                    raw=dict(loss=loss,terms=terms,optimizer_name='Wuji native NLopt LD_LBFGS',geometry_gradient_norm=float(np.linalg.norm(derivative[:51]))))))
        return loss
    optimizer.set_min_objective(callback);result=optimizer.optimize(start);status=optimizer.last_optimize_result()
    if status<0 and np.isfinite(best_loss):result=best
    if cancel.is_set():raise InterruptedError('标定已取消')
    with torch.no_grad():
        for k,v in objective.coordinates.geometry(torch.tensor(result[:51],device=model.device,dtype=torch.float64)).items():model.geometry[k].copy_(v)
    model._rest_cache=None;model.fixed=True;candidate=artifact()
    # Only the neutral thumb base posture is retained, not per-action angles.
    # Hinge rest angles remain physical straight. Runtime still uses raw EMF.
    flat=[i for i,row in enumerate(train) if row.get('static_endpoint')=='flat']
    if flat:
        neutral=result[51:].reshape(len(train),len(model.joint_names))[flat[-1]]
        for j,name in enumerate(model.joint_names):
            if '_thumb_' in name and name not in model.thumb_straight:
                candidate['bone_extrinsics']['rest_angles_rad'][j]=float(neutral[j])
        candidate['bone_extrinsics']['neutral_reference']='training_flat_thumb_base_v1'
    candidate['model_hash']=model_hash(candidate);validate_candidate(candidate)
    progress(dict(state='running',phase='validation',progress=None,message='核验Moxian原始EMF回放与旋转方向；不使用动作目标驱动IK'))
    fitted=replay(model,candidate,train,cancel);seed=replay(model,baseline,train,cancel)
    def errors(results,key):return np.asarray([[f[key] for f in r['ik_diagnostics']['fingers']] for r in results])
    p=errors(fitted,'position_error_mm');r=errors(fitted,'rotation_error_deg');sp=errors(seed,'position_error_mm');sr=errors(seed,'rotation_error_deg')
    position=float(np.sqrt(np.mean(p*p)));rotation=float(r.mean());reasons=[]
    if status not in (1,2,3,4):reasons.append('原生L-BFGS未正常收敛：'+str(status))
    if position>15 or position>np.sqrt(np.mean(sp*sp))*1.05+.5:reasons.append('EMF位置残差超标或较原Moxian模型退化')
    if rotation>15 or rotation>sr.mean()+2 or r.max()>35:reasons.append('EMF旋转残差超标或较原Moxian模型退化，保留原模型')
    accepted=not reasons
    report=dict(method=METHOD,fit_revision=REVISION,surface_backend='qnbot',state='completed',accepted=accepted,applied=False,
        reason='Moxian骨长与EMF外参拟合通过' if accepted else '；'.join(reasons),rejection_reasons=reasons,
        capture_protocol=PROTOCOL,capture_session_id=profile.get('capture_session_id'),source_capture_hash=source_hash,
        validation_complete=True,validation_pending=False,validation_scope='collected_six_pose_emf_replay_not_independent_holdout',
        model_family='moxian_native25_plus4_twist',solver_dof=len(model.joint_names),joint_limit_policy=candidate['bone_extrinsics']['joint_limit_policy'],flat_thumb_straight_sigma_deg=3,flat_thumb_hinge_allowance_deg=5,independent_geometry_parameters=51,selected_training_frames=6,
        position_rmse_mm=position,emf_orientation_error_deg=rotation,emf_orientation_max_deg=float(r.max()),
        baseline_position_rmse_mm=float(np.sqrt(np.mean(sp*sp))),baseline_emf_orientation_error_deg=float(sr.mean()),
        per_pose_emf_position_mm=p.tolist(),per_pose_emf_rotation_deg=r.tolist(),
        bone_lengths_m=candidate['bone_extrinsics']['bone_lengths_m'],T_wrist_TX=candidate['bone_extrinsics']['T_wrist_TX'],T_distal_RX=candidate['bone_extrinsics']['T_distal_RX'],
        end_to_end_s=time.monotonic()-began,online_optimization=True,delta_policy=False,
        frozen_geometry=['width','depth','root','axis','residual','material_binding','skin_weights'],
        solver=dict(**optimizer.provenance,optimizer_name='Wuji native NLopt LD_LBFGS',status=status,converged=status in (1,2,3,4),
            evaluations=evaluations,ftol_rel=1e-10,maxeval=profile.get('maxeval',20000),full_dimension=len(start),
            optimization_s=time.monotonic()-optimization_start,final_objective_value=evaluator(result)[0],objective_execution=evaluator.provenance(),
            objective_model='Moxian25 plus four axial joints FK and fixed Qnbot surface',wuji_builtin_hand_model_used=False))
    candidate['report']=report
    if cancel.is_set():raise InterruptedError('标定已取消')
    return candidate,report
