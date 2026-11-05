#!/usr/bin/env python3
"""Audit stored Moxian calibration; optional sensor-only refit. Never applies sizes."""
from __future__ import annotations
import argparse,copy,hashlib,json,sys,threading,time
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moxian_web.mano_runtime import ManoSurface
from moxian_web.hand_model import DEFAULT_MANO,DEFAULT_MATERIAL
from moxian_web.contact import ContactEngine,FINGERS
from moxian_web.hand_anatomy import GEOMETRY_REVISION
from moxian_web.hand_dimensions import DIMENSION_MODE
from moxian_web.mano_fit import choose_rows,fit_capture,FIT_REVISION

def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2));temp.replace(path)

def restore(m,p):
    m.proximal_lateral_scales=np.array(p.get('proximal_lateral_scales',[0.]*4),dtype=float)
    m.bone_reference=np.array(p['bone_reference']) if p.get('bone_reference') is not None else None
    m.dimension_parameters=np.array(p.get('dimension_parameters',np.zeros(20)));m.set_shape(p['shape'])
    m.raw_rotation=np.array(p['raw_rotation']);m.raw_translation=np.array(p['raw_translation']);m.raw_aligned=True
    m.mount_pos=np.array(p['mount_pos']);m.mount_rot=np.array(p['mount_rot']);m.flat_reference=copy.deepcopy(p.get('flat_reference'))
    m.sensor_joint_position=np.array(p['sensor_joint_position']) if p.get('sensor_joint_position') is not None else None
    m.mounting_calibrated=bool(p.get('mounting_calibrated'));m.mounting_estimated=p.get('report',{}).get('validation_pose_source')=='runtime_emf_only';m.has_wrist_targets=True
    if m.flat_reference:m.flat_seed=np.array(m.flat_reference.get('flat_seed',np.zeros((15,3))),np.float32);m.reference_root=np.eye(3,dtype=np.float32)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--serial',default='2DF5F3B00DA5');ap.add_argument('--initial-profile',type=Path);ap.add_argument('--refit',action='store_true');ap.add_argument('--publish-candidate',type=Path);ap.add_argument('--seconds',type=float,default=180);args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];folder=root/'output/moxian';home=folder/'hand_model'/hashlib.sha256(args.serial.encode()).hexdigest()[:16]
    capture_path=home/'capture.json';source_bytes=capture_path.read_bytes();capture=json.loads(source_bytes);rows=capture['rows']
    reference=json.loads((home/'sensor-reference.json').read_text());candidate_path=home/'candidate.json';candidate=json.loads(candidate_path.read_text()) if candidate_path.exists() else None
    original_candidate=candidate_path.read_bytes() if candidate_path.exists() else None
    m=ManoSurface(DEFAULT_MANO);m.set_dimensions(np.zeros(20));m.observations(rows[0]);m.last_frame=rows[0];m.install_flat_reference(reference)
    p={'schema_version':1,'serial':args.serial,'side':'left','operator':'local','weights_hash':m.weights_hash,'geometry_revision':GEOMETRY_REVISION,'calibrated':False,
       'dimension_mode':DIMENSION_MODE,'dimension_parameters':m.dimension_parameters.tolist(),'penetration_tolerance_m':.005,'shape':m.shape.tolist(),
       'material':dict(DEFAULT_MATERIAL),'bone_reference':m.bone_reference.tolist(),'raw_rotation':m.raw_rotation.tolist(),'raw_translation':m.raw_translation.tolist(),
       'mount_pos':m.mount_pos.tolist(),'mount_rot':m.mount_rot.tolist(),'sensor_joint_position':m.sensor_offsets().tolist(),'flat_reference':m.flat_reference,
       'observation_space':'raw_emf_sensor','calibration_fingerprint':reference['calibration_fingerprint'],'mounting_calibrated':False,'binding_overrides':{}}
    if (home/'current.json').exists():p=json.loads((home/'current.json').read_text());restore(m,p)
    if args.initial_profile:p=json.loads(args.initial_profile.read_text());restore(m,p)
    if (home/'fit-settings.json').exists():p['penetration_tolerance_m']=json.loads((home/'fit-settings.json').read_text())['penetration_tolerance_m']
    active_dimensions=m.dimension_summary();active_geometry=m.shape_version
    contact=ContactEngine(folder);contact.bind(args.serial,'left');mapping=contact.mapping()
    if args.refit or args.publish_candidate:
        (folder/'debug').mkdir(exist_ok=True);last=[-1]
        def progress(v):
            bucket=int(v.get('progress',0)*20)
            if bucket!=last[0]:print(json.dumps(v,ensure_ascii=False),flush=True);last[0]=bucket
        if args.publish_candidate:
            candidate=json.loads(args.publish_candidate.read_text());report=candidate['report']
            source_hash=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            if candidate['serial']!=args.serial or candidate['weights_hash']!=m.weights_hash or report.get('source_capture_hash')!=source_hash or report.get('validation_pose_source')!='runtime_emf_only':raise ValueError('候选身份、采集摘要或独立验证来源不匹配')
        else:
            candidate,report=fit_capture(DEFAULT_MANO,p,mapping,rows,progress,threading.Event(),max_seconds=args.seconds)
            candidate['report']=report
        write(folder/'debug'/'sensor-only-v9-candidate.json',candidate)
        # Do not overwrite a new capture or a concurrent user's candidate.
        latest=json.loads(capture_path.read_bytes())
        if latest['rows']!=capture['rows']:raise RuntimeError('采集原始帧已改变；私有结果保留，未发布候选')
        if candidate_path.exists() and candidate_path.read_bytes()!=original_candidate:raise RuntimeError('候选已被其他操作更新；私有结果保留')
        if original_candidate:(home/f'candidate-before-v9-{int(time.time())}.json').write_bytes(original_candidate)
        write(candidate_path,candidate)
        np.savez(home/'candidate.npz',dimension_parameters=candidate['dimension_parameters'],shape=candidate['shape'],bone_reference=candidate['bone_reference'],sensor_joint_position=candidate['sensor_joint_position'],mount_pos=candidate['mount_pos'],mount_rot=candidate['mount_rot'],raw_rotation=candidate['raw_rotation'],raw_translation=candidate['raw_translation'])
    candidate_dimensions={}
    if candidate:
        restore(m,candidate);candidate_dimensions=m.dimension_summary()
    train,valid=choose_rows(rows)
    audit={'schema_version':1,'created_at_ms':time.time()*1000,'serial':args.serial,'side':'left','fit_revision':candidate.get('report',{}).get('fit_revision') if candidate else None,'available_fit_revision':FIT_REVISION,
        'capture_frames':len(rows),'capture_complete':len(capture.get('completed_groups',[]))==10,'source_capture_file_sha256':hashlib.sha256(source_bytes).hexdigest(),
        'dimensions_applied':bool(p.get('calibrated')),'applied_revision':p.get('revision'),'active_geometry_version':active_geometry,'active_dimension_parameters':p['dimension_parameters'],
        'active_dimensions_mm':active_dimensions,'candidate_dimensions_mm':candidate_dimensions,'candidate_report':candidate.get('report') if candidate else None,
        'training_frames':len(train),'selected_holdout_frames':len(valid),'training_approach_frames':sum(r.get('load_role')=='approach' for r in train),'holdout_approach_frames':sum(r.get('load_role')=='approach' for r in valid),
        'runtime_contact_closure':False,'penetration_policy':'soft_penalty_only','thickness_identification':'contact_and_prior_only'}
    write(home/'calibration-audit.json',audit);write(folder/'debug'/'calibration-audit-v9.json',audit)
    lines=['# Moxian MANO 标定审计','',f'设备：{args.serial}，左手。尺寸已应用：{audit["dimensions_applied"]}。',
           f'采集 {len(rows)} 帧，训练 {len(train)} 帧（接近/释放 {audit["training_approach_frames"]}），留出 {len(valid)} 帧（接近/释放 {audit["holdout_approach_frames"]}）。',
           '', '长度为 MCP 到表面末端的链长；宽度/厚度为同一近节网格截面尺寸，均为模型值。厚度不能仅由 EMF 定位辨识。','',
           '| 手指 | 当前长度 mm | 候选长度 mm | 当前宽×厚 mm | 候选宽×厚 mm |','|---|---:|---:|---:|---:|']
    for f in FINGERS:
        a=active_dimensions[f];b=candidate_dimensions.get(f,a);lines.append(f'| {f} | {a["surface_length_mm"]:.2f} | {b["surface_length_mm"]:.2f} | {a["proximal_width_mm"]:.2f}×{a["proximal_depth_mm"]:.2f} | {b["proximal_width_mm"]:.2f}×{b["proximal_depth_mm"]:.2f} |')
    if candidate:
        r=candidate['report'];lines+=['',f'候选通过：{r["accepted"]}。{r["reason"]}',f'留出 EMF 定位残差：{r["baseline_position_rmse_mm"]:.2f} → {r["position_rmse_mm"]:.2f} mm。',f'非接触穿透：{r["noncontact_penetration_mm"]:.2f} mm，采用软惩罚，无 5 mm 硬性拒绝门槛。','', '本审计脚本只生成候选；网页 worker 的拟合通过后自动应用。当前尺寸和候选尺寸分别报告。']
    if candidate and 'together_gap_mm' in candidate['report']:
        r=candidate['report'];lines+=['',f'并拢侧面间隙：{r["baseline_together_gap_mm"]:.2f} → {r["together_gap_mm"]:.2f} mm。',f'相邻侧面分组：{r["together_pair_gaps_mm"]}。',f'接近／释放残差：{r["baseline_approach_sensor_rmse_mm"]:.2f} → {r["approach_sensor_rmse_mm"]:.2f} mm。',f'质量提示：{r.get("quality_warnings",[])}。','','| 手指 | 标定前长度 → 新长度 mm | 标定前宽×厚 → 新宽×厚 mm |','|---|---:|---:|']
        for f in FINGERS:
            a=r['dimensions_before_mm'][f];b=r['dimensions_after_mm'][f];lines.append(f'| {f} | {a["surface_length_mm"]:.2f} → {b["surface_length_mm"]:.2f} | {a["proximal_width_mm"]:.2f}×{a["proximal_depth_mm"]:.2f} → {b["proximal_width_mm"]:.2f}×{b["proximal_depth_mm"]:.2f} |')
    (folder/'debug'/'calibration-audit-v9.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'audit':str(home/'calibration-audit.json'),'accepted':candidate['report']['accepted'] if candidate else None,'applied':audit['dimensions_applied']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
