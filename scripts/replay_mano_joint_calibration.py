#!/usr/bin/env python3
"""Re-fit retained eight-gesture data without touching live calibration files.

Exports complete reports and paired meshes for reviewing failed candidates too.
"""
import argparse
import copy
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moxian_web.hand_model import DEFAULT_MANO,DEFAULT_MATERIAL
from moxian_web.mano_runtime import ManoSurface
from moxian_web.mano_fit import fit_capture,holdout_rows
from moxian_web.official_mano import OFFICIAL_MANO_MODE,reference_from_frame
from scripts.calibrate_wuji_mano import CHAINS


def install(model,profile):
    model.set_shape(profile['shape']);model.build_binding(profile['mapping'])
    for name in ('raw_rotation','raw_translation','mount_pos','mount_rot'):
        setattr(model,name,np.array(profile[name]))
    model.sensor_joint_position=np.array(profile['sensor_joint_position']) if profile.get('sensor_joint_position') is not None else None
    model.flat_reference=copy.deepcopy(profile.get('flat_reference'))
    model.mounting_calibrated=bool(profile.get('mounting_calibrated',False))
    model.mounting_estimated=bool(profile.get('mounting_estimated',False))
    model.calibration_pose_offsets=np.array(profile.get('calibration_pose_offsets',np.zeros((15,3))),np.float32)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--device-home',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-seconds',type=float,default=300)
    parser.add_argument('--frozen-candidate',type=Path,help='Only validate these shared parameters; no fitting')
    parser.add_argument('--initial-candidate',type=Path,help='Resume a complete same-version training snapshot; never applies it')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    source=args.device_home/'fit-wrist-targets.json'
    if not source.exists():source=args.device_home/'shared-capture.json'
    document=json.loads(source.read_text());rows=document['rows'];side=rows[0]['hand_side']
    capture=args.device_home/'shared-capture.json'
    serial=document.get('serial') or (json.loads(capture.read_text()).get('serial') if capture.exists() else None) or 'offline'
    if not all(r.get('skeleton') for r in rows):raise ValueError('Replay raw captures through prepare_wrist_targets first')
    path=DEFAULT_MANO.with_name('MANO_'+side.upper()+'.pkl')
    model=ManoSurface(path,side=side);model.enable_official_fk();model.build_binding(rows[0]['mapping'])
    model.install_flat_reference(reference_from_frame(next(r for r in rows if r['stage']=='open'),model.weights_hash,serial,'retained-training-reference'),allow_coarse=True)
    profile=dict(schema_version=1,serial=serial,side=side,weights_hash=model.weights_hash,
        tracking_source='official_qnbot_skeleton_fk_v1',shape=model.shape.tolist(),dimension_mode=OFFICIAL_MANO_MODE,
        material=dict(DEFAULT_MATERIAL),raw_rotation=model.raw_rotation.tolist(),raw_translation=model.raw_translation.tolist(),
        mount_pos=model.mount_pos.tolist(),mount_rot=model.mount_rot.tolist(),sensor_joint_position=model.sensor_offsets().tolist(),
        flat_reference=model.flat_reference,mapping=rows[0]['mapping'],penetration_tolerance_m=.002,
        calibration_pose_offsets=np.zeros((15,3)).tolist())
    current=args.device_home/'current.json'
    if current.exists():profile.update(json.loads(current.read_text()))
    flat=args.device_home/'flat-pose-calibration.json'
    if flat.exists():profile['calibration_pose_offsets']=json.loads(flat.read_text())['calibration_pose_offsets']
    def progress(message):print(json.dumps(message,ensure_ascii=False),flush=True)
    frozen=json.loads(args.frozen_candidate.read_text()) if args.frozen_candidate else None
    initial=json.loads(args.initial_candidate.read_text()) if args.initial_candidate else None
    candidate,report=fit_capture(path,profile,rows[0]['mapping'],rows,progress,threading.Event(),max_seconds=args.max_seconds,frozen_candidate=frozen,initial_candidate=initial)
    candidate['report']=report
    (args.output/'candidate.json').write_text(json.dumps(candidate,ensure_ascii=False,indent=2,allow_nan=False))
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    valid=holdout_rows(rows);meshes=[];poses=[];roots=[]
    for new in (False,True):
        install(model,candidate if new else profile);model.solver_enabled=new;model.contact_pose_enabled=False;model.tracking_time=None;model._mano_solver_previous=None
        sequence=[];qs=[];rs=[]
        for row in valid:
            v,*_=model.track(row);sequence.append(v);qs.append(model.last_pose.copy());rs.append(model.last_root.copy())
        meshes.append(np.array(sequence));poses.append(np.array(qs));roots.append(np.array(rs))
    np.savez_compressed(args.output/'heldout-meshes.npz',vertices_m=np.array(meshes),pose_axis_angle=np.array(poses),
        roots=np.array(roots),faces=model.faces,stages=np.array([r['stage'] for r in valid]),timestamps_ms=[r['timestamp_ms'] for r in valid])
    import trimesh
    for stage in dict.fromkeys(r['stage'] for r in valid):
        ids=[i for i,r in enumerate(valid) if r['stage']==stage];i=ids[len(ids)//2]
        for k,name in enumerate(('baseline','candidate')):
            trimesh.Trimesh(meshes[k][i],model.faces,process=False).export(args.output/f'{stage}-{name}.ply')
    summary=f'''# MANO 八动作离线重构验证

来源：`{source}`；手侧：{side}；训练 {report['selected_training_frames']} 帧，留出 {report['selected_validation_frames']} 帧。

- 候选通过：{report['accepted']}（未应用到实时服务）
- 留出有效观测 RMSE：{report['baseline_position_rmse_mm']:.3f} → {report['position_rmse_mm']:.3f} mm
- 对指材料点平均／最差间隙：{report['contact_gap_mm']:.3f}／{report['contact_max_gap_mm']:.3f} mm
- 静态／实时路径最大穿透：{report['max_penetration_mm']:.3f}／{report['runtime_max_penetration_mm']:.3f} mm
- 实时求解 p50／p95：{report['runtime_latency_ms']['p50']:.1f}／{report['runtime_latency_ms']['p95']:.1f} ms
- 结果：{report['reason']}

`report.json` 保存所有留出帧的碰撞、求解和逐动作指标。`heldout-meshes.npz` 第一轴为旧模型／新候选，长度单位为米；各动作另有成对 PLY 文件。离散顶点及面心碰撞查询不是连续相交的完整证明。
'''
    (args.output/'REPORT.md').write_text(summary)
    print(summary,flush=True)


if __name__=='__main__':main()
