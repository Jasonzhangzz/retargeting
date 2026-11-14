"""Validate source IK recovery and bounded MANO control on a frozen capture.

This does not fit shared parameters or apply a calibration. Use the paired
source solver replay JSONs and the matching raw shared capture, then run with
.venv-contact/bin/python. All source holdout frames are replayed twice.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from moxian_web.hand_model import DEFAULT_MANO
from moxian_web.mano_runtime import ManoSurface
from moxian_web.mano_objective import SOLVER_REVISION
from scripts.calibrate_wuji_mano import CHAINS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('before','after','capture','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    before=json.loads(a.before.read_text());after=json.loads(a.after.read_text())
    if before['source_sha']!=after['source_sha']:raise ValueError('Capture hashes differ')
    raw=json.loads(a.capture.read_text());lookup={r['timestamp_ms']:r for r in raw['rows']}
    selected=[r for r in after['rows'] if r['input']['partition']=='holdout']
    side=selected[0]['input']['hand_side'];m=ManoSurface(DEFAULT_MANO.with_name('MANO_'+side.upper()+'.pkl'),side=side);m.enable_official_fk()
    m.build_binding(lookup[selected[0]['input']['timestamp_ms']]['mapping'])
    frames=[{**copy.deepcopy(lookup[r['input']['timestamp_ms']]),'skeleton':r['pose'],'skeleton_targets':r['targets']} for r in selected]
    records={};meshes=[];poses=[]
    for mode,tactile in (('static',False),('runtime',True)):
        m.contact_pose_enabled=tactile;m.tracking_time=None;m._mano_solver_previous=None;m._mano_solver_cache=None
        results=[];vs=[];qs=[]
        for row in frames:
            v,*_=m.track(row);vs.append(v);qs.append(m.last_pose.copy())
            error=Rotation.from_matrix(Rotation.from_rotvec(m.raw_official_pose).as_matrix().transpose(0,2,1)@Rotation.from_rotvec(m.last_pose).as_matrix()).magnitude()
            results.append(dict(timestamp_ms=row['timestamp_ms'],stage=row['stage'],solver=copy.deepcopy(m.solver_report),
                                thumb_local_rotation_change_deg=np.rad2deg(error[12:]).tolist(),contacts=copy.deepcopy(m.contact_correction)))
        records[mode]=results;meshes.append(vs);poses.append(qs)
    np.savez_compressed(a.output/'holdout-meshes.npz',vertices_m=np.array(meshes),pose_axis_angle=np.array(poses),faces=m.faces,
                        modes=['static','runtime'],stages=[r['stage'] for r in frames],timestamps_ms=[r['timestamp_ms'] for r in frames])
    import trimesh
    for stage in dict.fromkeys(r['stage'] for r in frames):
        ids=[i for i,r in enumerate(frames) if r['stage']==stage];i=ids[len(ids)//2]
        trimesh.Trimesh(meshes[1][i],m.faces,process=False).export(a.output/(stage+'-runtime.ply'))
    source={}
    for label,document in (('before',before),('after',after)):
        rows=[r for r in document['rows'] if r['input']['partition']=='holdout']
        source[label]=dict(position_rmse_mm=float(np.sqrt(np.mean([[f['position_error_mm']**2 for f in r['ik_diagnostics']['fingers']] for r in rows]))),
            stages={st:{f:float(np.mean([r['ik_diagnostics']['fingers'][i]['position_error_mm'] for r in rows if r['input']['stage']==st]))
                       for i,f in enumerate(('thumb','index','middle','ring','pinky'))} for st in dict.fromkeys(r['input']['stage'] for r in rows)})
    report=dict(revision=SOLVER_REVISION,source_capture_sha256=after['source_sha'],capture_file_sha256=hashlib.sha256(a.capture.read_bytes()).hexdigest(),
                holdout_frames=len(frames),shape_fitted=False,calibration_applied=False,source=source,paths=records,
                summary={mode:dict(max_penetration_mm=max(r['solver']['collision']['max_penetration_m']*1000 for r in rows),
                                   physical_constraint_failures=sum(not r['solver']['constraints_satisfied'] for r in rows),
                                   source_tracking_failures=sum(not r['solver']['source_tracking_satisfied'] for r in rows),
                                   latency_p50_ms=float(np.median([r['solver']['elapsed_ms'] for r in rows])),
                                   latency_p95_ms=float(np.percentile([r['solver']['elapsed_ms'] for r in rows],95)),
                                   max_thumb_local_rotation_change_deg=max(max(r['thumb_local_rotation_change_deg']) for r in rows)) for mode,rows in records.items()})
    (a.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    lines=['# 拇指控制与小指卡弯回放','',f'同源八动作，各动作后半留出帧，共 {len(frames)} 帧；共享参数未拟合，标定未应用。',
        '',f'来源骨架全指定位 RMSE：{source["before"]["position_rmse_mm"]:.2f} → {source["after"]["position_rmse_mm"]:.2f} mm。',
        '','| 动作 | 小指误差修复前 / 后 (mm) | 拇指上游误差 (mm) |','|---|---:|---:|']
    for stage,values in source['after']['stages'].items():
        lines.append(f'| {stage} | {source["before"]["stages"][stage]["pinky"]:.2f} / {values["pinky"]:.2f} | {values["thumb"]:.2f} |')
    for mode,v in report['summary'].items():
        lines.extend(['',f'{mode}：最大穿透 {v["max_penetration_mm"]:.2f} mm；物理约束超限 {v["physical_constraint_failures"]}/{len(frames)}，来源观测超限 {v["source_tracking_failures"]}/{len(frames)}；耗时 p50/p95 {v["latency_p50_ms"]:.1f}/{v["latency_p95_ms"]:.1f} ms。'])
    lines.extend(['','拇指固定绑定按左右手各自 tip/distal 变换推导；六个来源关节到 MANO 的同帧初值及梯度由双侧回归测试核验。实时只能在当前帧输入周围作小幅修正，不能将整手张开以换取较小穿透。',
                  '','弯曲动作仍有不可忽略的来源目标／骨架残差和碰撞冲突，不能将本报告视为八动作标定通过或实物精度验收。原始数据及标定文件保留。'])
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':main()
