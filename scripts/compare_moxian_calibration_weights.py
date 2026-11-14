#!/usr/bin/env python3
"""Paired refits of retained raw calibration; never publishes a live profile."""
import argparse
import hashlib
import json
import sys
import threading
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moxian_web.hand_model import DEFAULT_MANO,DEFAULT_MATERIAL
from moxian_web.mano_runtime import ManoSurface
from moxian_web.mano_fit import fit_capture
from moxian_web.mano_objective import SolverSettings
from moxian_web.official_mano import OFFICIAL_MANO_MODE,reference_from_frame


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--device-home',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=600.)
    parser.add_argument('--adam-steps',type=int,default=480)
    parser.add_argument('--lm-steps',type=int,default=8)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    source=args.device_home/'fit-wrist-targets.json'
    document=json.loads(source.read_text());rows=document['rows']
    capture=json.loads((args.device_home/'shared-capture.json').read_text())
    serial=capture['serial'];side=rows[0]['hand_side'];path=DEFAULT_MANO.with_name('MANO_'+side.upper()+'.pkl')
    watched=[args.device_home/n for n in ('current.json','candidate.json','shared-capture.json')]
    before={str(p):p.read_bytes() if p.exists() else None for p in watched}
    m=ManoSurface(path,side=side);m.enable_official_fk();mapping=rows[0]['mapping'];m.build_binding(mapping)
    m.install_flat_reference(reference_from_frame(next(r for r in rows if r['stage']=='open'),m.weights_hash,serial,'paired-weight-reference'),allow_coarse=True)
    profile=dict(schema_version=1,serial=serial,side=side,weights_hash=m.weights_hash,
        tracking_source='official_qnbot_skeleton_fk_v1',shape=m.shape.tolist(),dimension_mode=OFFICIAL_MANO_MODE,
        material=dict(DEFAULT_MATERIAL),raw_rotation=m.raw_rotation.tolist(),raw_translation=m.raw_translation.tolist(),
        mount_pos=m.mount_pos.tolist(),mount_rot=m.mount_rot.tolist(),sensor_joint_position=m.sensor_offsets().tolist(),
        flat_reference=m.flat_reference,mapping=mapping,penetration_tolerance_m=.002,calibration_pose_offsets=np.zeros((15,3)).tolist())
    reports={}
    for name,position,normal,observation in [('original',80.,10.,1.),('pinch_priority',320.,40.,2.)]:
        settings=SolverSettings(adam_steps=args.adam_steps,lm_steps=args.lm_steps,
            calibration_contact_position_weight=position,calibration_contact_normal_weight=normal,pinch_observation_multiplier=observation)
        last=[None]
        def progress(v):
            key=(v.get('phase'),int(v.get('progress',0)*10))
            if key!=last[0]:print(json.dumps(dict(experiment=name,**v),ensure_ascii=False),flush=True);last[0]=key
        candidate,report=fit_capture(path,profile,mapping,rows,progress,threading.Event(),max_seconds=args.seconds,solver_settings=settings)
        (args.output/(name+'.json')).write_text(json.dumps(dict(profile=candidate,report=report),ensure_ascii=False,allow_nan=False,indent=2))
        reports[name]={k:report[k] for k in ('accepted','position_rmse_mm','contact_gap_mm','contact_max_gap_mm','max_penetration_mm','runtime_max_penetration_mm','trajectory_after','rejection_reasons','seconds')}
        reports[name]['solver']=dict(loss_weights=report['solver']['loss_weights'],budget_exhausted=report['solver']['budget_exhausted'],cost_terms=report['solver']['cost_terms'])
    untouched=all((p.read_bytes() if p.exists() else None)==before[str(p)] for p in watched)
    summary=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 raw_frames=len(rows),live_files_unchanged=untouched,validation_scope='static_eight_gestures_only',experiments=reports)
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,allow_nan=False,indent=2))
    lines=['# 保留八动作：对指权重对照','',f'来源：{source}；{len(rows)} 帧；实时文件未改变：{untouched}。',
           '', '| 条件 | 留出 EMF mm | 对指平均／最大 mm | 穿透 mm | 通过 |','|---|---:|---:|---:|---|']
    for name,r in reports.items():lines.append(f'| {name} | {r["position_rmse_mm"]:.2f} | {r["contact_gap_mm"]:.2f} / {r["contact_max_gap_mm"]:.2f} | {r["max_penetration_mm"]:.2f} | {r["accepted"]} |')
    lines+=['','本对照只验证旧静态动作；动态精度、完整重复轮次和新固定几何需重新采集验证。']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
