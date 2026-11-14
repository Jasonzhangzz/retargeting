#!/usr/bin/env python3
"""Paired real-EMF windows: pose-only SLSQP, tactile SLSQP and unchanged CEM.

Evaluation uses the SAME Moxian-derived material witnesses for every backend.
The source IK is cold-started at each sampled window; no gesture labels enter
either IK or retargeting. Stage labels only balance evaluation windows.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from moxian_web.moxian_replay import MoxianReplayReconstructor
from moxian_web.wuji_contact_cost import ContactObjective, ContactSLSQPSettings
from moxian_web.wuji_contact_slsqp import ContactSLSQPRetarget
from moxian_web.wuji_backends import create_retarget


def metrics(evaluator, frame, output):
    source = evaluator.source.read(frame); anchors = evaluator.material_contacts(source)
    evaluator.fk.set_contacts(anchors)
    geometry = evaluator.fk.forward(output['qpos'][6:])
    rotation = np.asarray(output['wrist_rotation']); origin = np.asarray(output['wrist_position_m'])
    c = geometry[2] @ rotation.T+origin; n = geometry[3] @ rotation.T
    target = np.asarray([a['position_m'] for a in anchors]).reshape(-1, 3)
    normal = np.asarray([a['direction'] for a in anchors]).reshape(-1, 3)
    errors = np.linalg.norm(c-target, axis=-1)*1000
    angular = np.degrees(np.arccos(np.clip((n*normal).sum(-1), -1, 1)))
    edges = source['edges']
    edge_errors = (np.linalg.norm((c[edges[:, 1]]-c[edges[:, 0]])-(target[edges[:, 1]]-target[edges[:, 0]]), axis=-1)*1000
                   if len(edges) else np.array([]))
    lap_error = None
    if len(edges):
        adjacency = np.zeros((len(anchors), len(anchors)))
        adjacency[edges[:, 0], edges[:, 1]] = 1; adjacency[edges[:, 1], edges[:, 0]] = 1
        degree = adjacency.sum(1); valid = degree > 0
        laplacian = np.diag(valid.astype(float))-adjacency/np.maximum(degree[:, None], 1.)
        differential_error = (laplacian @ (c-target))[valid]
        lap_error = float(np.sqrt(np.mean(np.sum(differential_error**2, axis=1)))*1000)
    objective = ContactObjective(source['joints'], target, normal, edges, np.asarray(output['qpos'][6:]),
        contact_links=[a['semantic_link'] for a in anchors])
    world_points = geometry[0] @ rotation.T+origin
    nodes = np.concatenate((world_points, c)) if objective.graph_enabled else world_points
    residual = objective.laplacian @ nodes-objective.differential
    unified_rms = float(np.sqrt(np.sum(objective.row_weights[:, None]*residual**2)/objective.row_weights.sum())*1000)
    return dict(contact_rmse_mm=float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
        normal_mean_deg=float(angular.mean()) if len(angular) else None,
        edge_rmse_mm=float(np.sqrt(np.mean(edge_errors**2))) if len(edge_errors) else None,
        contact_laplacian_rmse_mm=lap_error,
        unified_laplacian_rmse_mm=unified_rms,
        active_patches=len(anchors), edges=len(edges))


def run(capture, source_model, output, *, per_stage=5, device='cuda:0', include_cem=True):
    if per_stage < 1: raise ValueError('frames-per-stage must be positive')
    torch.set_num_threads(1)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    candidate = json.loads(Path(source_model).read_text())
    (output/'source-model.json').write_text(json.dumps(candidate, ensure_ascii=False, allow_nan=False)+'\n')
    raw = [json.loads(line) for line in Path(capture).open() if line.strip()]
    stages = list(dict.fromkeys(r.get('stage', 'unlabelled') for r in raw))
    reconstructor = MoxianReplayReconstructor(candidate); windows = []; topology = None
    for stage in stages:
        indices = [i for i, r in enumerate(raw) if r.get('stage', 'unlabelled') == stage]
        active = [i for i in indices if any((raw[i].get('contact') or {}).get('cell_contact', []))]
        middle = indices[len(indices)//2]
        centre = min(active, key=lambda i: abs(i-middle)) if active else middle
        selected = [i for i in indices if centre-per_stage//2 <= i < centre-per_stage//2+per_stage]
        reconstructor.ik.reset(); reconstructor.previous_stamp = None
        frames = [reconstructor.frame(raw[i]) for i in selected]
        topology = reconstructor.topology; windows.append((stage, frames))
    variants = ['slsqp_pose_only', 'slsqp_contact']+(['cem'] if include_cem else [])
    results = {}; all_metrics = []; joint_arrays = {}
    evaluator = ContactSLSQPRetarget(topology, output/'mappings', device)
    for variant in variants:
        print('Starting '+variant, flush=True)
        engine = create_retarget('cem' if variant == 'cem' else 'slsqp', topology, output/'mappings', device)
        measurements = []; joints = []
        for stage, frames in windows:
            engine.reset(); previous = None
            for frame in frames:
                began = time.perf_counter()
                result = engine.step(frame, contact_enabled=variant != 'slsqp_pose_only')
                if device.startswith('cuda'): torch.cuda.synchronize()
                wall = (time.perf_counter()-began)*1000
                measure = metrics(evaluator, frame, result)
                q = np.asarray(result['qpos'][6:]); dt = None if previous is None else (frame['timestamp_ms']-previous[0])/1000
                speed = float(np.linalg.norm(q-previous[1])/np.sqrt(20)/dt) if dt is not None and dt > 0 else None
                step = float(np.degrees(np.linalg.norm(q-previous[1])/np.sqrt(20))) if dt is not None else None
                previous = (frame['timestamp_ms'], q)
                measurements.append(dict(**measure, variant=variant, stage=stage, timestamp_ms=frame['timestamp_ms'],
                    feasible=result['solution_feasible'], elapsed_ms=wall, reported_ms=result['quality']['processing_ms'],
                    source_reconstruction_ms=frame['quality']['source_reconstruction_ms'],
                    joint_step_rms_deg=step, joint_speed_rms_rad_s=speed,
                    penetration_mm=result['collision']['self_mesh']['max_penetration_mm'],
                    native_metrics=result['quality'], solver=result['solver']))
                joints.append(q.tolist())
        def mean(key):
            values = [r[key] for r in measurements if r[key] is not None]
            return float(np.mean(values)) if values else None
        elapsed = np.array([r['elapsed_ms'] for r in measurements])
        results[variant] = dict(frames=len(measurements), contact_frames=sum(r['active_patches'] > 0 for r in measurements),
            **{key:mean(key) for key in ('contact_rmse_mm','normal_mean_deg','edge_rmse_mm','contact_laplacian_rmse_mm','unified_laplacian_rmse_mm','joint_step_rms_deg','joint_speed_rms_rad_s')},
            collision_pass_rate=float(np.mean([r['penetration_mm'] <= .5 for r in measurements])),
            feasible_rate=float(np.mean([r['feasible'] for r in measurements])),
            zero_gap_pass_rate=float(np.mean([r['penetration_mm'] <= .01 for r in measurements])),
            mean_ms=float(elapsed.mean()), p95_ms=float(np.percentile(elapsed,95)),
            warm_mean_ms=float(np.mean(elapsed[1:])) if len(elapsed)>1 else None,
            raw_emf_to_output_mean_ms=float(np.mean([r['elapsed_ms']+r['source_reconstruction_ms'] for r in measurements])),
            wrist_policy='optimized_26d' if variant == 'cem' else 'fixed_source_20d')
        all_metrics.extend(measurements); joint_arrays[variant] = np.asarray(joints)
        print(variant, json.dumps(results[variant]), flush=True)
    report = dict(capture=str(capture), capture_sha256=hashlib.sha256(Path(capture).read_bytes()).hexdigest(),
        source_model=str(source_model), source_model_snapshot='source-model.json', source_model_hash=candidate['model_hash'], source_surface='Moxian original SKN',
        robot_asset_sha256=evaluator.surface.asset_hash,
        implementation_sha256={name:hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
            for name in ('moxian_web/wuji_contact_cost.py','moxian_web/wuji_contact_source.py',
                         'moxian_web/wuji_contact_slsqp.py','moxian_web/wuji_contact_collision.py',
                         'moxian_web/moxian_replay.py','scripts/benchmark_moxian_wuji_contact.py')},
        source_reconstruction='raw_EMF_compiled_Studio_style_QP_cold_start_at_each_window_no_gesture_targets',
        evaluation='identical_fixed_Moxian_to_robot_material_witnesses; cold costs and final audit included',
        graph='single augmented skeleton/material graph with semantic attachments and boosted cross-link contact edges; all palm tiles share palm',
        device=device, settings=ContactSLSQPSettings().payload(), results=results)
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    with (output/'metrics.jsonl').open('w') as stream:
        for row in all_metrics: stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    np.savez_compressed(output/'joints.npz', **joint_arrays,
        timestamp_ms=[f['timestamp_ms'] for _, frames in windows for f in frames],
        stages=[stage for stage, frames in windows for _ in frames])
    lines = ['# Moxian glove → Wuji2 触觉接触图比较', '',
        '所有后端使用同一份 Moxian 原生表面和同一组材料点评估。',
        'CEM 允许腕部移动；SLSQP 固定源腕部，因此表格同时体现算法与自由度差异。', '',
        '新 SLSQP 将骨架和材料点合入一张图，计算单个高权重混合 Laplacian；接触子图、位置和法向另作诊断。统一图误差在所有后端使用同一完整图评估。', '',
        '| 后端 | 统一图 Laplacian mm | 接触子图 mm（诊断） | 边向量 RMS mm | 位置 mm（诊断） | 法向 °（诊断） | 零间距通过率 | 平均 / P95 ms |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, r in results.items():
        fmt = lambda key: '—' if r[key] is None else f'{r[key]:.2f}'
        lines.append(f'| {name} | {fmt("unified_laplacian_rmse_mm")} | {fmt("contact_laplacian_rmse_mm")} | {fmt("edge_rmse_mm")} | {fmt("contact_rmse_mm")} | {fmt("normal_mean_deg")} | {r["zero_gap_pass_rate"]:.1%} | {r["mean_ms"]:.1f} / {r["p95_ms"]:.1f} |')
    lines.extend(['', '耗时包含最终稠密碰撞验收和首次冷启动；原始 EMF 重建耗时另见 JSON。',
        '关节步长/速度是短窗口的运动连续性指标，不等同于静态噪声测量。',
        '同时报告不可行候选的几何误差；较低误差不代表通过碰撞验收。'])
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True); parser.add_argument('--source-model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--frames-per-stage', type=int, default=5)
    parser.add_argument('--device', default='cuda:0'); parser.add_argument('--no-cem', action='store_true')
    args = parser.parse_args()
    run(args.capture, args.source_model, args.output, per_stage=args.frames_per_stage, device=args.device, include_cem=not args.no_cem)
