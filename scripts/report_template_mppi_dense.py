#!/usr/bin/env python3
"""Report solved physical size grids and independently measured wrist continuity."""
import argparse,csv,json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retarget.common.results import write_json
from retarget.env.grasp.template_mppi_continuation import pose_change


def read(path):return json.loads(Path(path).read_text())


def report(root):
    root=Path(root).resolve();entries=read(root/'manifest.json')['cases'];plan=read(root/'continuation_plan.json')
    by_name={e['name']:e for e in entries};rows=[];families=[]
    for group in plan['groups']:
        names=[plan['nodes'][i]['name'] for i in group['indices']]
        cases=[by_name[n] for n in names]
        motions=[];previous=None
        for e in cases:
            case=(root/e['trajectory']).parent;d=read(case/'projection_force_closure.json')
            summary=read(root/e['summary']);q=np.load(case/'grasp.npz')['qpos']
            parent=read(root/e['continuation']) if e.get('continuation') else None
            row=dict(name=e['name'],template=e['template'],scale=e['primitive']['size_scale'],
                kind=e['primitive']['kind'],radius_mm=e['primitive']['radius_m']*1000,
                height_mm=e['primitive']['height_m']*1000,anchor=bool(e.get('continuation_anchor')),
                projected_closure=d['projected_closure'],patch_coverage=d['coverage'],
                fixed_anchor_coverage=d.get('template_point_coverage',d['coverage']),
                hand_object_depth_mm=d['max_hand_object_penetration_m']*1000,
                self_depth_mm=d['max_hand_hand_penetration_m']*1000,
                optimization_seconds=summary['optimization_seconds'],
                parent_case=parent['parent_case'] if parent else '',
                selected_attempt=parent['selected_attempt'] if parent else -1,
                object_translation_mm=parent['motion']['object_translation_m']*1000 if parent else 0,
                object_rotation_deg=parent['motion']['object_rotation_deg'] if parent else 0,
                max_joint_change_deg=parent['motion']['max_joint_change_deg'] if parent else 0,
                actual_response_max_n=d['max_response_normal_force_n'],nominal_preload_max_n=max(d['nominal_preloads_n']))
            rows.append(row)
            if previous is not None:
                m=pose_change(previous,q);motions.append(dict(left=cases[len(motions)]['name'],right=e['name'],**m))
            previous=q
        these=rows[-len(cases):]
        families.append(dict(template=group['template'],cases=len(cases),scales=group['scales'],
            projected_closed=sum(r['projected_closure'] for r in these),
            patch_80=sum(r['patch_coverage']>=.8-1e-7 for r in these),
            exact_pose_steps=sum(m['object_translation_m']<1e-7 and m['object_rotation_deg']<1e-4 for m in motions),
            max_adjacent_translation_mm=max((m['object_translation_m']*1000 for m in motions),default=0),
            max_adjacent_rotation_deg=max((m['object_rotation_deg'] for m in motions),default=0),
            max_adjacent_joint_deg=max((m['max_joint_change_deg'] for m in motions),default=0),motions=motions))
    all_motions=[m for f in families for m in f['motions']]
    jumps=[m for m in all_motions if m['object_translation_m']>.0030003 or m['object_rotation_deg']>3.0001]
    resources=read(root/'resources.json');audit=read(root/'independent_result_audit.json')
    result=dict(cases=len(rows),templates=len(families),new_solves=plan['recomputed'],anchors=plan['anchors'],
        subdivisions=plan['subdivisions'],passed=audit['passed'],
        projected_closed=sum(r['projected_closure'] for r in rows),patch_80=sum(r['patch_coverage']>=.8-1e-7 for r in rows),
        neighbor_pairs=len(all_motions),unresolved_large_object_jumps=jumps,
        exact_pose_neighbor_pairs=sum(f['exact_pose_steps'] for f in families),
        max_adjacent_translation_mm=max(m['object_translation_m']*1000 for m in all_motions),
        max_adjacent_rotation_deg=max(m['object_rotation_deg'] for m in all_motions),
        max_adjacent_joint_deg=max(m['max_joint_change_deg'] for m in all_motions),
        attempts={str(k):sum(r['selected_attempt']==k for r in rows) for k in (0,1,2)},
        wall_seconds=resources['wall_seconds'],peak_fraction_of_initial_free=resources['peak_fraction_of_initial_free'],
        native_evaluated=False,families=families)
    write_json(root/'dense_report.json',result)
    with (root/'dense_results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=['# 相邻尺寸连续求解结果','',
        f"{len(families)} 类、{len(rows)} 个尺寸；保留 {plan['anchors']} 个已审计的标准尺寸结果，重新求解 {plan['recomputed']} 格。",
        f"投影闭合 {result['projected_closed']}/{len(rows)}；patch 内部贴合率 ≥80%：{result['patch_80']}/{len(rows)}。",
        '投影闭合是虚拟接触下的力平衡证书；未运行原生动态物理评测。','',
        f"相邻物体位置最大变化 {result['max_adjacent_translation_mm']:.4f} mm，姿态最大变化 {result['max_adjacent_rotation_deg']:.4f}°。",
        f"{result['exact_pose_neighbor_pairs']}/{result['neighbor_pairs']} 对保持物体姿态不变；超过 3 mm / 3° 的跳变 {len(jumps)} 对。",
        f"批量耗时 {resources['wall_seconds']/60:.2f} 分钟；峰值使用初始可用显存的 {resources['peak_fraction_of_initial_free']:.1%}。",'',
        '| Type | 尺寸数 | 投影闭合 | patch ≥80% | 最大邻格位移 mm | 最大邻格转角 ° |','|---|---:|---:|---:|---:|---:|']
    for f in families:
        text.append(f"| {f['template']} | {f['cases']} | {f['projected_closed']} | {f['patch_80']} | {f['max_adjacent_translation_mm']:.4f} | {f['max_adjacent_rotation_deg']:.4f} |")
    (root/'dense_report.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='families'},ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('batch',type=Path);report(p.parse_args().batch)
