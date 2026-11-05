#!/usr/bin/env python3
"""Generate an auditable native Shadow result and timing report."""
from pathlib import Path
import argparse,collections,csv,json,statistics
import native_graspadmm_retest as runner


def run(root):
    read=lambda name:json.loads((root/name).read_text())
    manifest=read('MANIFEST.json');init=read('initialization.json');quality=read('quality_operation.json')
    evaluation=read('native_evaluation.json');diagnostics=read('diagnostics.json')
    latency=read('latency.json');throughput=read('throughput.json');consistency=read('consistency.json')
    runtime=read('runtime.json');phases=read('phase_diagnostic.json')
    physical={r['output']:r for r in evaluation['rows']}
    diag={r['input']:r for r in diagnostics['rows']};rows=[];failure_counts=collections.Counter()
    for i,r in enumerate(quality['rows']):
        d=diag[r['input']]
        if r.get('exception'):reason='operation_exception'
        elif not r['captured_final_state']:reason='missing_final_state'
        elif not r['native_operation_output']:
            forwards=[f for f in r['filters'] if 'check' not in f and not f['passed']]
            if forwards:
                failed=forwards[-1]
                checks=[f for f in r['filters'] if f.get('name')==failed['name'] and 'check' in f and not f['passed']]
                check=checks[-1]['check'].removeprefix('_').removesuffix('_filter') if checks else 'unspecified'
                reason='rejected_'+failed['name'].split()[-1]+'_'+check
            else:reason='missing_native_output'
        elif not physical.get(r['output'],{}).get('observed'):reason='evaluation_exception_or_missing'
        elif not physical[r['output']]['success']:reason='six_direction_failed'
        else:reason='native_gsr_pass'
        failure_counts[reason]+=1
        row=dict(d)
        row.update(index=i,scene=Path(r['input']).parent.name,input=r['input'],result=r.get('output'),
            final_state=str(root/'quality/cases'/f'{i:05d}'/'final_state.npz'),reason=reason,
            native_operation_pass=r['native_operation_output'],native_gsr_pass=reason=='native_gsr_pass',
            fallback_used=r['fallback_used'],operation_exception=r.get('exception'))
        rows.append(row)
    runner.write(root/'per_input_results.json',rows)
    columns=['index','scene','input','reason','fallback_used','native_operation_pass','native_gsr_pass',
        'slot_count','slot_distance_coverage','slot_normal_coverage',
        'slot_joint_coverage','required_groups_met','required_group_count','cln','cdc_mm','hand_object_depth_m','self_depth_m']
    with (root/'per_input_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    n=len(rows);ratio=lambda x:f'{x}/{n} ({x/n:.2%})' if n else '0/0（无输入）'
    lines=['# 原生 Shadow GraspADMM 复测','',
        f"范围：16 个固定物体—尺度组合，Shadow {runner.TYPE}，种子 20261104。此次使用原生模板重新初始化；与先前 MANO+LM 的 16 条姿态不是同初值对照。",'',
        f"源码：[{runner.COMMIT}](https://github.com/JYChen18/Dexonomy/tree/{runner.COMMIT})；Shadow 网格为官方 Menagerie 子模块 {runner.MENAGERIE}。源码、模型、场景与模板哈希保存在 MANIFEST.json 和 template_receipt.json，运行前后均检查。",
        '原生初始化、Clarabel QP、MuJoCo 虚拟力与碰撞、对偶更新、fallback、filter、squeeze、pregrasp 均直接调用上游源码。未使用 MANO、LM 或 GPU QP 替换精修。','',
        '## 输入覆盖与原生质量','',
        f"初始化输出 {init['samples']}/{manifest['max_initialization_outputs']} 条；有输出场景 {sum(v>0 for v in init['scene_counts'].values())}/16。初始化总 wall time {init['seconds']:.3f} s，包含场景/模型及首次 GPU 准备。初始化预算：{manifest['init_budget']}。",
        '官方初始化过滤会保留/填充候选以满足其最小数量规则，此处未改动；输出数不代表通过后续物理验收数。','',
        '| 指标 | 全部送入 ADMM 的初始化输出为分母 |','|---|---:|',
        f"| 原生 grasp + pregrasp filters 与输出 | {ratio(evaluation['native_operation_outputs'])} |",
        f"| 原生六方向 GSR | {ratio(evaluation['native_gsr_passed'])} |",
        f"| fallback 使用 | {ratio(sum(r['fallback_used'] for r in rows))} |",
        f"| 存在可取得的最终状态 | {ratio(diagnostics['valid_final_states'])} |",
        f"| 原生必要接触组全部满足 | {ratio(diagnostics['required_groups_passed'])} |",'',
        '## 独立接触几何诊断','',
        '| 指标 | 通过数 |','|---|---:|',
        f"| 槽位距离覆盖 ≥80% | {ratio(diagnostics['distance_passed'])} |",
        f"| 槽位法向覆盖 ≥80% | {ratio(diagnostics['normal_passed'])} |",
        f"| 槽位距离与法向联合覆盖 ≥80% | {ratio(diagnostics['slot_joint_passed'])} |",
        f"| 联合覆盖 + 原生无穿透 | {ratio(diagnostics['geometry_joint_passed'])} |",'',
        '槽位覆盖分母为各条原生 Shadow 更新后接触槽数，距离为 ±2 mm、法向为30°，并非 MANO 的固定四槽。几何法向使用原生手槽局部法向随 link 变换，物面为最近三角形插值顶点法向。物面目标与真实手点距离逐条报告，目标闭合不等于手已贴合。fallback 的 ADMM 目标另有标记。','',
        '| 场景 | 初始化数 | 原生输出 | GSR | 联合槽位覆盖 |','|---|---:|---:|---:|---:|---:|']
    for scene,count in sorted(init['scene_counts'].items()):
        group=[r for r in rows if r['scene']==scene]
        lines.append(f"| {scene} | {count} | {sum(r['native_operation_pass'] for r in group)} | {sum(r['native_gsr_pass'] for r in group)} | {sum(r['slot_joint_pass'] for r in group)} |")
    contacts=[r for r in rows if r.get('cln',0)>0]
    valid=[r for r in rows if r['valid_final_state']]
    lines+=['','失败分类：'+', '.join(f'{k}={v}' for k,v in sorted(failure_counts.items()))+'。','',
        f"可评估接触槽总数 {diagnostics['pooled_slots']}，距离/法向/联合满足数分别为 {diagnostics['pooled_distance']}/{diagnostics['pooled_normals']}/{diagnostics['pooled_joint']}；缺失状态另计失败，未加入槽位分母。",
        f"有最终几何状态 {len(valid)} 条；有原生接触 {len(contacts)} 条。CLN 均值（有效最终状态分母）{statistics.mean(r.get('cln',0) for r in valid) if valid else None}；CDC 均值 mm（有接触分母）{statistics.mean(r['cdc_mm'] for r in contacts) if contacts else None}；手物/自穿透最大值 m 为 {max((r['hand_object_depth_m'] for r in valid),default=None)}/{max((r['self_depth_m'] for r in valid),default=None)}。",'',
        '## 完整 operation 延迟与吞吐','',
        '以下直接运行原函数，未启用 profiler 或返回观察器；每轮独立输出目录，包含完整 operation 的模型/场景准备、求解、过滤和文件保存。失败尝试也计时。每 worker 的 BLAS/OpenMP 线程数为1。','']
    measured=[r['operation_seconds'] for r in latency['rows'] if not r['warmup']]
    lines.append(f"固定首条输入，预热1次、实测5次：中位数 {statistics.median(measured):.6f} s，范围 [{min(measured):.6f}, {max(measured):.6f}] s。")
    lines+=['','| Workers | 固定输入数 | 三轮 wall time / s | 吞吐中位数 / 条·s⁻¹ | 摊销秒 / pose |','|---:|---:|---|---:|---:|']
    for workers in (1,32):
        group=[r for r in throughput['records'] if r['workers']==workers]
        rates=[r['samples_per_second'] for r in group]
        seconds=[r['seconds'] for r in group]
        lines.append(f"| {workers} | {group[0]['samples']} | {', '.join(f'{s:.6f}' for s in seconds)} | {statistics.median(rates):.4f} | {statistics.median(seconds)/group[0]['samples']:.6f} |")
    lines+=['',f"测速中保存输出/最终成功 pose 不一致：{len(throughput['different_final_outputs'])} 条。独立非测速观察验证（包含失败最终状态和原生判定）1 worker 对32 workers不一致：{len(consistency['different_final_states_or_native_judgments'])} 条。差异清单原样保留，未挑选较快结果。",'',
        f"质量采集 operation 总 wall time {quality['wall_seconds']:.3f} s（含观察器开销）；原生六方向评测 {evaluation['wall_seconds']:.3f} s；独立几何诊断 {diagnostics['geometry_seconds']:.3f} s。",'',
        '独立分段诊断使用返回/调用事件，有测量开销，不混入上表。独占分段秒数：'+json.dumps(phases['exclusive_seconds'],ensure_ascii=False)+'。','',
        '[论文 v2 §IV-F](https://arxiv.org/html/2603.13832v2#S4.SS6) 报告约19条/秒，此处只与多进程吞吐比较；其倒数不作为单条独立延迟。硬件、物体尺度、抓型与外部负载均会影响结果。','',
        '## 运行来源与回放','',
        '独立环境：'+runtime['python']+'；版本：'+json.dumps(runtime['versions'],ensure_ascii=False)+'。',
        f"CPU affinity 可用 {len(runtime['cpu_affinity'])} 核。GPU 初始化前后负载与 worker 恢复收据保存在 initialization_resources.json。CPU 精修期间未挂起生产 GPU worker。四卡 CEM 配置和结果未切换。",'',
        '逐输入结果：per_input_results.json / .csv；原生成功结果：quality/grasp；全部可取得的最后状态：quality/cases/*/final_state.npz；每条诊断接触：diagnostic_contacts.npz。代表性索引见 representatives.json。',
        '此次结论仅适用于这16个物体—尺度组合和type7，不代表整篇论文的总体性能复现。','']
    (root/'report.md').write_text('\n'.join(lines))
    representatives={}
    for row in rows:
        key=row['reason']
        if key not in representatives:representatives[key]=row
    runner.write(root/'representatives.json',representatives)
    runner.write(root/'summary.json',dict(complete=True,scene_count=16,inputs=n,initialization_outputs=init['samples'],
        native_outputs=evaluation['native_operation_outputs'],gsr_passed=evaluation['native_gsr_passed'],
        slot_joint_passed=diagnostics['slot_joint_passed'],
        latency_median_seconds=statistics.median(measured),failures=dict(failure_counts),
        worker_mode_differences=len(consistency['different_final_states_or_native_judgments']),
        report_sha256=runner.sha(root/'report.md')))
    return root/'report.md'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();print(run(a.root.resolve()))
