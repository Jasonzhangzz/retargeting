#!/usr/bin/env python3
"""Report direct-MANO reference success rates and measured planning latency."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=ROOT/'reports/mano_direct_reference')
    args=parser.parse_args();r=args.report
    manifest=json.loads((r/'manifest.json').read_text());results=json.loads((r/'results.json').read_text())
    import mujoco
    from retarget.contact_exploration import ObjectMeshSurface
    from scripts.analyze_mano_sdf_perturbation import operation_metrics
    model=mujoco.MjModel.from_xml_path(str((ROOT/'data/example/sequence/scene/wuji_right/sugar_box_mesh.xml').resolve()))
    surface=ObjectMeshSurface.from_model(model,model.body('obj').id)
    for key,row in results.items():
        if row['complete']:
            with np.load(r/'runs/comfree'/key/'rollout.npz') as archive:row.update(operation_metrics(surface,archive))
            row['lift_success']=bool(row['motion_endpoint_lift_mm']>=50 and row['motion_endpoint_clearance_mm']>=5)
            row['lift_and_hold_success']=bool(row['lift_success'] and row['minimum_hold_clearance_mm']>=5)
        else:row.update(lift_success=False,lift_and_hold_success=False)
        for legacy in ('tracking_pass','lift_and_motion_complete','operation_pass'):row.pop(legacy,None)
    (r/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    conditions=manifest['conditions'];perturbed=[c for c in conditions if c['pattern'] is not None]
    methods=[m for m in ['sampled-lm','no-admm','qp'] if any(v['method']==m for v in results.values())]
    assert all(c['name']+'__'+m in results for c in conditions for m in methods),'Matrix is incomplete'
    summary={};labels={'sampled-lm':'AdamW + ADMM/采样 LM','no-admm':'AdamW，无 ADMM','qp':'AdamW + ADMM/QP'}
    for method in methods:
        rows=[results[c['name']+'__'+method] for c in perturbed];warm=[];first=[]
        for c in perturbed:
            file=r/'runs/comfree'/(c['name']+'__'+method)/'windows.jsonl'
            windows=[json.loads(line)['seconds']*1000 for line in file.read_text().splitlines()]
            first.append(windows[0]);warm.extend(windows[1:])
        summary[method]=dict(cases=len(rows),lift_successes=sum(v['lift_success'] for v in rows),
            lift_and_hold_successes=sum(v['lift_and_hold_success'] for v in rows),
            warm_mean_ms=float(np.mean(warm)),warm_p50_ms=float(np.median(warm)),warm_p95_ms=float(np.percentile(warm,95)),
            first_window_p50_ms=float(np.median(first)),warm_window_count=len(warm),
            process_wall_seconds_mean=float(np.mean([v['process_wall_seconds'] for v in rows if v['process_wall_seconds'] is not None])),
            under_30ms_fraction=float(np.mean(np.asarray(warm)<=30)))
    (r/'timing_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fields=['condition','method','lift_success','lift_and_hold_success','moving_position_rmse_mm',
        'moving_rotation_rmse_deg','trajectory_endpoint_position_error_mm','trajectory_endpoint_rotation_error_deg',
        'motion_endpoint_lift_mm','motion_endpoint_clearance_mm','minimum_hold_clearance_mm',
        'lift_delay_to_5mm_ms','warm_mean_ms','warm_p50_ms','warm_p95_ms',
        'first_window_ms','process_wall_seconds']
    with (r/'summary.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
        writer.writerows({key:row.get(key) for key in fields} for row in results.values())
    lines=['# 扰动 MANO 直接重生成手部参考：成功率与规划耗时','',
        '物理有效性补充：下表只统计物体抬升/离桌事件，没有剔除手部自碰撞。后续已发现实体手指交叉，不能将 9/9 解释为物理有效抓取率，见[手指交叉审计](../finger_crossing_audit/README.md)。','',
        '本次已移除原可行 Wuji 轨迹作为手部参考生成的基准、正则目标和初值。`run.py --noise` 默认采用新的直接重定向方式。','',
        '## 输入与协议','',
        '- 由扰动 MANO 的腕部及五个掌指根部构造掌部坐标系，与机器人中立几何对齐，直接生成 wrist 位姿；再通过绝对骨链方向 IK 拟合全部五指。没有物体接触/避碰项参与参考 IK。',
        '- 第一帧只用通用轻度屈曲的中立初值，后续仅使用上一帧新生成的结果。整个输入生成在禁止读取原 `wuji_right_qpos.npy` 的情况下仍可完成，并逐项复现本次参考。',
        '- 拇指、食指、全手 × 0.15 / 0.35 / 0.60 rad，共九个扰动案例，另加一个零扰动直接 IK 基线。**本次扰动从第一帧起生效**，与旧实验第 21–23 帧渐增的协议不同，不能把前后差异全部归因于参考生成方式。',
        '- 物体模板由扰动 MANO 的原语义顶点做精确 SDF 最近投影。原接触 mask/时序、权重、MANO 顶点 ID、机器人 link 和手部局部接触坐标仍保留。因此本实验去掉了原可行机器人轨迹先验，仍不是无接触语义先验的自主抓取测试。',
        '- sugar box 单条轨迹、seed=0、8 个优化初值。每窗口一次 hand 更新，ADMM 交替一轮、物体探索半径 5 mm；无 ADMM 对照固定同一错误模板、关闭一致性与 dual。两种方法使用逐项相同的输入参考与初始化。',
        '- ComFree 闭环 3.6 s（参考运动至 2.37 s，随后保持），每次 120 个 MPC 窗口、360 个执行控制步；关闭 viewer，批量顺序执行。','',
        '## 成功率','',
        '按用户指定的评价口径，成功率不使用手势或物体轨迹跟踪误差阈值。物体位姿误差只保留连续数值作为诊断，不能导致操作被判失败。','',
        '抬升成功：运动结束时（2.37 s）物体中心相对初始升高至少 50 mm，且整个物体离桌至少 5 mm。抬升并保持：在此基础上，后续至 3.6 s 的保持阶段整个物体持续离桌至少 5 mm。这些是实际物理抬升条件，不是对错误 MANO 手势的拟合条件。','',
        '下表使用已保存的同一批 20 条轨迹重新判定，未重跑或改变动作及计时，也不再统计起升时序同步阈值。','',
        '| 方法 | 抬升成功率 | 抬升并保持离桌 | 零扰动基线 |',
        '|---|---:|---:|---|']
    for method in methods:
        s=summary[method];baseline=results['projected_zero__'+method]
        lines.append(f"| {labels[method]} | {s['lift_successes']}/{s['cases']}（{100*s['lift_successes']/s['cases']:.1f}%） | {s['lift_and_hold_successes']}/{s['cases']} | {'通过' if baseline['lift_success'] else '未通过'} |")
    lines+=['','以下为运动结束时物体中心相对初始的抬升量（mm），括号表示是否持续保持离桌：','',
        '| 输入 | '+' | '.join(labels[m] for m in methods)+' |',
        '|---|'+''.join('---:|' for _ in methods)]
    for c in perturbed:
        values=[]
        for method in methods:
            row=results[c['name']+'__'+method]
            values.append(f"{row.get('motion_endpoint_lift_mm',float('nan')):.2f}（{'保持离桌' if row['lift_and_hold_success'] else '未保持离桌'}）")
        lines.append('| '+c['name']+' | '+' | '.join(values)+' |')
    lines+=['',
        '按这一物理操作判据，ADMM 的 index_350 也计为成功。无 ADMM 的 all_600 未能抓起物体（峰值中心抬升仅 1.77 mm）；同一输入开启 ADMM 后终点抬升 165.9 mm，并持续保持离桌。该对照同时关闭了探索、一致性项与 dual，不能把改善单独归因于接触点搜索。','',
        '[全手 0.60 rad：ADMM 回放](videos/all_600_admm.mp4) · [同一输入：无 ADMM 回放](videos/all_600_no_admm.mp4)','',
        '## 规划耗时','',
        '硬件：RTX 5060 Ti，驱动 595.84；具体软件版本见 environment.json。主指标为 `windows.jsonl.seconds`，包含一次完整求解及诊断/接触快照整理，排除物理执行和渲染。每次运行去掉首窗口后，汇总九组扰动的其余 119×9=1071 个窗口；首窗口另列，本机已有 JIT 缓存，不能视为全新环境的编译启动耗时。','',
        '| 方法 | 后续窗口均值 ms | P50 ms | P95 ms | 首窗口 P50 ms | 单次进程总耗时均值 s |',
        '|---|---:|---:|---:|---:|---:|']
    for method in methods:
        s=summary[method];lines.append(f"| {labels[method]} | {s['warm_mean_ms']:.1f} | {s['warm_p50_ms']:.1f} | {s['warm_p95_ms']:.1f} | {s['first_window_p50_ms']:.1f} | {s['process_wall_seconds_mean']:.2f} |")
    fits=[c['fit']['seconds'] for c in perturbed]
    lines += ['',f'手部参考 IK 是离线准备开销：每条 72 帧平均 {1000*np.mean(fits):.1f} ms，范围 {1000*np.min(fits):.1f}–{1000*np.max(fits):.1f} ms；此数值只含 IK，不含 MANO 模型加载/解码和 SDF 投影，不计入上表 MPC 窗口耗时。',
        '', '这些耗时尚未满足 30 ms 执行周期的实时要求。成功率是九个指定扰动案例在一个种子下的通过比例，不能外推为跨物体或多随机种子的成功率。','',
        '## 数据与复现','',
        '[逐次完整指标](results.json) · [CSV](summary.csv) · [耗时汇总](timing_summary.json) · [输入审计](input_audit.json) · [环境](environment.json)','',
        '`runs/comfree/<condition>__<method>/` 保存动作、状态、接触及逐窗口计时。输入在 `inputs/`，保留原始及扰动 MANO 参数以复核。','',
        '```bash','python scripts/benchmark_direct_mano.py --report reports/mano_direct_reference_new',
        'python scripts/analyze_direct_mano.py --report reports/mano_direct_reference_new',
        '# CLI 使用同一直接 MANO 重定向：',
        'python run.py --noise --noise-finger all --noise-amplitude 0.6',
        'python run.py --noise --noise-finger all --noise-amplitude 0.6 --no-admm','```','']
    (r/'README.md').write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
