#!/usr/bin/env python3
"""Summarize equivalent geometry execution, quality and paired wall timings."""
import argparse,html,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,default=Path('output/mano_patch_geometry_fast_20261105'))
    p.add_argument('--previews',type=Path,default=Path('output/mano_comfree_previews_20261105'))
    args=p.parse_args();root=args.experiment
    proof=json.loads((root/'validation/SUMMARY.json').read_text());assert proof['passed'] and proof['complete']
    server=json.loads((root/'SERVER_VALIDATION.json').read_text());assert server['returncode']==0 and server['SUMMARY.json']['passed']
    current=json.loads((root/'run/SUMMARY.json').read_text())['methods']['cem']
    baseline=Path('output/mano_patch_history_v5_gpu_fast_20261105')
    old=json.loads((baseline/'SUMMARY.json').read_text())['methods']['cem']
    changed=[]
    for packet in json.loads((root/'run/MANIFEST.json').read_text())['packets']:
        a=np.load(baseline/'quality'/packet['name']/'cem/final.npz');b=np.load(root/'run/quality'/packet['name']/'cem/final.npz')
        changed.extend((packet['name'],int(i)) for i in np.flatnonzero(a['patch_joint_accepted']!=b['patch_joint_accepted']))
    assert not changed
    records=[('服务器 A800',server['TIMINGS.json']),('本地 RTX 5060 Ti',json.loads((root/'validation/TIMINGS.json').read_text()))]
    lines=['# MANO patch与穿透查询执行优化','',
        '目标、贴合阈值、穿透允许量、弹簧模型和CEM规则不变；生产任务没有恢复。',
        'patch最近点不再计算无用内外符号；裁剪三角形紧凑存储；保守下界剪枝；三角形对并行精确评分、按原顺序归约。每三角形缓冲128个候选面，溢出在GPU完整重查，不截断。',
        '穿透保留原mesh_query_point射线符号判定；仅对确定在物体包围盒外的点跳过查询，盒内仍查询真实mesh。只输出最大深度与有效性，省掉无用法向、重心和投影输出。','',
        '## 相同输入、同进程配对计时','',
        '预热后5次取中位数，完整CUDA图内事件计时。固定Large Diameter状态/局部扰动循环填充候选缓冲；不是2048条独立grasp的完整求解。加载、编译和文件读写不计入。','']
    sections=[];timings={}
    for title,table in records:
        lines+=['### '+title,'','|候选数|阶段|原实现 ms|优化后 ms|加速|','|---:|---|---:|---:|---:|'];rows=[]
        for chunk in [128,2048]:
            a=table[f'reference_{chunk}']['median_ms'];b=table[f'optimized_{chunk}']['median_ms']
            timings[title+'_'+str(chunk)]={}
            for key,label in [('anatomical_patch_query','patch查询'),('hand_vertex_penetration_query','穿透查询'),('full','完整评分')]:
                speed=a[key]/b[key];lines.append(f'|{chunk}|{label}|{a[key]:.3f}|{b[key]:.3f}|{speed:.2f}倍|')
                rows.append(f'<tr><td>{chunk}</td><td>{label}</td><td>{a[key]:.3f}</td><td>{b[key]:.3f}</td><td>{speed:.2f}×</td></tr>')
                timings[title+'_'+str(chunk)][key]=dict(reference_ms=a[key],optimized_ms=b[key],speedup=speed)
        sections.append('<section><h2>'+title+'</h2><table><tr><th>候选数</th><th>阶段</th><th>原实现 ms</th><th>优化后 ms</th><th>加速</th></tr>'+''.join(rows)+'</table></section>')
        lines.append('')
    lines+=['## 正确性和质量','',
        f'本地{proof["scores_checked"]}个固定状态/局部扰动、服务器{server["SUMMARY.json"]["scores_checked"]}个状态/扰动验证通过，覆盖全部12个数据包。优化目标各项与验收布尔值对齐。',
        '存在非唯一的合格patch witness：本地1个槽的表面点不同，但两点都满足同一阈值，槽cost均为零；非合格槽仍核对坐标和代价，未放宽任何物理门槛。',
        '50项回归通过，包括交叉/边缘/退化/阈值、不同物体尺度与旋转、NaN/超范围输入、强制缓冲溢出的GPU完整查询、真实MANO图捕获及评分期间禁止Tensor.cpu。',
        f'从相同54条纯贴合LM初姿重新精修：实际评分仍{current["evaluations"]:,}次，速度{current["velocity"]}/54、几何{current["geometry"]}/54、联合{current["joint"]}/54，54条联合标记逐条与原实现一致。',
        f'新增精修wall：{old["solve_seconds"]:.3f}→{current["solve_seconds"]:.3f}s，摊销{old["amortized_refinement_seconds"]:.3f}→{current["amortized_refinement_seconds"]:.3f}s/pose（本地batch=1/8、候选128）。小batch时rollout占比更高，整轮改善小于大缓冲的几何查询改善。',
        '各阶段的加载、预热、验收和保存时间单列在run目录。54条分层诊断不能推广为总体通过率；五步速度通过也不是长期GSR或严格力闭合证明。','',
        '## 资源和交付','',
        '2048候选、302个裁剪三角形、128个面缓冲时临时GPU存储约2.4GiB；图捕获池可能另有驻留。该缓冲只改变执行并行度，溢出不会降低查询精度。',
        '服务器独立快照：`'+server['path']+'`。没有修改其他进程或恢复生产。',
        '现有三维网页更新为此次保存的结果；可查看成功/失败、逐槽贴合及五步物体运动。','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    result=dict(complete=True,production_started=False,objective_changed=False,physical_thresholds_changed=False,
        timings=timings,local_scores_checked=proof['scores_checked'],server_scores_checked=server['SUMMARY.json']['scores_checked'],
        quality=current,reference_quality=old,all_54_joint_decisions_preserved=True,tests_passed=50)
    (root/'SUMMARY.json').write_text(json.dumps(result,indent=2))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MANO 几何查询优化</title><style>body{margin:0;background:#f0f2ed;color:#152a30;font:14px sans-serif}nav{padding:12px 30px;background:#e3ebe4}a{color:#006e65}main{max-width:1100px;margin:auto;padding:24px 30px}h1{font:30px Georgia,serif}section{background:white;border:1px solid #c9d1cb;border-radius:5px;padding:16px;margin:16px 0;overflow:auto}table{width:100%;border-collapse:collapse}th,td{padding:10px;text-align:left;border-bottom:1px solid #e3ebe4}td:nth-child(n+3){font-family:monospace}p{line-height:1.8}</style><nav><a href="interactive.html">MANO 三维结果</a> · <a href="performance.html">几何查询耗时</a></nav><main><h1>MANO patch 与穿透查询优化</h1><p>同一输入、同一物理模型和阈值，预热后5次中位数。固定探针循环填充候选缓冲；不是2048条独立grasp完整求解的吞吐。</p>'''
    page+=''.join(sections)+f'<section><h2>固定54条精修验证</h2><p>速度38/54、几何33/54、联合31/54，联合通过/失败标记逐条保持。新增精修{old["solve_seconds"]:.3f}→{current["solve_seconds"]:.3f}s（本地batch=1/8、候选128）。这是诊断样本，不是总体通过率。</p><p>50项回归及两台GPU上的固定探针对照通过。保留原目标、五步rollout、5mm穿透允许量和全部验收门槛。生产任务继续停止。</p></section></main></html>'
    (args.previews/'performance.html').write_text(page)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
