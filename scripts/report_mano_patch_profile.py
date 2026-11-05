#!/usr/bin/env python3
"""Publish measured stage timings next to the current MANO inspection page."""
import argparse,html,json
from pathlib import Path

LABELS={'anatomical_patch_query':'实际三角形 patch 查询','rollout_total':'弹簧历史与物体 rollout',
        'mano_decode':'MANO 解码','hand_vertex_penetration_query':'手部顶点穿透查询',
        'mjwarp_collision':'MJWarp 碰撞','fixed_slot_surface_query':'固定槽物面投影',
        'fixed_hand_slots':'手面固定槽','other_scoring':'其余评分与聚合'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path,default=Path('output/mano_patch_profile_20261105'))
    p.add_argument('--previews',type=Path,default=Path('output/mano_comfree_previews_20261105'))
    args=p.parse_args();root=args.profile
    assert json.loads((root/'SUMMARY.json').read_text())['complete']
    local=json.loads((root/'PHASES.json').read_text())
    server=json.loads((root/'SERVER_PROFILE.json').read_text()) if (root/'SERVER_PROFILE.json').exists() else None
    records=[('本地 RTX 5060 Ti',local,json.loads((root/'CEM_WALL.json').read_text()))]
    if server:
        assert server['returncode']==0 and server['SUMMARY.json']['complete']
        records.insert(0,('服务器 A800',server['PHASES.json'],server['CEM_WALL.json']))
    lines=['# 当前MANO GPU评分耗时分布','',
        '同一 Large Diameter 物体、67个固定状态/局部扰动循环填满候选缓冲；GPU同步、预热后5次取中位数。',
        '分段CUDA事件在完整评分图内部计时；不把rollout内部子阶段重复相加。准备、编译、模型/文件加载均不混入稳态。',
        '这不是2048条独立grasp完整求解的吞吐。生产任务保持停止。','']
    lines+=['计时从已有MANO初姿开始，不含Shadow→MANO初始化IK；初始姿态评分包含在CEM计时中。','']
    sections=[]
    for title,phases,cem in records:
        lines+=['## '+title,'']
        cards=[]
        for chunk in [128,2048]:
            m=phases[f'chunk{chunk}_block256']['median_ms']
            lines+=[f'### {chunk}候选，当前默认block=256','',f'完整GPU评分 {m["full"]:.3f} ms。','',
                '|阶段|毫秒|占比|','|---|---:|---:|']
            rows=[]
            for key in sorted(LABELS,key=lambda k:m[k],reverse=True):
                percent=m['percent_of_full'][key]
                lines.append(f'|{LABELS[key]}|{m[key]:.3f}|{percent:.1f}%|')
                rows.append(f'<tr><td>{LABELS[key]}</td><td>{m[key]:.3f}</td><td><div class="bar" style="width:{percent:.2f}%"></div>{percent:.1f}%</td></tr>')
            cards.append(f'<article><h3>{chunk}候选 · {m["full"]:.2f} ms</h3><table><thead><tr><th>阶段</th><th>ms</th><th>占比</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></article>')
        lines+=['','|分块 / block|完整GPU评分 ms|rollout ms|patch ms|','|---|---:|---:|---:|']
        for key,v in phases.items():
            m=v['median_ms'];lines.append(f'|{key}|{m["full"]:.3f}|{m["rollout_total"]:.3f}|{m["anatomical_patch_query"]:.3f}|')
        wall=cem['wall_seconds']*1000;score=cem['adapter_gpu_stream_spans_ms']['evaluate_candidates'];validation=cem['adapter_gpu_stream_spans_ms']['validate']
        note=f'单条冻结MANO起点的 CEM 诊断：{cem["actual_pose_updates"]}轮、{cem["evaluations"]}次评分，wall {wall:.2f} ms；候选评分流跨度 {score:.2f} ms（{score/wall:.1%}），初始评分及验收 {validation:.2f} ms（{validation/wall:.1%}），采样、精英更新、主机调度等余项 {cem["remaining_wall_ms"]:.2f} ms（{cem["remaining_wall_ms"]/wall:.1%}）。'
        lines+=['',note,'','余项不是纯CPU算术时间；CPU profiler中的nonzero等待与GPU工作重叠，不能相加。该单条诊断的成败不混入固定54条质量统计。','']
        sections.append('<section><h2>'+title+'</h2><div class="grid">'+''.join(cards)+'</div><p>'+html.escape(note)+'</p></section>')
    recommendations=[
        '优先优化实际三角形patch查询：减少重复三角形/BVH工作，复用静态拓扑和物体局部数据；用保守包围体筛选后仍执行相同witness选择与失败回退。逐候选patch目标继续保留。',
        '让同抓型的多个独立pose共同填满候选缓冲。16个pose×128候选才是2048个有效候选；只增大缓冲并重复填充单条pose不会增加有效求解并行。该项仍需独立输入吞吐验证。',
        '其次合并投影和穿透查询中的重复坐标变换，复用MANO网格中间量；候选阶段按需输出诊断，最终结果仍保存完整指标。',
        'rollout block=32/64仅是小幅执行改善；更充分的接触并行仍值得测试，但应按其实际占比评估整体收益。',
        '再去除CEM动态nonzero/tolist同步、按pose发起随机采样与额外完整重评；完整GPU调度在大pose batch下需另测，当前单条诊断不支持把CPU称为主要瓶颈。']
    lines+=['## 优化顺序','',*['- '+v for v in recommendations],'',
        '线程布局实验只在诊断进程内替换block大小，评分在atol=1e-3、rtol=2e-4下对齐；没有修改生产配置、目标、物理参数或验收阈值。',
        '网页展示当前保存的6条诊断、每条3个真实阶段；其中最终结果3条联合通过、3条失败。该比例不代表整个数据集。','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MANO 计算耗时分布</title><style>
body{margin:0;background:#f0f2ed;color:#152a30;font:14px "Noto Sans CJK SC",sans-serif}main{max-width:1300px;margin:auto;padding:24px 30px}h1{font:30px Georgia,serif}nav{padding:12px 30px;background:#e3ebe4}a{color:#006e65}section{margin:24px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}article{background:white;border:1px solid #c9d1cb;border-radius:5px;padding:14px}table{width:100%;border-collapse:collapse}th,td{padding:10px;text-align:left;border-bottom:1px solid #e3ebe4}td:nth-child(2){font-family:monospace}td:nth-child(3){width:42%}.bar{height:5px;background:#006e65;min-width:1px;margin-bottom:4px}p,li{line-height:1.8}li{margin-bottom:10px}@media(max-width:850px){.grid{grid-template-columns:1fr}main{padding:18px}}
</style><nav><a href="interactive.html">MANO 三维结果</a> · <a href="performance.html">计算耗时分布</a></nav><main><h1>MANO 当前计算耗时分布</h1><p>完整GPU评分图内的CUDA事件计时，预热后5次取中位数。固定Large Diameter探针循环填充候选缓冲；不是2048条独立grasp完整求解。CEM从已有MANO初姿开始，不含Shadow→MANO初始化IK。生产任务保持停止。</p>'''
    page+=''.join(sections)+'<section><h2>优化顺序</h2><ol>'+''.join('<li>'+html.escape(v)+'</li>' for v in recommendations)+'</ol></section><p>GPU线程布局探针保持评分一致；没有切换生产配置。详细原始数据见项目 output/mano_patch_profile_20261105。</p></main></html>'
    target=args.previews/'performance.html';target.write_text(page)
    print(json.dumps(dict(report=str(root/'REPORT.md'),page=str(target),server_included=server is not None),ensure_ascii=False))


if __name__=='__main__':main()
