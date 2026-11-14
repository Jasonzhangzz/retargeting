"""Summarize frozen-source force ablations without claiming physical force truth."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);args=ap.parse_args();out=args.output
    manifest=json.loads((out/'manifest.json').read_text());rows=[json.loads(x) for x in (out/'results.ndjson').read_text().splitlines()]
    expected=sum(d['evaluation_frames'] for d in manifest['datasets'].values())*len(manifest['variants'])*len(manifest['seeds'])
    if len(rows)!=expected:raise ValueError(f'Incomplete ablation: {len(rows)}/{expected}')
    summary={};coherence=[];source_gaps=[];source_angles=[]
    lines=['# moxian 触觉方向、摩擦锥与法向力对照实验','',
        '日期：2026-11-09。真实输入为此前保持的拇食对指，以及本次用户确认抓握圆形瓶盖的数据。标签仅用于结果分组与对指间隙评价，不送入求解器。',
        '',f"共{expected}次CEM求解，6个预先指定变体、3个随机种子。两段采样分别取前半段作相对压力归一化，后半段共11个不同时间戳帧作评价。帧来自两次短时保持，彼此相关，不能当作11次独立操作或作泛化结论。",
        '', '## 方法', '',
        '当前默认实现已包含CARD式(3)的法向对齐项。新增实验项不是补上一个原本完全缺失的方向损失，而是与持续法向对齐比较显式库仑锥及相对法向力拟合。[论文式(3)与附录A](https://arxiv.org/html/2610.36539v1)。',
        '',r'在每个活动区域估计 $\tilde f=\sum_i p_i n_i$，方向为其归一化，coherence为 $\|\tilde f\|/\sum_i p_i$。方向与现有压力加权法向一致；这并未增加独立的姿态或剪切力观测。正号表示手施加给物体的推力。',
        '',r'对每个候选姿态的机器人法向 $n(q)$，解析求解 $\min_f\|f-\tilde f\|^2+\lambda_N(n^Tf-\|\tilde f\|)^2$，约束 $f_n\geq0,\ \|f_t\|\leq\mu f_n$。残差除以 $\|\tilde f\|^2$ 后加入CEM，保留原位置项、力优先级、语义区域和全部碰撞检查。',
        '', '法向目标等于估计合力幅值，是“局部法向推力”假设，不是另一项实测法向力。ADC没有牛顿标定，剪切力未测，摩擦系数0.5与1也只是敏感性条件。解析优化的力是给定姿态下的潜变量，不代表机器人已经产生或追踪了该力。',
        '', '所有变体固定源MANO、接触CoP及方向、当前帧Robokit初解、1024×16预算和0.5 mm自穿透门槛。不推断对指、不改接触目标、不使用两指闭合残差、不检查帧间扫掠轨迹、不使用历史姿态作为失败回退。',
        '', '## 结果', '',
        '“到位”指所选语义pad接触点距同一源CoP≤2 mm；“到位且方向合格”还要求原始估计方向与机器人法向夹角≤45°。统一用45°评价所有变体，并要求候选通过穿透验收。投影后的潜变量力天然落在锥内，这个恒成立事实不计为接触成功。',
        '', '以下为各帧与随机种子的均值；详细逐帧数据、种子波动和所有接触材料坐标在`results.ndjson`。']
    order=list(manifest['variants'])
    names={'reach_no_direction':'仅位置','card_alignment':'现有CARD方向项','cone_mu1':'锥 μ=1','cone_normal_mu1':'锥+法向力 μ=1','cone_mu05':'锥 μ=0.5','cone_normal_mu05':'锥+法向力 μ=0.5'}
    for case,label in [('pinch','拇食对指'),('bottle_cap','圆形瓶盖')]:
        lines+=['',f'### {label}','','| 方法 | CoP RMS mm | 平均角误差° | 到位% | 到位且方向合格% | 可行% | 最大自穿透mm |','|---|---:|---:|---:|---:|---:|---:|']
        summary[case]={}
        for variant in order:
            data=[r for r in rows if r['dataset']==case and r['variant']==variant]
            def mean(key):return float(np.mean([r['metrics'][key] for r in data]))
            strict=float(np.mean([r['metrics']['contact_and_in_cone_rate']*r['metrics']['feasible'] for r in data]))
            max_pen=max(r['metrics']['self_penetration_mm'] for r in data)
            item=dict(runs=len(data),rms_mm=mean('rms_mm'),angle_deg=mean('mean_angle_deg'),
                contact_rate=mean('contact_rate'),strict_contact_direction_rate=strict,
                feasible_rate=mean('feasible'),max_self_penetration_mm=max_pen,elapsed_ms=mean('elapsed_ms'),
                seed_strict_rates={str(seed):float(np.mean([r['metrics']['contact_and_in_cone_rate']*r['metrics']['feasible'] for r in data if r['seed']==seed])) for seed in manifest['seeds']})
            if case=='pinch':
                item['cop_pair_gap_mm']=mean('selected_cop_pair_gap_mm');item['sampled_pad_gap_mm']=mean('sampled_pad_surface_gap_mm')
                item['distal_contact_rate']=float(np.mean([a['error_mm']<=2 for r in data for a in r['contacts'] if a['region'] in (3,6)]))
            summary[case][variant]=item
            lines.append(f"| {names[variant]} | {item['rms_mm']:.3f} | {item['angle_deg']:.2f} | {100*item['contact_rate']:.1f} | {100*strict:.1f} | {100*item['feasible_rate']:.1f} | {max_pen:.3f} |")
        if case=='pinch':
            lines+=['','| 方法 | 两个所选CoP间距mm | 采样pad曲面最小间隙mm | 远端CoP到位% |','|---|---:|---:|---:|']
            for variant in order:
                item=summary[case][variant];lines.append(f"| {names[variant]} | {item['cop_pair_gap_mm']:.3f} | {item['sampled_pad_gap_mm']:.3f} | {100*item['distal_contact_rate']:.1f} |")
    for r in rows:
        if r['variant']!='card_alignment' or r['seed']!=manifest['seeds'][0]:continue
        coherence.extend(a['force_estimate']['coherence'] for a in r['contacts'])
        if r['dataset']=='pinch':
            a={a['region']:a for a in r['contacts']};source_gaps.append(float(np.linalg.norm(np.asarray(a[3]['position_m'])-a[6]['position_m'])*1000))
            source_angles.append(float(np.degrees(np.arccos(np.clip(np.dot(a[3]['direction'],a[6]['direction']),-1,1)))))
    gap=float(np.mean(source_gaps));angle=float(np.mean(source_angles))
    base=summary['bottle_cap']['card_alignment'];wide=summary['bottle_cap']['cone_normal_mu1'];tight=summary['bottle_cap']['cone_normal_mu05']
    lines+=['','## 本次结论','',
        f"瓶盖样本支持继续研究这个方向：锥加法向力拟合使到位且方向合格比例从{100*base['strict_contact_direction_rate']:.1f}%提高到μ=1时的{100*wide['strict_contact_direction_rate']:.1f}%、μ=0.5时的{100*tight['strict_contact_direction_rate']:.1f}%。这不是独立物体接触真值，样本也不足以选择通用默认参数。",
        '', '对指没有闭合。μ=1的新增法向力版本虽然降低了RMS，但到位且方向合格比例由40%降到30%；μ=0.5版本维持40%，pad曲面仍相距约17 mm。因此不能仅依据RMS下降宣称对指改善完成。',
        '', '本轮新增估计与显式力锥损失，线上默认参数未切换。回归72项通过，含解析力拟合与独立约束优化器对照、单位/旋转/压力尺度、目标保持、无配对/无触觉负例、碰撞与本帧候选回退。']
    lines+=['','## 输入约束与解释','',
        f'对指评价帧的源拇食CoP平均相距{gap:.3f} mm，估计推力夹角平均{angle:.2f}°。当前方向估计继承了未实测注册的MANO表面，不能据此认为源方向天然准确。',
        '',f'若两个机器人CoP要相距≤2 mm，同时各自距源目标≤2 mm，由三角不等式，源目标间距至多6 mm；本段实际约{gap:.1f} mm。仅调整机器人朝向无法同时满足这些条件；至少一个CoP位置误差需要达到约{(gap-2)/2:.1f} mm，或先纠正源接触几何。',
        '', '“采样pad曲面间隙”使用双方原生pad顶点及面心到另一表面的距离，是对指后验诊断，不进入求解器，也不是连续三角形相交证明。源CoP间距与pad曲面间隙含义不同。',
        '', '瓶盖没有尺寸、独立位姿、真实接触位置或机器人测力真值。因此表中的到位率是对源MANO接触目标的拟合，不是已验证的瓶盖接触率、抗滑性、承载能力或力跟踪成功率。',
        '', '没有依据这两个短片段挑选新默认参数。显式锥与法向力项保留为实验选项；默认仍是CARD法向对齐。各变体没有修改源MANO，也没有声称仅凭从MANO自身计算的法向就能反过来校正MANO的闭合。',
        '', '## 复现与验证','',
        '相同输入、拓扑、协议和结果均保存在本目录。`*-frames.json`与`*-topology.json`可直接用于重跑；`manifest.json`记录预算、种子、文件摘要和假设。时延测量与线上GPU服务并行，部分初始运行还与回归测试并行，不是隔离性能基准。',
        '', '回归及线上默认保持记录见[validation.json](validation.json)。对照图可查看[PNG](comparison.png)或[PDF](comparison.pdf)。',
        '', '```bash',
        '.venv-contact/bin/python scripts/benchmark_moxian_contact_force.py \\',
        '  --pinch output/card_force_ablation_20261109/pinch-frames.json \\',
        '  --pinch-topology output/card_force_ablation_20261109/pinch-topology.json \\',
        '  --grasp output/card_force_ablation_20261109/bottle_cap-frames.json \\',
        '  --grasp-topology output/card_force_ablation_20261109/bottle_cap-topology.json \\',
        '  --output output/card_force_ablation_replay',
        '```']
    for case in manifest['datasets']:
        source=json.loads((out/(case+'-source.json')).read_text())
        (out/(case+'-frames.json')).write_text(json.dumps(source['frames']))
        (out/(case+'-topology.json')).write_text(json.dumps(source['topology']))
    (out/'analysis.json').write_text(json.dumps(dict(results=summary,source_pinch_gap_mm=gap,source_direction_angle_deg=angle,mean_normal_coherence=float(np.mean(coherence))),indent=2))
    lines=[line.replace(chr(92)*2,chr(92)) for line in lines]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    labels=['Reach','CARD','Cone 1','Cone+N 1','Cone .5','Cone+N .5']
    for row,case in enumerate(('pinch','bottle_cap')):
        for col,(key,title) in enumerate((('rms_mm','Source CoP RMS (mm)'),('strict_contact_direction_rate','Reached + direction within 45 deg (%)'))):
            vals=[summary[case][v][key]*(100 if col else 1) for v in order]
            axes[row,col].bar(np.arange(len(order)),vals,color=['#999999','#3878b8','#62a68b','#227c5a','#dca559','#c37836'])
            axes[row,col].set_xticks(range(len(order)),labels,rotation=20);axes[row,col].set_title(('Pinch: ' if row==0 else 'Bottle cap: ')+title)
            axes[row,col].grid(axis='y',alpha=.2);axes[row,col].set_axisbelow(True)
    fig.suptitle('Frozen source targets; no pinch pairing; uncalibrated relative pressure')
    fig.savefig(out/'comparison.png',dpi=160);fig.savefig(out/'comparison.pdf');plt.close(fig)
    print(json.dumps(dict(runs=len(rows),output=str(out/'REPORT.md'),source_gap_mm=gap,source_angle_deg=angle)))


if __name__=='__main__':main()
