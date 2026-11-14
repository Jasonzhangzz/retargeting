#!/usr/bin/env python3
"""Summarize all saved trials without selecting candidates or changing gates."""
import argparse,json,statistics
from pathlib import Path


def span(values,digits=2):
    values=[float(x) for x in values if x is not None]
    if not values:return '—'
    lo,hi=min(values),max(values)
    return f'{lo:.{digits}f}' if abs(hi-lo)<10**(-digits) else f'{lo:.{digits}f}–{hi:.{digits}f}'


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('directories',type=Path,nargs='+');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    reports=[];lines=['# Moxian GPU 标定实测','',
        '本机 RTX 5060 Ti；保留记录 3969 帧，按现有覆盖规则抽取 349 帧训练，完整留出 1527 帧。',
        '各轮从同一初始化开始；原始记录不改写，不发布候选。无预处理缓存仍复用记录中已有的同期 IK，重新转换缺失的 240 帧。',
        '端到端包含预处理、拟合及全部验收；首次编译单列。整卡显存包含 Warp 与同时运行的实时服务。',
        '耗时通过只代表时间达标；最终通过还要求所有精度和完整留出检查通过。','']
    for directory in a.directories:
        warm=directory/'warmup.json'
        if warm.exists():lines.append(f'- `{directory.name}` 启动预热：{json.loads(warm.read_text())["seconds"]:.2f} 秒，单独计时。')
        records=json.loads((directory/'summary.json').read_text())
        for item in records:
            folder=directory/f'{item["method"]}-{item["cache"]}-{item["run"]}'
            report=json.loads((folder/'report.json').read_text()) if (folder/'report.json').exists() else None
            reports.append((item,report,folder))
    verification=a.output.parent/'gpu-service-verification.json'
    if verification.exists():
        service=json.loads(verification.read_text())
        lines+=['',f'真实网页服务本次重载至全部预热结束约 {service["service_start_to_warmup_observed_s"]:.1f} 秒（包含启动、三种模式及与实时显示并行）；与上面的独立基准预热条件不同。此启动成本仍需改进。', '已核实手套连接、实时网格恢复、3969 帧原始记录保留，当前继续使用默认模型。']
    lines+=['','| 方法 | 缓存 | 轮次 | 总秒数 | 编译秒数 | 总数减编译 | 时间≤60s | 全部验收 |','|---|---|---:|---:|---:|---:|---|---|']
    for item,r,_ in reports:
        seconds=item['end_to_end_s'];compile_s=item.get('compilation_s',0.)
        lines.append(f'| {item["method"]} | {item["cache"]} | {item["run"]} | {seconds:.2f} | {compile_s:.2f} | {seconds-compile_s:.2f} | {"通过" if item.get("performance_passed") else "未通过"} | {"通过" if item.get("accepted") else "未通过"} |')
        if item.get('error'):lines.append(f'\n该轮异常：`{item["error"]}`\n')
    if any(d.name=='flash-replay-v2' and (d.parent/'flash-replay-v1/DIAGNOSTIC_ONLY.json').exists() for d in a.directories):
        lines+=['','早期 `flash-replay-v1` 触发优化器对象绑定的重编译上限，已中止并保留诊断；表中 FlashSAC buffer 数字使用修正后的张量 Adam 版本 v2。']
    lines+=['','## 精度与显存','',
        '以下列出每个方法的全部实测范围，未挑选表现最好的轮次。间隙来自固定材料触点；穿透来自原始 MANO 三角面。','',
        '| 方法 | 对指均值 mm | 对指最大 mm | 纯 EMF 最大穿透 mm | 触觉路径最大穿透 mm | PyTorch 峰值 GiB | 整卡峰值 GiB |',
        '|---|---:|---:|---:|---:|---:|---:|']
    methods=list(dict.fromkeys(item['method'] for item,_,_ in reports))
    for method in methods:
        rr=[r for item,r,_ in reports if item['method']==method and r]
        if not rr:continue
        columns=[span([r.get(k) for r in rr]) for k in ('contact_gap_mm','contact_max_gap_mm','max_penetration_mm','runtime_max_penetration_mm')]
        columns += [span([r[k]/2**30 for r in rr if k in r]) for k in ('peak_cuda_memory_bytes','peak_device_memory_bytes')]
        lines.append('| '+method+' | '+' | '.join(columns)+' |')
    for method in methods:
        rr=[r for item,r,_ in reports if item['method']==method and r]
        if not rr:continue
        lines+=['',f'### {method}：逐指及四组接触','',
                '| 手指 | 留出 EMF RMSE mm |','|---|---:|']
        for finger in ('thumb','index','middle','ring','pinky'):
            lines.append(f'| {finger} | {span([r.get("skeleton_validation",{}).get("per_finger",{}).get(finger,{}).get("rmse_mm") for r in rr])} |')
        lines+=['','| 对指 | 单轮平均间隙的范围 mm | 单轮最大间隙的范围 mm | 单轮平均法向点积的范围 |','|---|---:|---:|---:|']
        for finger in ('index','middle','ring','pinky'):
            groups=[[c for c in r.get('contact_material_records',[]) if c['stage']=='pinch_'+finger] for r in rr]
            means=[statistics.mean(c['gap_m']*1000 for c in g) for g in groups if g]
            maxima=[max(c['gap_m']*1000 for c in g) for g in groups if g]
            normals=[statistics.mean(c['normal_dot'] for c in g) for g in groups if g]
            lines.append(f'| thumb–{finger} | {span(means)} | {span(maxima)} | {span(normals,3)} |')
        lines+=['',f'### {method}：阶段耗时与拒绝原因','',
                '| 阶段 | 秒数范围 |','|---|---:|']
        for stage in ('preparation','baseline','identification','sensor_validation','tactile_validation'):
            values=[r.get('preparation_s') if stage=='preparation' else r.get('stage_seconds',{}).get(stage) for r in rr]
            lines.append(f'| {stage} | {span(values)} |')
        lines+=['','阶段表的编译时间包含在其实际阶段耗时中；其余时间包括模型构造、指标汇总、碰撞交叉检查和序列化。','']
        reasons=list(dict.fromkeys(reason for r in rr for reason in r.get('rejection_reasons',[])))
        lines += ['- '+reason for reason in reasons]
    lines+=['','## 单帧延迟','',
        '原报告 `runtime_latency_ms` 记录一次 32 帧回放块的墙钟时间，不能当作单帧延迟。独立 `runtime-*.json` 为同一候选的单帧 GPU 姿态路径（含触觉关联和原始三角面查询）；不含串口、上游 IK 与网页传输。','',
        '| 方法 | 帧数 | p50 ms | p95 ms | 最大 ms |','|---|---:|---:|---:|---:|']
    for directory in a.directories:
        for path in sorted(directory.glob('runtime-*.json')):
            r=json.loads(path.read_text());lat=r['latency_ms']
            lines.append(f'| {r["method"]} | {r["frames"]} | {lat["p50"]:.2f} | {lat["p95"]:.2f} | {lat["max"]:.2f} |')
    lines+=['','FlashSAC buffer 入口按用户选择只复用 GPU buffer/编译更新方式，直接优化标定损失，没有 reward、rollout 或 SAC actor/critic。','',
        '精度要求保持：对指均值≤3 mm、最大≤5 mm、穿透≤2 mm，并通过其他动作、接近释放、骨架和完整留出检查。',
        '若全部验收栏未通过，候选只作诊断，继续使用原模型。']
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(lines)+'\n');print(a.output)
if __name__=='__main__':main()
