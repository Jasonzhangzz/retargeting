"""Rebuild complete single-trajectory timing table from original repeat receipts."""
import argparse
import json
from pathlib import Path

import numpy as np

LABELS = dict(sdk="SDK",network="Network initial guess",joint_lm="Joint LM · 32",causal_prefix="Causal prefix · 32",frozen_parallel="Frozen parallel · 32",frozen_history_previous="Frozen history / previous · 10",frozen_history_network="Frozen history / network · 10")


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True);a=ap.parse_args();r=a.root
    manifest=json.loads((r/"MANIFEST.json").read_text());rows=[]
    for name in manifest["methods"]:
        d=json.loads((r/name/"result.json").read_text());assert d["complete"] and len(d["timings"])==10 and d["frames"]==1133
        med={k:float(np.median([x[k] for x in d["timings"]])) for k in d["timings"][0]}
        assert all(abs(med[k]-d["median_s"][k])<1e-12 for k in med)
        rows.append(dict(d,label=LABELS[name]))
    sdk=next(d for d in rows if d['method']=='sdk')['median_s']['total']
    text=['# 拧螺丝右手：完整 1133 帧求解耗时','', 'builtin glove；每次从头重置整段历史，2次预热后测量10次。计时从CPU中的EMF数组开始，直到得到全部1133帧CPU关节角输出。包含特征/网络、人手重建/目标生成、机器人优化、滤波、API诊断和必要同步/输出拷贝。模型/资产加载、文件IO不进入稳态耗时；首次编译另行记录。SDK每次重新构造离线pipeline/session以重置历史，其初始化计入总耗时并单列。', '', '|方法|整段中位数/s|最小—最大/s|帧/s|SDK耗时÷本方法|', '|---|---:|---:|---:|---:|']
    for d in rows:
        t=d['median_s']['total'];text.append(f"|{d['label']}|{t:.6f}|{d['minimum_s']:.6f}—{d['maximum_s']:.6f}|{1133/t:.1f}|{sdk/t:.3f}×|")
    text+=['', 'Network initial guess 仅完成网络预测，没有人手IK、LM或额外滤波，不能视为相同优化质量的求解器。其余方法保留既有更新预算和权重。', '', '## 分阶段中位数（秒）','', '|方法|特征/网络|人手/目标|机器人|滤波|CPU输出|API等其余开销|','|---|---:|---:|---:|---:|---:|---:|']
    for d in rows:
        t=d['median_s'];other=t.get('api_overhead_and_diagnostics',0.)+t.get('initialize',0.)+t.get('emf_objects',0.)+t.get('python_loop_and_output',0.)
        robot=t.get('robot',t.get('robot_and_filter',0.));filt='包含于机器人' if d['method']=='sdk' or d.get('filter_fused_into_robot') else f"{t.get('filter',0.):.6f}"
        text.append(f"|{d['label']}|{t.get('network',0.):.6f}|{t.get('human_and_targets',0.):.6f}|{robot:.6f}|{filt}|{t.get('output_to_cpu',0.):.6f}|{other:.6f}|")
    text+=['', '分项各自取中位数，故它们相加未必恰好等于总耗时的中位数。frozen-history 的滤波融合在逐帧机器人图中，不能单列为独立测量。SDK的机器人时间也包含其内部滤波。', '', '## 加载、首次运行与一致性','', '|方法|环境/模型准备/s|首次整段运行/s|与回放缓存最大q差/rad|峰值驻留/MiB|','|---|---:|---:|---:|---:|']
    for d in rows:text.append(f"|{d['label']}|{d['setup_s']:.4f}|{d['cold_first']['total']:.4f}|{d['output_max_vs_cached_rad']:.3e}|{d.get('peak_nvml_resident_mib',0)}|")
    text+=['', 'CUDA均在同一A800（设备1）串行运行，驻留上限8GiB。设备有其他进程驻留，未操作它们；各方法开始前的利用率/显存快照保存在 *.gpu_before.json。本任务旧基准在测量期间暂挂，结束后恢复，收据见 PAUSED_BENCHMARK.json / RESUMED_BENCHMARK.json。结果只描述本机本次共享环境，不声称独占GPU性能。', '', '网络训练更新数为0。完整原始重复耗时、冷启动、输出结果、最大缓存差、编译和运行日志均保留。原五轨迹合计结果未改动。']
    if (r/'quality.json').exists():
        quality=json.loads((r/'quality.json').read_text());assert quality['complete']
        text+=['', '## 本次输出的原始目标检查','', '|方法|本次原始 F 总和|与旧缓存 F 差异|q 相对 SDK RMSE/rad|','|---|---:|---:|---:|']
        for q in quality['rows']:text.append(f"|{LABELS[q['method']]}|{q['common_F_raw_total']:.6f}|{q['F_change_percent']:+.6f}%|{q['q_sdk_rmse_rad']:.6f}|")
        text+=['', '本次为单条轨迹推理，旧缓存为五条轨迹批处理；网络初值的浮点数值存在差异，部分有限步求解输出也发生变化。上述 F 差异很小，但不能以此声称关节角逐位一致。参见 quality.json 和各方法 result.json 中 output_max_vs_cached_rad；SDK 和 frozen_history_previous 输出与各自缓存逐位一致，Joint LM 缓存差约5.05e-6 rad。Joint LM 与 SDK 仍是不同信息条件和数值解，耗时比不代表相同解精度加速。']
    (r/'REPORT.md').write_text('\n'.join(text)+'\n');(r/'summary.json').write_text(json.dumps(dict(complete=True,frames=1133,rows=rows,neural_optimizer_updates=0),indent=2))
    print('\n'.join(text[:15]))


if __name__=='__main__':main()
