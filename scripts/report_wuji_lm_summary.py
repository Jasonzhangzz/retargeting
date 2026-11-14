#!/usr/bin/env python3
"""Summarize the completed, quality-screened native CUDA benchmark."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();root=args.root
    read=lambda name:json.loads((root/name).read_text())
    metrics=read('metrics.json');timing=read('benchmark_robokit_compiled_report.json');sdk=read('sdk_benchmark_components.json')
    selection=read('selection.json');basins=read('basin_diagnosis.json');full=read('sdk_objective_diagnosis.json');fk=read('fk_and_startup_metrics.json')
    assert len(selection['trajectories'])==5 and len(full)==4 and len(timing['steady_repeats'])==3 and len(sdk['repeats'])==3
    summary=dict(control_threshold_rad=1e-4,frames_per_configuration=sum(sum(v['frames'] for v in d['sides'].values()) for d in selection['trajectories']),
        checkpoint=read('checkpoint_provenance.json'),speed={},aggregate_metrics=metrics['aggregate'],
        cold_solver_warmup_seconds=timing['cold_warmup_seconds_including_compile_and_result_capture'],
        benchmark_first_result_seconds=timing['first_result_including_import_model_urdf_validation_compilation_and_capture_seconds'],
        torch_peak_allocated_gib=timing['gpu_peak_allocated_bytes']/2**30,
        memory_scope='PyTorch allocator only; excludes native Warp allocations and driver context')
    for config in ('builtin','hl_urdf'):
        native=[sum(g['end_to_end_seconds'] for g in rep['groups'] if g['configuration']==config) for rep in timing['steady_repeats']]
        robot=[sum(g['robot_lm_seconds']+g['filter_seconds'] for g in rep['groups'] if g['configuration']==config) for rep in timing['steady_repeats']]
        reference=[rep['configuration_totals'][config]['end_to_end_seconds'] for rep in sdk['repeats']]
        reference_robot=[rep['configuration_totals'][config]['robot_retarget_seconds'] for rep in sdk['repeats']]
        summary['speed'][config]=dict(native_end_to_end_repeats_s=native,sdk_end_to_end_repeats_s=reference,
            native_end_to_end_median_s=float(np.median(native)),sdk_end_to_end_median_s=float(np.median(reference)),
            end_to_end_speedup=float(np.median(reference)/np.median(native)),
            native_robot_stage_median_s=float(np.median(robot)),sdk_robot_stage_median_s=float(np.median(reference_robot)),
            robot_stage_speedup=float(np.median(reference_robot)/np.median(robot)))
    rows=[]
    for config in ('builtin','hl_urdf'):
      for side in ('all','left','right'):
       for method,key in [('network','q_network_rad'),('native_lm_raw','q_lm_raw_rad'),('native_lm_filtered','q_lm_filtered_rad')]:
        differences=[]
        for path in (root/metrics['prediction_subdir']/config).glob('*.npz'):
            if side!='all' and not path.stem.endswith('_'+side):continue
            differences.append(np.load(path)[key]-np.load(root/'gt'/config/path.name)['q_sdk_rad'])
        error=np.concatenate(differences)
        order=read('joint_order_and_limits.json')['right' if side=='all' else side]['names']
        for j,name in enumerate(order):rows.append(dict(configuration=config,side=side,method=method,joint=j,
            joint_name=name[2:] if side=='all' else name,mae_rad=float(abs(error[:,j]).mean()),rmse_rad=float(np.sqrt(np.mean(error[:,j]**2))),max_error_rad=float(abs(error[:,j]).max())))
    with (root/'joint_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,stage,title in [(axes[0],'end_to_end','Full pipeline: all 5 episodes / both hands'),(axes[1],'robot_stage','Wuji2 q stage (including output filter)')]:
        x=np.arange(2);values=[summary['speed'][c] for c in ('builtin','hl_urdf')]
        native=[v['native_'+stage+'_median_s'] for v in values];ref=[v['sdk_'+stage+'_median_s'] for v in values]
        a=ax.bar(x-.17,ref,.34,label='SDK',color='#465364');b=ax.bar(x+.17,native,.34,label='GPU native LM',color='#278c76')
        ax.bar_label(a,fmt='%.2f s');ax.bar_label(b,fmt='%.2f s');ax.set(xticks=x,xticklabels=['Builtin glove','Fixed HL URDF override'],ylabel='Seconds (median of 3)',title=title);ax.legend();ax.grid(axis='y',alpha=.2)
        ax.set_ylim(0,max(ref)*1.2)
    fig.savefig(root/'figures/performance.png',dpi=160);plt.close(fig)
    lines=['# 干净实录数据：SDK 与并行 CUDA LM 对照','',
        '已替换为五条通过输入质量筛选的完整双手轨迹。GPU 求解加速成立；与 SDK 输出的 `1e-4 rad` 一致性尚未达到。',
        '', '[交互式完整轨迹与误差曲线](comparison.html) · [逐轨迹指标 CSV](trajectory_metrics.csv) · [逐关节指标 CSV](joint_metrics.csv) · [全部指标](metrics.json)',
        '', '## 数据与模型','',
        f"每个配置均使用 **{summary['frames_per_configuration']:,} 个单手帧**，包含全部首帧、初始化瞬态与原始保留时间戳。两种配置共评估 11,540 个单手帧。",
        '按 collection 名称日期降序，每个 collection 最多检查 40 个候选；共检查 538 条后在五个 collection 各选一条合格轨迹，随机候选顺序 seed=42。数据筛选没有使用网络输出或 Wuji2 q 误差。',
        '', '|索引|Collection|左/右帧数|最低置信度|TX→RX 最大距离|','|---|---|---:|---:|---:|']
    for d in selection['trajectories']:
        q=d['quality']['summary'];lines.append(f"|{d['index']}|{d['collection']}|{d['sides']['left']['frames']}/{d['sides']['right']['frames']}|{min(v['confidence_min'] for v in q.values()):.4f}|{max(v['receiver_distance_max_m'] for v in q.values())*100:.2f} cm|")
    lines+=['', '筛选条件：双手完整、120–1200 帧、全部有限、置信度≥0.8、TX→RX 距离1–25 cm、相邻位置跳变≤25 mm、旋转跳变≤0.5 rad、时间间隔≤50 ms；录制人手角度经内置 URDF 重建的每指 EMF 位置误差99分位≤10 mm、旋转误差99分位≤0.7 rad。门槛属于数据/模型自洽筛查，并非传感器测量精度声明。',
        '使用 H5 已保留的约60 Hz 时间戳序列；原离线导出设置 rate divider=2，本次没有再次重采样。source_seq 步长2按其导出协议解释，连续性使用时间戳检查。',
        '[候选淘汰记录](selection_audit.json) · [完整来源映射与 H5 SHA256](selection.json) · [输入质量比较图](figures/data_quality.png)',
        '', '模型为验证集最优 `temporal_causal/best.pt`，step **525**，4个block、宽1024，完整历史/当前 EMF 输入，无 q 反馈。版本：PyTorch 2.10.0+cu128、Warp 1.16.0、Wuji SDK 2026.8.31；测试在共享 A800 80GB 的 GPU 1 进行。',
        '训练缓存记录为 MANO 合成 EMF，标签由 SDK 对 MANO 21 点求解；本次为真实录制 EMF。[训练缓存溯源](teacher_cache_provenance.json)',
        '', '两种条件分别为：①SDK内置 glove URDF（未 override）；②显式固定 HL 左/右手 URDF override。SDK 与 LM 使用相同条件的几何及 wrist↔TX 变换。替换数据涉及多个佩戴者，HL配置作为固定URDF对照，未声明是每位佩戴者的专属标定。',
        '', '## 关节角误差','', '参考为官方 SDK 连续 session 输出，含其原生 alpha=.2 输出滤波。以下主对照为并行 LM 后加相同 alpha 的GPU并行线性滤波，按全部20维角度计算。SDK输出是本次参考（GT），不是实体机器人编码器测量。',
        '', '|配置|方法|RMSE(rad)|RMSE(°)|最大绝对误差(rad)|','|---|---|---:|---:|---:|']
    for d in metrics['aggregate']:
        if d['side']=='all':lines.append(f"|{d['configuration']}|{d['method']}|{d['rmse_rad']:.6f}|{np.rad2deg(d['rmse_rad']):.3f}|{d['max_error_rad']:.6f}|")
    lines+=['', '两种配置均没有任何一帧的全部20关节同时达到 `1e-4 rad`。原始 LM、滤波 LM、网络预测均保留。',
        '原样本的早期Torch-GPU原型曾记录RMSE约13.91°/12.69°；本次数据与求解实现均已更换，这些旧值仅保留审计，不用于隔离估计数据筛选或求解器的收益。',
        '', '|配置|Wuji2五指末端位置差 RMSE|末端位置差最大值|前1秒 q RMSE|1秒后 q RMSE|','|---|---:|---:|---:|---:|']
    for config in ('builtin','hl_urdf'):
        fr=[x for x in fk['metrics'] if x['configuration']==config and x['method']=='native_filtered'];sr=[x for x in fk['startup'] if x['configuration']==config and x['method']=='native_filtered']
        tip=np.sqrt(sum(x['fingertip_position_rmse_m']**2*x['frames'] for x in fr)/sum(x['frames'] for x in fr))
        warm=lambda key:np.sqrt(sum(x[key+'_q_rmse_rad']**2*x[key+'_frames'] for x in sr)/sum(x[key+'_frames'] for x in sr))
        lines.append(f"|{config}|{tip*1000:.3f} mm|{max(x['fingertip_position_max_m'] for x in fr)*1000:.3f} mm|{warm('first_second'):.6f} rad|{warm('after_first_second'):.6f} rad|")
    lines+=['', '末端位置差通过同一 Wuji2 URDF 对预测 q / SDK q 做 FK 后比较，属于对 SDK 参考的差异；人手层 EMF 拟合另存于 [FK与初始化指标](fk_and_startup_metrics.json)。主指标保留全部初始化阶段。输出全部满足所用 URDF 关节限位。',
        '', '## 并行方式与计时','', '每个几何/左右手分组包含全部五条轨迹，共2885帧；Transformer一次批量前向处理五条完整轨迹。人手层三个固定初值并行，共8655个21维系统；机器人层为2885个20维系统。LM按迭代推进，每一轮所有帧并行，没有前一帧 q 到后一帧求解器的依赖。',
        'LM法方程、Cholesky、接受/拒绝使用 Robokit 原生 CUDA 内核，Warp生成的 [.cu及SHA256](native_cuda/manifest.json) 已归档。FK、残差、解析Jacobian在GPU；编译版使用TorchInductor/Triton CUDA。线性系统为FP32，几何残差为FP64，网络为BF16 FlexAttention。',
        'Torch持有完整求解CUDA Graph内存池，Warp加入外部capture；Robokit内部capture关闭以避免两套allocator混用。主执行路径没有CPU线性求解或CPU优化器。CPU承担文件读取、固定模型构建及批量输入预处理。',
        'GPU输入/输出、网络、人手LM、目标准备、机器人LM、滤波均包含在下表完整流程内；磁盘读取、模型/URDF加载、验证、保存和报表在计时外。SDK采用原始session默认预算；LM人手70轮，机器人25+40轮。',
        '', '|配置（每项5770单手帧）|SDK完整流程|GPU完整流程|完整流程加速|SDK q阶段|GPU LM+滤波阶段|q阶段加速|','|---|---:|---:|---:|---:|---:|---:|']
    for config,v in summary['speed'].items():lines.append(f"|{config}|{v['sdk_end_to_end_median_s']:.3f}s|{v['native_end_to_end_median_s']:.3f}s|{v['end_to_end_speedup']:.2f}×|{v['sdk_robot_stage_median_s']:.3f}s|{v['native_robot_stage_median_s']:.3f}s|{v['robot_stage_speedup']:.2f}×|")
    lines += ['', '使用预热后3次重复的中位数。主SDK表使用最新分阶段计时；此前独立计时也完整保留。q阶段单独比较不含网络和人手重建；完整流程列包含这些开销。共享GPU/CPU导致计时波动，全部原始重复值见 [计时汇总](summary.json)、[SDK分阶段计时](sdk_benchmark_components.json)、[CUDA计时](benchmark_robokit_compiled_report.json)。',
        f"首次求解包含内核编译、构图、结果保存的预热轮共 **{summary['cold_solver_warmup_seconds']:.2f}s**；完整基准脚本从import到首轮全部结果为 **{summary['benchmark_first_result_seconds']:.2f}s**。因此本次加速结论适用于模型/内核已预热的批量求解，冷启动首轮更慢。",
        f"PyTorch allocator峰值 **{summary['torch_peak_allocated_gib']:.2f} GiB**，此数不含Warp原生buffer及driver context，未作为完整进程显存峰值。",
        '', '![速度对比](figures/performance.png)',
        '', '## 1e-4以上误差的诊断','', '每个配置/手抽取80帧：各轨迹按时间均匀取样并含该轨迹最大误差帧。以下计数是诊断样本结果，不外推为全数据失败率。SDK种子、SDK骨架和上一帧SDK q仅用于独立的诊断脚本；主速度与主误差结果没有这些输入。',
        '', '1. **左拇指存在初值依赖，符合不同局部盆地。** 主空间目标下，生产解继续优化后，内置左手6/80帧、HL左手5/80帧仍比候选最优代价高。右手对应均为0/80。',
        '2. **全部误差不能归因于网络温启。** 右手不同初值主要汇合。将上游人手骨架替换为SDK骨架后，右手对SDK raw q的RMSE从0.0820降到0.0285 rad（内置）、从0.0627降到0.0239 rad（HL），说明人手重建/状态策略差异影响明显。',
        '3. **匹配历史正则后仍有SDK停止策略差异。** 诊断进一步固定SDK骨架与真实上一帧SDK raw q，加入相同lambda_smooth=.01；即使从SDK自身q出发，继续GPU优化后80/80帧代价均进一步下降。右手最终解相互很接近，但相对SDK raw q仍有约0.0214/0.0177 rad的RMSE。该现象符合有限迭代/停止策略与充分收敛的LM解不同。左手内置条件下生产分支仍有8/80帧代价高于SDK种子分支。',
        '4. **编译没有引入差异，迭代/数值收敛仍有余量。** 普通CUDA残差与Triton编译残差的全部输出逐位相同，最大差为0。另行用FP64 GPU继续优化，个别关节最大可改变0.0398 rad；这包含精度、额外迭代和信赖域策略的变化，不能解释为单纯浮点舍入误差。',
        '主目标保留空间拓扑项和关节限位，但采用独立帧LM，去掉了SDK人手history项及机器人上一解正则；SDK还使用状态化小步/有限预算以及输出低通。两条管线并非对同一个时序优化过程的逐位复刻。诊断样本下拇指Jacobian按1e-7阈值均数值满秩，本次未以“普遍欠约束”解释全部误差。',
        '[初值/上游隔离诊断](basin_diagnosis.json) · [同SDK骨架和历史目标诊断](sdk_objective_diagnosis.json) · [编译逐位一致性](compiled_equivalence.json)',
        '', '## 曲线与缓存','',
        '交互页面可选择配置、左右手、轨迹与20个关节；提供q/SDK叠图、逐关节绝对误差、帧RMSE/最大误差和累计RMSE。全部保留原始点，静态图共20组关节曲线及4组误差曲线。',
        '图中仍有初始化瞬态和局部尖点，尤以左拇指分支变化明显；没有观察到所有轨迹随时间单调发散。这里是录制轨迹的开环逐帧比较，模型没有q反馈。',
        '- [内置左手误差](figures/error_builtin_left.png) · [内置右手误差](figures/error_builtin_right.png)',
        '- [HL左手误差](figures/error_hl_urdf_left.png) · [HL右手误差](figures/error_hl_urdf_right.png)',
        '- [左手2号轨迹20关节/SDK对照](figures/q_builtin_left_2.png)',
        '- `inputs/`：五条原始H5快照；`gt/`：官方SDK q/人手q/骨架/原始EMF与时间戳。',
        '- `parallel_robokit_compiled/`：网络q、LM raw q、LM滤波q、人手q、骨架、目标及代价。',
        '- `basin_controls/`：带明确DIAGNOSTIC_ONLY标记的候选结果；不进入主推理或主计时。',
        '', '## 复现与验证','',
        '核心实现：`retarget/learning/wuji_ik/batched_lm.py`、`robokit_lm.py`；入口脚本及快照位于 `implementation/`。官方SDK保持原样。Robokit源快照名为 `robokit-736f112`，实际文件SHA256见native_cuda manifest。',
        '```bash',
        'PY=/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/miniconda3/envs/pi0/bin/python',
        'export PYTHONPATH=/workspace/jasonzhang/robokit-736f112/src:$PWD',
        '# ROOT指定本报告目录；CKPT指定checkpoint_provenance.json中的canonical_checkpoint',
        'CUDA_VISIBLE_DEVICES=1 "$PY" scripts/benchmark_wuji_parallel_lm.py \\',
        '  --root "$ROOT" --checkpoint "$CKPT" --robokit-native --compile \\',
        '  --run-tag compiled --repeats 3 --memory-gib 10',
        '```',
        '同一RobokitCUDALM实例绑定固定几何、关节限位与batch形状；新形状需要新实例/重新构图。时序模型使用完整段，显存不足直接报错。',
        '验证包括：SDK重复生成q逐位一致；SDK q→URDF FK位置最大差<8e-9 m；四个配置/手的FK、解析Jacobian及空间代价/梯度对照通过；CUDA Graph换输入和跨图重放通过；预热后三次全输出可复现；编译前后逐位一致；完整输出关节限位检查通过。',
        '```text',
        'python -m pytest -q tests/learning/test_wuji_input_selection.py tests/learning/test_wuji_ik_batched_lm.py',
        '4 passed',
        '```','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    (root/'README.md').write_text('# Wuji2 并行 LM 复测\n\n[完整报告](REPORT.md) · [交互曲线](comparison.html)\n\n五条输入质量通过的实录轨迹，5770个单手帧/配置。GPU加速成立；SDK关节角1e-4 rad一致性未达到。\n')
    print('FINAL_REPORT_WRITTEN',json.dumps(summary['speed']),flush=True)


if __name__=='__main__':main()
