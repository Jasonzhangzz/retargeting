#!/usr/bin/env python3
"""Summarize frozen distilled-model native CUDA LM experiments without training."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    read=lambda n:json.loads((r/n).read_text())
    manifest=read('MANIFEST.json');bench=read('benchmark_robokit_compiled_report.json');sdk=read('sdk_benchmark_components.json');metrics=read('metrics.json');fk=read('fk_and_startup_metrics.json')
    assert len(bench['steady_repeats'])==len(sdk['repeats'])==3
    summary=dict(complete=True,optimizer_updates=0,checkpoint=manifest['checkpoint'],checkpoint_sha256=manifest['checkpoint_sha256'],checkpoint_step=bench['checkpoint_step'],checkpoint_epochs=bench['checkpoint_completed_epochs'],frames_per_configuration=5770,control_threshold_rad=1e-4,speed={},joint_metrics=[x for x in metrics['aggregate'] if x['side']=='all'],pose_metrics={},diagnostics=[])
    for config in ['builtin','hl_urdf']:
        gpu=[sum(g['end_to_end_seconds'] for g in x['groups'] if g['configuration']==config) for x in bench['steady_repeats']]
        gr=[sum(g['robot_lm_seconds']+g['filter_seconds'] for g in x['groups'] if g['configuration']==config) for x in bench['steady_repeats']]
        cpu=[x['configuration_totals'][config]['end_to_end_seconds'] for x in sdk['repeats']]
        cr=[x['configuration_totals'][config]['robot_retarget_seconds'] for x in sdk['repeats']]
        summary['speed'][config]=dict(gpu_full_seconds=float(np.median(gpu)),sdk_full_seconds=float(np.median(cpu)),full_speedup=float(np.median(cpu)/np.median(gpu)),gpu_robot_seconds=float(np.median(gr)),sdk_robot_seconds=float(np.median(cr)),robot_speedup=float(np.median(cr)/np.median(gr)),gpu_full_repeats=gpu,sdk_full_repeats=cpu)
        pose={}
        for method in ['network','native_raw','native_filtered','sdk','native_human_lm']:
            rows=[x for x in fk['metrics'] if x['configuration']==config and x['method']==method];n=sum(x['frames'] for x in rows)
            if method in ['sdk','native_human_lm']:
                pose[method]=dict(human_emf_position_rmse_m=float(np.sqrt(sum(x['human_emf_position_rmse_m']**2*x['frames'] for x in rows)/n)),human_emf_position_max_m=max(x['human_emf_position_max_m'] for x in rows),human_emf_orientation_rmse_rad=float(np.sqrt(sum(x['human_emf_orientation_rmse_rad']**2*x['frames'] for x in rows)/n)))
            else:
                pose[method]=dict(fingertip_difference_rmse_m=float(np.sqrt(sum(x['fingertip_position_rmse_m']**2*x['frames'] for x in rows)/n)),fingertip_difference_max_m=max(x['fingertip_position_max_m'] for x in rows),joint_limit_violation_max_rad=max(x['max_joint_limit_violation_rad'] for x in rows))
        summary['pose_metrics'][config]=pose
    if (r/'basin_diagnosis.json').exists():
        for d in read('basin_diagnosis.json'):
            prod=d['robot_targets']['production'];oracle=d['robot_targets']['sdk_skeleton_DIAGNOSTIC_ONLY']
            summary['diagnostics'].append(dict(configuration=d['configuration'],side=d['side'],frames=d['frames'],network_rmse_production_target_rad=prod['rmse_rad'][0],network_rmse_sdk_target_rad=oracle['rmse_rad'][0],network_vs_best_rmse_rad=prod['network_vs_best_solution_rmse_rad'],network_higher_cost_fraction=prod['network_seed_higher_cost_fraction'],production_higher_cost_fraction=prod['production_seed_higher_cost_fraction'],human_seed_solution_difference_rmse_rad=d['human_seed_solution_difference_rmse_rad']))
    summary['sdk_history_diagnostics']=read('sdk_objective_diagnosis.json') if (r/'sdk_objective_diagnosis.json').exists() else []
    summary['cold_solver_all_configurations_seconds']=bench['cold_warmup_seconds_including_compile_and_result_capture'];summary['process_to_first_all_results_seconds']=bench['first_result_including_import_model_urdf_validation_compilation_and_capture_seconds'];summary['torch_peak_gib']=bench['gpu_peak_allocated_bytes']/2**30
    (r/'summary.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained');names=['network','native_lm_raw','native_lm_filtered'];labels=['Network','Network + LM raw','Network + LM filtered'];colors=['#df9c2a','#336ac2','#cb4f63'];x=np.arange(2)
    for i,(method,label,color) in enumerate(zip(names,labels,colors)):
        vals=[next(v['rmse_rad'] for v in summary['joint_metrics'] if v['configuration']==c and v['method']==method) for c in ['builtin','hl_urdf']];bars=axes[0].bar(x+(i-1)*.23,vals,.23,label=label,color=color);axes[0].bar_label(bars,fmt='%.4f',fontsize=8)
    axes[0].set(xticks=x,xticklabels=['Builtin glove','Fixed HL glove URDF'],ylabel='q RMSE vs SDK (rad)',title='All 5 held-out trajectories / both hands');axes[0].legend(fontsize=8);axes[0].grid(axis='y',alpha=.2)
    for offset,key,label,color in [(-.17,'sdk_full_seconds','Official SDK','#465364'),(.17,'gpu_full_seconds','GPU complete pipeline','#278c76')]:
        vals=[summary['speed'][c][key] for c in ['builtin','hl_urdf']];bars=axes[1].bar(x+offset,vals,.34,label=label,color=color);axes[1].bar_label(bars,fmt='%.3fs')
    axes[1].set(xticks=x,xticklabels=['Builtin glove','Fixed HL glove URDF'],ylabel='Seconds, median of 3',title='5770 hand-frames per configuration, warm');axes[1].legend(fontsize=8);axes[1].grid(axis='y',alpha=.2);fig.savefig(r/'figures/summary.png',dpi=170);plt.close(fig)
    lines=['# 715516帧蒸馏模型 + 原生CUDA并行LM vs Wuji SDK','', '[完整交互轨迹/误差曲线](comparison.html) · [汇总JSON](summary.json) · [逐轨迹CSV](trajectory_metrics.csv) · [关节与FK指标](fk_and_startup_metrics.json)','', '已完成5条独立留出轨迹、左右手共5770帧的测试。内置glove和固定HL glove URDF各运行一次完整数据，均保留全部原始帧。模型为temporal_causal最终第200轮step4200；不训练或更新参数。','', '## 对SDK q的误差','', '|配置|方法|MAE(rad)|RMSE(rad)|最大误差(rad)|','|---|---|---:|---:|---:|']
    for d in summary['joint_metrics']:lines.append(f"|{d['configuration']}|{d['method']}|{d['mae_rad']:.6f}|{d['rmse_rad']:.6f}|{d['max_error_rad']:.6f}|")
    lines+=['','内置glove条件下，LM使对SDK q的平均误差变大；HL条件下，LM显著改善该几何条件下网络的初值误差。两种配置都没有一帧的全部20个关节达到1e-4rad一致性。SDK q是本次参考，不代表真实机器人编码器真值。','', '## 完整流程和q阶段计时','', '|配置|SDK完整流程(s)|GPU完整流程(s)|完整加速|SDK q阶段(s)|GPU LM+滤波(s)|q阶段加速|','|---|---:|---:|---:|---:|---:|---:|']
    for c,s in summary['speed'].items():lines.append(f"|{c}|{s['sdk_full_seconds']:.4f}|{s['gpu_full_seconds']:.4f}|{s['full_speedup']:.2f}×|{s['sdk_robot_seconds']:.4f}|{s['gpu_robot_seconds']:.4f}|{s['robot_speedup']:.2f}×|")
    lines+=['', 'GPU完整流程包含CPU EMF特征构建/传输、批量网络、人手LM、骨架目标准备、Wuji2 LM和并行滤波。SDK完整流程包含相同原始EMF的frame构建、离线pipeline/session初始化、人手IK及Wuji2 retarget。两者均排除磁盘I/O、模型加载、诊断和报表。预热后3次重复中位数；共享A800 GPU2及CPU上实测，原始重复值保留。',f"全部两配置首轮求解含编译/构图/保存预热{summary['cold_solver_all_configurations_seconds']:.2f}s；从进程import到首轮全部结果{summary['process_to_first_all_results_seconds']:.2f}s。PyTorch allocator峰值{summary['torch_peak_gib']:.2f}GiB，不包含Warp原生buffer。",'', '## 物理量比较','', '|配置|方法|Wuji2指尖相对SDK FK的RMSE(mm)|最大距离(mm)|','|---|---|---:|---:|']
    for c,p in summary['pose_metrics'].items():
        for m in ['network','native_raw','native_filtered']:lines.append(f"|{c}|{m}|{p[m]['fingertip_difference_rmse_m']*1000:.3f}|{p[m]['fingertip_difference_max_m']*1000:.3f}|")
    lines+=['','人手EMF拟合是中间人手模型对原始EMF的残差，不是Wuji2对EMF的直接FK；具体位置/旋转残差见summary.json的pose_metrics。所有输出均校验机器人关节限位。','', '## 并行和语义','', '每个几何/左右手分组将5条完整轨迹一次批量网络前向（各自因果注意力，无轨迹间注意力）；每组2885帧。人手三组固定初值合计8655个独立21维系统，机器人2885个独立20维系统；LM迭代依次推进，但每轮全部帧并行，没有前一帧q反馈。Robokit原生CUDA LM/Cholesky/接受拒绝，Torch CUDA残差和解析Jacobian，无CPU求解回退。','', '主求解没有读入SDK中间q或骨架。SDK保留原生状态及输出滤波；独立帧LM去掉人手history和机器人上一解正则，所以空间项一致不等于完整时序目标相同。报告保留raw和alpha=0.2的GPU并行线性滤波输出。','', '## 初值与目标隔离诊断（不进入主结果或速度）','', '每配置/左右手80帧，含每条轨迹最大误差点和均匀时间采样。SDK骨架、SDK初值、上一帧SDK q只用于这些诊断，不参与生产求解或主计时。','', '|配置|手|网络种子在LM骨架上对SDK raw q的RMSE|换SDK骨架后的RMSE|网络种子高于候选最低代价的帧数|','|---|---|---:|---:|---:|']
    for d in summary['diagnostics']:lines.append(f"|{d['configuration']}|{d['side']}|{d['network_rmse_production_target_rad']:.6f}|{d['network_rmse_sdk_target_rad']:.6f}|{round(d['network_higher_cost_fraction']*d['frames'])}/{d['frames']}|")
    lines+=['','[初值/骨架诊断](basin_diagnosis.json) · [固定SDK骨架与上一解的诊断](sdk_objective_diagnosis.json) · [空间目标/梯度与FK一致性](numerical_validation.json)','', '![误差与速度](figures/summary.png)','', '![内置左手逐帧误差](figures/error_builtin_left.png)','', '![内置右手逐帧误差](figures/error_builtin_right.png)']
    (r/'REPORT.md').write_text('\n'.join(lines)+'\n');print('DISTILLED_LM_SUMMARY_COMPLETE',json.dumps(summary['speed']),flush=True)


if __name__=='__main__':main()
