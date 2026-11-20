"""Human-readable final results and standalone training figures."""
import json
from pathlib import Path


def build_report(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(folder);report=json.loads((root/'report.json').read_text())
    candidates=json.loads((root/'candidate_reports.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
    for weight,r in sorted(candidates.items(),key=lambda item:float(item[0])):
        name='lambda_'+str(float(weight)).replace('.','p')
        history=json.loads((root/name/'history.json').read_text())
        axes[0].plot([x['epoch'] for x in history],[x['val_rmse_deg'] for x in history],label='lambda='+weight)
        axes[1].plot([x['epoch'] for x in history],[x['reconstruction'] for x in history],label='lambda='+weight)
    axes[0].set(xlabel='Epoch',ylabel='Validation angular RMSE (degrees)',title='Quantized reconstruction')
    axes[1].set(xlabel='Epoch',ylabel='Normalized joint MSE',title='Training reconstruction')
    for ax in axes:ax.grid(alpha=.25);ax.legend()
    fig.savefig(root/'training_curves.png',dpi=160);plt.close(fig)
    types=list(report['per_type']);values=[report['per_type'][name]['mae_deg'] for name in types]
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    ax.bar(range(len(types)),values,color='#3568a8');ax.axhline(1,color='#c6453d',linestyle='--',label='1 degree target')
    ax.set_xticks(range(len(types)),[name.split('_')[0] for name in types]);ax.set(xlabel='Grasp type ID',ylabel='Test MAE (degrees)',title='Held-out reconstruction by grasp type');ax.legend()
    fig.savefig(root/'per_type_error.png',dpi=160);plt.close(fig)
    lines=['# Wuji2 RVQ 训练结果','',f"- 选中语义权重：`{report['selected_weight']}`；模型输入／输出均为20维原生关节角（弧度）。",
        f"- 测试集：{report['samples']:,} 条；MAE {report['mae_deg']:.4f}°，RMSE {report['rmse_deg']:.4f}°，P95 {report['p95_deg']:.4f}°。",
        f"- 重建目标达到：**{report['reconstruction_target_met']}**；语义／重建折中达到：**{report['semantic_tradeoff_met']}**。",
        f"- 固定腕部指尖平均误差：{report['fingertips']['mean_mm']:.4f} mm，P95 {report['fingertips']['p95_mm']:.4f} mm。",
        '- 类型标签仅参与训练损失；独立decoder只需4个离散token。','',
        '## 验证集候选比较','','| λ | MAE ° | RMSE ° | P95 ° | 前两层均衡5-NN | 同/异类距离差 |','|---:|---:|---:|---:|---:|---:|']
    for weight,r in sorted(candidates.items(),key=lambda item:float(item[0])):
        sem=r['semantics'];lines.append(f"| {weight} | {r['mae_deg']:.4f} | {r['rmse_deg']:.4f} | {r['p95_deg']:.4f} | {100*sem['balanced_5nn_accuracy']:.2f}% | {sem['separation_gap']:.4f} |")
    lines+=['','候选仅按验证集选择；测试集不参与权重选择。语义折中要求验证RMSE不超过无语义对照的105%，同时5-NN和同/异类距离差改善。',
        '','![训练曲线](training_curves.png)','','![各抓型误差](per_type_error.png)','','## 测试集分类型结果','',
        '| Grasp type | 数量 | MAE ° | RMSE ° | P95 ° | 指尖 mm |','|---|---:|---:|---:|---:|---:|']
    for name,r in report['per_type'].items():
        lines.append(f"| {name} | {r['samples']} | {r['mae_deg']:.4f} | {r['rmse_deg']:.4f} | {r['p95_deg']:.4f} | {r['fingertip_mean_mm']:.4f} |")
    lines+=['','## 码本使用','','| 层 | 使用码字 | 总码字 | Perplexity |','|---:|---:|---:|---:|']
    for r in report['codebooks']:lines.append(f"| {r['level']+1} | {r['used']} | {r['entries']} | {r['perplexity']:.2f} |")
    lines+=['','完整逐关节及接触／稳定性分组统计见 [report.json](report.json)。本模型仅重建手型，不提供接触或动力学成功保证。',
        '','部署文件：[完整模型](export/model.pt)、[独立decoder](export/decoder.pt)、[接口及关节顺序](export/interface.json)。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return root/'REPORT.md'
