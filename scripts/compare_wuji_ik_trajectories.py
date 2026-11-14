#!/usr/bin/env python3
"""Compare completed causal checkpoints on the exact previously inspected test trajectories."""
import argparse
import base64
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from retarget.learning.wuji_ik.trajectory_diagnostics import analyze_trajectories, bootstrap_mean


COLORS = {'baseline': '#dc872f', 'best': '#1976bb', 'latest': '#239568', 'sdk': '#253442'}
FIXED = [(74, 3, 15, 'Previously typical'), (273, 3, 404, 'Longest'),
         (447, 3, 102, 'Previous worst increment'), (478, 6, 65, 'Previous spike candidate')]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def load_run(root):
    with np.load(root/'trajectories.npz') as source:
        data = {key: source[key] for key in source.files}
    data['export'] = json.loads((root/'export_report.json').read_text())
    data['metadata'] = json.loads((root/'segments.json').read_text())
    data['summary'], data['rows'], data['arrays'] = analyze_trajectories(
        data['q_pred_rad'], data['q_sdk_rad'], data['offsets'])
    data['online'] = {}
    if (root/'online_prefix.npz').exists():
        with np.load(root/'online_prefix.npz') as online:
            for index, start, end in zip(online['indices'], online['offsets'][:-1], online['offsets'][1:]):
                values = online['q_pred_rad'][start:end].astype(float)
                if len(values) != data['offsets'][index+1]-data['offsets'][index]:
                    raise ValueError('Online export and full-segment lengths differ')
                data['online'][int(index)] = np.rad2deg(values)
    for key, array in [('prediction', data['q_pred_rad']), ('sdk', data['q_sdk_rad'])]:
        acc = np.concatenate([np.diff(array[s:e].astype(float), n=2, axis=0)
                              for s, e in zip(data['offsets'][:-1], data['offsets'][1:]) if e-s >= 3])
        data['summary'][key+'_acceleration_rms_deg_per_frame2'] = float(np.rad2deg(np.sqrt(np.square(acc).mean())))
    data['degrees'] = np.rad2deg(data['q_pred_rad'].astype(float))
    return data


def assert_matched(reference, candidate):
    if reference['export']['data_signature'] != candidate['export']['data_signature']:
        raise ValueError('Data signatures differ')
    for key in ('q_sdk_rad', 'frame_ids', 'offsets'):
        np.testing.assert_array_equal(reference[key], candidate[key], err_msg=key+' mismatch')
    # The order is independently checked against the original source mapping.
    if reference['metadata'] != candidate['metadata']:
        raise ValueError('Source mappings or joint order differ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path, help='Contains best/ and latest/ exports')
    parser.add_argument('--baseline', required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    runs = {'baseline': load_run(args.baseline), 'best': load_run(root/'best'), 'latest': load_run(root/'latest')}
    baseline = runs['baseline']; best = runs['best']
    for run in runs.values():
        assert_matched(baseline, run)
        step = run['export']['checkpoint_step']
        run['label'] = ('Old baseline' if run is baseline else 'Validation best' if run is best else 'Final')+f' ({step})'
    offsets = baseline['offsets']; truth = np.rad2deg(baseline['q_sdk_rad'].astype(float))
    metadata = baseline['metadata']
    comparison = dict(generated_at=datetime.now(timezone.utc).isoformat(),
        matching='Exact equality of SDK q, frame IDs, offsets, all source mappings/joint orders, and data signature.',
        protocol='Same complete held-out segments; no q feedback or closed-loop robot rollout. GT means SDK teacher q.',
        limitation='Old checkpoint is an intermediate unsmoothed run. New runs also differ in training duration and learning-rate schedule; this is not a controlled smoothing-only ablation.',
        runs={}, fixed_trajectories=[], new_worst_cases=[])
    for name, run in runs.items():
        comparison['runs'][name] = dict(export=run['export'], diagnostics=run['summary'])
        if name != 'baseline':
            paired = np.asarray([row['mae_deg']-old['mae_deg'] for row, old in zip(run['rows'], baseline['rows'])])
            comparison['runs'][name]['paired_segment_mae_change_deg'] = bootstrap_mean(paired)
            destination = root/name
            write_json(destination/'diagnostics_summary.json', run['summary'])
            write_json(destination/'trajectory_metrics.json', run['rows'])
            with (destination/'trajectory_metrics.csv').open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(run['rows'][0])); writer.writeheader(); writer.writerows(run['rows'])
        online_report = []
        for index, values in run['online'].items():
            start, end = offsets[index:index+2]
            online_report.append(dict(index=index, frames=len(values),
                mae_deg=float(np.abs(values-truth[start:end]).mean()),
                maximum_difference_from_parallel_deg=float(np.abs(values-run['degrees'][start:end]).max())))
        comparison['runs'][name]['online_prefix_comparison'] = online_report
    for index, joint, event, label in FIXED:
        start, end = offsets[index:index+2]
        item = dict(index=index, joint=metadata[index]['joint_names'][joint], event_frame=event,
                    description=label, source=metadata[index], runs={})
        for name, run in runs.items():
            residual = run['degrees'][start:end]-truth[start:end]
            info = dict(run['rows'][index], selected_event_delta_error_deg=float(np.diff(residual, axis=0)[event-1, joint]))
            if index in run['online']:
                info['online_selected_event_delta_error_deg'] = float(np.diff(run['online'][index]-truth[start:end], axis=0)[event-1, joint])
            item['runs'][name] = info
        comparison['fixed_trajectories'].append(item)
    spike = max(best['rows'], key=lambda r: r['max_spike_strength_deg'])
    final_worst = max(runs['latest']['rows'], key=lambda r: r['max_delta_error_deg'])
    new_cases = [(spike['index'], spike['worst_spike_joint'], spike['worst_spike_frame'], 'Largest isolated error spike'),
                 (final_worst['index'], final_worst['worst_delta_joint'], final_worst['worst_delta_frame'], 'Final model largest increment error')]
    for index, joint, event, label in new_cases:
        start, end = offsets[index:index+2]
        comparison['new_worst_cases'].append(dict(index=index, joint=metadata[index]['joint_names'][joint],
            event_frame=event, description=label, source=metadata[index],
            runs={name:dict(run['rows'][index],
                selected_event_delta_error_deg=float(np.diff(run['degrees'][start:end]-truth[start:end],axis=0)[event-1,joint]),
                online_selected_event_delta_error_deg=float(np.diff(run['online'][index]-truth[start:end],axis=0)[event-1,joint])
                if index in run['online'] else None) for name,run in runs.items()}))
    write_json(root/'comparison_summary.json', comparison)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'figure.facecolor':'#f6f8fc', 'axes.facecolor':'white'})
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    for name, run in runs.items():
        summary = run['summary']; color = COLORS[name]
        progress = summary['normalized_progress']; x = (np.arange(20)+.5)*5
        axes[0, 0].plot(x, progress['mean_deg'], color=color, label=run['label'])
        axes[0, 0].fill_between(x, progress['ci95_low_deg'], progress['ci95_high_deg'], color=color, alpha=.12)
        fixed = summary['fixed_history_cohort']; x = np.arange(fixed['horizon'])
        axes[0, 1].plot(x, fixed['mean_deg'], color=color, label=run['label'])
        axes[0, 1].fill_between(x, fixed['ci95_low_deg'], fixed['ci95_high_deg'], color=color, alpha=.10)
        values = np.sort(np.max(np.abs(run['arrays']['delta_error_deg']), axis=1))
        axes[0, 2].plot(values, 1-np.arange(len(values))/len(values), color=color, label=run['label'])
    axes[0, 0].set(title='Same trajectories: error vs normalized progress (95% CI)', xlabel='Trajectory progress (%)', ylabel='MAE (deg)')
    fixed = best['summary']['fixed_history_cohort']
    axes[0, 1].set(title=f'Same {fixed["segments"]} trajectories: first {fixed["horizon"]} frames', xlabel='Frames since segment start', ylabel='MAE (deg)')
    axes[0, 2].set(title='SDK-relative increment residual survival', xlabel='Any-joint threshold (deg / frame)', ylabel='Fraction exceeding', xlim=(0, 20), ylim=(1e-5, 1))
    axes[0, 2].set_yscale('log'); axes[0, 2].axvline(5, color='gray', ls=':')
    for ax in axes[0]: ax.legend(fontsize=8); ax.grid(alpha=.2)
    old = np.array([r['mae_deg'] for r in baseline['rows']])
    for name in ('best', 'latest'):
        values = [r['mae_deg'] for r in runs[name]['rows']]
        axes[1, 0].scatter(old, values, s=10, alpha=.35, label=runs[name]['label'], color=COLORS[name])
    limit = max(ax for ax in [old.max()]+[max(r['mae_deg'] for r in runs[n]['rows']) for n in ('best', 'latest')])*1.04
    axes[1, 0].plot([0, limit], [0, limit], 'k--', lw=1)
    axes[1, 0].set(title='Paired error on each identical trajectory', xlabel='Old segment MAE (deg)', ylabel='New segment MAE (deg)')
    axes[1, 0].legend(fontsize=8)
    for i, (name, run) in enumerate(runs.items()):
        drift = run['summary']['paired_drift']['late_minus_early_deg']; mean = drift['mean']; lo, hi = drift['ci95']
        axes[1, 1].errorbar(i, mean, yerr=np.array([[mean-lo], [hi-mean]]), fmt='o', capsize=6, color=COLORS[name])
    axes[1, 1].set_xticks(range(3), [r['label'] for r in runs.values()], rotation=15)
    axes[1, 1].set(title='Last 20% minus first 20% MAE (95% trajectory CI)', ylabel='Late - early error (deg)')
    axes[1, 1].axhline(0, color='gray', ls='--')
    labels = ['SDK']+[run['label'] for run in runs.values()]
    values = [best['summary']['sdk_acceleration_rms_deg_per_frame2']]+[run['summary']['prediction_acceleration_rms_deg_per_frame2'] for run in runs.values()]
    axes[1, 2].bar(labels, values, color=[COLORS['sdk']]+[COLORS[k] for k in runs])
    axes[1, 2].tick_params(axis='x', labelrotation=15)
    axes[1, 2].set(title='Acceleration energy (lower alone does not imply better fit)', ylabel='RMS second difference (deg / frame²)')
    fig.suptitle(f'Completed temporal causal: {len(metadata)} identical held-out trajectories / {len(truth):,} frames\nSDK teacher comparison, no joint feedback; baseline is not a controlled smoothing-only ablation', fontsize=14)
    fig.savefig(root/'comparison_overview.png', dpi=155); plt.close(fig)

    fig, axes = plt.subplots(4, 3, figsize=(20, 15), constrained_layout=True)
    for (index, joint, event, label), (ax, bx, cx) in zip(FIXED, axes):
        start, end = offsets[index:index+2]; x = np.arange(end-start)
        ax.plot(x, truth[start:end, joint], color=COLORS['sdk'], label='SDK GT', lw=1.5)
        for name, run in runs.items():
            values = run['degrees'][start:end]; residual = values-truth[start:end]
            ax.plot(x, values[:, joint], color=COLORS[name], label=run['label'], lw=1.1, alpha=.9)
            bx.plot(x, np.abs(residual).mean(axis=1), color=COLORS[name], label=run['label'], lw=1)
            cx.plot(x[1:], np.diff(residual[:, joint]), color=COLORS[name], label=run['label'], lw=1)
        for plot in (ax, bx, cx): plot.axvline(event, color='#b74c55', ls=':', lw=1); plot.grid(alpha=.2)
        ax.set(title=f'{label} #{index}: {metadata[index]["joint_names"][joint]}', ylabel='Joint angle (deg)', xlabel='Local frame')
        bx.set(title=f'#{index}: 20-joint frame MAE', ylabel='Error (deg)', xlabel='Local frame')
        cx.set(title=f'#{index}: selected joint increment residual', ylabel='Delta error (deg / frame)', xlabel='Local frame')
        ax.legend(fontsize=7)
    fig.suptitle('Exact same four trajectories and joints as the previous inspection | dotted line = previously selected event\nNo trajectories were reselected to favor the new model', fontsize=14)
    fig.savefig(root/'same_trajectories.png', dpi=150); plt.close(fig)
    index, joint, event = 447, 3, 102; start, end = offsets[index:index+2]; x = np.arange(end-start)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True, sharex=True)
    axes[0].plot(x, truth[start:end, joint], label='SDK GT', color=COLORS['sdk'], lw=2)
    for name, run in runs.items():
        values = run['degrees'][start:end, joint]; residual = values-truth[start:end, joint]
        axes[0].plot(x, values, label=run['label'], color=COLORS[name])
        axes[1].plot(x, residual, label=run['label'], color=COLORS[name])
        axes[2].plot(x[1:], np.diff(residual), label=run['label'], color=COLORS[name])
        if index in run['online']:
            online = run['online'][index][:, joint]
            axes[0].plot(x, online, label=run['label']+' actual prefix', color=COLORS[name], ls=':', alpha=.8)
    for ax, ylabel in zip(axes, ['Joint angle (deg)', 'Signed error (deg)', 'Increment residual (deg / frame)']):
        ax.set_ylabel(ylabel); ax.axvline(event, color='#b74c55', ls=':'); ax.set_xlim(event-20, event+20); ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[-1].set_xlabel('Local frame')
    fig.suptitle('Previous worst jump revisited: same trajectory #447, r_thumb_ip, frame 102\nSolid = whole-segment causal parallel inference; dotted = actual growing EMF prefix calls', fontsize=14)
    fig.savefig(root/'previous_worst_jump.png', dpi=165); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(19, 9), constrained_layout=True)
    for (index, joint, event, label), (ax, bx, cx) in zip(new_cases, axes):
        start, end = offsets[index:index+2]; x = np.arange(end-start)
        ax.plot(x, truth[start:end,joint], label='SDK GT', color=COLORS['sdk'], lw=2)
        for name, run in runs.items():
            values = run['degrees'][start:end,joint]; residual = values-truth[start:end,joint]
            ax.plot(x, values, label=run['label'], color=COLORS[name])
            bx.plot(x, residual, label=run['label'], color=COLORS[name])
            cx.plot(x[1:], np.diff(residual), label=run['label'], color=COLORS[name])
            if index in run['online']:
                online = run['online'][index][:,joint]
                ax.plot(x, online, color=COLORS[name], ls=':', alpha=.8, label=run['label']+' actual prefix')
        for plot, ylabel in zip((ax,bx,cx), ['Joint angle (deg)','Signed error (deg)','Increment residual (deg / frame)']):
            plot.set(xlim=(max(0,event-15), min(len(x)-1,event+15)), xlabel='Local frame', ylabel=ylabel)
            plot.axvline(event, color='#b74c55', ls=':'); plot.grid(alpha=.2)
        ax.set_title(f'{label}\n#{index}, frame {event}, {metadata[index]["joint_names"][joint]}')
        ax.legend(fontsize=7)
    fig.suptitle('Remaining exceptions after training | same SDK target for all versions\nA lower average error or acceleration energy does not establish that large isolated errors are gone',fontsize=14)
    fig.savefig(root/'remaining_worst_cases.png',dpi=155); plt.close(fig)

    payload = []
    for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        item = dict(index=i, meta=metadata[i], frames=baseline['frame_ids'][start:end].tolist(),
                    q=np.round(truth[start:end], 5).tolist(), stats={}, online={})
        for name, run in runs.items():
            item[name] = np.round(run['degrees'][start:end], 5).tolist()
            item['stats'][name] = run['rows'][i]
            if i in run['online']: item['online'][name] = np.round(run['online'][i], 5).tolist()
        payload.append(item)
    picks = {key: max(range(len(metadata)), key=lambda i: best['rows'][i][field]) for key, field in
             [('worst', 'max_delta_error_deg'), ('growth', 'late_minus_early_deg'), ('spike', 'max_spike_strength_deg')]}
    picks.update(old_worst=447, longest=273, old_typical=74, old_spike=478, final_worst=final_worst['index'])
    table = []
    for name, run in runs.items():
        summary = run['summary']; threshold = summary['thresholds_degrees_per_frame']; drift = summary['paired_drift']['late_minus_early_deg']
        table.append(f'<tr><td>{run["label"]}</td><td>{summary["mae_deg"]:.3f}°</td><td>{summary["rmse_deg"]:.3f}°</td>'
                     f'<td>{threshold["5"]["frame_pair_counts"]["delta_error"]} ({100*threshold["5"]["frame_pair_rates"]["delta_error"]:.4f}%)</td>'
                     f'<td>{threshold["10"]["frame_pair_counts"]["delta_error"]}</td><td>{summary["delta_error_joint_deg"]["max"]:.3f}°</td><td>{summary["isolated_spike_frames_over_5deg"]}</td>'
                     f'<td>{drift["mean"]:+.3f}° [{drift["ci95"][0]:+.3f}, {drift["ci95"][1]:+.3f}]</td></tr>')
    prefix_reports = comparison['runs']['best']['online_prefix_comparison']
    prefix_max = max((r['maximum_difference_from_parallel_deg'] for r in prefix_reports), default=0)
    final_prefix_reports = comparison['runs']['latest']['online_prefix_comparison']
    final_prefix_max = max((r['maximum_difference_from_parallel_deg'] for r in final_prefix_reports), default=0)
    template = r'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>训练完成：EMF → Wuji2 同轨迹对比</title>
<style>body{margin:0;background:#f3f6fa;color:#172536;font:15px system-ui,sans-serif}main{max-width:1550px;margin:auto;padding:24px}.card{background:white;border-radius:12px;padding:18px;margin:16px 0;box-shadow:0 2px 8px #1725360c}.note{line-height:1.75;color:#4b5d70}table{width:100%;border-collapse:collapse}td,th{padding:12px;text-align:left;border-bottom:1px solid #dae2eb}select,button{font:inherit;padding:8px;margin:5px;border:1px solid #c6d1de;border-radius:6px;background:white}button{cursor:pointer}.plot{height:370px}#heat{height:520px}img{width:100%}#source{overflow-wrap:anywhere}@media(max-width:800px){main{padding:8px}.card{overflow:auto}}</style>
<main><h1>训练完成：同轨迹与 SDK GT 对比</h1><p class="note">temporal_causal：200 epochs / 5000 steps 已完成。验证集选择最佳 checkpoint，最后一步模型也独立测试。TESTSIZE。<br>GT 指 SDK 教师 q，误差不是实测手部误差或机器人闭环 rollout；模型只输入当前及历史 EMF，不回馈 q。每段完整推理，不跨缺帧或 reset。</p>
<div class="card"><table><thead><tr><th>模型（step）</th><th>MAE</th><th>RMSE</th><th>增量误差 &gt;5°</th><th>&gt;10°</th><th>最大增量误差</th><th>孤立误差尖点 &gt;5°</th><th>后20% − 前20% MAE（95% CI）</th></tr></thead><tbody>TABLE</tbody></table>
<p class="note">增量误差 = Δq_pred − Δq_SDK，计数以任一关节超过阈值的相邻帧对为单位，总计 PAIRS 对。孤立尖点要求同一关节误差左右增量反向且两侧都超过 5°。CI 按轨迹 bootstrap；后段−前段是相同 ≥40 帧轨迹的配对比较，不表示逐帧单调增长。<br>旧模型是中途 step 2480；新模型同时改变了训练时长、学习率与平滑项，不能将差异全部归因于平滑 loss。</p></div>
<div class="card note">所有版本的 SDK 标签、帧号、段边界、关节顺序、来源映射及数据签名已逐项确认完全一致。之前展示的 #74、#273、#447、#478 保持不变。最佳模型的 PREFIXCOUNT 条、最终模型的 FINALPREFIXCOUNT 条轨迹另外执行了实际逐帧 [0:t] prefix 推理，包含之前的 4 条和最终模型新的最大跳变。BF16 矩阵尺寸不同，与完整段并行预测的最大单点差分别为 PREFIXMAX° / FINALPREFIXMAX°。这是数值差异，不能用来证明时间泄漏；训练模型因果遮罩保持不变。</div>
<div class="card"><details open><summary>全测试集：误差趋势、跳变、加速度与配对比较</summary><img src="data:image/png;base64,OVERVIEW"></details><details><summary>之前相同的 4 条轨迹</summary><img src="data:image/png;base64,EXAMPLES"></details><details><summary>旧模型最大跳变局部对比</summary><img src="data:image/png;base64,JUMP"></details><details><summary>训练后仍存在的孤立尖点及最终模型最大跳变</summary><img src="data:image/png;base64,REMAINING"></details></div>
<div class="card"><label>新模型版本 <select id="model"><option value="best">验证集最佳</option><option value="latest">最后一步 5000</option></select></label><label>轨迹 <select id="seq"></select></label><label>关节 <select id="joint"></select></label><br>
<button data-key="old_typical">之前典型 #74</button><button data-key="longest">最长 #273</button><button data-key="old_worst">之前最大跳变 #447</button><button data-key="old_spike">之前尖点候选 #478</button><button data-key="worst">最佳模型最大增量误差</button><button data-key="spike">最佳模型最大尖点候选</button><button data-key="growth">最佳模型后段恶化最大</button><button data-key="final_worst">最终模型最大跳变</button><button id="zoom">放大旧模型跳变处</button><button id="reset">完整轨迹</button><p id="source" class="note"></p><p id="stat" class="note"></p></div>
<div class="card"><div id="angles" class="plot"></div><div id="errors" class="plot"></div><div id="delta" class="plot"></div><div id="heat"></div></div>
<p class="note">横轴为段内帧号，未假定 FPS。Running MAE 是起点至当前的平均，不是累积误差和。误差进度图使用同一批轨迹等权比较；固定帧号趋势图仅使用同一批长轨迹。两种新版本选定轨迹均有实际 prefix 细线，其余轨迹采用整段因果并行计算。JSON / CSV / NPZ 保存原始精度；交互图保留 0.00001°。</p></main>
<script>PLOTLY</script><script>
const DATA=PAYLOAD,PICKS=PICKSJSON,LABELS=LABELSJSON;const $=id=>document.getElementById(id),cfg={responsive:true,displaylogo:false};
const seq=$('seq'),joint=$('joint'),model=$('model');DATA.forEach((d,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`#${i} | ${d.meta.dataset} ${d.meta.side_name} | ${d.q.length} frames`;seq.appendChild(o)});
function line(x,y,name,color,dash){return {x,y,name,type:'scatter',mode:'lines',line:{color,width:1.5,dash}}}
function layout(title,y){return {title:{text:title,font:{size:15}},margin:{l:65,r:20,t:70,b:45},xaxis:{title:'段内帧序号'},yaxis:{title:y},legend:{orientation:'h',y:1.22},hovermode:'x unified'}}
function changeSeq(){const d=DATA[+seq.value];joint.replaceChildren();d.meta.joint_names.forEach((n,i)=>{const o=document.createElement('option');o.value=i;o.textContent=n;joint.appendChild(o)});joint.value=d.stats.baseline.worst_delta_joint;render()}
function render(){const d=DATA[+seq.value],m=model.value,j=+joint.value,n=d.q.length,x=Array.from({length:n},(_,i)=>i),p=d[m].map(v=>v[j]),old=d.baseline.map(v=>v[j]),q=d.q.map(v=>v[j]),error=p.map((v,i)=>v-q[i]),oe=old.map((v,i)=>v-q[i]);
const mae=v=>v.map((row,i)=>row.reduce((s,a,k)=>s+Math.abs(a-d.q[i][k]),0)/20),e=mae(d[m]),om=mae(d.baseline);let sum=0;const running=e.map((v,i)=>(sum+=v)/(i+1));
let traces=[line(x,q,'SDK GT','#253442'),line(x,old,LABELS.baseline,'#dc872f'),line(x,p,LABELS[m],m==='best'?'#1976bb':'#239568')];if(d.online[m])traces.push(line(x,d.online[m].map(v=>v[j]),'新模型实际 prefix','#a64d9e','dot'));if(d.online.baseline)traces.push(line(x,d.online.baseline.map(v=>v[j]),'旧模型实际 prefix','#aa7435','dot'));
Plotly.react('angles',traces,layout(d.meta.joint_names[j]+'：同轨迹与 SDK GT','角度 (°)'),cfg);
traces=[line(x,om,'旧模型逐帧 MAE','#dc872f'),line(x,e,'新模型逐帧 MAE','#1976bb'),line(x,running,'新模型 running MAE','#253442','dash'),line(x,error,'新模型选定关节有符号误差','#b74c55')];if(d.online[m])traces.push(line(x,mae(d.online[m]),'新模型实际 prefix MAE','#a64d9e','dot'));
Plotly.react('errors',traces,layout('逐帧误差与运行平均（MAE 对 20 关节取平均）','误差 (°)'),cfg);
const diff=v=>v.slice(1).map((a,i)=>a-v[i]);Plotly.react('delta',[line(x.slice(1),diff(q),'SDK Δq','#253442'),line(x.slice(1),diff(oe),'旧模型增量误差','#dc872f'),line(x.slice(1),diff(error),'新模型增量误差','#1976bb')],layout('跳变：Δq_pred − Δq_SDK（同一选定关节）','增量 (°/frame)'),cfg);
const h=layout('新模型全部关节绝对误差','关节');h.yaxis={automargin:true};h.margin.l=210;
Plotly.react('heat',[{type:'heatmap',x,y:d.meta.joint_names,z:Array.from({length:20},(_,k)=>d[m].map((v,i)=>Math.abs(v[k]-d.q[i][k]))),colorscale:'YlOrRd',colorbar:{title:'误差 °'},hovertemplate:'frame %{x}<br>%{y}<br>%{z:.3f}°<extra></extra>'}],h,cfg);
$('source').textContent=`test index ${d.index} · global segment ${d.meta.segment_id} · ${d.meta.source_id} · 原始帧 ${d.frames[0]}–${d.frames[n-1]}`;
const s=d.stats[m],b=d.stats.baseline;$('stat').textContent=`旧 → 新：MAE ${b.mae_deg.toFixed(3)}° → ${s.mae_deg.toFixed(3)}°；RMSE ${b.rmse_deg.toFixed(3)}° → ${s.rmse_deg.toFixed(3)}°；后段−前段 ${b.late_minus_early_deg.toFixed(3)}° → ${s.late_minus_early_deg.toFixed(3)}°；最大增量误差 ${b.max_delta_error_deg.toFixed(3)}° → ${s.max_delta_error_deg.toFixed(3)}°/frame（新模型 frame ${s.worst_delta_frame}, ${d.meta.joint_names[s.worst_delta_joint]}）`;
}
seq.onchange=changeSeq;joint.onchange=render;model.onchange=render;document.querySelectorAll('[data-key]').forEach(b=>b.onclick=()=>{if(b.dataset.key==='final_worst')model.value='latest';else if(['worst','spike','growth'].includes(b.dataset.key))model.value='best';seq.value=PICKS[b.dataset.key];changeSeq();if(b.dataset.key==='final_worst')joint.value=DATA[+seq.value].stats.latest.worst_delta_joint;else if(b.dataset.key==='spike')joint.value=DATA[+seq.value].stats.best.worst_spike_joint;render()});
$('zoom').onclick=()=>{const d=DATA[+seq.value];joint.value=d.stats.baseline.worst_delta_joint;render();const t=d.stats.baseline.worst_delta_frame;['angles','errors','delta','heat'].forEach(id=>Plotly.relayout(id,{'xaxis.range':[Math.max(0,t-20),Math.min(d.q.length-1,t+20)]}))};
$('reset').onclick=()=>['angles','errors','delta','heat'].forEach(id=>Plotly.relayout(id,{'xaxis.autorange':true}));seq.value=PICKS.old_worst;changeSeq();
</script></html>'''
    try:
        from plotly.offline import get_plotlyjs
    except ImportError:
        sys.path.insert(0, '/tmp/wuji_rollout_plotly')
        from plotly.offline import get_plotlyjs
    # Replace large payloads last to avoid rescanning/replacing any user-controlled source metadata.
    for key, value in dict(TABLE=''.join(table), TESTSIZE=f'测试集 {len(metadata)} 条连续段 / {len(truth):,} 帧',
        PAIRS=str(best['summary']['consecutive_frame_pairs']), FINALPREFIXMAX=f'{final_prefix_max:.3f}',
        PREFIXMAX=f'{prefix_max:.3f}', FINALPREFIXCOUNT=str(len(final_prefix_reports)), PREFIXCOUNT=str(len(prefix_reports)),
        PICKSJSON=json.dumps(picks), LABELSJSON=json.dumps({k:r['label'] for k,r in runs.items()})).items():
        template = template.replace(key, value)
    for token, filename in [('OVERVIEW', 'comparison_overview.png'), ('EXAMPLES', 'same_trajectories.png'), ('JUMP', 'previous_worst_jump.png'), ('REMAINING','remaining_worst_cases.png')]:
        template = template.replace(token, base64.b64encode((root/filename).read_bytes()).decode())
    template = template.replace('PLOTLY', get_plotlyjs())
    template = template.replace('PAYLOAD', json.dumps(payload, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c'))
    (root/'report.html').write_text(template)
    print(json.dumps({k:dict(step=r['export']['checkpoint_step'],mae=r['summary']['mae_deg'],rmse=r['summary']['rmse_deg']) for k,r in runs.items()}, indent=2))
    print('Report:', root/'report.html')


if __name__ == '__main__':
    main()
