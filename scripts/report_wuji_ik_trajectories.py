#!/usr/bin/env python3
"""Build static plots and a self-contained trajectory inspection report."""
import argparse
import base64
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from retarget.learning.wuji_ik.trajectory_diagnostics import analyze_trajectories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root
    a = np.load(root/'trajectories.npz')
    p, q, offsets = a['q_pred_rad'], a['q_sdk_rad'], a['offsets']
    pd, qd = np.rad2deg(p), np.rad2deg(q)
    summary, rows, arrays = analyze_trajectories(p, q, offsets)
    metadata = json.loads((root/'segments.json').read_text())
    export = json.loads((root/'export_report.json').read_text())
    (root/'diagnostics_summary.json').write_text(json.dumps(summary, indent=2))
    (root/'trajectory_metrics.json').write_text(json.dumps(rows, indent=2))
    with (root/'trajectory_metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    online = {}
    if (root/'online_prefix.npz').exists():
        b = np.load(root/'online_prefix.npz')
        for i, start, end in zip(b['indices'], b['offsets'][:-1], b['offsets'][1:]):
            online[int(i)] = np.rad2deg(b['q_pred_rad'][start:end]).astype(float)
        comparisons = []
        for i, values in online.items():
            start, end = offsets[i:i+2]
            error = values-qd[start:end]
            comparisons.append(dict(index=i, frames=len(values), mae_deg=float(np.abs(error).mean()),
                rmse_deg=float(np.sqrt(np.square(error).mean())),
                maximum_difference_from_parallel_deg=float(np.abs(values-pd[start:end]).max())))
        (root/'online_comparison.json').write_text(json.dumps(comparisons, indent=2))
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'figure.facecolor':'#f7f9fc','axes.facecolor':'white'})
    fig, axes = plt.subplots(2, 3, figsize=(17,9), constrained_layout=True)
    pr = summary['normalized_progress']; x = (np.arange(20)+.5)*5
    ax = axes[0,0]
    ax.fill_between(x, pr['ci95_low_deg'], pr['ci95_high_deg'], alpha=.2, color='#187cc0')
    ax.plot(x, pr['mean_deg'], label='Mean + 95% trajectory bootstrap CI')
    ax.plot(x, pr['median_deg'], '--', label='Median')
    ax.set(title=f'Error vs trajectory progress (same {pr["segments"]} trajectories)', xlabel='Trajectory progress (%)', ylabel='Frame MAE (deg)'); ax.legend(fontsize=8)
    pr = summary['fixed_history_cohort']; x = np.arange(pr['horizon'])
    ax = axes[0,1]
    ax.fill_between(x,pr['ci95_low_deg'],pr['ci95_high_deg'],alpha=.2)
    ax.plot(x,pr['mean_deg'])
    ax.set(title=f'Fixed cohort: {pr["segments"]} trajectories, first {pr["horizon"]} frames',xlabel='Frames since segment start',ylabel='Frame MAE (deg)')
    eligible = [r for r in rows if r['frames']>=40]
    early = np.array([r['early_mae_deg'] for r in eligible]); late = np.array([r['late_mae_deg'] for r in eligible])
    ax = axes[0,2]; ax.scatter(early,late,s=11,alpha=.4)
    lim = max(early.max(),late.max())*1.04; ax.plot([0,lim],[0,lim],'k--',lw=1)
    ax.set(title='Paired early / late error per trajectory',xlabel='First 20% MAE (deg)',ylabel='Last 20% MAE (deg)',xlim=(0,lim),ylim=(0,lim))
    ax = axes[1,0]
    for key,label in [('prediction_delta_deg','Prediction'),('sdk_delta_deg','SDK teacher'),('delta_error_deg','Increment residual')]:
        vals = np.sort(np.max(np.abs(arrays[key]),axis=1))
        ax.plot(vals, 1-np.arange(len(vals))/len(vals), label=label)
    ax.set_yscale('log'); ax.set(xlim=(0,25),ylim=(1e-5,1),title='Any-joint jump survival across adjacent frames',xlabel='Threshold (deg / frame)',ylabel='Fraction exceeding threshold'); ax.legend(fontsize=8)
    ax.axvline(5,color='gray',ls=':',lw=1); ax.axvline(10,color='gray',ls=':',lw=1)
    ax = axes[1,1]; ranked = sorted(rows,key=lambda r:r['max_delta_error_deg'],reverse=True)[:15]
    ax.bar(range(len(ranked)),[r['max_delta_error_deg'] for r in ranked],color='#e58e40')
    ax.set_xticks(range(len(ranked)),[r['index'] for r in ranked],rotation=60)
    ax.set(title='Largest SDK-relative increment jumps',xlabel='Test trajectory index',ylabel='Max increment residual (deg / frame)')
    ax = axes[1,2]; ax.hist(late-early,bins=45,color='#187cc0',alpha=.8); ax.axvline(0,color='k',ls='--')
    d = summary['paired_drift']['late_minus_early_deg']
    ax.axvline(d['mean'],color='#e58e40',lw=2)
    ax.set(title=f'Late minus early: mean {d["mean"]:+.3f} deg\n95% CI [{d["ci95"][0]:+.3f}, {d["ci95"][1]:+.3f}]',xlabel='Last 20% MAE - first 20% MAE (deg)',ylabel='Trajectories')
    fig.suptitle(f'Temporal causal | checkpoint {export["checkpoint_step"]} | {len(rows)} test trajectories, {len(p):,} frames\nEMF-only causal sequence prediction; no joint feedback; error is relative to SDK labels',fontsize=15)
    fig.savefig(root/'overview.png',dpi=160); plt.close(fig)
    typical = min(range(len(rows)),key=lambda i:abs(rows[i]['mae_deg']-np.median([r['mae_deg'] for r in rows])))
    longest = max(range(len(rows)),key=lambda i:rows[i]['frames'])
    worst = max(range(len(rows)),key=lambda i:rows[i]['max_delta_error_deg'])
    spike = max(range(len(rows)),key=lambda i:rows[i]['max_spike_strength_deg'])
    growth = max((i for i in range(len(rows)) if rows[i]['frames']>=40),key=lambda i:rows[i]['late_minus_early_deg'])
    selected = [('Typical trajectory',typical),('Longest trajectory',longest),('Largest increment residual',worst),('Largest isolated spike candidate',spike)]
    fig, axes = plt.subplots(4,2,figsize=(16,14),constrained_layout=True)
    for (label,i),(ax,bx) in zip(selected,axes):
        start,end = offsets[i:i+2]; row=rows[i]; j=row['worst_spike_joint'] if i==spike else row['worst_delta_joint']
        x=np.arange(end-start); pred=pd[start:end]; truth=qd[start:end]; e=np.abs(pred-truth).mean(axis=1)
        ax.plot(x,truth[:,j],label='SDK',color='black',lw=1.6);ax.plot(x,pred[:,j],label='Parallel causal',color='#187cc0',lw=1.2)
        if i in online: ax.plot(x,online[i][:,j],label='Actual growing prefix',color='#e58e40',lw=.9,alpha=.85)
        event=row['worst_spike_frame'] if i==spike else row['worst_delta_frame']
        ax.axvline(event,color='red',ls=':',lw=1)
        ax.set(title=f'{label} #{i} | {metadata[i]["dataset"]}, {metadata[i]["side_name"]}\n{metadata[i]["joint_names"][j]}',xlabel='Local frame',ylabel='Joint angle (deg)');ax.legend(fontsize=8)
        bx.plot(x,e,label='Frame MAE (20 joints)',color='#187cc0');bx.plot(x,np.cumsum(e)/(x+1),label='Running mean MAE',color='black',ls='--')
        if i in online: bx.plot(x,np.abs(online[i]-truth).mean(axis=1),color='#e58e40',lw=.8,label='Actual prefix MAE')
        bx.axvline(event,color='red',ls=':',lw=1);bx.set(title=f'MAE {row["mae_deg"]:.2f} deg; late - early {row["late_minus_early_deg"]:+.2f} deg',xlabel='Local frame',ylabel='Error (deg)');bx.legend(fontsize=8)
    fig.suptitle('Representative and deliberately selected worst cases | red line = selected event\nRunning MAE is a mean, not a cumulative sum; all panels use actual frame indices',fontsize=14)
    fig.savefig(root/'examples.png',dpi=150);plt.close(fig)
    i=worst;start,end=offsets[i:i+2];r=rows[i];j=r['worst_delta_joint'];event=r['worst_delta_frame'];x=np.arange(end-start)
    fig,axes=plt.subplots(3,1,figsize=(13,9),constrained_layout=True,sharex=True)
    axes[0].plot(x,qd[start:end,j],label='SDK',color='black');axes[0].plot(x,pd[start:end,j],label='Parallel causal')
    if i in online:axes[0].plot(x,online[i][:,j],label='Actual growing prefix',color='#e58e40',lw=1)
    axes[0].set_ylabel('Angle (deg)');axes[0].legend()
    axes[1].plot(x,pd[start:end,j]-qd[start:end,j],label='Signed prediction error');axes[1].axhline(0,color='gray',lw=.8);axes[1].set_ylabel('Error (deg)')
    for v,label in [(pd[start:end,j],'Prediction delta'),(qd[start:end,j],'SDK delta'),(pd[start:end,j]-qd[start:end,j],'Increment residual')]:axes[2].plot(x[1:],np.diff(v),label=label)
    axes[2].set(ylabel='Delta (deg / frame)',xlabel='Local frame');axes[2].legend()
    for ax in axes:ax.axvline(event,color='red',ls=':');ax.set_xlim(max(0,event-20),min(len(x)-1,event+20));ax.grid(alpha=.2)
    fig.suptitle(f'Largest SDK-relative jump: trajectory #{i}, frame {event}, {metadata[i]["joint_names"][j]}\nIncrement residual {r["max_delta_error_deg"]:.2f} deg/frame; this is a change in error, not necessarily a one-frame isolated spike',fontsize=13)
    fig.savefig(root/'worst_jump.png',dpi=160);plt.close(fig)
    try:
        from plotly.offline import get_plotlyjs
    except ImportError:
        sys.path.insert(0,'/tmp/wuji_rollout_plotly')
        from plotly.offline import get_plotlyjs
    payload=[]
    for i,(start,end) in enumerate(zip(offsets[:-1],offsets[1:])):
        payload.append(dict(index=i,meta=metadata[i],stats=rows[i],frames=a['frame_ids'][start:end].tolist(),
            p=np.round(pd[start:end].astype(float),5).tolist(),q=np.round(qd[start:end].astype(float),5).tolist(),
            online=np.round(online[i],5).tolist() if i in online else None))
    packed=json.dumps(payload,separators=(',',':')).replace('<','\\u003c')
    boot=json.dumps(dict(typical=typical,longest=longest,worst=worst,spike=spike,growth=growth))
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>EMF → Wuji2 轨迹诊断</title>
+<style>body{margin:0;background:#f3f6fa;color:#172536;font:15px system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:28px}h1{margin:0 0 10px}.card{background:white;border-radius:12px;padding:18px;margin:15px 0;box-shadow:0 2px 8px #1725360c}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.value{font-size:26px;font-weight:650}.note{color:#4b5d70;line-height:1.7}button,select{font:inherit;padding:8px;border:1px solid #b9c6d5;border-radius:6px;margin:4px;background:white}button{cursor:pointer}button:hover{background:#e8f2ff}.plot{height:350px}#heat{height:520px}#source{overflow-wrap:anywhere}img{width:100%}label{margin-right:16px}@media(max-width:800px){.stats{grid-template-columns:1fr 1fr}main{padding:10px}}</style>
+<main><h1>EMF → Wuji2 轨迹 rollout 诊断</h1><div class="note">temporal_causal · checkpoint STEP · 测试集 736 条连续轨迹 / 93,737 帧。角度单位均为度，时间为原始帧序号，未假定 FPS。<br>这里是相对 SDK 教师标签的轨迹误差，没有把网络输出 q 回馈给模型，也不是机器人闭环 rollout。全测试集采用完整段因果并行推理；选定轨迹另以每帧 [0:t] 历史实际调用 BF16 推理，橙线显示结果。BF16 的矩阵尺寸不同会引起数值差异；这 4 条补测轨迹中，与完整段预测的最大单点差为 0.513°。</div>
+<div class="stats"><div class="card">全测试 MAE<div class="value">MAE°</div></div><div class="card">全测试 RMSE<div class="value">RMSE°</div></div><div class="card">增量误差 &gt; 5°<div class="value">120 / 93,001</div>0.129% 相邻帧对</div><div class="card">后 20% − 前 20% MAE<div class="value">+0.217°</div>95% 轨迹 bootstrap CI [+0.129, +0.314]</div></div>
+<div class="card note">54.4% 的轨迹后段误差更高，存在轻度平均后段恶化；这不意味着误差逐帧单调增加，也不能据此断言 q 反馈误差累积。预测增量本身超过 5° 的比例为 6.66%，SDK 标签也为 6.69%。应重点看 <b>增量误差 Δq_pred − Δq_SDK</b>：超过 10° 共 4 对。将“孤立尖点”定义为同关节误差左右增量反向且两侧均超过 5° 时，计数为 0；更小的尖点和阶跃仍可能存在。</div>
+<div class="card"><details><summary>全测试统计：误差趋势、同批轨迹对照及跳变分布</summary><img src="data:image/png;base64,OVERVIEW"></details></div>
+<div class="card"><label>轨迹 <select id="seq"></select></label><label>关节 <select id="joint"></select></label><br><button data-key="typical">典型误差</button><button data-key="longest">最长轨迹</button><button data-key="worst">最大增量误差</button><button data-key="spike">最大孤立尖点候选</button><button data-key="growth">后段恶化最大</button><button id="zoom">放大最大增量误差 ±20 帧</button><button id="reset">完整轨迹</button><p id="source" class="note"></p><p id="stat" class="note"></p></div>
+<div class="card"><div id="angles" class="plot"></div><div id="errors" class="plot"></div><div id="delta" class="plot"></div><div id="heat"></div></div>
+<div class="card note">逐帧 MAE 对 20 个关节取平均；running MAE 为从段起点到当前帧的平均，并非误差累加和。所有增量、运行平均均在段内计算，不跨记录、缺帧或 reset。进度图只用同一批 ≥40 帧轨迹，每条轨迹等权；固定历史长度图使用相同的 ≥200 帧轨迹，避免样本减少造成趋势假象。CSV / NPZ 保留更高精度，交互图数值四舍五入至 0.00001°。长段无窗口截断。图中标签为 SDK q，不代表实测真值或下游 IK 收敛速度。</div></main>
+<script>PLOTLY</script><script>
+const DATA=PAYLOAD, PICKS=PICKSJSON;const $=id=>document.getElementById(id), cfg={responsive:true,displaylogo:false};
+const seq=$('seq'),joint=$('joint');DATA.forEach((d,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`#${i} | ${d.meta.dataset} ${d.meta.side_name} | ${d.p.length} frames | MAE ${d.stats.mae_deg.toFixed(2)}°`;seq.appendChild(o)});
+function line(x,y,name,color,dash){return {x,y,name,type:'scatter',mode:'lines',line:{color,width:1.6,dash}}}
+function layout(title,y){return {title:{text:title,font:{size:15}},margin:{l:65,r:20,t:55,b:45},xaxis:{title:'段内帧序号'},yaxis:{title:y},legend:{orientation:'h',y:1.17},hovermode:'x unified'}}
+function changeSeq(){const d=DATA[+seq.value];joint.replaceChildren();d.meta.joint_names.forEach((n,i)=>{const o=document.createElement('option');o.value=i;o.textContent=n;joint.appendChild(o)});joint.value=d.stats.worst_delta_joint;render()}
+function render(){const d=DATA[+seq.value],j=+joint.value,n=d.p.length,x=Array.from({length:n},(_,i)=>i),p=d.p.map(v=>v[j]),q=d.q.map(v=>v[j]),e=p.map((v,i)=>v-q[i]);
+const mae=d.p.map((v,i)=>v.reduce((s,a,k)=>s+Math.abs(a-d.q[i][k]),0)/20);let sum=0;const running=mae.map((v,i)=>(sum+=v)/(i+1));
+let traces=[line(x,q,'SDK','#263445'),line(x,p,'因果并行预测','#187cc0')];if(d.online)traces.push(line(x,d.online.map(v=>v[j]),'实际逐帧追加历史','#e58e40'));
+Plotly.react('angles',traces,layout(d.meta.joint_names[j]+'：关节角轨迹','角度 (°)'),cfg);
+traces=[line(x,mae,'逐帧 MAE（20关节）','#187cc0'),line(x,running,'Running mean MAE','#263445','dash'),line(x,e,'选定关节有符号误差','#bd5265')];if(d.online)traces.push(line(x,d.online.map((v,i)=>v.reduce((s,a,k)=>s+Math.abs(a-d.q[i][k]),0)/20),'实际 prefix MAE','#e58e40'));
+Plotly.react('errors',traces,layout('误差是否随历史增长：逐帧误差与运行平均','误差 (°)'),cfg);
+const diff=v=>v.slice(1).map((z,i)=>z-v[i]);Plotly.react('delta',[line(x.slice(1),diff(p),'Δq_pred','#187cc0'),line(x.slice(1),diff(q),'Δq_SDK','#263445'),line(x.slice(1),diff(e),'增量误差','#d25735')],layout('跳变诊断：相邻帧增量（只在段内）','增量 (°/frame)'),cfg);
+const h=layout('全部 20 关节绝对误差：指向尖点可查看数值','关节');h.yaxis={automargin:true};h.margin.l=210;
+Plotly.react('heat',[{type:'heatmap',x,y:d.meta.joint_names,z:Array.from({length:20},(_,k)=>d.p.map((v,i)=>Math.abs(v[k]-d.q[i][k]))),colorscale:'YlOrRd',colorbar:{title:'误差 °'},hovertemplate:'frame %{x}<br>%{y}<br>%{z:.3f}°<extra></extra>'}],h,cfg);
+$('source').textContent=`test index ${d.index} · global segment ${d.meta.segment_id} · ${d.meta.source_id} · source frames ${d.frames[0]}–${d.frames[n-1]}`;
+$('stat').textContent=`MAE ${d.stats.mae_deg.toFixed(3)}° · RMSE ${d.stats.rmse_deg.toFixed(3)}° · 后段−前段 ${d.stats.late_minus_early_deg.toFixed(3)}° · 最大增量误差 ${d.stats.max_delta_error_deg.toFixed(3)}°/frame（frame ${d.stats.worst_delta_frame}, ${d.meta.joint_names[d.stats.worst_delta_joint]}） · ${d.online?'已补测实际逐帧 prefix 推理':'完整段因果并行推理'}`;
+}
+seq.onchange=changeSeq;joint.onchange=render;document.querySelectorAll('[data-key]').forEach(b=>b.onclick=()=>{seq.value=PICKS[b.dataset.key];changeSeq()});
+$('zoom').onclick=()=>{const d=DATA[+seq.value];joint.value=d.stats.worst_delta_joint;render();const t=d.stats.worst_delta_frame;['angles','errors','delta','heat'].forEach(id=>Plotly.relayout(id,{'xaxis.range':[Math.max(0,t-20),Math.min(d.p.length-1,t+20)]}))};
+$('reset').onclick=()=>['angles','errors','delta','heat'].forEach(id=>Plotly.relayout(id,{'xaxis.autorange':true}));seq.value=PICKS.worst;changeSeq();
+</script></html>'''.replace('\n+','\n')
    html=html.replace('STEP',str(export['checkpoint_step'])).replace('MAE°',f'{summary["mae_deg"]:.3f}°').replace('RMSE°',f'{summary["rmse_deg"]:.3f}°')
    html=html.replace('OVERVIEW',base64.b64encode((root/'overview.png').read_bytes()).decode())
    html=html.replace('PLOTLY',get_plotlyjs()).replace('PAYLOAD',packed).replace('PICKSJSON',boot)
    (root/'report.html').write_text(html)
    print(json.dumps(dict(report=str(root/'report.html'),overview=str(root/'overview.png'),selected=dict(selected),online_indices=list(online)),indent=2))


if __name__=='__main__': main()
