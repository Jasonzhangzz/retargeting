#!/usr/bin/env python3
"""Summarize full-frame native CUDA initial-guess prior/clip ablations."""
import argparse,json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def error_metrics(q,gt,anchor,offsets):
    e=q-gt;d=np.concatenate([np.diff(e[u:v],axis=0) for u,v in zip(offsets[:-1],offsets[1:])]);steps=np.concatenate([np.diff(q[u:v],axis=0) for u,v in zip(offsets[:-1],offsets[1:])])
    return dict(frames=len(q),mae_rad=float(abs(e).mean()),rmse_rad=float(np.sqrt((e*e).mean())),max_error_rad=float(abs(e).max()),p99_abs_rad=float(np.quantile(abs(e),.99)),delta_rmse_rad=float(np.sqrt((d*d).mean())),max_frame_step_rad=float(abs(steps).max()),max_network_offset_rad=float(abs(q-anchor).max()),all_joints_within_1e4_fraction=float((abs(e).max(-1)<=1e-4).mean()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;data=json.loads((r/'results.json').read_text());assert data['complete'];cases=data['cases'];rows=[];selected={};options={};network={}
    for config in ['builtin','hl_urdf']:
        ds=[dict(np.load(r/f'{config}_{side}.npz')) for side in ['left','right']];gt=np.concatenate([x['q_sdk_rad'] for x in ds]);anchor=np.concatenate([x['q_network_rad'] for x in ds]);offsets=np.r_[ds[0]['offsets'],ds[0]['offsets'][-1]+ds[1]['offsets'][1:]];assert len(gt)==5770;network[config]=error_metrics(anchor,gt,anchor,offsets)
        for i,case in enumerate(cases):
            matching=[x for x in data['rows'] if x['configuration']==config and x['case']['name']==case['name']];assert len(matching)==2
            timing=[sum(x['robot_and_filter_seconds_repeats'][j] for x in matching) for j in range(3)]
            for method,key in [('raw','q_raw_rad'),('filtered','q_filtered_rad')]:
                q=np.concatenate([x[key][i] for x in ds]);m=error_metrics(q,gt,anchor,offsets);m.update(configuration=config,case=case['name'],method=method,weight=case['weight'],step=case['step'],radius=case['radius'],robot_and_filter_seconds=float(np.median(timing)));rows.append(m)
        best=min((x for x in rows if x['configuration']==config),key=lambda x:x['rmse_rad']);selected[config]=best;options[config]={k:best[k] for k in ['weight','step','radius']}
    summary=dict(complete=True,neural_optimizer_updates=0,all_frames_retained=True,frames_per_configuration=5770,network=network,selected_by_rmse=selected,robot_options=options,rows=rows,timing_scope=data['timing_scope'],selection_scope=data['selection_scope']);(r/'summary.json').write_text(json.dumps(summary,indent=2));(r/'selected_options.json').write_text(json.dumps(options,indent=2))
    with (r/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (r/'figures').mkdir(exist_ok=True);fig,axes=plt.subplots(2,2,figsize=(15,9),layout='constrained');x=np.arange(len(cases))
    for k,config in enumerate(['builtin','hl_urdf']):
      for j,metric in enumerate(['rmse_rad','max_error_rad']):
        ax=axes[k,j]
        for offset,method,color in [(-.18,'raw','#336ac2'),(.18,'filtered','#cb4f63')]:
            vals=[next(z[metric] for z in rows if z['configuration']==config and z['case']==c['name'] and z['method']==method) for c in cases];ax.bar(x+offset,vals,.36,label=method,color=color)
        ax.axhline(network[config][metric],color='#d19b22',ls='--',label='Network only');ax.set(xticks=x,xticklabels=[c['name'] for c in cases],ylabel=metric+' (rad)',title=config);ax.tick_params(axis='x',labelrotation=55,labelsize=8);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.savefig(r/'figures/parameter_sweep.png',dpi=150);plt.close(fig)
    lines=['# 网络初值正则与更新裁剪：完整轨迹对照','', '测试沿用最终第200轮step4200网络、5条留出轨迹、每配置左右手共5770帧。DiT训练保持运行。本轮为已经检查过的留出轨迹上的参数对照，最优设置是本轮调参结果，不作为新数据泛化结论。','', 'λ项采用0.5λ||q-q_network||²（rad空间），两个阶段都以同一个原始网络q为中心。step是每轮每关节的最大更新，box是各帧相对网络q的总L∞偏移限位，并与URDF物理限位相交。LM不做训练反传梯度裁剪；这里裁剪的是求解更新。','', '人手重建与网络初值使用同一份由真实EMF产生的缓存，用于隔离机器人LM的参数影响；SDK q仅用于结果打分，不进入目标或初始化。所有案例保留raw和alpha=.2的GPU并行滤波输出。滤波后不保证仍位于逐帧box内，因此总偏移边界验证作用在raw解。','', '## 全部参数结果','', '|配置|设置|raw RMSE(rad)|滤波RMSE(rad)|raw最大误差(rad)|滤波最大误差(rad)|','|---|---|---:|---:|---:|---:|']
    for config in ['builtin','hl_urdf']:
        lines.append(f"|{config}|network only|{network[config]['rmse_rad']:.6f}|—|{network[config]['max_error_rad']:.6f}|—|")
        for case in cases:
            raw=next(x for x in rows if x['configuration']==config and x['case']==case['name'] and x['method']=='raw');filtered=next(x for x in rows if x['configuration']==config and x['case']==case['name'] and x['method']=='filtered');lines.append(f"|{config}|{case['name']}|{raw['rmse_rad']:.6f}|{filtered['rmse_rad']:.6f}|{raw['max_error_rad']:.6f}|{filtered['max_error_rad']:.6f}|")
    lines+=['','## 本轮最小RMSE设置','']
    for config,b in selected.items():lines.append(f"- {config}：{b['case']}，输出{b['method']}，RMSE {b['rmse_rad']:.6f}rad，MAE {b['mae_rad']:.6f}rad，最大误差 {b['max_error_rad']:.6f}rad，最大相邻帧关节增量 {b['max_frame_step_rad']:.6f}rad。")
    lines+=['','这里的计时仅包括机器人LM和滤波，前序网络/人手重建使用缓存，不能当作完整EMF→q时延。最终选择设置另行运行完整流程，与同环境下重新运行的无正则基线和SDK计时比较。','', '![完整参数对照](figures/parameter_sweep.png)','', '[全部数值CSV](metrics.csv) · [汇总JSON](summary.json) · [CUDA预检](preflight.json)']
    (r/'REPORT.md').write_text('\n'.join(lines)+'\n');print('ANCHOR_SUMMARY_COMPLETE',json.dumps(selected),flush=True)


if __name__=='__main__':main()
