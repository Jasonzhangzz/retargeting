#!/usr/bin/env python3
"""Create reproducible full-resolution q/error curves and numerical summaries."""
import argparse
import csv
from datetime import datetime,timezone
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


METHODS={'network':'q_network_rad','native_lm_raw':'q_lm_raw_rad','native_lm_filtered':'q_lm_filtered_rad'}
COLORS={'sdk':'#20252b','network':'#df9c2a','native_lm_raw':'#336ac2','native_lm_filtered':'#cb4f63'}
JOINTS=[finger+' '+joint for finger in ['thumb','index','middle','ring','pinky'] for joint in ['MCP flex' if finger!='thumb' else 'CMC flex','MCP abd' if finger!='thumb' else 'CMC abd','PIP' if finger!='thumb' else 'MCP','DIP' if finger!='thumb' else 'IP']]


def metrics(pred,gt,threshold):
    error=pred-gt;difference=np.diff(pred,axis=0)-np.diff(gt,axis=0)
    return dict(rmse_rad=float(np.sqrt(np.mean(error**2))),mae_rad=float(np.mean(abs(error))),max_error_rad=float(abs(error).max()),
        within_threshold_fraction=float(np.mean(abs(error)<=threshold)),frames_all_joints_within_threshold_fraction=float(np.mean(abs(error).max(-1)<=threshold)),delta_rmse_rad=float(np.sqrt(np.mean(difference**2))) if len(difference) else None,
        max_predicted_frame_increment_rad=float(abs(np.diff(pred,axis=0)).max()) if len(difference) else None,
        per_joint_rmse_rad=np.sqrt(np.mean(error**2,axis=0)).tolist(),
        quarter_rmse_rad=[float(np.sqrt(np.mean(x**2))) for x in np.array_split(error,4)])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--threshold-rad',type=float,default=1e-4);p.add_argument('--prediction-subdir');a=p.parse_args();root=a.root
    prediction_subdir=a.prediction_subdir or ('parallel_robokit_compiled' if (root/'benchmark_robokit_compiled_report.json').exists() else 'parallel_robokit')
    benchmark_filename='benchmark_robokit_compiled_report.json' if prediction_subdir.endswith('_compiled') else 'benchmark_robokit_report.json'
    figures=root/'figures';figures.mkdir(exist_ok=True);rows=[];streams=[];overview={};quality=[]
    selection=json.loads((root/'selection.json').read_text())['trajectories']
    for config in ('builtin','hl_urdf'):
      overview[config]={name:[] for name in METHODS}
      for side in ('left','right'):
        figerr,axeserr=plt.subplots(5,1,figsize=(13,13),layout='constrained')
        for row_index,record in enumerate(selection):
          idx=record['index'];gt=dict(np.load(root/'gt'/config/f'trajectory_{idx}_{side}.npz'))
          data=dict(np.load(root/prediction_subdir/config/f'trajectory_{idx}_{side}.npz'))
          qgt=gt['q_sdk_rad'];seconds=(gt['timestamp_ns']-gt['timestamp_ns'][0])*1e-9
          if not np.array_equal(gt['timestamp_ns'],data['timestamp_ns']):raise ValueError('Timestamp mismatch')
          series={'sdk':qgt,**{name:data[key] for name,key in METHODS.items()}}
          fig,axes=plt.subplots(5,4,figsize=(17,13),sharex=True,layout='constrained')
          for j,ax in enumerate(axes.flat):
            for name,q in series.items():ax.plot(seconds,q[:,j],label=name,color=COLORS[name],lw=1.1,alpha=.85)
            ax.set_title(JOINTS[j],fontsize=10);ax.grid(alpha=.2)
            if j%4==0:ax.set_ylabel('q (rad)')
            if j>=16:ax.set_xlabel('Time (s)')
          axes[0,0].legend(fontsize=8)
          fig.suptitle(f'{config} / {side} / trajectory {idx} / {len(qgt)} native frames\nSDK continuous reference vs EMF-only causal warm start + parallel native Robokit LM',fontsize=12)
          filename=f'q_{config}_{side}_{idx}.png';fig.savefig(figures/filename,dpi=140);plt.close(fig)
          curve={}
          for method,key in METHODS.items():
            pred=data[key]
            if pred.shape!=qgt.shape or not np.isfinite(pred).all():raise ValueError('Invalid prediction shape/value')
            result=metrics(pred,qgt,a.threshold_rad)
            rows.append(dict(configuration=config,side=side,trajectory=idx,frames=len(qgt),method=method,**result))
            overview[config][method].append((side,idx,pred-qgt))
            error=pred-qgt;curve[method]=dict(frame_rmse_rad=np.sqrt(np.mean(error**2,axis=1)).tolist(),
                frame_max_error_rad=abs(error).max(1).tolist(),cumulative_rmse_rad=np.sqrt(np.cumsum((error**2).sum(1))/(20*np.arange(1,len(error)+1))).tolist())
            axeserr[row_index].plot(seconds,curve[method]['frame_rmse_rad'],label=method,color=COLORS[method],lw=1)
          axeserr[row_index].axhline(a.threshold_rad,color='#388f68',ls='--',label='1e-4 rad' if row_index==0 else None)
          axeserr[row_index].set(title=f'Trajectory {idx}',ylabel='Frame q RMSE (rad)');axeserr[row_index].grid(alpha=.2)
          if row_index==0:axeserr[row_index].legend(fontsize=8,ncol=4)
          if config=='builtin':
            pos=gt['position_tx_m'];conf=gt['confidence'];quat=gt['orientation_xyzw']
            quality.append(dict(trajectory=idx,side=side,frames=len(qgt),
                confidence_mean=conf.mean(0).tolist(),confidence_below_half_fraction=(conf<.5).mean(0).tolist(),
                confidence_zero_count=(conf==0).sum(0).tolist(),
                receiver_distance_from_tx_max_m=np.linalg.norm(pos,axis=-1).max(0).tolist(),
                receiver_distance_above_half_meter_count=(np.linalg.norm(pos,axis=-1)>.5).sum(0).tolist(),
                quaternion_norm_max_deviation=float(abs(np.linalg.norm(quat,axis=-1)-1).max()),
                timestamp_delta_min_s=float(np.diff(gt['timestamp_ns']).min()*1e-9),
                timestamp_delta_max_s=float(np.diff(gt['timestamp_ns']).max()*1e-9)))
          streams.append(dict(configuration=config,side=side,trajectory=idx,seconds=seconds.round(7).tolist(),
              q={name:np.round(q,7).tolist() for name,q in series.items()},curves=curve,figure='figures/'+filename))
        axeserr[-1].set_xlabel('Time (s)');figerr.suptitle(f'{config} / {side}: full-resolution rollout error (no q feedback)')
        figerr.savefig(figures/f'error_{config}_{side}.png',dpi=150);plt.close(figerr)
    aggregate=[]
    for config,methods in overview.items():
      for method,values in methods.items():
       for side in ('all','left','right'):
        selected=[v[2] for v in values if side=='all' or v[0]==side];err=np.concatenate(selected)
        aggregate.append(dict(configuration=config,side=side,method=method,frames=len(err),
            rmse_rad=float(np.sqrt(np.mean(err**2))),mae_rad=float(abs(err).mean()),max_error_rad=float(abs(err).max()),
            within_threshold_fraction=float(np.mean(abs(err)<=a.threshold_rad)),frames_all_joints_within_threshold_fraction=float(np.mean(abs(err).max(-1)<=a.threshold_rad)),per_joint_rmse_rad=np.sqrt(np.mean(err**2,axis=0)).tolist()))
    (root/'metrics.json').write_text(json.dumps(dict(prediction_subdir=prediction_subdir,control_threshold_rad=a.threshold_rad,aggregate=aggregate,trajectories=rows),indent=2))
    (root/'input_quality.json').write_text(json.dumps(quality,indent=2))
    columns=['configuration','side','trajectory','frames','method','rmse_rad','mae_rad','max_error_rad','within_threshold_fraction','frames_all_joints_within_threshold_fraction','delta_rmse_rad','max_predicted_frame_increment_rad']
    with (root/'trajectory_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    timing=json.loads((root/benchmark_filename).read_text());sdk=json.loads((root/'sdk_benchmark.json').read_text())
    print('SDK_SCHEMA',list(sdk),flush=True)
    dashboard='''<!doctype html><meta charset="utf-8"><title>Wuji native LM trajectory comparison</title>
<style>body{font:15px system-ui;margin:24px;color:#19212b;max-width:1500px}select{padding:8px;margin:5px}canvas{width:100%;height:340px;border:1px solid #ddd}p{line-height:1.5}.legend span{margin-right:20px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}a{color:#275d9d}</style>
<h1>EMF → Wuji2: SDK vs causal warm start + native Robokit CUDA LM</h1>
<p>SDK output is a stateful, filtered reference. Production LM has no SDK seeds or SDK intermediate targets. Orange: network; blue: raw LM; red: alpha=.2 parallel filtered LM. Joint control threshold: 1e-4 rad. All native frames are shown. Error is comparison to SDK q, not proof of physical pose accuracy.</p>
<label>Trajectory <select id="stream"></select></label><label>Joint <select id="joint"></select></label><label>Frame error <select id="metric"><option value="frame_rmse_rad">RMSE</option><option value="frame_max_error_rad">Maximum absolute error</option></select></label>
<div class="legend"><span style="color:#20252b">SDK</span><span style="color:#df9c2a">Network</span><span style="color:#336ac2">Native LM raw</span><span style="color:#cb4f63">Native LM filtered</span></div>
<h3 id="title"></h3><canvas id="q"></canvas><h3>Joint absolute error (rad)</h3><canvas id="e"></canvas>
<h3>Full-frame RMSE and cumulative rollout RMSE (rad)</h3><canvas id="rollout"></canvas>
<p>Solid: frame RMSE. Dotted: cumulative RMSE. A growing cumulative curve does not imply recursive q feedback: the model uses EMF history and LM solves each frame independently.</p>
<p><a href="metrics.json">Numerical metrics</a> · <a href="trajectory_metrics.csv">CSV</a> · <a href="input_quality.json">Input quality</a> · <a href="basin_diagnosis.json">Seed controls (SDK inputs diagnostic only)</a> · <a href="REPORT.md">Report</a></p>
<script>const DATA=__DATA__,COLORS=__COLORS__,NAMES=__NAMES__;
const s=document.getElementById('stream'),j=document.getElementById('joint');
DATA.forEach((d,i)=>s.add(new Option(`${d.configuration} / ${d.side} / trajectory ${d.trajectory}`,i)));
NAMES.forEach((d,i)=>j.add(new Option(d,i)));
function draw(id,x,curves,threshold=false){const c=document.getElementById(id),ratio=devicePixelRatio||1;c.width=c.clientWidth*ratio;c.height=340*ratio;const g=c.getContext('2d');g.scale(ratio,ratio);const W=c.clientWidth,H=340,L=74,R=20,T=20,B=35;let all=curves.flatMap(c=>c.y),mn=Math.min(...all),mx=Math.max(...all);if(threshold){mn=Math.min(mn,1e-4);mx=Math.max(mx,1e-4)}let pad=(mx-mn)*.08||.01;mn-=pad;mx+=pad;const px=t=>L+(t-x[0])/(x.at(-1)-x[0]||1)*(W-L-R),py=v=>H-B-(v-mn)/(mx-mn)*(H-T-B);g.clearRect(0,0,W,H);g.font='12px system-ui';for(let k=0;k<=5;k++){let v=mn+(mx-mn)*k/5;g.strokeStyle='#eee';g.beginPath();g.moveTo(L,py(v));g.lineTo(W-R,py(v));g.stroke();g.fillStyle='#555';g.fillText(v.toPrecision(4),4,py(v)+4);let t=x[0]+(x.at(-1)-x[0])*k/5;g.fillText(t.toFixed(2),px(t)-15,H-10)}for(const d of curves){g.strokeStyle=COLORS[d.name];g.setLineDash(d.dotted?[3,4]:[]);g.lineWidth=1.2;g.beginPath();d.y.forEach((v,i)=>i?g.lineTo(px(x[i]),py(v)):g.moveTo(px(x[i]),py(v)));g.stroke()}if(threshold){g.strokeStyle='#388f68';g.setLineDash([5,4]);g.beginPath();g.moveTo(L,py(1e-4));g.lineTo(W-R,py(1e-4));g.stroke()}g.setLineDash([])}
function update(){let d=DATA[+s.value],k=+j.value,x=d.seconds;document.getElementById('title').textContent=`${NAMES[k]} — ${d.configuration} / ${d.side} / trajectory ${d.trajectory} (${x.length} frames)`;draw('q',x,Object.entries(d.q).map(([name,q])=>({name,y:q.map(v=>v[k])})));draw('e',x,Object.entries(d.q).filter(([name])=>name!='sdk').map(([name,q])=>({name,y:q.map((v,i)=>Math.abs(v[k]-d.q.sdk[i][k]))})),true);draw('rollout',x,Object.entries(d.curves).flatMap(([name,c])=>[{name,y:c[document.getElementById('metric').value]},{name,y:c.cumulative_rmse_rad,dotted:true}]),true)}s.onchange=j.onchange=document.getElementById('metric').onchange=update;window.onresize=update;update();</script>'''
    dashboard=dashboard.replace('__DATA__',json.dumps(streams,separators=(',',':'))).replace('__COLORS__',json.dumps(COLORS)).replace('__NAMES__',json.dumps(JOINTS))
    (root/'comparison.html').write_text(dashboard)
    print(json.dumps(aggregate,indent=2),flush=True)


if __name__=='__main__':main()
