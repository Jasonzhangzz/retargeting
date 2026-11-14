#!/usr/bin/env python3
"""Compare temporal networks and their native LM rollouts on identical recordings."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


COLORS = {'sdk': '#20252b', 'causal_network': '#c88719', 'causal_lm': '#c54850',
          'bidirectional_network': '#5486c5', 'bidirectional_lm': '#208868'}
JOINTS = [f'{finger} {joint}' for finger in ('thumb', 'index', 'middle', 'ring', 'pinky')
          for joint in ('joint 1', 'joint 2', 'joint 3', 'joint 4')]


def summarize(error):
    frame_rmse = np.sqrt(np.mean(error**2, axis=-1))
    return dict(frames=len(error), rmse_rad=float(np.sqrt(np.mean(error**2))),
                mae_rad=float(np.mean(abs(error))), max_error_rad=float(abs(error).max()),
                frame_rmse_p50_rad=float(np.median(frame_rmse)),
                frame_rmse_p95_rad=float(np.quantile(frame_rmse, .95)),
                within_1e4_fraction=float(np.mean(abs(error) <= 1e-4)),
                all_joints_within_1e4_fraction=float(np.mean(abs(error).max(-1) <= 1e-4)),
                per_joint_rmse_rad=np.sqrt(np.mean(error**2, axis=0)).tolist(),
                thumb_squared_error_fraction=float((error[:, :4]**2).sum() / (error**2).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--causal-root', type=Path, required=True)
    parser.add_argument('--bidirectional-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    figures = out / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    selection = json.loads((args.causal_root / 'selection.json').read_text())['trajectories']
    rows, streams, collected = [], [], {}
    for config in ('builtin', 'hl_urdf'):
        for side in ('left', 'right'):
            figerr, axerr = plt.subplots(5, 1, figsize=(13, 13), layout='constrained')
            for row, item in enumerate(selection):
                index = item['index']
                filename = f'trajectory_{index}_{side}.npz'
                with np.load(args.causal_root / 'gt' / config / filename) as f:
                    gt, ts = f['q_sdk_rad'], f['timestamp_ns']
                series = {'sdk': gt}
                arrays = {}
                for model, root in [('causal', args.causal_root), ('bidirectional', args.bidirectional_root)]:
                    with np.load(root / 'parallel_robokit_compiled' / config / filename) as f:
                        if not np.array_equal(ts, f['timestamp_ns']):
                            raise ValueError('Prediction/GT timestamp mismatch')
                        arrays[model] = {k: f[k] for k in ('q_network_rad', 'q_lm_raw_rad', 'q_lm_filtered_rad')}
                    for method, key in [('network', 'q_network_rad'), ('lm', 'q_lm_filtered_rad')]:
                        series[f'{model}_{method}'] = arrays[model][key]
                seconds = (ts - ts[0]) * 1e-9
                curves = {}
                for method, q in series.items():
                    if q.shape != gt.shape or not np.isfinite(q).all():
                        raise ValueError('Invalid q')
                    if method == 'sdk':
                        continue
                    error = q - gt
                    stats = summarize(error)
                    stats['delta_rmse_rad'] = float(np.sqrt(np.mean(np.diff(error, axis=0)**2)))
                    rows.append(dict(configuration=config, side=side, trajectory=index, method=method, **stats))
                    collected.setdefault((config, side, method), []).append(error)
                    curves[method] = dict(frame_rmse_rad=np.sqrt((error**2).mean(-1)).tolist(),
                        frame_max_error_rad=abs(error).max(-1).tolist(),
                        cumulative_rmse_rad=np.sqrt(np.cumsum((error**2).sum(-1)) / (20*np.arange(1, len(error)+1))).tolist())
                    axerr[row].plot(seconds, curves[method]['frame_rmse_rad'], color=COLORS[method], lw=1, label=method)
                # Raw LM statistics are retained alongside the filtered primary comparison.
                for model, data in arrays.items():
                    rows.append(dict(configuration=config, side=side, trajectory=index,
                                     method=model+'_lm_raw', **summarize(data['q_lm_raw_rad']-gt)))
                    collected.setdefault((config, side, model+'_lm_raw'), []).append(data['q_lm_raw_rad']-gt)
                axerr[row].axhline(1e-4, color='#555', ls='--', lw=.7)
                axerr[row].set(title=f'Trajectory {index}', ylabel='Frame RMSE (rad)')
                axerr[row].grid(alpha=.2)
                if row == 0:
                    axerr[row].legend(ncol=4, fontsize=8)
                fig, axes = plt.subplots(5, 4, figsize=(17, 13), sharex=True, layout='constrained')
                for j, ax in enumerate(axes.flat):
                    for name, q in series.items():
                        ax.plot(seconds, q[:, j], color=COLORS[name], lw=1, label=name)
                    ax.set_title(JOINTS[j], fontsize=10)
                    ax.grid(alpha=.2)
                    if j % 4 == 0:
                        ax.set_ylabel('q (rad)')
                    if j >= 16:
                        ax.set_xlabel('Time (s)')
                axes[0, 0].legend(fontsize=7)
                fig.suptitle(f'{config} / {side} / trajectory {index}: SDK vs causal / bidirectional networks and parallel CUDA LM')
                fig.savefig(figures / f'q_{config}_{side}_{index}.png', dpi=135)
                plt.close(fig)
                streams.append(dict(configuration=config, side=side, trajectory=index,
                    seconds=seconds.tolist(), q={k: v.round(7).tolist() for k, v in series.items()}, curves=curves))
            axerr[-1].set_xlabel('Time (s)')
            figerr.suptitle(f'{config} / {side}: identical full recordings; filtered LM vs SDK reference')
            figerr.savefig(figures / f'error_{config}_{side}.png', dpi=145)
            plt.close(figerr)
    aggregate = []
    for config in ('builtin', 'hl_urdf'):
        for method in ('causal_network', 'causal_lm_raw', 'causal_lm', 'bidirectional_network', 'bidirectional_lm_raw', 'bidirectional_lm'):
            for side in ('all', 'left', 'right'):
                errors = [e for (c, s, m), es in collected.items()
                          if c == config and m == method and (side == 'all' or s == side) for e in es]
                aggregate.append(dict(configuration=config, side=side, method=method, **summarize(np.concatenate(errors))))
    (out / 'metrics.json').write_text(json.dumps(dict(control_threshold_rad=1e-4, aggregate=aggregate, trajectories=rows), indent=2))
    with (out / 'trajectory_metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['configuration', 'side', 'trajectory', 'method', 'frames', 'rmse_rad', 'mae_rad', 'max_error_rad', 'frame_rmse_p50_rad', 'frame_rmse_p95_rad'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    dashboard = '''<!doctype html><meta charset="utf-8"><title>Temporal model and CUDA LM comparison</title>
<style>body{font:15px system-ui;max-width:1500px;margin:24px;color:#19212b}canvas{width:100%;height:330px;border:1px solid #ddd}select{padding:8px;margin:6px}p{line-height:1.5}</style>
<h1>因果 / 双向时序模型与并行 CUDA LM</h1>
<p>同一批5条完整真实录制轨迹，每配置5770个单手帧。双向模型使用未来EMF，因果模型仅使用当前和历史；没有q反馈。LM曲线包含与SDK相同alpha=.2的输出滤波，但LM目标不含SDK历史正则，不是同一时序求解过程的复刻。</p>
<p id="legend"></p><select id="stream"></select><select id="joint"></select>
<select id="metric"><option value="frame_rmse_rad">Frame RMSE</option><option value="frame_max_error_rad">Frame maximum error</option></select>
<h3 id="title"></h3><canvas id="q"></canvas><h3>Joint absolute error (rad)</h3><canvas id="e"></canvas>
<h3>Frame error (solid) / cumulative RMSE (dotted), rad</h3><canvas id="rollout"></canvas>
<p>保留所有原始帧，包括初始化瞬态。网页q仅显示至小数点后7位；NPZ保存原始精度。1e-4 rad参考线为绿色虚线。<a href="REPORT.md">报告</a> · <a href="metrics.json">指标</a> · <a href="trajectory_metrics.csv">CSV</a></p>
<script>const DATA=__DATA__,COLORS=__COLORS__,NAMES=__NAMES__;
const s=document.getElementById('stream'),j=document.getElementById('joint');
document.getElementById('legend').innerHTML=Object.entries(COLORS).map(([k,v])=>`<span style="color:${v};margin-right:18px">${k}</span>`).join('');
DATA.forEach((d,i)=>s.add(new Option(`${d.configuration} / ${d.side} / trajectory ${d.trajectory}`,i)));
NAMES.forEach((d,i)=>j.add(new Option(d,i)));
function draw(id,x,curves,threshold=false){const c=document.getElementById(id),ratio=devicePixelRatio||1;c.width=c.clientWidth*ratio;c.height=330*ratio;const g=c.getContext('2d');g.scale(ratio,ratio);const W=c.clientWidth,H=330,L=74,R=20,T=20,B=35;let mn=Infinity,mx=-Infinity;for(const d of curves)for(const v of d.y){mn=Math.min(mn,v);mx=Math.max(mx,v)}if(threshold){mn=Math.min(mn,1e-4);mx=Math.max(mx,1e-4)}let pad=(mx-mn)*.08||.01;mn-=pad;mx+=pad;const px=t=>L+(t-x[0])/(x.at(-1)-x[0]||1)*(W-L-R),py=v=>H-B-(v-mn)/(mx-mn)*(H-T-B);g.clearRect(0,0,W,H);g.font='12px system-ui';for(let k=0;k<=5;k++){let v=mn+(mx-mn)*k/5;g.strokeStyle='#eee';g.beginPath();g.moveTo(L,py(v));g.lineTo(W-R,py(v));g.stroke();g.fillStyle='#555';g.fillText(v.toPrecision(4),4,py(v)+4);let t=x[0]+(x.at(-1)-x[0])*k/5;g.fillText(t.toFixed(2),px(t)-15,H-10)}for(const d of curves){g.strokeStyle=COLORS[d.name];g.setLineDash(d.dotted?[3,4]:[]);g.lineWidth=1.2;g.beginPath();d.y.forEach((v,i)=>i?g.lineTo(px(x[i]),py(v)):g.moveTo(px(x[i]),py(v)));g.stroke()}if(threshold){g.strokeStyle='#388f68';g.setLineDash([5,4]);g.beginPath();g.moveTo(L,py(1e-4));g.lineTo(W-R,py(1e-4));g.stroke()}g.setLineDash([])}
function update(){let d=DATA[+s.value],k=+j.value,x=d.seconds;document.getElementById('title').textContent=`${NAMES[k]} — ${d.configuration} / ${d.side} / trajectory ${d.trajectory} (${x.length} frames)`;draw('q',x,Object.entries(d.q).map(([name,q])=>({name,y:q.map(v=>v[k])})));draw('e',x,Object.entries(d.q).filter(([name])=>name!='sdk').map(([name,q])=>({name,y:q.map((v,i)=>Math.abs(v[k]-d.q.sdk[i][k]))})),true);draw('rollout',x,Object.entries(d.curves).flatMap(([name,c])=>[{name,y:c[document.getElementById('metric').value]},{name,y:c.cumulative_rmse_rad,dotted:true}]),true)}s.onchange=j.onchange=document.getElementById('metric').onchange=update;window.onresize=update;update();</script>'''
    for key, value in [('__DATA__', streams), ('__COLORS__', COLORS), ('__NAMES__', JOINTS)]:
        dashboard = dashboard.replace(key, json.dumps(value, separators=(',', ':')))
    (out / 'comparison.html').write_text(dashboard)
    print(json.dumps([x for x in aggregate if x['side'] == 'all'], indent=2))


if __name__ == '__main__':
    main()
