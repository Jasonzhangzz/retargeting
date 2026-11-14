#!/usr/bin/env python3
"""Recompute numerical evidence and standalone curves for the finite campaign."""
import argparse,csv,gzip,base64,json,math,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_json(path):
 for attempt in range(8):
  try:return json.loads(path.read_text())
  except OSError:
   if attempt==7:raise
   time.sleep(.05*(attempt+1))


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;manifest=read_json(r/'MANIFEST.json');resource_data=read_json(r/'resource_usage.json') if (r/'resource_usage.json').exists() else {};expected=[(job,k) for job in manifest.get('jobs',[]) for k in job['budgets']];groups=[];streams={};missing=[];details=[];record_rows=[];phase_rows=[];network_rows=[];source_records={v['index']:v for v in json.loads((r/'selection.json').read_text())['records']}
 for job,k in expected:
  path=r/'jobs'/job['id']/f'budget_{k}'/'result.json'
  if not path.exists():missing.append(dict(job=job['id'],budget=k));continue
  report=json.loads(path.read_text())
  if not report['complete'] or report['candidate_signature']!=manifest['candidate_signature']:missing.append(dict(job=job['id'],budget=k,reason='Incomplete or wrong signature'));continue
  timing=float(np.median([v['total'] for v in report['timings']]));sdk_rows=json.loads((r/f'sdk_{job["configuration"]}_timings.json').read_text());indices={v['trajectory'] for v in report['quality']};sdk_rows=[v for v in sdk_rows if v['side']==job['side'] and v['trajectory'] in indices];sdk_seconds=float(np.median([sum(v['timings'][rep]['total'] for v in sdk_rows) for rep in range(10)]));robot_sdk_seconds=float(np.median([sum(v['timings'][rep]['robot'] for v in sdk_rows) for rep in range(10)]))
  for v in report['quality']:
   source=source_records[v['trajectory']];dataset=source.get('dataset') or next((p for p in Path(source['source_h5']).parts if p.startswith('XRRD')),'unknown');record_rows.append(dict(split=job['split'],configuration=job['configuration'],backend=job['backend'],budget=k,reference=job['reference'],dataset=dataset,**v))
  measured=report['timings'];phase_row=dict(job=job['id'],budget=k,backend=job['backend'],configuration=job['configuration'],side=job['side'],split=job['split'],frames=report['frames'],segments=report['segments'])
  for name in ['network','human_and_targets','robot','filter','total']:
   phase_row[name+'_seconds']=None if measured[0][name] is None else float(np.median([v[name] for v in measured]))
  for name,value in (report.get('isolated_stage_profile') or {}).items():
   if isinstance(value,dict):phase_row['isolated_'+name+'_seconds']=float(np.median(value['seconds']));phase_row['isolated_'+name+'_max_difference_rad']=value['max_output_difference_rad']
  for component,records in report['trace'].items():
   if isinstance(records,list):
    for stage,record in enumerate(records):
     history=record.get('history',[])
     if history and 'solver_seconds' in history[-1]:phase_row[f'cpp_{component}_{stage}_last_repeat_seconds']=history[-1]['solver_seconds']
  phase_rows.append(phase_row)
  for method in ['raw','filtered']:
   errors=[];deltas=[];increments=[];per_record=[]
   for file in path.parent.glob('trajectory_*.npz'):
    data=np.load(file);pred=data['q_'+method];gt=data['q_sdk_raw_recovered' if method=='raw' else 'q_sdk_filtered'];error=pred-gt;errors.append(error);offsets=data['segment_offsets'];per_record.append(float(np.sqrt(np.square(error).mean())))
    for s,e in zip(offsets,offsets[1:]):
     if e-s>1:deltas.append(np.diff(error[s:e],axis=0));increments.append(np.diff(pred[s:e],axis=0))
    # Main curves use a fixed budget16, not a post-hoc selection on held-out q.
    if (k==16 and not job['reference']) or (k==200 and job['reference']):
     index=int(file.stem.split('_')[1]);key=f'{job["split"]}/{job["configuration"]}/{job["side"]}/{index}'
     if key not in streams:streams[key]=dict(name=key,time=((data['timestamp_ns']-data['timestamp_ns'][0])*1e-9).tolist(),sdk=np.round(data['q_sdk_filtered'],6).tolist(),sdk_raw=np.round(data['q_sdk_raw_recovered'],6).tolist(),methods={},offsets=offsets.tolist())
     streams[key]['methods'][job['backend']+('/reference200' if job['reference'] else '')+'/'+method]=np.round(pred,6).tolist()
     if 'q_network' in data:streams[key]['methods']['network']=np.round(data['q_network'],6).tolist()
   error=np.concatenate(errors);dd=np.concatenate(deltas);inc=np.concatenate(increments)
   row=dict(split=job['split'],configuration=job['configuration'],side=job['side'],backend=job['backend'],budget=k,reference=job['reference'],method=method,frames=len(error),rmse_rad=float(np.sqrt(np.square(error).mean())),mae_rad=float(abs(error).mean()),max_error_rad=float(abs(error).max()),above_1e4_fraction=float((abs(error)>1e-4).mean()),frames_above_1e4_fraction=float((abs(error).max(-1)>1e-4).mean()),sdk_q_within_1e4=bool(abs(error).max()<=1e-4),per_joint_rmse_rad=np.sqrt(np.square(error).mean(0)).tolist(),macro_record_rmse_rad=float(np.mean(per_record)),delta_rmse_rad=float(np.sqrt(np.square(dd).mean())),max_frame_increment_rad=float(abs(inc).max()),seconds=timing,peak_resident_mib=resource_data.get('job_peaks_mib',{}).get(job['id']),cold_first_solve_seconds=report['cold_first_solve']['total'],peak_reserved_gib=report.get('peak_reserved_bytes',0)/2**30,latency_p10_seconds=float(np.quantile([v['total'] for v in measured],.1)),latency_p90_seconds=float(np.quantile([v['total'] for v in measured],.9)),sdk_seconds=sdk_seconds,segments=report['segments'],repeat_max_drift_rad=report['repeat_max_drift_rad'],common_sdk_target_score=report['common_sdk_target_score'],full_speedup=sdk_seconds/timing,frames_per_second=len(error)/timing,peak_memory_gib=report['peak_memory_bytes']/2**30,information=report['information'],robot_objective_per_frame=sum(report['total_robot_objective'])/len(error),human_emf_error=report['human_emf_error'],fk_vs_sdk=report['fk_vs_sdk'])
   if report['timings'][0]['robot'] is not None:row['robot_speedup']=robot_sdk_seconds/float(np.median([v['robot']+v['filter'] for v in report['timings']]))
   groups.append(row)
  if job['backend']=='banded_lm' and k==16 and not job['reference']:
   nn_errors=[]
   for file in path.parent.glob('trajectory_*.npz'):
    with np.load(file) as data:
     if 'q_network' in data:nn_errors.append(data['q_network']-data['q_sdk_filtered'])
   if nn_errors:
    e=np.concatenate(nn_errors);network_rows.append(dict(split=job['split'],configuration=job['configuration'],side=job['side'],frames=len(e),rmse_rad=float(np.sqrt(np.square(e).mean())),max_error_rad=float(abs(e).max()),above_1e4_fraction=float((abs(e)>1e-4).mean())))
  details.append(dict(job=job['id'],budget=k,report=str(path.relative_to(r)),convergence=report['convergence'],terms=report['robot_objective_terms'],trace=report['trace']))
 summary=dict(complete=not missing,execution_complete=not missing,fixed_budget16_sdk_q_within_1e4=all(v['sdk_q_within_1e4'] for v in groups if v['budget']==16 and v['method']=='filtered' and not v['reference']),expected_cases=len(expected),executed_cases=len(details),missing=missing,rows=groups,details=details,phases=phase_rows,network_only=network_rows,candidate_signature=manifest.get('candidate_signature'),neural_optimizer_updates=0,scope='Shared A800 GPU; existing DiT not interrupted. Raw q uses approximate inverse-IIR SDK reference; filtered q uses exact SDK public output. Fixed budget16 curves plus development reference200; all budgets retained numerically.')
 (r/'summary.json').write_text(json.dumps(summary,indent=2));columns=['split','configuration','side','backend','budget','reference','method','frames','rmse_rad','mae_rad','max_error_rad','above_1e4_fraction','frames_above_1e4_fraction','sdk_q_within_1e4','macro_record_rmse_rad','delta_rmse_rad','max_frame_increment_rad','seconds','latency_p10_seconds','latency_p90_seconds','sdk_seconds','full_speedup','frames_per_second','peak_memory_gib','peak_reserved_gib','peak_resident_mib','cold_first_solve_seconds','robot_objective_per_frame']
 with (r/'metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,columns,extrasaction='ignore');w.writeheader();w.writerows(groups)
 for name,values in [('record_metrics.csv',record_rows),('phase_timings.csv',phase_rows)]:
  if values:
   fields=list(dict.fromkeys(k for row in values for k in row if not isinstance(row[k],(list,dict))))
   with (r/name).open('w',newline='') as f:
    w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(values)
 dataset_groups={}
 for row in record_rows:
  key=tuple(row[k] for k in ['split','configuration','side','backend','budget','reference','method','dataset']);dataset_groups.setdefault(key,[]).append(row)
 dataset_rows=[]
 for key,values in dataset_groups.items():
  n=sum(v['frames'] for v in values);dataset_rows.append(dict(zip(['split','configuration','side','backend','budget','reference','method','dataset'],key),frames=n,records=len(values),rmse_rad=float(np.sqrt(sum(v['frames']*v['rmse_rad']**2 for v in values)/n)),mae_rad=sum(v['frames']*v['mae_rad'] for v in values)/n,max_error_rad=max(v['max_error_rad'] for v in values),above_1e4_fraction=sum(v['frames']*v['above_1e4_fraction'] for v in values)/n))
 if dataset_rows:
  with (r/'dataset_metrics.csv').open('w',newline='') as f:
   writer=csv.DictWriter(f,list(dataset_rows[0]));writer.writeheader();writer.writerows(dataset_rows)
 summary['dataset_rows']=dataset_rows
 (r/'summary.json').write_text(json.dumps(summary,indent=2))
 figures=r/'figures';figures.mkdir(exist_ok=True)
 for split in ['development','heldout']:
  fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
  for config,ax in zip(['builtin','hl_urdf'],axes):
   for backend in sorted({v['backend'] for v in groups}):
    values=[v for v in groups if v['split']==split and v['configuration']==config and v['backend']==backend and v['method']=='filtered' and not v['reference']]
    for side in ['left','right']:
     points=sorted([v for v in values if v['side']==side],key=lambda v:v['budget'])
     if points:ax.plot([v['seconds'] for v in points],[v['rmse_rad'] for v in points],marker='o',label=backend+'/'+side)
   ax.set(title=config,xlabel='Full pipeline seconds / hand group',ylabel='Filtered q RMSE vs SDK (rad)');ax.grid(alpha=.2);ax.legend(fontsize=6)
  fig.suptitle(split+' / speed–accuracy, shared GPU');fig.savefig(figures/f'{split}_speed_accuracy.png',dpi=160);plt.close(fig)
 packed=base64.b64encode(gzip.compress(json.dumps(streams,separators=(',',':')).encode())).decode()
 html='''<!doctype html><meta charset="utf-8"><title>Wuji sparse trajectory results</title><style>body{font:15px system-ui;margin:24px}canvas{width:100%;height:300px;border:1px solid #ddd}select{padding:6px;margin:5px}label{display:inline-block}a{color:#246}</style><h1>Wuji2 causal and joint trajectory solvers</h1><p>Joint budget16, fixed causal update caps, and reference200 on development records; all frames. Joint methods read future EMF. Causal methods use previous raw q. Raw curves and metrics use inverse-IIR SDK q; filtered curves use the exact SDK public output. See the report for original weights and shared-GPU timing.</p><a href="REPORT.md">Report</a> · <a href="metrics.csv">All budgets CSV</a> · <a href="summary.json">Full numerical evidence</a><p id="loading">Loading full-resolution curves…</p><label>Stream <select id="stream"></select></label><label>Joint <select id="joint"></select></label><label>Solver <select id="method"></select></label><canvas id="q"></canvas><p>Absolute joint error (rad)</p><canvas id="error"></canvas><p>Frame and cumulative rollout RMSE (rad)</p><canvas id="rollout"></canvas><script>
const encoded='PACKED_DATA';let DATA;const colors=['#2265ba','#c65b43','#2a936f'];
function draw(id,x,series){let c=document.getElementById(id),W=c.clientWidth,H=300,r=devicePixelRatio||1;c.width=W*r;c.height=H*r;let g=c.getContext('2d');g.scale(r,r);let vals=series.flatMap(s=>s.y),lo=Math.min(...vals),hi=Math.max(...vals);if(hi-lo<1e-9)hi=lo+.01;let px=i=>65+(x[i]-x[0])/(x.at(-1)-x[0]||1)*(W-90),py=v=>265-(v-lo)/(hi-lo)*240;g.font='12px system-ui';for(let k=0;k<=5;k++){let v=lo+(hi-lo)*k/5;g.fillStyle='#444';g.fillText(v.toPrecision(4),2,py(v));g.strokeStyle='#eee';g.beginPath();g.moveTo(60,py(v));g.lineTo(W-20,py(v));g.stroke()}series.forEach((s,k)=>{g.strokeStyle=colors[k];g.setLineDash(s.dash?[4,4]:[]);g.beginPath();s.y.forEach((v,i)=>i?g.lineTo(px(i),py(v)):g.moveTo(px(i),py(v)));g.stroke();g.fillStyle=colors[k];g.fillText(s.name,80+220*k,15)});g.setLineDash([])}
function update(){let d=DATA[document.getElementById('stream').value],j=+document.getElementById('joint').value,m=document.getElementById('method').value,q=d.methods[m],sq=m.endsWith('/raw')?d.sdk_raw:d.sdk,n=q.length,x=d.time,e=q.map((v,i)=>Math.abs(v[j]-sq[i][j]));draw('q',x,[{name:m.endsWith('/raw')?'SDK raw (inverse IIR)':'SDK filtered',y:sq.map(v=>v[j])},{name:m,y:q.map(v=>v[j])}]);draw('error',x,[{name:'absolute error',y:e}]);let f=q.map((v,i)=>v.reduce((s,a,j)=>s+(a-sq[i][j])**2,0)/20),sum=0,c=f.map((v,i)=>Math.sqrt((sum+=v)/(i+1)));draw('rollout',x,[{name:'frame RMSE',y:f.map(Math.sqrt)},{name:'cumulative RMSE',y:c,dash:true}])}
(async()=>{let bytes=Uint8Array.from(atob(encoded),c=>c.charCodeAt(0));let ds=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));DATA=JSON.parse(await new Response(ds).text());let s=document.getElementById('stream'),j=document.getElementById('joint'),m=document.getElementById('method');Object.keys(DATA).forEach(k=>s.add(new Option(k,k)));for(let k=0;k<20;k++)j.add(new Option('q'+k,k));s.onchange=()=>{m.innerHTML='';Object.keys(DATA[s.value].methods).forEach(k=>m.add(new Option(k,k)));update()};j.onchange=m.onchange=update;s.onchange();document.getElementById('loading').remove()})().catch(e=>document.getElementById('loading').textContent=e);</script>'''.replace('PACKED_DATA',packed)
 html=html.replace('<h1>Wuji2 causal and joint trajectory solvers</h1>',f'<h1>Wuji2 causal and joint trajectory solvers</h1><p>Executed parameter cases: {len(details)}/{len(expected)}. Execution complete: {not missing}. Completion counts describe execution, not SDK-q precision equivalence.</p>')
 if (r/'oneway_comparison'/'summary.json').exists():
  html=html.replace('<h1>Wuji2 causal and joint trajectory solvers</h1>','<h1>Wuji2 causal and joint trajectory solvers</h1><p><a href="oneway_comparison/comparison.html">补充实验：单向因果、冻结历史、显式 KKT 验证与完整并行路径</a></p>')
 (r/'comparison.html').write_text(html)
 lines=['# Wuji2 稀疏时序求解对比','',f"完成 {len(details)}/{len(expected)} 个参数案例；完整完成：{not missing}。",'','[完整轨迹与误差交互曲线](comparison.html) · [全部数值](metrics.csv) · [来源和执行清单](MANIFEST.json)','','原始权重保持不变，不含网络初值正则或额外总偏移box。联合求解读取未来，因果版保留上一帧原始q状态。SDK提前停止和滤波使更低的优化目标不必对应更小的SDK q偏差。','', '联合模型曲线固定展示第二阶段16次预算；因果版首帧50/80、后续1/10次更新。避免依据留出集误差选择预算。原始q的SDK参照由FP32滤波逆推，filtered参照为SDK公共输出。所有帧、失败和未完成案例保留。','', '|划分|配置|手|后端|RMSE(rad)|最大误差(rad)|全流程(s)|SDK(s)|加速|','|---|---|---|---|---:|---:|---:|---:|---:|']
 for v in groups:
  if v['method']=='filtered' and v['budget']==16 and not v['reference']:lines.append(f"|{v['split']}|{v['configuration']}|{v['side']}|{v['backend']}|{v['rmse_rad']:.6f}|{v['max_error_rad']:.6f}|{v['seconds']:.4f}|{v['sdk_seconds']:.4f}|{v['full_speedup']:.2f}×|")
 lines+=['','所有时间为共享GPU环境下实测，正式网络和URDF加载、磁盘I/O、诊断评分不计入稳态全流程；输入特征、传输、网络、人手、目标构造、机器人和输出滤波计入。因果整帧图的阶段值为空，不将带有Python调度开销的拆分时延冒充融合图各阶段耗时。','','![开发集速度精度](figures/development_speed_accuracy.png)','','![留出集速度精度](figures/heldout_speed_accuracy.png)']
 lines+=['','[逐记录/collection误差](record_metrics.csv) · [数据集分组误差](dataset_metrics.csv) · [阶段与隔离性能测量](phase_timings.csv)','','同一报告中的成本必须按目标来源理解：各方法在自身人手重建目标上求解，`common_sdk_target_score`另外使用SDK人手目标做事后评分，未用于初始化。人手联合标量目标的曲率和接受规则与源码因果算法不同；因此较大的q偏差不能仅凭结果断言为不同盆地。','','冷启动时间保存在各案例`cold_first_solve`；稳态耗时包含网络及非SDK人手初值生成。融合因果图的隔离阶段时间不可相加代替正式全流程测量。GPU iLQR使用每条记录内部串行的原生CUDA Riccati递推；带状LM的PCR线性求解具有对数时间深度。Crocoddyl为3.2.1的C++ BoxFDDP，CPU/OpenMP8，单独标注；网络初值仍由GPU计算。','','SDK原始q参照由FP32滤波逆推，不能当作逐位可用的原始SDK状态。严格一致性结论使用滤波SDK输出。超过1e-4的偏差比例、最大误差和全部迭代预算保存在CSV/JSON中，不将更快但偏差更大的结果称为同精度加速。','','[原源码逐项目标与梯度验证](original_objective_validation.json) · [融合iLQR数值验证](ilqr_fusion_validation.json) · [分段/显存验证](structural_validation.json) · [原源码/SDK参考验证](sdk_reference_validation.json) · [执行修订及保留的失败](revisions/pre_fused_ilqr/repair.json)']
 if missing:lines+=['','未完成列表见 summary.json；不可把部分结果称为完整验收。']
 if (r/'oneway_comparison'/'summary.json').exists():lines+=['','[因果/冻结历史补充对照及并行边界](oneway_comparison/REPORT.md) · [补充交互曲线](oneway_comparison/comparison.html)']
 (r/'REPORT.md').write_text('\n'.join(lines)+'\n');print('REPORT',len(details),'/',len(expected),'CURVES',len(streams),flush=True)


if __name__=='__main__':main()
