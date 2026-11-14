#!/usr/bin/env python3
"""Re-score causal comparison arrays and build a standalone canvas report."""
import argparse,json,csv,gzip,base64
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;w=r/'oneway_comparison';rows=[];groups={};human=[];sdk_times={};joint_rows=[]
    parent_signature=json.loads((r/'MANIFEST.json').read_text())['candidate_signature']
    for config in ['builtin','hl_urdf']:
        sdk_times[config]=json.loads((r/f'sdk_{config}_timings.json').read_text())
        for side in ['left','right']:
            d=w/f'{config}_{side}';path=d/'result.json'
            if not path.exists():continue
            result=json.loads(path.read_text());data=np.load(d/'curves.npz');human.append(dict(configuration=config,side=side,**result['human_source_vs_sdk']))
            group=dict(configuration=config,side=side,offsets=data['record_offsets'].tolist(),records=data['record_indices'].tolist(),sdk=np.round(data['q_sdk_filtered'],6).tolist(),network=np.round(data['q_network'],6).tolist(),methods={})
            baseline_times=[v for v in sdk_times[config] if v['side']==side and v['trajectory'] in group['records']]
            sdk_robot_s=float(np.median([sum(v['timings'][i]['robot'] for v in baseline_times) for i in range(10)]))
            for row in result['rows']:
                key=f"{row['scenario']}__{row['mode']}__{row['updates']}";q=data[key];gt=data['q_sdk_filtered'];error=q-gt
                np.testing.assert_allclose(np.sqrt(np.square(error).mean()),row['rmse_rad'],rtol=1e-12)
                np.testing.assert_allclose(abs(error).max(),row['max_rad'],rtol=1e-12)
                row=dict(configuration=config,side=side,**row,sdk_robot_s=sdk_robot_s,robot_speedup=sdk_robot_s/row['median_s'],peak_resident_mib=result.get('peak_resident_mib'))
                rows.append(row);group['methods'][key]=np.round(q,6).tolist()
            # Existing all-time-parallel pipeline, same source/model/data, is
            # deliberately labeled bidirectional with a different human model.
            for budget in [16,32]:
                directory=r/'jobs'/f'development_banded_lm_{config}_{side}'/f'budget_{budget}'
                if not (directory/'result.json').exists():continue
                jr=json.loads((directory/'result.json').read_text())
                assert jr['complete'] and jr['candidate_signature']==parent_signature
                predictions=[]
                for index in group['records']:
                    jd=np.load(directory/f'trajectory_{index}_{side}.npz');predictions.append(jd['q_filtered'])
                jq=np.concatenate(predictions);assert jq.shape==data['q_sdk_filtered'].shape
                je=jq-data['q_sdk_filtered'];group['methods'][f'full_joint__{budget}']=np.round(jq,6).tolist()
                sdk_total=float(np.median([sum(v['timings'][i]['total'] for v in baseline_times) for i in range(10)]))
                joint_rows.append(dict(configuration=config,side=side,updates=budget,frames=len(jq),rmse_rad=float(np.sqrt(np.square(je).mean())),max_rad=float(abs(je).max()),total_s=float(np.median([v['total'] for v in jr['timings']])),sdk_total_s=sdk_total,phases={k:float(np.median([v[k] for v in jr['timings']])) for k in ['network','human_and_targets','robot','filter']},information='past_present_and_future; joint human scalar model differs from causal source rules',candidate_signature=jr['candidate_signature']))
            groups[config+'/'+side]=group
    aggregate=[]
    for scenario in sorted({v['scenario'] for v in rows}):
        for mode,updates in sorted({(v['mode'],v['updates']) for v in rows if v['scenario']==scenario}):
            values=[v for v in rows if v['scenario']==scenario and v['mode']==mode and v['updates']==updates];n=sum(v['frames'] for v in values)
            aggregate.append(dict(scenario=scenario,mode=mode,updates=updates,groups=len(values),frames=n,rmse_rad=float(np.sqrt(sum(v['frames']*v['rmse_rad']**2 for v in values)/n)),max_rad=max(v['max_rad'] for v in values),above_1e4_fraction=sum(v['frames']*v['above_1e4_fraction'] for v in values)/n,delta_rmse_rad=float(np.sqrt(sum((v['frames']-5)*v['delta_rmse_rad']**2 for v in values)/sum(v['frames']-5 for v in values))),robot_median_sum_s=sum(v['median_s'] for v in values),sdk_robot_median_sum_s=sum(v['sdk_robot_s'] for v in values),robot_speedup=sum(v['sdk_robot_s'] for v in values)/sum(v['median_s'] for v in values),fk_sdk_rmse_m=float(np.sqrt(sum(v['frames']*v['fk_sdk_rmse_m']**2 for v in values)/n)),fk_sdk_max_m=max(v['fk_sdk_max_m'] for v in values),projected_gradient_max=max(v['projected_gradient_max'] for v in values)))
    complete=len(rows)==80
    audit=json.loads((w/'sdk_causal_cpu_bound_audit.json').read_text()) if (w/'sdk_causal_cpu_bound_audit.json').exists() else None
    summary=dict(complete=complete,expected_cases=80,executed_cases=len(rows),rows=rows,aggregate=aggregate,human=human,full_joint_parallel_pipeline=joint_rows,sdk_bound_precision_audit=audit,neural_optimizer_updates=0,scope='Five development records, both hands and gloves. One-way timing is robot-only with cached neural initialization. Separately labeled joint pipeline includes measured full EMF-to-q latency and allows future information.',candidate_signature=json.loads((w/'MANIFEST.json').read_text())['candidate_signature'])
    fixed_path=w/'fixed_sdk_history_DIAGNOSTIC/summary.json';source_path=w/'source_slsqp_stopping/summary.json'
    fixed=json.loads(fixed_path.read_text()) if fixed_path.exists() else None
    source=json.loads(source_path.read_text()) if source_path.exists() else None
    summary['fixed_history_DIAGNOSTIC']=fixed;summary['source_slsqp_stopping']=source
    (w/'summary.json').write_text(json.dumps(summary,indent=2))
    for name,values in [('metrics.csv',rows),('aggregate.csv',aggregate)]:
        fields=[k for k,v in values[0].items() if not isinstance(v,(dict,list))] if values else []
        with (w/name).open('w',newline='') as f:writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(values)
    if joint_rows:
        values=[dict({k:v for k,v in row.items() if k!='phases'},**{k+'_s':v for k,v in row['phases'].items()}) for row in joint_rows]
        with (w/'joint_pipeline_metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,list(values[0]));writer.writeheader();writer.writerows(values)
    primary=[v for v in aggregate if v['scenario']=='source_causal_human_from_emf' and (v['updates'] in (32,200) or v['mode'].startswith('frozen_history'))]
    lines=['# 因果与冻结历史对照',f"\n完成 {len(rows)}/80 个配置；5 条完整开发记录，左右手、内置与 HL glove。",'\n每侧 2,885 帧，共 5,770 个独立手帧；两种 glove 合计 11,540 个手帧与 glove 配置，不能视为 11,540 个独立手帧。','\nSDK 公共输出为精确基准；raw SDK q 为逆滤波近似。网络仅作为初值，不加初值正则。', '\n| 方法 | final 更新 | RMSE rad | 最大误差 rad | FK RMSE mm | 机器人阶段加速 |', '|---|---:|---:|---:|---:|---:|']
    for v in primary:lines.append(f"| {v['mode']} | {v['updates']} | {v['rmse_rad']:.7f} | {v['max_rad']:.7f} | {1000*v['fk_sdk_rmse_m']:.4f} | {v['robot_speedup']:.2f}× |")
    lines+=['\n全轨迹并行对照：人手和机器人均批量计算导数，使用状态消元后的块三对角系统及 CUDA PCR。以下是真实测量的完整 EMF→q 时间，包含网络推理。该路线读取未来，人手联合标量目标也不同于源码因果接受规则，因此不应期待逐位复现 SDK。','\n| joint LM 更新 | 配置组 | RMSE rad | 最大误差 rad | 完整流程四组耗时和 s | SDK 耗时和 s | 加速 |','|---|---:|---:|---:|---:|---:|---:|']
    for budget in [16,32]:
        vs=[v for v in joint_rows if v['updates']==budget]
        if not vs:continue
        n=sum(v['frames'] for v in vs);rmse=np.sqrt(sum(v['frames']*v['rmse_rad']**2 for v in vs)/n);total=sum(v['total_s'] for v in vs);sdk=sum(v['sdk_total_s'] for v in vs)
        lines.append(f"| {budget} | {len(vs)} | {rmse:.7f} | {max(v['max_rad'] for v in vs):.7f} | {total:.4f} | {sdk:.4f} | {sdk/total:.2f}× |")
    lines+=['\n并行边界：joint banded LM 的各帧导数并行，PCR 为 O(log T) 轮；单向 causal_prefix 只将机器人时间线性化并行化，人手仍逐帧。frozen_history 两阶段按时间递推。GPU iLQR 的 Riccati backward pass 也有时间依赖。稀疏 KKT 经 shooting 状态消元得到本实现的块三对角系统，不因换一个通用求解器就自动改变非线性目标或 SDK 停止语义。']
    lines+=['\n两组并行方法 warm stage 为 4 次更新。逐帧方法 first=50/80、later=1/10 次 LM 更新，不等于 SDK 的函数评价上限。表中 online 的 10 表示 later final cap。', '\n因果/冻结历史表中的机器人阶段耗时按四组各自十次测量中位数求和，SDK 同样分组。该表排除网络推理、人手重建、目标生成和 IO；不能据此声称完整流程加速。整段联合表单独报告包含网络和人手重建的实测完整耗时。所有方法两次预热，不将编译计入稳态。', '\n`common_targets_DIAGNOSTIC` 直接使用 SDK 人手中间量，仅用于隔离机器人求解差异；`source_causal_human_from_emf` 从 EMF 独立重建。', '\ncausal_prefix 是单向非线性方程更新；frozen_parallel 冻结上一轮历史；frozen_history_previous 真正冻结已完成历史并以上一帧初始化；frozen_history_network 只改变当前帧初值。', '\n200 次更新不等于已证明收敛，仍需检查投影梯度。SDK q 偏差和 FK 几何误差分别报告，不能互相替代。']
    if audit:
        medians=[v['bound_tolerance_5e-06']['median'] for v in audit['rows']]
        lines.append(f'\nSDK 的因果投影梯度中位数为 {min(medians):.6g}–{max(medians):.6g}（使用 5e-6 rad 活动限位容差，缓解逆 IIR 的舍入影响）。该梯度指标不是 q 误差，不能与 1e-4 rad 控制误差阈值混用。SDK 使用 ftol_abs=1e-4 的目标变化停止条件及有限函数评价预算，而非梯度为零的验收。')
    if source:
        lines.append(f"\n完整轨迹的源码 SLSQP 与 SDK 输出逐位一致：{source['all_bitwise_equal']}。")
        for v in source['rows']:
            codes=v['statuses'][1];n=v['frames'];lines.append(f"\n{v['configuration']}/{v['side']}：final FTOL {int(codes.get('3',0))/n:.2%}，MAXEVAL {int(codes.get('5',0))/n:.2%}，fallback {v['fallback_frames']}。后续 warm maxeval=1 仅评价、不更新；LM 一次更新会改变 q，两者不能等同。")
    if fixed:
        lines+=['\n诊断：固定同一 source/SDK 历史 q 的独立帧优化。下表只统计非首帧，raw q 误差，不能当作生产轨迹性能。','\n| 初值 | warm 更新 | final 更新 | 四组 raw RMSE rad | 最大 raw 误差 rad |','|---|---:|---:|---:|---:|']
        for seed in ['source_previous_raw','network','source_current_raw_DIAGNOSTIC']:
            for warm in [0,1]:
                for final in [10,200]:
                    vs=[v for v in fixed['rows'] if v['seed']==seed and v['warm_updates']==warm and v['final_updates']==final]
                    if vs:
                        n=sum(v['frames']-5 for v in vs);rmse=np.sqrt(sum((v['frames']-5)*v['later_raw_sdk_rmse_rad']**2 for v in vs)/n);maximum=max(v['later_raw_sdk_max_rad'] for v in vs);lines.append(f'| {seed} | {warm} | {final} | {rmse:.7f} | {maximum:.7f} |')
        vs=[v for v in fixed['rows'] if v['seed']=='source_current_raw_DIAGNOSTIC' and v['warm_updates']==0 and v['final_updates']==200]
        if len(vs)==4:
            n=sum(v['frames']-5 for v in vs);rmse=np.sqrt(sum((v['frames']-5)*v['later_raw_sdk_rmse_rad']**2 for v in vs)/n)
            before=sum(v['baseline']['later_cost'] for v in vs);after=sum(v['later_cost'] for v in vs)
            linear=max(v['baseline']['native_linear_solve_max_error_vs_fp64'] for v in vs)
            lines.append(f'\n关键因果诊断：历史与目标完全固定，从已验证源码的当前 raw q 本身出发，不使用网络且跳过 later warm 更新，继续 200 次 final 优化后，相对原 raw q 的 RMSE 为 {rmse:.7f} rad；同一原始 stage2 目标由 {before:.9f} 降至 {after:.9f}。该结果只统计非首帧，主轨迹表仍保留全部首帧。它说明网络初值并不是 SDK 偏差的必要原因，有限步停止与更充分优化确有区别；并不排除其他初值带来的额外局部解差异。')
            lines.append(f'\n真实机器人活动集线性系统的 CUDA FP32 步与 FP64 步最大差为 {linear:.3g}；小规模显式稀疏 KKT 与状态消元 PCR 的 FP64 测试也已通过。尚未证明所有 200 步轨迹解都充分收敛，不能把预算名称当作最优性证书。')
    (w/'REPORT.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,scenario in zip(axes,['sdk_human_common_targets_DIAGNOSTIC','source_causal_human_from_emf']):
        for mode in ['causal_prefix','frozen_parallel','frozen_history_previous','frozen_history_network']:
            values=[v for v in aggregate if v['scenario']==scenario and v['mode']==mode]
            ax.plot([v['robot_median_sum_s'] for v in values],[v['rmse_rad'] for v in values],'-o',label=mode)
            for v in values:ax.annotate(str(v['updates']),(v['robot_median_sum_s'],v['rmse_rad']),fontsize=7)
        ax.set(xlabel='Sum of four robot group median latencies (s)',ylabel='Filtered q RMSE vs SDK (rad)',title='SDK human diagnostic' if 'DIAGNOSTIC' in scenario else 'Causal human from EMF');ax.grid(alpha=.3);ax.legend(fontsize=7)
    fig.savefig(w/'speed_error.png',dpi=160);plt.close(fig)
    payload=base64.b64encode(gzip.compress(json.dumps(groups,separators=(',',':')).encode())).decode()
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Wuji2 causal / frozen-history comparison</title><style>body{font:15px system-ui;margin:24px;background:#f6f7fa;color:#172033}select{margin:5px;padding:6px}canvas{background:white;width:100%;height:260px;border:1px solid #ddd;margin:6px 0}#legend{line-height:1.8}a{color:#246}</style><h2>Wuji2：因果、冻结历史与 SDK</h2><p>五条完整开发记录，左右手与两种 glove。图示保留到 1e-6 rad；原始数值见 NPZ / CSV。<a href="REPORT.md">数值报告</a> · <a href="metrics.csv">全部配置 CSV</a> · <a href="speed_error.png">速度—误差</a></p><p>本页因果路径保留逐帧人手重建；完整轨迹并行路径为 joint banded LM，允许未来信息，见数值报告。</p><p id="status"></p><select id="group"></select><select id="record"></select><select id="scenario"><option value="source_causal_human_from_emf">EMF 独立因果人手重建</option><option value="sdk_human_common_targets_DIAGNOSTIC">诊断：共用 SDK 人手目标</option></select><select id="budget"><option>4</option><option>16</option><option selected>32</option><option>200</option></select><select id="joint"></select><label><input type="checkbox" id="showjoint" checked>显示整段联合 LM（允许未来）</label><p id="legend"></p><div>关节轨迹（rad）</div><canvas id="trajectory"></canvas><div>逐帧 20 关节 RMSE（rad）</div><canvas id="error"></canvas><div>截至当前帧累计 RMSE（rad）</div><canvas id="cumulative"></canvas><p id="hover">移动鼠标查看帧号与数值。</p><script>
const packed='PAYLOAD';
async function start(){const raw=Uint8Array.from(atob(packed),c=>c.charCodeAt(0));const stream=new Blob([raw]).stream().pipeThrough(new DecompressionStream('gzip'));const groups=JSON.parse(await new Response(stream).text());const el=id=>document.getElementById(id);let traces=[];let activeFrame=null;
for(const key of Object.keys(groups))el('group').add(new Option(key,key));for(let j=0;j<20;j++)el('joint').add(new Option('joint '+j,j));
function recordOptions(){const g=groups[el('group').value];el('record').innerHTML='';g.records.forEach((r,i)=>el('record').add(new Option('record '+r+' ('+(g.offsets[i+1]-g.offsets[i])+' frames)',i)));draw();}
const colors=['#111827','#a3a3a3','#2563eb','#dc2626','#059669','#9333ea'];
function plot(id,series){const c=el(id),ratio=devicePixelRatio||1;c.width=c.clientWidth*ratio;c.height=260*ratio;const ctx=c.getContext('2d');ctx.scale(ratio,ratio);const W=c.clientWidth,H=260,pad=54;let lo=Infinity,hi=-Infinity;series.forEach(s=>s.y.forEach(v=>{lo=Math.min(lo,v);hi=Math.max(hi,v)}));const span=Math.max(hi-lo,1e-6);lo-=span*.06;hi+=span*.06;ctx.fillStyle='#536';ctx.font='12px monospace';for(let k=0;k<5;k++){const y=pad+(H-2*pad)*k/4;ctx.strokeStyle='#eee';ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(W-15,y);ctx.stroke();ctx.fillText((hi-(hi-lo)*k/4).toFixed(4),2,y+4);}series.forEach(s=>{ctx.strokeStyle=s.color;ctx.lineWidth=s.name==='SDK'?2:1.2;ctx.beginPath();s.y.forEach((v,i)=>{const x=pad+(W-pad-15)*i/Math.max(1,s.y.length-1),y=pad+(H-2*pad)*(hi-v)/(hi-lo);if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y);});ctx.stroke();});if(activeFrame!==null&&series.length){const x=pad+(W-pad-15)*activeFrame/Math.max(1,series[0].y.length-1);ctx.strokeStyle='#999';ctx.beginPath();ctx.moveTo(x,pad);ctx.lineTo(x,H-pad);ctx.stroke();}}
function draw(){const g=groups[el('group').value],r=+el('record').value,s=g.offsets[r],e=g.offsets[r+1],j=+el('joint').value,scenario=el('scenario').value,budget=el('budget').value,sdk=g.sdk.slice(s,e);traces=[{name:'SDK',q:sdk,color:colors[0]},{name:'network initial',q:g.network.slice(s,e),color:colors[1]}];['causal_prefix','frozen_parallel','frozen_history_previous','frozen_history_network'].forEach((mode,i)=>{let key=scenario+'__'+mode+'__'+(mode.startsWith('frozen_history')?'10':budget);if(g.methods[key])traces.push({name:mode,q:g.methods[key].slice(s,e),color:colors[i+2]});});if(scenario==='source_causal_human_from_emf'&&el('showjoint').checked&&g.methods['full_joint__'+budget])traces.push({name:'joint LM (own human, future allowed)',q:g.methods['full_joint__'+budget].slice(s,e),color:'#d97706'});el('legend').innerHTML=traces.map(t=>'<span style="color:'+t.color+'">● '+t.name+'</span>').join('　');plot('trajectory',traces.map(t=>({name:t.name,color:t.color,y:t.q.map(q=>q[j])})));let errs=traces.filter(t=>t.name!=='SDK').map(t=>({name:t.name,color:t.color,y:t.q.map((q,i)=>Math.sqrt(q.reduce((a,v,k)=>a+(v-sdk[i][k])**2,0)/20))}));plot('error',errs);plot('cumulative',errs.map(t=>{let acc=0;return {...t,y:t.y.map((v,i)=>{acc+=v*v;return Math.sqrt(acc/(i+1));})};}));el('status').textContent='CASECOUNT/80 个配置；200 次是更新预算，不是最优解证明。';}
for(const id of ['scenario','budget','joint','record','showjoint'])el(id).onchange=()=>{activeFrame=null;draw();};el('group').onchange=recordOptions;for(const id of ['trajectory','error','cumulative'])el(id).onmousemove=evt=>{if(!traces.length)return;activeFrame=Math.max(0,Math.min(traces[0].q.length-1,Math.round((evt.offsetX-54)/(el(id).clientWidth-69)*(traces[0].q.length-1))));el('hover').textContent='frame '+activeFrame+' | '+traces.map(t=>t.name+': '+t.q[activeFrame][+el('joint').value].toFixed(6)).join(' | ');draw();};window.onresize=draw;recordOptions();}start().catch(e=>document.getElementById('status').textContent=e.stack);
</script></html>'''.replace('PAYLOAD',payload).replace('CASECOUNT',str(len(rows)))
    (w/'comparison.html').write_text(html);print(json.dumps(dict(executed_cases=len(rows),complete=complete,aggregate=primary),indent=2))

if __name__=='__main__':main()
