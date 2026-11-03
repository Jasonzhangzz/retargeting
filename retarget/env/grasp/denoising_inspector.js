'use strict';
const $=id=>document.getElementById(id), node=(tag,text)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e};
const controls=document.querySelector('.controls');
controls.replaceChildren();
function select(id,label,items){const l=node('label',label),s=node('select');s.id=id;s.setAttribute('aria-label',label);for(const [value,text] of items){const o=node('option',text);o.value=String(value);s.append(o)}l.append(s);controls.append(l);return s}
function button(id,text){const b=node('button',text);b.id=id;controls.append(b);return b}
button('previous','←');select('grasp','抓型',DATA.cases.map((c,i)=>[i,c.label]));button('next','→');
select('seed','噪声种子',DATA.seeds.map(s=>[s,s]));
select('round','累计加噪轮数',Array.from({length:DATA.rounds+1},(_,r)=>[r,r===0?'0 · 无噪声':`${r} 轮`]));
button('play','播放加噪');
select('fraction','尺寸减少',DATA.fractions.map(f=>[f,`${(100*f).toFixed(0)}%`]));
select('stage-select','预测显示阶段',[
 ['clean','① 去噪结果'],['tight','② 去噪 → 沿尺寸收紧'],['noise','③ 收紧 → 加回原始高斯噪声'],
 ['removed','对照：收紧 → 加回各方法去除量'],['perpendicular','对照：收紧 → 加回 PCA 正交残差']]);
select('geometry','显示',[['mesh','真实手部网格'],['skeleton','骨架']]);
for(const [id,label,checked] of [['object','物体',true],['transparent','透明手部',false],['bones','骨架叠加',false],['sync','同步视角',true]]){const l=node('label'),i=node('input');i.type='checkbox';i.id=id;i.checked=checked;l.append(i,node('span',label));controls.append(l)}
button('front','正面');button('side','侧面');button('back','背面');button('download','导出当前数据');
const link=node('a','打开 Viser · 全部缓存 pose ↗');link.href=DATA.viser_url;link.target='_blank';link.rel='noopener';controls.append(link);
const style=node('style');style.textContent='.stage{grid-template-columns:1fr 1fr}.plot{height:430px}.controls{gap:10px}.controls select{max-width:340px}.panel h2{min-height:36px;gap:8px}.panel small{font-size:11px}.status-line{font:12px monospace;color:#566b6e;margin:12px 30px}.phase{margin:12px 30px;display:flex;gap:8px;flex-wrap:wrap}.phase button.active{background:#006e65;color:white}.details{max-height:none}.details p{font-size:12px}.details h2{font-size:16px}.warning-text{color:#ab4a19}.plot-joints{height:260px}@media(max-width:850px){.stage{grid-template-columns:1fr}.plot{height:400px}.controls{padding:12px}.controls select{max-width:280px}}';document.head.append(style);
document.title='Grasp pose · PCA / Diffusion';$('title').textContent='Grasp pose · 加噪、去噪与尺寸收紧';$('eyebrow').textContent='WUJI2 / DENOISING INSPECTOR';
$('subtitle').textContent=`${DATA.cases.length} 类的标准 1× 缓存姿态。${DATA.noise_mode==='fingers'?'随机选择 0–5 根手指，各指独立采样加噪轮数。':'全部手指逐轮加噪。'}单轮 σ=${DATA.sigma_step}，PCA 与 ${DATA.diffusion_method||'diffusion'} 共享同一输入。`;
const colors={truth:'#775c35',observed:'#0077c8',pca:'#00a080',diffusion:'#a84bce'};
$('legend').replaceChildren(...Object.entries({truth:'标准参考',observed:'同一加噪输入',pca:'PCA 预测',diffusion:'Diffusion 预测'}).map(([key,label])=>{const e=node('span'),dot=node('i');dot.className='dot';dot.style.background=colors[key];e.append(dot,node('span',label));return e}));
const panels=[], headings=[],subtitles=[];
for(const label of ['标准缓存姿态','累计高斯加噪','PCA','Diffusion']){const s=node('section'),h=node('h2'),title=node('span',label),small=node('small'),p=node('div');s.className='panel';p.className='plot';h.append(title,small);s.append(h,p);$('stage').append(s);panels.push(p);headings.push(title);subtitles.push(small)}
const phases=node('div');phases.className='phase';$('stage').before(phases);
for(const [value,label] of [['clean','① 预测去噪'],['tight','② 沿缓存尺寸收紧'],['noise','③ 加回同一原始噪声']]){const b=node('button',label);b.dataset.stage=value;b.onclick=()=>{$('stage-select').value=value;schedule()};phases.append(b)}
const status=node('div');status.className='status-line';$('stage').before(status);
const metricPanel=node('section');metricPanel.className='details';metricPanel.append(node('h2','当前姿态诊断'));const metricTable=node('table');metricPanel.append(metricTable);$('metrics').after(metricPanel);
const historyPanel=node('section');historyPanel.className='details';historyPanel.append(node('h2','每轮加噪与预测误差'));const historyPlot=node('div');historyPlot.className='plot-joints';historyPanel.append(historyPlot);metricPanel.after(historyPanel);
$('details').querySelector('b').textContent='20 个手指关节 · 当前阶段对照';$('details').querySelector('p').textContent='选择一行高亮相应手指；所有坐标以米表示，关节以弧度表示。前 6 个手腕坐标固定。';
const sourcePanel=node('details');sourcePanel.className='details';sourcePanel.append(node('summary','来源、计算定义与导出'));const sourceText=node('p');sourcePanel.append(sourceText);const exports=node('p');for(const [file,label] of [['experiment.json','完整实验 JSON'],['poses.npz','全精度阶段 pose NPZ'],['verification.json','数值核对']]){const a=node('a',label);a.href=file;exports.append(a,document.createTextNode('　'))}sourcePanel.append(exports);$('details').after(sourcePanel);
$('footer').textContent='手部使用真实 MuJoCo 网格与关节变换。坐标系和范围在同一抓型的各轮及各阶段保持固定，未逐姿态缩放。物体始终保持源尺寸；收紧仅沿该抓型缓存的关节尺寸曲线移动。原始候选不裁剪；越限、自碰撞、穿透在诊断中显示。这里只检验候选姿态，未进行动力学稳定性验证。页面脚本、网格和数据均内嵌，可离线打开。';
$('grasp').value=String(Math.max(0,DATA.cases.findIndex(c=>c.label.startsWith('24_'))));$('round').value=String(Math.min(4,DATA.rounds));$('fraction').value=String(DATA.fractions.includes(.5)?.5:DATA.fractions[0]);
$('stage-select').value='tight';
let selectedJoint=null,queue=Promise.resolve(),playing=false,playGeneration=0,busy=false;
let camera={eye:{x:0,y:2.05,z:.25},up:{x:0,y:0,z:1},center:{x:0,y:0,z:0}};
const cameras=Array.from({length:4},()=>structuredClone(camera)),linked=new Set(),ranges=new Map();
function current(){const c=DATA.cases[+$('grasp').value],path=c.paths.find(p=>p.seed===+$('seed').value),r=path.rounds[+$('round').value],f=+$('fraction').value;return {c,path,r,f,stage:$('stage-select').value}}
function fractionKey(f){return Object.keys(DATA.cases[0].curves).find(k=>Number(k)===Number(f))}
function methodPose(c,r,method,f,stage){const q=r.methods[method].clean.slice(),delta=c.curves[fractionKey(f)].delta;
 if(stage!=='clean')for(let i=0;i<20;i++)q[i+6]+=delta[i]*DATA.half[i];
 if(stage==='noise')for(let i=0;i<20;i++)q[i+6]+=r.noise[i]*DATA.half[i];
 if(stage==='removed')for(let i=6;i<26;i++)q[i]+=r.observed[i]-r.methods[method].clean[i];
 if(stage==='perpendicular')for(let i=0;i<20;i++)q[i+6]+=r.perpendicular[i]*DATA.half[i];return q}
function mesh(model,fr,object,color){const v=[],f=[];for(const g of model.geoms){if(g.object!==object)continue;const src=DATA.meshes[g.mesh],rotation=mm(fr[g.body].r,qr(g.quat)),p=point(fr,g.body,g.p),offset=v.length;for(const x of src.v)v.push(add(p,mv(rotation,x)));for(const face of src.f)f.push(face.map(i=>i+offset))}return {type:'mesh3d',x:v.map(p=>p[0]),y:v.map(p=>p[1]),z:v.map(p=>p[2]),i:f.map(p=>p[0]),j:f.map(p=>p[1]),k:f.map(p=>p[2]),color,opacity:object?.3:($('transparent').checked?.3:1),lighting:{ambient:.6,diffuse:.75,specular:.15},hoverinfo:'skip',showlegend:false}}
function draw(model,q,color){const fr=frames(model,q),points=jointPoints(model,fr),traces=[];
 if($('object').checked)traces.push(mesh(model,fr,true,'#dc8b09'));
 if($('geometry').value==='mesh')traces.push(mesh(model,fr,false,'#a9b8b1'));
 if($('geometry').value==='skeleton'||$('bones').checked)traces.push(skeleton(points,color));
 traces.push({type:'scatter3d',mode:'markers',x:points.filter((_,i)=>i>0&&i%4===0).map(p=>p[0]),y:points.filter((_,i)=>i>0&&i%4===0).map(p=>p[1]),z:points.filter((_,i)=>i>0&&i%4===0).map(p=>p[2]),marker:{size:3,color},hoverinfo:'skip',showlegend:false});
 if(selectedJoint!==null){const finger=Math.floor(selectedJoint/4),p=points.slice(1+4*finger,5+4*finger);traces.push({type:'scatter3d',mode:'lines+markers',x:p.map(v=>v[0]),y:p.map(v=>v[1]),z:p.map(v=>v[2]),line:{color,width:10},marker:{color,size:6},showlegend:false,hoverinfo:'skip'})}return traces}
function bounds(c){if(ranges.has(c.group))return ranges.get(c.group);const model=DATA.models[c.group],low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];
 const extend=points=>{for(const p of points)for(let k=0;k<3;k++){low[k]=Math.min(low[k],p[k]);high[k]=Math.max(high[k],p[k])}};
 const fr=frames(model,c.truth);for(const g of model.geoms){const rot=mm(fr[g.body].r,qr(g.quat)),p=point(fr,g.body,g.p);extend(DATA.meshes[g.mesh].v.map(v=>add(p,mv(rot,v))))}
 for(const path of c.paths)for(const r of path.rounds){extend(jointPoints(model,frames(model,r.observed)));for(const method of DATA.methods)for(const f of DATA.fractions)for(const s of ['clean','tight','noise','removed','perpendicular'])extend(jointPoints(model,frames(model,methodPose(c,r,method,f,s))))}
 const span=Math.max(...high.map((h,i)=>h-low[i]))*1.15,range=low.map((l,i)=>[(l+high[i]-span)/2,(l+high[i]+span)/2]);ranges.set(c.group,range);return range}
const fmt=(v,d=3)=>Number(v).toFixed(d),rmse=(a,b)=>Math.sqrt(a.reduce((sum,v,i)=>sum+(v-b[i])**2,0)/a.length);
function row(table,values,header=false){const tr=node('tr');for(const value of values)tr.append(node(header?'th':'td',String(value)));table.append(tr);return tr}
function metadata(ctx){const {c,path,r,f,stage}=ctx;
 const target=c.curves[fractionKey(f)].target,actual=(1-target/c.source_size)*100;
 $('coverage').textContent=`${+$('grasp').value+1} / ${DATA.cases.length} TYPES · ${DATA.cached_poses} 缓存 pose`;
 const phase=$('stage-select').selectedOptions[0].textContent;
 const timing=DATA.flow?`Flow t=${r.integration_start}→${r.integration_end} / Heun ${r.integration_steps} 步`:`DDPM 对应时刻 ${r.timestep}（近似各向同性 σ=${fmt(r.model_sigma,5)}）`;
 status.textContent=`${c.label} · ${c.primitive} / ${c.size_axis} · 源倍率 ${fmt(c.source_size,4)} / 曲线目标 ${fmt(target,4)} · 第 ${r.round} 轮 / RMS σ=${fmt(r.sigma,5)} · ${timing}`;
 subtitles[0].textContent=`原始 ${c.source_size}× / 节点 ${c.index}`;subtitles[1].textContent=`${r.round} 轮 / ${fmt(r.sigma,4)}`;for(let i=2;i<4;i++)subtitles[i].textContent=phase;
 $('note').textContent=stage==='clean'?`同一累计噪声输入同时送入两个方法。PCA 做类内正交投影；${DATA.diffusion_method||'diffusion'} 使用 seed ${DATA.diffusion_seed} checkpoint 和已知抓型、primitive、源尺寸。${DATA.flow?'Flow 固定观测 latent，推理不读取真实噪声、加噪轮数或 σ。':'DDPM 使用累计噪声的 RMS σ 作为各向同性近似。'}“去噪结果”是预测估计。`:stage==='tight'?`仅改变干净估计：q̂ + curve(${fmt(target,4)}) − curve(${fmt(c.source_size,4)})。两个方法使用同一条训练节点尺寸曲线；实际尺寸减少 ${fmt(actual,1)}%。` :stage==='noise'?'加回的是本轮输入中已知的完整累计高斯噪声 ε_total：q̂_tight + ε_total。两个方法加回完全相同的向量，不会重新采样。':stage==='removed'?'对照：加回各方法去掉的量 observed − clean_prediction。该量包含预测误差，不等同于已知高斯噪声；因此最终结果恒为 observed + 尺寸曲线位移。':'对照：加回 observed − PCA_projection(observed)。这是 PCA 正交残差，仅含噪声的正交分量及参考姿态的截断残差。';
 if(actual+1e-8<f*100)$('note').textContent+=` 当前请求减少 ${fmt(f*100,0)}%，已到该抓型缓存训练尺寸的下限，实际只能减少 ${fmt(actual,1)}%；继续提高比例不会再收紧。`;
 for(const b of phases.children)b.classList.toggle('active',b.dataset.stage===stage);
 const pm=r.methods.pca.fractions[fractionKey(f)][stage],dm=r.methods.diffusion.fractions[fractionKey(f)][stage],all=[c.truth_metrics,r.observed_metrics,pm,dm];
 metricTable.replaceChildren();row(metricTable,['诊断','标准参考','加噪输入','PCA · 当前阶段','Diffusion · 当前阶段'],true);
 for(const [label,key,digits] of [['相对标准关节 RMSE / 归一化','rmse',5],['模板接触包络 RMS / mm','enclosure_mm',2],['最大手物穿透 / mm','object_mm',3],['最大自穿透 / mm','self_mm',3],['最大关节越限 / rad','violation_rad',5]])row(metricTable,[label,...all.map(m=>fmt(m[key],digits))]);
 $('metrics').replaceChildren();for(const [label,value] of [['曲线目标尺寸减少',fmt(actual,1)+'%'],['PCA 当前包络 − 标准',fmt(pm.enclosure_mm-c.truth_metrics.enclosure_mm,2)+' mm'],['Diffusion 当前包络 − 标准',fmt(dm.enclosure_mm-c.truth_metrics.enclosure_mm,2)+' mm']]){const d=node('div');d.className='metric';d.append(node('b',value),node('span',label));$('metrics').append(d)}
 const maxPen=Math.max(pm.object_mm,dm.object_mm),maxViolation=Math.max(pm.violation_rad,dm.violation_rad);
 const warn=node('p',`尺寸曲线收紧不保证每类实际开口减小。当前两预测最大手物穿透 ${fmt(maxPen)} mm，最大关节越限 ${fmt(maxViolation,5)} rad。收紧及叠加阶段的 RMSE 是相对源标准姿态的偏移，不是目标尺寸真值误差。`);warn.className='warning-text';metricPanel.querySelector('p')?.remove();metricPanel.append(warn);
 $('thead').replaceChildren();row($('thead'),['关节','标准 / rad','加噪 / rad','PCA / rad','Diffusion / rad','累计噪声 / rad'],true);$('rows').replaceChildren();
 const pq=methodPose(c,r,'pca',f,stage),dq=methodPose(c,r,'diffusion',f,stage),fingers=['拇指','食指','中指','无名指','小指'];
 for(let i=0;i<20;i++){const tr=row($('rows'),[`${fingers[Math.floor(i/4)]} · ${i%4+1}`,...[c.truth,r.observed,pq,dq].map(q=>fmt(q[i+6],5)),fmt(r.noise[i]*DATA.half[i],5)]);tr.tabIndex=0;tr.classList.toggle('selected',selectedJoint===i);const pick=()=>{selectedJoint=selectedJoint===i?null:i;schedule()};tr.onclick=pick;tr.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();pick()}}}
 sourceText.textContent=`标准姿态来自 ${DATA.source}，源节点 ${c.source_index}，实验节点 ${c.index}（${c.split}）。源物体尺寸 ${c.dimensions_m.map(v=>fmt(1000*v,2)).join(' × ')} mm。训练与 PCA：${DATA.experiment}；${DATA.diffusion_method||'diffusion'} seed ${DATA.diffusion_seed}。噪声基础种子 ${path.seed}，本抓型有效种子 ${path.effective_seed}；各轮独立 N(0, ${DATA.sigma_step}² I)，各指累计 σ：${(r.sigma_fingers||Array(5).fill(r.sigma)).map(v=>fmt(v,4)).join(' / ')}。${path.finger_rounds?'各指总轮数：'+path.finger_rounds.join(' / ')+'。':''}归一化仅用于手指关节；手腕保持标准值。尺寸曲线仅使用训练尺寸节点，并限制在缓存训练范围内。碰撞和包络由原生 MuJoCo 计算。`;
 return {pq,dq};}
async function render(){const ctx=current(),{c,path,r}=ctx,{pq,dq}=metadata(ctx),m=DATA.models[c.group],range=bounds(c),qs=[c.truth,r.observed,pq,dq];busy=true;
 try{for(let i=0;i<4;i++){const scene={aspectmode:'cube',camera:cameras[i]};for(let k=0;k<3;k++)scene[['xaxis','yaxis','zaxis'][k]]={range:range[k],title:['x / m','y / m','z / m'][k],showbackground:false,gridcolor:'#e7ede7',zeroline:false,nticks:4,tickfont:{size:9}};
 await Plotly.react(panels[i],draw(m,qs[i],Object.values(colors)[i]),{scene,margin:{l:0,r:0,t:5,b:0},paper_bgcolor:'#fff',uirevision:'grasp-fixed'},{responsive:true,displaylogo:false,displayModeBar:false});
 if(!linked.has(i)){panels[i].on('plotly_relayout',async ev=>{if(busy||!ev['scene.camera'])return;camera=structuredClone(ev['scene.camera']);cameras[i]=structuredClone(camera);if(!$('sync').checked)return;busy=true;try{for(let j=0;j<4;j++)if(j!==i){cameras[j]=structuredClone(camera);await Plotly.relayout(panels[j],{'scene.camera':camera})}}finally{busy=false}});linked.add(i)}}
 const x=path.rounds.map(r=>r.round),series=[['加噪输入','observed'],['PCA','pca'],['Diffusion','diffusion']].map(([label,key],i)=>({type:'scatter',mode:'lines+markers',name:label,x,y:path.rounds.map(r=>key==='observed'?r.observed_metrics.rmse:r.methods[key].fractions[fractionKey(ctx.f)].clean.rmse),line:{color:[colors.observed,colors.pca,colors.diffusion][i],width:2},marker:{size:6}}));
 await Plotly.react(historyPlot,series,{margin:{l:65,r:25,t:15,b:45},xaxis:{title:'累计加噪轮数',dtick:1},yaxis:{title:'去噪后相对标准 RMSE'},legend:{orientation:'h',y:1.15},shapes:[{type:'line',xref:'x',yref:'paper',x0:r.round,x1:r.round,y0:0,y1:1,line:{color:'#566b6e',dash:'dot'}}],paper_bgcolor:'#fff',plot_bgcolor:'#fff'},{responsive:true,displaylogo:false,displayModeBar:false});
 }finally{busy=false}}
function fail(e){$('error').hidden=false;$('error').textContent='页面显示失败：'+e.message;console.error(e)}
function schedule(){queue=queue.then(render).catch(fail);return queue}
function stop(){playing=false;playGeneration++;$('play').textContent='播放加噪'}
for(const id of ['grasp','seed','round','fraction','stage-select','geometry','object','transparent','bones'])$(id).onchange=()=>{stop();if(id==='grasp')selectedJoint=null;schedule()};
$('previous').onclick=()=>{stop();$('grasp').value=String((+$('grasp').value-1+DATA.cases.length)%DATA.cases.length);selectedJoint=null;schedule()};
$('next').onclick=()=>{stop();$('grasp').value=String((+$('grasp').value+1)%DATA.cases.length);selectedJoint=null;schedule()};
$('sync').onchange=()=>{if($('sync').checked){for(let i=0;i<4;i++)cameras[i]=structuredClone(camera);schedule()}};
for(const [id,eye] of [['front',{x:0,y:2.05,z:.15}],['side',{x:2.05,y:0,z:.15}],['back',{x:0,y:-2.05,z:.15}]])$(id).onclick=()=>{camera={eye,up:{x:0,y:0,z:1},center:{x:0,y:0,z:0}};for(let i=0;i<4;i++)cameras[i]=structuredClone(camera);schedule()};
$('play').onclick=async()=>{if(playing){stop();return}playing=true;const generation=++playGeneration;$('play').textContent='停止播放';for(let r=0;r<=DATA.rounds&&playing&&generation===playGeneration;r++){$('round').value=String(r);await schedule();await new Promise(resolve=>setTimeout(resolve,900))}if(generation===playGeneration)stop()};
$('download').onclick=()=>{const {c,path,r,f,stage}=current(),value={grasp:c.label,reference_qpos:c.truth,observed_qpos:r.observed,noise_normalized:r.noise,noise_radians:r.noise.map((v,i)=>v*DATA.half[i]),seed:path.effective_seed,round:r.round,sigma:r.sigma,source_size:c.source_size,target_size:c.curves[fractionKey(f)].target,delta_normalized:c.curves[fractionKey(f)].delta,stage,pca_qpos:methodPose(c,r,'pca',f,stage),diffusion_qpos:methodPose(c,r,'diffusion',f,stage)};const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'})),a=node('a');a.href=url;a.download=`${c.label}_round${r.round}_${stage}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
schedule();
