// Offline model learning only. Missing measurements remain gaps, never zeros.
(()=>{
 const $=id=>document.getElementById(id),finite=v=>typeof v==='number'&&Number.isFinite(v);
 const names={extrinsics:'EMF外参',dimensions:'尺寸',kinematics:'运动学',tactile:'触觉注册',surface:'低维表面',mixed:'全量混训',reference:'整体手型',geometry:'非对指几何',pinches:'局部对指',all_pinches:'四组联合复测',pinch_index:'拇／食',pinch_middle:'拇／中',pinch_ring:'拇／无名',pinch_pinky:'拇／小'};
 let points=[],runId=null,cursor=0,pending=false,device=null,meta={},error='',liveFit=null;
 const methods={moxian_wuji:'Moxian六动作骨长与EMF外参',wuji_native:'Wuji原生六动作标定'};
 function activeFit(){return liveFit&&(!$('manoTrainingMethod').value||$('manoTrainingMethod').value===liveFit.method)&&(!runId||liveFit.run_id===runId)?liveFit:null;}
 function emptyMessage(){const fit=activeFit();return fit?.error?(['bone_extrinsics','wuji_native','moxian_wuji'].includes(fit.method)?'拟合失败：':'训练失败：')+fit.error:fit?.message?fit.message+'；尚未产生参数更新测量':'等待真实参数更新测量';}
 function selected(data,scope){
  if(scope==='all')return data;
  if(scope==='replay')return data.filter(p=>p.scope===p.phase||p.scope==='all_pinches');
  return data.filter(p=>p.scope===scope);
 }
 function segments(data,path){
  const result=[];let previous=null,line=null;
  for(const p of data){const value=path.reduce((o,k)=>o?.[k],p),key=p.phase+':'+p.scope;
   if(!finite(value)||!finite(p.step)){previous=null;line=null;continue;}
   if(key!==previous){line=[];result.push(line);}line.push([p.step,value]);previous=key;
  }return result;
 }
 // Helpers are exposed for contract tests and diagnostics, not model updates.
 window.ManoTrainingPlot={selected,segments};
 function plot(id,data,series,log=false){
  const canvas=$(id);if(!canvas)return;
  const ratio=window.devicePixelRatio||1,w=Math.max(260,canvas.clientWidth||400),h=170;
  canvas.width=w*ratio;canvas.height=h*ratio;
  const ctx=canvas.getContext('2d');if(!ctx)return;ctx.scale(ratio,ratio);ctx.clearRect(0,0,w,h);
  const transform=v=>log?Math.log10(1+Math.max(0,v)):v;
  const all=series.flatMap(s=>segments(data,s.path).flat()).filter(p=>finite(p[1]));
  ctx.font='10px sans-serif';ctx.fillStyle='#7c786f';
  if(!all.length){ctx.fillText(emptyMessage(),12,80);return;}
  const xmin=Math.min(...all.map(p=>p[0])),xmax=Math.max(xmin+1,...all.map(p=>p[0]));
  let ymin=Math.min(...all.map(p=>transform(p[1]))),ymax=Math.max(...all.map(p=>transform(p[1])));
  const pad=Math.max((ymax-ymin)*.08,Math.abs(ymax)*.03,.001);ymin=Math.max(0,ymin-pad);ymax+=pad;
  const X=x=>44+(x-xmin)/(xmax-xmin)*(w-58),Y=y=>h-27-(transform(y)-ymin)/(ymax-ymin)*(h-42);
  ctx.strokeStyle='#e0ddd7';ctx.lineWidth=1;
  for(let k=0;k<4;k++){const y=ymin+(ymax-ymin)*k/3,py=h-27-(y-ymin)/(ymax-ymin)*(h-42);ctx.beginPath();ctx.moveTo(42,py);ctx.lineTo(w-12,py);ctx.stroke();const value=log?10**y-1:y;ctx.fillText(value.toPrecision(3),0,py+3);}
  ctx.fillText(String(xmin),44,h-7);ctx.fillText(`${xmax} 步`,Math.max(48,w-72),h-7);
  for(const s of series){ctx.strokeStyle=s.color;ctx.fillStyle=s.color;ctx.lineWidth=1.6;ctx.setLineDash(s.dash?[4,3]:[]);
   for(const line of segments(data,s.path)){ctx.beginPath();line.forEach((p,i)=>i?ctx.lineTo(X(p[0]),Y(p[1])):ctx.moveTo(X(p[0]),Y(p[1])));ctx.stroke();for(const p of line){ctx.beginPath();ctx.arc(X(p[0]),Y(p[1]),1.7,0,2*Math.PI);ctx.fill();}}
  }ctx.setLineDash([]);
  canvas.onmousemove=e=>{const x=e.offsetX,step=xmin+(x-44)/(w-58)*(xmax-xmin);const p=data.reduce((a,b)=>!a||Math.abs(b.step-step)<Math.abs(a.step-step)?b:a,null);if(!p)return;
   const fmt=x=>finite(x)?x.toPrecision(5):'未知';
   $('manoTrainingHover').textContent=`第 ${p.step} 次目标评估 · 目标值 ${fmt(p.raw?.loss)}`+(finite(p.raw?.geometry_gradient_norm)?` · 参数梯度 ${fmt(p.raw.geometry_gradient_norm)}`:'');
  };
 }
 function render(){
  const data=points,log=$('manoTrainingLogScale').checked,fit=activeFit();
  const method=meta.method||fit?.method||$('manoTrainingMethod').value;
  plot('manoLossChart',data,[{path:['raw','loss'],color:'#287e63'}],log);
  $('manoPriorCaption').textContent='参数先验 · 骨长、骨长比例、EMF外参';
  const native=method==='wuji_native';
  for(const id of ['manoKLChart','manoGradientChart'])$(id).closest('figure').hidden=native;
  if(!native){
   plot('manoKLChart',data,[{path:['raw','terms','bone_prior'],color:'#377ca2'},{path:['raw','terms','bone_ratio_prior'],color:'#8751a1'},{path:['raw','terms','extrinsics_prior'],color:'#be7c35'}],log);
   plot('manoGradientChart',data,[{path:['raw','geometry_gradient_norm'],color:'#287e63'}],log);
  }
  const p=points.at(-1),loss=p?.raw?.loss;
  const outcome=fit?.validation_pending?'拟合完成，待独立复测':fit?.accepted?'独立验收通过':fit?.state==='completed'?'计算完成，结果见验收报告':'';
  $('manoTrainingStatus').textContent=error?'记录读取失败：'+error:`${methods[method]||'六动作标定'} · ${p?'第 '+p.step+' 次目标评估 · 目标值 '+(finite(loss)?loss.toPrecision(5):'未知'):(fit?.message||'等待点击拟合')} · ${outcome}`;
 }
 async function poll(){
  if(pending||document.hidden||!$('manoTrainingCurves').open)return;pending=true;const requested=device,requestedMethod=$('manoTrainingMethod').value;
  try{
   const q=new URLSearchParams({after:String(cursor)});if(runId)q.set('run_id',runId);
   const method=$('manoTrainingMethod').value;if(method)q.set('method',method);
   const response=await fetch('/api/hand_model/training_trace?'+q,{cache:'no-store',signal:AbortSignal.timeout(8000)});
   if(!response.ok)throw Error(`HTTP ${response.status}`);const result=await response.json();
   if(requested!==device||requestedMethod!==$('manoTrainingMethod').value)return;
   if(!Array.isArray(result.points))throw Error('无效的曲线响应');
   const changed=!!result.reset||result.run_id!==runId||result.points.length>0||meta.status!==result.status||!!error;
   if(result.reset||result.run_id!==runId)points=[];
   runId=result.run_id;meta=result;cursor=result.cursor;
   const seen=new Set(points.map(p=>p.sequence));
   points.push(...result.points.filter(p=>p.source==='train_only'&&finite(p.step)&&!seen.has(p.sequence)));points=points.slice(-4096);error='';if(changed)render();
  }catch(e){if(requested===device){error=e.message;render();}}
  finally{pending=false;}
 }
 window.updateTrainingCharts=state=>{
  const nextFit=state?.hand_model?.fit||null;
  const next=JSON.stringify([nextFit?.run_id,nextFit?.state,nextFit?.phase,nextFit?.message,nextFit?.accepted]);
  const old=JSON.stringify([liveFit?.run_id,liveFit?.state,liveFit?.phase,liveFit?.message,liveFit?.accepted]);liveFit=nextFit;
  const key=state?.device?.serial+':'+state?.side;
  if(key!==device){device=key;runId=null;cursor=0;points=[];poll();}
  if(next!==old)render();
 };
 $('manoTrainingLogScale').onchange=render;
 $('manoTrainingCurves').addEventListener('toggle',()=>{if($('manoTrainingCurves').open){poll();render();}});
 $('manoTrainingMethod').onchange=()=>{runId=null;cursor=0;points=[];meta={};const method=$('manoTrainingMethod').value;$('manoTrainingExport').href='/api/hand_model/training_trace?after=0'+(method?'&method='+encodeURIComponent(method):'');poll();render();};
 window.addEventListener?.('resize',render);
 if(window.currentMoxianState)window.updateTrainingCharts(window.currentMoxianState);else poll();
 setInterval(poll,1000);render();
})();
