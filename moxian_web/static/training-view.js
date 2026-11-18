// Offline model learning only. Missing measurements remain gaps, never zeros.
(()=>{
 const $=id=>document.getElementById(id),finite=v=>typeof v==='number'&&Number.isFinite(v);
 const names={extrinsics:'EMF外参',dimensions:'尺寸',kinematics:'运动学',tactile:'触觉注册',surface:'低维表面',mixed:'全量混训',reference:'整体手型',geometry:'非对指几何',pinches:'局部对指',all_pinches:'四组联合复测',pinch_index:'拇／食',pinch_middle:'拇／中',pinch_ring:'拇／无名',pinch_pinky:'拇／小'};
 let points=[],runId=null,cursor=0,pending=false,device=null,meta={},error='',liveFit=null;
 const methods={moxian_wuji:'Moxian六动作骨长与EMF外参',wuji_native:'Wuji Studio完整原生标定',bone_extrinsics:'骨长与EMF外参 · NLopt L-BFGS',parameters:'GPU 个体参数（历史）',flashsac:'FlashSAC buffer 版（历史）',delta:'几何＋Delta 联合版（历史）',least_squares:'分阶段最小二乘标定（历史）'};
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
   if(p.raw?.optimizer_name?.startsWith('Wuji native')){$('manoTrainingHover').textContent=`EMF与骨架拟合 · ${p.scope} · 第 ${p.step} 次目标评估 · 目标值 ${fmt(p.raw.loss)}`+(finite(p.raw.geometry_gradient_norm)?` · 参数梯度 ${fmt(p.raw.geometry_gradient_norm)}`:'');return;}
   if(p.solver==='bounded_block_LM'){$('manoTrainingHover').textContent=`第 ${p.step} 次迭代 · ${names[p.phase]||p.phase} · loss ${fmt(p.loss)} · 阻尼 ${fmt(p.damping)} · 步长 ${fmt(p.step_norm)} · ${p.accepted?'接受':'拒绝'} · ${p.linearized_frames} 帧线性化／${p.training_frames} 帧训练`;return;}
   const prior=finite(p.ema?.geometry_prior)?`几何软先验 ${fmt(p.ema.geometry_prior)}`:`β KL ${fmt(p.ema?.beta_kl)} · 几何 KL ${fmt(p.ema?.geometry_kl)}`;
   $('manoTrainingHover').textContent=`第 ${p.step} 步 · ${names[p.phase]||p.phase} / ${names[p.scope]||p.scope} · ${p.training_frames} 个独立训练帧 · 更新后 loss ${fmt(p.ema?.loss)} · ${prior} · EMF 朝向 ${fmt(p.ema?.emf_orientation_error_deg)}° · 指尖方向 ${fmt(p.ema?.tip_emf_angle_deg)}° · Delta RMS／最大 ${fmt(p.ema?.delta_rms_deg)}／${fmt(p.ema?.delta_max_deg)}° · 接触法向偏差 ${fmt(p.ema?.contact_normal_error_deg)}° · 几何梯度 ${fmt(p.raw?.geometry_gradient_norm)} · Delta 梯度 ${fmt(p.raw?.policy_gradient_norm)} · 限位饱和 ${fmt(p.raw?.joint_limit_saturation_fraction)}`;
  };
 }
 function render(){
  const data=selected(points,$('manoTrainingScope').value),log=$('manoTrainingLogScale').checked;
  const activeMethod=meta.method||activeFit()?.method,bones=['bone_extrinsics','wuji_native','moxian_wuji'].includes(activeMethod),lsq=activeMethod==='least_squares';
  for(const id of ['manoErrorChart','manoDirectionChart']){const figure=$(id)?.parentElement;if(figure)figure.hidden=bones;}
  $('manoTrainingLossPart').disabled=lsq||activeMethod==='wuji_native';
  for(const id of ['manoDeltaChart','manoLearningRateChart','manoSaturationChart']){const figure=$(id)?.parentElement;if(figure)figure.hidden=lsq||bones;}
  if(activeMethod==='wuji_native'){for(const id of ['manoKLChart','manoGradientChart']){const figure=$(id)?.parentElement;if(figure)figure.hidden=true;}}
  else{for(const id of ['manoKLChart','manoGradientChart']){const figure=$(id)?.parentElement;if(figure)figure.hidden=false;}}
  if(lsq){
   plot('manoLossChart',data,[{path:['loss'],color:'#287e63'}],log);
   $('manoPriorCaption').textContent='LM 阻尼（不是学习率或先验 KL）';
   plot('manoKLChart',data,[{path:['damping'],color:'#377ca2'}],log);
   plot('manoGradientChart',data,[{path:['gradient_norm'],color:'#287e63'}],log);
   for(const id of ['manoErrorChart','manoDirectionChart'])plot(id,[],[]);
   const p=points.at(-1),fit=activeFit();
   $('manoTrainingStatus').textContent=error?'记录读取失败：'+error:p?`分阶段最小二乘标定 · ${names[p.phase]||p.phase} · 第 ${p.step} 次迭代 · loss ${p.loss.toPrecision(5)} · 阻尼 ${p.damping.toPrecision(4)} · ${p.accepted?'接受':'拒绝'} · ${points.length} 个真实测量点 · 仅保存候选`:fit?.message||'等待最小二乘迭代记录';
   return;
  }
  const part=(activeMethod==='wuji_native'?'loss':$('manoTrainingLossPart').value||'loss').split('.');
  if(bones){
   plot('manoLossChart',data,[{path:['raw',...part],color:'#287e63'}],log);
   $('manoPriorCaption').textContent='参数先验 · 骨长、骨长比例、EMF外参';
   plot('manoKLChart',data,[{path:['raw','terms','bone_prior'],color:'#377ca2'},{path:['raw','terms','bone_ratio_prior'],color:'#8751a1'},{path:['raw','terms','extrinsics_prior'],color:'#be7c35'}],log);
   plot('manoGradientChart',data,[{path:['raw','geometry_gradient_norm'],color:'#287e63'}],log);
   const fit=activeFit(),p=points.at(-1),outcome=fit?.validation_pending?'拟合完成，待独立复测':fit?.accepted?(activeMethod==='wuji_native'?'原生标定成功':'独立验收通过'):fit?.state==='completed'?'计算完成，结果见验收报告':'';
   $('manoTrainingStatus').textContent=error?'记录读取失败：'+error:`EMF与骨架拟合 · Wuji原生 L-BFGS · ${p?'第 '+p.step+' 次目标评估 · 目标值 '+p.raw.loss.toPrecision(5):(fit?.message||'等待点击拟合')} · ${outcome}`;
   return;
  }

  plot('manoLossChart',data,[{path:['raw',...part],color:'#be7c35',dash:true},{path:['ema',...part],color:'#287e63'}],log);
  const native=finite(data.at(-1)?.raw?.geometry_prior)||liveFit?.surface_backend==='qnbot';
  const dimensions=finite(data.at(-1)?.raw?.terms?.dimension_prior);
  if($('manoPriorCaption'))$('manoPriorCaption').textContent=dimensions?'尺寸软先验 · 蓝：长宽厚，紫：宽厚比例':native?'几何软先验 · 蓝：参数先验，紫：mesh 边长':'先验 KL · 紫：MANO β，蓝：局部几何';
  plot('manoKLChart',data,dimensions?[{path:['raw','terms','dimension_prior'],color:'#377ca2'},{path:['raw','terms','section_aspect'],color:'#8751a1'}]:native?[{path:['raw','geometry_prior'],color:'#377ca2'},{path:['raw','terms','mesh_edge'],color:'#8751a1'}]:[{path:['raw','beta_kl'],color:'#8751a1',dash:true},{path:['ema','beta_kl'],color:'#8751a1'},{path:['raw','geometry_kl'],color:'#377ca2',dash:true},{path:['ema','geometry_kl'],color:'#377ca2'}],log);
  plot('manoErrorChart',data,[{path:['ema','contact_gap_mm'],color:'#287e63'},{path:['ema','sensor_rmse_mm'],color:'#be7c35'}]);
  plot('manoDirectionChart',data,[{path:['raw','tip_emf_angle_deg'],color:'#be7c35',dash:true},{path:['ema','tip_emf_angle_deg'],color:'#287e63'},{path:['ema','contact_normal_error_deg'],color:'#8751a1'},{path:['ema','emf_orientation_error_deg'],color:'#377ca2'}]);
  plot('manoDeltaChart',data,[{path:['ema','delta_rms_deg'],color:'#287e63'},{path:['ema','delta_max_deg'],color:'#8751a1'}]);
  plot('manoGradientChart',data,[{path:['raw','geometry_gradient_norm'],color:'#287e63'},{path:['raw','policy_gradient_norm'],color:'#8751a1'}],log);
  plot('manoLearningRateChart',data,[{path:['raw','geometry_learning_rate'],color:'#287e63'},{path:['raw','policy_learning_rate'],color:'#8751a1'}]);
  plot('manoSaturationChart',data,[{path:['raw','joint_limit_saturation_fraction'],color:'#287e63'}]);
  const p=points.at(-1),state={running:'训练中',cancelling:'正在取消',cancelled:'已取消',failed:'失败',completed:'训练完成',preview:'训练及验收完成',idle:'等待训练'}[meta.status]||meta.status||'等待训练';
  const fit=activeFit(),limit=fit?.training_options?.adam_steps,step=p?.step??0,method=methods[meta.method||fit?.method]||meta.method||'';
  const optimizer=p?.raw?.optimizer_name||(fit?.training_options?.optimizer==='muon'?'Muon＋Adam':fit?.training_options?'Adam':'');
  const finished=['preview','completed'].includes(meta.status),outcome=finished&&fit?.accepted===false?' · 验收未通过，未自动应用':'',budget=fit?.solver?.budget_exhausted?' · 时间预算已到，未跑满最大步数':'';
  const continuation=p?.previous_training_steps?` · 从 ${p.previous_training_steps} 步权重续训，累计 ${p.total_training_step} 步`:'';
  $('manoTrainingStatus').textContent=error?`曲线读取失败：${error}，下次轮询重试。`:p?`${state} · ${method}${optimizer?' · 优化器 '+optimizer:''} · ${names[p.phase]||p.phase} · 已更新 ${step}${limit?'/'+limit:''} 步${continuation} · 实际 batch ${p.training_frames} 帧 · ${points.length} 个真实测量点（约每秒一次）${budget}${outcome}`:fit?.run_id?`${method} · 已更新 0${limit?'/'+limit:''} 步 · ${emptyMessage()}。模板／基线验证不属于训练 loss。`:meta.run_id?`${state} · 尚无真实参数更新测量。`:'尚无训练曲线；开始拟合后显示真实数据，旧结果不补造曲线。';
 }
 async function poll(){
  if(pending||document.hidden)return;pending=true;const requested=device,requestedMethod=$('manoTrainingMethod').value;
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
 $('manoTrainingScope').onchange=render;$('manoTrainingLogScale').onchange=render;$('manoTrainingLossPart').onchange=render;
 $('manoTrainingMethod').onchange=()=>{runId=null;cursor=0;points=[];meta={};const method=$('manoTrainingMethod').value;$('manoTrainingExport').href='/api/hand_model/training_trace?after=0'+(method?'&method='+encodeURIComponent(method):'');poll();render();};
 window.addEventListener?.('resize',render);
 if(window.currentMoxianState)window.updateTrainingCharts(window.currentMoxianState);else poll();
 setInterval(poll,1000);render();
})();
