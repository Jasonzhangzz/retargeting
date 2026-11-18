// Only render the matched reference carried by the prediction, never latest ADC.
(()=>{
 const $=id=>document.getElementById(id),pct=v=>Number.isFinite(v)?(v*100).toFixed(1)+'%':'—';
 let current=null,signature=null,layout=null,loading=false,requested=null,retryAt=0;
 let chartsVisible=true;
 const panel=$('contactPrediction');
 if(typeof IntersectionObserver!=='undefined'&&panel){
  chartsVisible=false;
  const observer=new IntersectionObserver(entries=>{const visible=entries.some(e=>e.isIntersecting);if(visible!==chartsVisible){chartsVisible=visible;signature=null;if(visible)update(current);}});
  observer.observe(panel);
 }
 const dot=(a,b)=>a.reduce((v,x,i)=>v+x*b[i],0);
 const unit=a=>{const n=Math.hypot(...a);if(!Number.isFinite(n)||n<1e-8)throw Error('无效掌面方向');return a.map(x=>x/n);};
 function handLayout(t){
  if(!t?.ready||t.rest_vertices_m?.length!==778||t.sid_face_ids?.length!==460||t.sid_barycentric?.length!==460)throw Error('缺少手套曲面位置');
  const up=unit(t.palm_view.forward),normal=unit(t.palm_view.normal);
  const side=unit([up[1]*normal[2]-up[2]*normal[1],up[2]*normal[0]-up[0]*normal[2],up[0]*normal[1]-up[1]*normal[0]]);
  const project=v=>{if(v?.length!==3||!v.every(Number.isFinite))throw Error('无效曲面坐标');return [dot(v,side),-dot(v,up)];};
  const vertices=t.rest_vertices_m.map(project);
  if(!t.faces?.length||t.faces.some(f=>f.length!==3||f.some(i=>!Number.isInteger(i)||i<0||i>=778)))throw Error('无效曲面拓扑');
  const cells=t.sid_face_ids.map((face,sid)=>{
   const f=t.faces[face],b=t.sid_barycentric[sid];
   if(!f||b?.length!==3||!b.every(v=>Number.isFinite(v)&&v>=0)||Math.abs(b.reduce((a,v)=>a+v,0)-1)>1e-5)throw Error('无效 SID 位置');
   return {sid,point:[0,1].map(axis=>f.reduce((v,index,k)=>v+vertices[index][axis]*b[k],0)),label:t.patches?.[t.sid_patch_indices?.[sid]]?.label||'曲面位置'};
  });
  return {version:t.binding_version,side:t.hand_side,vertices,faces:t.faces,cells,
   tips:(t.tip_anchors_m||[]).map(project),min:[0,1].map(k=>Math.min(...vertices.map(v=>v[k]))),max:[0,1].map(k=>Math.max(...vertices.map(v=>v[k])))};
 }
 function ensureLayout(version){
  if(window.isMoxianReplay){
   const t=window.replayingManoTopology;
   if(t?.binding_version===version&&layout?.version!==version){try{layout=handLayout(t);}catch{layout=null;}}
   return;
  }
  if(!version||layout?.version===version||loading||Date.now()<retryAt)return;
  loading=true;requested=version;
  fetch('/api/hand_model/model',{signal:AbortSignal.timeout(5000)}).then(r=>{if(!r.ok)throw Error('手套曲面加载失败');return r.json();}).then(t=>{
   if(t.binding_version!==requested)throw Error('等待对应手型');layout=handLayout(t);retryAt=0;
  }).catch(()=>{retryAt=Date.now()+2000;}).finally(()=>{loading=false;signature=null;update(current);});
 }
 function grid(id,values,kind,contact,predicted,shape){
  const canvas=$(id),rect=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2);
  const w=Math.max(rect.width,200),h=Math.max(rect.height,200);
  if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}
  const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);c.clearRect(0,0,w,h);
  c.font='11px sans-serif';c.textAlign='center';c.fillStyle='#657565';canvas.onmousemove=null;
  if(!shape){c.fillText('等待对应手套曲面',w/2,h/2);canvas.title='等待手套曲面位置';return;}
  const scale=Math.min((w-54)/(shape.max[0]-shape.min[0]),(h-82)/(shape.max[1]-shape.min[1]));
  const xy=v=>[(w-(shape.max[0]-shape.min[0])*scale)/2+(v[0]-shape.min[0])*scale,43+(v[1]-shape.min[1])*scale];
  const vertices=shape.vertices.map(xy),radius=Math.max(1.4,Math.min(3.2,scale*.00125));
  // Union of projected MANO triangles gives the real rest-hand silhouette.
  // Rest geometry keeps the same SID location visible while the hand grasps.
  c.fillStyle='#e6e9df';c.beginPath();
  for(const f of shape.faces){
   const [a,b,d]=f.map(i=>vertices[i]),positive=(b[0]-a[0])*(d[1]-a[1])-(b[1]-a[1])*(d[0]-a[0])>=0;
   c.moveTo(...a);c.lineTo(...(positive?b:d));c.lineTo(...(positive?d:b));c.closePath();
  }
  c.fill('nonzero');
  c.fillStyle='#657565';c.fillText(`${shape.side==='left'?'左':'右'}手 · 掌面展开`,w/2,16);
  const names=['拇指','食指','中指','无名指','小指'];c.font='10px sans-serif';
  shape.tips.forEach((v,i)=>{const [x,y]=xy(v);c.fillText(names[i]||'',x,y-12);});
  c.fillText('腕部',w/2,h-15);
  const cells=shape.cells.map(cell=>({...cell,xy:xy(cell.point)}));
  for(const {sid,xy:point} of cells){
   const value=values?.[sid],truth=contact?.[sid];let color='#d5d4cf';
   if(kind==='diff'){
    if(typeof truth==='boolean'&&Number.isFinite(predicted?.[sid])){const p=predicted[sid]>=.52;color=p?(truth?'#438f70':'#d66a50'):(truth?'#568da9':'#e2e5df');}
   }else if(kind==='binary'){color=typeof value!=='boolean'?'#d5d4cf':value?'#438f70':'#e2e5df';}
   else if(Number.isFinite(value)){
    const t=Math.max(0,Math.min(1,kind==='adc'?value/255:value));
    color=kind==='adc'?`rgb(${Math.round(226-149*t)},${Math.round(231-75*t)},${Math.round(225-40*t)})`:`rgb(${Math.round(230+9*t)},${Math.round(233-142*t)},${Math.round(226-181*t)})`;
   }
   c.fillStyle=color;c.strokeStyle='#7b887370';c.lineWidth=.45;c.beginPath();c.arc(...point,radius,0,Math.PI*2);c.fill();c.stroke();
  }
  canvas.title='掌面展开 · 悬停触觉点查看部位和 SID';
  canvas.onmousemove=event=>{
   const box=canvas.getBoundingClientRect(),x=(event.clientX-box.left)*w/box.width,y=(event.clientY-box.top)*h/box.height;
   let nearest=null,distance=Math.max(7,radius*2)**2;
   for(const cell of cells){const d=(cell.xy[0]-x)**2+(cell.xy[1]-y)**2;if(d<distance){nearest=cell;distance=d;}}
   if(!nearest){canvas.title='掌面展开 · 悬停触觉点查看部位和 SID';return;}
   const {sid,label}=nearest,value=values?.[sid],truth=contact?.[sid],prob=predicted?.[sid];
   canvas.title=`${label} · SID ${sid} · ${kind==='diff'?`预测 ${Number.isFinite(prob)?prob.toFixed(3):'未知'} / 实测 ${typeof truth==='boolean'?Number(truth):'未知'}`:kind==='binary'?`实测 ${typeof value==='boolean'?Number(value):'未知'}`:kind==='adc'?`实测 ${value??'未知'} ADC`:`预测概率 ${Number.isFinite(value)?value.toFixed(3):'未知'}`}`;
  };
 }
 function update(state){
  current=state;const p=state?.contact_prediction||{},comp=p.comparison||{},ready=p.ready===true;
  if(document.hidden||!chartsVisible)return;
  const version=p.binding_version||state?.hand_model?.binding_version;ensureLayout(version);
  const shape=layout?.version===version&&layout?.side===(p.hand_side||state?.hand_model?.hand_side)?layout:null;
  $('predictionStatus').textContent=ready?(window.isMoxianReplay?'录制预测回放':'网络在线'):p.status==='warming_up'?`收集历史 ${p.history_frames||0}/${p.history_required||8}`:p.status==='starting'?'加载权重':p.status==='error'?'预测异常':'等待 MANO';
  $('predictionMessage').textContent=p.error||(ready?(comp.ready?'网络预测与同帧实测接触已对齐。':'网络已输出；实测接触未知或时间不匹配，请先去皮。'):p.status==='warming_up'?'请连续活动手指，集齐 8 帧后开始时序预测。':'连接手套并等待有效 MANO 姿态。');
  $('predictionF1').textContent=ready?pct(comp.illustrative?.f1):'—';
  $('predictionCumulative').textContent=ready?pct(comp.cumulative?.illustrative?.f1):'—';
  $('predictionVerified').textContent=ready?pct(comp.verified?.f1):'—';
  $('predictionFrame').textContent=ready?pct(p.frame_probability):'—';
  $('predictionInput').textContent=`仅 MANO 位姿 · ${p.history_frames||0}/${p.history_required||8} 帧 · 腕部运动未观测`;
  $('predictionMapping').textContent=`三图为同一手套掌面展开，悬停查看部位和 SID。${comp.verified_count||0}/460 个 SID 位置通过实测核验。当前／累计示意 F1 使用现有曲面映射；已核验 F1 仅统计通过位置核验的 SID。`;
  const m=comp.illustrative;
  $('predictionCounts').textContent=ready&&m?`命中 ${m.tp} · 误报 ${m.fp} · 漏报 ${m.fn} · 真阴性 ${m.tn} · 参考帧 ${comp.scored_frames}`:'实测接触未就绪，不计入评分';
  const lag=Number.isFinite(state?.hand_model?.timestamp_ms)?Math.max(0,state.hand_model.timestamp_ms-p.timestamp_ms):null;
  $('predictionTiming').textContent=ready?`CPU ${p.processing_ms.toFixed(1)} ms · MANO 延迟 ${lag==null?'—':lag.toFixed(0)} ms · 参考时间差 ${comp.timestamp_delta_ms==null?'—':comp.timestamp_delta_ms.toFixed(1)} ms`:'CPU 推理 · 接触阈值 0.52';
  const adc=$('predictionShowADC').checked;
  $('predictionTruthTitle').textContent=adc?'实测原始 ADC · 0–255':'实测接触状态 · 去皮后检测';
  const key=[p.timestamp_ms,ready,adc,shape?.version,Math.round($('predictionGrid').getBoundingClientRect().width)].join(':');
  if(key===signature)return;signature=key;
  grid('predictionGrid',ready?p.taxel_probability:null,'probability',null,null,shape);
  grid('predictionTruthGrid',ready?(adc?comp.actual_raw_adc:comp.actual_contact):null,adc?'adc':'binary',null,null,shape);
  grid('predictionDiffGrid',null,'diff',ready?comp.actual_contact:null,ready?p.taxel_probability:null,shape);
 }
 async function command(name){
  if(window.isMoxianReplay)return;
  try{const r=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command:name})});const value=await r.json();if(!r.ok)throw Error(value.error);}
  catch(error){$('predictionMessage').textContent=error.message;}
 }
 $('predictionReset').onclick=()=>command('contact_prediction_reset');
 $('predictionRestart').onclick=()=>command('contact_prediction_restart');
 $('predictionShowADC').onchange=()=>{signature=null;update(current);};
 $('predictionShowMano').onclick=()=>{const select=$('poseMode');select.value='mano';select.dispatchEvent(new Event('change'));};
 $('predictionOverlay').onchange=()=>window.updateManoView?.(current,$('poseMode').value==='mano');
 window.updateContactPrediction=update;window.addEventListener('resize',()=>{signature=null;update(current);});
 update(window.currentMoxianState);
})();
