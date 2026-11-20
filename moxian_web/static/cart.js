/* Arbitrary-motion capture, differentiable training jobs and pure-network inference. */
(()=>{
 const el=id=>document.getElementById(id);
 let state={},pending=null,refreshing=false,lastRefresh=0,recordings=[];
 window.cartHullVisible=el('cartShowHull').checked;el('cartError').textContent='';
 const selected=()=>Array.from(el('cartDatasets').selectedOptions).map(o=>o.value);
 function render(){
  const c=state.cart||{},r=c.recording||{},w=state.wuji2||{};
  const compatible=c.protocol==='cart_diff_ik_v6';
  const live=!!(compatible&&state.connected&&state.hand_model?.ready&&state.hand_model?.calibrated&&!c.replay_active&&!state.geort?.replay_active);
  if(!pending)el('cartSplit').value=c.split||'train';
  el('cartSplit').disabled=!!(pending||r.active||r.saving);
  el('cartRecordStart').disabled=!!(pending||!live||r.active||r.saving||state.geort?.recording?.active);
  el('cartRecordStop').disabled=!!(pending||!r.active);
  el('cartRecordStatus').textContent=`${r.active?'录制中':r.saving?'保存中':'未录制'} · ${r.frames||0}帧 · 无效 ${r.invalid_frames||0}帧`+(r.error?' · '+r.error:'');
  const geometry=c.capture_geometry;
  el('cartGeometryStatus').textContent=geometry?`Wrist 三维数据 · ${geometry.status==='completed'?'已保存':geometry.status==='failed'?'失败（原始帧已保留，可准备观测重试）':geometry.status==='cancelled'?'等待恢复':'保存中'} · ${geometry.frames||0}${geometry.raw_frames!=null?'/'+geometry.raw_frames:''} 帧${geometry.error?' · '+geometry.error:''}`:'';
  const h=compatible&&c.ready?c.hull:null;
  el('cartHullStatus').textContent=c.protocol&&!compatible?'CART 服务版本不匹配，请重启网页服务以加载同步触觉接触检测。':h?`自接触 ${h.self_contacts||0} · 物体接触 ${h.object_contacts||0} · 触觉 ${h.tactile_status==='valid'?'有效':'未知／过期'} · 预览延迟 ${c.preview_age_ms?.toFixed(0)||'—'} ms`:(c.error||'点击预览接触或开始录制');
  window.updateCartPreview?.(state);
  const job=c.job||{},running=['starting','running','cancelling'].includes(job.status);
  const stage={collect:'并行 rollout 与查询收集',geometry:'GPU 标签与碰撞网络更新',backward:'策略展开、合批损失与反传'}[job.compute_stage]||job.compute_stage;
  el('cartJob').textContent=job.error||`${job.action||'后台任务'} · ${job.status||'未运行'} · ${job.phase||''}${job.device?` · ${job.device}`:''} ${job.epoch?`${job.epoch}/${job.epochs}轮`:job.completed!=null?`${job.completed}/${job.total}帧`:job.frames!=null?`${job.frames}/${job.raw_frames??'…'}帧`:''}${job.batch?` · 更新 ${job.batch}/${job.batches??'…'}`:''}${job.phase==='mesh_validation'?` · 真实 mesh 验证 ${job.completed}/${job.total} 个窗口`:''}${job.parallel_batch_size?` · 实际并行 ${job.parallel_batch_size} 个窗口 · 每 batch 更新`:''}${job.gradient_accumulation===false?' · 无梯度累积':''}${job.compute_stage?` · ${stage}`:''}${job.transitions_per_s?` · ${job.transitions_per_s.toFixed(1)} 损失帧/秒`:''}`;
  el('cartDeleteDatasets').disabled=!!(pending||running||r.active||r.saving||c.replay_active||!selected().length);
  el('cartDeleteModel').disabled=!!(pending||running||!el('cartModels').value);
  for(const id of ['cartPreprocess','cartTrain','cartEvaluate'])el(id).disabled=!!(pending||running||r.active||r.saving||!selected().length);
  const chosen=recordings.filter(item=>selected().includes(item.id));
  const trainCount=chosen.filter(item=>item.split==='train').length,validationCount=chosen.filter(item=>item.split==='validation').length;
  el('cartTrainingHint').textContent=!chosen.length?'请选择训练会话和独立验证会话；也可先单独准备一段录制。':!trainCount||!validationCount?`已选训练 ${trainCount} 段、验证 ${validationCount} 段。${!validationCount?'请将采集分区切换为“验证”，另录一段会话，然后同时选中训练和验证数据。':'请同时选中至少一段训练会话。'}`:'已选择训练和独立验证会话；训练会自动准备缺失或过期的观测与 IK。';
  if(!trainCount||!validationCount)el('cartTrain').disabled=true;
  el('cartLosses').textContent=job.losses||job.sdf_losses?JSON.stringify({losses:job.losses,geometry_losses:job.sdf_losses,validation:job.validation_loss,geometry_validation:job.sdf_validation,metrics:job.metrics,timings:job.timings},null,2):job.result?JSON.stringify(job.result,null,2):'';
  const m=w.cart?.metrics;
  el('cartBufferStatus').textContent=w.cart?.buffering?`等待未来观测 ${w.cart.buffered_frames}/11 帧`:w.cart?.buffer_delay_ms!==undefined?`观测时间 ${w.cart.observation_timestamp_ms.toFixed(1)} ms · 实际缓冲 ${w.cart.buffer_delay_ms.toFixed(1)} ms`:'';
  el('cartContactStatus').textContent=m?`网络独立诊断 · 间隙 ${m.max_gap_mm.toFixed(2)} mm · 穿透 ${m.max_penetration_mm.toFixed(2)} mm · 骨轴误差 ${m.bone_direction_deg.toFixed(1)}° · ${w.contact_constraints_satisfied===null?'无活动接触':w.contact_constraints_satisfied?'接触达标':'接触未达标'}`:'';
  if(w.playback?.active&&c.replay_active){el('cartSeek').max=Math.max(0,w.playback.frames-1);el('cartSeek').value=w.playback.index;}
 }
 async function action(command,data={}){
  if(pending)return;pending=command;el('cartError').textContent='';render();
  try{
   const response=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command,...data}),signal:AbortSignal.timeout(15000)});
   const result=await response.json();if(!response.ok||result.error)throw Error(result.error||'CART 操作失败');
   if(command==='cart_record_start'||command==='cart_record_stop')state={...state,cart:{...state.cart,recording:result}};
   void refresh();return result;
  }catch(error){el('cartError').textContent=error.message;window.toast?.(error.message);}
  finally{pending=null;render();}
 }
 function options(select,items,label){
  const keep=new Set(Array.from(select.selectedOptions).map(o=>o.value));select.replaceChildren();
  for(const item of items){const o=document.createElement('option');o.value=item.id;o.textContent=label(item);o.selected=keep.has(item.id);select.appendChild(o);}
 }
 async function refresh(){
  if(refreshing)return;refreshing=true;
  try{
   const responses=await Promise.all(['/api/cart/recordings','/api/cart/models'].map(url=>fetch(url,{cache:'no-store'})));
   const [records,models]=await Promise.all(responses.map(async r=>{const d=await r.json();if(!r.ok)throw Error(d.error||'读取失败');return d.items||[];}));
   recordings=records;
   options(el('cartDatasets'),records,r=>`${r.split==='validation'?'验证':'训练'} · ${r.label||r.id} · ${r.frames}帧 ${r.complete?'已保存':'保存中'}${r.ik?` · IK 已缓存 ${r.ik.valid_frames}/${r.frames}帧`:''}${r.preprocessing?.protocol==='cart_diff_ik_v6'?' · IK／触点观测已准备':' · 需准备观测与 IK'}`);
   options(el('cartModels'),models,m=>`${m.id} · ${m.status}`);
   el('cartRecordings').replaceChildren(...records.map(r=>{const li=document.createElement('li'),a=document.createElement('a');a.href='/api/cart/recording?id='+encodeURIComponent(r.id);a.textContent=`${r.label||r.id} · 接触数据`;li.appendChild(a);if(r.capture_geometry?.status==='completed'){const geometry=document.createElement('a');geometry.href='/api/cart/geometry?id='+encodeURIComponent(r.id);geometry.textContent=' · Wrist 三维点完整记录';li.appendChild(geometry);}if(r.ik){const ik=document.createElement('a');ik.href='/api/cart/ik?id='+encodeURIComponent(r.id);ik.textContent=' · IK 轨迹缓存 NPZ';li.appendChild(ik);}return li;}));lastRefresh=Date.now();render();
  }catch(error){el('cartError').textContent=error.message;}finally{refreshing=false;}
 }
 async function remove(kind,ids){
  if(!ids.length||ids.some(id=>!id))return;
  if(!window.confirm(`永久删除 CART ${kind==='models'?'模型':'录制'}：\n${ids.join('\n')}\n此操作无法撤销。`))return;
  const result=await action(`cart_${kind}_delete`,{ids});
  if(result?.unloaded)state={...state,cart:{...state.cart,selected_model:null}};
  await refresh();render();
 }
 el('cartDeleteDatasets').onclick=()=>remove('recordings',selected());
 el('cartDeleteModel').onclick=()=>remove('models',[el('cartModels').value]);
 el('cartDatasets').onchange=el('cartModels').onchange=render;
 el('cartSplit').onchange=()=>action('cart_configure',{split:el('cartSplit').value});
 el('cartPreview').onclick=()=>action('cart_preview',{split:el('cartSplit').value});
 el('cartRecordStart').onclick=()=>action('cart_record_start',{split:el('cartSplit').value,label:el('cartLabel').value});
 el('cartRecordStop').onclick=()=>action('cart_record_stop');el('cartRefresh').onclick=refresh;
 el('cartPreprocess').onclick=()=>action('cart_preprocess',{datasets:selected()});
 el('cartTrain').onclick=()=>action('cart_train',{datasets:selected(),epochs:Number(el('cartEpochs').value),architecture:el('cartArchitecture').value||'attention',batch_size:Number(el('cartBatchSize').value)||0,initial_noise_rad:Number(el('cartInitialNoise').value)});
 el('cartCancel').onclick=()=>action('cart_cancel');
 el('cartSelect').onclick=async()=>{const result=await action('cart_model_select',{model:el('cartModels').value});if(result){window.wujiRequestedSolver='cart';el('wujiSolver').value='cart';}};
 el('cartEvaluate').onclick=()=>action('cart_evaluate',{model:el('cartModels').value,datasets:selected()});
 el('cartReplayLoad').onclick=()=>action('cart_replay_load',{dataset:selected()[0]});
 el('cartReplayPause').onclick=()=>action('cart_replay_pause',{paused:!!state.wuji2?.playback?.playing});
 el('cartReplayStop').onclick=()=>action('cart_replay_stop');el('cartSeek').onchange=()=>action('cart_replay_seek',{index:Number(el('cartSeek').value)});
 el('cartShowHull').onchange=()=>{window.cartHullVisible=el('cartShowHull').checked;window.updateManoView?.(state,el('poseMode').value==='mano');window.updateWujiView?.(state,el('poseMode').value==='wuji2');window.updateCartPreview?.(state);};
 window.updateCartUI=next=>{state=next||{};render();if(Date.now()-lastRefresh>5000)void refresh();};
 void refresh();
})();
