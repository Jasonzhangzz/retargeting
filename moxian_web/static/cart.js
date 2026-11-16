/* CART contact protocol: explicit action capture, independent network jobs. */
(()=>{
 const el=id=>document.getElementById(id);
 let state={},pending=null,refreshing=false,lastRefresh=0,stageKey='';
 window.cartHullVisible=el('cartShowHull').checked;el('cartError').textContent='';
 const selected=()=>Array.from(el('cartDatasets').selectedOptions).map(o=>o.value);
 function render(){
  const c=state.cart||{},r=c.recording||{},w=state.wuji2||{};
  const live=!!(state.connected&&state.hand_model?.ready&&state.hand_model?.calibrated&&!c.replay_active&&!state.geort?.replay_active);
  const stages=c.stages||[];
  if(stages.length&&stageKey!==JSON.stringify(stages)){
   stageKey=JSON.stringify(stages);el('cartStage').replaceChildren();
   for(const s of stages){const o=document.createElement('option');o.value=s.id;o.textContent=`${s.category}类 · ${s.label}`;el('cartStage').appendChild(o);const a=o.cloneNode(true);el('cartAction').appendChild(a);}
  }
  if(!pending){el('cartStage').value=c.stage||el('cartStage').value;el('cartSplit').value=c.split||'train';el('cartAction').value=c.manual||'auto';}
  const stage=stages.find(s=>s.id===c.stage);
  el('cartPrompt').textContent=`${stage?`${stage.index+1}/${stages.length} · ${stage.label}`:'等待协议'} · 完整周期 ${c.motion?.cycles||0}/2 · ${c.motion?.phase||'等待'}。闭合后再张开，不限速度；未完成可重录。`+(stage?.category===1?' 当前动作使用真实手部表面接触目标。':'');
  el('cartRecordStart').disabled=!!(pending||!live||r.active||r.saving||state.geort?.recording?.active);
  el('cartRecordStop').disabled=!!(pending||!r.active);
  for(const id of ['cartStage','cartSplit','cartNext'])el(id).disabled=!!(pending||r.active||r.saving);
  el('cartNext').disabled ||= (c.motion?.cycles||0)<2;
  el('cartRecordStart').textContent=pending==='cart_record_start'?'正在开始录制…':'录制当前子动作／重录';
  el('cartRecordStatus').textContent=`${r.active?'CART 录制中':r.saving?'后台保存中':'未录制'} · ${r.frames||0}帧 · 无效 ${r.invalid_frames||0}帧`+(r.seconds?` · ${r.seconds.toFixed(1)}秒`:'')+(r.error?' · '+r.error:'')+(!live?' · 请连接并应用六动作标定':'');
  const h=c.ready&&c.hull?c.hull:w.solver_backend==='cart'?w.cart_hull:null;
  el('cartVolume').textContent=h?.ready?`${h.volume_ml.toFixed(3)} mL`:'—';
  el('cartArea').textContent=h?.ready?`${h.area_cm2.toFixed(2)} cm²`:'—';
  const primitive=h?.primitive||h?.diagnostic_candidate?.primitive;
  const initialInfo=primitive?` · type ${primitive.reference_type} · ${primitive.kind} + delta · 半径 ${(primitive.radius_m*1000).toFixed(1)} mm${primitive.kind==='cylinder'?` · 长度 ${(primitive.height_m*1000).toFixed(1)} mm`:''} · 几何置信度 ${(primitive.confidence*100).toFixed(0)}%`:'';
  const tried=h?.initial_shapes_tried?.length?` · 已尝试 ${h.initial_shapes_tried.join(' / ')}`:'';
  const low=h?.fit_attempts?.length&&h.fit_attempts.every(a=>a.status==='low_confidence');
  const fingerNames={thumb:'拇指',index:'食指',middle:'中指',ring:'无名指',pinky:'小指'};
  const missing=h?.diagnostic_candidate?.contacts?.filter(a=>a.gap_m>.002||a.normal_error_deg>30).map(a=>fingerNames[a.patch.split('_')[0]]).join('、');
  const details=missing?` · 尚未达标：${missing}`:low?' · 请让拇指参与对向夹持或包围':'';
  el('cartHullStatus').textContent=h?`${h.contact_mode==='hand'?'真实手部表面接触目标':h.ready?'primitive 接触目标有效':h.diagnostic_candidate?'primitive 诊断候选（无效监督）':h.status==='low_confidence'||low?'几何置信度低，放弃拟合':'无有效 primitive'}${initialInfo}${tried}${details}${h.budget_exhausted?' · 搜索预算已用尽'+(h.ready?'，当前候选验收通过':''):''}${Number.isFinite(h.reference_gap_m)?` · 最大目标间距 ${(h.reference_gap_m*1000).toFixed(1)} mm`:''} · ${h.processing_ms?.toFixed(1)||'—'} ms · 估计延迟 ${c.preview_age_ms?.toFixed(0)||'—'} ms`:(c.error||'点击预览或录制，估计当前动作的 primitive');
  window.updateCartPreview?.(state);
  const job=c.job||{};el('cartJob').textContent=job.error||`${job.action||'后台任务'} · ${job.status||'未运行'} · ${job.phase||''} ${job.epoch?`${job.epoch}/${job.epochs}轮`:job.completed?`${job.completed}/${job.total}帧`:''}`;
  const running=['starting','running','cancelling'].includes(job.status);
  el('cartDeleteDatasets').disabled=!!(pending||running||r.active||r.saving||c.replay_active||!selected().length);
  el('cartDeleteModel').disabled=!!(pending||running||!el('cartModels').value);
  el('cartLosses').textContent=job.losses?JSON.stringify(job.losses,null,2):job.result?JSON.stringify(job.result,null,2):'';
  const m=w.cart?.metrics;el('cartContactStatus').textContent=m?`${w.cart.stage} · ${w.cart.phase} · ${w.contact_constraints_satisfied?'全部必需接触达标':'接触未全部达标'} · 间距 [${m.gaps_mm.map(v=>v.toFixed(2)).join(', ')}] mm · 最大穿透 ${w.cart.max_penetration_mm.toFixed(3)} mm · 识别 ${(w.cart.action_confidence*100).toFixed(0)}%`:'';
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
   const [records,models]=await Promise.all(responses.map(async r=>{const d=await r.json();if(!r.ok)throw Error(d.error||'读取 CART 数据失败');return d.items||[];}));
   options(el('cartDatasets'),records,r=>`${r.split==='validation'?'验证':'训练'} · ${r.stage} · ${r.label} · ${r.frames}帧 ${r.complete?'已保存':'保存中'} · ${r.preprocessing?(r.preprocessing.complete_cycles&&r.preprocessing.contact_targets_valid?'接触数据可用':'需补录／核对接触'):'未预处理'}`);
   options(el('cartModels'),models,m=>`${m.id} · ${m.status}`);
   el('cartRecordings').replaceChildren(...records.map(r=>{const li=document.createElement('li');const a=document.createElement('a');a.href='/api/cart/recording?id='+encodeURIComponent(r.id);a.textContent=`${r.label} · 下载动作／预处理接触目标`;li.appendChild(a);return li;}));lastRefresh=Date.now();
  }catch(error){el('cartError').textContent=error.message;}finally{refreshing=false;}
 }
 const config=()=>({stage:el('cartStage').value,split:el('cartSplit').value});
 async function remove(kind,ids){
  if(!ids.length||ids.some(id=>!id)){el('cartError').textContent='请选择要删除的录制或模型';return;}
  const active=kind==='models'&&ids.includes(state.cart?.selected_model);
  if(!window.confirm(`永久删除 CART ${kind==='models'?'模型':'录制'}：\n${ids.join('\n')}\n${active?'当前网络预览将停止。':''}此操作无法撤销。`))return;
  const result=await action(`cart_${kind}_delete`,{ids});
  if(result?.unloaded)state={...state,cart:{...state.cart,selected_model:null}};
  await refresh();render();
 }
 el('cartDeleteDatasets').onclick=()=>remove('recordings',selected());
 el('cartDeleteModel').onclick=()=>remove('models',[el('cartModels').value]);
 el('cartDatasets').onchange=el('cartModels').onchange=render;
 el('cartStage').onchange=()=>action('cart_stage_set',config());el('cartSplit').onchange=()=>action('cart_stage_set',config());
 el('cartPreview').onclick=()=>action('cart_stage_set',config());el('cartNext').onclick=()=>action('cart_stage_next');
 el('cartRecordStart').onclick=()=>action('cart_record_start',{...config(),label:el('cartLabel').value});
 el('cartRecordStop').onclick=()=>action('cart_record_stop');el('cartRefresh').onclick=refresh;
 el('cartPreprocess').onclick=()=>action('cart_preprocess',{datasets:selected()});
 el('cartTrain').onclick=()=>action('cart_train',{datasets:selected(),epochs:Number(el('cartEpochs').value)});
 el('cartCancel').onclick=()=>action('cart_cancel');el('cartSelect').onclick=async()=>{const result=await action('cart_model_select',{model:el('cartModels').value});if(result){window.wujiRequestedSolver='cart';el('wujiSolver').value='cart';}};
 el('cartEvaluate').onclick=()=>action('cart_evaluate',{model:el('cartModels').value,datasets:selected()});
 el('cartAction').onchange=()=>action('cart_action_set',{stage:el('cartAction').value});
 el('cartReplayLoad').onclick=()=>action('cart_replay_load',{dataset:selected()[0]});
 el('cartReplayPause').onclick=()=>action('cart_replay_pause',{paused:!!state.wuji2?.playback?.playing});
 el('cartReplayStop').onclick=()=>action('cart_replay_stop');el('cartSeek').onchange=()=>action('cart_replay_seek',{index:Number(el('cartSeek').value)});
 el('cartShowHull').onchange=()=>{window.cartHullVisible=el('cartShowHull').checked;window.updateManoView?.(state,el('poseMode').value==='mano');window.updateWujiView?.(state,el('poseMode').value==='wuji2');};
 window.updateCartUI=next=>{state=next||{};render();if(Date.now()-lastRefresh>5000)void refresh();};
 void refresh();
})();
