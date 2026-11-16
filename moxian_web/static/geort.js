/* GeoRT controls use the same server/worker path as the CLI. */
(()=>{
 const el=id=>document.getElementById(id),selected=()=>Array.from(el('geortDatasets').selectedOptions,o=>o.value);
 let pending=false,pendingAction=null,lastState={},lastRefresh=0,loading=false,lastCatalogKey='',lastCoverageRefresh=0,coverageLoading=false,coverageGeneration=0;
 el('geortError').textContent='';
 async function api(command,data={}){
  if(pending)throw Error('上一项操作正在处理，请稍候');
  pending=true;pendingAction=command;el('geortError').textContent='';window.updateGeortUI?.(lastState);
  try{
   const response=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command,...data}),signal:AbortSignal.timeout(10000)});
   const result=await response.json();if(!response.ok||result.error)throw Error(result.error||'GeoRT 操作失败');
   if(command==='geort_record_start'||command==='geort_record_stop'){
    lastState={...lastState,geort:{...lastState.geort,recording:result}};
   }
   if(command==='geort_record_start'){
    ++coverageGeneration;lastCoverageRefresh=0;
    lastState.geort.coverage={...lastState.geort.coverage,baseline_ids:[],frames:0,segment:{frames:0},recording:true,notice:null};
    window.resetGeortCoverageRecording?.();
   }
   if(command==='geort_diagnostics_set')lastState={...lastState,geort:{...lastState.geort,diagnostics:result}};
   return result;
  }catch(error){el('geortError').textContent=error.message;window.toast?.(error.message);throw error;}
  finally{pending=false;pendingAction=null;window.updateGeortUI?.(lastState);}
 }
 function bind(id,action){el(id).onclick=()=>Promise.resolve().then(action).catch(error=>{el('geortError').textContent=error.message;});}
 function options(id,items,label){
  const node=el(id),chosen=new Set(Array.from(node.selectedOptions,o=>o.value));
  node.replaceChildren(...items.map(item=>{const option=document.createElement('option');option.value=item.id;option.textContent=label(item);option.selected=chosen.has(item.id);return option;}));
 }
 async function refresh(){
  if(loading)return;loading=true;
  try{
   const responses=await Promise.all(['/api/geort/recordings','/api/geort/models'].map(url=>fetch(url,{cache:'no-store'}).then(r=>r.json())));
   options('geortDatasets',responses[0].items||[],d=>`${d.label||d.id} · ${d.frames}帧 · ${d.source.hand_side} · ${d.complete?'已保存':'保存中'}`);
   options('geortModels',responses[1].items||[],m=>`${m.id}${m.run_id?' / '+m.run_id:''} · last.pth · ${m.status} · ${m.reproduction_profile?.includes('smoke_test')?'测试模型':m.training_mode==='contact_v1'?'GeoRT · 接触约束':'GeoRT · 官方'}`);
   lastRefresh=Date.now();
   window.updateGeortUI?.(lastState);
  }catch(error){el('geortError').textContent=error.message;}finally{loading=false;}
 }
 bind('geortRefresh',refresh);
 async function remove(kind,ids){
  if(!ids.length||ids.some(id=>!id))throw Error('请先选择要删除的录制或模型');
  const active=kind==='models'&&ids.includes(lastState.geort?.selected_model);
  if(!window.confirm(`永久删除${kind==='models'?'模型':'录制'}：\n${ids.join('\n')}\n${active?'当前网络预览将停止。':''}此操作无法撤销。`))return;
  const result=await api(`geort_${kind}_delete`,{ids});
  if(result.unloaded)lastState={...lastState,geort:{...lastState.geort,selected_model:null}};
  await refresh();
 }
 bind('geortDeleteDatasets',()=>remove('recordings',selected()));
 bind('geortDeleteModel',()=>remove('models',[el('geortModels').value]));
 async function coverageSelect(ids){
  const result=await api('geort_coverage_select',{datasets:ids});
  lastState={...lastState,geort:{...lastState.geort,coverage:result}};
  window.updateGeortUI(lastState);lastCoverageRefresh=0;void refreshCoverage();
 }
 bind('geortCoverageSelect',()=>{if(!selected().length)throw Error('请选择覆盖基线的录制数据');return coverageSelect(selected());});
 bind('geortCoverageReset',()=>coverageSelect([]));
 async function refreshCoverage(){
  if(coverageLoading||Date.now()-lastCoverageRefresh<1000)return;
  coverageLoading=true;lastCoverageRefresh=Date.now();
  const generation=coverageGeneration;
  try{
   const response=await fetch('/api/geort/coverage?finger='+encodeURIComponent(el('geortWorkspaceFinger').value||'thumb'),{cache:'no-store',signal:AbortSignal.timeout(10000)}),c=await response.json();
   if(generation!==coverageGeneration)return;
   if(!response.ok||c.error)throw Error(c.error||'覆盖评估读取失败');
   window.updateGeortCoverageCloud?.(c);
  }catch(error){if(generation===coverageGeneration){el('geortCoverageStatus').textContent=error.message;window.updateGeortCoverageCloud?.(null);}}
  finally{coverageLoading=false;if(generation!==coverageGeneration)void refreshCoverage();}
 }
 window.refreshGeortCoverage=()=>{lastCoverageRefresh=0;void refreshCoverage();};
 bind('geortWorkspaceRebuild',()=>api('geort_workspace_rebuild'));
 el('geortDatasets').onchange=el('geortModels').onchange=()=>window.updateGeortUI(lastState);
 bind('geortRecordStart',()=>api('geort_record_start',{label:el('geortLabel').value}));
 bind('geortRecordStop',()=>api('geort_record_stop'));
 bind('geortDiagnosticsToggle',()=>api('geort_diagnostics_set',{enabled:!lastState.geort?.diagnostics?.enabled||lastState.geort?.diagnostics?.status==='error'}));
 bind('geortTrain',()=>api('geort_train',{datasets:selected(),epochs:Number(el('geortEpochs').value),fk_epochs:Number(el('geortFkEpochs').value),mode:el('geortMode').value||'official'}));
 bind('geortCancel',()=>api('geort_train_cancel'));
 bind('geortSelect',async()=>{
  await api('geort_model_select',{model:el('geortModels').value});
  window.wujiRequestedSolver='geort';el('wujiSolver').value='geort';
 });
 bind('geortEvaluate',()=>api('geort_evaluate',{model:el('geortModels').value,datasets:selected()}));
 bind('geortReplayLoad',async()=>{
  const ids=selected();if(ids.length!==1)throw Error(el('geortError').textContent='回放时请选择一段录制');
  await api('geort_replay_load',{dataset:ids[0]});window.wujiRequestedSolver='geort';
 });
 bind('geortReplayPause',()=>api('geort_replay_pause',{paused:!!lastState.wuji2?.playback?.playing}));
 bind('geortReplayStop',()=>api('geort_replay_stop'));
 el('geortReplaySeek').onchange=()=>api('geort_replay_seek',{index:Number(el('geortReplaySeek').value)}).catch(()=>{});
 window.updateGeortUI=state=>{
  state=state||{};lastState=state;const g=state.geort||{},r=g.recording||{},t=g.training||{},play=state.wuji2?.playback;
  const blocked=state.geort?.replay_active?'回放中，请先返回实时输入':!state.connected?'手套未连接':!state.hand_model?.ready?'等待有效手部模型':!state.hand_model?.calibrated?'请先完成并应用六动作标定':'';
  el('geortRecordStart').disabled=!!(pending||r.active||r.saving||blocked);
  el('geortRecordStop').disabled=pending||!r.active;
  el('geortRecordStart').textContent=pendingAction==='geort_record_start'?'正在开始录制…':'开始录制动作';
  el('geortRecordStop').textContent=pendingAction==='geort_record_stop'?'正在停止录制…':'停止并保存';
  el('geortRecordingStatus').textContent=`${pendingAction==='geort_record_start'?'正在开始录制':r.active?'录制中':r.saving?'后台保存中':blocked||'未录制'} · ${r.frames||0} 有效帧 · ${r.invalid_frames||0} 无效帧 · ${(r.seconds||0).toFixed(1)}秒`+(r.error?' · '+r.error:'');
  const running=['starting','running','cancelling'].includes(t.status);
  const diagnostic=g.diagnostics||{};
  el('geortDiagnosticsToggle').textContent=diagnostic.status==='error'?'重新开始表面诊断':diagnostic.enabled?'停止表面诊断':'开始表面诊断';
  el('geortDiagnosticsToggle').disabled=pending;
  el('geortDiagnosticsStatus').textContent=!diagnostic.enabled?'诊断未启动':diagnostic.ready?
   `${diagnostic.input_kind==='replay'?'回放':'实时'}抽样诊断 · 源时间戳 ${diagnostic.timestamp_ms.toFixed(1)} ms · ${diagnostic.device} · 计算 ${diagnostic.processing_ms.toFixed(1)} ms · 结果年龄 ${(diagnostic.age_ms||0).toFixed(0)} ms${diagnostic.model_run_id?' · '+diagnostic.model_run_id:''}`:
   diagnostic.error||'诊断进程初始化中…';
  el('geortDiagnosticsSummary').textContent=diagnostic.ready?`全手非相邻区域采样最大穿透：Glove ${diagnostic.source.max_penetration_mm.toFixed(3)} mm；Wuji2 ${diagnostic.robot?.query_valid?diagnostic.robot.max_penetration_mm.toFixed(3)+' mm':'未诊断'}。${diagnostic.robot_reason||''}`:'';
  const labels={contact:'接触近邻',penetrating:'穿透',separated:'未检出接触',normal_mismatch:'近接·法向不符',unknown:'未知'};
  const names={index:'食指',middle:'中指',ring:'无名指',pinky:'小指'};
  const cell=value=>{const td=document.createElement('td');td.textContent=value?`${labels[value.state]||'未知'} · 间隙 ${value.gap_mm.toFixed(3)} mm · 穿透 ${value.penetration_mm.toFixed(3)} mm · ${value.normal_error_deg.toFixed(1)}°`:'未诊断';td.className='diagnostic-'+(value?.state||'unknown');return td;};
  el('geortDiagnosticsRows').replaceChildren(...(diagnostic.ready?diagnostic.pairs||[]:[]).map(pair=>{const tr=document.createElement('tr'),name=document.createElement('td');name.textContent='拇指—'+names[pair.finger];tr.appendChild(name);tr.appendChild(cell(pair.source));tr.appendChild(cell(pair.robot));return tr;}));
  el('geortDeleteDatasets').disabled=!!(pending||running||r.active||r.saving||g.replay_active||!selected().length);
  el('geortDeleteModel').disabled=!!(pending||running||!el('geortModels').value);
  el('geortCoverageSelect').disabled=!!(pending||r.active||r.saving||!selected().length);
  el('geortCoverageReset').disabled=!!(pending||r.active||r.saving);
  window.renderGeortCoverage?.(g.coverage);void refreshCoverage();
  el('geortTrain').disabled=running||r.active||r.saving;el('geortCancel').disabled=!running;el('geortMode').disabled=running;
  el('geortTrainingStatus').textContent=(t.status?`${t.status} · ${t.phase||'准备'}${t.epoch?` · ${t.epoch}/${t.epochs}轮`:''}${t.elapsed_s?` · ${t.elapsed_s.toFixed(1)}秒`:''}${t.error?' · '+t.error:''}`:'官方默认：100,000 FK 样本 · FK 200轮 · 网络200轮')+` · 网络训练 ${t.device?.gpu_name||'CUDA GPU'} · FK采样／体素预处理 ${t.device?.robot_fk==='cuda:0'?'CUDA GPU':t.device?.preprocessing||'CUDA GPU'}`+(g.model_update?.error?' · 新模型预览停止：'+g.model_update.error:'');
  el('geortTrainingStatus').textContent+=t.training_mode==='contact_v1'?' · 接触约束模式（非官方loss）':'';
  if(t.training_mode==='contact_v1')el('geortTrainingStatus').textContent=el('geortTrainingStatus').textContent.replace('FK采样／体素预处理','FK／接触几何');
  const weights=t.constraint_weights||{penetration:10,surface_contact:2,surface_normal:.2,bone_direction:.5,bending:.05,flat_pose:.5,pinch_position:1,flat_direction:5};
  const lossNote=k=>t.training_mode==='contact_v1'&&k in weights?`（权重 ${weights[k]}${k==='penetration'?'，0.5 mm归一化':''}，参与反向传播）`:'（越低越好）';
  el('geortLosses').textContent=t.losses?Object.entries(t.losses).map(([k,v])=>`${k}: ${v.toExponential(5)}${k==='direction'?'（越接近 −1 越好）':k==='collision'?'（原版未启用；增强穿透另列penetration）':lossNote(k)}`).join('\n'):t.loss!==undefined?`FK MSE: ${t.loss.toExponential(5)}（越接近0越好）`:'';
  el('geortReplayStatus').textContent=play?.active?`动作回放 · ${play.playing?'播放中':'已暂停'} · ${play.index+1}/${play.frames}帧 · ${play.seconds.toFixed(1)}/${play.duration_s.toFixed(1)}秒`:`实时输入 · 模型 ${g.selected_model||'未选择'}`;
  if(play?.active){el('geortReplaySeek').max=Math.max(1,play.frames-1);if(document.activeElement!==el('geortReplaySeek'))el('geortReplaySeek').value=play.index;}
  if(g.evaluation)el('geortEvaluation').textContent=JSON.stringify(g.evaluation,null,2);
  const key=[r.active,r.saving,r.frames,t.status,g.selected_model].join(':');
  if(key!==lastCatalogKey&&Date.now()-lastRefresh>2000){lastCatalogKey=key;void refresh();}
 };
 void refresh();
})();
