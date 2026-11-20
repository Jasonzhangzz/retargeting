/* Same-frame mesh/MLP collision audit; never changes q or solver selection. */
(()=>{
 const el=id=>document.getElementById(id);
 let latest={},pending=false,example=null,examplePending=false;
 const mm=value=>Number.isFinite(value)?value.toFixed(3):'—';
 const names={palm:'掌部',thumb:'拇指',index:'食指',middle:'中指',ring:'无名指',pinky:'小指'};
 const link=name=>name==='palm'?names.palm:name.replace(/^(thumb|index|middle|ring|pinky)_link(\d)$/,(_,finger,n)=>`${names[finger]} L${n}`);
 function row(values,className=''){
  const tr=document.createElement('tr');tr.className=className;
  values.forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.appendChild(td);});return tr;
 }
 function method(name,result){
  const known=result?.query_valid===true&&typeof result.collision==='boolean';
  return row([name,known?(result.collision?'碰撞':'未检出碰撞'):'不可用',
   known?`${mm(result.max_penetration_mm)} mm`:'—',known?String(result.collision_pairs):'—',
   result?`${mm(result.processing_ms)} ms`:'—'],known?(result.collision?'diagnostic-penetrating':'diagnostic-contact'):'diagnostic-unknown');
 }
 el('wujiCollisionToggle').onclick=async()=>{
  if(pending)return;pending=true;window.updateWujiCollisionUI({wuji_collision:latest});
  try{
   const response=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({command:'wuji2_collision_set',enabled:!latest.enabled||latest.status==='error'}),signal:AbortSignal.timeout(10000)});
   const result=await response.json();if(!response.ok||result.error)throw Error(result.error||'碰撞对照操作失败');
   window.updateWujiCollisionUI({wuji_collision:result});
  }catch(error){el('wujiCollisionStatus').textContent=error.message;}
  finally{pending=false;el('wujiCollisionToggle').disabled=false;}
 };
 async function loadExample(index){
  const response=await fetch('/api/wuji_collision/example?index='+encodeURIComponent(index),{cache:'no-store',signal:AbortSignal.timeout(10000)}),d=await response.json();
  if(!response.ok||!d.ready)throw Error(d.error||'测试姿态不可用');
  example=d;el('wujiCollisionExampleSelect').hidden=false;el('wujiCollisionLive').hidden=false;
  render();
 }
 el('wujiCollisionExample').onclick=async()=>{
  if(examplePending)return;examplePending=true;el('wujiCollisionExample').disabled=true;
  try{
   const response=await fetch('/api/wuji_collision/examples',{cache:'no-store',signal:AbortSignal.timeout(10000)}),catalog=await response.json();
   if(!response.ok||!catalog.ready||!catalog.items?.length)throw Error(catalog.error||'暂无测试姿态');
   const select=el('wujiCollisionExampleSelect');select.replaceChildren(...catalog.items.map(item=>{const option=document.createElement('option');option.value=String(item.index);option.textContent=item.label;return option;}));
   const item=catalog.items.find(item=>item.disagreement_pairs>0)||catalog.items[Math.min(1,catalog.items.length-1)];
   select.value=String(item.index);await loadExample(select.value);
  }catch(error){el('wujiCollisionStatus').textContent=error.message;}
  finally{examplePending=false;el('wujiCollisionExample').disabled=false;}
 };
 el('wujiCollisionExampleSelect').onchange=()=>loadExample(el('wujiCollisionExampleSelect').value).catch(error=>{el('wujiCollisionStatus').textContent=error.message;});
 el('wujiCollisionLive').onclick=()=>{example=null;el('wujiCollisionExampleSelect').hidden=true;el('wujiCollisionLive').hidden=true;render();};
 function render(){
  const d=example||latest;
  el('wujiCollisionToggle').disabled=pending;
  el('wujiCollisionToggle').textContent=latest.status==='error'?'重新开始碰撞对照':latest.enabled?'停止碰撞对照':'开始碰撞对照';
  const ready=d.ready===true;
  el('wujiCollisionStatus').textContent=example?`独立测试姿态 #${d.example.pose_id} · 非实时输入 · ${d.device} · 已保存的真实网格与模型预测`:
   !d.enabled?'碰撞对照已停止':ready?
   `${d.input_kind==='replay'?'回放':'实时'} · ${d.solver_backend||'Wuji2'} · 同帧源时间戳 ${mm(d.timestamp_ms)} ms · 结果年龄 ${mm(d.age_ms)} ms · ${d.device} · 总计算 ${mm(d.processing_ms)} ms`:
   d.error||(d.status==='starting'?'正在加载碰撞模型…':'等待当前 Wuji2 姿态');
  el('wujiCollisionMethods').replaceChildren(...(ready?[
   method('真实网格 · 密集',d.dense_mesh),method('真实网格 · 同点',d.matched_mesh),method('Neural SDF · 同点',d.neural)]:[]));
  const c=d.comparison;
  el('wujiCollisionDifference').textContent=ready?(c?
   `密集网格／MLP：${c.dense_agrees?'一致':'不一致'}；同点网格／MLP：${c.matched_agrees?'一致':'不一致'}。最大穿透差（密集／同点）${mm(c.dense_depth_error_mm)} / ${mm(c.matched_depth_error_mm)} mm；同点距离 MAE ${mm(c.point_mae_mm)} mm。累计 ${c.frames} 帧，不一致 ${c.dense_disagreement_frames} / ${c.matched_disagreement_frames} 帧。`:
   `MLP 不可用：${d.model_error||'等待模型'}；真实网格结果仍显示。`):'';
  el('wujiCollisionDifference').className=c&&!c.dense_agrees?'diagnostic-penetrating':'';
  el('wujiCollisionDomain').textContent=ready?
   `${d.query_points} 个同点查询 · 每 link ${d.samples_per_link} 个曲面样本 · 模型 ${d.model?.name||'不可用'}${d.model?.epoch?' · '+d.model.epoch+'轮':''}。${d.in_training_domain?'当前关节角在训练域内。':'当前姿态超出模型训练域，MLP 为域外预测：'+(d.out_of_domain_joints||[]).join('、')}`:'';
  const pairSelect=el('wujiCollisionPair'),chosen=pairSelect.value;
  const all=document.createElement('option');all.value='';all.textContent='全手';
  const pairOptions=ready?(d.pairs||[]).map(p=>{const option=document.createElement('option');option.value=[p.a,p.b].join(':');option.textContent=`${link(p.link_a)} ↔ ${link(p.link_b)}`;return option;}):[];
  pairSelect.replaceChildren(all,...pairOptions);pairSelect.value=pairOptions.some(o=>o.value===chosen)?chosen:'';
  const rows=ready?(d.pairs||[]).map(p=>{const tr=row([`${link(p.link_a)} ↔ ${link(p.link_b)}`,mm(p.dense_mm),mm(p.matched_mm),mm(p.neural_mm),
   p.agrees===null?'MLP 不可用':p.agrees?'一致':'不一致'],p.agrees===false?'diagnostic-penetrating':'');tr.dataset.pair=[p.a,p.b].join(':');tr.onclick=()=>window.focusWujiCollisionPair?.(p.a,p.b);return tr;}):[];
  if(ready&&!rows.length)rows.push(row(['未检出深度超过 0.1 mm 的 link 对','—','—','—','—']));
  el('wujiCollisionPairs').replaceChildren(...rows);
  window.currentWujiCollisionVisual=d;
  window.updateWujiCollisionView?.(d);
 }
 window.updateWujiCollisionUI=state=>{
  latest=state?.wuji_collision||{};render();
 };
 if(new URLSearchParams(window.location?.search||'').get('collision')==='test')void el('wujiCollisionExample').onclick();
})();
