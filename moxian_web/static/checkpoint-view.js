// Training artifacts are explicitly previewed; selection never publishes a model.
(()=>{
 const $=id=>document.getElementById(id),select=$('manoCheckpoint');
 const methods=[['moxian_wuji','Moxian六动作骨长与EMF外参'],['wuji_native','Wuji 原生六动作标定'],['bone_extrinsics','Moxian 17动作标定（历史）'],['parameters','xMimic 思路版（历史）'],['flashsac','FlashSAC buffer 版（历史）'],['delta','几何＋Delta 联合训练（历史）'],['least_squares','分阶段最小二乘标定（历史）']];
 // Names remain visible even before live state or artifact requests complete.
 select.replaceChildren(...methods.map(([method,label])=>new Option(label,method)));
 let current=null,items=[],pending=false,loading=false,lastLoad=0,device=null,message='';
 const number=v=>Number.isFinite(v)?v.toFixed(2):'—';
 function render(){
  const h=current?.hand_model||{},fit=h.fit||{},shown=h.checkpoint_preview;
  const count=items.filter(item=>(item.surface_backend||'mano')===(h.surface_backend||'mano')).length;
  const working=['running','cancelling'].includes(fit.state)||h.capture?.state==='collecting';
  const chosen=items.find(item=>item.method===select.value&&(item.surface_backend||'mano')===(h.surface_backend||'mano'));
  select.disabled=pending||working;
  $('manoCheckpointLoad').disabled=pending||working||!current?.connected||!h.ready||!chosen||chosen.compatible===false;
  $('manoCheckpointEnd').disabled=pending||!shown;
  $('manoCheckpointRefresh').disabled=loading||pending;
  $('manoCheckpointSummary').textContent=chosen?`${chosen.accepted?'验收通过':'未通过验收，仅供对比'} · 对指均值 ${number(chosen.contact_gap_mm)} mm · 最大 ${number(chosen.contact_max_gap_mm)} mm · 穿透 ${number(chosen.max_penetration_mm)} mm${chosen.incompatible_reason?' · '+chosen.incompatible_reason:''}`:loading?'正在读取所选训练结果…':'尚未读取到此训练结果，请点击刷新列表。';
  if(chosen?.method==='wuji_native')$('manoCheckpointSummary').textContent=`${chosen.accepted?'Wuji 原生标定成功':'原生结果仅供查看'} · 21自由度 · ${chosen.incompatible_reason||'原生URDF与Moxian显示适配'}`;
  $('manoCheckpointStatus').textContent=pending?'正在切换模型…':message?message:shown?`正在预览：${methods.find(([method])=>method===shown.method)?.[1]||shown.label}。当前标定未替换，返回后恢复原模型。`:!current?.connected?`已读取 ${count} 个训练结果；等待手套连接后预览。`:`当前使用${h.calibrated?'已应用标定':'默认模型'}；可对比 ${count} 个训练结果。`;
 }
 async function refresh(force=false){
  if(loading||(!force&&Date.now()-lastLoad<10000))return;
  const requestedDevice=device;
  loading=true;lastLoad=Date.now();render();
  try{
   const response=await fetch('/api/hand_model/checkpoints',{cache:'no-store',signal:AbortSignal.timeout(10000)});
   const result=await response.json();if(!response.ok)throw Error(result.error||'权重列表读取失败');
   if(requestedDevice!==device)return;
   if(!Array.isArray(result.items))throw Error('权重接口返回格式异常');
   items=result.items.filter(item=>methods.some(([method])=>method===item.method));message=result.error||'';
  }catch(error){if(requestedDevice===device)message='权重读取失败：'+error.message+'；请点击刷新列表重试。';}
  finally{loading=false;render();if(requestedDevice!==device)refresh(true);}
 }
 async function change(action,data={}){
  if(pending)return;pending=true;message='';render();
  try{
   const response=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command:'mano_'+action,...data}),signal:AbortSignal.timeout(15000)});
   const result=await response.json();if(!response.ok)throw Error(result.error||'权重切换失败');
  }catch(error){message=error.message;}
  finally{pending=false;render();}
 }
 select.onchange=render;
 $('manoCheckpointLoad').onclick=()=>{const backend=current?.hand_model?.surface_backend||'mano';const chosen=items.find(item=>item.method===select.value&&(item.surface_backend||'mano')===backend);if(chosen&&chosen.compatible!==false)change('checkpoint_preview',{id:chosen.id});};
 $('manoCheckpointEnd').onclick=()=>change('checkpoint_end');
 $('manoCheckpointRefresh').onclick=()=>refresh(true);
 window.updateCheckpointComparison=state=>{
  current=state;const key=state?.device?.serial+':'+state?.side;
  if(key!==device){device=key;lastLoad=0;items=[];}
  render();refresh();
 };
 if(window.currentMoxianState)window.updateCheckpointComparison(window.currentMoxianState);
 else refresh(true);
 setInterval(()=>refresh(),10000);
})();
