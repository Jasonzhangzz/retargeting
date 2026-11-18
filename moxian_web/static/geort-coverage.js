/* Independent source and robot clouds; only exact FK reference points have poses. */
(()=>{
 const el=id=>document.getElementById(id);
 let signature='',spatial={},yaw=.6,pitch=.35,drag=null,hits=[],selected=null,request=0;
 const chosen=()=>el('geortWorkspaceFinger').value||'thumb';
 const referenceFinger=()=>spatial.workspace?.fingers?.find(f=>f.finger===chosen());
 function selectionStatus(pose,count){
  const f=['thumb','index','middle','ring','pinky'].indexOf(pose.finger),q=pose.joint_positions_rad?.slice(f*4,f*4+4);
  const names=f===0?['CMC屈伸','CMC外展','MCP','IP']:['MCP屈伸','MCP外展','PIP','DIP'];
  const angles=q?` · ${q.map((v,i)=>`${names[i]} ${(v*180/Math.PI).toFixed(1)}°`).join(' / ')}`:'';
  const version=pose.reference_id?` · 参考版 ${pose.reference_id.slice(0,8)}`:'';
  el('geortWorkspacePoseStatus').textContent=`参考姿态 #${pose.sample_id}${version}${angles} · 所选指尖 [${pose.point_mm.map(v=>v.toFixed(1)).join(', ')}] mm · 该手指5 mm单元${count>0?`已采集（${count} 帧）`:'尚未采集'}。蓝环对应预览中的金色指尖；不代表整手姿态覆盖。`;
 }
 function clearSelection(){
  selected=null;++request;window.clearGeortWorkspacePose?.();
  el('geortWorkspacePoseStatus').textContent='点击参考点，查看生成该点的Wuji2整手姿态。';
 }
 window.renderGeortCoverage=c=>{
  const last=c||{},key=JSON.stringify(last);if(key===signature)return;signature=key;
  const ids=last.baseline_ids||[],w=last.workspace||{};
  el('geortCoverageScope').textContent=ids.length?`叠加录制 ${ids.length} 段 ＋ 当前一段`:'仅显示当前一段录制';
  const domain=w.output_domain_filtered?` · 输出域外剔除 ${w.rejected_output_domain_samples?.toLocaleString()||0}`:'';
  const policy=w.reference_domain?.robot_extension_policy||w.reference_domain?.glove_extension_policy;
  const strict=!!w.reference_domain?.robot_extension_policy;
  const extension=w.extension_domain_filtered?` · 过度反伸剔除 ${w.rejected_extension_samples?.toLocaleString()||0}${policy?` · 反伸上限 MCP ${policy.mcp_extension_deg}° / PIP ${policy.pip_extension_deg}° / DIP ${policy.dip_extension_deg}° / 拇指 ${policy.thumb_extension_deg}°`:''}`:'';
  el('geortWorkspaceStatus').textContent=w.status==='completed'?
   `Wuji2姿态参考 · ${w.samples?.toLocaleString()||'—'} 组训练FK姿态 · 保留 ${w.accepted_samples?.toLocaleString()||'—'} · 穿透 >${w.penetration_threshold_mm??5} mm剔除 ${w.rejected_penetration_samples?.toLocaleString()||0}${domain}${extension} · 无效 ${w.rejected_invalid_samples||0}${strict?' · 手指屈伸下限0°':w.extension_domain_filtered?' · 已按标定反伸范围筛选':' · 旧参考尚未按标定反伸范围筛选'}`:
   w.status==='failed'?`参考生成失败：${w.error||'请重建参考'}`:
   w.status==='waiting_training_samples'?(w.error||'等待当前GeoRT训练FK样本'):
   `${w.status==='running'||w.status==='starting'?'正在筛选训练FK姿态':'等待有效标定模型'}${w.completed!==undefined?` · 已检查 ${w.completed}/${w.total}`:''}；已有录制点仍可查看`;
  el('geortCoverageStatus').textContent=last.notice||(!last.ready?'开始录制或叠加已有录制，查看采集散点。':
   `${last.recording?'录制中':'已保存录制'} · 累计 ${last.frames||0} 帧 · 当前段 ${last.segment?.frames||0} 帧`);
 };
 const canvas=el('geortCoverageCanvas'),ctx=canvas.getContext?.('2d');
 function draw(){
  if(!ctx)return;
  const width=canvas.clientWidth||640,height=canvas.clientHeight||360,dpr=window.devicePixelRatio||1;
  canvas.width=width*dpr;canvas.height=height*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#f4f6f2';ctx.fillRect(0,0,width,height);
  const source=spatial.fingers?.find(f=>f.finger===chosen())?.cloud_mm||[];
  const f=referenceFinger(),allReference=f?.reference_cloud_mm||[],counts=f?.reference_cloud_counts||[],ids=f?.reference_sample_ids||[];
  const reference=el('geortShowReference').checked!==false?allReference:[];
  const onlyMissing=el('geortOnlyMissing').checked===true;
  const shown=reference.map((p,i)=>({p,i})).filter(v=>!onlyMissing||counts[v.i]===0);
  hits=[];el('geortMissingPose').disabled=!ids.length||!shown.some(v=>counts[v.i]===0);
  el('geortClearPose').disabled=!selected;
  el('geortScatterStatus').textContent=`当前显示：${source.length} 个采集点 · ${shown.length} 个Wuji2参考点（固定毫米尺度，显示可抽样，非动作完成进度）；${ids.length?'点击参考点查看姿态，拖动旋转。':allReference.length?'重建姿态参考后可点选预览。':'等待姿态参考。'}`;
  const points=[...allReference,...source];
  if(!points.length){ctx.fillStyle='#59685d';ctx.fillText('开始录制或叠加已有录制以显示散点',20,30);return;}
  // Keep the millimetre scale fixed across reference changes and fingers.
  const center=[0,0,50],scale=Math.min(width,height)*.72/300;
  const project=p=>{const [x,y,z]=p.map((v,i)=>(v-center[i])),u=Math.cos(yaw)*y+Math.sin(yaw)*x,v=-Math.sin(yaw)*y+Math.cos(yaw)*x;return [width/2+u*scale,height/2-(Math.cos(pitch)*z+Math.sin(pitch)*v)*scale];};
  const origin=project([0,0,0]);
  for(let i=0;i<3;i++){const p=[0,0,0];p[i]=50;const end=project(p);ctx.strokeStyle=['#be5555','#43865c','#547cae'][i];ctx.beginPath();ctx.moveTo(...origin);ctx.lineTo(...end);ctx.stroke();ctx.fillStyle=ctx.strokeStyle;ctx.fillText('XYZ'[i]+' 50mm',end[0]+3,end[1]);}
  for(const {p,i} of shown){
   const [x,y]=project(p);ctx.fillStyle=counts[i]===0?'#c6813d':'#aab4ab';
   ctx.beginPath();ctx.arc(x,y,counts[i]===0?2:1.6,0,Math.PI*2);ctx.fill();
   if(Number.isInteger(ids[i]))hits.push({x,y,sample:ids[i],point:p,count:counts[i]});
  }
  ctx.fillStyle='#249b65';for(const p of source){const [x,y]=project(p);ctx.beginPath();ctx.arc(x,y,2.1,0,Math.PI*2);ctx.fill();}
  const hit=selected&&hits.find(h=>h.sample===selected.sample);
  if(hit){ctx.strokeStyle='#273b78';ctx.lineWidth=2;ctx.beginPath();ctx.arc(hit.x,hit.y,6,0,Math.PI*2);ctx.stroke();ctx.lineWidth=1;}
  if(selected?.pose){const index=ids.indexOf(selected.sample);if(index>=0)selectionStatus(selected.pose,counts[index]||0);}
 }
 async function select(hit){
  const reference_id=spatial.workspace?.reference_id;if(!reference_id)return;
  const token=++request;selected=hit;draw();
  el('geortWorkspacePoseStatus').textContent=`正在加载参考姿态 #${hit.sample}…`;
  try{
   const url='/api/geort/workspace/pose?'+new URLSearchParams({reference_id,sample_id:hit.sample,finger:chosen()});
   const response=await fetch(url,{cache:'no-store',signal:AbortSignal.timeout(10000)}),pose=await response.json();
   if(token!==request)return;
   if(!response.ok||!pose.ready)throw Error(pose.error||'姿态预览不可用');
   if(pose.reference_id!==reference_id||pose.sample_id!==hit.sample||pose.finger!==chosen())throw Error('参考已变化，请重新选择');
   if(!window.showGeortWorkspacePose)throw Error('姿态预览正在初始化，请稍后再点击');
   await window.showGeortWorkspacePose(pose);
   if(token!==request)return;
   selected.pose=pose;selectionStatus(pose,pose.recorded_cell_frames||0);
  }catch(error){if(token===request){window.clearGeortWorkspacePose?.();el('geortWorkspacePoseStatus').textContent=error.message;}}
 }
 window.resetGeortCoverageRecording=()=>{
  spatial={...spatial,baseline_ids:[],frames:0,segment:{frames:0},recording:true,notice:null,fingers:[],
   workspace:{...spatial.workspace,fingers:spatial.workspace?.fingers?.map(f=>({...f,reference_cloud_counts:(f.reference_cloud_mm||[]).map(()=>0)}))}};
  window.renderGeortCoverage(spatial);draw();
 };
 window.updateGeortCoverageCloud=c=>{
  const next=c?.compatible===false?{}:c||{};
  if(spatial.workspace?.reference_id!==next.workspace?.reference_id)clearSelection();
  spatial=next;draw();
 };
 el('geortWorkspaceFinger').onchange=()=>{clearSelection();draw();window.refreshGeortCoverage?.();};
 el('geortShowReference').onchange=()=>{clearSelection();draw();};
 el('geortOnlyMissing').onchange=()=>{clearSelection();draw();};
 el('geortClearPose').onclick=()=>{clearSelection();draw();};
 el('geortMissingPose').onclick=()=>{
  const missing=hits.filter(h=>h.count===0);if(!missing.length)return;
  const i=missing.findIndex(h=>h.sample===selected?.sample);void select(missing[(i+1)%missing.length]);
 };
 canvas.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,lastX:e.clientX,lastY:e.clientY,moved:false};canvas.setPointerCapture?.(e.pointerId);};
 canvas.onpointermove=e=>{
  if(!drag)return;
  if(Math.hypot(e.clientX-drag.x,e.clientY-drag.y)>4)drag.moved=true;
  if(drag.moved){yaw+=(e.clientX-drag.lastX)*.01;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-drag.lastY)*.01));draw();}
  drag.lastX=e.clientX;drag.lastY=e.clientY;
 };
 canvas.onpointerup=e=>{
  const was=drag;drag=null;if(!was||was.moved||Math.hypot(e.clientX-was.x,e.clientY-was.y)>4)return;
  const rect=canvas.getBoundingClientRect?.()||{left:0,top:0};
  const x=e.clientX-rect.left,y=e.clientY-rect.top;
  const hit=hits.map(h=>({...h,d:Math.hypot(x-h.x,y-h.y)})).filter(h=>h.d<=8).sort((a,b)=>a.d-b.d)[0];
  if(hit)void select(hit);
 };
 canvas.onpointercancel=()=>{drag=null;};
 window.addEventListener?.('resize',draw);draw();
})();
