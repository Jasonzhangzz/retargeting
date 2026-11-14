import * as THREE from 'three';
import {OrbitControls} from '/vendor/OrbitControls.js';
import {CurvedSurface} from './curved-surface.js';
import {createTaxels,TAXEL_PURPLE,TAXEL_YELLOW} from './taxel-view.js';

const canvas=document.getElementById('wujiCanvas'),panel=document.getElementById('wujiCompare');
const basis=p=>[p[1],p[0],-p[2]];
let renderer,camera,controls,scenes,mano,robot=[],surface,topology,model,version,loading=false,current,visible=false,centered=false,drawSignature,displayedFrameKey,displayedFrame;
let renderSize='';
let dots=[],normals=[],rays=[],errors,anchorDots=[],contactGraph=[];
let packedGeometryKey,packedGeometryData,packedGeometry;
const backendOf=w=>w?.solver_profile==='cem_contact'?'slsqp_cem':w?.solver_profile==='edge_laplacian'?'slsqp_edge':w?.solver_profile==='contact_graph'?'slsqp_graph':w?.solver_backend||(w?.algorithm?.startsWith('card_')?'cem':w?.algorithm?.startsWith('moxian_')?'slsqp':null);
const viewKey=(w,h={})=>[w.mapping_version,w.geometry_version||h.geometry_version,w.source_binding_version||h.binding_version,backendOf(w),w.algorithm,w.topology_version].map(x=>x??'').join(':');
function geometryFrame(w){
 const descriptor=w.geometry_f32;if(!descriptor)return w;
 const key=[w.solver_session_id,w.timestamp_ms,w.mapping_version].join(':');
 if(key!==packedGeometryKey||descriptor.data!==packedGeometryData){
  if(descriptor.format!=='wuji-f32-v1')throw Error('Wuji2 几何传输版本不匹配');
  const raw=atob(descriptor.data),bytes=Uint8Array.from(raw,c=>c.charCodeAt(0)),geometry={};
  for(const [name,field] of Object.entries(descriptor.fields)){
   const {shape,offset}=field,count=shape.reduce((a,b)=>a*b,1);
   if(![2,3].includes(shape.length)||shape.at(-1)!==3||offset<0||offset%4||offset+count*4>bytes.length)throw Error('Wuji2 几何帧损坏');
   const values=new Float32Array(bytes.buffer,offset,count),rows=Array.from({length:count/3},(_,i)=>Array.from(values.subarray(3*i,3*i+3)));
   geometry[name]=shape.length===3?Array.from({length:shape[0]},(_,i)=>rows.slice(3*i,3*i+3)):rows;
  }
  packedGeometryKey=key;packedGeometryData=descriptor.data;packedGeometry=geometry;
 }
 return {...w,...packedGeometry};
}
const transform=(p,m)=>m.map(row=>row.reduce((s,x,k)=>s+x*p[k],0));
const lineObject=color=>new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color,depthTest:false,transparent:true,opacity:.65}));
function init(){
 renderer=new THREE.WebGLRenderer({canvas,antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor('#29342d');
 camera=new THREE.PerspectiveCamera(38,1,.001,3);controls=new OrbitControls(camera,canvas);controls.enableDamping=false;controls.minDistance=.06;controls.maxDistance=.9;
 scenes=[new THREE.Scene(),new THREE.Scene()];
 scenes.forEach((s,i)=>{s.add(new THREE.HemisphereLight('#fff4df','#526556',2.5));const l=new THREE.DirectionalLight('#fff1dc',2);l.position.set(.2,.4,.3);s.add(l);dots[i]=createTaxels();s.add(dots[i]);normals[i]=lineObject('#a9d3dc');rays[i]=lineObject('#bdadca');s.add(normals[i],rays[i]);});
 scenes.forEach((scene,i)=>{anchorDots[i]=new THREE.Points(new THREE.BufferGeometry(),new THREE.PointsMaterial({color:'#e8a05c',size:.0035,sizeAttenuation:true,depthTest:false}));anchorDots[i].renderOrder=6;scene.add(anchorDots[i]);});
 scenes.forEach((scene,i)=>{contactGraph[i]=lineObject('#e8a05c');scene.add(contactGraph[i]);});
 errors=lineObject('#e8ba69');scenes[1].add(errors);controls.addEventListener('change',render);
}
function render(){
 if(!renderer||!visible)return;
 const box=canvas.getBoundingClientRect();if(!box.width||!box.height)return;
 const size=[box.width,box.height,renderer.getPixelRatio()].join(':');
 if(size!==renderSize){renderer.setSize(box.width,box.height,false);renderSize=size;}
 const stacked=box.width<600;
 const w=stacked?box.width:box.width/2,h=stacked?box.height/2:box.height;
 canvas.parentElement.classList.toggle('stacked',stacked);
 camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setScissorTest(true);
 scenes.forEach((s,i)=>{const x=stacked?0:i*w,y=stacked?(1-i)*h:0;renderer.setViewport(x,y,w,h);renderer.setScissor(x,y,w,h);renderer.render(s,camera);});
 renderer.setScissorTest(false);
}
// Only complete, finite endpoint results may replace a displayed pair. Worker
// freshness controls contact overlays, not the lifetime of the canvas geometry.
const vectors=a=>Array.isArray(a)&&a.length>0&&a.every(p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite));
function completeFrame(w,b=model,a=topology){
 const count=a?.rest_vertices_m?.length;
 return vectors(w.source_vertices_m)&&(!count||w.source_vertices_m.length===count)&&
  vectors(w.link_positions_m)&&Array.isArray(w.link_rotations)&&w.link_rotations.length===w.link_positions_m.length&&w.link_rotations.every(r=>r.length===3&&vectors(r))&&
  (!b?.link_names||w.link_positions_m.length===b.link_names.length)&&
  ['source_taxel_positions_m','taxel_positions_m','source_taxel_normals','taxel_normals','source_joints_m'].every(k=>vectors(w[k]))&&
  w.source_taxel_positions_m.length===w.source_taxel_normals.length&&w.taxel_positions_m.length===w.taxel_normals.length;
}
async function load(wanted,state){
 if(loading)return;loading=true;
 try{
  if(window.isMoxianReplay&&(!window.replayingManoTopology||!window.replayingWujiTopology))throw Error('录制缺少对照拓扑，请用离线脚本重新求解');
  const [a,b]=await Promise.all([window.replayingManoTopology||fetch('/api/hand_model/model').then(r=>r.json()),window.replayingWujiTopology||fetch('/api/wuji2/model').then(r=>r.json())]);
  if(!a.ready||!b.ready)return;
  if(b.mapping.geometry_version!==a.geometry_version||b.mapping.binding_version!==a.binding_version||b.mapping.version!==state.wuji2.mapping_version)throw Error('等待同版本源手套／Wuji2 映射');
  const next=current?.wuji2;
  if(!(next?.ready||next?.display_available)||next.mapping_version!==b.mapping.version||wanted!==viewKey(next,a)||
     (backendOf(b)&&backendOf(b)!==backendOf(next))||(b.algorithm&&b.algorithm!==next.algorithm)||
     (b.topology_version&&b.topology_version!==next.topology_version)||!completeFrame(next,b,a))return;
  const nextSurface=new CurvedSurface(a),nextRobot=[];
  const g=new THREE.BufferGeometry();g.setIndex(nextSurface.faces.flatMap(f=>a.hand_side==='left'?[f[0],f[2],f[1]]:f));g.setAttribute('position',new THREE.Float32BufferAttribute(nextSurface.positions.length,3));
  const nextMano=new THREE.Mesh(g,new THREE.MeshStandardMaterial({color:'#d4ba96',roughness:.75,side:THREE.DoubleSide,polygonOffset:true,polygonOffsetFactor:1,polygonOffsetUnits:1}));nextMano.frustumCulled=false;
  b.link_names.forEach((_,owner)=>{
   const ids=b.vertex_owners.map((v,i)=>v===owner?i:-1).filter(i=>i>=0);if(!ids.length)return;
   const start=ids[0],local=b.local_vertices_m.slice(start,start+ids.length);
   const faces=b.faces.filter(f=>b.vertex_owners[f[0]]===owner).flatMap(f=>f.map(i=>i-start));
   const g=new THREE.BufferGeometry();g.setIndex(faces);g.setAttribute('position',new THREE.Float32BufferAttribute(local.flatMap(basis),3));g.computeVertexNormals();
   const m=new THREE.Mesh(g,new THREE.MeshStandardMaterial({color:'#aeb8b7',roughness:.65,side:THREE.DoubleSide,polygonOffset:true,polygonOffsetFactor:1,polygonOffsetUnits:1}));m.matrixAutoUpdate=false;m.userData.owner=owner;m.frustumCulled=false;nextRobot.push(m);
  });
  // Build off scene; swap both models together only when their first frame exists.
  if(mano){scenes[0].remove(mano);mano.geometry.dispose();mano.material.dispose();}
  robot.forEach(m=>{scenes[1].remove(m);m.geometry.dispose();m.material.dispose();});surface?.dispose();
  topology=a;model=b;version=wanted;surface=nextSurface;mano=nextMano;robot=nextRobot;drawSignature=null;displayedFrame=null;displayedFrameKey=null;
  scenes[0].add(mano);robot.forEach(m=>scenes[1].add(m));
  update(current||state,visible);
  window.wujiRenderError=null;
 }catch(e){document.getElementById('wujiStatus').textContent=e.message;window.wujiRenderError=e.message;}
 finally{loading=false;}
}
function points(object,positions,w){
 const g=object.geometry,n=positions.length;
 if(g.attributes.position?.count!==n){g.setAttribute('position',new THREE.Float32BufferAttribute(n*3,3));g.setAttribute('color',new THREE.Float32BufferAttribute(n*3,3));}
 const purple=new THREE.Color(TAXEL_PURPLE),yellow=new THREE.Color(TAXEL_YELLOW),invalid=new THREE.Color('#e26c65'),rejected=new Set(w.normal_rejected_sids||[]);
 positions.forEach((p,i)=>{g.attributes.position.setXYZ(i,...basis(p));const c=rejected.has(i)?invalid:w.contact_ready&&w.contact_mask?.[i]===true?yellow:purple;g.attributes.color.setXYZ(i,c.r,c.g,c.b);});
 g.attributes.position.needsUpdate=g.attributes.color.needsUpdate=true;g.computeBoundingSphere();
}
function lines(object,values){object.geometry.setAttribute('position',new THREE.Float32BufferAttribute(values,3));object.visible=values.length>0;}
function update(state,active){
 const began=performance.now();
 const activated=active&&!visible;
 current=state;visible=active;panel.hidden=!active;
 if(!active)return;
 try{
  if(!renderer)init();
  let w=geometryFrame(state?.wuji2||{});const h=state?.hand_model||{};
  current={...state,wuji2:w};
  if(window.isMoxianReplay&&(!window.replayingManoTopology||!window.replayingWujiTopology)){
   drawSignature=null;displayedFrameKey=null;displayedFrame=null;if(mano)mano.visible=false;robot.forEach(m=>m.visible=false);dots.forEach(d=>d.visible=false);anchorDots.forEach(d=>d.visible=false);[...normals,...rays,...contactGraph,errors].forEach(l=>l.visible=false);
   document.getElementById('wujiStatus').textContent='录制缺少对照拓扑，请用离线脚本重新求解';render();return;
  }
  const expected=window.wujiRequestedSolver||w.requested_solver_backend||w.active_solver_backend||backendOf(w);
  if(displayedFrame&&expected&&backendOf(displayedFrame)!==expected){
   displayedFrame=null;displayedFrameKey=null;drawSignature=null;
  }
  const wanted=viewKey(w,h);
  if(w.mapping_version&&h.geometry_version&&version!==wanted&&(!expected||backendOf(w)===expected))load(wanted,current);
  const count=model?.mapping?.records.filter(Boolean).length||0,q=w.quality||{};
  const robokit=w.solver?.initializer?.method==='robokit_hand_retargeter_online';
  const canPresent=!!(w.ready||w.display_available)&&(!expected||backendOf(w)===expected)&&
   (!w.active_solver_session_id||w.active_solver_session_id===w.solver_session_id)&&
   w.mapping_version===model?.mapping.version&&version===wanted&&completeFrame(w);
  const holding=!canPresent&&!!displayedFrame;
  const warp=w.solver?.backend==='robokit_population_solver_warp_card',timing=w.solver?.timing_ms;
  const slsqp=w.solver_backend==='slsqp';
  const selector=document.getElementById('wujiSolver');if(selector&&expected)selector.value=expected;
  if(canPresent&&window.wujiRequestedSolver===backendOf(w))window.wujiRequestedSolver=null;
  const shown=canPresent?w:holding?displayedFrame:null,shownBackend=backendOf(shown);
  const shownLabel=shownBackend==='slsqp'?'SLSQP · 统一图':shownBackend==='cem'?'CEM · CARD':'未知后端';
  document.querySelector('.wuji-right-label').textContent='WUJI2.2 · '+(shown?shownLabel:`等待 ${(expected||'solver').toUpperCase()} 首帧`);
  document.getElementById('wujiFrameIdentity').textContent=shown?
   `画面后端：${shownLabel} · ${window.isMoxianReplay?'录制回放':holding?'暂停画面':'实时输出'} · ${shown.algorithm||'旧录制未记录算法版本'} · 会话 ${shown.solver_session_id?.slice(0,8)||'—'} · 源帧 ${shown.source_timestamp_ms?.toFixed(0)||'—'} ms`:
   `画面后端：等待 ${(expected||'solver').toUpperCase()} 首帧 · 切换期间不显示其他后端的旧姿态`;
  const phase=slsqp&&Number.isFinite(timing?.slsqp)?` · SLSQP ${timing.slsqp.toFixed(1)} ms · 验收 ${timing.final_audit.toFixed(1)} ms`:Number.isFinite(timing?.robokit)&&Number.isFinite(timing?.cem)?` · 初解 ${timing.robokit.toFixed(1)} ms · CEM含验收 ${timing.cem.toFixed(1)} ms`:'';
  const failedCandidate=canPresent&&w.solution_feasible===false;
  const solverLabel=slsqp?(backendOf(w)==='slsqp_cem'?'Moxian → SLSQP · 纯 CEM 接触 cost':backendOf(w)==='slsqp_graph'?'Moxian → SLSQP · 离散接触图 + Laplacian':backendOf(w)==='slsqp_edge'?'Moxian → SLSQP · 接触 edges + Laplacian':'Moxian → SLSQP'):`${warp?'Robokit → Warp CARD CEM':robokit?'Robokit → CEM':'CARD CEM'} 1024 × 16`;
  document.getElementById('wujiStatus').textContent=(w.error||(!state?.connected?'设备离线':failedCandidate?solverLabel+' · 显示本帧候选 · 约束超限'+(w.status==='solving'?' · 下一帧计算中':''):w.ready?`${solverLabel} · ${w.contact_anchors?.length||0} 个锚点 · ${w.contact_edges?.length||0} 条跨 link 边${phase} · 整帧 ${q.processing_ms?.toFixed(1)||'—'} ms · 源延迟 ${w.source_age_ms?.toFixed(0)||'—'} ms${w.status==='solving'?' · 下一帧计算中':''}`:'Wuji2 '+({mapping:'建立表面映射中',initializing:'初始化求解器',solving:w.input_live?'求解中':'输入暂停',infeasible:'显示本帧候选 · 约束超限',starting:'启动中'}[w.status]||'等待有效 Moxian mesh')))+(holding?' · 保留最后画面，接触状态未知':'');
  if(expected&&backendOf(w)!==expected)document.getElementById('wujiStatus').textContent=`切换至 ${expected.toUpperCase()}，等待该后端的新结果`;
  if(slsqp)document.getElementById('wujiStatus').textContent+=` · 输出 ${q.output_hz?.toFixed(1)||'—'} Hz · 输入排队 ${q.input_queue_ms?.toFixed(1)||'—'} ms · 初值 ${w.solver?.initial_guess==='previous_raw_solution'?'上一原始解':w.solver?.initial_guess||'—'}`+(w.solver?.reset_reason?` · 重置 ${w.solver.reset_reason}`:'');
  if(slsqp&&w.solver?.execution_mode==='penetration_restoration')document.getElementById('wujiStatus').textContent+=w.solver.escape_continuation_active?' · 穿透恢复：沿已验收姿态分步退出':' · 穿透恢复：优先减小负间距';
  if(w.taxel_projection?.normal_rejected_sids?.length)document.getElementById('wujiStatus').textContent+=` · ${w.taxel_projection.normal_rejected_sids.length} 个SID法向不匹配，不参与接触图`;
  if(slsqp&&state?.contact?.mapping_confirmed===0)document.getElementById('wujiStatus').textContent+=' · SID分区尚未实物核验';
  const mesh=w.collision?.self_mesh;
  const seed=w.solver?.tactile_seed,pairs=w.self_contact_pairs||[];
  document.getElementById('wujiError').textContent=(q.contact_rmse_mm==null?'接触误差：—':`源接触点 RMS ${q.contact_rmse_mm.toFixed(1)} mm · 中心 ${q.patch_center_rmse_mm?.toFixed(1)||'—'} mm · 最大 ${q.contact_max_error_mm.toFixed(1)} mm · 法向 ${q.contact_normal_mean_deg.toFixed(1)}°`)+(seed?.applied?` · 触觉初解 ${seed.gap_before_mm.toFixed(1)} → ${seed.gap_after_mm.toFixed(1)} mm`:'')+(pairs.length?` · 最终对指点距 ${Math.max(...pairs.map(p=>p.distance_mm)).toFixed(1)} mm${w.contact_constraints_satisfied?'':'（未达闭合门槛）'}`:'')+(mesh?` · 自穿透 ${mesh.max_penetration_mm.toFixed(2)} / ${w.collision.self_hard_limit_mm} mm · ${w.collision.object_hard_gate_enabled===false?'推测物体侵入（软约束）':'物体穿透'} ${w.collision.object_max_penetration_mm.toFixed(2)} mm`:'')+` · ${q.joint_bounds_saturated??0} 关节到达限位 · SDF ${w.sdf?.type||'—'} ${w.sdf?.ready?'可用':'点集不足'} · 硬拒绝 ${w.solver?.hard_rejected??0}`+(q.cost_terms?' · '+Object.entries(q.cost_terms).map(([k,v])=>`${k} ${v.toFixed(1)}`).join(' / '):'');
  const thumbEdges=(w.contact_edges||[]).filter(e=>e.semantic_links?.includes('thumb_distal')&&e.semantic_links.some(l=>!l.startsWith('thumb')&&l.endsWith('_distal'))&&Array.isArray(e.source_vector_m)&&Array.isArray(e.robot_vector_m));
  const sourceError=q.source_quality?.position_rmse_mm;
  if(slsqp){
   const mm=value=>Number.isFinite(value)?value.toFixed(2)+' mm':'—';
   document.getElementById('wujiError').textContent=`统一图 Laplacian RMS ${mm(q.unified_laplacian_rmse_mm)} · 接触子图 RMS ${mm(q.contact_laplacian_rmse_mm)}（诊断） · 边向量 RMS ${mm(q.contact_edge_rmse_mm)} · 自穿透 ${mm(mesh?.max_penetration_mm)} · 物体穿透 ${mm(w.collision?.object_max_penetration_mm)} · ${w.collision?.separation_satisfied?'零间距验收通过':'等待零间距验收'} · ${q.joint_bounds_saturated??0} 关节到达限位`+(q.cost_terms?' · '+Object.entries(q.cost_terms).map(([k,v])=>`${k} ${v.toFixed(3)}`).join(' / '):'');
  }
  if(thumbEdges.length){const nearest=thumbEdges.reduce((a,b)=>Math.hypot(...a.source_vector_m)<Math.hypot(...b.source_vector_m)?a:b);document.getElementById('wujiError').textContent+=` · 拇指最近跨指材料点距：源 ${(Math.hypot(...nearest.source_vector_m)*1000).toFixed(1)} / Wuji ${(Math.hypot(...nearest.robot_vector_m)*1000).toFixed(1)} mm`;}
  const cg=w.source_contact_geometry,verified=cg?.active_verified_sids?.length||0,unverified=cg?.active_unverified_sids?.length||0;
  const sourceRegistration=slsqp?` · Moxian 原始表面材料绑定 · ${cg?.personalized_moxian?'已加载个体骨长':'默认骨长'}`:cg?` · 源接触位置已实测核验 ${verified}/${verified+unverified} SID`+(cg.personalized_mano?'':' · 个性化未验收')+(cg.sensor_mounting_status==='multi_pose_calibrated'?'':' · EMF 安装未验收'):' · 源接触位置未实测核验';
  document.getElementById('wujiSourceQuality').textContent='源 EMF 安装诊断残差：'+(Number.isFinite(sourceError)?`${sourceError.toFixed(1)} mm`:'不可用或原始配准未就绪')+sourceRegistration+` · 自接触 ${w.self_contact_pairs?.length||0} 对`+(w.self_contact_pairs?.length?` · 最大点距 ${Math.max(...w.self_contact_pairs.map(x=>x.distance_mm)).toFixed(1)} mm`:'')+` · 机器人虚拟投影 ${w.taxel_projection?.mapped_sids??count} SID`+` · 力权重为相对强度；实时未滤波`+(robokit?` · Robokit 本帧初解 ${w.solver.initializer.elapsed_ms.toFixed(1)} ms`:Number.isFinite(w.initializer_quality?.held_out_joint_mean_error_deg)?` · 拇指初值留出均差 ${w.initializer_quality.held_out_joint_mean_error_deg.toFixed(1)}°`:'');
  if(q.source_quality?.source_ik_solver)document.getElementById('wujiSourceQuality').textContent+=` · 源骨架 ${q.source_quality.source_ik_solver} ${q.source_quality.source_ik_processing_ms?.toFixed(2)||'—'} ms`;
  if(!mano||!surface||!model||!(canPresent||holding)){if(mano)mano.visible=false;robot.forEach(m=>m.visible=false);dots.forEach(d=>d.visible=false);anchorDots.forEach(d=>d.visible=false);[...normals,...rays,...contactGraph,errors].forEach(l=>l.visible=false);render();return;}
  if(holding)w=displayedFrame;
  const frameKey=[backendOf(w),w.algorithm,w.solver_session_id,w.mapping_version,w.timestamp_ms].join(':');
  const contactReady=!!w.contact_ready&&!holding;
  const signature=[frameKey,version,contactReady,...['wujiSid','wujiNormals','wujiRays','wujiErrors'].map(id=>document.getElementById(id).type==='checkbox'?document.getElementById(id).checked:document.getElementById(id).value)].join(':');
  if(signature===drawSignature){if(activated)render();return;}drawSignature=signature;
  mano.visible=true;robot.forEach(m=>m.visible=true);
  const curved=surface.update(w.source_vertices_m),attr=mano.geometry.attributes.position;
  for(let i=0;i<curved.length;i+=3)attr.setXYZ(i/3,curved[i+1],curved[i],-curved[i+2]);attr.needsUpdate=true;mano.geometry.computeVertexNormals();
  // B maps source world coordinates into the displayed Three.js coordinates.
  const B=new THREE.Matrix4().set(0,1,0,0,1,0,0,0,0,0,-1,0,0,0,0,1);
  robot.forEach(m=>{const i=m.userData.owner,r=w.link_rotations[i],p=w.link_positions_m[i];m.matrix.copy(B).multiply(new THREE.Matrix4().set(...r[0],p[0],...r[1],p[1],...r[2],p[2],0,0,0,1)).multiply(B);});
  anchorDots.forEach((dot,i)=>{dot.geometry.setAttribute('position',new THREE.Float32BufferAttribute((w.contact_anchors||[]).flatMap(a=>basis(i===0?a.position_m:a.robot_position_m)),3));dot.geometry.computeBoundingSphere();dot.visible=contactReady;});
  const sourceLabel=topology?.surface_backend==='qnbot'?'Moxian glove':'MANO';
  document.querySelector('.wuji-left-label').textContent=w.source_hand_side==='left'?sourceLabel+' · 左手镜像':sourceLabel+' · 右手';
  dots[0].visible=true;dots[1].visible=!!model.mapping.ready;const contact={contact_ready:contactReady,contact_mask:w.contact_mask};
  const sourceContactReady=Array.isArray(w.observed_contact_mask)?!holding&&w.input_live!==false:contactReady;
  points(dots[0],w.source_taxel_positions_m,{contact_ready:sourceContactReady,contact_mask:w.observed_contact_mask||w.contact_mask});points(dots[1],w.taxel_positions_m,{...contact,normal_rejected_sids:w.taxel_projection?.normal_rejected_sids});
  const sid=Math.max(0,Math.min(459,Number(document.getElementById('wujiSid').value)||0));
  const record=model.mapping.records[sid];document.getElementById('wujiSidInfo').textContent=record?`虚拟 SID ${sid} · ${record.link} · 法向映射偏差 ${record.normal_error_deg.toFixed(1)}°${record.normal_valid===false?' · 法向不匹配，不参与接触图':''}`:`SID ${sid} 未映射`;
  const source=[w.source_taxel_positions_m,w.taxel_positions_m],ns=[w.source_taxel_normals,w.taxel_normals];
  for(let i=0;i<2;i++){
   const p=source[i][sid],n=ns[i][sid];lines(normals[i],document.getElementById('wujiNormals').checked?[...basis(p),...basis(p.map((x,k)=>x+n[k]*.008))]:[]);
   const origin=i===0?w.source_joints_m[record?.source_joint_index||0]:w.link_positions_m[record?.owner||0];
   lines(rays[i],document.getElementById('wujiRays').checked?[...basis(origin),...basis(p)]:[]);
  }
  const errorPoints=[];if(document.getElementById('wujiErrors').checked&&contactReady)(w.contact_anchors||[]).forEach(a=>errorPoints.push(...basis(a.position_m),...basis(a.robot_position_m)));lines(errors,errorPoints);
  contactGraph.forEach((object,i)=>{
   const values=[];
   if(contactReady){
    (w.contact_edges||[]).forEach(edge=>edge.nodes.forEach(k=>{const a=w.contact_anchors[k];values.push(...basis(i===0?a.position_m:a.robot_position_m));}));
    const graph=w.retarget_graph,nodes=i===0?graph?.source_nodes_m:graph?.robot_nodes_m;
    if(nodes)(graph.edges||[]).filter(edge=>edge.kind==='material_attachment'&&edge.nodes[0]<edge.nodes[1]).forEach(edge=>edge.nodes.forEach(k=>values.push(...basis(nodes[k]))));
   }
   lines(object,values);
  });
  if(!centered){
   const box=new THREE.Box3();w.source_vertices_m.forEach(p=>box.expandByPoint(new THREE.Vector3(...basis(p))));controls.target.copy(box.getCenter(new THREE.Vector3()));window.setWujiView('front');centered=true;
  }
  render();
  displayedFrameKey=frameKey;
  displayedFrame=w;
  window.wujiViewStats={solver_backend:backendOf(w),algorithm:w.algorithm,solver_session_id:w.solver_session_id,source_timestamp_ms:w.source_timestamp_ms,holding,update_ms:performance.now()-began,render_submission_latency_ms:Date.now()-w.source_timestamp_ms};
 }catch(e){document.getElementById('wujiStatus').textContent=e.message;window.wujiRenderError=e.message;}
}
window.updateWujiView=update;
window.setWujiView=mode=>{
 if(!controls||!topology||!model)return;
 const M=model.mapping.mirror_matrix,n=new THREE.Vector3(...basis(transform(topology.palm_view.normal,M))).normalize(),up=new THREE.Vector3(...basis(transform(topology.palm_view.forward,M))).normalize();
 const side=new THREE.Vector3().crossVectors(up,n).normalize();camera.up.copy(up);camera.position.copy(controls.target).addScaledVector(n,.4);
 if(mode==='top')camera.position.addScaledVector(up,.18);else if(mode!=='front')camera.position.addScaledVector(side,.1);
 controls.update();render();
};
for(const id of ['wujiSid','wujiNormals','wujiRays','wujiErrors'])document.getElementById(id).addEventListener('input',()=>update(current,visible));
window.addEventListener('resize',render);
if(window.currentMoxianState)update(window.currentMoxianState,document.getElementById('poseMode').value==='wuji2');
window.debugWujiView=()=>({visible,version,loading,hasFrame:!!displayedFrame,frameKey:displayedFrameKey,solver_backend:backendOf(displayedFrame),algorithm:displayedFrame?.algorithm,requested_solver:window.wujiRequestedSolver,mano:!!mano?.visible,robot:robot.every(m=>m.visible),canvas:[canvas.width,canvas.height],error:window.wujiRenderError});
