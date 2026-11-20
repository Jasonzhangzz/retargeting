import * as THREE from 'three';
import {OrbitControls} from '/vendor/OrbitControls.js';
import {CurvedSurface} from './curved-surface.js';
import {createTaxels,TAXEL_PURPLE,TAXEL_YELLOW} from './taxel-view.js';

const canvas=document.getElementById('wujiCanvas'),panel=document.getElementById('wujiCompare');
const basis=p=>[p[1],p[0],-p[2]];
let renderer,camera,controls,scenes,mano,robot=[],rvqRobot=[],surface,topology,model,version,loading=false,current,visible=false,centered=false,drawSignature,displayedFrameKey,displayedFrame;
let renderSize='';
let dots=[],normals=[],rays=[],errors,anchorDots=[],contactGraph=[];
let packedGeometryKey,packedGeometryData,packedGeometry;
const backendOf=w=>w?.solver_profile==='contact_graph'?'slsqp_graph':w?.solver_backend||(w?.algorithm?.startsWith('card_')?'cem':w?.algorithm?.startsWith('moxian_')?'slsqp':null);
function syncPredictionVisibility(){
 const showGrasp=document.getElementById('wujiShowGraspType').checked;
 document.getElementById('wujiRVQ').hidden=!showGrasp;
 document.getElementById('wujiGraspPreview').hidden=!showGrasp;
 if(!document.getElementById('wujiRVQOverlay').checked)rvqRobot.forEach(m=>m.visible=false);
}
function collisionLinkName(name){
 if(name==='right_palm_link'||name==='left_palm_link')return 'palm';
 return String(name).replace(/^(?:right|left)_finger([1-5])_link([1-4])$/,(_,f,k)=>`${['thumb','index','middle','ring','pinky'][Number(f)-1]}_link${k}`);
}
function nonAdjacentCollisionPair(names){
 if(names.length!==2||names[0]===names[1])return false;
 const parsed=names.map(name=>/^(thumb|index|middle|ring|pinky)_link([1-4])$/.exec(name));
 if(names.includes('palm')){
  const finger=parsed[names[0]==='palm'?1:0];
  return !!finger&&Number(finger[2])>1;
 }
 if(parsed.some(link=>!link))return false;
 return parsed[0][1]!==parsed[1][1]||Math.abs(Number(parsed[0][2])-Number(parsed[1][2]))>1;
}
function selfCollisionLinks(report,valid,threshold=.5){
 const links=new Set();
 if(valid&&report?.query_valid===true){
  for(const [pair,depth] of Object.entries(report.pair_depths_mm||{})){
   const names=pair.split('__').map(collisionLinkName);
   if(Number.isFinite(depth)&&depth>threshold&&nonAdjacentCollisionPair(names))names.forEach(name=>links.add(name));
  }
 }
 return links;
}
function updateSelfCollisionColors(report,valid){
 const links=selfCollisionLinks(report,valid);let changed=false;
 for(const mesh of robot){
  const name=model?.link_names?.[mesh.userData.owner];
  const color=links.has(collisionLinkName(name))?'#ed4545':'#aeb8b7';
  if(mesh.userData.collisionColor!==color){mesh.material.color.set(color);mesh.userData.collisionColor=color;changed=true;}
 }
 return changed;
}
function cardWindowLabel(w){
 if(!Number.isFinite(w?.buffer_delay_frames))return '';
 if(w.buffer_delay_frames===0)return ' · 单帧实时 · 无未来帧缓冲';
 return ` · 前瞻 ${w.buffer_delay_frames} 帧 · 缓冲 ${w.buffer_delay_ms?.toFixed(1)||'—'} ms · 总延迟 ${w.source_age_ms?.toFixed(0)||'—'} ms`;
}
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
  const nextRVQ=nextRobot.map(original=>{
   const m=new THREE.LineSegments(new THREE.EdgesGeometry(original.geometry,20),new THREE.LineBasicMaterial({color:'#58c8ee',transparent:true,opacity:.8,depthWrite:false}));
   m.matrixAutoUpdate=false;m.userData.owner=original.userData.owner;m.frustumCulled=false;m.visible=false;return m;
  });
  // Build off scene; swap both models together only when their first frame exists.
  if(mano){scenes[0].remove(mano);mano.geometry.dispose();mano.material.dispose();}
  rvqRobot.forEach(m=>{scenes[1].remove(m);m.geometry.dispose();m.material.dispose();});
  robot.forEach(m=>{scenes[1].remove(m);m.geometry.dispose();m.material.dispose();});surface?.dispose();
  topology=a;model=b;version=wanted;surface=nextSurface;mano=nextMano;robot=nextRobot;rvqRobot=nextRVQ;drawSignature=null;displayedFrame=null;displayedFrameKey=null;
  scenes[0].add(mano);robot.forEach(m=>scenes[1].add(m));rvqRobot.forEach(m=>scenes[1].add(m));
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
function updateRVQPanel(annotation,holding,sourceError){
 syncPredictionVisibility();
 const r=annotation,ready=!!r?.ready;
 const title=name=>String(name||'—').replace(/^(\d+)_/,'$1 · ').replaceAll('_',' ');
 const image=document.getElementById('wujiPrimitiveImage'),primitive=ready?model?.rvq?.primitives?.[r.grasp_type]:null;
 document.getElementById('wujiGraspType').textContent=ready?title(r.grasp_type):'等待预测';
 document.getElementById('wujiGraspConfidence').textContent=ready?`${(r.probability*100).toFixed(1)}%`:'—';
 if(primitive){
  if(image.dataset.graspType!==r.grasp_type){image.src=primitive.image;image.alt=`${title(r.grasp_type)} primitive 静态示意图`;image.dataset.graspType=r.grasp_type;}
  image.hidden=false;
 }else{image.hidden=true;image.removeAttribute('src');delete image.dataset.graspType;}
 document.getElementById('wujiPrimitiveStatus').textContent=ready?(primitive?'标准 primitive · 静态示意':model?.rvq?.primitive_error||'该类别暂无 primitive 示意图'):
  holding?'等待恢复预测':'等待手套姿态';
 document.getElementById('wujiRVQLabel').textContent=ready?title(r.grasp_type):'—';
 document.getElementById('wujiRVQProbability').textContent=ready?`${(r.probability*100).toFixed(1)}%`:'—';
 document.getElementById('wujiRVQError').textContent=ready?`${r.reconstruction_mae_deg.toFixed(2)}°`:'—';
 document.getElementById('wujiRVQTop3').textContent=ready?'Top-3：'+r.top3.map(item=>`${title(item.grasp_type)} ${(item.probability*100).toFixed(1)}%`).join(' ／ '):'Top-3：—';
 document.getElementById('wujiRVQTokens').textContent=ready?r.tokens.join(' · '):'—';
 document.getElementById('wujiRVQStatus').textContent=ready?`${holding?'暂停画面':'同帧推理'} · ${r.inference_ms.toFixed(2)} ms · Wuji2 20 关节 → RVQ → 重建与分类`:
  r?.status==='disabled'?'RVQ 已关闭':r?.error||sourceError||'等待 Wuji2 重定向结果';
 document.getElementById('wujiRVQOverlay').disabled=!ready;
}
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
   updateRVQPanel(null,false,'录制缺少对照拓扑');
   drawSignature=null;displayedFrameKey=null;displayedFrame=null;if(mano)mano.visible=false;robot.forEach(m=>m.visible=false);rvqRobot.forEach(m=>m.visible=false);dots.forEach(d=>d.visible=false);anchorDots.forEach(d=>d.visible=false);[...normals,...rays,...contactGraph,errors].forEach(l=>l.visible=false);
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
  const warp=['robokit_population_solver_warp_card','main_robokit_population_solver_window'].includes(w.solver?.backend),timing=w.solver?.timing_ms;
  const slsqp=!!backendOf(w)?.startsWith('slsqp'),graphSlsqp=slsqp;
  const selector=document.getElementById('wujiSolver');if(selector&&expected)selector.value=expected;
  if(canPresent&&window.wujiRequestedSolver===backendOf(w))window.wujiRequestedSolver=null;
  const shown=canPresent?w:holding?displayedFrame:null,shownBackend=backendOf(shown);
  const collisionChanged=updateSelfCollisionColors(shown?.collision?.self_mesh,canPresent&&!holding);
  updateRVQPanel(shown?.rvq,holding,w.error);
  const shownLabel=shownBackend==='cart'?'CART · 接触网络':shownBackend==='geort'?(shown.algorithm==='geort_contact_v1'?'GeoRT · 接触约束网络':'GeoRT · 官方网络'):shownBackend==='slsqp_graph'?'SLSQP · 离散接触图 + Laplacian':shownBackend==='wuji-retargeting'?'wuji-retargeting':shownBackend==='slsqp'?'SLSQP · 统一图':shownBackend==='cem'?'CEM · CARD':'未知后端';
  document.querySelector('.wuji-right-label').textContent='WUJI2.2 · '+(shown?shownLabel:`等待 ${(expected||'solver').toUpperCase()} 首帧`);
  document.getElementById('wujiFrameIdentity').textContent=shown?
   `画面后端：${shownLabel} · ${window.isMoxianReplay||shown.playback?.active?'录制回放':holding?'暂停画面':'实时输出'} · ${shown.algorithm||'旧录制未记录算法版本'} · 会话 ${shown.solver_session_id?.slice(0,8)||'—'} · 源帧 ${shown.source_timestamp_ms?.toFixed(0)||'—'} ms`:
   `画面后端：等待 ${(expected||'solver').toUpperCase()} 首帧 · 切换期间不显示其他后端的旧姿态`;
  const phase=slsqp&&Number.isFinite(timing?.slsqp)?` · SLSQP ${timing.slsqp.toFixed(1)} ms · 验收 ${timing.final_audit.toFixed(1)} ms`:Number.isFinite(timing?.robokit)&&Number.isFinite(timing?.cem)?` · 初解 ${timing.robokit.toFixed(1)} ms · CEM含验收 ${timing.cem.toFixed(1)} ms`:'';
  const failedCandidate=canPresent&&w.solution_feasible===false;
  const solverLabel=slsqp?(backendOf(w)==='slsqp_graph'?'Moxian → SLSQP · 离散接触图 + Laplacian':'Moxian → SLSQP'):`${warp?'Robokit → Warp CARD CEM':robokit?'Robokit → CEM':'CARD CEM'} 1024 × 16`;
  document.getElementById('wujiStatus').textContent=(w.error||(!state?.connected?'设备离线':failedCandidate?solverLabel+' · 显示本帧候选 · 约束超限'+(w.status==='solving'?' · 下一帧计算中':''):w.ready?`${solverLabel} · ${w.contact_anchors?.length||0} 个锚点 · ${w.contact_edges?.length||0} 条跨 link 边${phase} · 整帧 ${q.processing_ms?.toFixed(1)||'—'} ms · 源延迟 ${w.source_age_ms?.toFixed(0)||'—'} ms${w.status==='solving'?' · 下一帧计算中':''}`:'Wuji2 '+({mapping:'建立表面映射中',initializing:'初始化求解器',solving:w.input_live?'求解中':'输入暂停',infeasible:'显示本帧候选 · 约束超限',starting:'启动中'}[w.status]||'等待有效 Moxian mesh')))+(holding?' · 保留最后画面，接触状态未知':'');
  if(expected&&backendOf(w)!==expected)document.getElementById('wujiStatus').textContent=`切换至 ${expected.toUpperCase()}，等待该后端的新结果`;
  if(w.status==='buffering')document.getElementById('wujiStatus').textContent=w.message||`CARD 等待真实未来帧 · ${w.buffering_frames||0}/${w.required_frames||6}`;
  if(shownBackend==='cem')document.getElementById('wujiStatus').textContent+=cardWindowLabel(shown);
  if(slsqp)document.getElementById('wujiStatus').textContent+=` · 输出 ${q.output_hz?.toFixed(1)||'—'} Hz · 输入排队 ${q.input_queue_ms?.toFixed(1)||'—'} ms · 初值 ${w.solver?.initial_guess==='previous_raw_solution'?'上一原始解':w.solver?.initial_guess||'—'}`+(w.solver?.reset_reason?` · 重置 ${w.solver.reset_reason}`:'');
  if(slsqp&&w.solver?.execution_mode==='penetration_restoration')document.getElementById('wujiStatus').textContent+=w.solver.escape_continuation_active?' · 穿透恢复：沿已验收姿态分步退出':' · 穿透恢复：优先减小负间距';
  if(w.taxel_projection?.normal_rejected_sids?.length)document.getElementById('wujiStatus').textContent+=` · ${w.taxel_projection.normal_rejected_sids.length} 个SID法向不匹配，不参与接触图`;
  if(slsqp&&state?.contact?.mapping_confirmed===0)document.getElementById('wujiStatus').textContent+=' · SID分区尚未实物核验';
  const mesh=w.collision?.self_mesh;
  const seed=w.solver?.tactile_seed,pairs=w.self_contact_pairs||[];
  document.getElementById('wujiError').textContent=(q.contact_rmse_mm==null?'接触误差：—':`源接触点 RMS ${q.contact_rmse_mm.toFixed(1)} mm · 中心 ${q.patch_center_rmse_mm?.toFixed(1)||'—'} mm · 最大 ${q.contact_max_error_mm.toFixed(1)} mm · 法向 ${q.contact_normal_mean_deg.toFixed(1)}°`)+(seed?.applied?` · 触觉初解 ${seed.gap_before_mm.toFixed(1)} → ${seed.gap_after_mm.toFixed(1)} mm`:'')+(pairs.length?` · 最终对指点距 ${Math.max(...pairs.map(p=>p.distance_mm)).toFixed(1)} mm${w.contact_constraints_satisfied?'':'（未达闭合门槛）'}`:'')+(mesh?` · ${w.collision.mesh_role==='diagnostic_only'?'网格穿透诊断':'自穿透'} ${mesh.max_penetration_mm.toFixed(2)} / ${w.collision.mesh_role==='diagnostic_only'?w.collision.mesh_diagnostic_limit_mm:w.collision.self_hard_limit_mm} mm · ${w.collision.object_hard_gate_enabled===false?'推测物体侵入（软约束）':'物体穿透'} ${w.collision.object_max_penetration_mm.toFixed(2)} mm`:'')+` · ${q.joint_bounds_saturated??0} 关节到达限位 · SDF ${w.sdf?.type||'—'} ${w.sdf?.ready?'可用':'点集不足'} · 硬拒绝 ${w.solver?.hard_rejected??0}`+(q.cost_terms?' · '+Object.entries(q.cost_terms).map(([k,v])=>`${k} ${v.toFixed(1)}`).join(' / '):'');
  const thumbEdges=(w.contact_edges||[]).filter(e=>e.semantic_links?.includes('thumb_distal')&&e.semantic_links.some(l=>!l.startsWith('thumb')&&l.endsWith('_distal'))&&Array.isArray(e.source_vector_m)&&Array.isArray(e.robot_vector_m));
  const sourceError=q.source_quality?.position_rmse_mm;
  if(graphSlsqp){
   const mm=value=>Number.isFinite(value)?value.toFixed(2)+' mm':'—';
   document.getElementById('wujiError').textContent=`统一图 Laplacian RMS ${mm(q.unified_laplacian_rmse_mm)} · 接触子图 RMS ${mm(q.contact_laplacian_rmse_mm)}（诊断） · 边向量 RMS ${mm(q.contact_edge_rmse_mm)} · 自穿透 ${mm(mesh?.max_penetration_mm)} · 物体穿透 ${mm(w.collision?.object_max_penetration_mm)} · ${w.collision?.separation_satisfied?'零间距验收通过':'等待零间距验收'} · ${q.joint_bounds_saturated??0} 关节到达限位`+(q.cost_terms?' · '+Object.entries(q.cost_terms).map(([k,v])=>`${k} ${v.toFixed(3)}`).join(' / '):'');
  }
  if(thumbEdges.length){const nearest=thumbEdges.reduce((a,b)=>Math.hypot(...a.source_vector_m)<Math.hypot(...b.source_vector_m)?a:b);document.getElementById('wujiError').textContent+=` · 拇指最近跨指材料点距：源 ${(Math.hypot(...nearest.source_vector_m)*1000).toFixed(1)} / Wuji ${(Math.hypot(...nearest.robot_vector_m)*1000).toFixed(1)} mm`;}
  const cg=w.source_contact_geometry,verified=cg?.active_verified_sids?.length||0,unverified=cg?.active_unverified_sids?.length||0;
  const sourceRegistration=graphSlsqp?` · Moxian 原始表面材料绑定 · ${cg?.personalized_moxian?'已加载个体骨长':'默认骨长'}`:cg?` · 源接触位置已实测核验 ${verified}/${verified+unverified} SID`+(cg.personalized_mano?'':' · 个性化未验收')+(cg.sensor_mounting_status==='multi_pose_calibrated'?'':' · EMF 安装未验收'):' · 源接触位置未实测核验';
  document.getElementById('wujiSourceQuality').textContent='源 EMF 安装诊断残差：'+(Number.isFinite(sourceError)?`${sourceError.toFixed(1)} mm`:'不可用或原始配准未就绪')+sourceRegistration+` · 自接触 ${w.self_contact_pairs?.length||0} 对`+(w.self_contact_pairs?.length?` · 最大点距 ${Math.max(...w.self_contact_pairs.map(x=>x.distance_mm)).toFixed(1)} mm`:'')+` · 机器人虚拟投影 ${w.taxel_projection?.mapped_sids??count} SID`+` · 力权重为相对强度；实时未滤波`+(robokit?` · Robokit 本帧初解 ${w.solver.initializer.elapsed_ms.toFixed(1)} ms`:Number.isFinite(w.initializer_quality?.held_out_joint_mean_error_deg)?` · 拇指初值留出均差 ${w.initializer_quality.held_out_joint_mean_error_deg.toFixed(1)}°`:'');
  if(q.source_quality?.source_ik_solver)document.getElementById('wujiSourceQuality').textContent+=` · 源骨架 ${q.source_quality.source_ik_solver} ${q.source_quality.source_ik_processing_ms?.toFixed(2)||'—'} ms`;
  if(backendOf(w)==='wuji-retargeting'){
   document.getElementById('wujiStatus').textContent=w.error||`wuji-retargeting · ${w.ready?'实时输出':w.status==='solving'?'求解中':'等待标定骨架'} · 整帧 ${q.processing_ms?.toFixed(1)||'—'} ms · 输出 ${q.output_hz?.toFixed(1)||'—'} Hz`+(holding?' · 保留最后画面':'');
   document.getElementById('wujiError').textContent='当前已标定 Moxian 骨架 → 原版 Wuji2 重定向 · 原版低通 α=0.2 · 不使用触觉或碰撞约束';
   document.getElementById('wujiSourceQuality').textContent=`输入：21 个骨架节点 · ${w.solver?.first?'首帧初始化':'连续热启动'} · ${w.solver?.frames_computed??0} 帧`+(w.source_model_hash?` · 标定模型 ${w.source_model_hash.slice(0,12)}`:'');
  }
  if(!mano||!surface||!model||!(canPresent||holding)){window.updateCartHullOverlay?.(scenes[0],null,basis);window.updateCartHullOverlay?.(scenes[1],null,basis);if(mano)mano.visible=false;robot.forEach(m=>m.visible=false);rvqRobot.forEach(m=>m.visible=false);dots.forEach(d=>d.visible=false);anchorDots.forEach(d=>d.visible=false);[...normals,...rays,...contactGraph,errors].forEach(l=>l.visible=false);render();return;}
  if(holding)w=displayedFrame;
  if(['geort','cart'].includes(backendOf(w))){
   document.getElementById('wujiStatus').textContent=(w.error||`${backendOf(w)==='cart'?'CART':'GeoRT'} · ${w.model_id||'—'} · ${w.playback?.active?'动作回放':'实时网络'} · ${q.processing_ms?.toFixed(2)||'—'} ms`)+(holding?' · 保留最后画面':'');
   document.getElementById('wujiError').textContent=backendOf(w)==='cart'?'CART 接触损失训练 · 直接网络推理 · 当前接触是否达标见 CART 面板':w.algorithm==='geort_contact_v1'?'Glove反伸限位与原生网格接触损失训练 · 网络直接输出20个关节角 · 推理无追加求解 · 达标情况见表面诊断':'网络直接输出20个关节角 · 真实 FK 显示 · 未进行接触或碰撞约束优化';
   document.getElementById('wujiSourceQuality').textContent='Moxian 标定后指尖 → GeoRT · 无滤波';
  }
  const frameKey=[backendOf(w),w.algorithm,w.solver_session_id,w.mapping_version,w.timestamp_ms].join(':');
  const contactReady=!!w.contact_ready&&!holding;
  const signature=[frameKey,version,contactReady,window.cartHullVisible,...['wujiSid','wujiNormals','wujiRays','wujiErrors','wujiRVQOverlay'].map(id=>document.getElementById(id).type==='checkbox'?document.getElementById(id).checked:document.getElementById(id).value)].join(':');
  if(signature===drawSignature){if(activated||collisionChanged)render();return;}drawSignature=signature;
  mano.visible=true;robot.forEach(m=>m.visible=true);
  const curved=surface.update(w.source_vertices_m),attr=mano.geometry.attributes.position;
  for(let i=0;i<curved.length;i+=3)attr.setXYZ(i/3,curved[i+1],curved[i],-curved[i+2]);attr.needsUpdate=true;mano.geometry.computeVertexNormals();
  // B maps source world coordinates into the displayed Three.js coordinates.
  const B=new THREE.Matrix4().set(0,1,0,0,1,0,0,0,0,0,-1,0,0,0,0,1);
  robot.forEach(m=>{const i=m.userData.owner,r=w.link_rotations[i],p=w.link_positions_m[i];m.matrix.copy(B).multiply(new THREE.Matrix4().set(...r[0],p[0],...r[1],p[1],...r[2],p[2],0,0,0,1)).multiply(B);});
  const showRVQ=document.getElementById('wujiRVQOverlay').checked&&w.rvq?.ready&&vectors(w.rvq.link_positions_m)&&Array.isArray(w.rvq.link_rotations)&&w.rvq.link_rotations.length===w.rvq.link_positions_m.length&&w.rvq.link_rotations.every(r=>r.length===3&&vectors(r));
  rvqRobot.forEach(m=>{
   const i=m.userData.owner;m.visible=!!showRVQ;
   if(showRVQ){const r=w.rvq.link_rotations[i],p=w.rvq.link_positions_m[i];m.matrix.copy(B).multiply(new THREE.Matrix4().set(...r[0],p[0],...r[1],p[1],...r[2],p[2],0,0,0,1)).multiply(B);}
  });
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
  if(['geort','cart','wuji-retargeting'].includes(backendOf(w))){
   dots.forEach(d=>d.visible=false);
   [...normals,...rays,...contactGraph,errors].forEach(line=>line.visible=false);
  }
  window.updateCartHullOverlay?.(scenes[0],canPresent?w.cart_hull:null,basis);
  window.updateCartHullOverlay?.(scenes[1],canPresent?w.cart_robot_hull:null,basis);
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
for(const id of ['wujiRVQOverlay','wujiShowGraspType'])document.getElementById(id).checked=false;
syncPredictionVisibility();
for(const id of ['wujiSid','wujiNormals','wujiRays','wujiErrors','wujiRVQOverlay','wujiShowGraspType'])document.getElementById(id).addEventListener('input',()=>{syncPredictionVisibility();update(current,visible);render();});
window.addEventListener('resize',render);
if(window.currentMoxianState)update(window.currentMoxianState,document.getElementById('poseMode').value==='wuji2');
window.debugWujiView=()=>({rvq:displayedFrame?.rvq,rvqOverlay:rvqRobot.some(m=>m.visible),graspTypeVisible:!document.getElementById('wujiGraspPreview').hidden,collisionLinks:robot.filter(m=>m.userData.collisionColor==='#ed4545').map(m=>model?.link_names?.[m.userData.owner]),visible,version,loading,hasFrame:!!displayedFrame,frameKey:displayedFrameKey,solver_backend:backendOf(displayedFrame),algorithm:displayedFrame?.algorithm,requested_solver:window.wujiRequestedSolver,mano:!!mano?.visible,robot:robot.every(m=>m.visible),canvas:[canvas.width,canvas.height],error:window.wujiRenderError});
