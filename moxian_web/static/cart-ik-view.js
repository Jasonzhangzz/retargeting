import * as THREE from 'three';
import {OrbitControls} from '/vendor/OrbitControls.js';
import {CurvedSurface} from './curved-surface.js';

const el=id=>document.getElementById(id),canvas=el('cartIKCanvas'),timeline=el('cartIKTimeline');
const basis=p=>[p[1],p[0],-p[2]],B=new THREE.Matrix4().set(0,1,0,0,1,0,0,0,0,0,-1,0,0,0,0,1);
let renderer,camera,controls,scenes,source,surface,robot=[],trails=[],topology,dataset=null;
let frame=null,wanted=0,pending=false,playing=false,scrubbing=false,anchor=0,offset=0,generation=0,lastSize='';
function init(){
 renderer=new THREE.WebGLRenderer({canvas,antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor('#29342d');
 camera=new THREE.PerspectiveCamera(38,1,.001,3);controls=new OrbitControls(camera,canvas);controls.minDistance=.05;controls.maxDistance=1;
 controls.addEventListener('change',render);scenes=[new THREE.Scene(),new THREE.Scene()];
 scenes.forEach(scene=>{scene.add(new THREE.HemisphereLight('#fff4df','#526556',2.4));const light=new THREE.DirectionalLight('#fff1dc',2);light.position.set(.2,.4,.3);scene.add(light);});
}
function disposeModels(){
 if(source){scenes[0].remove(source);source.geometry.dispose();source.material.dispose();source=null;}
 for(const mesh of robot){scenes[1].remove(mesh);mesh.geometry.dispose();mesh.material.dispose();}
 robot=[];surface?.dispose();
 trails.forEach((line,i)=>{scenes[i].remove(line);line.geometry.dispose();line.material.dispose();});trails=[];
}
function makeTrails(values,segments){
 const coords=[];
 for(let i=1;i<values.length;i++){
  if(!values[i]||!values[i-1]||topology.timestamps_ms[i]-topology.timestamps_ms[i-1]>250||
     (segments&&segments[i]!==segments[i-1]))continue;
  for(let f=0;f<5;f++)coords.push(...basis(values[i-1][f]),...basis(values[i][f]));
 }
 const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(coords,3));
 return new THREE.LineSegments(geometry,new THREE.LineBasicMaterial({color:'#c9b781',transparent:true,opacity:.22,depthWrite:false}));
}
function makeModels(t){
 if(!renderer)init();disposeModels();topology=t;
 surface=new CurvedSurface(t.source);const geometry=new THREE.BufferGeometry();geometry.setIndex(surface.faces.flat());
 geometry.setAttribute('position',new THREE.Float32BufferAttribute(surface.positions.length,3));
 source=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:'#d4ba96',roughness:.8,side:THREE.DoubleSide}));source.frustumCulled=false;scenes[0].add(source);
 const model=t.robot;
 model.link_names.forEach((_,owner)=>{
  const ids=model.vertex_owners.map((v,i)=>v===owner?i:-1).filter(i=>i>=0);if(!ids.length)return;
  const start=ids[0],local=model.local_vertices_m.slice(start,start+ids.length),g=new THREE.BufferGeometry();
  g.setIndex(model.faces.filter(f=>model.vertex_owners[f[0]]===owner).flatMap(f=>f.map(i=>i-start)));
  g.setAttribute('position',new THREE.Float32BufferAttribute(local.flatMap(basis),3));g.computeVertexNormals();
  const solid=new THREE.Mesh(g,new THREE.MeshStandardMaterial({color:'#b9c5be',roughness:.68,side:THREE.DoubleSide}));
  solid.matrixAutoUpdate=false;solid.userData.owner=owner;solid.frustumCulled=false;solid.renderOrder=2;robot.push(solid);scenes[1].add(solid);

 });
 trails=[makeTrails(t.source_tip_trajectories_m),makeTrails(t.robot_tip_trajectories_m,t.segments)];trails.forEach((line,i)=>scenes[i].add(line));
 controls.target.set(0,0,-.075);camera.up.set(0,0,-1);camera.position.set(.13,.38,-.015);controls.update();
}
function geometryPacket(packet){
 const descriptor=packet.geometry_f32;if(descriptor?.format!=='cart-ik-f32-v1')throw Error('IK 几何传输版本不匹配');
 const raw=atob(descriptor.data),bytes=Uint8Array.from(raw,c=>c.charCodeAt(0)),arrays={};
 for(const [name,{offset,shape}] of Object.entries(descriptor.fields)){
  const count=shape.reduce((a,b)=>a*b,1);
  if(offset<0||offset%4||offset+count*4>bytes.length)throw Error('IK 几何帧损坏');
  arrays[name]={values:new Float32Array(bytes.buffer,offset,count),shape};
 }
 return arrays;
}
function poseMatrix(arrays,pose,owner){
 const r=arrays.link_rotations.values,p=arrays.link_positions_m.values,ri=(pose*21+owner)*9,pi=(pose*21+owner)*3;
 return new THREE.Matrix4().copy(B).multiply(new THREE.Matrix4().set(r[ri],r[ri+1],r[ri+2],p[pi],r[ri+3],r[ri+4],r[ri+5],p[pi+1],r[ri+6],r[ri+7],r[ri+8],p[pi+2],0,0,0,1)).multiply(B);
}
function present(packet){
 const arrays=geometryPacket(packet),vertices=arrays.source_vertices_m.values;
 const controls=[];for(let i=0;i<vertices.length;i+=3)controls.push(Array.from(vertices.subarray(i,i+3)));
 surface.update(controls);const values=surface.positions,positions=source.geometry.attributes.position;
 for(let i=0;i<values.length;i+=3)positions.setXYZ(i/3,values[i+1],values[i],-values[i+2]);
 positions.needsUpdate=true;source.geometry.computeVertexNormals();source.visible=true;
 robot.forEach(mesh=>{mesh.visible=packet.selected;if(packet.selected)mesh.matrix.copy(poseMatrix(arrays,0,mesh.userData.owner));});
 frame=packet;if(!scrubbing)el('cartIKSeek').value=packet.index;
 el('cartIKTime').textContent=`${packet.seconds.toFixed(2)} / ${topology.duration_s.toFixed(2)} s · 帧 ${packet.index+1} / ${topology.frames} · 原始时间 ${packet.timestamp_ms.toFixed(1)} ms`;
 el('cartIKStatus').textContent=`${topology.label} · ${packet.selected?'缓存 IK 状态':'本帧 IK 无效'}${packet.domain_limited?' · 已限制到 CART 关节域':''}${packet.solver_fallback?' · 原版求解器回退':''} · 手套与机器人同帧回放`;
 drawTimeline();render();window.cartIKViewError=null;
}
function render(){
 if(!renderer||!dataset||el('cartIKStage').hidden)return;
 let box=canvas.getBoundingClientRect();if(!box.width||!box.height)return;
 const stacked=box.width<600;canvas.parentElement.classList.toggle('stacked',stacked);box=canvas.getBoundingClientRect();
 const size=[box.width,box.height,renderer.getPixelRatio()].join(':');if(size!==lastSize){renderer.setSize(box.width,box.height,false);lastSize=size;}
 const w=stacked?box.width:box.width/2,h=stacked?box.height/2:box.height;
 camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setScissorTest(true);
 trails.forEach(line=>line.visible=el('cartIKTrails').checked);
 scenes.forEach((scene,i)=>{const x=stacked?0:i*w,y=stacked?(1-i)*h:0;renderer.setViewport(x,y,w,h);renderer.setScissor(x,y,w,h);renderer.render(scene,camera);});
 renderer.setScissorTest(false);
}
function drawTimeline(){
 if(!topology)return;const width=timeline.clientWidth;if(!width)return;timeline.width=Math.round(width*devicePixelRatio);timeline.height=15*devicePixelRatio;
 const ctx=timeline.getContext('2d'),n=topology.frames;
 for(let i=0;i<n;i++){ctx.fillStyle=topology.available[i]?'#8bb59c':'#d4a579';const x=i/n*timeline.width;ctx.fillRect(x,0,Math.ceil(timeline.width/n),timeline.height);}
 if(frame){ctx.fillStyle='#254e47';ctx.fillRect(frame.index/Math.max(1,n-1)*(timeline.width-2),0,2*devicePixelRatio,timeline.height);}
}
async function json(url){const response=await fetch(url,{cache:'no-store',signal:AbortSignal.timeout(50000)});const data=await response.json();if(!response.ok||data.error)throw Error(data.error||'读取 IK 失败');return data;}
async function seek(index,force=false){
 if(!dataset||!topology)return;wanted=Math.max(0,Math.min(topology.frames-1,Math.round(index)));if(pending)return;
 pending=true;const current=generation,key=dataset;
 try{
  while(dataset===key&&generation===current&&(force||frame?.index!==wanted)){
   force=false;
   const requested=wanted,packet=await json(`/api/cart/ik/viewer/frame?id=${encodeURIComponent(key)}&index=${requested}`);
   if(dataset!==key||generation!==current)break;
   // Playback may outrun mesh reconstruction. Present completed frames while
   // playing; a paused scrub still discards responses for obsolete targets.
   if(requested===wanted||playing)present(packet);
  }
 }catch(error){if(generation===current){setPlaying(false);el('cartIKStatus').textContent=error.message;window.cartIKViewError=error.message;}}
 finally{pending=false;if(generation!==current&&dataset&&topology)void seek(wanted);}
}
function setPlaying(value){
 playing=value;el('cartIKPlay').textContent=value?'暂停':'播放';
 if(value){offset=topology.timestamps_ms[frame?.index||0]-topology.timestamps_ms[0];anchor=performance.now();requestAnimationFrame(tick);}
}
function tick(){
 if(!playing||!dataset)return;
 const elapsed=offset+(performance.now()-anchor)*Number(el('cartIKSpeed').value),stamps=topology.timestamps_ms,start=stamps[0];
 let lo=0,hi=stamps.length;
 while(lo<hi){const mid=(lo+hi)>>1;if(stamps[mid]-start<=elapsed)lo=mid+1;else hi=mid;}
 void seek(Math.max(0,lo-1));
 if(elapsed>=topology.duration_s*1000){setPlaying(false);return;}
 requestAnimationFrame(tick);
}
el('cartIKLoad').onclick=async()=>{
 const ids=Array.from(el('cartDatasets').selectedOptions).map(o=>o.value);
 if(ids.length!==1){el('cartIKStatus').textContent='请只选择一段已准备 IK 的录制会话。';return;}
 setPlaying(false);const current=++generation;dataset=null;topology=null;frame=null;wanted=0;
 el('cartIKLoad').disabled=true;el('cartIKStatus').textContent='加载已保存的手套与 IK…';el('cartIKStage').hidden=true;
 try{
  const t=await json('/api/cart/ik/viewer/topology?id='+encodeURIComponent(ids[0]));if(generation!==current)return;
  makeModels(t);dataset=ids[0];el('cartIKSeek').max=t.frames-1;el('cartIKStage').hidden=false;
  await seek(Math.max(0,t.available.findIndex(Boolean)));
 }catch(error){el('cartIKStatus').textContent=error.message;window.cartIKViewError=error.message;}
 finally{el('cartIKLoad').disabled=false;}
};
el('cartIKPlay').onclick=()=>{if(!dataset)return;if(!playing&&frame?.index===topology.frames-1){frame=null;wanted=0;void seek(0);offset=0;setPlaying(true);}else setPlaying(!playing);};
el('cartIKValid').onclick=()=>{
 if(!dataset||!topology)return;setPlaying(false);
 const start=(frame?.index??-1)+1;
 let index=topology.available.findIndex((ok,i)=>ok&&i>=start);
 if(index<0)index=topology.available.findIndex(Boolean);
 if(index<0){el('cartIKStatus').textContent=`此缓存没有有效 IK 帧。`;return;}
 void seek(index);
};
el('cartIKClose').onclick=()=>{setPlaying(false);generation++;dataset=null;frame=null;topology=null;el('cartIKStage').hidden=true;el('cartIKStatus').textContent='IK 回放已关闭。';void fetch('/api/cart/ik/viewer/close').catch(()=>{});};
el('cartIKSeek').oninput=()=>{setPlaying(false);void seek(Number(el('cartIKSeek').value));};
el('cartIKSeek').addEventListener('pointerdown',()=>{scrubbing=true;setPlaying(false);});
window.addEventListener('pointerup',()=>{scrubbing=false;});
window.addEventListener('pointercancel',()=>{scrubbing=false;});
el('cartIKSpeed').onchange=()=>{if(playing)setPlaying(true);};
el('cartIKTrails').onchange=render;
window.addEventListener('resize',()=>{render();drawTimeline();});
el('cartPanel').addEventListener('toggle',()=>{if(!el('cartPanel').open)setPlaying(false);else{render();drawTimeline();}});
window.debugCartIKView=()=>({dataset,index:frame?.index,wanted,playing,pending,selected:frame?.selected,
 source_visible:!!source?.visible,robot_visible:robot.some(mesh=>mesh.visible),frames:topology?.frames,
 source_timestamp_ms:frame?.source_timestamp_ms,robot_timestamp_ms:frame?.robot_timestamp_ms,error:window.cartIKViewError});
