import * as THREE from 'three';
import {OrbitControls} from '/vendor/OrbitControls.js';
import './cart-overlay.js?v=collection-preview-v1';

// Source collection geometry has its own viewer, independent of robot inference.
const canvas=document.getElementById('cartPreviewCanvas'),status=document.getElementById('cartPreviewStatus');
const basis=p=>[p[1],p[0],-p[2]];
let renderer,scene,camera,controls,mesh,topology,version,frameKey,current,pending=false,lastFetch=0;
function render(){
 if(!renderer||!canvas.clientWidth||!canvas.clientHeight)return;
 renderer.setSize(canvas.clientWidth,canvas.clientHeight,false);camera.aspect=canvas.clientWidth/canvas.clientHeight;
 camera.updateProjectionMatrix();renderer.render(scene,camera);
}
function init(){
 renderer=new THREE.WebGLRenderer({canvas,antialias:true});renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));renderer.setClearColor('#29342d');
 scene=new THREE.Scene();camera=new THREE.PerspectiveCamera(38,1,.001,3);
 controls=new OrbitControls(camera,canvas);controls.minDistance=.05;controls.maxDistance=1;controls.addEventListener('change',render);
 scene.add(new THREE.HemisphereLight('#fff4df','#526556',2.4));
 const light=new THREE.DirectionalLight('#fff1dc',2);light.position.set(.2,.4,.3);scene.add(light);
}
function hide(message){
 if(window.cartPreviewStats)window.cartPreviewStats.visible=false;
 if(mesh)mesh.visible=false;if(scene)window.updateCartHullOverlay?.(scene,null,basis);
 status.textContent=message;render();
}
async function updateFrame(frame,hull){
 if(!renderer)init();
 const wanted=frame.geometry_version+':'+frame.binding_version;
 if(version!==wanted){
  const response=await fetch('/api/cart/preview/topology',{signal:AbortSignal.timeout(5000)}),t=await response.json();
  if(!response.ok||!t.ready||t.geometry_version!==frame.geometry_version||t.binding_version!==frame.binding_version)throw Error('等待同一标定的 CART 预览拓扑');
  if(mesh){scene.remove(mesh);mesh.geometry.dispose();mesh.material.dispose();}
  topology=t;version=wanted;frameKey=null;
  const geometry=new THREE.BufferGeometry();geometry.setIndex(t.faces.flat());
  geometry.setAttribute('position',new THREE.Float32BufferAttribute((t.rest_vertices_m||[]).length*3,3));
  const material=new THREE.MeshStandardMaterial({color:'#ffffff',roughness:.8,side:THREE.DoubleSide,vertexColors:true});
  mesh=new THREE.Mesh(geometry,material);mesh.frustumCulled=false;scene.add(mesh);
 }
 if(!current?.cart?.ready||window.cartHullVisible===false)return hide('接触预览已暂停或输入失效');
 if(hull?.timestamp_ms!==frame.timestamp_ms)throw Error('CART 接触目标与网格时间戳不同步');
 if(frameKey!==frame.timestamp_ms){
  const count=frame.vertices_m.length,position=mesh.geometry.attributes.position;
  if(position.count!==count)throw Error('CART 预览顶点数与标定拓扑不符');
  frame.vertices_m.forEach((p,i)=>position.setXYZ(i,...basis(p)));position.needsUpdate=true;mesh.geometry.computeVertexNormals();
  const color=new THREE.Color('#d4ba96'),highlight=new THREE.Color('#4fcb9e'),values=new Float32Array(count*3);
  for(let i=0;i<count;i++)values.set([color.r,color.g,color.b],i*3);
  for(const patch of hull?.required_patches||[])for(const face of topology.regions[patch]||[])for(const i of topology.faces[face])values.set([highlight.r,highlight.g,highlight.b],i*3);
  if(!mesh.geometry.attributes.color)mesh.geometry.setAttribute('color',new THREE.Float32BufferAttribute(values,3));
  else{mesh.geometry.attributes.color.array.set(values);mesh.geometry.attributes.color.needsUpdate=true;}
  if(frameKey===null){
   mesh.geometry.computeBoundingBox();const center=mesh.geometry.boundingBox.getCenter(new THREE.Vector3());
   controls.target.copy(center);camera.up.set(...basis(topology.palm_view.forward));
   camera.position.copy(center).addScaledVector(new THREE.Vector3(...basis(topology.palm_view.normal)).normalize(),.38);controls.update();
  }
  frameKey=frame.timestamp_ms;
 }
 let shown=hull;
 if(!hull?.ready&&hull?.diagnostic_candidate)shown={...hull.diagnostic_candidate,preview_only:true,timestamp_ms:hull.timestamp_ms,geometry_version:hull.geometry_version};
 mesh.visible=true;window.updateCartHullOverlay?.(scene,shown,basis);render();
 window.cartPreviewRenderError=null;
 status.textContent=hull?.contact_mode==='hand'?'源手套 · 绿色为指定真实表面区域，亮点与连线为当前最近目标；完全闭合时显示真实手部接触。':hull?.ready?'源手套 · primitive + delta 已通过全部关键手指接触及碰撞检查。':hull?.diagnostic_candidate?'源手套 · 橙色 primitive 为诊断候选，不作为有效训练目标。':hull?.status==='low_confidence'?'源手套 · 几何置信度低，已放弃拟合。':'源手套 · 当前没有通过检查的 primitive，请让拇指参与对向夹持或包围。';
 const primitive=shown?.primitive;if(primitive)status.textContent+=` type ${primitive.reference_type} · ${primitive.kind} + delta · 几何置信度 ${(primitive.confidence*100).toFixed(0)}% · ${hull.processing_ms?.toFixed(1)||'—'} ms`;

 window.cartPreviewStats={visible:true,timestamp_ms:frame.timestamp_ms,vertices:frame.vertices_m.length,contact_mode:hull?.contact_mode,hull_visible:!!shown?.ready,diagnostic:!!shown?.preview_only,solver_independent:true};
}
async function refresh(){
 if(pending)return;
 const c=current?.cart;
 if(!c?.ready||window.cartHullVisible===false||!document.getElementById('cartPanel').open){
  if(window.cartPreviewStats)window.cartPreviewStats.visible=false;
  return hide(c?.error|| (c?.recording?.active?'接触预览初始化或等待有效几何；原始动作持续录制。':'点击“预览当前接触目标”或开始录制，显示独立三维预览。'));
 }
 if(Date.now()-lastFetch<200)return;lastFetch=Date.now();pending=true;
 try{
  if(c.preview_frame)await updateFrame(c.preview_frame,c.hull);
  else{
   const response=await fetch('/api/cart/preview',{cache:'no-store',signal:AbortSignal.timeout(5000)}),p=await response.json();
   if(!response.ok||!p.ready)throw Error(p.error||'等待有效 CART 接触预览');
   const raw=atob(p.mesh_f32.data),bytes=Uint8Array.from(raw,c=>c.charCodeAt(0)),values=new Float32Array(bytes.buffer);
   if(values.length!==(p.mesh_f32.vertices+p.mesh_f32.joints)*3)throw Error('CART 预览数据长度无效');
   const vertices=Array.from({length:p.mesh_f32.vertices},(_,i)=>Array.from(values.subarray(i*3,i*3+3)));
   await updateFrame({...p,vertices_m:vertices},p.hull);
  }
 }catch(error){hide(error.message);window.cartPreviewRenderError=error.message;}
 finally{pending=false;}
}
window.updateCartPreview=state=>{current=state;void refresh();};
window.addEventListener('resize',render);
if(window.currentMoxianState)window.updateCartPreview(window.currentMoxianState);
