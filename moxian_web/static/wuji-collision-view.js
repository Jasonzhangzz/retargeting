import * as THREE from 'three';
import {OrbitControls} from '/vendor/OrbitControls.js';
import {basis,visualState,linkLabel} from './wuji-collision-colors.js';

const el=id=>document.getElementById(id),canvas=el('wujiCollisionCanvas');
const B=new THREE.Matrix4().set(0,1,0,0,1,0,0,0,0,0,-1,0,0,0,0,1);
let renderer,camera,controls,scenes,models=[[],[]],dots=[],asset,loading=false,data=null,key='',shape='',paint,retryAfter=0;
const raycaster=new THREE.Raycaster();raycaster.params.Points.threshold=.003;
const selectedPair=()=>el('wujiCollisionPair').value?el('wujiCollisionPair').value.split(':').map(Number):null;
const mm=v=>Number.isFinite(v)?v.toFixed(3):'—';
function init(){
 renderer=new THREE.WebGLRenderer({canvas,antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor('#243139');
 camera=new THREE.PerspectiveCamera(38,1,.001,3);camera.up.set(0,0,1);
 controls=new OrbitControls(camera,canvas);controls.enableDamping=false;controls.minDistance=.10;controls.maxDistance=.9;controls.addEventListener('change',render);
 scenes=[new THREE.Scene(),new THREE.Scene()];
 scenes.forEach(s=>{
  s.add(new THREE.HemisphereLight('#edf6ff','#53666e',2.5));
  const light=new THREE.DirectionalLight('#fff2df',2.5);light.position.set(.4,.2,.5);s.add(light);
  const axes=new THREE.AxesHelper(.025);s.add(axes);
  const points=new THREE.Points(new THREE.BufferGeometry(),new THREE.PointsMaterial({size:.003,vertexColors:true,depthTest:false,transparent:true,opacity:.95}));
  points.visible=false;points.renderOrder=10;s.add(points);dots.push(points);
 });
 reset();new ResizeObserver(()=>{if(canvas.clientWidth){apply();render();}}).observe(canvas.parentElement);
}
function reset(){if(!controls)return;controls.target.set(0,0,.075);camera.position.set(.33,.13,.19);controls.update();render();}
function render(){
 if(!renderer||!canvas.clientWidth)return;
 const box=canvas.getBoundingClientRect(),stacked=box.width<640,w=stacked?box.width:box.width/2,h=stacked?box.height/2:box.height;
 const next=[box.width,box.height,renderer.getPixelRatio()].join(':');if(next!==shape){renderer.setSize(box.width,box.height,false);shape=next;}
 canvas.parentElement.classList.toggle('stacked',stacked);camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setScissorTest(true);
 scenes.forEach((s,i)=>{const x=stacked?0:i*w,y=stacked?(1-i)*h:0;renderer.setViewport(x,y,w,h);renderer.setScissor(x,y,w,h);renderer.render(s,camera);});
 renderer.setScissorTest(false);
}
async function loadModel(wanted){
 if(loading)return;loading=true;
 try{
  const response=await fetch('/api/wuji_collision/model',{cache:'no-store'}),m=await response.json();
  if(!response.ok||!m.ready)throw Error(m.error||'等待原生手模型');
  if(m.asset_sha256!==wanted)throw Error('可视化资产与检测帧不匹配，请重新初始化碰撞检测');
  const disposed=new Set();models.forEach((meshes,s)=>meshes.forEach(mesh=>{scenes[s].remove(mesh);if(!disposed.has(mesh.geometry)){mesh.geometry.dispose();disposed.add(mesh.geometry);}mesh.material.dispose();}));models=[[],[]];
  m.meshes.forEach((link,owner)=>{
   const geometry=new THREE.BufferGeometry();geometry.setIndex(link.faces.flat());geometry.setAttribute('position',new THREE.Float32BufferAttribute(link.vertices_m.flatMap(basis),3));geometry.computeVertexNormals();
   scenes.forEach((s,i)=>{const mesh=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:'#b7c1c4',roughness:.65,side:THREE.DoubleSide,transparent:true}));
    mesh.matrixAutoUpdate=false;mesh.frustumCulled=false;mesh.userData={owner,name:link.name};s.add(mesh);models[i].push(mesh);});
  });asset=m.asset_sha256;key='';apply();render();
 }catch(error){el('wujiCollisionVisualStatus').textContent=error.message;retryAfter=Date.now()+1000;}
 finally{loading=false;}
}
function apply(){
 if(!renderer)return;
 const v=data?.ready?data.visualization:null;
 if(!v||v.asset_sha256!==asset){models.flat().forEach(m=>m.visible=false);dots.forEach(d=>d.visible=false);return;}
 const pair=selectedPair(),reference=el('wujiCollisionReference').value||'matched';paint=visualState(data,reference,pair);
 el('wujiCollisionMeshOnlyLabel').textContent=reference==='dense'?'仅密集网格检出 · 含采样差异':'仅网格检出 · MLP 漏检';
 el('wujiCollisionNeuralOnlyLabel').textContent=reference==='dense'?'仅 MLP 检出 · 含采样差异':'仅 MLP 检出 · 误报';
 models.forEach((meshes,s)=>meshes.forEach(mesh=>{
  const i=mesh.userData.owner,r=v.link_rotations[i],p=v.link_positions_m[i];
  mesh.visible=s===0||!!data.neural;mesh.matrix.copy(B).multiply(new THREE.Matrix4().set(...r[0],p[0],...r[1],p[1],...r[2],p[2],0,0,0,1)).multiply(B);
  mesh.material.color.set(paint.colors[s][i]);mesh.material.opacity=pair&&!pair.includes(i)?.13:1;
  mesh.material.depthWrite=mesh.material.opacity===1;mesh.material.wireframe=el('wujiCollisionWireframe').checked;
 }));
 dots.forEach((dot,s)=>{
  const ids=paint.points[s],positions=ids.flatMap(i=>basis(v.witnesses.positions_m[i]));
  const colors=paint.pointColors[s].flatMap(color=>{const c=new THREE.Color(color);return [c.r,c.g,c.b];});
  dot.geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));dot.geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));dot.geometry.computeBoundingSphere();
  dot.visible=el('wujiCollisionPoints').checked&&!!ids.length;dot.userData.indices=ids;
 });
 el('wujiCollisionLeftLabel').textContent=`真实网格 · ${reference==='dense'?'密集':'同点'} · ${mm((reference==='dense'?data.dense_mesh:data.matched_mesh)?.max_penetration_mm)} mm`;
 el('wujiCollisionRightLabel').textContent=data.neural?`Neural SDF · ${mm(data.neural.max_penetration_mm)} mm`:'Neural SDF · 不可用';
 el('wujiCollisionVisualStatus').textContent=`同一姿态、同一尺度、同步视角 · ${v.witnesses.displayed_queries} 个穿透查询点（按 link 对与误报／漏报类型保留最深点）${reference==='dense'?' · 点位置来自同点采样':''} · 拖动旋转，滚轮缩放，悬停查看距离`;
}
function update(next){
 data=next;
 if(!canvas.clientWidth)return;
 try{
  if(!renderer)init();
  const k=[next?.ready,next?.context_key,next?.timestamp_ms,next?.example?.pose_id,next?.request_id].join(':');
  if(!next?.ready||!next.visualization){
   apply();render();el('wujiCollisionVisualStatus').textContent='等待有效姿态；可点击“测试姿态”查看独立样本';return;
  }
  if(asset!==next.visualization.asset_sha256){if(Date.now()>=retryAfter)void loadModel(next.visualization.asset_sha256);return;}
  if(k!==key){key=k;apply();render();}
 }catch(error){el('wujiCollisionVisualStatus').textContent='三维视图：'+error.message;}
}
window.updateWujiCollisionView=update;
window.focusWujiCollisionPair=(a,b)=>{el('wujiCollisionPair').value=[Math.min(a,b),Math.max(a,b)].join(':');apply();render();};
for(const id of ['wujiCollisionReference','wujiCollisionPair','wujiCollisionPoints','wujiCollisionWireframe'])el(id).onchange=()=>{apply();render();};
el('wujiCollisionResetView').onclick=reset;
el('wujiCollisionPanel').addEventListener('toggle',()=>update(data));
canvas.addEventListener('pointerleave',()=>{el('wujiCollisionTooltip').hidden=true;});
canvas.addEventListener('pointermove',event=>{
 const tooltip=el('wujiCollisionTooltip');tooltip.hidden=true;if(!renderer||!data?.ready||!paint)return;
 const box=canvas.getBoundingClientRect(),x=event.clientX-box.left,y=event.clientY-box.top,stacked=box.width<640;
 const side=stacked?(y<box.height/2?0:1):(x<box.width/2?0:1),w=stacked?box.width:box.width/2,h=stacked?box.height/2:box.height;
 const px=stacked?x:x-side*w,py=stacked?y-side*h:y;
 raycaster.setFromCamera(new THREE.Vector2(px/w*2-1,-py/h*2+1),camera);
 const hit=dots[side].visible?raycaster.intersectObject(dots[side])[0]:null;
 if(hit){
  const i=dots[side].userData.indices[hit.index],v=data.visualization.witnesses;
  tooltip.textContent=`${linkLabel(models[0][v.source_links[i]].userData.name)} → ${linkLabel(models[0][v.target_links[i]].userData.name)}\nSDF：网格 ${mm(v.mesh_sdf_mm[i])} mm / MLP ${mm(v.neural_sdf_mm?.[i])} mm（负值在内部）`;
 }else{
  const hit=raycaster.intersectObjects(models[side].filter(m=>m.visible))[0];if(!hit)return;
  const i=hit.object.userData.owner;tooltip.textContent=`${linkLabel(hit.object.userData.name)} · 最大穿透 ${mm(paint.depths[side][i])} mm`;
 }
 tooltip.style.left=Math.min(x+12,Math.max(0,box.width-280))+'px';tooltip.style.top=Math.max(32,y-48)+'px';tooltip.hidden=false;
});
if(window.currentWujiCollisionVisual)update(window.currentWujiCollisionVisual);
