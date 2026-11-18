import * as THREE from 'three';
import {OrbitControls} from '/vendor/OrbitControls.js';
import {createTaxels,updateTaxels} from './taxel-view.js';

const canvas=document.getElementById('manoCanvas');
let renderer,scene,camera,controls,mesh,lines,topology,version,loading=false,error=null,current=null,active=false,binding=false,targetMarkers=[],targetLinks,contactDots;
let surface;
let predictionSnapshot=null,predictionReceivedAt=0;
let displayedModel,displayedContact,receivedAt=0,frameKey=null,drawSignature=null,geometryUpdates=0,renderSize='';
const modelVersion=m=>m.geometry_version+':'+m.binding_version+':'+m.coordinate_frame+':'+(m.sensor_mounting?.flat_reference_id||'unregistered');
const finitePoints=a=>ArrayBuffer.isView(a)?a.length>0&&a.length%3===0&&a.every(Number.isFinite):Array.isArray(a)&&a.length>0&&a.every(p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite));
function completeModel(m,t){
 const count=t?.rest_vertices_m?.length||t?.vertices_m?.length;
 return finitePoints(m?.vertices_m)&&finitePoints(m?.joints_m)&&(!count||m.vertices_m.length===(ArrayBuffer.isView(m.vertices_m)?count*3:count))&&
  (!t||t.joint_chains.every(chain=>chain.every(i=>i<m.joints_m.length)));
}
const basis=p=>[p[1],p[0],-p[2]];
function palmView(){
 const rotate=p=>{const r=current?.hand_model?.root_rotation;return r?p.map((_,i)=>r[i].reduce((s,v,k)=>s+v*p[k],0)):p;};
 const normal=new THREE.Vector3(...basis(rotate(topology?.palm_view?.normal||[0,-1,0]))).normalize();
 const up=new THREE.Vector3(...basis(rotate(topology?.palm_view?.forward||[1,0,0]))).normalize();
 return {normal,up,side:new THREE.Vector3().crossVectors(up,normal).normalize()};
}
function init(){
  try{
    renderer=new THREE.WebGLRenderer({canvas,antialias:true,alpha:false});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor('#29342d');
    scene=new THREE.Scene();camera=new THREE.PerspectiveCamera(38,1,.001,3);camera.position.set(.16,.13,.27);
    controls=new OrbitControls(camera,canvas);controls.enableDamping=false;controls.minDistance=.06;controls.maxDistance=.8;
    scene.add(new THREE.HemisphereLight('#fff4df','#526556',2.4));const light=new THREE.DirectionalLight('#fff1dc',2.1);light.position.set(.2,.4,.3);scene.add(light);
    const fill=new THREE.DirectionalLight('#ece3d4',1);fill.position.set(-.3,0,-.1);scene.add(fill);
    const grid=new THREE.GridHelper(.4,20,'#52634e','#39473d');grid.position.y=-.035;
    // A reference backdrop, not a floor that slices the moving palm. Its depth
    // must never compete with the skin or material contact points.
    grid.material.depthTest=false;grid.material.depthWrite=false;grid.renderOrder=-10;scene.add(grid);
    contactDots=createTaxels();scene.add(contactDots);
    targetMarkers=Array.from({length:5},(_,i)=>{const marker=new THREE.Mesh(new THREE.SphereGeometry(.0028,12,8),new THREE.MeshBasicMaterial({color:[0xe5a476,0xa5be8b,0x91b5c6,0xd7b978,0xc3a3c7][i],wireframe:true,depthTest:false}));marker.renderOrder=5;marker.visible=false;scene.add(marker);return marker;});
    targetLinks=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineDashedMaterial({color:'#ffffff',dashSize:.002,gapSize:.0015,depthTest:false,transparent:true,opacity:.7}));targetLinks.renderOrder=4;scene.add(targetLinks);
    controls.addEventListener('change',render);
  }catch(e){error=e.message;window.manoRenderError=error;}
}
async function loadTopology(wanted){
  if(loading)return;loading=true;
  try{
    const t=window.replayingManoTopology||await (await fetch('/api/hand_model/model',{signal:AbortSignal.timeout(5000)})).json();
    if(!t.ready||!current?.hand_model?.ready||!completeModel(current.hand_model,t)||modelVersion(current.hand_model)!==wanted||t.geometry_version!==current.hand_model.geometry_version||t.binding_version!==current.hand_model.binding_version)return;
    const centered=!!mesh?.userData.centered;
    const nextSurface={faces:t.faces,topology:t,positions:new Float32Array((t.rest_vertices_m?.length||t.vertices_m?.length||778)*3),dispose(){},mapHit(face,barycentric){return {face,barycentric};},update(vertices){if(ArrayBuffer.isView(vertices))this.positions.set(vertices);else this.positions.set(vertices.flat());return this.positions;},parent:t.faces.map((_,i)=>i)};
    const g=new THREE.BufferGeometry();g.setIndex(nextSurface.faces.flat());g.setAttribute('position',new THREE.Float32BufferAttribute(nextSurface.positions.length,3));
    const nextMesh=new THREE.Mesh(g,new THREE.MeshStandardMaterial({color:'#d4ba96',roughness:.75,metalness:0,side:THREE.DoubleSide,polygonOffset:true,polygonOffsetFactor:1,polygonOffsetUnits:1}));nextMesh.frustumCulled=false;nextMesh.userData.centered=centered;
    const nextLines=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:'#f7e9ca',depthTest:false,transparent:true,opacity:.7}));nextLines.renderOrder=3;
    if(mesh){scene.remove(mesh);mesh.geometry.dispose();mesh.material.dispose();}
    surface?.dispose();surface=nextSurface;mesh=nextMesh;scene.add(mesh);topology=t;version=wanted;
    if(lines){scene.remove(lines);lines.geometry.dispose();lines.material.dispose();}
    lines=nextLines;scene.add(lines);
    displayedModel=null;predictionSnapshot=null;frameKey=null;drawSignature=null;if(current)update(current,active);
  }catch(e){window.manoRenderError=e.message;}finally{loading=false;}
}
function render(){
 if(!renderer||!active)return;
 const rect=canvas.getBoundingClientRect();if(!rect.width||!rect.height)return;
 const size=[rect.width,rect.height,renderer.getPixelRatio()].join(':');
 if(size!==renderSize){renderer.setSize(rect.width,rect.height,false);renderSize=size;}
 camera.aspect=rect.width/rect.height;camera.updateProjectionMatrix();renderer.render(scene,camera);
}
function update(state,visible){
 const activated=visible&&!active;active=visible;current=state;if(canvas.hidden===visible)canvas.hidden=!visible;
 if(!renderer){init();if(error)return;}
 if(!visible)return;
 let m=state?.hand_model;
 const cartFrame=state?.cart?.preview_frame;
 if(window.cartHullVisible&&state.cart.ready&&cartFrame?.geometry_version===m?.geometry_version){m={...m,...cartFrame,mesh_frame:null,input_targets:null};}
 const wanted=m&&modelVersion(m);
 if(m?.ready&&completeModel(m)&&version!==wanted)loadTopology(wanted);
 const canPresent=!!m?.ready&&version===wanted&&!!topology&&completeModel(m,topology);
 const holding=!canPresent&&!!displayedModel;
 const now=performance.now();
 const key=canPresent?version+':'+(m.timestamp_ms??JSON.stringify(m.vertices_m)):frameKey;
 const newFrame=canPresent&&key!==frameKey;
 if(newFrame){receivedAt=now;displayedContact=state.contact;}
 else if(canPresent)displayedContact=state.contact;
 // Readiness has a 250 ms cutoff, shorter than some worker frames. A short
 // delay holds one complete visual snapshot instead of blinking overlays off
 // and on. Its status explicitly identifies historical data. At one second,
 // or on an actual worker error, contact/target overlays become unknown once.
 const age=Math.max(now-receivedAt,Number.isFinite(state?.hand_model?.age_ms)?state.hand_model.age_ms:0);
 const snapshot=holding&&!m?.ready&&wanted===version&&age<1000&&!state?.hand_model?.error&&['default','calibrated','preview',undefined].includes(state?.hand_model?.status);
 if(mesh)mesh.visible=canPresent||holding;
 if(lines)lines.visible=(canPresent||holding)&&document.getElementById('manoWire').checked;
 const shown=holding?displayedModel:m;
 const showTargets=(canPresent||snapshot)&&!!shown?.input_targets&&Math.abs(shown.input_targets.timestamp_ms-shown.timestamp_ms)<=25&&document.getElementById('showTipTargets').checked;
 targetMarkers.forEach(t=>t.visible=showTargets);targetLinks.visible=showTargets;
 contactDots.visible=canPresent||holding;
 // A delayed worker result must not erase the last finite surface. Contacts and
 // input target overlays are explicitly unknown until a fresh frame arrives.
 document.getElementById('poseEmpty').hidden=canPresent||holding;
 window.manoViewHasFrame=canPresent||holding;
 if(holding){m=displayedModel;document.getElementById('poseStatus').textContent=snapshot?`等待新姿态 · 保留上帧及历史叠层（${Math.round(age)} ms）`:(state?.hand_model?.error||'等待新姿态')+' · 保留最后画面，接触状态未知';}
 if(!(canPresent||holding)){window.updateCartHullOverlay?.(scene,null,basis);render();return;}
 if(!mesh||!topology)return;
 const contact=(canPresent||snapshot)?displayedContact:null;
 const prediction=state?.contact_prediction;
 const predictionEnabled=!!document.getElementById('predictionOverlay')?.checked;
 if(predictionEnabled&&prediction?.ready===true&&prediction.vertex_probability?.length===778&&prediction.hand_side===m.hand_side&&prediction.binding_version===m.binding_version&&Math.abs(prediction.timestamp_ms-m.timestamp_ms)<=250&&prediction.timestamp_ms!==predictionSnapshot?.timestamp_ms){
  const previous=predictionSnapshot?.binding_version===prediction.binding_version?predictionSnapshot.vertex_probability:null;
  predictionSnapshot={...prediction,vertex_probability:prediction.vertex_probability.map((p,i)=>{p=Number.isFinite(p)?Math.max(0,Math.min(1,p)):0;return previous?previous[i]*.7+p*.3:p;})};predictionReceivedAt=now;
 }
 const overlay=predictionEnabled&&predictionSnapshot?.binding_version===m.binding_version&&now-predictionReceivedAt<500;
 const heat=overlay?predictionSnapshot:null;
 window.updateCartHullOverlay?.(scene,canPresent&&m.cart_hull?.timestamp_ms===m.timestamp_ms?m.cart_hull:null,basis);
 const signature=JSON.stringify([key,window.cartHullVisible,contact?.ready===true,contact?.ready===true?contact.cell_contact:null,document.getElementById('manoWire').checked,showTargets,overlay,overlay?heat.timestamp_ms:null]);
 if(signature===drawSignature){if(activated)render();return;}
 drawSignature=signature;
 const positions=mesh.geometry.attributes.position;
 if(mesh.material){
  if(overlay){
   if(!mesh.geometry.attributes.color)mesh.geometry.setAttribute('color',new THREE.Float32BufferAttribute(778*3,3));
   const colors=mesh.geometry.attributes.color;
   const neutral=new THREE.Color('#d4ba96'),warm=new THREE.Color('#e6a23c');
   heat.vertex_probability.forEach((p,i)=>colors.setXYZ(i,neutral.r+(warm.r-neutral.r)*p,neutral.g+(warm.g-neutral.g)*p,neutral.b+(warm.b-neutral.b)*p));colors.needsUpdate=true;
  }
  if(mesh.material.vertexColors!==overlay){mesh.material.vertexColors=overlay;mesh.material.color.set(overlay?'#ffffff':'#d4ba96');mesh.material.needsUpdate=true;}
 }
 const curved=newFrame?surface.update(m.vertices_m):surface.positions;
 if(newFrame){for(let i=0;i<curved.length;i+=3)positions.setXYZ(i/3,curved[i+1],curved[i],-curved[i+2]);positions.needsUpdate=true;mesh.geometry.computeVertexNormals();geometryUpdates++;}
 updateTaxels(contactDots,curved,surface.topology,contact,basis);
 if(!mesh.userData.centered){mesh.geometry.computeBoundingBox();const center=mesh.geometry.boundingBox.getCenter(new THREE.Vector3());controls.target.copy(center);const view=palmView();camera.up.copy(view.up);camera.position.copy(center).addScaledVector(view.normal,.42).addScaledVector(view.side,.08).addScaledVector(view.up,.03);controls.update();mesh.userData.centered=true;}
 const linkPoints=[];
 if(showTargets)m.input_targets.positions_local.forEach((p,i)=>{targetMarkers[i].position.set(...basis(p));linkPoints.push(...basis(m.sensor_positions_m?.[i]||m.tip_positions_m?.[i]||m.vertices_m[[744,320,443,554,671][i]]),...basis(p));});
 targetLinks.geometry.setAttribute('position',new THREE.Float32BufferAttribute(linkPoints,3));targetLinks.computeLineDistances();
 if(newFrame){const points=[];for(const chain of topology.joint_chains)for(let k=1;k<chain.length;k++)points.push(...basis(m.joints_m[chain[k-1]]),...basis(m.joints_m[chain[k]]));lines.geometry.setAttribute('position',new THREE.Float32BufferAttribute(points,3));}
 displayedModel=m;frameKey=key;render();
}
window.updateManoView=update;
window.setManoView=mode=>{
 if(!controls)return;
 const view=palmView(),v=view.normal.clone().multiplyScalar(.42);
 if(mode==='top')v.addScaledVector(view.up,.16);
 else if(mode!=='front')v.addScaledVector(view.side,.08).addScaledVector(view.up,.03);
 camera.position.copy(controls.target).add(v);camera.up.copy(view.up);controls.update();render();
};
window.startManoBinding=()=>{binding=true;canvas.classList.add('binding');};
canvas.addEventListener('click',e=>{
 if(!binding||!mesh||!topology)return;
 const box=canvas.getBoundingClientRect();const mouse=new THREE.Vector2((e.clientX-box.left)/box.width*2-1,-(e.clientY-box.top)/box.height*2+1);
 const ray=new THREE.Raycaster();ray.setFromCamera(mouse,camera);const hit=ray.intersectObject(mesh)[0];if(!hit)return;
 const index=hit.faceIndex;const ids=surface.faces[index],pos=mesh.geometry.attributes.position;
 const a=new THREE.Vector3().fromBufferAttribute(pos,ids[0]),b=new THREE.Vector3().fromBufferAttribute(pos,ids[1]),c=new THREE.Vector3().fromBufferAttribute(pos,ids[2]);
 const bary=new THREE.Vector3();THREE.Triangle.getBarycoord(hit.point,a,b,c,bary);
 const material=surface.mapHit(index,bary.toArray());
 binding=false;canvas.classList.remove('binding');window.bindManoPoint?.(material.face,material.barycentric);
});
document.getElementById('manoWire').addEventListener('change',()=>update(current,active));
window.addEventListener('resize',render);
if(window.currentMoxianState)update(window.currentMoxianState,document.getElementById('poseMode').value==='mano');

window.debugManoView=()=>({active,version,loading,frameKey,geometryUpdates,hasMesh:!!mesh,surface:surface?.stats,targets:targetMarkers.filter(t=>t.visible).length,taxels:contactDots?.userData,taxelSphereCount:contactDots?.contactSpheres?.count||0,meshColor:mesh?.material?.color?.getHexString?.(),predictionOverlay:mesh?.material?.vertexColors===true,taxelPositions:contactDots?.geometry.attributes.position?Array.from(contactDots.geometry.attributes.position.array):[],taxelColors:contactDots?.geometry.attributes.color?Array.from(contactDots.geometry.attributes.color.array):[],visible:mesh?.visible,positionCount:mesh?.geometry.attributes.position.count,draw:renderer?.info.render,box:mesh?.geometry.boundingBox,camera:camera?.position,target:controls?.target,canvas:[canvas.width,canvas.height],error});
window.debugCurvedSurface=()=>({positions:surface?.positions,topology:surface?.topology,source:surface?.source,recipes:surface?.recipes,mapHit:(f,b)=>surface.mapHit(f,b),stats:surface?.stats});
