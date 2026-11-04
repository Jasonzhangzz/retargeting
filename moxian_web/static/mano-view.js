import * as THREE from 'three';
import {OrbitControls} from '/vendor/OrbitControls.js';
const canvas=document.getElementById('manoCanvas');
let renderer,scene,camera,controls,mesh,lines,topology,version,loading=false,error=null,current=null,active=false,binding=false,targetMarkers=[],targetLinks,contactDots;
const basis=p=>[p[1],p[0],-p[2]];
function init(){
  try{
    renderer=new THREE.WebGLRenderer({canvas,antialias:true,alpha:false});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor('#29342d');
    scene=new THREE.Scene();camera=new THREE.PerspectiveCamera(38,1,.001,3);camera.position.set(.16,.13,.27);
    controls=new OrbitControls(camera,canvas);controls.enableDamping=false;controls.minDistance=.06;controls.maxDistance=.8;
    scene.add(new THREE.HemisphereLight('#fff4df','#526556',2.4));const light=new THREE.DirectionalLight('#fff1dc',2.1);light.position.set(.2,.4,.3);scene.add(light);
    const fill=new THREE.DirectionalLight('#a7bfce',1);fill.position.set(-.3,0,-.1);scene.add(fill);
    const grid=new THREE.GridHelper(.4,20,'#52634e','#39473d');grid.position.y=-.035;scene.add(grid);
    const dotCanvas=document.createElement('canvas');dotCanvas.width=dotCanvas.height=64;
    const dc=dotCanvas.getContext('2d'),gradient=dc.createRadialGradient(32,32,0,32,32,32);gradient.addColorStop(0,'rgba(255,255,255,1)');gradient.addColorStop(.25,'rgba(255,255,255,1)');gradient.addColorStop(.6,'rgba(255,255,255,.7)');gradient.addColorStop(1,'rgba(255,255,255,0)');dc.fillStyle=gradient;dc.fillRect(0,0,64,64);
    contactDots=new THREE.Points(new THREE.BufferGeometry(),new THREE.PointsMaterial({color:'#39ff76',size:.006,map:new THREE.CanvasTexture(dotCanvas),transparent:true,alphaTest:.02,depthTest:false,depthWrite:false}));contactDots.renderOrder=10;contactDots.frustumCulled=false;scene.add(contactDots);
    targetMarkers=Array.from({length:5},(_,i)=>{const marker=new THREE.Mesh(new THREE.SphereGeometry(.0028,12,8),new THREE.MeshBasicMaterial({color:[0xe5a476,0xa5be8b,0x91b5c6,0xd7b978,0xc3a3c7][i],wireframe:true,depthTest:false}));marker.renderOrder=5;marker.visible=false;scene.add(marker);return marker;});
    targetLinks=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineDashedMaterial({color:'#ffffff',dashSize:.002,gapSize:.0015,depthTest:false,transparent:true,opacity:.7}));targetLinks.renderOrder=4;scene.add(targetLinks);
    controls.addEventListener('change',render);
  }catch(e){error=e.message;window.manoRenderError=error;}
}
async function loadTopology(wanted){
  if(loading)return;loading=true;
  try{
    const t=window.replayingManoTopology||await (await fetch('/api/hand_model/model',{signal:AbortSignal.timeout(5000)})).json();
    if(!t.ready)return;
    topology=t;version=wanted;
    if(mesh){scene.remove(mesh);mesh.geometry.dispose();mesh.material.dispose();}
    const g=new THREE.BufferGeometry();g.setIndex(t.faces.flat());g.setAttribute('position',new THREE.Float32BufferAttribute(new Float32Array(778*3),3));
    mesh=new THREE.Mesh(g,new THREE.MeshStandardMaterial({color:'#d4ba96',roughness:.75,metalness:0,side:THREE.DoubleSide}));mesh.frustumCulled=false;scene.add(mesh);
    if(lines){scene.remove(lines);lines.geometry.dispose();lines.material.dispose();}
    lines=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:'#f7e9ca',depthTest:false,transparent:true,opacity:.7}));lines.renderOrder=3;scene.add(lines);
    if(current)update(current,true);
  }catch(e){window.manoRenderError=e.message;}finally{loading=false;}
}
function render(){
 if(!renderer||!active)return;
 const rect=canvas.getBoundingClientRect();if(!rect.width||!rect.height)return;
 renderer.setSize(rect.width,rect.height,false);camera.aspect=rect.width/rect.height;camera.updateProjectionMatrix();renderer.render(scene,camera);
}
function update(state,visible){
 active=visible;current=state;canvas.hidden=!visible;
 if(!renderer){init();if(error)return;}
 if(!visible)return;
 const m=state?.hand_model;
 if(mesh)mesh.visible=!!m?.ready;
 if(lines)lines.visible=!!m?.ready&&document.getElementById('manoWire').checked;
 targetMarkers.forEach(t=>t.visible=!!m?.ready&&!!m?.input_targets&&document.getElementById('showTipTargets').checked);targetLinks.visible=!!m?.ready&&!!m?.input_targets&&document.getElementById('showTipTargets').checked;
 contactDots.visible=!!m?.ready;contactDots.geometry.setAttribute('position',new THREE.Float32BufferAttribute((m?.ready?(m.contact_points||[]).filter(p=>p.state==='confirmed'):[]).flatMap(p=>basis(p.position_m)),3));
 if(!m?.ready){render();return;}
 const wanted=m.geometry_version+':'+m.binding_version+':'+m.coordinate_frame+':'+(m.sensor_mounting?.flat_reference_id||'unregistered');
 if(version!==wanted)loadTopology(wanted);
 if(!mesh||!topology)return;
 const positions=mesh.geometry.attributes.position;
 m.vertices_m.forEach((p,i)=>positions.setXYZ(i,...basis(p)));positions.needsUpdate=true;mesh.geometry.computeVertexNormals();
 if(!mesh.userData.centered){mesh.geometry.computeBoundingBox();const center=mesh.geometry.boundingBox.getCenter(new THREE.Vector3());controls.target.copy(center);camera.up.set(0,0,-1);camera.position.copy(center).add(new THREE.Vector3(.08,.27,.11));controls.update();mesh.userData.centered=true;}
 const linkPoints=[];
 if(m.input_targets&&Math.abs(m.input_targets.timestamp_ms-m.timestamp_ms)<=25)m.input_targets.positions_local.forEach((p,i)=>{targetMarkers[i].position.set(...basis(p));linkPoints.push(...basis(m.sensor_positions_m?.[i]||m.tip_positions_m?.[i]||m.vertices_m[[744,320,443,554,671][i]]),...basis(p));});
 targetLinks.geometry.setAttribute('position',new THREE.Float32BufferAttribute(linkPoints,3));targetLinks.computeLineDistances();
 const points=[];
 for(const chain of topology.joint_chains)for(let k=1;k<chain.length;k++)points.push(...basis(m.joints_m[chain[k-1]]),...basis(m.joints_m[chain[k]]));
 lines.geometry.setAttribute('position',new THREE.Float32BufferAttribute(points,3));lines.visible=document.getElementById('manoWire').checked;render();
}
window.updateManoView=update;
window.setManoView=mode=>{
 if(!controls)return;
 const v=mode==='top'?new THREE.Vector3(0,.30,.001):mode==='front'?new THREE.Vector3(0,.015,.30):new THREE.Vector3(.08,.27,.11);
 camera.position.copy(controls.target).add(v);camera.up.set(0,mode==='front'?1:0,mode==='front'?0:-1);controls.update();render();
};
window.startManoBinding=()=>{binding=true;canvas.classList.add('binding');};
canvas.addEventListener('click',e=>{
 if(!binding||!mesh||!topology)return;
 const box=canvas.getBoundingClientRect();const mouse=new THREE.Vector2((e.clientX-box.left)/box.width*2-1,-(e.clientY-box.top)/box.height*2+1);
 const ray=new THREE.Raycaster();ray.setFromCamera(mouse,camera);const hit=ray.intersectObject(mesh)[0];if(!hit)return;
 const index=hit.faceIndex;const ids=topology.faces[index],pos=mesh.geometry.attributes.position;
 const a=new THREE.Vector3().fromBufferAttribute(pos,ids[0]),b=new THREE.Vector3().fromBufferAttribute(pos,ids[1]),c=new THREE.Vector3().fromBufferAttribute(pos,ids[2]);
 const bary=new THREE.Vector3();THREE.Triangle.getBarycoord(hit.point,a,b,c,bary);
 binding=false;canvas.classList.remove('binding');window.bindManoPoint?.(index,bary.toArray());
});
document.getElementById('manoWire').addEventListener('change',()=>update(current,active));
window.addEventListener('resize',render);
if(window.currentMoxianState)update(window.currentMoxianState,document.getElementById('poseMode').value==='mano');

window.debugManoView=()=>({active,version,loading,hasMesh:!!mesh,targets:targetMarkers.filter(t=>t.visible).length,contactPoints:contactDots?.geometry.attributes.position?.count,contactColor:contactDots?.material.color.getHexString(),visible:mesh?.visible,positionCount:mesh?.geometry.attributes.position.count,draw:renderer?.info.render,box:mesh?.geometry.boundingBox,camera:camera?.position,target:controls?.target,canvas:[canvas.width,canvas.height],error});
