import * as THREE from 'three';
import {OrbitControls} from '/vendor/OrbitControls.js';
import {workspaceFrames} from './geort-workspace-fk.js';

// Read-only native robot preview; live solver/capture state is never changed.
const canvas=document.getElementById('geortWorkspacePoseCanvas');
const basis=p=>[p[1],p[0],-p[2]],B=new THREE.Matrix4().set(0,1,0,0,1,0,0,0,0,0,-1,0,0,0,0,1);
let renderer,scene,camera,controls,meshes=[],marker,version,topology,epoch=0;
function render(){
 if(!renderer||!canvas.clientWidth||!canvas.clientHeight)return;
 renderer.setSize(canvas.clientWidth,canvas.clientHeight,false);camera.aspect=canvas.clientWidth/canvas.clientHeight;
 camera.updateProjectionMatrix();renderer.render(scene,camera);
}
function init(){
 renderer=new THREE.WebGLRenderer({canvas,antialias:true});renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));renderer.setClearColor('#29342d');
 scene=new THREE.Scene();camera=new THREE.PerspectiveCamera(38,1,.001,3);
 controls=new OrbitControls(camera,canvas);controls.minDistance=.07;controls.maxDistance=1;controls.addEventListener('change',render);
 scene.add(new THREE.HemisphereLight('#fff4df','#526556',2.4));
 const light=new THREE.DirectionalLight('#fff1dc',2);light.position.set(.2,.4,.3);scene.add(light);
 marker=new THREE.Mesh(new THREE.SphereGeometry(.003,16,12),new THREE.MeshBasicMaterial({color:'#ffd075',depthTest:false}));
 marker.renderOrder=5;scene.add(marker);marker.visible=false;
}
function clear(){
 ++epoch;meshes.forEach(m=>{m.visible=false;});if(marker)marker.visible=false;
 window.geortWorkspacePreviewStats={visible:false};render();
}
window.clearGeortWorkspacePose=clear;
window.showGeortWorkspacePose=async pose=>{
 const token=++epoch;
 if(!renderer)init();
 if(version!==pose.reference_id){
  const response=await fetch('/api/geort/workspace/model?'+new URLSearchParams({reference_id:pose.reference_id}),{signal:AbortSignal.timeout(10000)}),t=await response.json();
  if(token!==epoch)return;
  if(!response.ok||!t.ready)throw Error(t.error||'参考模型尚未就绪');
  const next=[];
  for(let owner=0;owner<t.link_names.length;owner++){
   const ids=t.vertex_owners.map((v,i)=>v===owner?i:-1).filter(i=>i>=0);if(!ids.length)continue;
   const start=ids[0],vertices=t.local_vertices_m.slice(start,start+ids.length);
   const g=new THREE.BufferGeometry();g.setIndex(t.faces.filter(f=>t.vertex_owners[f[0]]===owner).flatMap(f=>f.map(i=>i-start)));
   g.setAttribute('position',new THREE.Float32BufferAttribute(vertices.flatMap(basis),3));g.computeVertexNormals();
   const m=new THREE.Mesh(g,new THREE.MeshStandardMaterial({color:'#aeb8b7',roughness:.7,side:THREE.DoubleSide}));m.matrixAutoUpdate=false;m.frustumCulled=false;m.userData.owner=owner;next.push(m);
  }
  meshes.forEach(m=>{scene.remove(m);m.geometry.dispose();m.material.dispose();});meshes=next;meshes.forEach(m=>scene.add(m));version=pose.reference_id;topology=t;
  controls.target.set(...basis([0,0,.065]));camera.up.set(...basis([0,0,1]));camera.position.copy(controls.target).add(new THREE.Vector3(...basis([.33,-.12,.10])));controls.update();
 }
 if(token!==epoch)return;
 const frames=workspaceFrames(topology.kinematics,pose.joint_positions_rad);
 if(frames.link_positions_m.length!==topology.link_names.length)throw Error('姿态与Wuji2模型不匹配');
 const finger=['thumb','index','middle','ring','pinky'].indexOf(pose.finger),start=1+finger*4;
 if(Math.hypot(...frames.keypoints_m[finger].map((v,i)=>v-pose.point_mm[i]/1000))>1e-6)throw Error('参考pose的指尖与散点坐标不一致');
 meshes.forEach(m=>{
  const i=m.userData.owner,p=frames.link_positions_m[i],r=frames.link_rotations[i];
  if(![...p,...r.flat()].every(Number.isFinite))throw Error('参考姿态包含无效几何');
  const matrix=new THREE.Matrix4().set(r[0][0],r[0][1],r[0][2],p[0],r[1][0],r[1][1],r[1][2],p[1],r[2][0],r[2][1],r[2][2],p[2],0,0,0,1);
  m.matrix.copy(B).multiply(matrix).multiply(B);m.matrixWorldNeedsUpdate=true;m.visible=true;
  m.material.color.set(i>=start&&i<start+4?'#dfb876':'#aeb8b7');
 });
 marker.position.set(...basis(pose.point_mm.map(v=>v/1000)));marker.visible=true;render();
 window.geortWorkspacePreviewStats={visible:true,sample_id:pose.sample_id,reference_id:pose.reference_id,finger:pose.finger,point_mm:pose.point_mm,joint_positions_rad:pose.joint_positions_rad,links:meshes.length,cell_observed:pose.cell_observed,solver_independent:true,rendering:'on_click_fk',cached_pose_only:true};
};
window.addEventListener('resize',render);
