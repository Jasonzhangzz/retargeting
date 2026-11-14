"""Local Three.js replay of source MANO/object, IK physics and optimized physics."""
from pathlib import Path
import json,shutil
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def export_viewer(model,motion,qpos,baseline,reference,output,max_frames=600,device='cuda:0'):
    out=Path(output);n=len(qpos);ids=np.unique(np.linspace(0,n-1,min(n,max_frames)).astype(int))
    chosen=[];seen=set();meshes=[]
    # Render each rigid visual mesh once, despite duplicate collision geoms.
    for g in range(model.ngeom):
        if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
        key=(int(model.geom_bodyid[g]),int(model.geom_dataid[g]),*model.geom_pos[g],*model.geom_quat[g])
        if key in seen:continue
        seen.add(key);chosen.append(g);mesh=int(model.geom_dataid[g])
        va,vn=int(model.mesh_vertadr[mesh]),int(model.mesh_vertnum[mesh]);fa,fn=int(model.mesh_faceadr[mesh]),int(model.mesh_facenum[mesh])
        meshes.append(dict(vertices=model.mesh_vert[va:va+vn].tolist(),faces=model.mesh_face[fa:fa+fn].tolist(),object=g==model.geom('tracked_object_geom').id))
    from robokit.sim import MjWarpSim
    import warp as wp
    frame_sim=MjWarpSim(model,len(ids),mujoco.MjData(model),nconmax=32,njmax=256,device=device)
    def frames(values):
        with wp.ScopedDevice(device),wp.ScopedStream(frame_sim.stream):
            frame_sim.data.qpos.assign(np.asarray(values[ids],np.float32));frame_sim.kinematics()
            wp.synchronize_stream(frame_sim.stream)
            pos=frame_sim.data.geom_xpos.numpy()[:,chosen]
            matrices=frame_sim.data.geom_xmat.numpy()[:,chosen]
        quat=Rotation.from_matrix(matrices.reshape(-1,3,3)).as_quat().reshape(len(ids),len(chosen),4)
        return np.concatenate([pos,quat],axis=-1).round(7).tolist()
    packet=dict(meshes=meshes,optimized=frames(qpos),baseline=frames(baseline),reference=frames(reference),
                human_vertices=motion['mano_vertices'][1:len(qpos)+1][ids].round(6).tolist(),human_faces=motion['mano_faces'].tolist(),
                time_s=motion['time_s'][1:len(qpos)+1][ids].tolist(),frame_ids=motion['frame_ids'][1:len(qpos)+1][ids].tolist())
    (out/'replay.json').write_text(json.dumps(packet,separators=(',',':')))
    vendor=Path(__file__).resolve().parents[2]/'moxian_web/static/vendor'
    shutil.copy(vendor/'three.module.min.js',out/'three.module.min.js')
    shutil.copy(vendor/'OrbitControls.js',out/'OrbitControls.js')
    html='''<!doctype html><meta charset="utf-8"><title>HO-3D → Wuji2 · DIAL-MPC</title>
<style>body{margin:0;background:#15201e;color:#e9eee7;font:14px system-ui}header{padding:16px 24px;background:#21312b;display:flex;gap:20px;align-items:center;flex-wrap:wrap}header strong{margin-right:auto}label{cursor:pointer}button{padding:7px 16px;cursor:pointer}#scene{width:100%;height:75vh}footer{padding:14px 24px;line-height:1.8;color:#b2c0b8}input[type=range]{width:min(55vw,700px)}#status{font:13px monospace}</style>
<script type="importmap">{"imports":{"three":"./three.module.min.js"}}</script>
<header><strong>HO-3D → Wuji2 · DIAL-MPC</strong><label><input id="human" type="checkbox" checked>青：MANO 参考</label><label><input id="ghost" type="checkbox" checked>橙：参考物体</label><label><input id="robot" type="checkbox" checked>紫：优化结果</label><label><input id="base" type="checkbox">灰：IK 控制物理回放</label></header>
<div id="scene"></div><footer><button id="play">暂停</button> <input id="seek" type="range" min="0" max="1" value="0"> <span id="status">加载回放…</span><br>鼠标拖动旋转，滚轮缩放。物体轨迹与接触由物理仿真产生；灰色网格仅为视觉参照，不代表存在支撑桌面。<br>仿真使用原始物体网格的凸包碰撞；质量、摩擦和相机重力方向为明确配置假设。<a href="result.json" style="color:#b7d7b8">查看指标及参数</a></footer>
<script type="module">
import * as THREE from 'three';import {OrbitControls} from './OrbitControls.js';
const el=id=>document.getElementById(id),data=await(await fetch('./replay.json')).json();
const scene=new THREE.Scene();scene.background=new THREE.Color('#15201e');const camera=new THREE.PerspectiveCamera(42,1,.001,20);camera.up.set(0,0,1);
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));el('scene').append(renderer.domElement);
scene.add(new THREE.HemisphereLight(0xffffff,0x505853,2));const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(1,-1,2);scene.add(light);
const controls=new OrbitControls(camera,renderer.domElement);const object=data.optimized[0][data.meshes.findIndex(m=>m.object)];controls.target.set(...object.slice(0,3));camera.position.copy(controls.target).add(new THREE.Vector3(.4,-.5,.35));controls.update();
const grid=new THREE.GridHelper(2,40,0x607267,0x2c3e34);grid.rotation.x=Math.PI/2;scene.add(grid);
function group(key,color,opacity){const g=new THREE.Group();scene.add(g);const meshes=data.meshes.map(m=>{const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(m.vertices.flat(),3));geo.setIndex(m.faces.flat());geo.computeVertexNormals();const material=new THREE.MeshStandardMaterial({color:m.object&&key==='optimized'?0xe8af4d:color,transparent:opacity<1,opacity,roughness:.65,depthWrite:opacity===1});const mesh=new THREE.Mesh(geo,material);g.add(mesh);if(key==='reference'&&!m.object)mesh.visible=false;return mesh;});return {g,meshes,key};}
const groups=[group('optimized',0xac8de1,1),group('baseline',0x9aa7a0,.35),group('reference',0xed9852,.25)];
const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(data.human_vertices[0].flat(),3));geo.setIndex(data.human_faces.flat());geo.computeVertexNormals();const human=new THREE.Mesh(geo,new THREE.MeshStandardMaterial({color:0x62d8ce,transparent:true,opacity:.24,side:THREE.DoubleSide,depthWrite:false}));scene.add(human);
let playing=true,frame=0,last=performance.now(),elapsed=0;el('seek').max=data.time_s.length-1;
el('play').onclick=()=>{playing=!playing;el('play').textContent=playing?'暂停':'播放';last=performance.now();};el('seek').oninput=()=>{frame=+el('seek').value;elapsed=data.time_s[frame]-data.time_s[0];draw();};
for(const id of ['human','ghost','robot','base'])el(id).onchange=draw;
function draw(){for(const row of groups){row.g.visible=el({optimized:'robot',baseline:'base',reference:'ghost'}[row.key]).checked;row.meshes.forEach((m,i)=>{const v=data[row.key][frame][i];m.position.set(...v.slice(0,3));m.quaternion.set(...v.slice(3));});}human.visible=el('human').checked;geo.attributes.position.array.set(data.human_vertices[frame].flat());geo.attributes.position.needsUpdate=true;geo.computeVertexNormals();el('seek').value=frame;el('status').textContent=`帧 ${data.frame_ids[frame]} · ${data.time_s[frame].toFixed(2)} s`;}
const observer=new ResizeObserver(()=>{const b=el('scene').getBoundingClientRect();renderer.setSize(b.width,b.height);camera.aspect=b.width/b.height;camera.updateProjectionMatrix();});observer.observe(el('scene'));
function animate(now){requestAnimationFrame(animate);if(playing){elapsed+=(now-last)/1000;const duration=data.time_s.at(-1)-data.time_s[0];if(elapsed>duration+.5){elapsed=0;frame=0;}while(frame<data.time_s.length-1&&data.time_s[frame+1]-data.time_s[0]<=elapsed)frame++;draw();}last=now;controls.update();renderer.render(scene,camera);}draw();requestAnimationFrame(animate);
</script>'''
    (out/'index.html').write_text(html)
