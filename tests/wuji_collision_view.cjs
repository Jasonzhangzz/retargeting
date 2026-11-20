/* Execute scene updates with real Three.js math and a stubbed GPU renderer. */
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),{pathToFileURL}=require('node:url');
function element(){return {value:'',checked:false,hidden:false,textContent:'',style:{},children:[],classList:{toggle(){}},
 clientWidth:1000,addEventListener(){},getBoundingClientRect(){return {left:0,top:0,width:1000,height:440};}};}
(async()=>{
 const T=await import(pathToFileURL(path.resolve('moxian_web/static/vendor/three.module.min.js')));
 const nodes=new Map([...fs.readFileSync('moxian_web/static/index.html','utf8').matchAll(/id="([^"]+)"/g)].map(m=>[m[1],element()]));
 const canvas=nodes.get('wujiCollisionCanvas');canvas.parentElement=element();nodes.get('wujiCollisionReference').value='matched';nodes.get('wujiCollisionPoints').checked=true;
 const renders=[];
 class Renderer{
  constructor(){this.ratio=1;}setPixelRatio(v){this.ratio=v;}getPixelRatio(){return this.ratio;}setClearColor(){}setSize(){}setScissorTest(){}setViewport(){}setScissor(){}
  render(scene,camera){scene.updateMatrixWorld(true);renders.push({scene,camera});}
 }
 class Controls{constructor(camera){this.target=new T.Vector3();this.camera=camera;}addEventListener(){}update(){this.camera.lookAt(this.target);}}
 globalThis.__collisionThree={...T,WebGLRenderer:Renderer};globalThis.__collisionControls=Controls;
 globalThis.document={getElementById:id=>nodes.get(id)};globalThis.window={};globalThis.devicePixelRatio=1;
 globalThis.ResizeObserver=class{observe(){}};
 const geometry={ready:true,asset_sha256:'test',meshes:Array.from({length:21},(_,i)=>({name:i?'index_link1':'palm',vertices_m:[[0,0,0],[.005,0,0],[0,.005,0]],faces:[[0,1,2]]}))};
 globalThis.fetch=async()=>({ok:true,json:async()=>geometry});
 let source=fs.readFileSync('moxian_web/static/wuji-collision-view.js','utf8');
 source=source.replace("import * as THREE from 'three';","const THREE=globalThis.__collisionThree;")
  .replace("import {OrbitControls} from '/vendor/OrbitControls.js';","const OrbitControls=globalThis.__collisionControls;")
  .replace("'./wuji-collision-colors.js'",JSON.stringify(pathToFileURL(path.resolve('moxian_web/static/wuji-collision-colors.js')).href));
 await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
 const d={ready:true,request_id:1,neural:{query_valid:true,max_penetration_mm:2},matched_mesh:{max_penetration_mm:2},
 pairs:[{a:4,b:8,dense_mm:2,matched_mm:2,neural_mm:0}],
 visualization:{asset_sha256:'test',link_positions_m:Array.from({length:21},(_,i)=>[i*.01,0,0]),link_rotations:Array.from({length:21},()=>[[1,0,0],[0,1,0],[0,0,1]]),
 witnesses:{positions_m:[[.04,.01,.02]],source_links:[4],target_links:[8],mesh_sdf_mm:[-2],neural_sdf_mm:[1],displayed_queries:1}}};
 window.updateWujiCollisionView(d);await new Promise(resolve=>setImmediate(resolve));
 let scenes=[...new Set(renders.map(r=>r.scene))];assert.equal(scenes.length,2);
 const meshes=s=>s.children.filter(o=>o.isMesh),points=s=>s.children.find(o=>o.isPoints);
 assert.equal(meshes(scenes[0]).length,21);assert.equal(meshes(scenes[1]).length,21);
 assert.equal(meshes(scenes[0])[4].material.color.getHexString(),'58bffa');
 assert.equal(meshes(scenes[1])[4].material.color.getHexString(),'b7c1c4');
 assert.deepEqual(meshes(scenes[0])[4].matrix.elements.slice(12,15),[0,.04,0]);
 assert.deepEqual(meshes(scenes[0])[4].matrix.elements,meshes(scenes[1])[4].matrix.elements);
 assert.equal(points(scenes[0]).geometry.attributes.position.count,1);assert.equal(points(scenes[1]).visible,false);
 assert(renders.every(r=>r.camera===renders[0].camera)); // One synchronized orbit camera.
 window.updateWujiCollisionView({ready:false});
 assert(scenes.every(s=>meshes(s).every(m=>!m.visible)&&!points(s).visible));
 console.log('Collision 3D scenes: real matrix math, synchronized camera, collision colors/points and stale clearing passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
