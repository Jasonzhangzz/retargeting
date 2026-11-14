const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');

function view(){
 const elements={};
 const element=id=>elements[id]??={checked:true,addEventListener(){},classList:{add(){},remove(){}}};
 class Attribute{constructor(values,size){this.count=typeof values==='number'?values/size:values.length/size;this.values=[];}setXYZ(i,...v){this.values[i]=v;}}
 class Geometry{constructor(){this.attributes={};}setAttribute(k,v){this.attributes[k]=v;}computeVertexNormals(){}}
 const context=vm.createContext({document:{getElementById:element},window:{addEventListener(){}},
  THREE:{Float32BufferAttribute:Attribute},Geometry,Float32Array,Number,performance:{now:()=>0},
  updateTaxels:(dots,vertices,topology,contact)=>{dots.userData={ready:!!contact?.ready};},
  fetch:async()=>({json:async()=>({ready:false})}),AbortSignal});
 vm.runInContext(fs.readFileSync(path.join(__dirname,'../moxian_web/static/mano-view.js'),'utf8').replace(/^import .*;\n/gm,''),context);
 vm.runInContext(`
  let renders=0;const originalRender=render;
  const object=()=>({visible:false,userData:{centered:true},geometry:new Geometry(),computeLineDistances(){}});
  renderer={};render=()=>{renders++;};mesh=object();mesh.geometry.setAttribute('position',new THREE.Float32BufferAttribute(6,3));
  lines=object();contactDots=object();targetMarkers=Array.from({length:5},object);targetLinks=object();
  topology={rest_vertices_m:[[0,0,0],[1,1,1]],joint_chains:[[0,1]]};
  surface={topology,update(vertices){this.positions=new Float32Array(vertices.flat());return this.positions;}};
  version='geometry:binding:mano_wrist_local:unregistered';
  globalThis.inspect=()=>({visible:mesh.visible,lines:lines.visible,empty:!document.getElementById('poseEmpty').hidden,
    contact:contactDots.userData?.ready,target:targetMarkers.some(t=>t.visible),
    position:mesh.geometry.attributes.position.values[0],status:document.getElementById('poseStatus').textContent,renders,geometryUpdates});
 `,context);
 const frame={ready:true,geometry_version:'geometry',binding_version:'binding',coordinate_frame:'mano_wrist_local',timestamp_ms:1000,
  vertices_m:[[0,0,0],[1,1,1]],joints_m:[[0,0,0],[1,1,1]],input_targets:{timestamp_ms:1000,positions_local:[]}};
 const update=(changes={})=>{context.window.updateManoView({hand_model:{...frame,...changes},contact:{ready:true}},true);return context.inspect();};
 return {update,run:source=>vm.runInContext(source,context)};
}

test('MANO short readiness flaps hold one labelled snapshot without blinking overlays or rebuilding geometry',()=>{
 const v=view(),first=v.update();assert.ok(first.visible&&first.lines&&first.contact&&first.target&&!first.empty);
 for(let i=0;i<10;i++){
  const held=v.update({ready:false,vertices_m:[],timestamp_ms:2000+i});
  assert.ok(held.visible&&held.lines&&held.contact&&held.target&&!held.empty);
  assert.deepEqual(held.position,first.position);assert.match(held.status,/历史叠层/);
  assert.equal(held.geometryUpdates,first.geometryUpdates);assert.equal(held.renders,first.renders);
  assert.ok(v.update().contact);
 }
 const next=v.update({timestamp_ms:3000,vertices_m:[[1,2,3],[4,5,6]]});
 assert.deepEqual(Array.from(next.position),[2,1,-3]);
});

test('invalid MANO candidates and pending topology changes cannot corrupt the previous frame',()=>{
 for(const changes of [{vertices_m:[]},{vertices_m:[[NaN,0,0],[0,0,0]]},{joints_m:[]},{geometry_version:'new'},{binding_version:'new'}]){
  const v=view(),first=v.update(),held=v.update(changes);
  assert.ok(held.visible&&!held.contact&&!held.empty);assert.deepEqual(held.position,first.position);
 }
 const v=view(),initial=v.update({ready:false});assert.ok(!initial.visible&&initial.empty);
});

test('MANO long stalls and worker errors clear historical overlays once, while new frames resume immediately',()=>{
 for(const changes of [{ready:false,age_ms:1001},{ready:false,error:'worker failed'}]){
  const v=view(),first=v.update(),held=v.update(changes);
  assert.ok(held.visible&&held.lines&&!held.contact&&!held.target&&!held.empty);
  assert.match(held.status,/接触状态未知/);assert.equal(held.geometryUpdates,first.geometryUpdates);
  assert.equal(v.update(changes).renders,held.renders);
  assert.ok(v.update({timestamp_ms:2000}).contact);
 }
});

test('MANO duplicate polls do not upload or render geometry; wire changes still redraw',()=>{
 const v=view(),a=v.update();for(let i=0;i<20;i++){const b=v.update();assert.equal(b.renders,a.renders);assert.equal(b.geometryUpdates,a.geometryUpdates);}
 v.run("document.getElementById('manoWire').checked=false");const b=v.update();assert.ok(!b.lines);assert.equal(b.renders,a.renders+1);assert.equal(b.geometryUpdates,a.geometryUpdates);
});

test('MANO same-size renders leave the WebGL buffer allocated',()=>{
 const v=view(),result=v.run(`
  let resizes=0,width=800;active=true;
  canvas.getBoundingClientRect=()=>({width,height:330});
  renderer={getPixelRatio:()=>1,setSize(){resizes++;},render(){}};camera={updateProjectionMatrix(){}};
  originalRender();originalRender();originalRender();const initial=resizes;width=900;originalRender();
  ({initial,total:resizes});
 `);
 assert.equal(result.initial,1);assert.equal(result.total,2);
});
