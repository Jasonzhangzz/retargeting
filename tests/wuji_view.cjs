const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');

function view(){
 const elements={};
 const element=id=>elements[id]??={value:'0',checked:true,type:id==='wujiSid'?'number':'checkbox',addEventListener(){}};
 class Attribute{constructor(values,size){this.count=typeof values==='number'?values/size:values.length/size;this.values=[];}setXYZ(i,...values){this.values[i]=values;}}
 class Geometry{constructor(){this.attributes={};}setAttribute(name,attribute){this.attributes[name]=attribute;}computeBoundingSphere(){}computeVertexNormals(){}}
 class Matrix{set(){return this;}copy(){return this;}multiply(){return this;}}
 class Color{constructor(color){this.r=color==='yellow'?1:0;this.g=0;this.b=0;}}
 const context=vm.createContext({document:{getElementById:element,querySelector:element},window:{addEventListener(){}},
  THREE:{Float32BufferAttribute:Attribute,Matrix4:Matrix,Color},TAXEL_PURPLE:'purple',TAXEL_YELLOW:'yellow',
  performance:{now:()=>0},Date,console,atob,fetch:async()=>({json:async()=>({ready:false})}),THREEGeometry:Geometry});
 const source=fs.readFileSync(path.join(__dirname,'../moxian_web/static/wuji-view.js'),'utf8').replace(/^import .*;\n/gm,'');
 vm.runInContext(source,context);
 vm.runInContext(`
  let renders=0,meshUpdates=0;const originalRender=render;
  const object=()=>({visible:false,geometry:new THREEGeometry(),matrix:new THREE.Matrix4(),userData:{owner:0}});
  renderer={};render=()=>{renders++;};centered=true;
  mano=object();mano.geometry.setAttribute('position',new THREE.Float32BufferAttribute(3,3));
  robot=[object()];dots=[object(),object()];anchorDots=[object(),object()];normals=[object(),object()];rays=[object(),object()];errors=object();contactGraph=[object(),object()];
  surface={update:vertices=>{meshUpdates++;return vertices.flat();}};
  model={solver_backend:'slsqp',algorithm:'moxian_unified_graph_slsqp_v3',topology_version:'topology',mapping:{version:'mapping',ready:true,records:[{link:'pinky',normal_error_deg:0,owner:0,source_joint_index:0}]}};
  version=viewKey({...model,mapping_version:'mapping',geometry_version:'geometry',source_binding_version:'binding'});
  globalThis.inspect=()=>({mano:mano.visible,robot:robot[0].visible,anchor:anchorDots[0].visible,
    error:errors.visible,color:dots[0].geometry.attributes.color?.values[0][0],renders,meshUpdates,
    status:document.getElementById('wujiStatus').textContent,errorText:document.getElementById('wujiError').textContent,identity:document.getElementById('wujiFrameIdentity').textContent,rightLabel:document.querySelector('.wuji-right-label').textContent,frameKey:displayedFrameKey,position:mano.geometry.attributes.position.values[0],graph:contactGraph.map(g=>g.visible),graphVertices:contactGraph.map(g=>g.geometry.attributes.position?.count)});
 `,context);
 const frame={status:'solving',ready:true,display_available:true,input_live:true,contact_ready:true,contact_mask:[true],
  solver_backend:'slsqp',algorithm:'moxian_unified_graph_slsqp_v3',topology_version:'topology',
  timestamp_ms:1000,solver_session_id:'session',mapping_version:'mapping',source_hand_side:'right',source_timestamp_ms:1000,
  source_vertices_m:[[0,0,0]],link_positions_m:[[0,0,0]],link_rotations:[[[1,0,0],[0,1,0],[0,0,1]]],
  source_taxel_positions_m:[[0,0,0]],taxel_positions_m:[[0,0,0]],source_taxel_normals:[[0,0,1]],taxel_normals:[[0,0,1]],source_joints_m:[[0,0,0]],
  contact_anchors:[{position_m:[0,0,0],robot_position_m:[0,0,0]}]};
 const update=(changes={},connected=true)=>{context.window.updateWujiView({connected,hand_model:{geometry_version:'geometry',binding_version:'binding'},wuji2:{...frame,...changes}},true);assert.equal(context.window.wujiRenderError,undefined);return context.inspect();};
 return {update,run:source=>vm.runInContext(source,context)};
}

test('polling a slow solve retains paired geometry without redundant rebuilds',()=>{
 const v=view(),first=v.update();assert.ok(first.mano&&first.robot&&first.anchor);
 for(let i=0;i<10;i++){
  const next=v.update({source_age_ms:500+i*100});assert.ok(next.mano&&next.robot);
  assert.equal(next.meshUpdates,first.meshUpdates);assert.equal(next.renders,first.renders);
 }
 assert.ok(v.update({timestamp_ms:2000}).meshUpdates>first.meshUpdates);
});

test('invalid correspondence preserves measured source touch and reports recovery truthfully',()=>{
 const v=view();
 const shown=v.update({solution_feasible:false,contact_ready:false,contact_mask:[false],observed_contact_mask:[true],taxel_projection:{normal_rejected_sids:[0]},
  solver:{execution_mode:'penetration_restoration',escape_continuation_active:true}});
 assert.equal(shown.color,1);
 assert.match(shown.status,/穿透恢复/);assert.match(shown.status,/1 个SID法向不匹配/);
});

test('pinch display exposes the nonzero source target as well as robot point gap',()=>{
 const v=view();const shown=v.update({contact_anchors:[{position_m:[0,0,0],robot_position_m:[0,0,0]},
  {position_m:[.017,0,0],robot_position_m:[.023,0,0]}],contact_edges:[{semantic_links:['thumb_distal','index_distal'],
  source_vector_m:[.017,0,0],robot_vector_m:[.023,0,0],nodes:[0,1]}]});
 assert.match(shown.errorText,/源 17.0 \/ Wuji 23.0 mm/);
});

test('switching from CEM cannot hold or relabel its old pose as SLSQP',()=>{
 const v=view();
 v.run(`model.solver_backend='cem';model.algorithm='card_test';version=viewKey({...model,mapping_version:'mapping',geometry_version:'geometry',source_binding_version:'binding'});`);
 const cem=v.update({solver_backend:'cem',algorithm:'card_test'});
 assert.ok(cem.mano&&cem.robot);assert.match(cem.identity,/画面后端：CEM/);
 const pending=v.update({solver_backend:'cem',algorithm:'card_test',requested_solver_backend:'slsqp',ready:false,display_available:false});
 assert.ok(!pending.mano&&!pending.robot);assert.match(pending.identity,/等待 SLSQP 首帧/);assert.doesNotMatch(pending.identity,/画面后端：CEM/);
 v.run(`model.solver_backend='slsqp';model.algorithm='moxian_unified_graph_slsqp_v3';version=viewKey({...model,mapping_version:'mapping',geometry_version:'geometry',source_binding_version:'binding'});`);
 const slsqp=v.update({solver_session_id:'new-slsqp',timestamp_ms:2000});
 assert.ok(slsqp.mano&&slsqp.robot);assert.match(slsqp.identity,/画面后端：SLSQP/);assert.match(slsqp.rightLabel,/SLSQP/);
 assert.match(slsqp.identity,/moxian_unified_graph_slsqp_v3/);
});

test('a requested backend hides the old result before its server acknowledgement',()=>{
 const v=view();v.update();
 v.run(`window.wujiRequestedSolver='cem';`);
 const pending=v.update();assert.ok(!pending.mano&&!pending.robot);assert.match(pending.identity,/等待 CEM 首帧/);
});

test('compact geometry follows same-frame metadata without reusing cached status',()=>{
 const v=view(),first=v.update();
 const floats=new Float32Array([1,2,3]),data=Buffer.from(floats.buffer).toString('base64');
 const packed={format:'wuji-f32-v1',fields:{source_vertices_m:{offset:0,shape:[1,3]}},data};
 const next=v.update({source_vertices_m:undefined,geometry_f32:packed,timestamp_ms:2000});
 assert.ok(next.mano&&next.robot);assert.deepEqual(Array.from(next.position),[2,1,-3]);assert.notEqual(next.frameKey,first.frameKey);
 const held=v.update({source_vertices_m:undefined,geometry_f32:packed,timestamp_ms:2000,ready:false,display_available:false});
 assert.ok(held.mano&&held.robot);assert.match(held.identity,/暂停画面/);
});

test('transient timeout holds the same frame and clears/restores contacts without a pose timestamp change',()=>{
 const v=view();assert.equal(v.update().color,1);
 const held=v.update({ready:false,display_available:false,input_live:false,contact_ready:false});
 assert.ok(held.mano&&held.robot);assert.equal(held.color,0);assert.ok(!held.anchor&&!held.error);assert.match(held.status,/保留最后画面/);
 const recovered=v.update();assert.ok(recovered.anchor&&recovered.error);assert.equal(recovered.color,1);
 const offline=v.update({ready:false,display_available:false,contact_ready:false},false);
 assert.ok(offline.mano&&offline.robot);assert.match(offline.status,/设备离线/);assert.equal(offline.color,0);
});

test('pending, restarted or failed solves retain the displayed pair without admitting their new pose',()=>{
 for(const changes of [{timestamp_ms:2000},{solver_session_id:'new-session'},{status:'error',error:'failed'},{status:'mapping'},{mapping_version:'new-mapping'}]){
  const v=view(),first=v.update();
  for(let i=0;i<3;i++){
   const held=v.update({...changes,source_vertices_m:[[9,9,9]],ready:false,display_available:false,contact_ready:false});
   assert.ok(held.mano&&held.robot&&!held.anchor);assert.equal(held.frameKey,first.frameKey);
   assert.deepEqual(held.position,first.position);assert.match(held.status,/保留最后画面/);
  }
  const recovered=v.update({timestamp_ms:3000,source_vertices_m:[[1,2,3]]});
  assert.ok(recovered.mano&&recovered.robot&&recovered.anchor);assert.notEqual(recovered.frameKey,first.frameKey);
  assert.deepEqual(Array.from(recovered.position),[2,1,-3]);
 }
 const v=view();assert.ok(!v.update({ready:false,display_available:false}).mano);
});

test('incomplete or nonfinite candidates cannot blank or corrupt the displayed pair',()=>{
 for(const changes of [{source_vertices_m:[]},{source_vertices_m:[[NaN,0,0]]},{link_positions_m:[]},{link_rotations:[]},{taxel_normals:[[0,Infinity,0]]}]){
  const v=view(),first=v.update(),held=v.update({...changes,timestamp_ms:2000});
  assert.ok(held.mano&&held.robot&&!held.anchor);assert.equal(held.frameKey,first.frameKey);assert.deepEqual(held.position,first.position);
 }
});

test('same-size redraws do not reset the WebGL drawing buffer',()=>{
 const v=view();
 const result=v.run(`
  let resizes=0,width=800;visible=true;
  canvas.getBoundingClientRect=()=>({width,height:520});canvas.parentElement={classList:{toggle(){}}};
  renderer={getPixelRatio:()=>1,setSize(){resizes++;},setScissorTest(){},setViewport(){},setScissor(){},render(){}};
  camera={updateProjectionMatrix(){}};scenes=[{},{}];
  originalRender();originalRender();originalRender();const initial=resizes;width=900;originalRender();
  ({initial,total:resizes});
 `);
 assert.equal(result.initial,1);assert.equal(result.total,2);
});

test('a finite current candidate replaces the previous pose even if constraints failed',()=>{
 const v=view(),first=v.update();
 const next=v.update({timestamp_ms:2000,status:'infeasible',ready:false,display_available:true,contact_ready:false});
 assert.ok(next.mano&&next.robot&&!next.anchor);assert.ok(next.meshUpdates>first.meshUpdates);
 assert.match(next.status,/本帧候选/);assert.match(next.status,/超限/);
 const solving=v.update({timestamp_ms:2000,status:'solving',solution_feasible:false,ready:false,display_available:true,contact_ready:false});
 assert.ok(solving.mano&&solving.robot);assert.match(solving.status,/超限/);assert.match(solving.status,/下一帧计算中/);
});

test('SLSQP status and semantic graph use the supplied edges and clear stale overlays',()=>{
 const v=view();
 const first=v.update({solver_backend:'slsqp',solver:{timing_ms:{slsqp:10,final_audit:2}},
  quality:{unified_laplacian_rmse_mm:3.45,contact_laplacian_rmse_mm:1.23,contact_edge_rmse_mm:2.34},collision:{separation_satisfied:true},
  retarget_graph:{source_nodes_m:[[0,0,0],[0,1,0],[1,1,0]],robot_nodes_m:[[0,0,0],[0,1,0],[1,1,0]],edges:[
   {nodes:[0,2],kind:'material_attachment'},{nodes:[2,0],kind:'material_attachment'},
   {nodes:[0,1],kind:'skeleton'},{nodes:[1,2],kind:'contact'}]},
  contact_anchors:[{position_m:[0,0,0],robot_position_m:[0,0,0]},{position_m:[0,1,0],robot_position_m:[0,1,0]}],
  contact_edges:[{nodes:[0,1],semantic_links:['thumb_distal','palm']}]});
 assert.match(first.status,/SLSQP/);assert.match(first.status,/1 条跨 link 边/);
 assert.match(first.errorText,/统一图 Laplacian RMS 3.45 mm/);assert.match(first.errorText,/接触子图 RMS 1.23 mm（诊断）/);assert.match(first.errorText,/边向量 RMS 2.34 mm/);assert.match(first.errorText,/零间距验收通过/);
 assert.deepEqual(Array.from(first.graphVertices),[4,4]); // One contact and one attachment, without directed duplicates.
 assert.deepEqual(Array.from(first.graph),[true,true]);
 const stale=v.update({ready:false,display_available:false,contact_ready:false});
 assert.deepEqual(Array.from(stale.graph),[false,false]);
});
