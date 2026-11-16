const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');

function ui(){
 const nodes={},requests=[];
 const element=()=>({value:'',checked:true,textContent:'',children:[],selectedOptions:[],
  appendChild(child){this.children.push(child);},replaceChildren(...children){this.children=children;},cloneNode(){return {...this};}});
 const el=id=>nodes[id]??=element();
 const window={confirm:()=>true};
 const context=vm.createContext({document:{getElementById:el,createElement:element},window,Date,Promise,AbortSignal,encodeURIComponent,
  fetch:async(url,options)=>{requests.push({url,body:options?.body&&JSON.parse(options.body)});return {ok:true,json:async()=>options?{active:true,capture_mode:'cart_v3',frames:0}:{items:[]}};}});
 vm.runInContext(fs.readFileSync(path.join(__dirname,'../moxian_web/static/cart.js'),'utf8'),context);
 return {el,window,requests,tick:()=>new Promise(resolve=>setImmediate(resolve))};
}
const stages=[{id:'pinch_index',index:0,category:1,label:'拇指对食指'}];
test('CART records the selected synchronized protocol stage and partition',async()=>{
 const h=ui();await h.tick();h.el('cartLabel').value='motion';
 h.window.updateCartUI({connected:true,hand_model:{ready:true,calibrated:true},cart:{ready:true,stages,stage:'pinch_index',split:'train',recording:{},hull:{ready:true,volume_ml:12.345,area_cm2:4}}});
 assert.equal(h.el('cartVolume').textContent,'12.345 mL');assert.equal(h.el('cartRecordStart').disabled,false);
 await h.el('cartRecordStart').onclick();
 const request=h.requests.find(r=>r.body?.command==='cart_record_start');
 assert.deepEqual(request.body,{command:'cart_record_start',label:'motion',stage:'pinch_index',split:'train'});
 assert.match(h.el('cartRecordStatus').textContent,/CART 录制中/);
});
test('CART closure shows hand contact instead of inventing a 3D thickness',async()=>{
 const h=ui();await h.tick();
 h.window.updateCartUI({connected:true,hand_model:{ready:true,calibrated:true},cart:{ready:true,stages,stage:'pinch_index',recording:{},hull:{ready:false,contact_mode:'hand'},motion:{cycles:1,phase:'hold'}}});
 assert.match(h.el('cartHullStatus').textContent,/真实手部表面接触/);assert.equal(h.el('cartVolume').textContent,'—');
 assert.equal(h.el('cartNext').disabled,true);
 h.window.updateCartUI({connected:false,hand_model:{ready:false},cart:{recording:{}}});
 assert.equal(h.el('cartRecordStart').disabled,true);
 h.el('cartShowHull').checked=false;h.el('cartShowHull').onchange();assert.equal(h.window.cartHullVisible,false);
});
test('CART shows fitted primitive size, confidence and invalid budget-limited candidates',async()=>{
 const h=ui();await h.tick();
 const primitive={kind:'sphere',reference_type:13,radius_m:.012,confidence:.9};
 h.window.updateCartUI({cart:{ready:true,recording:{},hull:{ready:true,contact_mode:'hull',volume_ml:1,area_cm2:2,primitive,initial_shapes_tried:['cylinder','sphere'],processing_ms:35}}});
 assert.match(h.el('cartHullStatus').textContent,/primitive 接触目标有效/);
 assert.match(h.el('cartHullStatus').textContent,/type 13 · sphere \+ delta · 半径 12.0 mm · 几何置信度 90%/);
 assert.match(h.el('cartHullStatus').textContent,/已尝试 cylinder \/ sphere/);
 h.window.updateCartUI({cart:{ready:true,recording:{},hull:{ready:false,contact_mode:'unavailable',budget_exhausted:true,diagnostic_candidate:{primitive}}}});
 assert.match(h.el('cartHullStatus').textContent,/无效监督/);
 assert.match(h.el('cartHullStatus').textContent,/搜索预算已用尽/);
 assert.equal(h.el('cartVolume').textContent,'—');
});
test('CART action progress uses the actual protocol action count',async()=>{
 const h=ui();await h.tick();
 const actions=Array.from({length:8},(_,index)=>({id:'action'+index,index,category:index<4?1:index-2,label:'action'}));
 h.window.updateCartUI({cart:{ready:true,stages:actions,stage:'action7',recording:{}}});
 assert.match(h.el('cartPrompt').textContent,/8\/8/);
 assert.equal(h.el('cartStage').children.length,8);
});

test('CART distinguishes low confidence from a candidate validated after its search budget',async()=>{
 const h=ui();await h.tick();
 h.window.updateCartUI({cart:{ready:true,recording:{},hull:{ready:false,contact_mode:'unavailable',status:'budget_exhausted',
  budget_exhausted:true,fit_attempts:[{type_id:6,status:'low_confidence',confidence:.1}]}}});
 assert.match(h.el('cartHullStatus').textContent,/几何置信度低/);
 assert.match(h.el('cartHullStatus').textContent,/拇指参与对向夹持/);
 h.window.updateCartUI({cart:{ready:true,recording:{},hull:{ready:true,contact_mode:'hull',volume_ml:1,area_cm2:2,
  budget_exhausted:true}}});
 assert.match(h.el('cartHullStatus').textContent,/接触目标有效/);
 assert.match(h.el('cartHullStatus').textContent,/当前候选验收通过/);
 h.window.updateCartUI({cart:{ready:true,recording:{},hull:{ready:false,contact_mode:'unavailable',diagnostic_candidate:{
  contacts:[{patch:'thumb_distal',gap_m:.004,normal_error_deg:10}]}}}});
 assert.match(h.el('cartHullStatus').textContent,/尚未达标：拇指/);
});
test('CART training sends independent selected datasets and explicit epochs',async()=>{
 const h=ui();await h.tick();h.el('cartDatasets').selectedOptions=[{value:'train1'},{value:'validation1'}];h.el('cartEpochs').value='200';
 await h.el('cartTrain').onclick();assert.deepEqual(h.requests.find(r=>r.body?.command==='cart_train').body,{command:'cart_train',datasets:['train1','validation1'],epochs:200});
});

test('CART delete controls target the CART store',async()=>{
 const h=ui();await h.tick();h.el('cartDatasets').selectedOptions=[{value:'old-stage'}];
 await h.el('cartDeleteDatasets').onclick();
 assert.deepEqual(h.requests.find(r=>r.body?.command==='cart_recordings_delete').body,{command:'cart_recordings_delete',ids:['old-stage']});
 h.el('cartModels').value='old-network';await h.el('cartDeleteModel').onclick();
 assert.deepEqual(h.requests.find(r=>r.body?.command==='cart_models_delete').body,{command:'cart_models_delete',ids:['old-network']});
});

test('CART collection targets remain visible with GeoRT selected and no CART model',async()=>{
 const h=ui();await h.tick();let shown;
 h.window.updateCartPreview=s=>{shown=s;};
 h.window.updateCartUI({wuji2:{solver_backend:'geort'},cart:{ready:true,recording:{active:true},stages,stage:'pinch_index',
  hull:{ready:false,contact_mode:'hand',reference_gap_m:.01,contact_points_m:[[0,0,0],[.01,0,0]]}}});
 assert.equal(shown.wuji2.solver_backend,'geort');
 assert.match(h.el('cartPrompt').textContent,/使用真实手部表面接触/);
 assert.match(h.el('cartHullStatus').textContent,/10.0 mm/);
});

test('CART overlay uploads current triangles and hides on stale input or toggle',()=>{
 class Geometry{constructor(){this.attributes={};}dispose(){}deleteAttribute(name){delete this.attributes[name];}setAttribute(name,value){this.attributes[name]=value;}computeBoundingSphere(){}computeVertexNormals(){}}
 class Attribute{constructor(values,size){this.array=new Float32Array(values);this.count=values.length/size;}}
 class Object3D{constructor(geometry,material){this.geometry=geometry;this.material=material;this.children=[];}add(...children){this.children.push(...children);}}
 const THREE={BufferGeometry:Geometry,Float32BufferAttribute:Attribute,Group:Object3D,Mesh:Object3D,LineSegments:Object3D,Points:Object3D,
  MeshStandardMaterial:class{},LineBasicMaterial:class{},PointsMaterial:class{},DoubleSide:2};
 const scene=new Object3D(),window={cartHullVisible:true},context=vm.createContext({window,THREE});
 const source=fs.readFileSync(path.join(__dirname,'../moxian_web/static/cart-overlay.js'),'utf8').replace(/^import .*;\n/gm,'');vm.runInContext(source,context);
 const hull={ready:true,timestamp_ms:1,dimension:3,geometry_version:'g',vertices_m:[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],faces:[[0,1,2],[0,3,1]],edges:[[0,1]],finger_points_m:[[0,0,0],[1,0,0],[0,1,0],[0,0,1],[.1,.1,.1]]};
 window.updateCartHullOverlay(scene,hull,p=>[p[1],p[0],-p[2]]);
 assert.equal(scene.children.length,1);const group=scene.children[0];assert.equal(group.children[0].geometry.attributes.position.count,6);
 assert.equal(group.children[2].geometry.attributes.position.count,5);
 window.updateCartHullOverlay(scene,null,p=>p);assert.equal(group.visible,false);
 window.cartHullVisible=false;window.updateCartHullOverlay(scene,hull,p=>p);assert.equal(group.visible,false);
});
