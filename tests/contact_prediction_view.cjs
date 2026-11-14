const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs'),vm=require('node:vm');

function page(topology,observerClass){
 const elements={},draws={};
 const element=id=>elements[id]??={checked:false,textContent:'',getBoundingClientRect:()=>({left:0,top:0,width:300,height:330}),
  getContext:()=>({setTransform(){},clearRect(){draws[id]={circles:[],texts:[],paths:[]};},fillText(t){draws[id].texts.push(t);},fillRect(){},beginPath(){},moveTo(x,y){draws[id].paths.push([x,y]);},lineTo(){},closePath(){},fill(){},stroke(){},arc(...args){draws[id].circles.push({args,color:this.fillStyle});}})};
 const context=vm.createContext({document:{getElementById:element},window:{addEventListener(){}},IntersectionObserver:observerClass,devicePixelRatio:1,Number,Math,AbortSignal,fetch:async()=>({ok:true,json:async()=>topology}),Event:class{}});
 vm.runInContext(fs.readFileSync('moxian_web/static/contact-prediction-view.js','utf8'),context);
 return {context,elements,draws,update:state=>context.window.updateContactPrediction(state)};
}

test('live comparison renders measured, illustrative and verified metrics separately',()=>{
 const p=page();p.update({contact_prediction:{ready:true,status:'ready',history_frames:8,history_required:8,timestamp_ms:1000,
  processing_ms:9.1,frame_probability:.8,taxel_probability:Array(460).fill(.7),comparison:{ready:true,timestamp_delta_ms:2,
   illustrative:{f1:.6,tp:5,fp:3,fn:2,tn:450},cumulative:{illustrative:{f1:.4}},verified:null,verified_count:0,
   actual_contact:Array(460).fill(false),actual_raw_adc:Array(460).fill(12),scored_frames:20}}});
 assert.equal(p.elements.predictionF1.textContent,'60.0%');assert.equal(p.elements.predictionCumulative.textContent,'40.0%');
 assert.equal(p.elements.predictionVerified.textContent,'—');assert.match(p.elements.predictionMapping.textContent,/0\/460/);
 assert.match(p.elements.predictionCounts.textContent,/误报 3/);
 p.elements.predictionShowADC.checked=true;p.elements.predictionShowADC.onchange();
 assert.match(p.elements.predictionTruthTitle.textContent,/ADC/);
});

test('unknown tactile never displays a zero error score or fabricated verification',()=>{
 const p=page();p.update({contact_prediction:{ready:true,status:'ready',timestamp_ms:20,processing_ms:8,comparison:{ready:false}}});
 assert.equal(p.elements.predictionF1.textContent,'—');assert.equal(p.elements.predictionVerified.textContent,'—');
 assert.match(p.elements.predictionMessage.textContent,/未知/);
 p.update({contact_prediction:{ready:false,status:'warming_up',history_frames:3,history_required:8}});
 assert.match(p.elements.predictionStatus.textContent,/3\/8/);
});

function shape(version='v1',side='right'){
 return {ready:true,binding_version:version,hand_side:side,palm_view:{normal:[0,0,1],forward:[0,1,0]},
  rest_vertices_m:Array.from({length:778},(_,i)=>i===1?[1,0,0]:i===2?[0,1,0]:[0,0,0]),
  // Opposite triangle windings must both fill the silhouette rather than cancel.
  faces:[[0,1,2],[0,2,1]],sid_face_ids:Array(460).fill(0),
  sid_barycentric:Array.from({length:460},(_,i)=>[i/459,1-i/459,0]),sid_patch_indices:Array(460).fill(0),
  patches:[{label:'食指 · 指腹'}],tip_anchors_m:[[0,0,0],[.2,.2,0],[.4,.4,0],[.6,.6,0],[.8,.8,0]]};
}
const settle=()=>new Promise(resolve=>setImmediate(resolve));

test('offscreen prediction charts do not draw and resume with the latest snapshot',async()=>{
 let notify;const p=page(shape(),class{constructor(callback){notify=callback;}observe(){}});
 const state={contact_prediction:{ready:true,binding_version:'v1',hand_side:'right',timestamp_ms:40,processing_ms:8,taxel_probability:Array(460).fill(.7),comparison:{}}};
 p.update(state);assert.equal(p.draws.predictionGrid,undefined);
 notify([{isIntersecting:true}]);await settle();assert.equal(p.draws.predictionGrid.circles.length,460);
 notify([{isIntersecting:false}]);const drawn=p.draws.predictionGrid;p.update({...state,contact_prediction:{...state.contact_prediction,timestamp_ms:41}});assert.equal(p.draws.predictionGrid,drawn);
});

test('all three views use the same hand silhouette and barycentric SID positions with anatomical hover labels',async()=>{
 const p=page(shape());p.update({contact_prediction:{ready:true,binding_version:'v1',hand_side:'right',timestamp_ms:40,processing_ms:8,
  taxel_probability:Array(460).fill(.7),comparison:{ready:true,actual_contact:Array(460).fill(true)}}});
 await settle();
 const a=p.draws.predictionGrid,b=p.draws.predictionTruthGrid,c=p.draws.predictionDiffGrid;
 assert.equal(a.circles.length,460);
 assert.ok(Math.abs(a.circles[200].args[0]-(27+(1-200/459)*246))<1e-8);
 assert.equal(a.circles[200].args[1],289);assert.deepEqual(a.circles.map(v=>v.args),b.circles.map(v=>v.args));
 assert.deepEqual(a.circles.map(v=>v.args),c.circles.map(v=>v.args));
 assert.ok(a.texts.includes('右手 · 掌面展开'));assert.ok(a.texts.includes('拇指'));assert.ok(a.texts.includes('小指'));assert.ok(a.texts.includes('腕部'));
 assert.equal(c.circles[200].color,'#438f70');
 const [x,y]=a.circles[200].args;
 p.elements.predictionGrid.onmousemove({clientX:x,clientY:y});
 assert.match(p.elements.predictionGrid.title,/食指 · 指腹 · SID 200/);
 assert.match(p.elements.predictionGrid.title,/0.700/);
 // Opposite input windings have the same first edge after orientation correction.
 assert.deepEqual(a.paths[0],a.paths[1]);
});

test('missing or mismatched hand mapping never falls back to a rectangular SID grid or another hand',async()=>{
 const p=page(shape());
 p.update({contact_prediction:{ready:false,binding_version:'v1',hand_side:'right'}});await settle();
 assert.equal(p.draws.predictionGrid.circles.length,460);
 assert.equal(p.draws.predictionGrid.circles[0].color,'#d5d4cf');
 p.update({contact_prediction:{ready:false,binding_version:'v1',hand_side:'left'}});
 assert.equal(p.draws.predictionGrid.circles.length,0);
 assert.ok(p.draws.predictionGrid.texts.includes('等待对应手套曲面'));
 p.update({contact_prediction:{ready:false,binding_version:'another',hand_side:'right'}});await settle();
 assert.equal(p.draws.predictionGrid.circles.length,0);
});
