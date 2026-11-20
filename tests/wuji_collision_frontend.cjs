const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
function element(){return {textContent:'',disabled:false,className:'',children:[],dataset:{},appendChild(v){this.children.push(v);},replaceChildren(...v){this.children=v;}};}
const html=fs.readFileSync('moxian_web/static/index.html','utf8');
const ids=new Map([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],element()]));
const requests=[];
const context={document:{getElementById:id=>{assert(ids.has(id));return ids.get(id);},createElement:element},window:{},AbortSignal,Number,URLSearchParams,
 fetch:async(url,options)=>{
  if(options?.body){requests.push(JSON.parse(options.body));return {ok:true,json:async()=>({enabled:false,ready:false,status:'disabled'})};}
  return {ok:true,json:async()=>url.endsWith('/examples')?{ready:true,items:[{index:4,pose_id:42,label:'对照样本',disagreement_pairs:1}]}:
   {...d,example:{pose_id:42},input_kind:'test'}};
 }};
vm.runInNewContext(fs.readFileSync('moxian_web/static/wuji-collision.js','utf8'),context);
const update=d=>context.window.updateWujiCollisionUI({wuji_collision:d});
const result={query_valid:true,collision:true,max_penetration_mm:1.2,collision_pairs:2,processing_ms:2};
const d={enabled:true,ready:true,status:'ready',timestamp_ms:100,age_ms:10,device:'cpu',processing_ms:5,
 solver_backend:'cart',dense_mesh:result,matched_mesh:result,neural:{...result,collision:false},
 model:{name:'mlp100k_v1',epoch:30},in_training_domain:true,query_points:1000,samples_per_link:128,
 comparison:{dense_agrees:false,matched_agrees:false,dense_depth_error_mm:1.2,matched_depth_error_mm:1.2,point_mae_mm:.2,frames:1,dense_disagreement_frames:1,matched_disagreement_frames:1},
 pairs:[{a:4,b:8,link_a:'thumb_link4',link_b:'index_link4',dense_mm:1.2,matched_mm:1.2,neural_mm:.1,agrees:false}]};
update(d);
assert.equal(ids.get('wujiCollisionMethods').children.length,3);
assert.match(ids.get('wujiCollisionDifference').textContent,/不一致/);
assert.match(ids.get('wujiCollisionPairs').children[0].children[0].textContent,/拇指 L4 ↔ 食指 L4/);
assert.match(ids.get('wujiCollisionStatus').textContent,/cart/);
update({...d,in_training_domain:false,out_of_domain_joints:['j1']});
assert.match(ids.get('wujiCollisionDomain').textContent,/超出模型训练域/);
update({...d,ready:false,status:'stale',error:'诊断帧已过期'});
assert.equal(ids.get('wujiCollisionMethods').children.length,0);
assert.equal(ids.get('wujiCollisionPairs').children.length,0);
assert.equal(ids.get('wujiCollisionDifference').textContent,'');
update({...d,neural:null,comparison:null,model_error:'模型不存在'});
assert.match(ids.get('wujiCollisionDifference').textContent,/模型不存在/);
ids.get('wujiCollisionToggle').onclick().then(async()=>{
 assert.equal(requests[0].command,'wuji2_collision_set');assert.equal(requests[0].enabled,false);
 assert.match(ids.get('wujiCollisionStatus').textContent,/已停止/);
 await ids.get('wujiCollisionExample').onclick();
 assert.match(ids.get('wujiCollisionStatus').textContent,/非实时输入/);
 assert.equal(context.window.currentWujiCollisionVisual.example.pose_id,42);
 update({enabled:true,ready:false,status:'waiting_input'});
 assert.equal(context.window.currentWujiCollisionVisual.example.pose_id,42); // Explicit test mode survives live polling.
 ids.get('wujiCollisionLive').onclick();
 assert.equal(context.window.currentWujiCollisionVisual.ready,false);
 assert.equal(ids.get('wujiCollisionMethods').children.length,0);
 console.log('Wuji collision UI: matched frames, disagreement, stale clearing, domain and toggle passed');
}).catch(error=>{console.error(error);process.exitCode=1;});
