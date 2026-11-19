/* Execute the controller against a small DOM/API fixture; no device actions. */
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const html=fs.readFileSync('moxian_web/static/index.html','utf8');
const ids=new Map([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],element()]));
function element(){return {children:[],value:'',checked:true,disabled:false,textContent:'',
 get selectedOptions(){return this.children.filter(c=>c.selected);},
 replaceChildren(...values){this.children=values;},appendChild(value){this.children.push(value);}};}
const requests=[];
let recordings=[];
const context={document:{getElementById:id=>{assert(ids.has(id),`Missing DOM id ${id}`);return ids.get(id);},createElement:element},
 window:{confirm:()=>true},AbortSignal,Date,JSON,Array,Set,Number,
 fetch:async(url,options)=>{
  if(options?.body)requests.push(JSON.parse(options.body));
  return {ok:true,json:async()=>url.includes('/recordings')?{items:recordings}:
   url.includes('/models')?{items:[]}:{active:true,frames:1}};
 }};
async function main(){
 vm.runInNewContext(fs.readFileSync('moxian_web/static/cart.js','utf8'),context);
 await new Promise(resolve=>setImmediate(resolve));
 const state={connected:true,hand_model:{ready:true,calibrated:true},cart:{protocol:'cart_diff_ik_v6',split:'train',recording:{},
  ready:true,hull:{self_contacts:1,object_contacts:2,tactile_status:'valid'},preview_age_ms:1},wuji2:{}};
 context.window.updateCartUI(state);
 assert.equal(ids.get('cartRecordStart').disabled,false);
 assert.match(ids.get('cartHullStatus').textContent,/自接触 1 · 物体接触 2/);
 assert.equal(ids.has('cartStage'),false);assert.equal(ids.has('cartAction'),false);
 assert.equal(ids.has('cartTubeLoad'),false);assert.equal(ids.has('cartIKLoad'),true);
 recordings=[{id:'train',split:'train',protocol:'cart_diff_v5',emf_recorded:true,frames:100,complete:true}];
 await ids.get('cartRefresh').onclick();ids.get('cartDatasets').children[0].selected=true;ids.get('cartDatasets').onchange();
 assert.equal(ids.get('cartTrain').disabled,true);assert.equal(ids.get('cartPreprocess').disabled,false);
 assert.match(ids.get('cartTrainingHint').textContent,/训练 1 段、验证 0 段/);
 assert.match(ids.get('cartDatasets').children[0].textContent,/需准备观测与 IK/);
 recordings.push({id:'validation',split:'validation',protocol:'cart_diff_v5',emf_recorded:true,frames:100,complete:true});
 await ids.get('cartRefresh').onclick();ids.get('cartDatasets').children[1].selected=true;ids.get('cartDatasets').onchange();
 assert.equal(ids.get('cartTrain').disabled,false);
 assert.match(ids.get('cartTrainingHint').textContent,/已选择训练和独立验证会话/);
 ids.get('cartArchitecture').value='fsq';
 ids.get('cartInitialNoise').value='0.1';
 await ids.get('cartTrain').onclick();assert.equal(requests.at(-1).architecture,'fsq');
 assert.equal(requests.at(-1).batch_size,0);assert.equal('micro_batch_size' in requests.at(-1),false);
 assert.equal(ids.has('cartMicroBatchSize'),false);
 assert.equal(requests.at(-1).initial_noise_rad,.1);
 await ids.get('cartPreview').onclick();assert.equal(requests.at(-1).command,'cart_preview');
 ids.get('cartLabel').value='揉搓';await ids.get('cartRecordStart').onclick();
 assert.equal(requests.at(-1).command,'cart_record_start');assert.equal(requests.at(-1).label,'揉搓');
 assert.equal('stage' in requests.at(-1),false);
 state.cart.protocol='cart_contact_v3';context.window.updateCartUI(state);
 assert.equal(ids.get('cartRecordStart').disabled,true);
 assert.match(ids.get('cartHullStatus').textContent,/版本不匹配/);
 state.cart.protocol='cart_diff_ik_v6';
 state.cart.job={status:'running'};context.window.updateCartUI(state);
 assert.equal(ids.get('cartTrain').disabled,true);assert.equal(ids.get('cartPreprocess').disabled,true);
 console.log('CART controller: DOM bindings, free-motion commands, evidence display and busy guards passed');
}
main().catch(error=>{console.error(error);process.exitCode=1;});
