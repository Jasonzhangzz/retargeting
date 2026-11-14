const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('moxian_web/static/checkpoint-view.js','utf8');
const items=[{id:'parameters-id',method:'parameters',label:'xMimic 思路版',compatible:true,accepted:false,contact_gap_mm:22},{id:'flash-id',method:'flashsac',label:'FlashSAC buffer 版',compatible:true,accepted:false,contact_gap_mm:23}];
const connected={connected:true,device:{serial:'glove'},side:'right',hand_model:{ready:true}};
const flush=()=>new Promise(resolve=>setImmediate(resolve));
function page(fetcher){
 const elements={},requests=[];
 const element=id=>elements[id]??={value:'',disabled:false,textContent:'',replaceChildren(...options){this.options=options;this.value=options[0]?.value||'';}};
 const context=vm.createContext({document:{getElementById:element},window:{},Option:class{constructor(label,value){this.label=label;this.value=value;}},Number,Date,AbortSignal,setInterval(){},fetch:async(url,options)=>{requests.push({url,options});return fetcher(url,options);}});
 vm.runInContext(source,context);
 return {elements,requests,update:state=>context.window.updateCheckpointComparison(state)};
}
const response=()=>({ok:true,json:async()=>({items})});

test('native default and five historical methods load without any live-state callback',async()=>{
 const p=page(response);await flush();
 assert.deepEqual(p.elements.manoCheckpoint.options.map(x=>x.label),['Moxian六动作骨长与EMF外参','Wuji 原生六动作标定','Moxian 17动作标定（历史）','xMimic 思路版（历史）','FlashSAC buffer 版（历史）','几何＋Delta 联合训练（历史）','分阶段最小二乘标定（历史）']);
 p.elements.manoCheckpoint.value='parameters';p.elements.manoCheckpoint.onchange();
 assert.equal(p.requests.length,1);assert.match(p.elements.manoCheckpointSummary.textContent,/22.00/);
 assert.match(p.elements.manoCheckpointStatus.textContent,/已读取 2/);assert.equal(p.elements.manoCheckpointLoad.disabled,true);
 p.update(connected);await flush();
 p.elements.manoCheckpoint.value='flashsac';p.elements.manoCheckpoint.onchange();
 assert.equal(p.elements.manoCheckpointLoad.disabled,false);assert.match(p.elements.manoCheckpointSummary.textContent,/23.00/);
 await p.elements.manoCheckpointLoad.onclick();await flush();
 assert.deepEqual(JSON.parse(p.requests.at(-1).options.body),{command:'mano_checkpoint_preview',id:'flash-id'});
});

test('failed fetch never empties choices or retries on every state frame; refresh recovers',async()=>{
 let fail=true;const p=page(()=>{if(fail)throw Error('network unavailable');return response();});await flush();
 assert.equal(p.elements.manoCheckpoint.options.length,7);assert.match(p.elements.manoCheckpointStatus.textContent,/network unavailable/);
 p.elements.manoCheckpoint.value='parameters';
 p.update(connected);await flush();const count=p.requests.length;
 for(let i=0;i<20;i++)p.update(connected);await flush();assert.equal(p.requests.length,count);
 fail=false;await p.elements.manoCheckpointRefresh.onclick();await flush();
 assert.equal(p.elements.manoCheckpointLoad.disabled,false);assert.match(p.elements.manoCheckpointStatus.textContent,/可对比 2/);
 assert.equal(p.elements.manoCheckpoint.options.length,7);
});

test('an in-flight response from a previous device is discarded before preview can be enabled',async()=>{
 let resolve,calls=0;const p=page(()=>++calls===1?new Promise(r=>resolve=r):response());
 p.elements.manoCheckpoint.value='parameters';
 p.update(connected);assert.equal(p.elements.manoCheckpointLoad.disabled,true);
 resolve({ok:true,json:async()=>({items:[{...items[0],id:'wrong-device-id'}]})});await flush();
 assert.equal(calls,2);assert.equal(p.elements.manoCheckpointLoad.disabled,false);
 p.elements.manoCheckpointLoad.onclick();await flush();
 assert.equal(JSON.parse(p.requests.at(-1).options.body).id,'parameters-id');
});
