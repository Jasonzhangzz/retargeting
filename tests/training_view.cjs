const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('moxian_web/static/training-view.js','utf8'),flush=()=>new Promise(r=>setImmediate(r));
function page(responses){
 const elements={},requests=[];let poll;
 const ctx=new Proxy({},{get:(o,k)=>k in o?o[k]:()=>{},set:(o,k,v)=>(o[k]=v,true)});
 const element=id=>elements[id]??={value:id==='manoTrainingScope'?'replay':'',checked:true,textContent:'',clientWidth:400,getContext:()=>ctx};
 const window={devicePixelRatio:1,addEventListener(){}};
 vm.runInNewContext(source,{document:{getElementById:element},window,Number,Math,Set,URLSearchParams,AbortSignal,setInterval:fn=>poll=fn,fetch:async url=>{requests.push(String(url));return {ok:true,json:async()=>responses.shift()};}});
 return {elements,requests,helpers:window.ManoTrainingPlot,poll,window};
}
const point=(step,scope='reference')=>({sequence:step,sample_id:step,step,phase:scope.startsWith('pinch')?'pinches':scope,scope,source:'train_only',training_frames:349,raw:{loss:4,beta_kl:1,geometry_kl:2},ema:{loss:3,beta_kl:1,geometry_kl:2,contact_gap_mm:null}});
test('unknown values are gaps and phase/action losses never connect',async()=>{
 const p=page([{points:[],cursor:0,run_id:null}]);await flush();
 const data=[point(1),{...point(2),ema:{loss:null}},point(3),point(4,'geometry'),point(5,'pinch_index')];
 const segments=p.helpers.segments(data,['ema','loss']);
 assert.equal(segments.length,4);assert.equal(segments.flat().length,4);
 assert.equal(p.helpers.selected(data,'replay').length,4);
 assert.equal(p.helpers.selected(data,'pinch_index').length,1);
});
test('incremental polling deduplicates, excludes validation and resets on a new run',async()=>{
 const p=page([
  {points:[point(1),{...point(2),source:'validation'}],cursor:2,run_id:'a',reset:true,status:'running'},
  {points:[point(1),point(3)],cursor:3,run_id:'a',reset:false,status:'running'},
  {points:[point(1,'geometry')],cursor:1,run_id:'b',reset:true,status:'completed'}
 ]);await flush();assert.match(p.elements.manoTrainingStatus.textContent,/1 个真实/);
 await p.poll();await flush();assert.match(p.requests[1],/after=2&run_id=a/);assert.match(p.elements.manoTrainingStatus.textContent,/2 个真实/);
 await p.poll();await flush();assert.match(p.elements.manoTrainingStatus.textContent,/1 个真实/);assert.match(p.elements.manoTrainingStatus.textContent,/训练完成/);
});

test('empty plots show the actual baseline phase and configured step limit',async()=>{
 const response={points:[],cursor:0,run_id:'a',reset:true,status:'running',method:'parameters'};
 const p=page([response,response]);await flush();
 p.window.updateTrainingCharts({device:{serial:'test'},side:'right',hand_model:{fit:{state:'running',run_id:'a',method:'parameters',message:'基线验证 300/3380 帧',training_options:{adam_steps:480}}}});await flush();
 assert.match(p.elements.manoTrainingStatus.textContent,/0\/480/);assert.match(p.elements.manoTrainingStatus.textContent,/基线验证 300\/3380/);
});

test('completed curves distinguish the maximum from actual steps and failed acceptance',async()=>{
 const response={points:[point(129,'mixed')],cursor:129,run_id:'a',reset:true,status:'preview',method:'parameters'};
 const p=page([response,response]);await flush();
 p.window.updateTrainingCharts({device:{serial:'test'},side:'right',hand_model:{fit:{state:'preview',run_id:'a',method:'parameters',accepted:false,training_options:{adam_steps:480},solver:{budget_exhausted:true}}}});await flush();
 const text=p.elements.manoTrainingStatus.textContent;assert.match(text,/129\/480/);assert.match(text,/时间预算已到/);assert.match(text,/未自动应用/);
});

test('curves identify the actual hybrid optimizer from measurements',async()=>{
 const sample=point(3,'mixed');sample.raw.optimizer_name='Muon+Adam';
 const p=page([{points:[sample],cursor:3,run_id:'muon',reset:true,status:'running',method:'delta'}]);await flush();
 assert.match(p.elements.manoTrainingStatus.textContent,/优化器 Muon\+Adam/);
});

test('continuation curves distinguish additional updates from cumulative steps',async()=>{
 const sample=point(10,'mixed');sample.previous_training_steps=480;sample.total_training_step=490;
 const p=page([{points:[sample],cursor:10,run_id:'resume',reset:true,status:'running',method:'delta'}]);await flush();
 assert.match(p.elements.manoTrainingStatus.textContent,/从 480 步权重续训，累计 490 步/);
});

test('least squares shows measured LM loss and damping without Adam or EMA telemetry',async()=>{
 const sample={sequence:1,sample_id:1,step:1,source:'train_only',phase:'dimensions',scope:'dimensions',solver:'bounded_block_LM',loss:12.5,damping:.05,accepted:true,step_norm:.1,gradient_norm:.2,training_frames:3309,linearized_frames:96};
 const p=page([{points:[sample],cursor:1,run_id:'lsq',reset:true,status:'completed',method:'least_squares'}]);await flush();
 assert.match(p.elements.manoTrainingStatus.textContent,/最小二乘/);assert.match(p.elements.manoTrainingStatus.textContent,/阻尼/);
 assert.doesNotMatch(p.elements.manoTrainingStatus.textContent,/Adam|EMA|学习率/);
 assert.equal(p.elements.manoTrainingLossPart.disabled,true);
 assert.match(p.elements.manoPriorCaption.textContent,/LM 阻尼/);
});


test('EMF skeleton fitting plots native objective evaluations without network training telemetry',async()=>{
 const sample={sequence:1,sample_id:1,step:5,phase:'mixed',scope:'mixed',source:'train_only',raw:{loss:12.5,optimizer_name:'Wuji native NLopt LD_LBFGS',geometry_gradient_norm:.2,terms:{bone_prior:1,bone_ratio_prior:2,extrinsics_prior:3}}};
 const p=page([{points:[sample],cursor:1,run_id:'bones',reset:true,status:'completed',method:'bone_extrinsics'}]);await flush();
 assert.match(p.elements.manoTrainingStatus.textContent,/EMF与骨架拟合/);
 assert.match(p.elements.manoTrainingStatus.textContent,/5 次目标评估/);
 assert.doesNotMatch(p.elements.manoTrainingStatus.textContent,/训练|Adam|EMA|Delta/);
});

test('complete native calibration plots the actual total objective and hides unavailable network telemetry',async()=>{
 const sample={sequence:1,sample_id:'wujihand:100',step:100,phase:'wujihand',scope:'wujihand',source:'train_only',raw:{loss:3.5,optimizer_name:'Wuji native calibration'}};
 const response={points:[sample],cursor:1,run_id:'native',reset:true,status:'completed',method:'wuji_native'};
 const p=page([response,response]);await flush();
 p.window.updateTrainingCharts({device:{serial:'test'},side:'right',hand_model:{fit:{method:'wuji_native',state:'completed',run_id:'native',accepted:true}}});await flush();
 assert.match(p.elements.manoTrainingStatus.textContent,/100 次目标评估/);assert.match(p.elements.manoTrainingStatus.textContent,/原生标定成功/);
 assert.doesNotMatch(p.elements.manoTrainingStatus.textContent,/独立验收通过|Adam|EMA|Delta/);assert.equal(p.elements.manoTrainingLossPart.disabled,true);
});
