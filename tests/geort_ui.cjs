const {test}=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
const path=require('node:path');

function harness(){
 const nodes={},requests=[],confirmations=[],created=[],scatter=[];let commandResponse=null,getResponse=null,coverageResponse=null,confirmed=true;
 const element=()=>({value:'',textContent:'',disabled:false,selectedOptions:[],children:[],appendChild(child){this.children.push(child);},replaceChildren(...options){this.children=options;this.options=options;this.selectedOptions=options.filter(o=>o.selected);}});
 const node=id=>nodes[id]??=element();
 node('geortEpochs').value='200';node('geortFkEpochs').value='200';
 const ctx={setTransform(){},fillRect(){},fillText(){},beginPath(){},moveTo(){},lineTo(){},stroke(){},arc(x,y,r){scatter.push({x,y,r,color:this.fillStyle});},fill(){}};
 node('geortCoverageCanvas').getContext=()=>ctx;node('geortShowReference').checked=true;
 const document={getElementById:node,createElement:tag=>{created.push(tag);return element();},activeElement:null};
 const window={confirm:message=>{confirmations.push(message);return confirmed;}};
 const context=vm.createContext({document,window,console,Date,Promise,AbortSignal,URLSearchParams,fetch:async(url,options)=>{
  requests.push({url,body:options?.body&&JSON.parse(options.body)});
  if(options?.method==='POST'&&commandResponse)return commandResponse();
  if(coverageResponse&&url.includes('/api/geort/coverage?'))return coverageResponse(url);
  if(getResponse&&url.includes('/workspace/'))return getResponse(url);
  return {ok:true,json:async()=>url.includes('recordings')?{items:[{id:'session',label:'<unsafe>',frames:50,complete:true,source:{hand_side:'right'}}]}:url.includes('models')?{items:[{id:'model',status:'completed'}]}:{ok:true}};
 }});
 vm.runInContext(fs.readFileSync(path.join(__dirname,'../moxian_web/static/geort-coverage.js'),'utf8'),context);
 vm.runInContext(fs.readFileSync(path.join(__dirname,'../moxian_web/static/geort.js'),'utf8'),context);
 return {node,window,requests,confirmations,created,scatter,setConfirmed:value=>{confirmed=value;},setCommandResponse:fn=>{commandResponse=fn;},setGetResponse:fn=>{getResponse=fn;},setCoverageResponse:fn=>{coverageResponse=fn;},tick:()=>new Promise(resolve=>setImmediate(resolve))};
}

test('capture requires calibrated live input and exposes counters',async()=>{
 const h=harness();await h.tick();
 h.window.updateGeortUI(null);assert.equal(h.node('geortRecordStart').disabled,true);
 h.window.updateGeortUI({connected:true,hand_model:{ready:true,calibrated:false},geort:{recording:{}}});
 assert.equal(h.node('geortRecordStart').disabled,true);
 h.window.updateGeortUI({connected:true,hand_model:{ready:true,calibrated:true},geort:{recording:{active:true,frames:42,invalid_frames:2,started_at:Date.now()/1000}}});
 assert.equal(h.node('geortRecordStop').disabled,false);
 assert.match(h.node('geortRecordingStatus').textContent,/42 有效帧 · 2 无效帧/);
 assert.match(h.node('geortDatasets').options[0].textContent,/<unsafe>/); // textContent, never HTML.
});

test('training forwards selected sessions and official epoch defaults',async()=>{
 const h=harness();await h.tick();h.node('geortDatasets').selectedOptions=[{value:'a'},{value:'b'}];
 await h.node('geortTrain').onclick();
 const request=h.requests.find(r=>r.body?.command==='geort_train');
 assert.deepEqual(request.body,{command:'geort_train',datasets:['a','b'],epochs:200,fk_epochs:200,mode:'official'});
});

test('constrained training is explicit and labels its separate native-mesh losses',async()=>{
 const h=harness();await h.tick();h.node('geortDatasets').selectedOptions=[{value:'real-session'}];
 h.node('geortMode').value='contact_v1';await h.node('geortTrain').onclick();
 assert.equal(h.requests.find(r=>r.body?.command==='geort_train').body.mode,'contact_v1');
 h.window.updateGeortUI({geort:{training:{status:'running',training_mode:'contact_v1',losses:{collision:0,penetration:.3,surface_contact:.2,pinch_position:.5}}}});
 assert.match(h.node('geortTrainingStatus').textContent,/非官方loss/);
 assert.match(h.node('geortLosses').textContent,/penetration: 3.00000e-1（权重 10，0.5 mm归一化，参与反向传播）/);
 assert.match(h.node('geortLosses').textContent,/pinch_position: 5.00000e-1（权重 1，参与反向传播）/);
 h.window.updateGeortUI({geort:{training:{status:'running',training_mode:'contact_v1',constraint_weights:{penetration:1},losses:{penetration:.3}}}});
 assert.match(h.node('geortLosses').textContent,/权重 1，0.5 mm归一化/);
 assert.equal(h.node('geortMode').disabled,true);
});

test('record button gives immediate progress, start acknowledgement and visible failure',async()=>{
 const h=harness();await h.tick();
 h.window.updateGeortUI({connected:true,hand_model:{ready:true,calibrated:true},geort:{recording:{}}});
 let finish;h.setCommandResponse(()=>new Promise(resolve=>{finish=resolve;}));
 const pending=h.node('geortRecordStart').onclick();await h.tick();
 assert.match(h.node('geortRecordStart').textContent,/正在开始录制/);
 assert.equal(h.node('geortRecordStart').disabled,true);
 finish({ok:true,json:async()=>({active:true,saving:false,frames:0})});await pending;
 assert.match(h.node('geortRecordingStatus').textContent,/录制中/);assert.equal(h.node('geortRecordStop').disabled,false);
 h.setCommandResponse(async()=>({ok:false,json:async()=>({error:'录制保存失败'})}));
 await h.node('geortRecordStop').onclick();assert.match(h.node('geortError').textContent,/录制保存失败/);
});

test('new recording clears green points and observed reference cells, ignoring an older in-flight cloud',async()=>{
 const h=harness();await h.tick();
 const workspace={reference_id:'same-reference',fingers:[{finger:'thumb',reference_cloud_mm:[[0,0,0]],reference_sample_ids:[7],reference_cloud_counts:[3]}]};
 const old={baseline_ids:['old'],frames:3,fingers:[{finger:'thumb',cloud_mm:[[10,20,30]]}],workspace};
 h.window.updateGeortCoverageCloud(old);
 let oldResponse,newResponse;
 h.setCoverageResponse(()=>new Promise(resolve=>{if(!oldResponse)oldResponse=resolve;else newResponse=resolve;}));
 h.window.refreshGeortCoverage();await h.tick();
 h.window.updateGeortUI({connected:true,hand_model:{ready:true,calibrated:true},geort:{recording:{},coverage:old}});
 h.setCommandResponse(async()=>({ok:true,json:async()=>({id:'new',active:true,frames:0})}));
 h.scatter.length=0;await h.node('geortRecordStart').onclick();
 assert.equal(h.scatter.filter(p=>p.color==='#249b65').length,0);
 assert.equal(h.scatter.filter(p=>p.color==='#c6813d').length,1);
 assert.match(h.node('geortScatterStatus').textContent,/0 个采集点/);
 assert.match(h.node('geortCoverageScope').textContent,/仅显示当前一段/);
 oldResponse({ok:true,json:async()=>old});await h.tick();
 assert.equal(h.scatter.filter(p=>p.color==='#249b65').length,0);
 assert.equal(typeof newResponse,'function');
 newResponse({ok:true,json:async()=>({...old,baseline_ids:[],frames:1,fingers:[{finger:'thumb',cloud_mm:[[40,50,60]]}]})});await h.tick();
 assert.equal(h.scatter.filter(p=>p.color==='#249b65').length,1);
 assert.match(h.node('geortScatterStatus').textContent,/1 个采集点/);
});

test('failed recording start preserves the existing distribution',async()=>{
 const h=harness();await h.tick();
 h.window.updateGeortCoverageCloud({fingers:[{finger:'thumb',cloud_mm:[[10,20,30]]}]});
 h.setCommandResponse(async()=>({ok:false,json:async()=>({error:'手套未连接'})}));
 await h.node('geortRecordStart').onclick();
 assert.match(h.node('geortScatterStatus').textContent,/1 个采集点/);
 assert.match(h.node('geortError').textContent,/手套未连接/);
});

test('model selection requests GeoRT and playback pause/seek uses server controls',async()=>{
 const h=harness();await h.tick();h.node('geortModels').value='model';
 await h.node('geortSelect').onclick();assert.equal(h.window.wujiRequestedSolver,'geort');
 h.window.updateGeortUI({geort:{selected_model:'model'},wuji2:{playback:{active:true,playing:true,index:4,frames:20,seconds:1,duration_s:4}}});
 assert.match(h.node('geortReplayStatus').textContent,/动作回放/);
 await h.node('geortReplayPause').onclick();
 assert.deepEqual(h.requests.at(-1).body,{command:'geort_replay_pause',paused:true});
 h.node('geortReplaySeek').value='12';await h.node('geortReplaySeek').onchange();
 assert.equal(h.requests.at(-1).body.index,12);
});

test('delete sends only explicit selected IDs and respects confirmation cancellation',async()=>{
 const h=harness();await h.tick();h.node('geortDatasets').selectedOptions=[{value:'a'},{value:'b'}];
 h.setConfirmed(false);await h.node('geortDeleteDatasets').onclick();
 assert.equal(h.requests.some(r=>r.body?.command==='geort_recordings_delete'),false);
 h.setConfirmed(true);await h.node('geortDeleteDatasets').onclick();
 assert.deepEqual(h.requests.find(r=>r.body?.command==='geort_recordings_delete').body,{command:'geort_recordings_delete',ids:['a','b']});
 h.node('geortModels').value='active';h.window.updateGeortUI({geort:{selected_model:'active'}});
 await h.node('geortDeleteModel').onclick();assert.match(h.confirmations.at(-1),/当前网络预览将停止/);
 assert.deepEqual(h.requests.find(r=>r.body?.command==='geort_models_delete').body,{command:'geort_models_delete',ids:['active']});
});

test('collection panel uses scatter without coverage progress or percentage tables',async()=>{
 const h=harness();await h.tick();
 h.window.updateGeortUI({geort:{recording:{active:true},coverage:{ready:true,recording:true,baseline_ids:['a'],frames:500,segment:{frames:150},workspace:{status:'completed',samples:100000,accepted_samples:495,rejected_penetration_samples:28996,rejected_output_domain_samples:70509,rejected_invalid_samples:0,output_domain_filtered:true,penetration_threshold_mm:5}}}});
 assert.equal(h.created.includes('progress'),false);assert.equal(h.created.includes('tr'),false);
 assert.match(h.node('geortCoverageStatus').textContent,/500 帧/);
 assert.doesNotMatch(h.node('geortCoverageStatus').textContent,/%|优先补采/);
 assert.match(h.node('geortWorkspaceStatus').textContent,/保留 495/);
 assert.match(h.node('geortWorkspaceStatus').textContent,/输出域外剔除 70,509/);
 assert.match(h.node('geortWorkspaceStatus').textContent,/旧参考尚未按标定反伸范围筛选/);
 assert.equal(h.node('geortCoverageReset').disabled,true);
 h.window.updateGeortUI({geort:{coverage:{ready:true,workspace:{status:'completed',extension_domain_filtered:true,rejected_extension_samples:70509,reference_domain:{glove_extension_policy:{mcp_extension_deg:15,pip_extension_deg:10,dip_extension_deg:20,thumb_extension_deg:10}}}}}});
 assert.match(h.node('geortWorkspaceStatus').textContent,/过度反伸剔除 70,509/);
 assert.match(h.node('geortWorkspaceStatus').textContent,/反伸上限 MCP 15° \/ PIP 10° \/ DIP 20°/);
 assert.match(h.node('geortWorkspaceStatus').textContent,/已按标定反伸范围筛选/);
 h.window.updateGeortUI({geort:{coverage:{workspace:{status:'completed',extension_domain_filtered:true,reference_domain:{glove_extension_policy:{dip_extension_deg:20},robot_extension_policy:{mcp_extension_deg:0,pip_extension_deg:0,dip_extension_deg:0,thumb_extension_deg:0}}}}}});
 assert.match(h.node('geortWorkspaceStatus').textContent,/反伸上限 MCP 0° \/ PIP 0° \/ DIP 0°/);
 assert.match(h.node('geortWorkspaceStatus').textContent,/手指屈伸下限0°/);
});

test('reference rebuild keeps the same point at a fixed millimetre scale',async()=>{
 const h=harness();await h.tick();
 const source={finger:'thumb',cloud_mm:[[20,30,40]]};
 h.window.updateGeortCoverageCloud({fingers:[source],workspace:{fingers:[{finger:'thumb',reference_cloud_mm:[[0,0,0]]}]}});
 const before=h.scatter.filter(p=>p.color==='#249b65').at(-1);
 h.window.updateGeortCoverageCloud({fingers:[source],workspace:{fingers:[{finger:'thumb',reference_cloud_mm:[[220,180,200]]}]}});
 const after=h.scatter.filter(p=>p.color==='#249b65').at(-1);
 assert.equal(after.x,before.x);assert.equal(after.y,before.y);
 assert.match(h.node('geortScatterStatus').textContent,/固定毫米尺度/);
});

test('scatter draws source outside reference and while no robot reference is ready',async()=>{
 const h=harness();await h.tick();h.scatter.length=0;
 h.window.updateGeortCoverageCloud({fingers:[{finger:'thumb',cloud_mm:[[150,20,30]]}],workspace:{status:'completed',fingers:[{finger:'thumb',reference_cloud_mm:[[0,0,0]]}]}});
 assert.equal(h.scatter.length,2);
 assert.equal(h.scatter[0].color,'#aab4ab');assert.equal(h.scatter[1].color,'#249b65');
 assert.ok(h.scatter.every(p=>Number.isFinite(p.x)&&Number.isFinite(p.y)));
 h.scatter.length=0;h.node('geortShowReference').checked=false;h.node('geortShowReference').onchange();
 assert.equal(h.scatter.length,1);assert.equal(h.scatter[0].color,'#249b65');
 h.scatter.length=0;
 h.window.updateGeortCoverageCloud({fingers:[{finger:'thumb',cloud_mm:[[150,20,30]]}],workspace:{status:'running'}});
 assert.equal(h.scatter.length,1);
 h.window.updateGeortUI({geort:{coverage:{ready:true,frames:4,workspace:{status:'running',completed:100,total:1000}}}});
 assert.match(h.node('geortWorkspaceStatus').textContent,/已有录制点仍可查看/);
 assert.doesNotMatch(h.node('geortWorkspaceStatus').textContent,/%/);
});

test('training exposes GPU device and distinguishes negative cosine from zero targets',async()=>{
 const h=harness();await h.tick();
 h.window.updateGeortUI({geort:{training:{status:'running',device:{gpu_name:'NVIDIA RTX test'},losses:{total:-.9,direction:-.96,chamfer:.001,collision:0}}}});
 assert.match(h.node('geortTrainingStatus').textContent,/NVIDIA RTX test/);
 assert.match(h.node('geortLosses').textContent,/越接近 −1 越好/);
 assert.match(h.node('geortLosses').textContent,/原版未启用/);
});

test('coverage baseline requests selected sessions and clears explicitly',async()=>{
 const h=harness();await h.tick();h.node('geortDatasets').selectedOptions=[{value:'a'}];
 await h.node('geortCoverageSelect').onclick();
 assert.deepEqual(h.requests.find(r=>r.body?.command==='geort_coverage_select').body,{command:'geort_coverage_select',datasets:['a']});
 await h.node('geortCoverageReset').onclick();
 assert.deepEqual(h.requests.filter(r=>r.body?.command==='geort_coverage_select').at(-1).body,{command:'geort_coverage_select',datasets:[]});
});

test('surface diagnostics keep zero-gap penetration red and never change training',async()=>{
 const h=harness();await h.tick();
 h.window.updateGeortUI({geort:{diagnostics:{enabled:true,ready:true,input_kind:'live',timestamp_ms:123,processing_ms:20,device:'cuda:0',age_ms:10,source:{max_penetration_mm:7.7},robot:{query_valid:true,max_penetration_mm:0},pairs:[{finger:'index',source:{state:'penetrating',gap_mm:0,penetration_mm:7.7,normal_error_deg:0},robot:{state:'contact',gap_mm:.2,penetration_mm:0,normal_error_deg:10}}]}}});
 const row=h.node('geortDiagnosticsRows').children[0];
 assert.match(row.children[1].textContent,/穿透.*7.700/);assert.equal(row.children[1].className,'diagnostic-penetrating');
 assert.match(row.children[2].textContent,/接触近邻/);
 assert.match(h.node('geortDiagnosticsStatus').textContent,/源时间戳 123.0/);
 assert.equal(h.requests.some(r=>r.body?.command==='geort_train'),false);
 h.window.updateGeortUI({geort:{diagnostics:{enabled:true,ready:false,status:'stale',error:'诊断帧已过期',pairs:[{}]}}});
 assert.equal(h.node('geortDiagnosticsRows').children.length,0);
 assert.match(h.node('geortDiagnosticsStatus').textContent,/已过期/);
});

test('surface diagnostics toggle uses a separate command and shows initialization',async()=>{
 const h=harness();await h.tick();h.setCommandResponse(async()=>({ok:true,json:async()=>({enabled:true,ready:false,status:'starting'})}));
 await h.node('geortDiagnosticsToggle').onclick();
 assert.deepEqual(h.requests.at(-1).body,{command:'geort_diagnostics_set',enabled:true});
 assert.match(h.node('geortDiagnosticsStatus').textContent,/初始化/);
 assert.equal(h.node('geortDiagnosticsToggle').textContent,'停止表面诊断');
});

test('clicking an exact reference point previews its pose without solver or recording controls',async()=>{
 const h=harness();await h.tick();const poses=[];
 h.window.showGeortWorkspacePose=async pose=>poses.push(pose);
 h.setGetResponse(async url=>{const q=new URL(url,'http://local').searchParams;return {ok:true,json:async()=>({ready:true,reference_id:q.get('reference_id'),sample_id:Number(q.get('sample_id')),finger:q.get('finger'),point_mm:[0,0,0],cell_observed:false})};});
 h.window.updateGeortCoverageCloud({workspace:{reference_id:'ref',fingers:[{finger:'thumb',reference_cloud_mm:[[0,0,0]],reference_sample_ids:[71],reference_cloud_counts:[0]}]}});
 const point=h.scatter.at(-1),c=h.node('geortCoverageCanvas');
 c.onpointerdown({clientX:point.x,clientY:point.y});c.onpointerup({clientX:point.x,clientY:point.y});await h.tick();
 assert.equal(poses.length,1);assert.equal(poses[0].sample_id,71);
 assert.match(h.node('geortWorkspacePoseStatus').textContent,/尚未采集/);
 assert.equal(h.requests.some(r=>r.body),false);
 h.window.updateGeortCoverageCloud({workspace:{reference_id:'ref',fingers:[{finger:'thumb',reference_cloud_mm:[[0,0,0]],reference_sample_ids:[71],reference_cloud_counts:[4]}]}});
 assert.match(h.node('geortWorkspacePoseStatus').textContent,/已采集（4 帧）/);
 // Dragging the same point rotates the cloud and never issues another request.
 c.onpointerdown({clientX:point.x,clientY:point.y});c.onpointermove({clientX:point.x+20,clientY:point.y});c.onpointerup({clientX:point.x+20,clientY:point.y});await h.tick();
 assert.equal(poses.length,1);
});

test('missing-only filter and next-point inspection preserve original sample IDs',async()=>{
 const h=harness();await h.tick();const poses=[];let cleared=0;
 h.window.showGeortWorkspacePose=async pose=>poses.push(pose);h.window.clearGeortWorkspacePose=()=>cleared++;
 h.setGetResponse(async url=>{const q=new URL(url,'http://local').searchParams;return {ok:true,json:async()=>({ready:true,reference_id:'ref',sample_id:Number(q.get('sample_id')),finger:'thumb',point_mm:[1,2,3],cell_observed:false})};});
 h.window.updateGeortCoverageCloud({workspace:{reference_id:'ref',fingers:[{finger:'thumb',reference_cloud_mm:[[0,0,0],[30,0,0]],reference_sample_ids:[3,92],reference_cloud_counts:[8,0]}]}});
 h.node('geortOnlyMissing').checked=true;h.scatter.length=0;h.node('geortOnlyMissing').onchange();
 assert.equal(h.scatter.length,1);assert.equal(h.scatter[0].color,'#c6813d');
 h.node('geortMissingPose').onclick();await h.tick();assert.equal(poses[0].sample_id,92);
 h.window.updateGeortCoverageCloud({workspace:{reference_id:'new',fingers:[]}});
 assert.ok(cleared>0);assert.equal(h.node('geortClearPose').disabled,true);
});

test('an older async point response cannot replace a newer selection',async()=>{
 const h=harness();await h.tick();const finish=[],poses=[];
 h.window.showGeortWorkspacePose=async pose=>poses.push(pose);
 h.setGetResponse(url=>new Promise(resolve=>{const q=new URL(url,'http://local').searchParams;finish.push(()=>resolve({ok:true,json:async()=>({ready:true,reference_id:'ref',sample_id:Number(q.get('sample_id')),finger:'thumb',point_mm:[1,2,3]})}));}));
 h.window.updateGeortCoverageCloud({workspace:{reference_id:'ref',fingers:[{finger:'thumb',reference_cloud_mm:[[0,0,0],[30,0,0]],reference_sample_ids:[3,92],reference_cloud_counts:[0,0]}]}});
 h.node('geortMissingPose').onclick();h.node('geortMissingPose').onclick();
 finish[1]();await h.tick();finish[0]();await h.tick();
 assert.equal(poses.length,1);assert.equal(poses[0].sample_id,92);
});
