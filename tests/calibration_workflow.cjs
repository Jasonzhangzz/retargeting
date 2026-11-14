const {test}=require('node:test'),assert=require('node:assert/strict');
const {view}=require('../moxian_web/static/calibration-workflow.js');
const hand=()=>({status:'default',ready:false,capture:{state:'ready',source:'continuous_motion',index:33,total:34,actions_per_pass:17,completed_train:17,completed_validation:16,capture_pass:'validation',reference_ready:true,prompt:'五指伸直、张开并拢两次'}});
test('last outstanding repeat is an enabled capture action, never a silent disabled fit',()=>{
 const h=hand(),v=view(h);assert.equal(v.action,'capture');assert.equal(v.disabled,false);assert.equal(v.canFit,false);assert.match(v.label,/最后一次复测.*5/);assert.match(v.phase,/五指伸直/);
});
test('finishing the last repeat immediately enables fitting, independent of live mesh freshness or saving',()=>{
 const h=hand();h.capture.state='complete';h.capture.completed_validation=17;h.capture.index=34;
 h.capture_storage={active:true,current:{phase:'encoding',completed:64,total:6917}};
 const v=view(h);assert.equal(v.action,'fit');assert.equal(v.disabled,false);assert.equal(v.canFit,true);assert.match(v.phase,/64\/6917/);assert.equal(v.progress,64/6917);
});
test('capture, model preparation and cancellation display their real state',()=>{
 const h=hand();h.capture.state='collecting';h.capture.seconds_remaining=2;assert.equal(view(h).disabled,true);assert.equal(view(h).progress,.6);
 h.capture.state='preparing';h.capture.error='回放初始化 30/240';assert.match(view(h).phase,/30\/240/);
 h.fit={state:'cancelling'};assert.match(view(h).label,/取消/);assert.equal(view(h).disabled,true);
});
test('copy and compilation are visible at the main button, with real counts and no invented compilation percentage',()=>{
 const h=hand();h.fit={state:'running',phase:'copy',completed:64,total:7000,progress:.001,message:'准备快照',estimated_remaining_s:2};
 const v=view(h);assert.match(v.phase,/64\/7000/);assert.equal(v.remaining,2);assert.equal(v.progress,.001);
 h.fit.phase='compile';h.fit.progress=.5;assert.equal(view(h).progress,null);assert.match(view(h).phase,/GPU 编译/);
});
test('busy, offline and preview have explicit blocked reasons',()=>{
 const h=hand();assert.equal(view(h,true,true).disabled,true);assert.match(view(h,false).reason,/离线/);
 h.checkpoint_preview={label:'test'};assert.match(view(h).reason,/返回当前模型/);
 h.capture.reference_ready=false;assert.equal(view(h).disabled,true);
});
test('a submitted fit request has immediate feedback before the first worker report',()=>{
 const h=hand();h.capture.state='complete';const v=view(h,true,true,'mano_fit');
 assert.match(v.phase,/提交拟合请求/);assert.match(v.label,/启动拟合/);assert.equal(v.progress,null);assert.equal(v.disabled,true);
});
test('static holds advance through capture and only the new completed protocol can fit Qnbot',()=>{
 const h=hand();h.surface_backend='qnbot';h.capture={...h.capture,protocol:'contact_static_v5',source:'static_holds',window_seconds:2,index:35,total:36};
 assert.equal(view(h).action,'capture');assert.equal(view(h).disabled,false);
 h.capture.state='collecting';h.capture.motion_phase='hold';assert.match(view(h).phase,/检查稳定/);
 h.capture.state='complete';assert.equal(view(h).canFit,true);
 h.capture.protocol='contact_motion_v4';assert.equal(view(h).canFit,false);
});
test('first static pass offers EMF skeleton fitting and never silently starts a repeat',()=>{
 const h=hand();h.surface_backend='qnbot';h.capture={protocol:'contact_static_v5',source:'static_holds',state:'fit_ready',index:18,total:36,reference_ready:true,completed_train:17,completed_validation:0};
 const v=view(h);assert.equal(v.action,'fit');assert.equal(v.canFit,true);assert.equal(v.disabled,false);assert.match(v.label,/EMF.*骨架拟合/);
 h.fit={state:'completed',validation_pending:true};
 const next=view(h);assert.equal(next.action,'capture');assert.equal(next.canFit,false);assert.match(next.label,/开始独立复测/);
 h.capture.state='complete';const final=view(h);assert.equal(final.action,'fit');assert.equal(final.canFit,true);assert.match(final.label,/验收/);
});
test('a pending fit from another capture does not replace the new fit button with validation',()=>{
 const h=hand();h.surface_backend='qnbot';h.capture={protocol:'contact_static_v5',source:'static_holds',state:'fit_ready',metadata:{motion_session_id:'new'}};
 h.fit={state:'completed',validation_pending:true,capture_session_id:'old'};
 const v=view(h);assert.equal(v.action,'fit');assert.equal(v.canFit,true);
});
test('native L-BFGS reports evaluations and objective rather than a percentage of its evaluation limit',()=>{
 const h=hand();h.fit={state:'running',method:'bone_extrinsics',phase:'joint_fit',completed:300,evaluation_limit:20000,objective_value:123.45,evaluations_per_s:100,progress:.1};
 const v=view(h);assert.equal(v.progress,null);assert.match(v.phase,/300 次/);assert.match(v.phase,/123.450/);assert.doesNotMatch(v.phase,/20000/);
});

test('Wuji default keeps old 17 actions read-only and offers a fresh native capture',()=>{
 const h={...hand(),calibration_method:'wuji_native',surface_backend:'qnbot'};
 const v=view(h);assert.equal(v.action,'start');assert.equal(v.canFit,false);assert.match(v.label,/六动作/);assert.match(v.reason,/旧17动作/);
});
test('Wuji six actions advance automatically, finish at a manual fit and remain terminal after application',()=>{
 const h={calibration_method:'wuji_native',status:'ready',capture:{protocol:'wuji_studio_native_v1',state:'collecting',index:2,motion_phase:'hold',metadata:{motion_session_id:'native'}}};
 assert.equal(view(h).disabled,true);assert.match(view(h).phase,/3\/6.*保持1秒/);
 h.capture.state='complete';assert.equal(view(h).action,'fit');assert.equal(view(h).canFit,true);assert.match(view(h).phase,/不会自动重新采集/);
 h.fit={method:'wuji_native',accepted:true,applied:true,capture_session_id:'native'};assert.equal(view(h).action,'none');assert.equal(view(h).disabled,true);assert.match(view(h).label,/已应用/);
 h.capture.metadata.motion_session_id='new';assert.equal(view(h).canFit,true);
 h.fit={method:'wuji_native',state:'running',message:'原生 wujihand · 评估100次 · 目标值12'};assert.equal(view(h).progress,null);assert.match(view(h).phase,/评估100/);
});

test('Moxian six-pose fit never inherits acceptance from the withdrawn Wuji hand model',()=>{
 const h={calibration_method:'moxian_wuji',status:'ready',capture:{protocol:'wuji_studio_native_v1',state:'complete',metadata:{motion_session_id:'six'}},fit:{method:'wuji_native',accepted:true,applied:true,capture_session_id:'six'}};
 assert.equal(view(h).action,'fit');assert.equal(view(h).canFit,true);assert.match(view(h).label,/Moxian/);
 h.fit={method:'moxian_wuji',accepted:true,applied:true,capture_session_id:'six'};assert.equal(view(h).canFit,false);assert.match(view(h).label,/已应用/);
});

test('expanded Moxian joint model offers refitting the retained six poses while the old model stays active',()=>{
 const h={calibration_method:'moxian_wuji',calibration_fit_revision:'v2',status:'ready',capture:{protocol:'wuji_studio_native_v1',state:'complete',metadata:{motion_session_id:'six'}},fit:{method:'moxian_wuji',fit_revision:'v1',accepted:true,applied:true,capture_session_id:'six'}};
 assert.equal(view(h).action,'fit');assert.equal(view(h).canFit,true);
 h.fit.fit_revision='v2';assert.equal(view(h).canFit,false);
});
