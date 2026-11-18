/* One source of button eligibility and the visible reason for its next step. */
(function(root){
 'use strict';
 function nativeView(h,online,busy,pending){
  const method=h.calibration_method||'wuji_native',moxian=method==='moxian_wuji',cap=h.capture||{},fit=h.fit||{},native=cap.protocol==='wuji_studio_native_v1',working=['running','cancelling'].includes(fit.state);
  const unavailable=!online||['error','not_started','unavailable'].includes(h.status);
  let action='none',label='等待标定',reason='',phase='',disabled=busy||unavailable,progress=null,canFit=false;
  if(working){label=fit.state==='cancelling'?'正在取消原生标定…':'正在执行 Wuji 原生标定…';disabled=true;phase=fit.message||'原生objective与L-BFGS求解中';reason='使用Wuji原生21自由度模型；成功后应用原生URDF。';}
  else if(!native){action='start';label='开始 Wuji 原生六动作标定';reason='旧17动作记录保留为历史；原生objective需要重新采集六个指定手势。';phase='Wuji Studio 2026.7.1 原生标定';}
  else if(cap.state==='complete'){
   if(fit.method===method&&fit.accepted&&(!h.calibration_fit_revision||fit.fit_revision===h.calibration_fit_revision)&&fit.capture_session_id===cap.metadata?.motion_session_id){label=fit.applied?'Wuji 原生标定已应用':'原生标定成功，待应用';disabled=true;phase=fit.application_error||fit.reason||'完成';reason='可重新采集六动作或恢复上一模型。';progress=1;}
   else{action='fit';label='开始 Wuji 原生 EMF 与骨架拟合';reason='六动作采集完成，点击后直接执行原生objective和solver。';phase='数据就绪；不会自动重新采集';progress=1;canFit=!disabled;}
  }else if(['ready','paused'].includes(cap.state)){action='capture';label=cap.state==='paused'?'继续当前原生动作':'就位，开始自动六动作采集';phase=cap.error||cap.prompt;reason='通过后自动切换下一动作。';}
  else if(cap.state==='collecting'){label='原生动作采集中…';disabled=true;const names={waiting_movement:'等待变换手势',waiting_pose:'等待原生姿态约束',waiting_stable:'等待稳定',hold:'稳定保持1秒',collect:'采集1秒'};phase=`动作 ${Math.min((cap.index||0)+1,6)}/6 · ${names[cap.motion_phase]||'采集中'}`+(cap.constraints_ok===false?' · '+(cap.constraint_failures||[]).slice(0,2).join('；'):'');reason='姿态与稳定性满足时显示绿色通过。';}
  else{action='start';label='重新开始 Wuji 六动作标定';phase=cap.error||'请开始新标定';}
  if(h.checkpoint_preview){disabled=true;canFit=false;reason='先结束历史模型预览。';}
  if(unavailable){disabled=true;canFit=false;reason=online?(h.error||'模型未就绪'):'网页离线，请等待重新连接。';}
  if(pending==='mano_fit'&&!working){label='正在启动 Wuji 原生标定…';phase='已提交原生求解请求';disabled=true;canFit=false;}
  if(moxian){label=label.replace('Wuji 原生','Moxian 六动作').replace('原生标定','Moxian标定');reason=reason.replace('原生objective和solver','Moxian骨架目标与Wuji原生L-BFGS').replace('原生URDF','Moxian骨长与EMF外参');if(working)reason='Moxian25＋四指轴向扭转；末节可反折，使用原始mesh。';}
  return {action,label,disabled,reason,phase,progress,remaining:null,canFit};
 }
 function view(h={},online=true,busy=false,pending=null){
  if(['wuji_native','moxian_wuji'].includes(h.calibration_method)||h.capture?.protocol==='wuji_studio_native_v1')return nativeView(h,online,busy,pending);
  const cap=h.capture||{},fit=h.fit||{},storage=h.capture_storage||{},current=storage.current||{},warmup=h.gpu_warmup||{};
  const working=['running','cancelling'].includes(fit.state),missing=cap.missing_actions||[];
  const count=cap.actions_per_pass||16,train=cap.completed_train??Math.min(cap.index||0,count),valid=cap.completed_validation??Math.max(0,(cap.index||0)-count);
  const fitReady=cap.protocol==='contact_static_v5'&&cap.state==='fit_ready',pendingValidation=fit.validation_pending===true&&(!cap.metadata?.motion_session_id||fit.capture_session_id===cap.metadata.motion_session_id);
  const complete=cap.state==='complete'&&(h.surface_backend!=='qnbot'||cap.protocol==='contact_static_v5'),unavailable=!online||['error','not_started','unavailable'].includes(h.status);
  let action='none',label='等待采集状态',reason='',phase='',progress=null,remaining=null,disabled=busy||unavailable;
  if(working){
   label=fit.state==='cancelling'?'正在取消拟合…':'正在拟合…';disabled=true;
   const names={prepare:'数据准备',copy:'数据快照',compile:'GPU 编译',templates:'接触模板',validation:'独立验收'};
   phase=(names[fit.phase]||'参数更新')+' · '+(fit.message||'正在处理');
   const total=fit.total,done=fit.completed;
   if(Number.isFinite(total)&&Number.isFinite(done))phase+=` · ${done}/${total}`;
   progress=fit.phase==='compile'?null:Number.isFinite(fit.progress)?fit.progress:null;
   if(fit.method==='bone_extrinsics'&&fit.phase==='joint_fit'){
    phase=`EMF与骨架拟合 · 已评估 ${fit.completed||0} 次`;
    if(Number.isFinite(fit.objective_value))phase+=` · 目标值 ${fit.objective_value.toPrecision(6)}`;
    if(Number.isFinite(fit.evaluations_per_s))phase+=` · ${fit.evaluations_per_s.toFixed(1)} 次/秒`;
    progress=null;
   }
   remaining=fit.estimated_remaining_s;reason='拟合进度会持续更新；通过完整验收才会应用参数。';
  }else if(fitReady){
   action=pendingValidation?'capture':'fit';label=pendingValidation?'开始独立复测':'开始 EMF 与骨架拟合';
   reason=pendingValidation?'拟合候选已保存；独立复测通过后才应用。':'17个动作已采集完成；点击后优化骨长和EMF外参。';
   phase=pendingValidation?'EMF与骨架拟合完成，等待独立复测':'采集完成，等待点击拟合';progress=1;
  }else if(complete){
   action='fit';label=cap.protocol==='contact_static_v5'?(pendingValidation?'验收 EMF 与骨架拟合结果':'开始 EMF 与骨架拟合'):'开始拟合';reason='采集已完成，可以开始拟合或验收。';phase='数据就绪';progress=1;
   if(warmup.state==='running'){phase='GPU 预热 · '+(warmup.message||'编译计算图');progress=null;reason+=' 点击后会先等待预热，并显示训练进度。';}
  }else if(cap.state==='preparing'){
   label='正在准备参考模型…';phase=cap.error||'正在恢复参考模型';disabled=true;
  }else if(cap.state==='collecting'){
   label='正在采集…';phase=`${cap.capture_pass==='validation'?'独立复测':'拟合采集'} · 已识别 ${cap.detected_repetitions||0}/2 次 · 窗口剩余 ${Math.ceil(cap.seconds_remaining||0)} 秒`;
   progress=Math.max(0,Math.min(1,1-(cap.seconds_remaining||0)/(cap.window_seconds||5)));disabled=true;
   if(cap.extended)phase+=` · 自动延长，等待${cap.waiting_for||'展平回位'}`;
  }else if(['ready','paused'].includes(cap.state)&&['continuous_motion','static_holds'].includes(cap.source)){
   action='capture';const last=(cap.total-cap.index)===1;
   label=last?`开始最后一次复测（${cap.window_seconds||5} 秒）`:'继续采集当前动作';
   if(cap.state==='paused')label=last?'补录最后一次复测':'补录当前动作';
   reason=last?'尚未开始拟合：只缺最后一次独立复测。完成后此按钮会变为“开始拟合”。':`尚未开始拟合：还缺 ${Math.max(0,count-train)} 个拟合动作、${Math.max(0,count-valid)} 个复测动作。`;
   phase=(cap.capture_pass==='validation'?'待采独立复测：':'待采拟合动作：')+(missing[0]?.label||cap.prompt||'请按下方示意操作');
   disabled=disabled||!cap.reference_ready;
  }else{disabled=true;reason='请先完成联合标定采集。';}
  if(h.surface_backend==='qnbot'&&cap.protocol!=='contact_static_v5'&&!working){action='none';disabled=true;label='请开始静态17动作';reason='历史记录只读，请重新采集静态17动作。';}
  if(storage.active&&!working){
   if(current.phase==='encoding'){
    const done=current.completed||0,total=current.total||0;
    phase=`后台保存：编码 ${done}/${total} 帧`;progress=total?done/total:null;
   }else if(current.phase==='copying'){phase=`后台保存：提交 ${current.files_completed||0}/${current.files_total||0} 份文件`;progress=null;}
   remaining=current.estimated_remaining_s;
   if(complete)reason+=' 保存与拟合相互独立。';
  }
  if(h.checkpoint_preview){disabled=true;reason='先返回当前模型，再开始拟合。';}
  if(cap.protocol==='contact_static_v5'&&cap.state==='collecting'){progress=(cap.stability_frames||0)/(cap.stability_frames_required||20);phase=`${cap.capture_pass==='validation'?'独立复测':'拟合采集'} · ${{waiting_pose:'等待目标姿态',waiting_stable:'等待稳定',hold:'检查稳定',collect:'采集完成'}[cap.motion_phase]||'静态保持'} · 稳定窗口 ${cap.stability_frames||0}/${cap.stability_frames_required||20} 帧`+(cap.constraints_ok===false?' · '+(cap.constraint_failures||[]).slice(0,2).join('；'):'');}
  if(pending==='mano_fit'&&!working){label='正在启动拟合…';phase='已提交拟合请求，等待工作进程确认';progress=null;disabled=true;}
  if(unavailable)reason=!online?'网页服务离线，等待重新连接。':'模型工作进程未就绪：'+(h.error||h.status);
  if(storage.error){phase='保存失败：'+storage.error;progress=null;}
  return {action,label,disabled,reason,phase,progress,remaining,canFit:(complete||fitReady&&!pendingValidation)&&!working&&!busy&&!unavailable&&!h.checkpoint_preview};
 }
 root.MoxianCalibrationWorkflow={view};
 if(typeof module!=='undefined')module.exports=root.MoxianCalibrationWorkflow;
})(typeof window==='undefined'?globalThis:window);
