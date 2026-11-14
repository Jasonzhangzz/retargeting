/* Repo-native vector instructions, matching the existing gesture guide. */
(function(root,definition){
 'use strict';
 const fingers=['thumb','index','middle','ring','pinky'],labels=['拇指','食指','中指','无名指','小指'];
 let targets=definition;
 const expected=stage=>targets?.stages?.[stage];
 function sequence(stage){
  if(stage==='motion_spread_together')return [['extend','五指伸直','保持伸直，不握拳'],['spread','自然张开',''],['together','并拢、稍作停留',''],['spread','松开并展平张开','两次完整循环 · 至少6秒']];
  if(stage.startsWith('contact_'))return [['apart','全手展平',''],['approach','自然接近','连续记录目标patch位置与法向'],['touch','贴合，稍作停留',''],['release','松开，回到全手展平','两次完整循环 · 至少6秒，不足自动延长；不规定速度或保持时长']];
  if(stage.startsWith('pinch_'))return [['apart','分开','1 秒'],['approach','缓慢接近','1 秒'],['touch','指腹轻触','1.5 秒'],['release','释放分开','1.5 秒']];
  if(stage==='thumb_opposition')return [['spread','拇指外展',''],['oppose','扫向各指',''],['return','自然展开','']];
  if(stage.endsWith('_repeat'))return [['hold',stage==='open_repeat'?'自然张开':'伸直并拢','保持 4 秒']];
  return [['extend','伸直',''],['curl','弯曲／钩状',''],['return','再次伸直','']];
 }
 function illustration(stage,phase,side){
  if(stage.startsWith('contact_'))return contactIllustration(stage,phase,side);
  const target=stage.startsWith('pinch_')?fingers.indexOf(stage.slice(6)):stage.startsWith('range_')?fingers.indexOf(stage.slice(6)):0;
  const pinch=stage.startsWith('pinch_'),closed=pinch&&['approach','touch'].includes(phase);
  let paths=[[[72,175],[49,155],[34,132],[26,110]],[[84,132],[80,95],[78,66],[78,46]],[[111,124],[110,84],[110,50],[110,30]],[[140,126],[141,91],[141,62],[141,43]],[[167,137],[176,111],[181,86],[183,67]]];
  if(stage==='motion_spread_together'){
   if(phase==='together'||phase==='extend')paths=[[[72,175],[61,153],[52,133],[48,115]],[[84,132],[87,95],[88,66],[89,46]],[[111,124],[110,84],[110,50],[110,30]],[[140,126],[133,91],[132,62],[131,43]],[[167,137],[155,111],[153,86],[151,67]]];
   if(phase==='spread')for(let f=1;f<5;f++)for(let k=1;k<4;k++)paths[f][k][0]+=(f-2.5)*k*5;
  }
  if(stage==='together_repeat')paths=[paths[0],[[87,132],[87,94],[87,66],[87,46]],[[112,127],[112,84],[112,50],[112,30]],[[137,129],[137,91],[137,62],[137,43]],[[162,139],[162,108],[162,83],[162,65]]];
  if(stage.startsWith('range_')&&phase==='curl'){
   const [x,y]=paths[target][0];paths[target]=target?[[x,y],[x-10,y+18],[x+4,y+30],[x+13,y+12]]:[[72,175],[67,149],[84,142],[95,155]];
  }
  if(stage==='thumb_opposition'&&phase==='oppose')paths[0]=[[72,175],[81,151],[108,137],[134,125]];
  let contact=null;
  if(closed){
   const x=[0,83,111,137,159][target],y=[0,104,106,116,132][target],gap=phase==='touch'?0:12;
   const base=paths[target][0];paths[target]=[base,[base[0]+7,base[1]-27],[x+11,y-15],[x+gap,y]];
   paths[0]=[[72,175],[57,151],[x-17,y+9],[x-gap,y]];if(phase==='touch')contact=[x,y];
  }
  const muted='#a2ac9c',ink='#697b62',accent='#dc592d';
  const id=('motion-'+stage+'-'+phase+'-'+side).replace(/[^a-zA-Z0-9_-]/g,'');
  let body='<path d="M76 215 L72 174 Q58 155 71 137 L84 129 L110 120 L140 123 L169 137 Q179 155 170 181 L157 215" fill="#e7ece1" stroke="'+ink+'" stroke-width="2"/>';
  for(let f=0;f<5;f++){
   const active=stage==='motion_spread_together'||stage.endsWith('_repeat')||f===target||pinch&&f===0;
   const color=active?accent:muted;
   body+='<path d="M'+paths[f].map(p=>p.join(',')).join(' L')+'" fill="none" stroke="'+color+'" stroke-width="13" stroke-linecap="round" stroke-linejoin="round"/>';
   body+=paths[f].slice(0,3).map(p=>'<circle cx="'+p[0]+'" cy="'+p[1]+'" r="2.4" fill="#fafaf5"/>').join('');
  }
  const spec=expected(stage);
  if(spec?.kind==='opposing_pads')for(const f of [0,target]){
   const a=paths[f][2],b=paths[f][3],x=.3*a[0]+.7*b[0],y=.3*a[1]+.7*b[1];
   body+='<ellipse data-target-patch="'+spec.patches[f===0?0:1]+'" cx="'+x+'" cy="'+y+'" rx="10" ry="7" fill="'+(phase==='touch'?'#ffe15b':'#fff3af')+'" stroke="#b34324" stroke-width="2"'+(phase==='touch'?'':' stroke-dasharray="3 2"')+'/>';
  }
  if(spec?.kind==='side_prior')for(let f=1;f<4;f++)for(const [finger,offset] of [[f,7],[f+1,-7]]){
   const a=paths[finger][1],b=paths[finger][2];body+='<path data-target-region="adjacent_finger_side" d="M'+(a[0]+offset)+','+a[1]+' L'+(b[0]+offset)+','+b[1]+'" stroke="#e3a322" stroke-width="4"/>';
  }
  if(contact)body+='<circle cx="'+contact[0]+'" cy="'+contact[1]+'" r="17" fill="none" stroke="'+accent+'" stroke-width="2" stroke-dasharray="4 3"/>';
  const arrow=pinch?(phase==='release'?'M73 84 Q40 72 22 90':phase==='approach'?'M24 88 Q45 66 68 84':null):stage.endsWith('_repeat')?null:stage==='thumb_opposition'?'M29 95 Q90 55 151 98':phase==='curl'?'M198 75 Q220 120 186 160':'M192 155 Q215 110 190 72';
  if(arrow)body+='<path d="'+arrow+'" fill="none" stroke="'+accent+'" stroke-width="2.5" marker-end="url(#'+id+')"/>';
  const transform=side==='left'?'translate(230,0) scale(-1,1)':'';
  return '<svg viewBox="0 0 230 230" role="img" aria-label="'+(side==='left'?'左手':'右手')+' '+(pinch?'拇指与'+labels[target]+'对指':stage.startsWith('range_')?labels[target]+'活动范围':'手势')+'示意"><defs><marker id="'+id+'" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10" fill="'+accent+'"/></marker></defs><g transform="'+transform+'">'+body+'</g></svg>';
 }
 function contactIllustration(stage,phase,side){
  const spec=expected(stage);if(!spec)return '';
  const active=spec.active_fingers||[],closed=['approach','touch'].includes(phase),fold=stage==='contact_fold',fist=stage==='contact_fist';
  let paths=[[[72,175],[49,155],[34,132],[26,110]],[[84,132],[80,95],[78,66],[78,46]],[[111,124],[110,84],[110,50],[110,30]],[[140,126],[141,91],[141,62],[141,43]],[[167,137],[176,111],[181,86],[183,67]]];
  const mcp={index:[84,146],middle:[110,140],ring:[138,142],pinky:[162,151]};
  const palmFinger=stage.startsWith('contact_palm_')?stage.slice(13):null;
  if(closed){
   if(fist||fold){for(let f=1;f<5;f++){const [x,y]=paths[f][0];paths[f]=fold?[[x,y],[x-5,y-28],[x+16,y-32],[x+10,y-13]]:[[x,y],[x-7,y-26],[x+16,y-10],mcp[fingers[f]]];}}
   else if(palmFinger){const [x,y]=mcp[palmFinger];paths[0]=[[72,175],[65,150],[x-16,y+8],[x,y]];}
   else{
    const fs=active.filter(f=>f!=='thumb'),cx=fs.reduce((a,f)=>a+paths[fingers.indexOf(f)][0][0],0)/fs.length;
    fs.forEach((f,k)=>{const i=fingers.indexOf(f),base=paths[i][0],x=cx+(k-(fs.length-1)/2)*15;paths[i]=[base,[base[0]+8,base[1]-27],[x+10,88],[x,106+(k%2)*6]];});
    paths[0]=[[72,175],[64,149],[cx-18,122],[cx-5,113]];
   }
  }
  if(closed&&(fist||fold)){for(let f=1;f<5;f++){const path=paths[f],point=fold?path[0].map((v,k)=>.35*v+.65*path[1][k]):mcp[fingers[f]];path[3]=point.map((v,k)=>(v-.35*path[2][k])/.65);}}
  if(closed&&!palmFinger&&!fist&&!fold&&active.length===2){const f=fingers.indexOf(active[1]),point=[paths[f][3][0],110];for(const i of [0,f])paths[i][3]=point.map((v,k)=>(v-.35*paths[i][2][k])/.65);}
  if(phase==='approach')for(let i=0;i<5;i++)if(active.includes(fingers[i]))for(const k of [2,3])paths[i][k][1]+=i===0?6:-6;
  let body='<path d="M76 215 L72 174 Q58 155 71 137 L84 129 L110 120 L140 123 L169 137 Q179 155 170 181 L157 215" fill="#e7ece1" stroke="#697b62" stroke-width="2"/>';
  paths.forEach((path,i)=>{const on=active.includes(fingers[i]);body+='<path data-finger="'+fingers[i]+'" data-observation-weight="'+(on?1:0)+'" d="M'+path.map(p=>p.join(',')).join(' L')+'" fill="none" stroke="'+(on?'#dc592d':'#a2ac9c')+'" stroke-width="13" stroke-linecap="round" stroke-linejoin="round"/>';});
  for(const region of spec.patches){
   let point;
   if(region.startsWith('palm_mcp_'))point=mcp[region.slice(9)];
   else{const i=fingers.indexOf(region.split('_')[0]),path=paths[i],segment=region.endsWith('_proximal')?0:2;point=path[segment].map((v,k)=>.35*v+.65*path[segment+1][k]);}
   body+='<ellipse data-target-patch="'+region+'" cx="'+point[0]+'" cy="'+point[1]+'" rx="9" ry="6" fill="#ffe15b" stroke="#8d4824" stroke-width="2" stroke-dasharray="3 2"/>';
  }
  if(spec.kind==='distal_cluster'&&closed)body+='<ellipse cx="'+paths[0][3][0]+'" cy="108" rx="'+(active.length*8)+'" ry="20" fill="none" stroke="#916b14" stroke-dasharray="3 3"/>';
  const note=fist||fold?'<text x="12" y="227" fill="#697b62" font-size="10">拇指自然向外 · EMF 权重 0</text>':'';
  const transform=side==='left'?'translate(230,0) scale(-1,1)':'';
  return '<svg viewBox="0 0 230 235" role="img" aria-label="'+(side==='left'?'左手':'右手')+' '+spec.label+'示意"><g transform="'+transform+'">'+body+'</g>'+note+'</svg>';
 }
 function trajectoryMarkup(cap){
  const data=cap.trajectory||[];if(data.length<2)return '<p class="footnote">等待轨迹；接触帧由完整接近—松开过程识别。</p>';
  const first=data[0].t_ms,last=data.at(-1).t_ms,lo=Math.min(...data.map(p=>p.gap_m)),hi=Math.max(lo+.002,...data.map(p=>p.gap_m));
  const X=t=>15+570*(t-first)/Math.max(1,last-first),Y=g=>65-50*(g-lo)/(hi-lo);
  const points=data.map(p=>X(p.t_ms)+','+Y(p.gap_m)).join(' ');
  const flats=data.filter(p=>p.flat).map(p=>'<circle cx="'+X(p.t_ms)+'" cy="'+Y(p.gap_m)+'" r="2" fill="#6578c9"/>').join('');
  const events=(cap.contact_events||[]).filter(e=>e.timestamp_ms>=first&&e.timestamp_ms<=last).map(e=>'<circle data-contact-event="'+e.repetition+'" cx="'+X(e.timestamp_ms)+'" cy="'+Y(e.gap_m)+'" r="5" fill="#dc592d"/>').join('');
  const patchText=Object.entries(data.at(-1).patches||{}).map(([name,p])=>name+'：位置 ['+p.position_m.map(x=>(x*1000).toFixed(1)).join(', ')+'] mm · 法向 ['+p.normal.map(x=>x.toFixed(2)).join(', ')+']').join('；');
  return '<p class="footnote">连续关键patch（冻结参考推算）：'+patchText+'</p><svg viewBox="0 0 600 85" role="img" aria-label="冻结参考模型距离轨迹及动作极小值"><polyline points="'+points+'" fill="none" stroke="#287e63" stroke-width="2"/>'+flats+events+'<text x="15" y="82" font-size="11" fill="#697b62">参考距离 '+(lo*1000).toFixed(1)+'–'+(hi*1000).toFixed(1)+' mm · '+(cap.stage_id==='motion_spread_together'?'红点：并拢时刻；不施加指尖接触等式':'红点：动作假定接触 · 蓝点：全手展平；非实测皮肤真值')+'</text></svg>'+patchCurves(data);
 }
 function patchCurves(data){
  const names=Object.keys(data.at(-1)?.patches||{}),colors=['#287e63','#dc592d','#8751a1','#377ca2','#9a751a','#b44f7c'];
  if(!names.length)return '';
  const first=data[0].t_ms,last=data.at(-1).t_ms,X=t=>20+560*(t-first)/Math.max(1,last-first);
  function chart(normal){
   const series=names.map(name=>{const start=data.find(p=>p.patches?.[name]?.valid)?.patches[name];if(!start)return [];
    return data.filter(p=>p.patches?.[name]?.valid).map(p=>{const r=p.patches[name];const value=normal?Math.acos(Math.max(-1,Math.min(1,r.normal.reduce((s,x,i)=>s+x*start.normal[i],0))))*180/Math.PI:Math.hypot(...r.position_m.map((x,i)=>x-start.position_m[i]))*1000;return [p.t_ms,value];});});
   const max=Math.max(1,...series.flat().map(p=>p[1]));return '<svg viewBox="0 0 600 100" role="img" aria-label="'+(normal?'patch法向变化':'patch位置变化')+'">'+series.map((points,i)=>'<polyline fill="none" stroke="'+colors[i%colors.length]+'" stroke-width="1.5" points="'+points.map(([t,v])=>X(t)+','+(75-60*v/max)).join(' ')+'"/>').join('')+'<text x="20" y="96" font-size="11" fill="#697b62">'+(normal?'法向相对起点变化 °':'位置相对起点位移 mm')+' · 0–'+max.toFixed(1)+'</text></svg>';
  }
  return chart(false)+chart(true)+'<p class="footnote">'+names.map((n,i)=>'<span style="color:'+colors[i%colors.length]+'">'+n+'</span>').join(' · ')+'</p>';
 }
 function markup(stage,phase,side){
  const spec=expected(stage),hint=stage.startsWith('contact_')?'虚线黄色：期望接触区域；实测触觉另以实心黄色小球显示。'+(spec?.kind==='distal_cluster'?'形成连通指尖簇，尽量各指与拇指接触。':''):(spec?(spec.kind==='opposing_pads'?'黄色：'+spec.label+'；仅“轻触”且双侧触觉有效时要求贴合。':spec.label):'接触目标定义加载中…');
  return sequence(stage).map(([key,title,time],i)=>'<figure class="motion-step'+(key===phase?' active':'')+'"><span class="motion-step-number">'+(i+1)+'</span>'+illustration(stage,key,side)+'<figcaption><strong>'+title+'</strong><small>'+time+'</small></figcaption></figure>').join('')+'<p class="motion-target-note">'+hint+'</p>';
 }
 function staticMarkup(stage,endpoint,side){
  const phase=stage==='motion_spread_together'?endpoint:'touch';
  const label=stage==='motion_spread_together'?(endpoint==='spread'?'五指伸直张开':'五指伸直并拢'):expected(stage)?.label||'目标姿态';
  return '<figure class="motion-step active">'+illustration(stage,phase,side)+'<figcaption><strong>'+label+'</strong><small>就位后保持静止：最近20帧稳定后自动采集</small></figcaption></figure><p class="motion-target-note">黄色虚线为动作要求的接触区域；实测触觉单独显示。</p>';
 }
 function overview(side){return (targets?.protocols?.contact_motion_v4?.stages||targets?.protocols?.contact_motion_v3_spread_v1?.stages||[]).map(stage=>'<figure class="motion-step">'+illustration(stage,stage==='motion_spread_together'?'together':'touch',side)+'<figcaption><strong>'+(stage==='motion_spread_together'?'五指伸直，分别保持张开与并拢':expected(stage).label)+'</strong><small>20帧稳定即采集 · '+(stage==='motion_spread_together'?'张开与并拢分别保持':'目标姿态保持静止')+'</small></figcaption></figure>').join('');}
 root.MoxianMotionGuide={sequence,illustration,markup,staticMarkup,overview,expected,trajectoryMarkup,get revision(){return targets?.revision||'loading';}};
 if(!targets&&root.fetch)root.fetch('/calibration-contact-targets.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('接触目标定义读取失败');return r.json();}).then(value=>{targets=value;}).catch(error=>{root.MoxianMotionGuide.error=error.message;});
})(typeof window==='undefined'?globalThis:window,typeof module!=='undefined'&&module.exports?require('./calibration-contact-targets.json'):null);
