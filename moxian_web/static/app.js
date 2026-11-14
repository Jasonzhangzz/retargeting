'use strict';
const $ = id => document.getElementById(id);
const fingers = ['thumb', 'index', 'middle', 'ring', 'pinky'];
const names = ['拇指', '食指', '中指', '无名指', '小指'];
const colors = ['#ed875b', '#d0bc74', '#a8c794', '#7aafa4', '#81a0c6'];
let state = null, heatMode = 'hand', busy = false, lastSequence = null, history = [];
let lastFetch = 0, apiAlive = false, toastTimer,pendingCommand=null;
let statePollPending=false,statePollTimer;
let senseMode = 'contact', poseMode = 'mano', mapping = null, mappingDirty = false, selectedSids = new Set();
const requestedView=new URLSearchParams(window.location.search).get('view');
const requestedSolver=new URLSearchParams(window.location.search).get('solver');
let initialSolverApplied=false;
if([...$('poseMode').options].some(option=>option.value===requestedView))poseMode=requestedView;
$('poseMode').value=poseMode;
let mappingLoading = false, sceneInitialized = false;
let returnToManoAfterCalibration=null;
let replaying=false,replayFrames=[],replayIndex=0,replayPlaying=false,replayLast=0;
const trails = Object.fromEntries(fingers.map(f => [f, []]));
$('fingerRows').innerHTML = fingers.map((f, i) => `<tr id="row-${f}"><td><i style="background:${colors[i]}"></i>${names[i]}</td><td>等待</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>`).join('');

function toast(message) {
  $('toast').textContent = message; $('toast').hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').hidden = true, 3500);
}
async function command(command, data = {}) {
  if(replaying){toast('正在回放，返回实时后可操作设备。');return;}
  if (busy) return;
  busy = true;pendingCommand=command;if(state)updateUI();
  try {
    const response = await fetch('/api/command', {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({command, ...data}), signal: AbortSignal.timeout(6500)});
    const result = await response.json();
    if (!response.ok) throw Error(result.error || '操作失败');
    if (command === 'baseline') toast('正在采集 50 帧，请保持手套无接触。');
    if (command === 'clear_baseline') toast('已清除当前设备基线。');
    if (command === 'record_start') toast('开始记录真实姿态与触觉快照。');
    if (result.recording&&state)state.recording=result.recording;
    if (command === 'record_stop') toast(result.recording?.saving?'正在后台保存，完成后可下载 JSONL。':'录制已保存，可下载 JSONL。');
    return result;
  } catch (error) { toast(error.message); }
  finally { busy = false;pendingCommand=null;if(state)updateUI(); }
}
$('connect').onclick = () => command(state?.connected || ['waiting','connecting','stale'].includes(state?.status) ? 'disconnect' : 'connect');
$('mobileConnect').onclick = () => $('connect').click();
$('baseline').onclick = () => command('baseline');
$('clearBaseline').onclick = () => command('clear_baseline');
$('record').onclick = () => command(state?.recording?.active ? 'record_stop' : 'record_start');

function updateUI() {
  const live = !!(apiAlive && state?.connected);
  if(returnToManoAfterCalibration&&state?.skeleton?.ready&&state.skeleton.capture?.session_state==='completed'){poseMode='mano';$('poseMode').value='mano';returnToManoAfterCalibration=null;sceneInitialized=false;}
  const messages = {connecting:'重新连接中', waiting:'等待数据', disconnected:'已断开', stale:'数据已过期', streaming:'数据流在线'};
  const status = apiAlive ? messages[state?.status] || '等待连接' : 'SDK 服务离线';
  $('liveText').textContent = status;
  $('deviceStatus').textContent = status;
  $('deviceDot').classList.toggle('on', live);
  $('liveBadge').querySelector('.dot').classList.toggle('on', live);
  $('connect').textContent = state?.status !== 'disconnected' && apiAlive ? '断开手套' : '连接手套';
  $('mobileConnect').textContent = state?.status !== 'disconnected' && apiAlive ? '断开' : '连接';
  $('port').textContent = state?.device?.port || '—';
  $('serial').textContent = state?.device?.serial || '—';
  $('side').textContent = state?.side ? (state.side === 'left' ? '左手 / LEFT' : '右手 / RIGHT') : '—';
  $('heatSide').textContent = state?.side ? (state.side === 'left' ? '左手 · 固件报告' : '右手 · 固件报告') : '等待手侧信息';
  $('warning').hidden = !state?.side_mismatch;
  $('warning').textContent = '固件报告为'+(state?.side==='right'?'右手':'左手')+'；USB PID ' + (state?.device?.pid || '') + ' 标识为' + (state?.device?.usb_side === 'right' ? '右手' : '左手') + '，与实际及固件报告不一致。骨骼 IK 使用固件报告的手侧。';
  $('error').hidden = apiAlive && !state?.error;
  $('error').textContent = apiAlive ? state?.error || '' : '无法连接本地 SDK 服务，请运行 scripts/start_moxian_web.sh。';
  $('hz').textContent = live ? (state.hz || 0).toFixed(1) : '—';
  const valid = fingers.filter(f => state?.pose?.raw_valid_mask?.[f]).length;
  $('valid').textContent = live && state?.pose ? valid : '—';
  $('peak').textContent = live && state?.tactile ? state.tactile.peak : '—';
  $('active').textContent = live && state?.tactile ? state.tactile.active : '—';
  $('streamNote').textContent = live ? `累计 ${Number(state.frames).toLocaleString()} 帧` : '等待实时数据';
  $('poseStatus').textContent = live ? `${valid} / 5 指尖有效` : status;
  $('poseEmpty').hidden = live && !!state?.pose;
  $('heatEmpty').hidden = live && !!state?.tactile;
  $('poseEmptyText').textContent = live ? '等待手套姿态数据' : status;
  $('heatEmptyText').textContent = live ? '等待触觉数据' : status;
  $('baselineTag').textContent = state?.tactile?.has_baseline ? '去基线' : '原始值';
  $('baselineStatus').textContent = state?.baseline_collecting ? '采集基线中，请勿触碰手套…' : state?.tactile?.has_baseline ? '已应用当前设备的静态基线' : '尚未去除静态基线';
  $('baseline').disabled = !live || state?.baseline_collecting;
  $('baseline').textContent = state?.baseline_collecting ? '采集中…' : '采集基线';
  const recording = state?.recording;
  $('record').disabled = busy||recording?.saving||(!live && !recording?.active);
  $('record').innerHTML = pendingCommand==='record_start'?'正在开始录制…':pendingCommand==='record_stop'||recording?.saving?'正在保存…':recording?.active ? '<span>■</span> 停止录制' : '<span>●</span> 开始录制';
  $('recordIndicator').classList.toggle('on', !!recording?.active);
  $('recordInfo').textContent = recording?.error?recording.error:recording?.saving?`后台保存中 · 已写 ${recording.frames} 帧 · 待写 ${recording.queued_frames||0} 帧`:recording?.active ? `${recording.frames} 个快照 · ${recording.seconds.toFixed(1)} s` : recording?.file ? `已保存 ${recording.frames} 个快照` : '保存姿态、触觉与时间戳快照。';
  $('export').hidden = !recording?.file || !!recording.active||!!recording.saving;
  $('footerState').textContent = live ? `V1 · CRC 错误 ${state.crc_errors} · 数据年龄 ${state.age_ms?.toFixed(0)} ms` : status;
  for (const [i, finger] of fingers.entries()) {
    const cells = $(`row-${finger}`).children;
    const pose = state?.pose?.fingertip_local?.[finger];
    const isValid = live && state?.pose?.raw_valid_mask?.[finger];
    cells[1].textContent = !live ? '离线' : isValid ? '● 有效' : state?.pose?.held_mask?.[finger] ? '保持旧值' : '无效';
    cells[1].style.color = isValid ? '#6d8b5e' : '#b08f6c';
    [2,3,4].forEach((c,k) => cells[c].textContent = live && pose ? (pose.position[k]*1000).toFixed(1) : '—');
    cells[5].textContent = live && pose ? pose.quaternion_xyzw.map(x => x.toFixed(3)).join('  ') : '—';
  }
  if (live && state.sequence !== lastSequence) {
    lastSequence = state.sequence;
    history.push(senseMode === 'contact' ? (contactForView()?.ready ? contactForView().count/20*255 : 0) : state.tactile.peak); if (history.length > 100) history.shift();
    $('spark').querySelector('path').setAttribute('d', history.map((v,i) => `${i ? 'L' : 'M'}${i*130/99},${33 - v/255*30}`).join(' '));
    for (const finger of fingers) {
      if (!state.pose.raw_valid_mask[finger]) { trails[finger] = []; continue; }
      trails[finger].push(state.pose.fingertip_local[finger].position.map(x=>x*1000));
      if (trails[finger].length > 18) trails[finger].shift();
    }
  }
  updateModes(live);
  window.currentMoxianState=state;window.updateCheckpointComparison?.(state);window.updateContactPrediction?.(state);updateHandUI(live);if(poseMode!=='mano')window.updateManoView?.(state,false);if(poseMode!=='wuji2')window.updateWujiView?.(state,false);
  if(replaying){$('liveText').textContent='录制回放 · 本地文件';$('deviceStatus').textContent='录制回放';$('footerState').textContent='REPLAY · 不操作设备';}
}

async function poll() {
  if(statePollPending)return;
  if(document.hidden){scheduleStatePoll(1000);return;}
  if(replaying){scheduleStatePoll(100);return;}
  statePollPending=true;
  const start = performance.now();
  try {
    const response = await fetch('/api/state?view='+encodeURIComponent(poseMode)+'&geometry=packed', {cache:'no-store', signal:AbortSignal.timeout(2500)});
    if (!response.ok) throw Error('SDK HTTP ' + response.status);
    const nextState=await response.json();if(replaying){statePollPending=false;scheduleStatePoll(100);return;}
    if(poseMode==='mano')await hydrateNativeMesh(nextState);
    state=nextState; apiAlive = true; lastFetch = performance.now();
    if(!initialSolverApplied&&poseMode==='wuji2'&&state.connected&&state.wuji2?.solver_backend&&['slsqp_cem','slsqp_edge','slsqp_graph','cem'].includes(requestedSolver)&&!busy){
     initialSolverApplied=true;if(state.wuji2.solver_backend!==requestedSolver)void selectWujiSolver(requestedSolver);
    }
    if (!mappingDirty && state.contact?.mapping_version && mapping?.version !== state.contact.mapping_version) loadMapping();
  } catch { apiAlive = false; }
  updateUI(); drawPose(); drawHeat();
  // Include fetch and rendering time in the frame budget; keep one request in flight.
  statePollPending=false;scheduleStatePoll(Math.max(0,(poseMode==='wuji2'?1000/60:33.3)-(performance.now()-start)));
}
let nativeMeshCache=null;
async function hydrateNativeMesh(next){
 const m=next.hand_model,d=m?.mesh_frame;if(m?.surface_backend!=='qnbot'||!m.ready||!d)return;
 const matches=c=>c&&c.sequence===d.sequence&&c.geometry_version===m.geometry_version&&c.binding_version===m.binding_version;
 if(!matches(nativeMeshCache)){
  const response=await fetch('/api/hand_model/mesh?sequence='+d.sequence,{cache:'no-store',signal:AbortSignal.timeout(2000)});
  if(!response.ok)throw Error('官方mesh帧读取失败');const bytes=await response.arrayBuffer(),view=new DataView(bytes);
  if(bytes.byteLength<128||view.getUint32(0,true)!==0x31424e51||view.getUint32(4,true)!==1)throw Error('官方mesh格式不匹配');
  const seq=view.getUint32(8,true),nv=view.getUint32(12,true),nj=view.getUint32(16,true),timestamp=view.getFloat64(24,true),decoder=new TextDecoder();
  const text=(offset,n)=>decoder.decode(new Uint8Array(bytes,offset,n)).split('\0')[0];
  if(nv!==d.vertex_count||nj!==d.joint_count||bytes.byteLength!==128+12*(nv+nj)||text(32,32)!==m.geometry_version||text(64,64)!==m.binding_version)throw Error('官方mesh拓扑已切换');
  if(seq!==d.sequence){m.ready=false;return;}
  const points=new Float32Array(bytes,128),joints=Array.from({length:nj},(_,i)=>Array.from(points.subarray((nv+i)*3,(nv+i+1)*3)));
  nativeMeshCache={sequence:seq,geometry_version:m.geometry_version,binding_version:m.binding_version,vertices:points.subarray(0,nv*3),joints,timestamp};
 }
 m.vertices_m=nativeMeshCache.vertices;m.joints_m=nativeMeshCache.joints;m.timestamp_ms=nativeMeshCache.timestamp;
}
function scheduleStatePoll(ms){clearTimeout(statePollTimer);statePollTimer=setTimeout(poll,ms);}
document.addEventListener('visibilitychange',()=>{if(!document.hidden&&!statePollPending){clearTimeout(statePollTimer);poll();}});

function canvasContext(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const {width, height} = canvas.getBoundingClientRect();
  if (canvas.width !== Math.round(width*dpr) || canvas.height !== Math.round(height*dpr)) {canvas.width = Math.round(width*dpr); canvas.height = Math.round(height*dpr);}
  const ctx = canvas.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,width,height);
  return {ctx, width, height};
}
let camera = {yaw:-.3, pitch:.52, zoom:1};
let cameraCenter = [65, 0, -45];
function project(point, w, h) {
  const [x,y,z] = point, a = y-cameraCenter[1], b = x-cameraCenter[0], c = -(z-cameraCenter[2]);
  const rx = a*Math.cos(camera.yaw)+c*Math.sin(camera.yaw);
  const rz = -a*Math.sin(camera.yaw)+c*Math.cos(camera.yaw);
  const ry = b*Math.cos(camera.pitch)-rz*Math.sin(camera.pitch);
  const depth = b*Math.sin(camera.pitch)+rz*Math.cos(camera.pitch);
  const perspective = 650 / (650 - depth);
  const scale = Math.min(w/240,h/200)*camera.zoom*perspective;
  return [w/2+rx*scale,h*.51-ry*scale,depth,scale];
}
function quatRotate(v, q) {
  const [x,y,z,w]=q, [vx,vy,vz]=v;
  const tx=2*(y*vz-z*vy),ty=2*(z*vx-x*vz),tz=2*(x*vy-y*vx);
  return [vx+w*tx+y*tz-z*ty,vy+w*ty+z*tx-x*tz,vz+w*tz+x*ty-y*tx];
}
function drawPose() {
  if(poseMode==='mano'){window.updateManoView?.(state,true);return;}
  if(poseMode==='wuji2'){window.updateWujiView?.(state,true);return;}
  if (['skeleton','official_skeleton'].includes(poseMode)) { drawSkeleton(); return; }
  const {ctx:c,width:w,height:h}=canvasContext($('poseCanvas'));
  const P=p=>project(p,w,h);
  function line(a,b,color,width=1,dash=[]) {const A=P(a),B=P(b); c.beginPath();c.moveTo(A[0],A[1]);c.lineTo(B[0],B[1]);c.strokeStyle=color;c.lineWidth=width;c.setLineDash(dash);c.stroke();c.setLineDash([]);}
  for(let x=-30;x<=190;x+=20)line([x,-110,0],[x,110,0],x===-30?'#435141':'#364137',.6);
  for(let y=-110;y<=110;y+=20)line([-30,y,0],[190,y,0],'#364137',.6);
  const mapped=poseMode==='fingertips'&&state?.skeleton?.ready?state.skeleton.targets:null;
  if(mapped&&!sceneInitialized){cameraCenter=[0,1,2].map(k=>(Math.min(...mapped.positions_local.map(p=>p[k]))+Math.max(...mapped.positions_local.map(p=>p[k])))*500);sceneInitialized=true;}
  const origin=P([0,0,0]);
  c.beginPath();c.arc(origin[0],origin[1],3,0,Math.PI*2);c.fillStyle='#98a68a';c.fill();
  c.font='8px "DejaVu Sans Mono",monospace';c.fillStyle='#82917b';c.fillText(mapped?'WRIST ORIGIN':'EMF ORIGIN',origin[0]+8,origin[1]+3);
  [[[25,0,0],'#d68361','X'],[[0,25,0],'#91ad75','Y'],[[0,0,25],'#7fa7bb','Z']].forEach(([v,col,t])=>{line([0,0,0],v,col,1.5);const a=P(v);c.fillStyle=col;c.fillText(t,a[0]+4,a[1]);});
  if (!apiAlive || !state?.connected || !state.pose) return;
  const tips=mapped?Object.fromEntries(fingers.map((f,i)=>[f,{position:mapped.positions_local[i],quaternion_xyzw:mapped.quaternions_local_xyzw[i]}])):state.pose.fingertip_local;
  const nodes=fingers.map((f,i)=>({f,i,tip:tips[f],point:tips[f]?.position.map(x=>x*1000)})).filter(n=>n.tip&&state.pose.raw_valid_mask[n.f]).sort((a,b)=>P(a.point)[2]-P(b.point)[2]);
  for(const {f,i,tip,point} of nodes) {
    line([0,0,0],point,'#87968055',.7,[3,5]);
    const trail=mapped?[]:trails[f];
    for(let k=1;k<trail.length;k++)line(trail[k-1],trail[k],colors[i]+Math.round(k/trail.length*100).toString(16).padStart(2,'0'),1.5);
    const A=P(point);
    const glow=c.createRadialGradient(A[0],A[1],1,A[0],A[1],20);glow.addColorStop(0,colors[i]+'32');glow.addColorStop(1,colors[i]+'00');c.fillStyle=glow;c.beginPath();c.arc(A[0],A[1],20,0,Math.PI*2);c.fill();
    [[[13,0,0],'#d28a70'],[[0,13,0],'#a3bb87'],[[0,0,13],'#80a9c0']].forEach(([v,col])=>{const q=quatRotate(v,tip.quaternion_xyzw);line(point,point.map((x,k)=>x+q[k]),col,1.5);});
    c.fillStyle=colors[i];c.beginPath();c.arc(A[0],A[1],4,0,Math.PI*2);c.fill();c.strokeStyle='#eef2e3';c.lineWidth=.8;c.stroke();
    c.font='10px "Noto Sans CJK SC",sans-serif';c.fillStyle=colors[i];c.fillText(names[i],A[0]+10,A[1]-10);
    c.font='7px "DejaVu Sans Mono",monospace';c.fillStyle='#8b9a82';c.fillText(point.map(x=>x.toFixed(0)).join(', '),A[0]+10,A[1]+3);
  }
}
let drag=null;
$('poseCanvas').addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];$('poseCanvas').setPointerCapture(e.pointerId);});
$('poseCanvas').addEventListener('pointermove',e=>{if(!drag)return;camera.yaw+=(e.clientX-drag[0])*.008;camera.pitch=Math.max(-1.5,Math.min(1.5,camera.pitch+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY];drawPose();});
$('poseCanvas').addEventListener('pointerup',()=>drag=null);
$('poseCanvas').addEventListener('pointercancel',()=>drag=null);
$('poseCanvas').addEventListener('wheel',e=>{e.preventDefault();camera.zoom=Math.max(.4,Math.min(3,camera.zoom*Math.exp(-e.deltaY*.001)));drawPose();},{passive:false});
function chooseCamera(mode) {if(poseMode==='mano')window.setManoView?.(mode);if(poseMode==='wuji2')window.setWujiView?.(mode);camera={yaw:mode==='front'?0:-.3,pitch:mode==='top'?0:mode==='front'?1.5:.52,zoom:1};document.querySelectorAll('[data-camera]').forEach(b=>b.classList.toggle('selected',b.dataset.camera===mode));drawPose();}
document.querySelectorAll('[data-camera]').forEach(b=>b.onclick=()=>chooseCamera(b.dataset.camera));
$('resetCamera').onclick=()=>chooseCamera('orbit');

const palette=[[0,[232,237,222]],[35,[186,201,156]],[85,[124,170,105]],[145,[220,196,89]],[200,[229,140,66]],[255,[215,71,36]]];
function heatColor(v) {for(let i=1;i<palette.length;i++){if(v<=palette[i][0]){const [a,A]=palette[i-1],[b,B]=palette[i],t=(v-a)/(b-a);return `rgb(${A.map((x,k)=>Math.round(x+(B[k]-x)*t)).join(',')})`;}}return 'rgb(215,71,36)';}
let heatCells=[];
let handPatches=[];
function drawHeat() {
  const {ctx:c,width:w,height:h}=canvasContext($('heatCanvas')); heatCells=[]; handPatches=[];
  const values=apiAlive&&state?.connected?state.tactile?.values:null;
  const contact=contactForView(), ready=apiAlive&&state?.connected&&contact?.ready;
  const assignments=mapping?.assignments;
  const patches=contact?.patches||[];
  const scale=Math.min(w/370,(h-65)/310),ox=(w-340*scale)/2,oy=35;
  const patchColor=index=>senseMode==='contact' ? (ready&&patches[index]?.contact ? '#39cf68' : '#e4e6df') : '#e9edde';
  function cell(x,y,size,sid){const X=ox+x*scale,Y=oy+y*scale,S=size*scale;const index=assignments?.[sid];
    c.fillStyle=senseMode==='contact' ? (ready&&patches[index]?.contact?'#39cf68':'#e4e6df') : heatColor(values?.[sid]||0);
    if(selectedSids.has(sid)&&$('mappingEditor').open)c.fillStyle='#dc592d';
    c.beginPath();c.roundRect(X,Y,S-1,S-1,1.1);c.fill();heatCells.push({x:X,y:Y,size:S,sid});}
  c.font='8px "Noto Sans CJK SC",sans-serif';c.textAlign='center';c.fillStyle='#88977b';
  if(heatMode==='matrix'){
    const size=11,offsetX=60;
    for(let row=0;row<23;row++)for(let col=0;col<20;col++)cell(offsetX+col*size,14+row*size,size,(22-row)*20+col);
    c.fillStyle='#89977b';c.fillText('指尖侧 · ROW 22',ox+(offsetX+110)*scale,oy+5*scale);
    c.fillText('腕侧 · ROW 0',ox+(offsetX+110)*scale,oy+285*scale);
  }else if(mapping){
    function patch(id,x,y,bw,bh,short){const i=mapping.patches.findIndex(p=>p.id===id);if(i<0)return;
      const X=ox+x*scale,Y=oy+y*scale,W=bw*scale,H=bh*scale;
      c.fillStyle=patchColor(i);c.beginPath();c.roundRect(X,Y,W,H,3*scale);c.fill();
      // Contact is fill-only and binary; borders never encode another state.
      if(senseMode==='tactile'){const sids=assignments.map((v,sid)=>v===i?sid:-1).filter(sid=>sid>=0);const cols=id.startsWith('palm')?10:4;const rows=Math.ceil(sids.length/cols);const cw=bw/cols,ch=bh/rows;
        sids.forEach((sid,k)=>{const cx=X+(k%cols)*cw*scale,cy=Y+Math.floor(k/cols)*ch*scale;c.fillStyle=heatColor(values?.[sid]||0);c.fillRect(cx+.6,cy+.6,Math.max(.2,cw*scale-1),Math.max(.2,ch*scale-1));});}
      c.fillStyle=senseMode==='contact'&&ready&&patches[i]?.contact?'#174e2a':'#a0aa96';c.font='7px "DejaVu Sans Mono",monospace';c.fillText(senseMode==='contact'?`${short} ${ready?(patches[i]?.contact?'1':'0'):'?'}`:short,X+W/2,Y+H/2+2);
      handPatches.push({x:X,y:Y,width:W,height:H,index:i});}
    const xs=[17,78,139,200,261],ys=[92,43,20,43,77];
    fingers.forEach((f,i)=>{const x=xs[i],y=ys[i];c.fillStyle='#89977b';c.font='8px "Noto Sans CJK SC",sans-serif';c.fillText(names[i],ox+(x+20)*scale,oy+(y-10)*scale);
      patch(f+'_distal',x,y,40,29,'D');
      if(i===0)patch(f+'_proximal',x,y+35,40,50,'P');
      else{patch(f+'_intermediate',x,y+35,40,23,'M');patch(f+'_proximal',x,y+64,40,26,'P');}});
    for(let r=0;r<3;r++)for(let col=0;col<2;col++)patch(`palm_${r}_${col}`,80+col*101,245-r*39,94,33,`PALM ${r*2+col+1}`);
    c.font='7px "Noto Sans CJK SC",sans-serif';c.fillStyle='#9da98f';c.fillText('D 末端 / M 中段 / P 近端 · 掌部分区为示意',ox+170*scale,oy+297*scale);
  }
  c.textAlign='left';
}
document.querySelectorAll('[data-heat]').forEach(b=>b.onclick=()=>{heatMode=b.dataset.heat;document.querySelectorAll('[data-heat]').forEach(n=>n.classList.toggle('selected',n===b));$('heatTooltip').hidden=true;drawHeat();});
function hitCell(e){const rect=$('heatCanvas').getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;return heatCells.find(p=>x>=p.x&&x<p.x+p.size&&y>=p.y&&y<p.y+p.size);}
let selecting=false, selectAdd=true;
function selectCell(cell){if(!cell)return;if(selectAdd)selectedSids.add(cell.sid);else selectedSids.delete(cell.sid);$('selectionInfo').textContent=`已选 ${selectedSids.size} 个 SID`;drawHeat();}
$('heatCanvas').addEventListener('pointerdown',e=>{if(!$('mappingEditor').open||heatMode!=='matrix')return;const cell=hitCell(e);if(!cell)return;selecting=true;selectAdd=!selectedSids.has(cell.sid);e.target.setPointerCapture(e.pointerId);selectCell(cell);});
$('heatCanvas').addEventListener('pointerup',()=>selecting=false);
$('heatCanvas').addEventListener('pointercancel',()=>selecting=false);
$('heatCanvas').addEventListener('pointermove',e=>{const rect=e.target.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;const cell=hitCell(e);if(selecting){selectCell(cell);return;}
  const block=handPatches.find(p=>x>=p.x&&x<p.x+p.width&&y>=p.y&&y<p.y+p.height);
  const index=cell?mapping?.assignments[cell.sid]:block?.index;const tip=$('heatTooltip');tip.hidden=index===undefined||!mapping;
  if(!tip.hidden){const patch=contactForView()?.patches?.[index];const description=senseMode==='contact' ? (!apiAlive||!state?.connected||!contactForView()?.ready?'接触状态未知':patch?.contact?'1 · 接触':'0 · 未确认接触') : cell?`${state?.tactile?.values[cell.sid]??'—'} ADC`:'触觉阵列';tip.textContent=`${mapping.patches[index].label} · ${cell?'SID '+cell.sid+' · ':''}${description}`;tip.style.left=Math.max(0,Math.min(x+10,rect.width-225))+'px';tip.style.top=Math.max(0,y-30)+'px';}});
$('heatCanvas').addEventListener('pointerleave',()=>$('heatTooltip').hidden=true);

async function loadMapping(force=false){if(mappingLoading||mappingDirty&&!force)return;mappingLoading=true;try{const r=await fetch('/api/mapping',{signal:AbortSignal.timeout(6000)});const m=await r.json();if(m.ready&&!mappingDirty){const selectedId=mapping?.patches[Number($('patchSelect').value)]?.id;mapping=m;$('patchSelect').replaceChildren(...m.patches.map((p,i)=>{const o=document.createElement('option');o.value=i;o.textContent=p.label;return o;}));const selectedIndex=m.patches.findIndex(p=>p.id===selectedId);if(selectedIndex>=0)$('patchSelect').value=selectedIndex;updateMappingUI();drawHeat();}}catch(e){if(force)toast(e.message);}finally{mappingLoading=false;}}
function updateMappingUI(){$('mappingStatus').textContent=`${mapping?.confirmed?.length||0} / 20 区已核对`;$('mappingDirty').textContent=mappingDirty?'尚未保存':'';const patch=mapping?.patches[Number($('patchSelect').value)];$('confirmPatch').textContent=patch&&mapping.confirmed.includes(patch.id)?'当前区已核对 ✓':'已核对当前区';}
$('mappingEditor').addEventListener('toggle',()=>{if($('mappingEditor').open){heatMode='matrix';document.querySelectorAll('[data-heat]').forEach(b=>b.classList.toggle('selected',b.dataset.heat==='matrix'));loadMapping();}$('heatCanvas').classList.toggle('editing',$('mappingEditor').open);drawHeat();});
$('clearSelection').onclick=()=>{selectedSids.clear();$('selectionInfo').textContent='已选 0 个 SID';drawHeat();};
$('patchSelect').onchange=()=>{selectedSids.clear();if(mapping){const i=Number($('patchSelect').value);mapping.assignments.forEach((p,sid)=>{if(p===i)selectedSids.add(sid);});}$('selectionInfo').textContent=`已选 ${selectedSids.size} 个 SID`;updateMappingUI();drawHeat();};
$('assignPatch').onclick=()=>{if(!mapping||!selectedSids.size)return toast('请先在矩阵选择 SID');const i=Number($('patchSelect').value);const affected=new Set([...selectedSids].map(sid=>mapping.assignments[sid]));affected.add(i);selectedSids.forEach(sid=>mapping.assignments[sid]=i);mapping.confirmed=mapping.confirmed.filter(id=>!affected.has(mapping.patches.findIndex(p=>p.id===id)));mappingDirty=true;updateMappingUI();drawHeat();};
$('confirmPatch').onclick=()=>{if(!mapping)return;const id=mapping.patches[Number($('patchSelect').value)].id;if(!mapping.confirmed.includes(id))mapping.confirmed.push(id);mappingDirty=true;updateMappingUI();};
$('saveMapping').onclick=async()=>{if(!mapping)return;const r=await command('mapping_save',{assignments:mapping.assignments,confirmed:mapping.confirmed});if(r?.ok){mappingDirty=false;await loadMapping(true);toast('分区映射已保存');}};
$('restoreMapping').onclick=async()=>{const r=await command('mapping_restore');if(r?.ok){mappingDirty=false;selectedSids.clear();await loadMapping(true);toast('已恢复默认示意分区');}};
$('senseMode').onchange=e=>{senseMode=e.target.value;history=[];updateUI();drawHeat();};
$('poseMode').onchange=e=>{poseMode=e.target.value;sceneInitialized=false;cameraCenter=[65,0,-45];updateUI();drawPose();};
$('tare').onclick=async()=>{const r=await command('tare');if(r?.ok)toast('去皮中：请保持 3 秒无接触，采集基线与噪声。');};
$('clearTare').onclick=async()=>{const r=await command('clear_tare');if(r?.ok)toast('已清除接触去皮；旧触觉均值基线保留。');};
function ikCommand(action){const stage=state?.skeleton?.capture?.stage;return command('ik_'+action,{request_id:stage?.request_id,stage_id:stage?.stage_id});}
$('ikConfirm').onclick=()=>ikCommand('confirm');$('ikRetry').onclick=()=>ikCommand('retry');$('ikCancel').onclick=()=>ikCommand('cancel');$('ikRestart').onclick=()=>command('ik_restart');
$('ikRecalibrate').onclick=()=>{$('poseMode').value='mano';$('poseMode').dispatchEvent(new Event('change'));return $('manoStart').onclick();};$('ikLegacyInitialize').onclick=()=>command('ik_recalibrate');$('ikRestoreCalibration').onclick=()=>command('ik_restore_calibration');
$('manoGoSkeleton').onclick=()=>{$('poseMode').value='skeleton';$('poseMode').dispatchEvent(new Event('change'));};
$('wujiRestart').onclick=()=>command('wuji2_restart');
$('wujiRebuild').onclick=()=>command('wuji2_mapping_rebuild');
async function selectWujiSolver(solver){
 if(busy||replaying)return;
 window.wujiRequestedSolver=solver;window.updateWujiView?.(state,poseMode==='wuji2');
 const result=await command('wuji2_solver_set',{solver});
 if(!result)window.wujiRequestedSolver=null;
 window.updateWujiView?.(state,poseMode==='wuji2');return result;
}
$('wujiSolver').onchange=()=>selectWujiSolver($('wujiSolver').value);
const gestureSteps = [
  {id:'close',title:'五指自然并拢',view:'厂商原图 · 手背视图',hint:'手指伸直，自然靠拢。掌心展开，注意这一步不是握拳。'},
  {id:'open',title:'五指自然张开',view:'厂商原图 · 手背视图',hint:'五指舒适地分开，掌心展开；不必用力撑到极限。'},
  {id:'mcp90',title:'根关节弯曲 90°',view:'厂商原图 · 对照左侧手势',hint:'看左图：四指从指根向掌心方向折约 90°；中段和末端保持一条直线。不要卷起指尖。'},
  {id:'mcp90_pip90',title:'自然握拳',view:'厂商原图 · 对照右侧手势',hint:'看右图：四指向掌心卷曲，中间指节也弯曲。与上一步的平直手指不同，自然握拳即可。'},
  {id:'pinch_first',title:'拇指 ↔ 食指',view:'掌面视图 · 橙色为接触指腹',hint:'拇指与食指的指腹轻轻接触；其余手指自然放松。'},
  {id:'pinch_middle',title:'拇指 ↔ 中指',view:'掌面视图 · 橙色为接触指腹',hint:'拇指与中指的指腹轻轻接触；其余手指自然放松。'},
  {id:'pinch_ring',title:'拇指 ↔ 无名指',view:'掌面视图 · 橙色为接触指腹',hint:'拇指与无名指的指腹轻轻接触；其余手指自然放松。'},
  {id:'pinch_little',title:'拇指 ↔ 小指',view:'掌面视图 · 橙色为接触指腹',hint:'拇指与小指的指腹轻轻接触；其余手指自然放松。'},
];
const officialGestureImages = {"together.png":"/gestures/together.png","spread.png":"/gestures/spread.png","knuckle-and-fist.png":"/gestures/knuckle-and-fist.png"};
function gestureSvg(id){
  const officialName={close:"together.png",open:"spread.png",mcp90:"knuckle-and-fist.png",mcp90_pip90:"knuckle-and-fist.png"}[id];
  if(officialName)return `<img class="official-gesture" src="${officialGestureImages[officialName]}" alt="厂商原图：${gestureSteps.find(g=>g.id===id).title}${id.startsWith("mcp90")?"，左图根关节弯曲、右图握拳对照":""}">`;
  const ink='#65715c', accent='#dc592d', paper='#f8f8f1';
  let body='';
  const path=(d,color=ink,width=12)=>`<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round"/>`;
  const line=(points,color=ink,width=12)=>path('M'+points.map(p=>p.join(',')).join(' L'),color,width);
  const dot=(p,color=ink,r=2.8)=>`<circle cx="${p[0]}" cy="${p[1]}" r="${r}" fill="${color}"/>`;
    // Palm-facing LEFT hand: thumb is on the viewer's right. Fingers are
    // drawn little/ring/middle/index from left to right, with explicit joints.
    body+=`<path d="M85 215 L82 176 Q70 157 71 128 L156 124 Q166 144 174 157 Q161 180 147 189 L147 215" fill="#e7eade" stroke="${ink}" stroke-width="2"/>`;
    const target={pinch_first:3,pinch_middle:2,pinch_ring:1,pinch_little:0}[id];
    const bases=[[79,132],[104,126],[129,124],[153,129]];
    const tips=[[78,72],[102,45],[126,36],[151,55]];
    let contact=null;
    for(let f=0;f<4;f++){
      const a=bases[f], tip=tips[f], selected=f===target;
      const pts=selected?[[...a],[a[0]-2,96],[a[0]+8,84],[a[0]+16,101]]:[a,[a[0]+(tip[0]-a[0])*.4,a[1]+(tip[1]-a[1])*.4],[a[0]+(tip[0]-a[0])*.75,a[1]+(tip[1]-a[1])*.75],tip];
      body+=line(pts,selected?accent:target!==undefined?'#a9b29e':ink,12);
      pts.slice(0,3).forEach(p=>body+=dot(p,paper,2.2));
      if(selected)contact=pts.at(-1);
    }
    const thumb=[[163,163],[176,139],[contact[0]+20,113],[contact[0]+3,109]];
    body+=line(thumb,contact?accent:ink,14);
    thumb.slice(0,3).forEach(p=>body+=dot(p,paper,2.2));
    body+=path('M91 170 Q112 151 145 160','#bdc5b1',1.5);
    if(contact){
      body+=`<circle cx="${contact[0]+2}" cy="${contact[1]+4}" r="13" fill="none" stroke="${accent}" stroke-width="1.5" stroke-dasharray="3 3"/>`;
    }
    if(state?.side==='right')body=`<g transform="translate(270,0) scale(-1,1)">${body}</g>`;
    body+=`<text x="94" y="235" fill="${ink}" font-size="10">${state?.side==='right'?'右手':'左手'} · 掌面</text>`;
  const meta=gestureSteps.find(s=>s.id===id);
  return `<svg viewBox="25 20 220 225" role="img" aria-label="${state?.side==='right'?'右手':'左手'}${meta.title}，${meta.view}" xmlns="http://www.w3.org/2000/svg">${body}</svg>`;
}
let shownGesture=null;
$('gestureSteps').innerHTML=gestureSteps.map((g,i)=>`<figure data-gesture="${g.id}">${gestureSvg(g.id)}<figcaption><b>${String(i+1).padStart(2,'0')}</b>${g.title}</figcaption></figure>`).join('');
function updateGestureGuide(stage,live,boneReady){
  const id=stage?.stage_id;
  const gesture=gestureSteps.find(g=>g.id===id);
  $('gestureGuide').hidden=!gesture||boneReady;
  const gestureKey=id+':'+(state?.side||'left');
  if(gesture&&shownGesture!==gestureKey){
    $('gestureImage').innerHTML=gestureSvg(id);
    $('gestureTitle').textContent=gesture.title;
    $('gestureView').textContent=gesture.view;
    $('gestureHint').textContent=gesture.hint;
    shownGesture=gestureKey;
  }
  $('gestureGuide').classList.toggle('inactive',!live);
  const official=['close','open','mcp90','mcp90_pip90'].includes(id);
  $('gestureGuide').classList.toggle('official',official);
  $('gestureSource').textContent=official?'厂商包原图 · 未修改':'对指补充示意 · 非官方原图';
  $('gestureSteps').querySelectorAll('figure').forEach(el=>{
    const current=el.dataset.gesture===id&&!boneReady;
    el.classList.toggle('current',current);
    if(current)el.setAttribute('aria-current','step');else el.removeAttribute('aria-current');
  });
}
function updateModes(live){
  const contact=contactForView(),skeleton=state?.skeleton,ready=live&&contact?.ready,isContact=senseMode==='contact';
  $('contactMetricLabel').innerHTML=isContact?'接触分区 <small>CONTACT PATCHES</small>':'触觉峰值 <small>PEAK PRESSURE</small>';
  $('responseMetricLabel').innerHTML=isContact?'接触状态 <small>BINARY CONTACT</small>':'接触单元 <small>ACTIVE CELLS</small>';
  $('peak').textContent=isContact?(ready?contact.count:'—'):(live?state.tactile.peak:'—');$('peakUnit').textContent=isContact?'/ 20':'ADC';
  $('active').textContent=isContact?(ready?(contact.count>0?'1':'0'):'—'):(live?state.tactile.active:'—');$('activeUnit').textContent=isContact?'0 / 1':'/ 460';
  $('responseNote').textContent=isContact?(ready?'1 有确认接触 / 0 无确认接触':'未就绪 · 状态未知'):'阈值 > 5 ADC';
  const collecting=state?.contact?.tare?.state==='collecting';
  $('tare').disabled=!live||collecting;$('clearTare').disabled=!live||collecting;
  $('tareStatus').textContent=collecting?`采集中 ${Math.round((contact.tare.progress||0)*100)}%`:contact?.tare?.error||(!live?'设备离线':state?.contact?.ready?'去皮有效':'请先去皮');
  $('baselineTag').textContent=isContact?(collecting?'采集中':ready?'接触识别':'待就绪'):(state?.tactile?.has_baseline?'去基线':'原始值');
  $('heatEmpty').hidden=!!(live&&state?.tactile);
  $('heatEmptyText').textContent=!live?'等待实时数据':'点击去皮，记录无接触信号';
  $('heatMaxLabel').textContent=isContact?'1 · 接触':'255 ADC';document.querySelector('.color-scale').classList.toggle('binary',isContact);document.querySelector('.heat-legend>span:first-child').textContent=isContact?'0 · 无接触':'0';$('arrayLabel').textContent=heatMode==='hand'?'20 ANATOMICAL PATCHES':'23 × 20 SENSING ARRAY';
  $('baselineStatus').textContent=isContact?(ready?'未接触区域灰显':'接触状态未知 · 请完成去皮'):(state?.tactile?.has_baseline?'已应用旧触觉基线':'原始触觉值');
  $('heatUnitNote').textContent=isContact?'二值分布：灰 0 / 绿 1；未知显示 ?':'ADC 原始量，不等同于牛顿';
  const detector=contact?.detector;
  $('detectorInfo').textContent=detector?`轻触检测 · EMA ${detector.ema_tau_ms} ms · 进入 max(${detector.enter_adc}, ${detector.enter_sigma}σ) ADC · 相邻双单元确认`:'连接后显示接触检测参数。';
  const active=ready?contact.patches.filter(p=>p.contact):[];
  $('contactList').replaceChildren(...(isContact&&active.length?active.map(p=>{const chip=document.createElement('span');chip.className='contact-chip';chip.textContent=p.label;return chip;}):[document.createTextNode(!isContact?'切换接触分布可查看接触 patch':!live?'设备离线，接触状态未知':!ready?collecting?'正在采集无接触信号…':'尚未去皮，接触状态未知':'当前无接触')]));
  if(!mappingDirty)updateMappingUI();
  const bone=poseMode==='skeleton';const boneReady=live&&skeleton?.ready;
  $('poseTitle').textContent=bone?'骨骼空间姿态':poseMode==='raw_fingertips'?'EMF 传感器姿态':'指尖目标姿态';$('coordinateLabel').textContent=bone?'WRIST LOCAL · 25 JOINTS':'EMF LOCAL · 5 SENSORS';$('poseSource').textContent=bone?'OFFICIAL SKELETON IK':'POSITION / ORIENTATION';$('frameLabel').textContent=bone?'腕部局部坐标系':'EMF 坐标系';
  $('poseLegend').textContent=bone?'官方 25 节点 / 24 骨段':'EMF 原点 · 虚线为原点连线';$('skeletonAge').textContent=bone?(boneReady?`骨架年龄 ${Math.round(skeleton.age_ms)} ms`:'QnBot SDK 1.7.0'):'传感器方向轴';
  $('poseEmpty').hidden=bone?!!boneReady:!!(live&&state?.pose);
  const mapped=poseMode==='fingertips'&&boneReady&&skeleton?.targets;
  if(mapped){$('coordinateLabel').textContent='WRIST LOCAL · 5 TARGETS';$('frameLabel').textContent='腕部局部坐标系';$('poseLegend').textContent='官方映射指尖目标 · 与骨架同一坐标系';$('poseSource').textContent='OFFICIAL MAPPED TARGETS';}
  if(bone&&boneReady&&skeleton?.targets&&$('showTipTargets').checked){const e=skeleton.alignment?.tip_errors_mm?.thumb;$('poseLegend').textContent='实心：骨架末端 · 空心：指尖目标';$('skeletonAge').textContent=`拇指目标差 ${Number.isFinite(e)?e.toFixed(1):'—'} mm · 动力学关闭`;}
  const stage=skeleton?.capture?.stage;
  updateGestureGuide(stage,live,boneReady);
  const statuses={input_invalid:'标定位姿缺失或无效',starting:'官方 IK 初始化中',needs_calibration:'需要骨架采集 / 标定',calibrating:'官方 IK 标定计算中',waiting_pose:'等待骨架输出',error:'骨架运行时异常',cancelled:'骨架采集已取消',stale:'骨架数据过期',unavailable:'骨架运行时未安装',not_started:'骨架未启动',ready:'骨架在线'};
  if(bone){$('poseStatus').textContent=boneReady?(Math.max(...Object.values(skeleton.alignment?.tip_errors_mm||{}))>15?'骨架在线 · 指尖目标残差较大':'25 节点骨架在线'):statuses[skeleton?.status]||'等待骨架';$('poseEmptyText').textContent=!live?'等待手套连接':skeleton?.error||statuses[skeleton?.status]||'请按下方提示完成八动作标定';}
  $('ikPrompt').textContent=skeleton?.error||(!live?'连接手套后可进行骨架标定。':boneReady?'标定已就绪，正在显示官方 IK 骨架。':skeleton?.capture?.session_state==='cancelled'?'采集已取消，点击重启骨架以继续。':stage?.prompt||statuses[skeleton?.status]||'等待官方 IK 运行时');
  if(boneReady&&skeleton.alignment?.thumb_unreliable)$('ikPrompt').textContent=`拇指骨架目标残差 ${skeleton.alignment.tip_errors_mm.thumb.toFixed(1)} mm，多个关节到达限位。当前目标未被骨架匹配；请对照实际动作、原始观测与骨段，重启会复用已有记录。`;
  const officialIK=skeleton?.runtime_policy?.mode==='official';
  const thumbZeroIK=skeleton?.runtime_policy?.mode==='thumb_straight_zero';
  const slsqpIK=skeleton?.runtime_policy?.mode==='emf_slsqp';
  const localIK=!!skeleton?.runtime_policy&&!officialIK&&!thumbZeroIK&&!slsqpIK;
  if(bone&&slsqpIK)$('poseSource').textContent='EMF · SLSQP IK';
  if(boneReady&&slsqpIK){$('poseLegend').textContent='25 节点 / 24 骨段 · SLSQP IK';$('ikPrompt').textContent='SLSQP：EMF 位置／旋转、姿态及相邻帧约束 · 固定安装标定 · 动力学关闭。';}
  if(bone&&officialIK)$('poseSource').textContent='QNBOT OFFICIAL IK';
  if(boneReady&&officialIK)$('ikPrompt').textContent='官方标定配置与原始骨骼资源 · 动力学关闭。';
  if(bone&&thumbZeroIK)$('poseSource').textContent='QNBOT GN · THUMB STRAIGHT ZERO';
  if(boneReady&&thumbZeroIK){
    const e=skeleton.alignment?.tip_errors_mm?.thumb;
    $('ikPrompt').textContent='官方 GN 求解器 · 拇指零位与关节轴顺序修正 · 动力学关闭。'+(Number.isFinite(e)&&e>15?` 拇指目标残差 ${e.toFixed(1)} mm，仍需核对定位与标定。`:'');
  }
  if(bone&&localIK)$('poseSource').textContent='QNBOT GN · LOCAL CONFIG';
  if(boneReady&&localIK)$('ikPrompt').textContent='当前为 QnBot GN 加本地映射、关节与权重修正，并非官方原配置。';
  if(boneReady&&skeleton?.runtime_policy?.fixed_input_registration?.quality?.physical_validation==='failed_flat_repeat'){
    $('ikPrompt').textContent='本地修正版尚未通过平摊实物复测；当前为调试输出，尺寸已标定不代表姿态正确。';
    if(bone)$('poseStatus').textContent='调试输出 · 实物复测未通过';
  }
  const steps=gestureSteps.map(g=>g.id);
  $('ikStep').textContent=stage?`${steps.indexOf(stage.stage_id)+1} / 8 · ${stage.collected_sample_count} / ${stage.required_sample_count}`:(skeleton?.hand_side||state?.side||'left').toUpperCase()+' HAND';
  $('ikProgress').max=stage?.required_sample_count||30;$('ikProgress').value=stage?.collected_sample_count||0;
  const running=skeleton?.capture?.session_state==='running';
  $('ikConfirm').disabled=!live||!running||stage?.state!=='awaiting_confirmation';$('ikCancel').disabled=!live||!running||!stage;$('ikRetry').disabled=!live||!running||!stage;$('ikRestart').disabled=!live;
  $('ikRecalibrate').disabled=!live||running;$('ikLegacyInitialize').disabled=!live||running;$('ikRestoreCalibration').disabled=!live;
}
function drawSkeleton(){
  const skeleton=poseMode==='official_skeleton'?state?.skeleton_official:state?.skeleton;
  const {ctx:c,width:w,height:h}=canvasContext($('poseCanvas'));
  const shared=poseMode==='skeleton'&&state?.hand_model?.calibrated&&state?.hand_model?.ready?state.hand_model.shared_skeleton_pose:null;
  const pose=shared||skeleton?.pose;const chains=shared?fingers.map(f=>['wrist',...[0,1,2,3].map(k=>`${f}_${k}`)]):skeleton?.chains;
  if(!apiAlive||!state?.connected||!skeleton?.ready||!pose)return;
  const nodes=Object.fromEntries(pose.names.map((name,i)=>[name,pose.positions_local[i].map(x=>x*1000)]));
  if(!sceneInitialized){cameraCenter=[0,1,2].map(k=>(Math.min(...Object.values(nodes).map(p=>p[k]))+Math.max(...Object.values(nodes).map(p=>p[k])))/2);sceneInitialized=true;}
  const P=p=>project(p,w,h);
  function line(a,b,color,width=1){const A=P(a),B=P(b);c.beginPath();c.moveTo(A[0],A[1]);c.lineTo(B[0],B[1]);c.strokeStyle=color;c.lineWidth=width;c.lineCap='round';c.stroke();}
  const center=cameraCenter;
  for(let n=-120;n<=120;n+=20){line([center[0]+n,center[1]-120,0],[center[0]+n,center[1]+120,0],'#364137',.6);line([center[0]-120,center[1]+n,0],[center[0]+120,center[1]+n,0],'#364137',.6);}
  (chains||[]).forEach((chain,i)=>{for(let k=1;k<chain.length;k++)line(nodes[chain[k-1]],nodes[chain[k]],colors[i]+'bb',3);chain.slice(1).forEach((name,k)=>{const A=P(nodes[name]);c.beginPath();c.arc(A[0],A[1],k===chain.length-2?4:2.6,0,Math.PI*2);c.fillStyle=colors[i];c.fill();});const tip=P(nodes[chain.at(-1)]);c.font='10px "Noto Sans CJK SC",sans-serif';c.fillStyle=colors[i];c.fillText(names[i],tip[0]+9,tip[1]-7);});
  const targets=skeleton.targets;
  if($('showTipTargets').checked&&targets&&Math.abs(targets.timestamp_ms-pose.timestamp_ms)<=25)targets.positions_local.forEach((p,i)=>{const point=p.map(x=>x*1000),tip=nodes[chains[i].at(-1)],A=P(point);c.setLineDash([3,4]);line(tip,point,colors[i],1);c.setLineDash([]);c.beginPath();c.arc(A[0],A[1],7,0,Math.PI*2);c.strokeStyle=colors[i];c.lineWidth=2;c.stroke();});
  const wrist=P(nodes.wrist);c.beginPath();c.arc(wrist[0],wrist[1],4,0,Math.PI*2);c.fillStyle='#ccd6bc';c.fill();c.font='8px monospace';c.fillText('WRIST',wrist[0]+9,wrist[1]+4);
}
window.addEventListener('resize',()=>{drawPose();drawHeat();});
setInterval(()=>{$('clock').textContent=new Date().toLocaleTimeString('zh-CN',{hour12:false});if(apiAlive&&performance.now()-lastFetch>1500){apiAlive=false;updateUI();drawPose();drawHeat();}},500);
setTimeout(poll,0);

function contactForView(){
 // 02 is strictly the tactile detector. MANO readiness, geometry and fitting
 // never gate or relabel sensor contacts.
 return state?.contact||null;
}

function manoControl(action,data={}){return command('mano_'+action,data);}
$('manoStart').onclick=async()=>{if(!['wuji_native','moxian_wuji'].includes(state?.hand_model?.calibration_method)&&state?.hand_model?.capture?.state==='fit_ready'&&!state?.hand_model?.fit?.validation_pending)return $('manoFit').onclick();if(state?.hand_model?.checkpoint_preview){const result=await manoControl('checkpoint_end');if(!result?.ok)return;}if(state?.skeleton?.ready)return manoControl('capture_start');returnToManoAfterCalibration={session:state?.skeleton?.capture?.session_id};$('poseMode').value='skeleton';$('poseMode').dispatchEvent(new Event('change'));toast('先完成八动作初始化，再开始静态 EMF 与骨架标定。');};
$('manoMotionConfirm').onclick=()=>manoControl(state?.hand_model?.capture?.state==='fit_ready'?'capture_resume':'capture_confirm',{request_id:state?.hand_model?.capture?.request_id});
$('manoMotionRetry').onclick=()=>manoControl('capture_retry',{stage_id:state?.hand_model?.capture?.stage_id});$('manoMotionCancel').onclick=()=>manoControl('capture_cancel',{stage_id:state?.hand_model?.capture?.stage_id});
function selectedTrainingOptions(){return {adam_steps:Number($('manoTrainSteps').value),max_seconds:Number($('manoTrainSeconds').value),batch_size:Number($('manoTrainBatch').value),optimizer:$('manoTrainOptimizer').value};}
$('manoFit').onclick=()=>{
 const method=$('manoFitMethod').value;
 if(['bone_extrinsics','wuji_native','moxian_wuji'].includes(method))return manoControl('fit',{method});
 if(method==='least_squares')return manoControl('fit',{method,solver_options:{max_seconds:Number($('manoLsqSeconds').value),max_iterations:Number($('manoLsqIterations').value),stages:[['extrinsics','Extrinsics'],['dimensions','Dimensions'],['kinematics','Kinematics'],['tactile','Tactile'],['surface','Surface']].filter(([,id])=>$('manoLsq'+id).checked).map(([name])=>name)}});
 return manoControl('fit',{method,training_options:selectedTrainingOptions()});
};$('manoFitCancel').onclick=()=>manoControl('fit_cancel');
$('manoResumeFit').onclick=()=>manoControl('fit',{method:'delta',resume:true,resume_run_id:state?.hand_model?.fit?.run_id,training_options:selectedTrainingOptions()});
$('manoStartFit').onclick=()=>{const next=window.MoxianCalibrationWorkflow.view(state?.hand_model||{},apiAlive,busy);if(next.disabled)return toast(next.reason||next.phase);if(next.action==='start')return $('manoStart').onclick();if(next.action==='capture'){$('manoMotionGuide').scrollIntoView({behavior:'smooth',block:'center'});return $('manoMotionConfirm').onclick();}if(next.action==='fit')return $('manoFit').onclick();toast(next.reason||next.phase);};$('manoFitSettings').onclick=()=>$('manoFitMethod').scrollIntoView({behavior:'smooth',block:'center'});$('manoFitMethod').onchange=()=>updateUI();
$('manoRestart').onclick=()=>manoControl('restart');$('manoApply').onclick=()=>manoControl('apply');$('manoRestore').onclick=()=>manoControl('restore');
$('surfaceBackend').onchange=()=>manoControl('restart',{surface_backend:$('surfaceBackend').value});
$('manoPreview').onclick=()=>manoControl('preview',{enabled:!state?.hand_model?.preview});
$('manoFitOptions').onclick=()=>manoControl('fit_options',{penetration_tolerance_mm:Number($('manoPenetration').value)});
$('manoMaterial').onclick=()=>manoControl('material',{material:{young_pa:Number($('manoYoung').value)*1000,thickness_m:Number($('manoThickness').value)/1000,max_compression_m:.002}});
$('officialRestart').onclick=()=>command('ik_official_restart');
$('showTipTargets').onchange=()=>{updateUI();drawPose();};
$('bindSid').onclick=()=>{if(!selectedSids.size)return toast('先在 02 矩阵选中 SID，再在模型曲面点选位置。');$('poseMode').value=poseMode='mano';updateUI();window.startManoBinding?.();toast('在 MANO 曲面点击对应位置；必须属于同一解剖 patch。');};
window.bindManoPoint=async(face,barycentric)=>{const r=await manoControl('bind_sid',{sids:[...selectedSids],face,barycentric});if(r?.ok)toast('SID 曲面位置已保存；请再次触碰核对。');};
let shownManoDimensions=null;
function updateHandUI(live){
 $('showTipTargets').parentElement.lastChild.textContent=poseMode==='mano'?' EMF 安装匹配检查':' 指尖匹配检查';
 const mano=poseMode==='mano',h=state?.hand_model||{},cap=h.capture||{},fit=h.fit||{},working=['running','cancelling'].includes(fit.state);
 const officialFk=h.tracking_source==='official_qnbot_skeleton_fk_v1';
 $('manoCanvas').hidden=!mano;$('poseCanvas').hidden=mano;$('manoGuide').hidden=!mano;$('ikGuide').hidden=mano||poseMode==='official_skeleton';
 const reference=poseMode==='official_skeleton',o=state?.skeleton_official||{},policy=o.runtime_policy||{};
 $('officialGuide').hidden=!reference;$('officialRestart').disabled=!live;
 $('officialStatus').textContent=o.error||`QnBot SDK ${o.sdk_version||policy.sdk_version||'—'} · ${o.ready?(replaying?'录制输出回放':'同源 EMF 对照在线'):o.status||'等待启动'} · ${Number.isFinite(o.processing_ms)?o.processing_ms.toFixed(1)+' ms/帧':'等待骨架'} · 官方原标定 ${policy.calibration_sha256?.slice(0,12)||'—'}`;
 if(reference){
  $('poseTitle').textContent='官方 SDK 骨架对照';$('coordinateLabel').textContent='WRIST LOCAL · 25 JOINTS';$('frameLabel').textContent='腕部局部坐标系';
  $('poseSource').textContent=`QNBOT SDK ${o.sdk_version||policy.sdk_version||'—'} · UNMODIFIED`;$('poseLegend').textContent='官方原始 25 节点 · 无本地姿态修正';
  $('skeletonAge').textContent=replaying?'录制骨架快照':o.ready?`骨架年龄 ${Math.round(o.age_ms)} ms`:'独立官方 SDK worker';
  $('poseStatus').textContent=o.ready?(replaying?'官方 SDK 对照回放':'官方 SDK 对照在线'):o.error||o.status||'等待官方 SDK';
  $('poseEmpty').hidden=!!(live&&o.ready);$('poseEmptyText').textContent=!live?'等待手套连接':o.error||'等待官方 SDK 同帧输出';
 }
 const wuji=poseMode==='wuji2';document.querySelector('.visual-grid').classList.toggle('wuji-mode',wuji);document.querySelector('.pose-stage').hidden=wuji;
 if(wuji){$('ikGuide').hidden=true;$('poseEmpty').hidden=true;$('poseTitle').textContent='Moxian／Wuji2 触觉对照';$('coordinateLabel').textContent='SHARED VIRTUAL WORLD · METRES';$('poseLegend').textContent='同步视角 · 同一尺度 · 紫色阵列 / 黄色接触 · 跨语义 link 连线';$('skeletonAge').textContent=replaying?'离线重算回放':Number.isFinite(state?.wuji2?.source_age_ms)?`输出年龄 ${Math.round(state.wuji2.source_age_ms)} ms`:'独立 Wuji2 worker';}
 if(mano){
  $('poseTitle').textContent='MANO 手部模型';$('coordinateLabel').textContent=h.sensor_mounting?.flat_reference_id?'MANO WRIST · EMF REGISTERED':'WRIST LOCAL · 778 VERTICES';$('poseSource').textContent=officialFk?'QNBOT ROTATIONS → OFFICIAL MANO':'OFFICIAL MANO';$('frameLabel').textContent=h.sensor_mounting?.flat_reference_id?'MANO 腕部参考 · EMF 刚性配准':'腕部局部坐标系';
  $('poseLegend').textContent=$('showTipTargets').checked?'淡紫：触觉点阵 · 黄色小球：实测接触 · 空心：EMF 原点':'淡紫：触觉点阵 · 黄色小球：实测接触 · 动力学关闭';$('skeletonAge').textContent=h.ready?`EMF 残差 ${h.quality?.position_rmse_mm?.toFixed(1)||'—'} mm · ${Math.round(h.age_ms)} ms`:'独立 GPU worker';
  if(officialFk){$('coordinateLabel').textContent='MANO WRIST · QNBOT FK';$('frameLabel').textContent='MANO 腕部参考 · 官方骨架驱动';$('skeletonAge').textContent=h.ready?`官方骨架同帧 · ${Math.round(h.age_ms)} ms`:'等待官方骨架';}
  $('poseEmpty').hidden=!!(window.manoViewHasFrame||(live&&h.ready));$('poseEmptyText').textContent=h.error||'等待 MANO 与新鲜骨架输入';$('poseStatus').textContent=h.ready?(Math.max(...Object.values(h.quality?.alignment?.tip_errors_mm||{}))>15?(h.calibrated?'已标定 · 输入残差较大':'输入残差较大 · 未标定'):h.calibrated?'已标定 · 个性化手型在线':'未标定 · 默认手型'):h.error||'模型初始化中';
 }
 $('manoBadge').textContent=h.checkpoint_preview?'权重对比 · 未应用':h.status==='input_invalid'?'位姿输入异常':working?'标定计算中':h.preview?'候选预览':h.calibrated?'已标定 ✓':fit.accepted?'候选通过 · 待应用':cap.state==='complete'?'采集完成 · 尺寸未应用':'默认固定尺寸';
 $('manoPenetration').disabled=working;$('manoFitOptions').disabled=working;
 if(document.activeElement!==$('manoPenetration'))$('manoPenetration').value=(h.penetration_tolerance_m??.002)*1000;
 if(h.surface_backend==='qnbot'){if(document.activeElement!==$('surfaceBackend'))$('surfaceBackend').value='qnbot';if(mano){$('poseTitle').textContent='Qnbot 官方手部 mesh';$('coordinateLabel').textContent='QNBOT WRIST · 11,598 VERTICES';$('poseSource').textContent='NATIVE 25 DOF · FIXED MODEL';$('frameLabel').textContent='官方腕坐标 · 原生骨架与蒙皮';}}else if(h.ready&&document.activeElement!==$('surfaceBackend'))$('surfaceBackend').value='mano';
 const shared=cap.metadata||{},nativeCalibration=['wuji_native','moxian_wuji'].includes(h.calibration_method)||cap.protocol==='wuji_studio_native_v1',nativeCapture=cap.protocol==='wuji_studio_native_v1',staticCapture=['contact_static_v5','wuji_studio_native_v1'].includes(cap.protocol),motion=['continuous_motion','static_holds','wuji_native_static'].includes(cap.source),contactV3=['contact_motion_v3','contact_motion_v4'].includes(cap.protocol||shared.protocol);
 $('manoMotionGuide').hidden=!motion||['complete','fit_ready'].includes(cap.state)||h.surface_backend==='qnbot'&&!staticCapture;
 $('manoMotionPrompt').textContent=[cap.error,cap.prompt].filter(Boolean).join('。');
 const side=h.hand_side||shared.side||state?.side||'right',guide=window.MoxianMotionGuide;
 const guidePhase=cap.motion_phase||guide?.sequence(cap.stage_id||'range_thumb')[0][0];
 const guideKey=[cap.stage_id,staticCapture?cap.static_endpoint:guidePhase,side,guide?.revision].join(':');
 if(nativeCapture&&$('manoMotionIllustrations').dataset.key!==guideKey){const labels={pinch_index:'拇指／食指对指',pinch_middle:'拇指／中指对指',pinch_ring:'拇指／无名指对指',pinch_pinky:'拇指／小指对指',four_finger_bend_90:'四指弯曲90°',flat_open:'伸直并拢、手掌展平'};const stage=cap.stage_id||'flat_open';$('manoMotionIllustrations').innerHTML=`<figure class="motion-step active"><img src="/wuji-calibration-guides/${stage}-${side}.png" alt="Wuji Studio ${labels[stage]}" style="width:100%;max-height:260px;object-fit:contain"><figcaption><strong>${labels[stage]}</strong><small>Wuji Studio 2026.7.1 原版动作示意</small></figcaption></figure>`;$('manoMotionIllustrations').dataset.key=guideKey;}
 else if(motion&&guide&&$('manoMotionIllustrations').dataset.key!==guideKey){$('manoMotionIllustrations').innerHTML=staticCapture?guide.staticMarkup(cap.stage_id,cap.static_endpoint,side):guide.markup(cap.stage_id||'range_thumb',guidePhase,side);$('manoMotionIllustrations').dataset.key=guideKey;}
 const targetOverview=$('manoContactTargetOverview');
 if(guide&&guide.revision!=='loading'&&targetOverview.dataset.key!==side+guide.revision){targetOverview.innerHTML=guide.overview(side);targetOverview.dataset.key=side+guide.revision;}
 const templateStatus=$('manoContactTemplateStatus'),template=fit.contact_templates;
 const templateKey=[fit.run_id,template?.revision,fit.state].join(':');
 if(templateStatus&&templateStatus.dataset.key!==templateKey){
  templateStatus.dataset.key=templateKey;
  if(template?.actions?.length){templateStatus.replaceChildren(...template.actions.map(a=>{const p=document.createElement('p'),edges=a.edges.filter(e=>e.required);p.textContent=`${guide?.expected(a.stage)?.label||a.stage}：${a.feasible?'参考模板可行':'参考模板仍有残差（保留样本）'} · 最大间隙 ${Math.max(...edges.map(e=>e.template_gap_mm)).toFixed(2)} mm · 法向偏差 ${Math.max(...edges.map(e=>e.template_normal_error_deg)).toFixed(1)}° · 穿透 ${Math.max(...edges.map(e=>e.template_penetration_mm)).toFixed(2)} mm`;return p;}));}
  else templateStatus.textContent='完成新流程采集后生成；独立复测不参与模板选择。';
 }
 $('manoMotionIllustrationNote').textContent=`${side==='left'?'左手':'右手'}掌面示意 · 黄色为期望接触区域 · 橙色为活动手指 · 不是触觉测量${guide?.error?' · '+guide.error:''}`;
 $('manoMotionPhase').textContent=cap.state==='collecting'?`第 ${cap.index+1}/${cap.total} 组 · 第 ${cap.cycle} 轮${cap.cycle===3?'（独立验证）':''} · ${({apart:'分开',approach:'缓慢接近',touch:'指腹轻触并保持',release:'释放并分开',extend:'伸直',curl:'缓慢弯曲／钩状',return:'再次展开',spread:'拇指外展',oppose:'扫向各指',hold:'保持姿态',motion:'缓慢覆盖活动范围'})[cap.motion_phase]||'准备'} · 本轮剩余 ${Math.ceil(cap.cycle_remaining_s||0)} 秒 · 采集预计剩余 ${Math.ceil(cap.estimated_remaining_s||0)} 秒（不含准备与拟合）`:cap.state==='paused'?`已暂停 · 已保留前 ${cap.retained_cycles||0} 轮；点击继续，仅重做第 ${cap.cycle} 轮。`:cap.state==='cancelled'?'已取消；可重试当前动作，已完成动作保留。':'对照示意图摆好起始姿势，点击开始；第三轮只作验证。';
 if(contactV3){
  const pass=cap.capture_pass==='validation'?'独立复测':'拟合采集';
  const missing=Math.max(0,(cap.required_repetitions||2)-(cap.detected_repetitions||0));
  $('manoMotionPhase').textContent=`${pass} · 动作 ${(cap.action_index||0)+1}/${cap.actions_per_pass||17} · 已识别 ${cap.detected_repetitions||0}/2 次 · 窗口剩余 ${Math.ceil(cap.seconds_remaining||0)} 秒 · 预计剩余 ${Math.ceil(cap.estimated_remaining_s||0)} 秒（不含切换与补录）`+(cap.state==='paused'?(missing?` · ${cap.error||`已确认事件保留，只需补录 ${missing} 次`}`:' · 接触次数已满足，点击进入下一动作'):cap.extended?` · 自动延长：等待${cap.waiting_for||'展平回位'}`:cap.protocol==='contact_motion_v4'?` · ${cap.flat_valid?'已展平':'等待展平'} · 两次展平→贴合→展平`:cap.state==='ready'?' · 按自然速度完成两次接触—松开':'');
 }
 const trajectory=$('manoMotionTrajectory');if(trajectory){trajectory.hidden=!contactV3;if(contactV3&&guide){const points=cap.trajectory||[],key=[cap.stage_id,cap.index,points.length,points.at(-1)?.t_ms,cap.detected_repetitions].join(':');if(trajectory.dataset.key!==key){trajectory.innerHTML=guide.trajectoryMarkup(cap);trajectory.dataset.key=key;}}}
 $('manoMotionProgress').value=cap.progress||0;$('manoMotionConfirm').disabled=busy||!['ready','paused','fit_ready'].includes(cap.state)||((contactV3||staticCapture)&&!cap.reference_ready);$('manoMotionConfirm').textContent=cap.state==='paused'?`继续第 ${cap.cycle} 轮`:'已就位，开始此动作';$('manoMotionRetry').disabled=busy||['preparing','complete'].includes(cap.state);$('manoMotionRetry').textContent=contactV3?'重做当前动作（原始数据保留）':'重做整组';if(contactV3)$('manoMotionConfirm').textContent=cap.state==='paused'?'补录缺失次数':`已就位，开始 ${cap.window_seconds||5} 秒采集`;$('manoMotionCancel').disabled=busy||cap.state==='cancelled';if(pendingCommand==='mano_capture_confirm')$('manoMotionConfirm').textContent='正在开始采集…';
 if(contactV3&&cap.state==='paused'&&cap.detected_repetitions>=2&&!busy)$('manoMotionConfirm').textContent='进入下一动作';
 $('manoMotionPhase').classList.remove('static-passed');
 if(staticCapture){
  const posePassed=cap.state==='collecting'&&cap.constraints_ok===true;
  const previousPassed=(cap.aggregate_count||0)>0&&(['ready','complete'].includes(cap.state)||(cap.constraint_failures||[]).some(s=>s.includes('上一动作已通过')));
  $('manoMotionPhase').classList.toggle('static-passed',posePassed||previousPassed);
  $('manoMotionPhase').setAttribute('aria-live','polite');
  $('manoMotionPhase').textContent=`${cap.capture_pass==='validation'?'独立复测':'拟合采集'} · 动作 ${(cap.action_index||0)+1}/17 · ${{waiting_pose:'等待目标姿态',waiting_stable:'等待稳定',hold:'稳定保持',collect:'采集观测'}[cap.motion_phase]||'请就位'} · 稳定窗口 ${cap.stability_frames||0}/${cap.stability_frames_required||20} 帧`+(cap.constraints_ok===false?' · '+(cap.constraint_failures||[]).slice(0,2).join('；'):'');
  if(previousPassed)$('manoMotionPhase').textContent='✓ 上一姿态采集通过 · '+$('manoMotionPhase').textContent;
  if(posePassed)$('manoMotionPhase').textContent='✓ 姿态通过 · '+$('manoMotionPhase').textContent+(cap.motion_phase==='waiting_stable'?' · 请保持稳定':' · 稳定性通过');
  $('manoMotionConfirm').textContent=cap.state==='paused'?'重新保持当前姿态':'开始自动采集（通过后自动切换）';
 }
 $('manoSharedStatus').textContent=contactV3?`${cap.actions_per_pass||17} 动作 × ${cap.window_seconds||5} 秒 × 训练与独立复测 · ${cap.samples||0} 帧 · ${cap.state==='complete'?'独立复测完整留出':'原始 EMF 与同期触觉保存；仅轨迹选定帧施加接触等式'}`:motion?`连续标定 · ${cap.samples||0} 帧 · ${cap.state==='complete'?'完整第三轮及最终复测留出':'原始 EMF 与同期触觉连续保存'}`:cap.source==='official_eight'&&cap.state==='complete'?`已复用八动作初始化 · ${shared.frames||cap.samples} 帧；请开始连续联合标定。`:cap.error||'先完成骨架八动作初始化，再采集 17 个动作及独立复测。';
 if(h.capture_storage_error)$('manoSharedStatus').textContent='采集后台保存失败：'+h.capture_storage_error+'；请保留当前会话并检查磁盘。';
 if(staticCapture)$('manoSharedStatus').textContent=`静态17动作 · 拟合采集 ${cap.completed_train||0}/17 · 独立复测 ${cap.completed_validation||0}/17 · 有效保持片段 ${cap.aggregate_count||0} · 通过后自动下一动作，17类完成后停下，点击 EMF 与骨架拟合；独立复测单独开始。`;
 if(nativeCalibration){
  $('manoSharedStatus').textContent=nativeCapture?`Wuji 原生六动作 · 已完成 ${cap.aggregate_count||0}/6 · 每动作保持1秒＋采集1秒 · 自动切换；全部完成后点击原生拟合。`:'新默认流程：Wuji 原生六动作、原生21自由度objective与solver；历史17动作只读保留。';
  if(nativeCapture){const phases={waiting_movement:'请变换到目标手势',waiting_pose:'等待姿态满足',waiting_stable:'等待稳定',hold:'稳定保持1秒',collect:'采集1秒',done:'六动作采集完成'};$('manoMotionPhase').textContent=(cap.constraints_ok?'✓ 姿态通过 · ':'')+`动作 ${Math.min((cap.index||0)+1,6)}/6 · ${phases[cap.motion_phase]||'请就位'}`+(cap.constraints_ok===false?' · '+(cap.constraint_failures||[]).slice(0,2).join('；'):'');$('manoMotionConfirm').textContent=cap.state==='paused'?'继续当前原生动作':'开始自动六动作采集';}
 }
 const betaValues=h.betas||fit.official_betas;$('manoBetaStatus').textContent=betaValues?`官方 betas（10 维）：${betaValues.map(x=>x.toFixed(3)).join(' · ')}；固定尺寸和安装参数另行辨识。`:'全部训练动作混训，联合辨识手型、几何与安装；Delta 版同时更新姿态策略。';
 if(h.surface_backend==='qnbot')$('manoBetaStatus').textContent='51个共享参数：15段独立骨长＋五个EMF安装变换30维＋全局对齐6维。固定宽厚、指根、关节轴、顶点残差与材料绑定；骨长驱动纵向mesh拉伸，实时QP使用同一模型。';
 const q=h.quality||{};$('manoStatus').textContent=h.error||(!live?'手套离线':h.ready?`${h.calibrated?'已标定':'默认'} · ${q.processing_ms?.toFixed(0)||'—'} ms/帧 · EMF 残差 ${q.position_rmse_mm?.toFixed(1)||'—'} mm · ${h.sensor_mounting?.status==='position_prior'?'安装位置为先验，待多动作标定':h.sensor_mounting?.status==='multi_pose_calibrated'?'安装位置已多动作标定':h.sensor_mounting?.status==='multi_pose_candidate'?'候选安装位置，尚未应用':'EMF 参考未建立'} · 触觉接触单元 ${state.contact?.ready?(state.contact.cell_contact||[]).filter(x=>x===true).length:'未知'} · 无接触力反馈`:'正在初始化所选 GPU 手部模型');
 if(officialFk&&h.ready){const kind=h.checkpoint_preview?'候选权重预览':h.calibrated?'已标定尺寸':'默认尺寸';$('manoStatus').textContent=`${kind} · 固定模型前向 · ${q.processing_ms?.toFixed(0)||'—'} ms/帧 · 无在线优化 · 触觉仅显示`;if(mano)$('poseStatus').textContent=kind+' · 固定模型前向';}
 const inputInvalid=state.skeleton?.status==='input_invalid';
 const thumbUnreliable=inputInvalid||!!(live&&state.skeleton?.ready&&state.skeleton.alignment?.thumb_unreliable);
 $('manoSourceWarning').hidden=!thumbUnreliable;$('manoGoSkeleton').hidden=!thumbUnreliable;
 if(inputInvalid)$('manoSourceWarning').textContent=state.skeleton.error;
 else if(thumbUnreliable)$('manoSourceWarning').textContent=`拇指上游 IK 未匹配：目标残差 ${state.skeleton.alignment.tip_errors_mm.thumb.toFixed(1)} mm，多个关节到达限位。已标定尺寸不代表当前姿态已匹配；请在骨骼视图对照原始目标与实际手势。`;
 $('manoStart').disabled=busy||!live||working||(cap.state==='preparing'||cap.state==='collecting'||['ready','paused'].includes(cap.state)&&(h.surface_backend!=='qnbot'||['contact_motion_v4','contact_static_v5'].includes(cap.protocol)))&&motion||state?.skeleton?.capture?.session_state==='running';
 if(h.surface_backend==='qnbot'&&!staticCapture)$('manoStart').disabled=busy||!live||working||cap.state==='preparing'||state?.skeleton?.capture?.session_state==='running';
 $('manoStart').textContent=pendingCommand==='mano_capture_start'?'正在启动 17 动作标定…':cap.state==='preparing'?'正在后台准备参考模型…':'开始 17 动作联合标定 · 约 204 秒';
 if(h.surface_backend==='qnbot'){$('manoStart').textContent=cap.state==='preparing'?'正在准备静态标定…':cap.state==='fit_ready'&&!fit.validation_pending?'开始 EMF 与骨架拟合':cap.fit_data_ready?'重新采集静态17动作':'开始静态17动作标定';}
 const workflow=window.MoxianCalibrationWorkflow.view(h,apiAlive,busy,pendingCommand);$('manoFit').disabled=!workflow.canFit;$('manoFitCancel').disabled=!working;$('manoApply').disabled=fit.method==='least_squares'||!!h.checkpoint_preview||!fit.accepted||working||!live||!h.ready||!!fit.applied;$('manoPreview').disabled=['geort_motion_v2','contact_motion_v3','contact_motion_v4'].includes(fit.capture_protocol)||!fit.accepted||fit.state!=='preview'||!live||!h.ready;$('manoPreview').textContent=h.preview?'结束预览':'预览候选';
 $('manoResumeFit').disabled=!workflow.canFit||!!h.preview||!!h.checkpoint_preview||fit.method!=='delta'||!fit.solver?.joint_geometry_policy||!fit.run_id;
 const warmup=h.gpu_warmup||{};$('gpuWarmupStatus').hidden=!['running','failed'].includes(warmup.state);$('gpuWarmupStatus').textContent=warmup.state==='failed'?`GPU 预热失败：${warmup.error}；拟合时将重新尝试。`:`GPU 首次编译预热：${warmup.message||'准备固定形状计算图'}（单独计时）`;
 const eta=fit.estimated_remaining_s;const duration=s=>s>=60?`${Math.floor(s/60)} 分 ${Math.ceil(s%60)} 秒`:`${Math.ceil(s)} 秒`;
 $('manoFitETA').hidden=!working;$('manoFitETA').textContent=Number.isFinite(eta)?`${['optimization_budget','budget'].includes(fit.eta_scope)?'优化预算剩余':fit.eta_scope==='stage'?'本阶段预计剩余':'拟合预计剩余'} ${duration(eta)} · 已用 ${duration(Number.isFinite(fit.started_at_ms)?Math.max(0,(Date.now()-fit.started_at_ms)/1000):fit.elapsed_s||0)}（随处理速度更新）`:'正在估算剩余时长…';
 const count=cap.actions_per_pass||16,trainDone=cap.completed_train??Math.min(cap.index||0,count),validDone=cap.completed_validation??Math.max(0,(cap.index||0)-count),storage=h.capture_storage||{},saved=storage.last_saved;
 $('manoDatasetStatus').textContent=`拟合采集 ${trainDone}/${count} · 独立复测 ${validDone}/${count} · 内存 ${cap.samples||0} 帧`+(saved?` · 已保存 ${saved.samples} 帧`:storage.active?' · 正在保存':' · 数据已恢复');
 $('manoDatasetLocation').textContent=saved?.paths?.[0]?`自动保存位置：${saved.paths[0]}`:'自动保存在 output/moxian/hand_model/<设备目录>/motion-capture.json；与下方会话 JSONL 录制分开保存。';
 if(nativeCapture){$('manoDatasetStatus').textContent=`原生标定采集 ${cap.aggregate_count||0}/6 · 原始 EMF ${cap.samples||0} 帧`+(storage.active?' · 正在保存':' · 自动保存');$('manoDatasetLocation').textContent=saved?.paths?.[0]?`自动保存位置：${saved.paths[0]}`:'自动保存在 output/moxian/hand_model/<设备目录>/qnbot/wuji-calibration-capture.json。';$('manoMotionIllustrationNote').textContent=`${side==='left'?'左手':'右手'} · Wuji Studio 原版图示 · 绿色文字表示姿态通过`+(cap.metadata?.gesture_frame?' · 已对齐 Moxian 接收器方向，沿用原版角度范围':'');}
 $('manoCaptureDownload').hidden=!(saved?.samples||cap.samples);
 const fitButton=$('manoStartFit');fitButton.disabled=workflow.disabled;fitButton.textContent=workflow.label;
 $('manoFitBlockedReason').textContent=workflow.reason;
 $('manoWorkflowPhase').textContent=workflow.phase+(Number.isFinite(workflow.remaining)?` · 预计剩余 ${Math.ceil(workflow.remaining)} 秒`:working||storage.active||cap.state==='preparing'?' · 正在估算剩余时长':'');
 const workflowProgress=$('manoWorkflowProgress');workflowProgress.hidden=!(working||pendingCommand==='mano_fit'||storage.active||['collecting','preparing'].includes(cap.state)||warmup.state==='running');if(Number.isFinite(workflow.progress))workflowProgress.value=workflow.progress;else workflowProgress.removeAttribute('value');
 if(h.surface_backend==='mano'&&$('manoFitMethod').value==='least_squares'&&!working)$('manoFitMethod').value='parameters';
 const leastSquares=$('manoFitMethod').value==='least_squares';
 const bones=h.surface_backend==='qnbot';
 for(const option of $('manoFitMethod').options){option.hidden=bones?option.value!==(nativeCalibration?(h.calibration_method||'wuji_native'):'bone_extrinsics'):['bone_extrinsics','wuji_native','moxian_wuji'].includes(option.value);option.disabled=option.hidden;}
 if(bones&&!working)$('manoFitMethod').value=nativeCalibration?(h.calibration_method||'wuji_native'):'bone_extrinsics';
 if(!bones&&['bone_extrinsics','wuji_native','moxian_wuji'].includes($('manoFitMethod').value))$('manoFitMethod').value='parameters';
 $('manoLegacyTrainingNotes').hidden=leastSquares;$('manoLegacyTrainingFooter').hidden=leastSquares;
 if(leastSquares)$('manoBetaStatus').textContent='最小二乘：15维长宽厚 → 小幅运动学 → 区域触觉注册 → 低维表面，按可辨识性开放。离线交替求姿态，实时只做固定前向；没有网络或EMA。';
 $('manoLeastSquaresSettings').hidden=!leastSquares;$('manoLegacyTrainingControls').hidden=leastSquares;$('manoLegacyTrainingExplanation').hidden=leastSquares;
 for(const suffix of ['Seconds','Iterations','Extrinsics','Dimensions','Kinematics','Tactile','Surface'])$('manoLsq'+suffix).disabled=working;
 $('manoFit').textContent=leastSquares?'开始最小二乘标定（仅保存候选）':'开始拟合（通过验收后应用）';
 const muonAvailable=$('manoFitMethod').value==='delta';$('manoTrainOptimizer').querySelector('option[value="muon"]').disabled=!muonAvailable;
 if(!muonAvailable&&!working)$('manoTrainOptimizer').value='adam';
 $('manoOptimizerHint').textContent=muonAvailable?'可选 Muon＋Adam：Delta 隐藏层矩阵用 Muon，几何、偏置和输出层用 Adam；Muon 学习率 0.001，几何 Adam 为 0.01。':'当前模式使用 Adam。Muon 针对网络隐藏层二维矩阵；选择 几何＋Delta 后可启用。';
 if(leastSquares)$('manoOptimizerHint').textContent='分阶段最小二乘：按证据开放共享参数，消去逐帧姿态检查可辨识性；没有学习率、网络或EMA。';
 $('manoFitMethod').querySelector('option[value="least_squares"]').disabled=h.surface_backend!=='qnbot';
 if(bones){
  for(const id of ['manoLeastSquaresSettings','manoLegacyTrainingControls','manoLegacyTrainingExplanation','manoLegacyTrainingNotes','manoLegacyTrainingFooter','manoResumeFit'])$(id).hidden=true;
  $('manoOptimizerHint').textContent='NLopt LD_LBFGS · ftol_rel 1e-10 · 最多20000次评估；通过原始EMF驱动QP的独立复测后应用。';
  $('manoFit').textContent=fit.validation_pending&&cap.state==='complete'?'验收 EMF 与骨架拟合结果':'开始 EMF 与骨架拟合';
  if(fit.bone_lengths_m)$('manoBetaStatus').textContent+=' 骨长 mm：'+Object.entries(fit.bone_lengths_m).map(([f,segments])=>f+' '+Object.values(segments).map(x=>(x*1000).toFixed(1)).join('/')).join(' · ');
  $('manoExtrinsicsResult').hidden=!fit.T_wrist_TX;
  if(fit.T_wrist_TX)$('manoExtrinsicsValues').textContent=JSON.stringify({units:{translation:'m',rotation:'matrix'},T_wrist_TX:fit.T_wrist_TX,T_distal_RX:fit.T_distal_RX},null,2);
  if(fit.contact_templates)$('manoContactTemplateStatus').textContent='接触材料点与连接关系由拟合观测确定后冻结；独立复测未参与选择。';
  if(h.model_hash)$('manoStatus').textContent=`${h.calibrated?'已标定':'预览'} · 实时逐指QP · ${q.processing_ms?.toFixed(1)||'—'} ms mesh/帧 · EMF残差 ${q.position_rmse_mm?.toFixed(1)||'—'} mm · 模型 ${h.model_hash.slice(0,12)}`;
 }
 if(nativeCalibration){
  $('manoStart').textContent=nativeCapture?'重新采集 Wuji 六动作':'开始 Wuji 原生六动作标定';
  $('manoStart').disabled=busy||!live||working||cap.state==='preparing'||nativeCapture&&cap.state==='collecting';
  $('manoFit').textContent='开始 Wuji 原生 EMF 与骨架拟合';
  $('manoBetaStatus').textContent='Wuji原生21自由度手模型；51个共享几何/EMF外参变量。原生六动作、边界、先验、objective和L-BFGS均由固定版本SDK执行。Moxian mesh仅做显示适配。';
  $('manoOptimizerHint').textContent='直接调用 Wuji Studio 2026.7.1 完整原生calibration；成功后应用原生URDF与同版本SDK实时IK。';
  $('manoPenetration').closest('label').hidden=true;$('manoFitOptions').hidden=true;
  $('manoLegacyTargets').hidden=true;
  $('manoMethodExplanation').textContent='原生动作：四组对指、四指弯曲90°、伸直并拢平掌。保持与采样规则、姿态阈值使用固定版本原生规则；直接执行原生objective和L-BFGS，生成原生URDF。实时21自由度IK由同版SDK计算，Moxian mesh仅适配显示。';
 }
 if(h.calibration_method==='moxian_wuji'){
  $('manoBetaStatus').textContent='Moxian原生25关节＋四指各一个指根轴向旋转角，共29自由度；DIP反折≤20°、PIP及拇指反折≤10°；指根侧旋由EMF驱动且限±30°，实时限制整帧转角。仍只拟合15骨长和36维EMF外参，固定宽厚、指根与蒙皮。';
  $('manoMethodExplanation').textContent='沿用Wuji六个静态动作与原生L-BFGS。目标函数使用Moxian原生FK、指腹接触、展平与桌面姿态约束；不再拟合Wuji内置人手模型，不使用21→25骨架显示适配。';
  $('manoOptimizerHint').textContent='Wuji原生NLopt LD_LBFGS · Moxian29自由度FK · 51共享参数＋每姿态29辅助角；检查原始EMF位置与旋转后应用。';
  $('manoFit').textContent='拟合 Moxian 骨长与EMF外参';if(mano)$('poseSource').textContent=h.solver_dof===29?'MOXIAN 25 + 4 FINGER TWISTS':'MOXIAN NATIVE 25 DOF';
  $('manoStart').textContent=nativeCapture?'重新采集六动作':'开始六动作标定';
 }
 $('manoFitMethod').disabled=working;
 for(const id of ['manoTrainSteps','manoTrainSeconds','manoTrainBatch','manoTrainOptimizer'])$(id).disabled=working;
 if(working&&fit.training_options){$('manoTrainSteps').value=fit.training_options.adam_steps;$('manoTrainSeconds').value=fit.training_options.max_seconds;$('manoTrainBatch').value=fit.training_options.batch_size;$('manoTrainOptimizer').value=fit.training_options.optimizer||'adam';}
 window.updateTrainingCharts?.(state);
 $('manoFitProgress').hidden=!working;if(!Number.isFinite(fit.progress))$('manoFitProgress').removeAttribute('value');else $('manoFitProgress').value=fit.progress;
 const liveSolver=h.mano_solver||{};
 const contactGap=Math.max(0,...(h.tactile_pose_correction?.pairs||[]).map(p=>p.gap_mm||0));
 const liveWarnings=liveSolver.constraints_satisfied===false?[`实时约束未满足：最大穿透 ${((liveSolver.collision?.max_penetration_m||0)*1000).toFixed(2)} mm，接触间隙 ${contactGap.toFixed(2)} mm；继续显示当前最优姿态`]:[];
 if(liveSolver.source_tracking_satisfied===false){const names={thumb:'拇指',index:'食指',middle:'中指',ring:'无名指',pinky:'小指'};liveWarnings.push('骨架目标未匹配：'+Object.entries(liveSolver.source_tracking_errors||{}).filter(([,v])=>!v.satisfied).map(([k,v])=>`${names[k]||k} ${v.position_error_mm.toFixed(1)} mm`).join('、'));}
 $('manoCalibrationWarnings').textContent=[...(h.calibrated?h.calibration_warnings||[]:fit.quality_warnings||[]),...liveWarnings].join('；');
 if(h.surface_refinement&&$('manoCalibrationWarnings').textContent)$('manoCalibrationWarnings').textContent='历史完整标定提示：'+$('manoCalibrationWarnings').textContent;
 const refined=null;const togetherGaps=refined?.together_pair_gaps_mm||fit.together_pair_gaps_mm||h.calibration_pair_gaps_mm;$('manoTogetherStatus').textContent=togetherGaps?(refined?'已应用近端修正 · 三截面留出误差：':'留出并拢侧面间隙：')+Object.entries(togetherGaps).map(([pair,gap])=>`${({thumb_index:'拇指—食指',index_middle:'食指—中指',middle_ring:'中指—无名指',ring_pinky:'无名指—小指'})[pair]||pair} ${gap.toFixed(2)} mm`).join(' · '):'';
 const sideDiagnostics=liveSolver.side_contact_diagnostics||{};const sideLines=['thumb_index','middle_ring'].filter(k=>sideDiagnostics[k]?.available).map(k=>`${k==='thumb_index'?'拇—食':'中—无'} ${sideDiagnostics[k].section_gaps_mm.map(x=>x.toFixed(1)).join('/')} mm`);if(sideLines.length)$('manoTogetherStatus').textContent+=' · 实时侧面三截面间隙（几何测量）：'+sideLines.join(' · ');
 if(h.rigid_spacing_report?.accepted){const ds=h.rigid_spacing_report.held_out_surface_distances_mm.flat();$('manoTogetherStatus').textContent=`已应用固定指根标定 · 近端三截面真实曲面间隙 ≤ ${Math.max(...ds).toFixed(2)} mm · 骨段长度固定`; }
 $('manoFitStatus').textContent=(fit.method?`${({parameters:'GPU 个体参数',delta:'几何＋Delta 同步联合训练',flashsac:'FlashSAC buffer 实验（直接损失）'})[fit.method]||fit.method} · `:'')+(fit.error||fit.message||fit.reason||(cap.state==='complete'?(motion?'连续采集已完成；联合拟合通过后整体应用。':'八动作初始化已复用；请开始连续联合标定。'):'完成采集后点击“开始拟合”；实时几何保持固定。'))+(Number.isFinite(fit.position_rmse_mm)?` · 留出定位 ${fit.position_rmse_mm.toFixed(2)} mm · 并拢侧面间隙 ${Number.isFinite(fit.together_gap_mm)?fit.together_gap_mm.toFixed(2):'—'} mm · 法向筛选间隙 ${fit.contact_gap_mm?.toFixed(2)||'—'} mm · 非接触穿透 ${fit.noncontact_penetration_mm?.toFixed(2)||'—'} mm`:'');
 if(staticCapture&&!working&&cap.state==='fit_ready')$('manoFitStatus').textContent=fit.validation_pending?'EMF与骨架拟合完成，候选已保存；点击开始独立复测。':'17类静态动作已完成；点击开始 EMF 与骨架拟合，不会自动重新采集。';
 if(officialFk&&h.calibrated&&!h.calibration_matches_tracking&&!working)$('manoFitStatus').textContent='旧驱动版本的标定报告（当前骨架已更新）：'+$('manoFitStatus').textContent;
 if(nativeCalibration){
  const currentNativeFit=fit.method===(h.calibration_method||'wuji_native')&&(!nativeCapture||fit.capture_session_id===cap.metadata?.motion_session_id);
  $('manoFitStatus').textContent=currentNativeFit?(fit.error||fit.application_error||fit.message||fit.reason||'Wuji 原生标定结果'):(nativeCapture&&cap.state==='complete'?'六动作已采集完成，请点击原生 EMF 与骨架拟合。':'请开始 Wuji 原生六动作标定；历史17动作只读保留。');
  if(working)$('manoFitETA').textContent='原生求解中 · '+(fit.native_profile||'准备原生输入')+' · 耗时由原生收敛决定';
  if(h.model_hash&&h.tracking_source==='wuji_sdk_native21'){
   $('manoStatus').textContent=`Wuji 原生21自由度 IK → Moxian mesh · ${q.processing_ms?.toFixed(1)||'—'} ms mesh/帧 · EMF残差 ${q.position_rmse_mm?.toFixed(1)||'—'} mm · 模型 ${h.model_hash.slice(0,12)}`;
   if(mano){$('poseSource').textContent='WUJI NATIVE 21 DOF → QNBOT MESH';$('frameLabel').textContent='原生腕坐标经固定旋转适配显示';$('poseStatus').textContent='原生标定已应用 · SDK实时IK';}
  }
 }
 if(mano&&h.quality?.skeleton_solver?.startsWith('moxian_emf_pose_slsqp')){$('poseSource').textContent='EMF SLSQP → OFFICIAL MANO';$('frameLabel').textContent='MANO 腕部参考 · SLSQP 骨架驱动';}
 const audit=h.calibration_audit||{},appliedComparison=!!(h.calibrated&&fit.applied&&fit.dimensions_before_mm),current=h.checkpoint_preview?h.checkpoint_preview.base_dimensions_mm:appliedComparison?fit.dimensions_before_mm:h.preview?(audit.active_dimensions_mm||fit.dimensions_before_mm):(h.dimensions_mm||audit.active_dimensions_mm),candidate=h.checkpoint_preview?h.dimensions_mm:appliedComparison?h.dimensions_mm:fit.dimensions_after_mm||audit.candidate_dimensions_mm;
 $('manoDimensionsTitle').textContent=h.checkpoint_preview?'原模型与预览权重尺寸':appliedComparison?'标定前与已应用尺寸':'当前尺寸与候选尺寸';document.querySelectorAll('.mano-dimension-direction').forEach(e=>e.textContent=appliedComparison?'标定前 → 已应用':'当前 → 候选');
 $('manoApplicationStatus').textContent=h.checkpoint_preview?`权重对比预览：${h.checkpoint_preview.label}；原标定未更改。`:`当前：${h.calibrated?'已标定，尺寸已生效 '+(h.applied_revision||'').slice(0,8):'默认尺寸，未应用候选'}；候选：${fit.applied?'已自动应用':fit.accepted?'等待自动应用':fit.state==='preview'?'未通过检查，可预览':'尚未拟合'}${h.preview?'；正在预览候选':''}。`;
 if(fit.method==='least_squares'&&!h.checkpoint_preview)$('manoApplicationStatus').textContent='最小二乘结果仅保存为独立候选；请在“已训练权重”选择“分阶段最小二乘标定”预览，当前模型未替换。';
 if(h.flat_pose_calibration?.applied){
  const flat=h.flat_pose_calibration.validation;
  $('manoApplicationStatus').textContent+=` 平摊姿态标定已生效 · 留出最大弯折 ${flat?.max_bend_deg?.toFixed(1)||'—'}°。`;
  if(!working&&!h.calibrated)$('manoBadge').textContent='平摊姿态已标定 · 默认尺寸';
 }
 const num=x=>Number.isFinite(x)?x.toFixed(1):'—';const segments=a=>a?.segment_lengths_mm?.map(num).join(' / ')||'—';
 const dimensionKey=JSON.stringify([current,candidate]);if(dimensionKey!==shownManoDimensions){shownManoDimensions=dimensionKey;
 $('manoDimensionsRows').innerHTML=fingers.map((f,i)=>{const a=current?.[f],b=candidate?.[f];return `<tr><td>${names[i]}</td><td>${num(a?.surface_length_mm)} → ${num(b?.surface_length_mm)}<br><small>分段 ${segments(a)}<br>→ ${segments(b)}</small></td><td>${num(a?.proximal_width_mm)} × ${num(a?.proximal_depth_mm)}<br>→ ${num(b?.proximal_width_mm)} × ${num(b?.proximal_depth_mm)}</td></tr>`;}).join('');}
 if(h.material&&document.activeElement!==$('manoYoung')&&document.activeElement!==$('manoThickness')){if(!$('manoYoung').dataset.edited)$('manoYoung').value=h.material.young_pa/1000;if(!$('manoThickness').dataset.edited)$('manoThickness').value=h.material.thickness_m*1000;}

}
$('manoYoung').addEventListener('input',()=>$('manoYoung').dataset.edited='1');$('manoThickness').addEventListener('input',()=>$('manoThickness').dataset.edited='1');


$('replayLoad').onclick=()=>$('replayFile').click();
$('replayFile').onchange=async e=>{
 const file=e.target.files[0];if(!file)return;
 if(file.size>100*1024*1024)return toast('回放上限 100 MB；请截取需要的 JSONL 时间段。');
 try{
  const frames=[];let last=-Infinity,recordedTopology=null,recordedWuji=null;const text=await file.text();
  for(const line of text.split('\n')){
   if(!line.trim())continue;const row=JSON.parse(line);if(!row.pose||!row.tactile)continue;
   if(row.hand_model_topology?.ready)recordedTopology=row.hand_model_topology;delete row.hand_model_topology;row.recorded_topology=recordedTopology;
   if(row.wuji2_topology?.ready)recordedWuji=row.wuji2_topology;delete row.wuji2_topology;row.recorded_wuji_topology=recordedWuji;
   const t=(row.timestamp||0)*1000;if(t-last<32)continue;last=t;
   delete row.hand_model_input;delete row.raw_6dof;
   row.connected=true;row.age_ms=0;row.recording={active:false,frames:0,seconds:0,file:null};frames.push(row);
  }
  if(!frames.length)throw Error('文件中没有可回放的手套快照');
  replayFrames=frames;replayIndex=0;replaying=true;replayPlaying=true;replayLast=performance.now();$('replayControls').hidden=false;$('replaySeek').max=frames.length-1;sceneInitialized=false;showReplayFrame();toast(`已加载 ${frames.length} 个回放快照；设备操作已锁定。`);
 }catch(err){toast('回放读取失败：'+err.message);}
};
function showReplayFrame(){
 state=replayFrames[replayIndex];window.isMoxianReplay=true;window.replayingManoTopology=state.recorded_topology;window.replayingWujiTopology=state.recorded_wuji_topology;apiAlive=true;lastFetch=performance.now();$('replaySeek').value=replayIndex;
 const t=state.timestamp-replayFrames[0].timestamp;$('replayTime').textContent=`${t.toFixed(1)} / ${(replayFrames.at(-1).timestamp-replayFrames[0].timestamp).toFixed(1)} s`;
 updateUI();drawPose();drawHeat();
}
$('replayPlay').onclick=()=>{replayPlaying=!replayPlaying;replayLast=performance.now();$('replayPlay').textContent=replayPlaying?'暂停':'播放';};
$('replaySeek').oninput=e=>{replayIndex=Number(e.target.value);replayLast=performance.now();showReplayFrame();};
$('replayExit').onclick=()=>{replaying=false;window.isMoxianReplay=false;replayPlaying=false;replayFrames=[];window.replayingManoTopology=null;window.replayingWujiTopology=null;$('replayControls').hidden=true;$('replayFile').value='';sceneInitialized=false;};
setInterval(()=>{
 if(!replaying||!replayPlaying||!replayFrames.length)return;
 const now=performance.now();const dt=(replayFrames[Math.min(replayIndex+1,replayFrames.length-1)].timestamp-replayFrames[replayIndex].timestamp)*1000;
 if(now-replayLast>=Math.max(25,dt)){if(replayIndex<replayFrames.length-1)replayIndex++;else replayPlaying=false;replayLast=now;showReplayFrame();}
},25);
