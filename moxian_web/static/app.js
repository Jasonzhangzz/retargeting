'use strict';
const $ = id => document.getElementById(id);
const fingers = ['thumb', 'index', 'middle', 'ring', 'pinky'];
const names = ['拇指', '食指', '中指', '无名指', '小指'];
const colors = ['#ed875b', '#d0bc74', '#a8c794', '#7aafa4', '#81a0c6'];
let state = null, heatMode = 'hand', busy = false, lastSequence = null, history = [];
let lastFetch = 0, apiAlive = false, toastTimer;
let senseMode = 'contact', poseMode = 'skeleton', mapping = null, mappingDirty = false, selectedSids = new Set();
let mappingLoading = false, sceneInitialized = false;
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
  busy = true;
  try {
    const response = await fetch('/api/command', {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({command, ...data}), signal: AbortSignal.timeout(6500)});
    const result = await response.json();
    if (!response.ok) throw Error(result.error || '操作失败');
    if (command === 'baseline') toast('正在采集 50 帧，请保持手套无接触。');
    if (command === 'clear_baseline') toast('已清除当前设备基线。');
    if (command === 'record_start') toast('开始记录真实姿态与触觉快照。');
    if (command === 'record_stop') toast('录制已保存，可下载 JSONL。');
    return result;
  } catch (error) { toast(error.message); }
  finally { busy = false; }
}
$('connect').onclick = () => command(state?.connected || ['waiting','connecting','stale'].includes(state?.status) ? 'disconnect' : 'connect');
$('mobileConnect').onclick = () => $('connect').click();
$('baseline').onclick = () => command('baseline');
$('clearBaseline').onclick = () => command('clear_baseline');
$('record').onclick = () => command(state?.recording?.active ? 'record_stop' : 'record_start');

function updateUI() {
  const live = !!(apiAlive && state?.connected);
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
  $('warning').textContent = '已确认佩戴左手；USB PID ' + (state?.device?.pid || '') + ' 标识为' + (state?.device?.usb_side === 'right' ? '右手' : '左手') + '，与实际及固件报告不一致。骨骼 IK 使用已确认的左手模型。';
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
  $('record').disabled = !live && !recording?.active;
  $('record').innerHTML = recording?.active ? '<span>■</span> 停止录制' : '<span>●</span> 开始录制';
  $('recordIndicator').classList.toggle('on', !!recording?.active);
  $('recordInfo').textContent = recording?.active ? `${recording.frames} 个快照 · ${recording.seconds.toFixed(1)} s` : recording?.file ? `已保存 ${recording.frames} 个快照` : '保存姿态、触觉与时间戳快照。';
  $('export').hidden = !recording?.file || !!recording.active;
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
  window.currentMoxianState=state;updateHandUI(live);if(poseMode!=='mano')window.updateManoView?.(state,false);
  if(replaying){$('liveText').textContent='录制回放 · 本地文件';$('deviceStatus').textContent='录制回放';$('footerState').textContent='REPLAY · 不操作设备';}
}

async function poll() {
  if(replaying){setTimeout(poll,100);return;}
  const start = performance.now();
  try {
    const response = await fetch('/api/state', {cache:'no-store', signal:AbortSignal.timeout(2500)});
    if (!response.ok) throw Error('SDK HTTP ' + response.status);
    const nextState=await response.json();if(replaying){setTimeout(poll,100);return;}state=nextState; apiAlive = true; lastFetch = performance.now();
    if (!mappingDirty && state.contact?.mapping_version && mapping?.version !== state.contact.mapping_version) loadMapping();
  } catch { apiAlive = false; }
  updateUI(); drawPose(); drawHeat();
  // Include fetch and rendering time in the frame budget; keep one request in flight.
  setTimeout(poll, document.hidden ? 500 : Math.max(0, 16.7 - (performance.now()-start)));
}

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
  if (poseMode === 'skeleton') { drawSkeleton(); return; }
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
function chooseCamera(mode) {if(poseMode==='mano')window.setManoView?.(mode);camera={yaw:mode==='front'?0:-.3,pitch:mode==='top'?0:mode==='front'?1.5:.52,zoom:1};document.querySelectorAll('[data-camera]').forEach(b=>b.classList.toggle('selected',b.dataset.camera===mode));drawPose();}
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
    body+=`<text x="94" y="235" fill="${ink}" font-size="10">左手 · 掌面</text>`;
  const meta=gestureSteps.find(s=>s.id===id);
  return `<svg viewBox="25 20 220 225" role="img" aria-label="左手${meta.title}，${meta.view}" xmlns="http://www.w3.org/2000/svg">${body}</svg>`;
}
let shownGesture=null;
$('gestureSteps').innerHTML=gestureSteps.map((g,i)=>`<figure data-gesture="${g.id}">${gestureSvg(g.id)}<figcaption><b>${String(i+1).padStart(2,'0')}</b>${g.title}</figcaption></figure>`).join('');
function updateGestureGuide(stage,live,boneReady){
  const id=stage?.stage_id;
  const gesture=gestureSteps.find(g=>g.id===id);
  $('gestureGuide').hidden=!gesture||boneReady;
  if(gesture&&shownGesture!==id){
    $('gestureImage').innerHTML=gestureSvg(id);
    $('gestureTitle').textContent=gesture.title;
    $('gestureView').textContent=gesture.view;
    $('gestureHint').textContent=gesture.hint;
    shownGesture=id;
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
  const statuses={starting:'官方 IK 初始化中',needs_calibration:'需要骨架采集 / 标定',calibrating:'官方 IK 标定计算中',waiting_pose:'等待骨架输出',error:'骨架运行时异常',cancelled:'骨架采集已取消',stale:'骨架数据过期',unavailable:'骨架运行时未安装',not_started:'骨架未启动',ready:'骨架在线'};
  if(bone){$('poseStatus').textContent=boneReady?(Math.max(...Object.values(skeleton.alignment?.tip_errors_mm||{}))>15?'骨架在线 · 指尖目标残差较大':'25 节点骨架在线'):statuses[skeleton?.status]||'等待骨架';$('poseEmptyText').textContent=!live?'等待手套连接':skeleton?.error||statuses[skeleton?.status]||'请按下方提示完成八动作标定';}
  $('ikPrompt').textContent=skeleton?.error||(!live?'连接手套后可进行骨架标定。':boneReady?'标定已就绪，正在显示官方 IK 骨架。':skeleton?.capture?.session_state==='cancelled'?'采集已取消，点击重启骨架以继续。':stage?.prompt||statuses[skeleton?.status]||'等待官方 IK 运行时');
  const steps=gestureSteps.map(g=>g.id);
  $('ikStep').textContent=stage?`${steps.indexOf(stage.stage_id)+1} / 8 · ${stage.collected_sample_count} / ${stage.required_sample_count}`:'LEFT HAND';
  $('ikProgress').max=stage?.required_sample_count||30;$('ikProgress').value=stage?.collected_sample_count||0;
  const running=skeleton?.capture?.session_state==='running';
  $('ikConfirm').disabled=!live||!running||stage?.state!=='awaiting_confirmation';$('ikCancel').disabled=!live||!running||!stage;$('ikRetry').disabled=!live||!running||!stage;$('ikRestart').disabled=!live;
}
function drawSkeleton(){
  const {ctx:c,width:w,height:h}=canvasContext($('poseCanvas'));const pose=state?.skeleton?.pose;
  if(!apiAlive||!state?.connected||!state.skeleton?.ready||!pose)return;
  const nodes=Object.fromEntries(pose.names.map((name,i)=>[name,pose.positions_local[i].map(x=>x*1000)]));
  if(!sceneInitialized){cameraCenter=[0,1,2].map(k=>(Math.min(...Object.values(nodes).map(p=>p[k]))+Math.max(...Object.values(nodes).map(p=>p[k])))/2);sceneInitialized=true;}
  const P=p=>project(p,w,h);
  function line(a,b,color,width=1){const A=P(a),B=P(b);c.beginPath();c.moveTo(A[0],A[1]);c.lineTo(B[0],B[1]);c.strokeStyle=color;c.lineWidth=width;c.lineCap='round';c.stroke();}
  const center=cameraCenter;
  for(let n=-120;n<=120;n+=20){line([center[0]+n,center[1]-120,0],[center[0]+n,center[1]+120,0],'#364137',.6);line([center[0]-120,center[1]+n,0],[center[0]+120,center[1]+n,0],'#364137',.6);}
  (state.skeleton.chains||[]).forEach((chain,i)=>{for(let k=1;k<chain.length;k++)line(nodes[chain[k-1]],nodes[chain[k]],colors[i]+'bb',3);chain.slice(1).forEach((name,k)=>{const A=P(nodes[name]);c.beginPath();c.arc(A[0],A[1],k===chain.length-2?4:2.6,0,Math.PI*2);c.fillStyle=colors[i];c.fill();});const tip=P(nodes[chain.at(-1)]);c.font='10px "Noto Sans CJK SC",sans-serif';c.fillStyle=colors[i];c.fillText(names[i],tip[0]+9,tip[1]-7);});
  const targets=state.skeleton.targets;
  if($('showTipTargets').checked&&targets&&Math.abs(targets.timestamp_ms-pose.timestamp_ms)<=25)targets.positions_local.forEach((p,i)=>{const point=p.map(x=>x*1000),tip=nodes[state.skeleton.chains[i].at(-1)],A=P(point);c.setLineDash([3,4]);line(tip,point,colors[i],1);c.setLineDash([]);c.beginPath();c.arc(A[0],A[1],7,0,Math.PI*2);c.strokeStyle=colors[i];c.lineWidth=2;c.stroke();});
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
$('manoFlatReference').onclick=()=>manoControl('flat_reference');
$('manoStart').onclick=()=>manoControl('capture_start');$('manoResume').onclick=()=>manoControl('capture_resume');
$('manoConfirm').onclick=()=>manoControl('capture_confirm',{request_id:state?.hand_model?.capture?.request_id});
$('manoRetry').onclick=()=>manoControl('capture_retry',{request_id:state?.hand_model?.capture?.request_id});
$('manoCancel').onclick=()=>manoControl('capture_cancel');$('manoFit').onclick=()=>manoControl('fit');$('manoFitCancel').onclick=()=>manoControl('fit_cancel');
$('manoRestart').onclick=()=>manoControl('restart');$('manoApply').onclick=()=>manoControl('apply');$('manoRestore').onclick=()=>manoControl('restore');
$('manoPreview').onclick=()=>manoControl('preview',{enabled:!state?.hand_model?.preview});
$('manoVerify').onclick=async()=>{const patch_ids=state?.hand_model?.capture?.expected_patches||[];const r=await manoControl('verify',{patch_ids});if(r?.ok)toast('已记录你对当前指腹分区的核对；位置不符时请先在 02 修正 SID。');};
$('manoFitOptions').onclick=()=>manoControl('fit_options',{penetration_tolerance_mm:Number($('manoPenetration').value)});
$('manoMaterial').onclick=()=>manoControl('material',{material:{young_pa:Number($('manoYoung').value)*1000,thickness_m:Number($('manoThickness').value)/1000,max_compression_m:.002}});
$('showTipTargets').onchange=()=>{updateUI();drawPose();};
$('bindSid').onclick=()=>{if(!selectedSids.size)return toast('先在 02 矩阵选中 SID，再在模型曲面点选位置。');$('poseMode').value=poseMode='mano';updateUI();window.startManoBinding?.();toast('在 MANO 曲面点击对应位置；必须属于同一解剖 patch。');};
window.bindManoPoint=async(face,barycentric)=>{const r=await manoControl('bind_sid',{sids:[...selectedSids],face,barycentric});if(r?.ok)toast('SID 曲面位置已保存；请再次触碰核对。');};
let shownManoGesture=null;
function updateHandUI(live){
 $('showTipTargets').parentElement.lastChild.textContent=poseMode==='mano'?' EMF 安装匹配检查':' 指尖匹配检查';
 const mano=poseMode==='mano',h=state?.hand_model||{},cap=h.capture||{},fit=h.fit||{},working=fit.state==='running';
 $('manoCanvas').hidden=!mano;$('poseCanvas').hidden=mano;$('manoGuide').hidden=!mano;$('ikGuide').hidden=mano;
 if(mano){
  $('poseTitle').textContent='MANO 手部模型';$('coordinateLabel').textContent=h.sensor_mounting?.flat_reference_id?'MANO WRIST · EMF REGISTERED':'WRIST LOCAL · 778 VERTICES';$('poseSource').textContent='PERSONALIZED MANO';$('frameLabel').textContent=h.sensor_mounting?.flat_reference_id?'MANO 腕部参考 · EMF 刚性配准':'腕部局部坐标系';
  $('poseLegend').textContent=$('showTipTargets').checked?'绿色：接触点 · 空心：EMF 原点 · 虚线：安装残差':'绿色亮点：确认接触 · 动力学关闭';$('skeletonAge').textContent=h.ready?`EMF 残差 ${h.quality?.position_rmse_mm?.toFixed(1)||'—'} mm · ${Math.round(h.age_ms)} ms`:'独立 GPU worker';
  $('poseEmpty').hidden=!!(live&&h.ready);$('poseEmptyText').textContent=h.error||'等待 MANO 与新鲜骨架输入';$('poseStatus').textContent=h.ready?(Math.max(...Object.values(h.quality?.alignment?.tip_errors_mm||{}))>15?'目标残差较大 · 待核对':h.calibrated?'个性化手型在线':'默认固定尺寸 · 未应用拟合'):h.error||'模型初始化中';
 }
 $('manoBadge').textContent=h.preview?'候选预览':h.calibrated?'尺寸已应用':working?'尺寸拟合中':fit.accepted?'候选通过 · 待应用':cap.state==='complete'?'采集完成 · 尺寸未应用':'默认固定尺寸';
 $('manoPenetration').disabled=working;$('manoFitOptions').disabled=working;
 if(document.activeElement!==$('manoPenetration'))$('manoPenetration').value=(h.penetration_tolerance_m??.005)*1000;
 $('manoFlatReference').disabled=!live||!h.ready||working||cap.state==='collecting';
 const q=h.quality||{};$('manoStatus').textContent=h.error||(!live?'手套离线':h.ready?`${h.calibrated?'已标定':'默认'} · ${q.processing_ms?.toFixed(0)||'—'} ms/帧 · EMF 残差 ${q.position_rmse_mm?.toFixed(1)||'—'} mm · ${h.sensor_mounting?.status==='position_prior'?'安装位置为先验，待多动作标定':h.sensor_mounting?.status==='multi_pose_calibrated'?'安装位置已多动作标定':'EMF 参考未建立'} · 接触点 ${h.contact_points?.length||0} · 无接触力反馈`:'正在初始化 MANO / MuJoCo Warp');
 $('manoCapture').hidden=!cap.stage_id||cap.state==='idle';$('manoStep').textContent=`${(cap.index||0)+1} / 10 · ${cap.stage_id||''}`;$('manoCycles').textContent=cap.stage_id?.startsWith('pinch_')?`${cap.cycles||0} / 2 次接触释放`:'';
 $('manoPrompt').textContent=cap.prompt||'';$('manoQuality').textContent=cap.error||`状态：${cap.state||'等待'} · ${cap.samples||0} 帧；未核对分区不能用于强尺寸约束。`;
 $('manoProgress').value=cap.progress||0;
 const alias={open:'open',together:'close',mcp90:'mcp90',hook:'mcp90_pip90',fist:'mcp90_pip90',pinch_index:'pinch_first',pinch_middle:'pinch_middle',pinch_ring:'pinch_ring',pinch_pinky:'pinch_little',open_repeat:'open'};
 if(cap.stage_id&&shownManoGesture!==cap.stage_id){$('manoGesture').innerHTML=cap.stage_id==='hook'?hookGestureSvg():gestureSvg(alias[cap.stage_id])+(cap.stage_id==='fist'?'<p>参照右侧握拳图，自然虚握。</p>':'');shownManoGesture=cap.stage_id;}
 const patchChecked=(cap.expected_patches||[]).every(id=>mapping?.confirmed?.includes(id));$('manoConfirm').disabled=!live||cap.state!=='awaiting_confirmation'||!patchChecked;
 if(!patchChecked&&!cap.error)$('manoQuality').textContent='先在 02 的触觉矩阵核对拇指和当前手指末端，再点击确认当前接触分区；位置不符请先修正 SID。';
 $('manoVerify').disabled=!live||cap.state==='collecting'||patchChecked;$('manoVerify').textContent=patchChecked?'当前接触分区已核对 ✓':'确认当前接触分区';$('manoRetry').disabled=!live||!['awaiting_confirmation','collecting','failed'].includes(cap.state);$('manoCancel').disabled=!live||!['collecting','awaiting_confirmation','failed'].includes(cap.state);
 $('manoVerify').hidden=!cap.expected_patches?.length;$('manoStart').disabled=!live||!h.ready||working;$('manoResume').disabled=!live||!cap.completed_groups?.length||cap.state==='complete';
 $('manoFit').disabled=cap.state!=='complete'||working;$('manoFitCancel').disabled=!working;$('manoApply').disabled=!fit.accepted||working;$('manoPreview').disabled=fit.state!=='preview';$('manoPreview').textContent=h.preview?'结束预览':'预览候选';
 $('manoFitProgress').hidden=!working;$('manoFitProgress').value=fit.progress||0;
 $('manoFitStatus').textContent=fit.error||fit.message||(fit.state==='preview'?`${fit.reason} · 留出定位 ${fit.position_rmse_mm.toFixed(2)} mm · 接触间隙 ${fit.contact_gap_mm.toFixed(2)} mm · 非接触穿透 ${fit.noncontact_penetration_mm.toFixed(2)} mm`:cap.state==='complete'?'十动作采集已完成；请拟合并应用候选，当前仍使用默认尺寸。':'采集完成后仍需拟合与应用；实时几何保持固定。');
 if(h.material&&document.activeElement!==$('manoYoung')&&document.activeElement!==$('manoThickness')){if(!$('manoYoung').dataset.edited)$('manoYoung').value=h.material.young_pa/1000;if(!$('manoThickness').dataset.edited)$('manoThickness').value=h.material.thickness_m*1000;}

}
$('manoYoung').addEventListener('input',()=>$('manoYoung').dataset.edited='1');$('manoThickness').addEventListener('input',()=>$('manoThickness').dataset.edited='1');

function hookGestureSvg(){return `<svg viewBox="0 0 240 185" role="img" aria-label="钩指侧视补充示意，指根伸直，中末节弯曲"><path d="M80 164 L80 95 Q80 86 91 86 L104 94 L104 164" fill="#e4e8dc" stroke="#73836a" stroke-width="2"/><path d="M91 101 L91 52 L132 52 L134 78 M95 106 L97 58 L139 58 L141 84 M100 112 L104 65 L145 65 L146 91 M104 117 L111 73 L151 73 L151 98" fill="none" stroke="#7b8b6c" stroke-width="9" stroke-linecap="round" stroke-linejoin="round"/><path d="M104 140 Q140 146 132 118" fill="none" stroke="#a4b196" stroke-width="12" stroke-linecap="round"/><text x="22" y="86" fill="#899b77" font-size="11">指根伸直</text><text x="148" y="43" fill="#dc592d" font-size="11">中末节弯曲</text><text x="55" y="182" fill="#a7b19b" font-size="9">钩指 · 非官方补充示意</text></svg>`;}

$('replayLoad').onclick=()=>$('replayFile').click();
$('replayFile').onchange=async e=>{
 const file=e.target.files[0];if(!file)return;
 if(file.size>100*1024*1024)return toast('回放上限 100 MB；请截取需要的 JSONL 时间段。');
 try{
  const frames=[];let last=-Infinity,recordedTopology=null;const text=await file.text();
  for(const line of text.split('\n')){
   if(!line.trim())continue;const row=JSON.parse(line);if(!row.pose||!row.tactile)continue;
   if(row.hand_model_topology?.ready)recordedTopology=row.hand_model_topology;delete row.hand_model_topology;row.recorded_topology=recordedTopology;
   const t=(row.timestamp||0)*1000;if(t-last<32)continue;last=t;
   delete row.hand_model_input;delete row.raw_6dof;
   row.connected=true;row.age_ms=0;row.recording={active:false,frames:0,seconds:0,file:null};frames.push(row);
  }
  if(!frames.length)throw Error('文件中没有可回放的手套快照');
  replayFrames=frames;replayIndex=0;replaying=true;replayPlaying=true;replayLast=performance.now();$('replayControls').hidden=false;$('replaySeek').max=frames.length-1;sceneInitialized=false;showReplayFrame();toast(`已加载 ${frames.length} 个回放快照；设备操作已锁定。`);
 }catch(err){toast('回放读取失败：'+err.message);}
};
function showReplayFrame(){
 state=replayFrames[replayIndex];window.replayingManoTopology=state.recorded_topology;apiAlive=true;lastFetch=performance.now();$('replaySeek').value=replayIndex;
 const t=state.timestamp-replayFrames[0].timestamp;$('replayTime').textContent=`${t.toFixed(1)} / ${(replayFrames.at(-1).timestamp-replayFrames[0].timestamp).toFixed(1)} s`;
 updateUI();drawPose();drawHeat();
}
$('replayPlay').onclick=()=>{replayPlaying=!replayPlaying;replayLast=performance.now();$('replayPlay').textContent=replayPlaying?'暂停':'播放';};
$('replaySeek').oninput=e=>{replayIndex=Number(e.target.value);replayLast=performance.now();showReplayFrame();};
$('replayExit').onclick=()=>{replaying=false;replayPlaying=false;replayFrames=[];window.replayingManoTopology=null;$('replayControls').hidden=true;$('replayFile').value='';sceneInitialized=false;};
setInterval(()=>{
 if(!replaying||!replayPlaying||!replayFrames.length)return;
 const now=performance.now();const dt=(replayFrames[Math.min(replayIndex+1,replayFrames.length-1)].timestamp-replayFrames[replayIndex].timestamp)*1000;
 if(now-replayLast>=Math.max(25,dt)){if(replayIndex<replayFrames.length-1)replayIndex++;else replayPlaying=false;replayLast=now;showReplayFrame();}
},25);
