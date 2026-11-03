// DOM and numeric checks only: this does not validate WebGL rendering.
// NODE_PATH=/path/to/node_modules node scripts/check_grasp_denoising_page.cjs [page.html] [poses.json]
const fs=require('fs'),vm=require('vm'),assert=require('assert'),path=require('path');
const {JSDOM}=require('jsdom');
const html=fs.readFileSync(process.argv[2]||path.join(__dirname,'../output/grasp_denoising_inspector/index.html'),'utf8');
// Execute the actual application against a DOM; Plotly's drawing is mocked.
const marker='<script>\nconst DATA=',start=html.indexOf(marker);
assert(start>0);
const code=html.slice(start+'<script>\n'.length,html.indexOf('</script>',start));
const dom=new JSDOM(html.slice(0,start).replace(/<script>[\s\S]*?<\/script>/g,''),{url:'http://127.0.0.1:8093/',runScripts:'outside-only'});
const ctx=dom.getInternalVMContext(),window=dom.window;
window.structuredClone=structuredClone;
window.Plotly={react:async(p,data,layout)=>{p.traces=data;p.layout=layout;p.on=(name,cb)=>{p[name]=cb}},relayout:async(p,layout)=>{p.lastRelayout=layout}};
vm.runInContext(code,ctx);
const evaluate=s=>vm.runInContext(s,ctx);
async function flush(){await evaluate('queue');assert(window.document.getElementById('error').hidden,window.document.getElementById('error').textContent)}
async function set(id,v){const e=window.document.getElementById(id);e.value=String(v);e.onchange();await flush()}
(async()=>{
 await flush();assert.equal(window.document.querySelectorAll('.panel').length,4);
 assert.equal(window.document.getElementById('stage-select').value,'tight');
 if(evaluate('DATA.fractions.includes(.5)'))assert.equal(window.document.getElementById('fraction').value,'0.5');
 assert.equal(window.document.getElementById('grasp').options.length,31);
 assert.equal(window.document.querySelectorAll('#rows tr').length,20);
 assert.equal(window.document.querySelectorAll('.plot')[0].traces.filter(t=>t.type==='mesh3d').length,2);
 for(const stage of ['tight','noise','removed','perpendicular','clean']){await set('stage-select',stage);assert(window.document.querySelector('.note').textContent.length>20)}
 if(evaluate('DATA.fractions.includes(.9)')){await set('grasp',17);await set('fraction',.9);assert(window.document.querySelector('.note').textContent.includes('下限'));}
 for(const g of [0,17,21,23,30]){await set('grasp',g);for(const r of [0,1,5])await set('round',r)}
 await set('seed',23);await set('fraction',.2);await set('geometry','skeleton');
 window.document.querySelector('#rows tr').click();await flush();assert(window.document.querySelector('#rows tr').classList.contains('selected'));
 for(const id of ['object','bones','transparent']){const e=window.document.getElementById(id);e.checked=!e.checked;e.onchange();await flush()}
 window.document.getElementById('side').click();await flush();
 const poses=evaluate(`JSON.stringify(DATA.cases.flatMap(c=>c.paths.flatMap(path=>path.rounds.flatMap(r=>DATA.methods.flatMap(method=>DATA.fractions.flatMap(f=>['clean','tight','noise','removed','perpendicular'].map(stage=>({group:c.group,seed:path.seed,round:r.round,method,fraction:f,stage,q:methodPose(c,r,method,f,stage)}))))))))`);
 if(process.argv[3])fs.writeFileSync(process.argv[3],poses);
 // Exercise camera synchronization, numeric bounds, and non-default controls.
 const panels=window.document.querySelectorAll('.plot');await panels[0]['plotly_relayout']({'scene.camera':{eye:{x:1,y:1,z:1}}});
 assert(panels[1].lastRelayout['scene.camera']);
 assert(evaluate('DATA.cases.every(c=>bounds(c).every(a=>a.every(Number.isFinite)))'));
 console.log(JSON.stringify({passed:true,dom_only:true,types:31,panels:4,rows:20,exportedPoseCount:JSON.parse(poses).length,controls:'cases, rounds, seeds, fractions, all stages, layers, table highlight, cameras'},null,2));
})().catch(e=>{console.error(e);process.exitCode=1});
