/* Bounded local browser verification after the completion handoff. */
const fs=require('fs'),path=require('path');
const {chromium}=require('/home/zz/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root=path.resolve(process.argv[2]);const port=Number(process.argv[3]||8105);const deadline=Date.now()+3600000;
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
(async()=>{
 while(true){
  const statusPath=path.join(root,'delivery_status.json');
  if(fs.existsSync(statusPath)){
   const status=JSON.parse(fs.readFileSync(statusPath));
   if(status.state==='failed')throw new Error('Completion handoff failed: '+status.error);
   if(status.state==='complete')break;
  }
  if(Date.now()>deadline)throw new Error('Timed out awaiting completed visualization');
  await sleep(15000);
 }
 const data=path.join(root,'viewer');const manifest=JSON.parse(fs.readFileSync(path.join(data,'manifest.json')));
 if(manifest.seed_kind!=='shadow_gsr_transfer'||manifest.samples!==490)throw new Error('Wrong experiment dataset');
 if(!manifest.cases.every(r=>r.source_shadow&&r.source_shadow.native_gsr_pass))throw new Error('Unverified Shadow source');
 const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--enable-unsafe-swiftshader','--use-angle=swiftshader']});
 try{
  const page=await browser.newPage({viewport:{width:1650,height:1080}});const errors=[];page.on('pageerror',e=>errors.push(String(e)));
  const response=await page.goto(`http://127.0.0.1:${port}/`,{waitUntil:'networkidle'});if(response.status()!==200)throw new Error('HTTP '+response.status());
  await page.getByRole('heading',{level:1,name:'MANO 刚体 · GraspADMM',exact:true}).waitFor({timeout:30000});await sleep(12000);
  await page.locator('input:not([type]), input[type="text"]').nth(5).click();await page.getByRole('option',{name:'三阶段对比',exact:true}).click();await sleep(1800);
  await page.getByRole('button',{name:'下一条',exact:true}).click();await sleep(1200);
  const text=await page.locator('body').innerText();if(!text.includes('Shadow 来源')||!text.includes('映射后接触点'))throw new Error('Source provenance or mapping details missing');
  await page.getByText('显示法向',{exact:true}).click();await sleep(500);
  await page.screenshot({path:path.join(data,'preview.png')});
  if(errors.length)throw new Error(errors.join('\n'));
  fs.writeFileSync(path.join(data,'browser_validation.json'),JSON.stringify({passed:true,samples:490,all_sources_gsr_passed:true,checks:['HTTP 200','MANO heading','three-stage comparison','next sample','source provenance','mapping errors','normal toggle'],errors},null,2));
 }finally{await browser.close()}
})().catch(e=>{
 fs.writeFileSync(path.join(root,'browser_check_failure.json'),JSON.stringify({passed:false,error:String(e)},null,2));console.error(e);process.exitCode=1;
});
