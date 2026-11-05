#!/usr/bin/env python3
"""Headless application verification: UI, actual FK uploads and motion continuity."""
import json,argparse,time
from pathlib import Path
from playwright.sync_api import sync_playwright

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--url',default='http://127.0.0.1:8892');a=p.parse_args();samples=[];errors=[]
 with sync_playwright() as w:
  b=w.chromium.launch(executable_path='/opt/google/chrome/chrome',headless=True,args=['--no-sandbox','--disable-dev-shm-usage','--use-angle=swiftshader','--enable-unsafe-swiftshader']);page=b.new_page(viewport={'width':1500,'height':1080});page.on('pageerror',lambda e:errors.append(str(e)));page.goto(a.url);page.wait_for_function('!!window.debugPrimitive',timeout=30000)
  page.evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))');assert page.evaluate('window.debugPrimitive.views.every(v=>!v.renderer.getContext().isContextLost() && v.renderer.info.render.triangles>0)')
  names=page.locator('#grasp option').evaluate_all('(es)=>es.map(e=>e.value)')
  for name in names:
   page.select_option('#grasp',name);page.wait_for_function('(name)=>window.debugPrimitive.grid.name===name',arg=name)
   scales=page.evaluate('window.debugPrimitive.grid.scales');chosen=[scales[0],scales[len(scales)//2],scales[-1]]
   wrist_samples=[]
   for x in chosen:
    result=page.evaluate('(x)=>window.debugPrimitive.sample(x)',x);result.update(name=name);samples.append(result);wrist_samples.append(result['manoJoints'][0])
   if page.evaluate('!!window.debugPrimitive.grid.fixed_wrist'):
    drift=max(abs(x-y) for sample in wrist_samples for x,y in zip(sample,wrist_samples[0]));assert drift<1e-7,(name,drift)
   assert page.evaluate('window.debugPrimitive.views[0].object.position.distanceTo(window.debugPrimitive.views[1].object.position)<1e-10')
   assert page.evaluate('Math.abs(window.debugPrimitive.views[0].object.quaternion.dot(window.debugPrimitive.views[1].object.quaternion))>1-1e-10')
   mid=(scales[len(scales)//2]+scales[len(scales)//2+1])/2
   bone_error=page.evaluate('''x=>{const s=window.debugPrimitive.sample(x),m=window.debugPrimitive.assets.mano;return Math.max(...m.parents.slice(1).map((p,i)=>Math.abs(Math.hypot(...s.manoJoints[i+1].map((v,k)=>v-s.manoJoints[p][k]))-Math.hypot(...m.joints[i+1].map((v,k)=>v-m.joints[p][k])))))}''',mid);assert bone_error<1e-7,(name,bone_error)
   moving=[page.evaluate('(x)=>window.debugPrimitive.sample(x).mano',mid+d) for d in (-1e-7,1e-7)]
   jump=max(abs(x-y) for x,y in zip(*moving));assert jump<1e-4,(name,jump)
   assert page.evaluate('window.debugPrimitive.views[0].object.scale.equals(window.debugPrimitive.views[1].object.scale)')
   if name.startswith('24_') or name.startswith('16_'):page.screenshot(path=str(a.output/(name+'.png')),full_page=True)
  page.select_option('#grasp',names[0]);page.wait_for_function('(name)=>window.debugPrimitive.grid.name===name',arg=names[0]);page.evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))');page.screenshot(path=str(a.output/'comparison-desktop.png'),full_page=True)
  before=page.evaluate('window.currentScale');page.locator('#size').focus();page.locator('#size').press('ArrowRight');after=page.evaluate('window.currentScale');assert before!=after
  page.evaluate('window.debugPrimitive.views[0].camera.position.x+=.01;window.debugPrimitive.views[0].controls.update()');assert page.evaluate('window.debugPrimitive.views[0].camera.position.distanceTo(window.debugPrimitive.views[1].camera.position)<1e-10');page.locator('#reset').click()
  page.locator('#contacts').uncheck();assert page.evaluate('window.debugPrimitive.views[0].markers.every(m=>!m.visible)');page.locator('#contacts').check()
  page.locator('#wire').check();assert page.evaluate('window.debugPrimitive.views.every(v=>v.hand.every(m=>m.material.wireframe))');page.locator('#wire').uncheck()
  page.locator('#transparent').uncheck();assert page.evaluate('window.debugPrimitive.views.every(v=>v.object.material.opacity===1)');page.locator('#transparent').check()
  page.locator('#play').click();first=page.evaluate('window.currentScale');page.wait_for_timeout(250);second=page.evaluate('window.currentScale');assert first!=second;page.locator('#play').click()
  page.set_viewport_size({'width':390,'height':844});page.screenshot(path=str(a.output/'comparison-mobile.png'),full_page=True);assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
  assert not errors,errors
  (a.output/'browser_samples.json').write_text(json.dumps(samples));report={'passed':True,'types':len(names),'knot_samples':len(samples),'js_errors':errors,'object_scales_equal':True,'object_poses_equal':True,'fixed_wrist_checked':True,'binary_contact_toggle':True,'interpolated_bone_lengths_fixed':True,'camera_synchronized':True,'native_slider_keyboard':True,'headless_backend':'SwiftShader','mobile_overflow':False,'scan_playback':True,'submillimetre_slider_continuity':True};(a.output/'browser_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));b.close()
if __name__=='__main__':main()
