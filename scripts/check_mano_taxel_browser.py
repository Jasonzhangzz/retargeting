#!/usr/bin/env python3
"""Headless rendered-surface and binary-color checks, independent of hardware."""
import argparse,json
from pathlib import Path
from playwright.sync_api import sync_playwright

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--url',default='http://127.0.0.1:8892');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/opt/google/chrome/chrome',headless=True,args=['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']);page=browser.new_page(viewport={'width':1500,'height':1050});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(a.url);page.wait_for_function('window.debugPrimitive');names=page.locator('#grasp option').evaluate_all('(es)=>es.map(e=>e.value)');samples=[]
        for name in names:
            page.select_option('#grasp',name);page.wait_for_function('n=>debugPrimitive.grid.name===n',arg=name)
            for t in (.0,.37,1.):
                result=page.evaluate('''u=>{
                  const d=debugPrimitive,x=d.grid.scales[0]*(1-u)+d.grid.scales.at(-1)*u;
                  const mesh=d.sample(x).mano,top=d.taxelTopology,points=d.taxels.geometry.attributes.position.array,colors=d.taxels.geometry.attributes.color.array;
                  let error=0;for(let s=0;s<460;s++)for(let k=0;k<3;k++){
                    const expected=top.faces[top.sid_face_ids[s]].reduce((v,i,j)=>v+top.sid_barycentric[s][j]*mesh[i*3+k],0);
                    error=Math.max(error,Math.abs(expected-points[s*3+k]));
                  }
                  const unique=new Set();for(let i=0;i<colors.length;i+=3)unique.add(Array.from(colors.slice(i,i+3)).join(','));
                  return {error,colors:unique.size,taxels:d.taxels.userData,count:d.taxels.geometry.attributes.position.count};
                }''',t)
                assert result['error']<1e-7 and result['colors']<=2 and result['count']==460,result
                samples.append({'name':name,**result})
        page.locator('#taxels').uncheck();assert page.evaluate('!debugPrimitive.taxels.visible');page.locator('#taxels').check()
        page.screenshot(path=str(a.output/'taxels-desktop.png'),full_page=True);page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1');page.screenshot(path=str(a.output/'taxels-mobile.png'),full_page=True)
        assert not errors,errors
        result={'passed':True,'types':len(names),'samples':len(samples),'max_surface_error_m':max(s['error'] for s in samples),'binary_colors':True,'fixed_material_binding':True,'browser_errors':errors,'samples_detail':samples};(a.output/'taxel_browser_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='samples_detail'}));browser.close()
if __name__=='__main__':main()
