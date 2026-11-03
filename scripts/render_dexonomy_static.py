#!/usr/bin/env python3
"""Render a static HTML gallery from cached original Shadow targets."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,html,json,time
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco,numpy as np
from PIL import Image,ImageDraw
from retarget.viewer.snapshot import get_snapshot
from retarget.env.grasp.dexonomy_replay import write_json

def render(batch):
    batch=Path(batch).resolve();image_dir=batch/'images';image_dir.mkdir(exist_ok=True);records=[];started=time.time()
    for manifest in sorted(batch.glob('*/shards/*/manifest.json')):
        snapshot=get_snapshot(manifest.parent)
        for i,entry in enumerate(snapshot.manifest['cases']):
            model=snapshot.model(i);data=mujoco.MjData(model)
            model.geom_group[(model.geom_contype!=0)|(model.geom_conaffinity!=0)]=3
            model.vis.headlight.active=1
            model.vis.headlight.ambient[:]=[.55,.55,.55]
            model.vis.headlight.diffuse[:]=[.8,.8,.8]
            model.vis.headlight.specular[:]=[.1,.1,.1]
            # Display-only colors make hand/object overlap easy to inspect.
            for g in range(model.ngeom):
                if model.geom_group[g]==3:continue
                body_name=model.body(int(model.geom_bodyid[g])).name
                model.geom_matid[g]=-1
                model.geom_rgba[g]=[.65,.74,.83,1] if body_name.startswith('hand-') else [.92,.58,.24,1]
            opt=mujoco.MjvOption();opt.geomgroup[3]=False
            opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]=True
            camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
            camera.azimuth=135.;camera.elevation=-25.
            rec={'grasp_type':entry['grasp_type'],'case':entry['name'],'object':Path(entry['object']['source_record']).parts[-3],
                'source_row':entry['object']['source_row'],'object_scale':snapshot.poses['object_scale'][i].tolist(),
                'stage_contact_counts':snapshot.poses['stage_contact_counts'][i].tolist(),'images':{}}
            with mujoco.Renderer(model,height=480,width=640) as renderer:
                for si,stage in enumerate(('pregrasp','grasp','squeeze')):
                    data.qpos[:]=snapshot.poses['stage_qpos'][i,si]
                    mujoco.mj_kinematics(model,data);mujoco.mj_collision(model,data);mujoco.mj_camlight(model,data)
                    ids=[b for b in range(1,model.nbody) if model.body_mocapid[b]<0]
                    positions=data.xpos[ids]
                    lower=positions.min(0);upper=positions.max(0)
                    for g in range(model.ngeom):
                        if model.geom_group[g]==3:continue
                        radius=model.geom_rbound[g];lower=np.minimum(lower,data.geom_xpos[g]-radius);upper=np.maximum(upper,data.geom_xpos[g]+radius)
                    camera.lookat[:]=(lower+upper)*.5;camera.distance=max(.3,float(np.linalg.norm(upper-lower))*1.15)
                    renderer.update_scene(data,camera=camera,scene_option=opt)
                    rgb=renderer.render().copy()
                    if rgb.std()<2:raise ValueError('Rendered image is blank')
                    image=Image.fromarray(rgb);draw=ImageDraw.Draw(image)
                    draw.rectangle((0,0,640,45),fill=(22,27,34))
                    draw.text((12,7),entry['grasp_type']+' / '+stage,fill=(240,240,245))
                    draw.text((12,25),'Static original target | geometric contacts: '+str(rec['stage_contact_counts'][si]),fill=(160,220,240))
                    filename=f"{entry['grasp_type']}_{entry['name']}_{stage}.png"
                    temp=image_dir/(filename+'.tmp.png');image.save(temp);temp.replace(image_dir/filename);rec['images'][stage]='images/'+filename
            records.append(rec)
        print('RENDERED',manifest.parent.parent.parent.name,len(records),flush=True)
    payload=json.dumps(records,ensure_ascii=False).replace('</','<\/')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shadow Hand 原始抓取静态预览</title>
<style>body{font:15px system-ui;background:#111820;color:#e8eef5;margin:0;padding:28px}h1{font-size:25px}p{color:#aebfd0;line-height:1.6}select{background:#243342;color:white;border:1px solid #567;padding:8px;border-radius:6px}#grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px;margin-top:24px}article{background:#1b2733;border:1px solid #34485a;border-radius:10px;overflow:hidden}article img{width:100%;display:block}article header,article footer{padding:12px}h2{font-size:16px;margin:0 0 8px}small{color:#adc2d4;word-break:break-all}article select{max-width:100%}.toolbar{position:sticky;top:0;background:#111820ee;padding:12px 0;z-index:1}</style>
<h1>Shadow Hand 原始抓取 · 静态还原</h1>
<p>100 条原始姿态，31 个 grasp type。完整保留原始手姿态、物体位姿和尺寸。只做运动学和几何碰撞查询；未执行动作仿真、接触力求解或力闭合检查。<br>接触标记表示当前目标姿态上的几何接触。名义 grasp 阶段可能留有间隙，可切换到 squeeze 查看收紧目标。</p>
<div class="toolbar">阶段 <select id="stage"><option value="grasp">grasp 抓取目标</option><option value="pregrasp">pregrasp 预抓取目标</option><option value="squeeze">squeeze 收紧目标</option></select> <span id="stats"></span></div>
<div id="grid"></div><script>const records=PAYLOAD;const groups=new Map();for(const r of records){if(!groups.has(r.grasp_type))groups.set(r.grasp_type,[]);groups.get(r.grasp_type).push(r)}const grid=document.querySelector('#grid');for(const[type,rows]of groups){const a=document.createElement('article');const header=document.createElement('header');const title=document.createElement('h2');title.textContent=type;header.append(title);const sel=document.createElement('select');rows.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent='样本 '+(i+1)+' · '+r.object;sel.append(o)});header.append(sel);const img=document.createElement('img');img.loading='lazy';const footer=document.createElement('footer');a.append(header,img,footer);const update=()=>{const r=rows[Number(sel.value)],stage=document.querySelector('#stage').value,si={pregrasp:0,grasp:1,squeeze:2}[stage];img.src=r.images[stage];img.alt=type+' '+stage;footer.textContent='几何接触 '+r.stage_contact_counts[si]+' 个 · 缩放 '+r.object_scale.map(x=>x.toPrecision(4)).join(', ')};sel.onchange=update;a.update=update;grid.append(a);update()}document.querySelector('#stage').onchange=()=>[...grid.children].forEach(a=>a.update());document.querySelector('#stats').textContent=records.length+' 个样本 / '+groups.size+' 个抓型';</script></html>'''.replace('PAYLOAD',payload)
    (batch/'index.html').write_text(page)
    write_json(batch/'preview.json',{'cases':len(records),'images':len(records)*3,'types':len({r['grasp_type'] for r in records}),'seconds':time.time()-started,'static_only':True,'renderer':'MuJoCo EGL','records':records})
    print(json.dumps({'cases':len(records),'images':len(records)*3,'seconds':time.time()-started,'html':str(batch/'index.html')}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('batch',type=Path);render(p.parse_args().batch)
