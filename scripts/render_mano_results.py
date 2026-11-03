#!/usr/bin/env python
"""Offline EGL previews of actual solved MANO cache rows; no optimization."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,mujoco
from PIL import Image,ImageDraw,ImageFont
from retarget.viewer.mano_results import *
from scripts.serve_mano_results import palm_camera

def numbers(x):return ' '.join(map(str,np.asarray(x).reshape(-1).tolist()))
def render(row,view):
    s=row['snapshot'];i=row['index'];v,f=s.mano_mesh(i);ov,of=s.object_mesh(i);width,height=780,680
    xml='<mujoco><visual><global offwidth="780" offheight="680"/><rgba haze=".06 .09 .13 1"/></visual><asset>'
    for name,a,b in [('hand',v,f),('object',ov,of)]:xml+='<mesh name="'+name+'" vertex="'+numbers(a)+'" face="'+numbers(b)+'"/>'
    xml+='</asset><worldbody><light pos="0 0 2"/><geom type="mesh" mesh="hand" contype="0" conaffinity="0" rgba=".87 .67 .53 1"/><geom type="mesh" mesh="object" contype="0" conaffinity="0" rgba=".32 .56 .85 .22"/></worldbody></mujoco>'
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);mujoco.mj_kinematics(model,data);mujoco.mj_camlight(model,data)
    model.vis.headlight.ambient[:]=.7;model.vis.headlight.diffuse[:]=.6;model.vis.headlight.active=1
    eye,center,up=palm_camera(row)
    if view=='oblique':
        axis=np.cross(center-eye,up);axis/=np.linalg.norm(axis);eye=center-(eye-center)*.75+axis*np.linalg.norm(center-eye)*.65
    forward=center-eye;forward/=np.linalg.norm(forward);right=np.cross(forward,up);right/=np.linalg.norm(right);up=np.cross(right,forward)
    hp=s.poses['mapped_hand_points'][i];op=s.poses['mapped_object_points'][i]
    good=(np.abs(s.poses['mapped_distances_m'][i])<=.002)&(np.sum(s.poses['mapped_hand_normals'][i]*s.poses['mapped_object_normals'][i],axis=-1)<=-np.cos(np.deg2rad(30)))
    with mujoco.Renderer(model,height=height,width=width) as renderer:
        renderer.update_scene(data);scene=renderer.scene
        for camera in scene.camera:camera.pos[:]=eye;camera.forward[:]=forward;camera.up[:]=up
        for j,(a,b) in enumerate(zip(hp,op)):
            color=np.array([.1,.9,.35,1]) if good[j] else np.array([1,.48,.1,1])
            for point,rgba,size in ((a,color,.0028),(b,np.array([.15,.5,1,1]),.0018)):
                g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([size,0,0]),point,np.eye(3).reshape(-1),rgba);scene.ngeom+=1
            g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_LINE,np.zeros(3),a,np.eye(3).reshape(-1),color);mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_LINE,2,a,b);scene.ngeom+=1
        image=Image.fromarray(renderer.render().copy())
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    draw=ImageDraw.Draw(image);focal=height/(2*np.tan(np.deg2rad(model.vis.global_.fovy)*.5))
    for j,point in enumerate(hp):
        rel=point-eye;depth=rel@forward
        if depth>0:draw.text((width/2+focal*(rel@right)/depth+5,height/2-focal*(rel@up)/depth-10),str(j+1),font=small,fill='white',stroke_width=2,stroke_fill='black')
    return image

def main():
    index=SolvedIndex(DEFAULT_RESULTS).refresh();cache=ClosureCache()
    out=DEFAULT_CACHE/'previews';out.mkdir(parents=True,exist_ok=True)
    results=[]
    for quality in ('几何达标','几何未达标'):
        refs,_=index.page(quality=quality,limit=1);r=snapshot(refs[0]['path']).row(refs[0]['index']);report=cache.evaluate(r)
        image=Image.new('RGB',(1560,810),'#15212e');draw=ImageDraw.Draw(image)
        title=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
        draw.text((18,12),'SOLVED MPPI / '+r['key'],font=title,fill='white')
        d=r['metric']['geometry_diagnostics']
        draw.text((18,44),f"Geometry: {'PASS' if r['metric']['geometry_accepted'] else 'FAIL'} | coverage {d['mapped_coverage']:.0%} | penetration {d['max_hand_object_penetration_m']*1000:.3f} mm | cost {r['metric']['final_cost']:.4f}",font=small,fill='#bdd3e8')
        for j,view in enumerate(('palm','oblique')):image.paste(render(r,view),(j*780,80))
        text=f"Projected closure: {'PASS' if report['projected']['certified'] else 'UNPROVEN'} | near-surface closure: {'PASS' if report['near_surface']['certified'] else 'UNPROVEN'} | dynamics: not evaluated"
        draw.text((18,770),text,font=small,fill='white')
        name=r['snapshot'].path.name+'_'+str(r['index'])+'.jpg';image.save(out/name,quality=92)
        write(out/(name+'.json'),dict(pose_key=r['key'],entry=r['entry'],metrics=r['metric'],closure=report))
        results.append(dict(image=name,key=r['key'],geometry_accepted=r['metric']['geometry_accepted'],closure=report))
        print('RENDERED',name,flush=True)
    write(out/'index.json',results)
    public=ROOT/'output/dexonomy_shadow_mano_mapping_20261103'
    link=public/'solved_previews'
    if not link.exists():link.symlink_to(out,target_is_directory=True)
    page='<html><meta charset="utf-8"><title>已求解 MANO 抓取</title><style>body{background:#15212e;color:white;font:17px system-ui;padding:20px}a{color:#89c8ff}img{max-width:100%;margin:20px 0}</style><h1>已求解 MANO 抓取</h1><p><a href="http://192.168.0.32:8101/">打开交互 Viser 页面：完整已求解缓存、筛选、接触点、力闭合</a></p><p>以下图片来自实际 MPPI 最终结果。绿色为贴合槽、橙色为未贴合槽、蓝色为物体投影。示例含几何达标和未达标姿态；投影闭合不代表已满足几何贴合。未执行动力学稳定性测试。</p>'
    for item in results:page+='<h2>'+item['key']+'</h2><img src="solved_previews/'+item['image']+'">'
    (public/'solved.html').write_text(page+'</html>')
if __name__=='__main__':main()
