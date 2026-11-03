#!/usr/bin/env python3
"""Render numbered fixed Shadow -> MANO contact mappings for review."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,mujoco
from PIL import Image,ImageDraw,ImageFont
from scripts.inspect_shadow_mano_mapping import COLORS,palm

def human_model(vertices,faces,width,height):
    def numbers(a):return ' '.join(map(str,np.asarray(a).reshape(-1).tolist()))
    xml='<mujoco><visual><global offwidth="'+str(width)+'" offheight="'+str(height)+'"/></visual><asset><mesh name="mano" vertex="'+numbers(vertices)+'" face="'+numbers(faces)+'"/></asset><worldbody><geom type="mesh" mesh="mano" contype="0" conaffinity="0" rgba=".78 .58 .43 1"/></worldbody></mujoco>'
    return mujoco.MjModel.from_xml_string(xml)
def render_side(model,data,points,normals,records,center,eye,up,width,height):
    model.vis.global_.offwidth=width;model.vis.global_.offheight=height
    model.vis.headlight.ambient[:]=.65;model.vis.headlight.diffuse[:]=.65;model.vis.headlight.active=1
    option=mujoco.MjvOption();option.geomgroup[3]=False;option.sitegroup[:]=False
    with mujoco.Renderer(model,height=height,width=width) as renderer:
        renderer.update_scene(data,scene_option=option)
        scene=renderer.scene
        forward=center-eye;forward/=np.linalg.norm(forward)
        right=np.cross(forward,up);right/=np.linalg.norm(right);up=np.cross(right,forward)
        for camera in scene.camera:camera.pos[:]=eye;camera.forward[:]=forward;camera.up[:]=up
        for i,(p,n,rec) in enumerate(zip(points,normals,records)):
            color=np.r_[COLORS[rec['finger_index']]/255.,1.]
            geom=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.0022,0,0]),p,np.eye(3).reshape(-1),color);scene.ngeom+=1
            geom=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),p,np.eye(3).reshape(-1),color)
            mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,.00065,p,p+np.asarray(n)*.011);scene.ngeom+=1
        image=Image.fromarray(renderer.render().copy())
    draw=ImageDraw.Draw(image);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    focal=height/(2*np.tan(np.deg2rad(model.vis.global_.fovy)*.5))
    for i,p in enumerate(points):
        rel=np.array(p)-eye;depth=rel@forward
        if depth<=0:continue
        x=width/2+focal*(rel@right)/depth;y=height/2-focal*(rel@up)/depth
        draw.text((x+5,y-12),str(i+1),font=font,fill='white',stroke_width=2,stroke_fill='#102030')
    return image
def main(folder):
    raw=json.loads((folder/'data.json').read_text());geometry=np.load(folder/'geometry.npz',allow_pickle=False)
    width,height=640,620;shadow=mujoco.MjModel.from_xml_path(str(ROOT/'data/dexonomy/shadow_assets/right.xml'))
    for i in range(shadow.ngeom):
        if shadow.geom_contype[i]==0:shadow.geom_matid[i]=-1;shadow.geom_rgba[i]=[.57,.66,.73,1]
    chosen=[r for r in raw['cases'] if r['mode']=='template' and r['name'].split('_')[0] in ('1','6','9','11','14','24')]
    previews=folder/'previews';previews.mkdir(exist_ok=True)
    results=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    for row in chosen:
        data=mujoco.MjData(shadow);data.qpos[:]=row['source_qpos'][7:];mujoco.mj_kinematics(shadow,data);mujoco.mj_camlight(shadow,data)
        vertices=np.array(row['mano_vertices']);human=human_model(vertices,geometry['mano_faces'],width,height);hd=mujoco.MjData(human);mujoco.mj_kinematics(human,hd);mujoco.mj_camlight(human,hd)
        sj=np.array(row['shadow_joints']);basis=palm(sj);combined=np.r_[sj,vertices]
        center=(combined.min(0)+combined.max(0))*.5;distance=max(.3,float(np.linalg.norm(combined.max(0)-combined.min(0)))*1.4)
        source_normals=np.array(row['shadow_normals']);selection=[i for i,r in enumerate(row['contact_mapping']) if r['shadow_body']=='rh_palm']
        normal=source_normals[selection[0]] if selection else basis[:,2]*np.sign(np.mean(source_normals,0)@basis[:,2])
        for view in ('palm','oblique'):
            eye=center+normal*distance+(basis[:,0]*distance*.6+basis[:,1]*distance*.15 if view=='oblique' else basis[:,1]*distance*.05)
            pair=Image.new('RGB',(width*2,height+100),'#15212e');draw=ImageDraw.Draw(pair)
            draw.text((16,10),row['name']+' / '+view,font=font,fill='white')
            draw.text((16,43),'Shadow annotation: numbered surface anchors',font=small,fill='#b8d6ed')
            draw.text((width+16,43),'MANO: same IDs / anatomical triangle mapping',font=small,fill='#e4c5a6')
            for side,(m,d,key,nkey) in enumerate([(shadow,data,'shadow_contacts','shadow_normals'),(human,hd,'mano_contacts','mano_normals')]):
                pair.paste(render_side(m,d,np.array(row[key]),np.array(row[nkey]),row['contact_mapping'],center,eye,basis[:,1],width,height),(width*side,72))
            draw.text((16,height+79),'Template annotations, not per-sample contact truth. MANO bone IK only; object MPPI not applied.',font=small,fill='#d3deea')
            p=previews/(row['name']+'_'+view+'.png');pair.save(p);results.append(p.name);print('RENDERED',p.name,flush=True)
    thumbs=[Image.open(previews/f).resize((640,360)) for f in results if f.endswith('_palm.png')]
    sheet=Image.new('RGB',(1280,360*((len(thumbs)+1)//2)),'#15212e')
    for i,im in enumerate(thumbs):sheet.paste(im,((i%2)*640,(i//2)*360))
    sheet.save(folder/'mapping_overview.jpg',quality=94)
    index='<html><meta charset="utf-8"><title>Shadow → MANO 接触映射</title><style>body{background:#14202d;color:white;font:16px system-ui;padding:20px}img{max-width:100%}a{color:#83c8ff}</style><h1>Shadow → MANO 接触映射检查</h1><p>同编号、同颜色对应同一模板标注点。左：Shadow；右：MANO。短箭头为表面法向。MANO 是骨架初始化，尚未执行物体 MPPI。</p><p>这里展示模板标注映射，不是原始数据集逐样本接触真值。</p><p><a href="http://192.168.0.32:8099/">交互 3D：31 个抓型、模板／真实缓存姿态、逐点筛选</a></p><img src="mapping_overview.jpg">'
    for f in results:index+='<h3>'+f+'</h3><a href="previews/'+f+'"><img loading="lazy" src="previews/'+f+'"></a>'
    (folder/'index.html').write_text(index+'</html>')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);main(p.parse_args().folder)
