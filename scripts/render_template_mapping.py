#!/usr/bin/env python3
"""Render the exported paired templates with real meshes and numbered contacts."""
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from retarget.env.grasp.hand import hand_tree
from scripts.visualize_template_mapping import canonical

COLORS = [(168,75,206),(0,119,200),(220,139,9),(0,160,128),(219,92,97),(119,92,53)]


def main(folder, mesh_folder):
    raw = json.loads((folder/'data.json').read_text())
    tree = ET.parse(ROOT/'data/grasp/templates/assets/hand/shadow/right.xml').getroot()
    tree.find('compiler').set('meshdir', str(mesh_folder.resolve()))
    models = [mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding='unicode')),
              mujoco.MjModel.from_xml_string(ET.tostring(hand_tree('wuji2'), encoding='unicode'))]
    width, height = 640, 580
    for index, model in enumerate(models):
        model.vis.global_.offwidth=width
        model.vis.global_.offheight=height
        model.vis.headlight.ambient[:]=.6
        model.vis.headlight.diffuse[:]=.6
        # Neutral inspection colors keep the black factory palm visible on
        # the offscreen background without changing the mesh geometry.
        shade=[.47,.53,.54,1.] if index==0 else [.68,.73,.70,1.]
        model.mat_rgba[:]=shade
        model.geom_rgba[:]=shade
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    renderers = [mujoco.Renderer(m,height=height,width=width) for m in models]
    images=[]
    try:
        for case in raw['cases']:
            # Use identical camera distance and target for both hands.
            poses=case['poses']['grasp']
            landmarks=np.concatenate([poses[h]['joints'] for h in ('shadow','wuji')])
            low, high=landmarks.min(0),landmarks.max(0)
            center=(low+high)/2
            distance=max(float(np.max(high-low))*1.4,.27)
            pair=Image.new('RGB',(width*2,height+90),'#f0f2ed')
            draw=ImageDraw.Draw(pair)
            draw.text((20,12),case['name'],font=font,fill='#152a30')
            for side,(model,renderer) in enumerate(zip(models,renderers)):
                hand=('shadow','wuji')[side]
                data=mujoco.MjData(model)
                data.qpos[:]=case['shadow_qpos'] if side==0 else np.r_[np.zeros(6),case['finger_qpos']]
                mujoco.mj_forward(model,data)
                origin,basis=canonical(model,data,side==0)
                option=mujoco.MjvOption()
                option.geomgroup[:]=0
                option.geomgroup[2 if side==0 else 1]=1
                option.sitegroup[:]=0
                renderer.update_scene(data,scene_option=option)
                scene=renderer.scene
                position=origin+basis@(center+np.array([0,distance,0]))
                for camera in scene.camera:
                    camera.pos[:]=position
                    camera.forward[:]=basis@np.array([0,-1.,0])
                    camera.up[:]=basis@np.array([0,0,1.])
                points=poses[hand]['contacts' if side==0 else 'effective']
                for i,(point,normal) in enumerate(zip(points['p'],points['n'])):
                    record=case['mapping']['effective'][i]
                    f=int(record['wuji_body'][len('right_finger')])-1 if record['wuji_body'].startswith('right_finger') else 5
                    color=np.r_[np.array(COLORS[f])/255.,1.]
                    world=origin+basis@point
                    geom=scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.0027,0,0]),world,np.eye(3).flatten(),color)
                    scene.ngeom+=1
                    geom=scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),world,np.eye(3).flatten(),color)
                    mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,.0009,world,world+basis@np.array(normal)*.012)
                    scene.ngeom+=1
                image=Image.fromarray(renderer.render())
                # Project the exact exported coordinates to the same camera.
                painter=ImageDraw.Draw(image)
                camera=scene.camera[0]
                focal=height/(2*np.tan(np.deg2rad(model.vis.global_.fovy)/2))
                for i,point in enumerate(points['p']):
                    relative=np.array(point)-center
                    depth=distance-relative[1]
                    # camera right = forward x up = -X in the shared frame.
                    x=width/2-focal*relative[0]/depth
                    y=height/2-focal*relative[2]/depth
                    painter.text((x+7,y-16),str(i+1),font=small,fill='white',stroke_width=2,stroke_fill='#152a30')
                pair.paste(image,(side*width,80))
                draw.text((20+side*width,49),'Shadow / source annotation' if side==0 else 'Wuji2 / current ADMM template',font=small,fill='#566b6e')
            path=folder/'previews'/f'{case["name"]}.png'
            path.parent.mkdir(exist_ok=True)
            pair.save(path)
            images.append((case['name'],pair))
        chosen=[im for name,im in images if name.partition('_')[0] in ('6','9','11','24')]
        sheet=Image.new('RGB',(1280,670*len(chosen)),'white')
        for i,im in enumerate(chosen):sheet.paste(im,(0,670*i))
        sheet.save(folder/'preview.png')
        # A compact gallery allows checking all types without a 3-D browser.
        thumb_w,thumb_h=640,335
        sheet=Image.new('RGB',(thumb_w*3,thumb_h*((len(images)+2)//3)),'#f0f2ed')
        for i,(_,im) in enumerate(images):
            sheet.paste(im.resize((thumb_w,thumb_h)),((i%3)*thumb_w,(i//3)*thumb_h))
        sheet.save(folder/'all_templates.jpg',quality=90)
        print(f'Rendered {len(images)} paired template previews')
    finally:
        for renderer in renderers:renderer.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,default=ROOT/'output/template_mapping')
    parser.add_argument('--shadow-meshes',type=Path,default=ROOT/'.cache/shadow_hand_visual')
    args=parser.parse_args()
    main(args.folder,args.shadow_meshes)
