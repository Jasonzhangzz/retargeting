#!/usr/bin/env python3
"""Uniform random committed pose pairs; real meshes, shared object/cameras, no optimization."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('MUJOCO_EGL_DEVICE_ID','0');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import sys,argparse,json,xml.etree.ElementTree as ET,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,mujoco
from PIL import Image,ImageDraw,ImageFont
from retarget.viewer.mano_results import ManoSnapshot,retry
from retarget.env.grasp.hand import hand_tree,rigid_mesh_arrays,HAND_BODIES
from scripts.retarget_dexonomy_mano import read,write,sha
_read,_sha=read,sha
read=lambda path:retry(_read,path)
sha=lambda path:retry(_sha,path)
W,H=640,510
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def pool(root):
    groups={}
    if (root/'reused_shards.json').exists():
        for r in read(root/'reused_shards.json'):groups[r['key']]=dict(path=r['path'],samples=r['samples'])
    for line in (root/'server.log').read_text().splitlines():
        try:r=json.loads(line)
        except ValueError:continue
        if isinstance(r,dict) and all(k in r for k in ('worker','source','samples')):
            typ,part=r['source'].split('/');groups[r['source']]=dict(path=str(root/typ/'shards'/f'{int(part):07d}'),samples=r['samples'])
    return [(k,v) for k,v in sorted(groups.items()) if v['samples']>0]

def numbers(a):return ' '.join(format(float(x),'.9g') for x in np.asarray(a).reshape(-1))

def scene_image(meshes,obj,eye,center,up,hp,op):
    xml=f'<mujoco><visual><global offwidth="{W}" offheight="{H}" fovy="42"/><rgba haze=".045 .065 .09 1"/></visual><asset>'
    for i,(v,f) in enumerate([*meshes,obj]):xml+=f'<mesh name="g{i}" vertex="{numbers(v)}" face="{numbers(f)}"/>'
    xml+='</asset><worldbody><light pos="0 0 2" dir="0 0 -1" ambient=".55 .55 .55" diffuse=".65 .65 .65"/>'
    for i in range(len(meshes)+1):
        rgba='.35 .65 .93 .42' if i==len(meshes) else '.86 .69 .54 1'
        xml+=f'<geom type="mesh" mesh="g{i}" contype="0" conaffinity="0" rgba="{rgba}"/>'
    xml+='</worldbody></mujoco>';m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_kinematics(m,d);mujoco.mj_camlight(m,d)
    m.vis.headlight.ambient[:]=.65;m.vis.headlight.diffuse[:]=.55
    forward=center-eye;forward/=np.linalg.norm(forward);right=np.cross(forward,up);right/=np.linalg.norm(right);vertical=np.cross(right,forward)
    with mujoco.Renderer(m,height=H,width=W) as renderer:
        renderer.update_scene(d);s=renderer.scene
        for cam in s.camera:cam.pos[:]=eye;cam.forward[:]=forward;cam.up[:]=vertical
        for a,b in zip(hp,op):
            for p,c,size in [(a,[1,.65,.08,1],.002),(b,[.1,1,.72,1],.0017)]:
                g=s.geoms[s.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([size,0,0]),p,np.eye(3).reshape(-1),np.array(c));s.ngeom+=1
            g=s.geoms[s.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_LINE,np.zeros(3),a,np.eye(3).reshape(-1),np.array([1,.65,.08,.8]));mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_LINE,1,a,b);s.ngeom+=1
        return Image.fromarray(renderer.render().copy())

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT/'results/dexonomy_wuji2_current');p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=6);p.add_argument('--seed',type=int,default=20261104);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    groups=pool(a.source.resolve());counts=np.array([v['samples'] for _,v in groups],np.int64);ends=counts.cumsum();rng=np.random.default_rng(a.seed);draws=rng.choice(int(ends[-1]),size=a.count,replace=False)
    model=mujoco.MjModel.from_xml_string(ET.tostring(hand_tree('wuji2'),encoding='unicode'));data=mujoco.MjData(model)
    local=[(model.body(name).id,*rigid_mesh_arrays(model,model.body(name).id)) for name in HAND_BODIES]
    title=ImageFont.truetype(FONT,23);font=ImageFont.truetype(FONT,17);small=ImageFont.truetype(FONT,15);items=[]
    for number,draw in enumerate(draws,1):
        gi=int(np.searchsorted(ends,draw,side='right'));key,group=groups[gi];row=int(draw-(ends[gi-1] if gi else 0));folder=Path(group['path'])
        verification=read(folder/'verification.json');assert verification['success'] and all(verification['checks'].values());assert sha(folder/'poses.npz')==read(folder/'checksums.json')['poses.npz']
        manifest=read(folder/'manifest.json');source=Path(manifest['source_mano_shard']);mano=ManoSnapshot(source);entry=manifest['cases'][row];mi=int(entry.get('source_mano_case',entry['cached_case']))
        with np.load(folder/'poses.npz',allow_pickle=False) as z:wuji={k:z[k][row].copy() for k in z.files}
        np.testing.assert_array_equal(wuji['object_pose_world'],mano.poses['object_pose_world'][mi]);np.testing.assert_allclose(wuji['target_object_contact_points'],mano.poses['mapped_object_points'][mi],atol=1e-7,rtol=0)
        mv,mf=mano.mano_mesh(mi);ov,of=mano.object_mesh(mi);data.qpos[:26]=wuji['qpos'];mujoco.mj_kinematics(model,data)
        meshes=[(v@data.xmat[bid].reshape(3,3).T+data.xpos[bid],f) for bid,v,f in local]
        allv=np.concatenate([mv,ov,*[v for v,_ in meshes]]);center=(allv.min(0)+allv.max(0))/2;radius=np.linalg.norm(allv-center,axis=1).max();distance=max(.25,radius/np.sin(np.deg2rad(21))*1.13)
        j=mano.poses['mano_joints'][mi];up=j[9]-j[0];up/=np.linalg.norm(up);x=j[5]-j[17];normal=np.cross(x,up);normal/=np.linalg.norm(normal)
        if np.dot(normal,ov.mean(0)-j[[0,5,9,13,17]].mean(0))<0:normal=-normal
        side=np.cross(up,normal);side/=np.linalg.norm(side);directions=[normal,(normal*.65+side*.7+up*.2)/np.linalg.norm(normal*.65+side*.7+up*.2)]
        mm=mano.summary(mi);wm=bool(wuji['geometry_accepted']);mg=bool(mm['geometry_accepted']);gap=float(wuji['max_contact_gap_m'])*1000;penetration=float(wuji['max_penetration_m'])*1000
        canvas=Image.new('RGB',(2*W,2*H+155),'#101b28');d=ImageDraw.Draw(canvas)
        d.text((18,12),f'Random pair {number:02d}  |  {key}  |  row {row}',font=title,fill='white')
        d.text((18,47),f'MANO geometry: {"PASS" if mg else "FAIL"}    |    WUJI2 geometry: {"PASS" if wm else "FAIL"}',font=font,fill='#c1d5e8')
        d.text((18,72),f'WUJI2 max gap {gap:.2f} mm; sampled penetration {penetration:.2f} mm. Same object, world frame and cameras.',font=small,fill='#c1d5e8')
        for view,direction in enumerate(directions):
            eye=center+direction*distance
            for col,(meshes_hand,hp,op,label) in enumerate([([(mv,mf)],mano.poses['mapped_hand_points'][mi],mano.poses['mapped_object_points'][mi],'MANO'),(meshes,wuji['hand_contact_points'],wuji['target_object_contact_points'],'WUJI2')]):
                im=scene_image(meshes_hand,(ov,of),eye,center,up,hp,op);draw_im=ImageDraw.Draw(im);draw_im.text((18,12),label+(' / contact view' if view==0 else ' / oblique view'),font=title,fill='white',stroke_width=2,stroke_fill='#101b28');canvas.paste(im,(col*W,100+view*H))
        d.text((18,2*H+118),'Orange: final hand anchors. Green: matched object targets. Lines: residual gaps. No quality filtering; no optimization.',font=small,fill='#c1d5e8')
        name=f'pair_{number:02d}.jpg';canvas.save(a.output/name,quality=94)
        item=dict(number=number,image=name,key=key,wuji_shard=str(folder),wuji_row=row,mano_shard=str(source),mano_row=mi,mano_geometry_accepted=mg,wuji_geometry_accepted=wm,wuji_max_gap_mm=gap,wuji_max_sampled_penetration_mm=penetration);items.append(item);print(json.dumps(item),flush=True)
    for page in range((len(items)+2)//3):
        selected=items[page*3:page*3+3];sheet=Image.new('RGB',(2*W,len(selected)*(H+100)+55),'#101b28');d=ImageDraw.Draw(sheet);d.text((18,12),'Random paired final grasps | MANO (left) / WUJI2 (right)',font=title,fill='white')
        for ri,item in enumerate(selected):
            im=Image.open(a.output/item['image']);sheet.paste(im.crop((0,0,2*W,H+100)),(0,55+ri*(H+100)))
        sheet.save(a.output/f'overview_{page+1}.jpg',quality=94)
    write(a.output/'selection.json',dict(seed=a.seed,population_rows=int(ends[-1]),sampling='Uniform without replacement over all committed WUJI2 pose rows represented in historical references and completion log; no quality filtering.',items=items))
    with zipfile.ZipFile(a.output/'comparisons.zip','w',zipfile.ZIP_DEFLATED) as z:
        for file in a.output.glob('*.jpg'):z.write(file,file.name)
        z.write(a.output/'selection.json','selection.json')
    print('DONE',str(a.output),flush=True)
if __name__=='__main__':main()
