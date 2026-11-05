#!/usr/bin/env python3
"""Inspect fixed anatomical Shadow annotation -> MANO triangle correspondences."""
from __future__ import annotations
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import argparse,json,time,sys,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch,mujoco
from scipy.spatial.transform import Rotation
from retarget.env.grasp.mano_solver import ManoDecoder,ShadowSkeleton,Settings,lm_ik,initialize,vertex_normals,sha
from retarget.task.grasp.dexonomy import load_template
from retarget.env.geometry import closest_triangles
from scripts.mano_campaign_inputs import GeometryLoader,write

FINGERS=('thumb','index','middle','ring','pinky','palm')
COLORS=np.array([[209,85,189],[47,151,245],[236,167,32],[40,186,135],[240,91,92],[160,133,89]],np.uint8)
CHAINS=((13,14,15),(1,2,3),(4,5,6),(10,11,12),(7,8,9))
TIPS=(744,320,443,554,671)
PREFIX=('th','ff','mf','rf','lf')

def palm(joints):
    x=joints[5]-joints[17];x/=np.linalg.norm(x)
    y=joints[9]-joints[0];y-=x*np.dot(x,y);y/=np.linalg.norm(y)
    return np.stack((x,y,np.cross(x,y)),1)
def bone_frame(start,end,palm_x):
    z=end-start;z/=np.linalg.norm(z);x=palm_x-z*np.dot(palm_x,z)
    if np.linalg.norm(x)<1e-8:x=np.cross(z,[0,0,1])
    x/=np.linalg.norm(x);return np.stack((x,np.cross(z,x),z),1)
def barycentric(point,triangle):
    a,b,c=triangle;v=b-a;w=c-a;r=point-a
    v2=np.dot(v,v);w2=np.dot(w,w);vw=np.dot(v,w)
    d=v2*w2-vw*vw
    beta=(w2*np.dot(r,v)-vw*np.dot(r,w))/d;gamma=(v2*np.dot(r,w)-vw*np.dot(r,v))/d
    return np.array([1-beta-gamma,beta,gamma])

def map_contacts(model,contacts,mano):
    """Neutral, object-independent mapping restricted to the same anatomical segment."""
    native=mujoco.MjData(model);native.qpos[:]=0;mujoco.mj_kinematics(model,native)
    b=mano.base
    zero=lambda *s:torch.zeros(s,device=mano.device)
    with torch.no_grad():mv,mj,_=b(zero(10),zero(),zero(5),zero(5),zero(1,15,3))
    mv=mv[0].cpu().numpy();mj=mj[0].cpu().numpy();weights=b.weights.cpu().numpy();faces=b.faces
    sroots=np.array([native.xpos[model.body('rh_'+p+'proximal').id] for p in PREFIX])
    sj=np.zeros((21,3));sj[0]=native.xpos[model.body('rh_palm').id]
    for f,p in enumerate(PREFIX):
        ids=[model.body('rh_'+p+x).id for x in ('proximal','middle','distal')]
        sj[1+4*f:4+4*f]=native.xpos[ids]
        sj[4+4*f]=native.xpos[ids[-1]]+native.xmat[ids[-1]].reshape(3,3)@np.array([0,0,.027])
    mj21=np.zeros_like(sj);mj21[0]=mj[0]
    for f,chain in enumerate(CHAINS):mj21[1+4*f:4+4*f]=mj[list(chain)];mj21[4+4*f]=mv[TIPS[f]]
    sp=palm(sj);mp=palm(mj21);records=[]
    for index,(body,local) in enumerate(contacts):
        bid=model.body(body).id;sr=native.xmat[bid].reshape(3,3)
        sw=native.xpos[bid]+sr@local[:3];sn=sr@local[3:];sn/=np.linalg.norm(sn)
        if body=='rh_palm' or 'metacarpal' in body:
            f=5;j=0;ratio=np.linalg.norm(mj21[5]-mj21[17])/np.linalg.norm(sj[5]-sj[17])
            desired=mj[0]+mp@(sp.T@(sw-sj[0])*ratio);normal=mp@sp.T@sn
            allowed=weights[:,0]>.35
        else:
            f=next((f for f,p in enumerate(PREFIX) if body.startswith('rh_'+p)),None)
            if f is None:raise ValueError('Unsupported annotated body '+body)
            segment=next((k for k,s in enumerate(('proximal','middle','distal')) if body.endswith(s)),None)
            if segment is None:raise ValueError('No anatomical segment for '+body)
            j=CHAINS[f][segment];start=sj[1+4*f+segment];end=sj[2+4*f+segment]
            mstart=mj21[1+4*f+segment];mend=mj21[2+4*f+segment]
            sb=bone_frame(start,end,sp[:,0]);mb=bone_frame(mstart,mend,mp[:,0])
            length=np.linalg.norm(mend-mstart)/np.linalg.norm(end-start)
            desired=mstart+mb@(sb.T@(sw-start)*length);normal=mb@sb.T@sn
            # Hard anatomical restriction prevents nearest-surface jumps to another finger.
            allowed=(weights[:,j]>.25)&(weights[:,list(CHAINS[f])].sum(-1)>.65)
        face_ids=np.flatnonzero(allowed[faces].sum(-1)>=2)
        if len(face_ids)==0:raise ValueError('No MANO surface patch for '+body)
        triangles=mv[faces[face_ids]];points=closest_triangles(desired[None],triangles)[0]
        normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        normals/=np.maximum(np.linalg.norm(normals,axis=-1,keepdims=True),1e-12)
        score=((points-desired)**2).sum(-1)+.006**2*(1-normals@normal)
        chosen=int(np.argmin(score));face=int(face_ids[chosen]);point=points[chosen];bn=barycentric(point,mv[faces[face]])
        error=float(np.linalg.norm(point-desired));angle=float(np.rad2deg(np.arccos(np.clip(normals[chosen]@normal,-1,1))))
        records.append(dict(id=index+1,shadow_body=body,shadow_local_point=local[:3].tolist(),shadow_local_normal=local[3:].tolist(),
            finger=FINGERS[f],finger_index=f,mano_joint_id=j,mano_face_id=face,mano_vertex_ids=faces[face].tolist(),mano_barycentric=bn.tolist(),
            neutral_desired_point=desired.tolist(),neutral_mapped_point=point.tolist(),neutral_mapping_error_mm=error*1000,
            neutral_normal_error_deg=angle,anatomical_patch_restricted=True,
            needs_review=bool(error>.01 or angle>60)))
    return records

def posed_contacts(model,data,contacts,vertices,mapping,faces):
    source=[];source_n=[];target=[];target_n=[]
    for (body,local),record in zip(contacts,mapping):
        bid=model.body(body).id;r=data.xmat[bid].reshape(3,3)
        source.append(data.xpos[bid]+r@local[:3]);source_n.append(r@local[3:])
        triangle=vertices[faces[record['mano_face_id']]]
        target.append(np.asarray(record['mano_barycentric'])@triangle)
        normal=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]);normal/=np.linalg.norm(normal);target_n.append(normal)
    return np.asarray(source),np.asarray(source_n),np.asarray(target),np.asarray(target_n)

def project(points,vertices,faces):
    triangles=vertices[faces];cp=closest_triangles(points,triangles)
    d=((cp-points[:,None])**2).sum(-1);best=d.argmin(-1)
    return cp[np.arange(len(points)),best],np.sqrt(d[np.arange(len(points)),best])

def prepare(folder,device):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    decoder=ManoDecoder(ROOT/'data/grasp/assets/mano/MANO_RIGHT.pkl',device)
    skeleton=ShadowSkeleton(ROOT/'data/dexonomy/shadow_assets/right.xml')
    names=sorted([p.stem for p in (ROOT/'data/grasp/templates/assets/hand/shadow/raw_anno').glob('*.yaml') if p.stem.split('_')[0].isdigit()],key=lambda x:int(x.split('_')[0]))
    preview=ROOT/'results/dexonomy_shadow_grasp_preview_20261103'
    names=[n for n in names if (preview/n).exists()]
    rows=[];inputs=[];models={}
    for name in names:
        sm,sd,contacts,path=load_template(ROOT/'data/grasp/templates',name)
        actual=next((preview/name).glob('shards/*/manifest.json'));manifest=json.loads(actual.read_text());e=manifest['cases'][0]
        with np.load(actual.parent/'poses.npz',allow_pickle=False) as a:
            q=a['grasp_qpos'][0].copy();object_pose=a['object_pose_world'][0].copy();count=int(a['contact_counts'][0])
        canonical=np.r_[np.zeros(3),1.,np.zeros(3),sd.qpos]
        for mode,pose in [('template',canonical),('cached',q)]:
            rows.append(dict(name=name,mode=mode,source_qpos=pose.tolist(),annotation=str(path),entry=e if mode=='cached' else None,
                source_actual_geometric_contacts=count if mode=='cached' else None,object_pose=np.asarray(object_pose).reshape(-1).tolist() if mode=='cached' else None))
            inputs.append(pose)
        models[name]=(sm,contacts)
    targets=skeleton(np.asarray(inputs))
    target=torch.tensor(targets,dtype=torch.float32,device=device)
    best,_,diagnostics=lm_ik(decoder,initialize(decoder,targets),target,Settings())
    best_cost=diagnostics['final_cost']
    with torch.no_grad():vertices,joints,_=decoder(best,full=True)
    vertices=vertices.cpu().numpy();joints=joints.cpu().numpy()
    mappings={name:map_contacts(*models[name],decoder) for name in names}
    loader=GeometryLoader(ROOT/'data/dexonomy');geometry={};case_geometry={}
    full=skeleton.model;native=skeleton.data
    for i,row in enumerate(rows):
        q=np.asarray(row['source_qpos']);native.qpos[:]=q[7:];mujoco.mj_kinematics(full,native)
        r=Rotation.from_quat(q[3:7],scalar_first=True).as_matrix()
        source_points=[];source_normals=[]
        for rec in mappings[row['name']]:
            body=full.body(rec['shadow_body']).id;br=native.xmat[body].reshape(3,3)
            source_points.append((native.xpos[body]+br@rec['shadow_local_point'])@r.T+q[:3])
            source_normals.append((br@rec['shadow_local_normal'])@r.T)
        mv=vertices[i];mp=[];mn=[]
        for rec in mappings[row['name']]:
            tri=mv[decoder.base.faces[rec['mano_face_id']]];mp.append(np.asarray(rec['mano_barycentric'])@tri)
            normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);mn.append(normal/np.linalg.norm(normal))
        row.update(variables=best[i].detach().cpu().tolist(),mano_vertices=mv.tolist(),mano_joints=joints[i].tolist(),
            shadow_joints=targets[i].tolist(),shadow_contacts=np.asarray(source_points).tolist(),shadow_normals=np.asarray(source_normals).tolist(),
            mano_contacts=np.asarray(mp).tolist(),mano_normals=np.asarray(mn).tolist(),bone_fit_cost=float(best_cost[i]),contact_mapping=mappings[row['name']],
            contact_kind='original grasp-type template annotation replayed at source pose; not dataset-provided contact truth',
            mano_pose_kind='bone IK initialization for mapping inspection; object CEM not yet applied')
        if row['mode']=='cached':
            key,g=loader.get(row['entry']);gid=hashlib.sha256(str(key).encode()).hexdigest()[:16]
            if gid not in geometry:
                geometry[gid+'_vertices']=g['vertices'];geometry[gid+'_faces']=g['faces']
            pose=np.array(row['object_pose']);rr=Rotation.from_quat(pose[3:],scalar_first=True).as_matrix();factor=row['entry']['scene_scale']
            world=g['vertices']*factor@rr.T+pose[:3]
            sp,sd=project(np.asarray(source_points),world,g['faces']);tp,td=project(np.asarray(mp),world,g['faces'])
            row.update(object_geometry=gid,shadow_object_projection=sp.tolist(),mano_object_projection=tp.tolist(),
                shadow_projection_distance_mm=(sd*1000).tolist(),mano_projection_distance_mm=(td*1000).tolist())
        row['case']=row['name']+'/'+row['mode']
        print('MAPPED',row['case'],len(mp),flush=True)
    np.savez_compressed(folder/'geometry.npz',mano_faces=decoder.base.faces,**geometry)
    write(folder/'data.json',dict(created_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),cases=rows,mappings=mappings,
        model_sha256=sha(decoder.base.path),source='Original Dexonomy per-type annotated surface points; semantic segment mapping, fixed neutral triangle barycentrics',
        dataset_contact_labels_available=False,mano_optimization='bone IK only; this inspection does not certify object contact or closure'))
    summary=dict(types=len(names),cases=len(rows),contacts=sum(len(v) for v in mappings.values()),
        review_slots=[dict(type=n,**r) for n,m in mappings.items() for r in m if r['needs_review']],
        all_anatomical_patch_restricted=all(r['anatomical_patch_restricted'] for m in mappings.values() for r in m))
    write(folder/'summary.json',summary);print('PREPARED',str(folder),summary['types'],flush=True)
    return folder

def shadow_meshes(q):
    global SHADOW_NATIVE
    if 'SHADOW_NATIVE' not in globals():SHADOW_NATIVE=ShadowSkeleton(ROOT/'data/dexonomy/shadow_assets/right.xml')
    skeleton=SHADOW_NATIVE;m,d=skeleton.model,skeleton.data;d.qpos[:]=q[7:];mujoco.mj_kinematics(m,d)
    r=Rotation.from_quat(q[3:7],scalar_first=True).as_matrix();items=[]
    for i in range(m.ngeom):
        if int(m.geom_type[i])!=int(mujoco.mjtGeom.mjGEOM_MESH) or m.geom_contype[i]!=0:continue
        mesh=int(m.geom_dataid[i]);a,n=m.mesh_vertadr[mesh],m.mesh_vertnum[mesh];b,k=m.mesh_faceadr[mesh],m.mesh_facenum[mesh]
        v=m.mesh_vert[a:a+n]@d.geom_xmat[i].reshape(3,3).T+d.geom_xpos[i];v=v@r.T+q[:3]
        items.append((v.copy(),m.mesh_face[b:b+k].copy()))
    return items

def serve(folder,host,port):
    import viser
    raw=json.loads((folder/'data.json').read_text());cases={(r['name'],r['mode']):r for r in raw['cases']}
    geometry=np.load(folder/'geometry.npz',allow_pickle=False);faces=geometry['mano_faces']
    server=viser.ViserServer(host=host,port=port,label='Shadow → MANO 接触映射检查',verbose=False)
    server.scene.set_up_direction('+z')
    names=list(raw['mappings'])
    selection=server.gui.add_dropdown('抓型',options=names,initial_value=names[0])
    mode=server.gui.add_dropdown('姿态',options=['模板姿态','缓存原始 grasp'],initial_value='模板姿态')
    finger=server.gui.add_dropdown('显示手指',options=['全部',*FINGERS],initial_value='全部')
    contact=server.gui.add_slider('单独查看接触编号（0=全部）',min=0,max=40,step=1,initial_value=0)
    show_links=server.gui.add_checkbox('显示 Shadow → MANO 对应连线',initial_value=True)
    show_normals=server.gui.add_checkbox('显示表面法向',initial_value=True)
    show_object=server.gui.add_checkbox('显示物体与投影',initial_value=True)
    notes=server.gui.add_markdown('')
    handles=[]
    def add(handle):handles.append(handle);return handle
    def update(_=None):
        for h in handles:h.remove()
        handles.clear()
        case=cases[(selection.value,'template' if mode.value=='模板姿态' else 'cached')]
        source=np.array(case['shadow_contacts']);target=np.array(case['mano_contacts']);mapping=case['contact_mapping']
        sj=np.array(case['shadow_joints']);mv=np.array(case['mano_vertices'])
        center=np.mean(np.r_[sj,mv],axis=0);offset=np.array([.16,0,0])
        left=-offset-center;right=offset-center
        for k,(v,f) in enumerate(shadow_meshes(np.array(case['source_qpos']))):
            add(server.scene.add_mesh_simple('/shadow/mesh'+str(k),vertices=v+left,faces=f,color=(155,175,195),opacity=.8,side='double'))
        add(server.scene.add_mesh_simple('/mano/mesh',vertices=mv+right,faces=faces,color=(227,184,143),opacity=.8,side='double'))
        add(server.scene.add_label('/shadow/title','Shadow / 原始抓型模板标注重放',position=tuple(sj.mean(0)+left+[0,0,.14])))
        add(server.scene.add_label('/mano/title','MANO / 固定三角面重心映射',position=tuple(mv.mean(0)+right+[0,0,.14])))
        lines=[];linecolors=[];normal_lines=[];normal_colors=[]
        distances=[]
        for i,rec in enumerate(mapping):
            if finger.value!='全部' and rec['finger']!=finger.value:continue
            if contact.value and rec['id']!=contact.value:continue
            color=COLORS[rec['finger_index']];a=source[i]+left;b=target[i]+right
            for side,p in [('shadow',a),('mano',b)]:
                add(server.scene.add_icosphere('/'+side+'/point'+str(i),radius=.0022,color=tuple(int(c) for c in color),position=tuple(p)))
                add(server.scene.add_label('/'+side+'/label'+str(i),'#'+str(i+1)+' '+rec['finger'],position=tuple(p+[0,0,.004])))
            if show_links.value:lines.append([a,b]);linecolors.append([color,color])
            if show_normals.value:
                normal_lines.extend([[a,a+np.array(case['shadow_normals'][i])*.012],[b,b+np.array(case['mano_normals'][i])*.012]])
                normal_colors.extend([[color,color],[color,color]])
            if case['mode']=='cached' and show_object.value:
                sp=np.array(case['shadow_object_projection'][i])+left;mp=np.array(case['mano_object_projection'][i])+right
                lines.extend([[a,sp],[b,mp]]);linecolors.extend([[[250,220,70]]*2]*2)
            suffix=''
            if case['mode']=='cached':suffix=f" · 物体间隙 Shadow {case['shadow_projection_distance_mm'][i]:.1f} / MANO {case['mano_projection_distance_mm'][i]:.1f} mm"
            distances.append(f"#{i+1} {rec['shadow_body']} → MANO face {rec['mano_face_id']} · 中性映射偏差 {rec['neutral_mapping_error_mm']:.1f} mm · 法向偏差 {rec['neutral_normal_error_deg']:.0f}°"+suffix)
        if lines:add(server.scene.add_line_segments('/links',points=np.asarray(lines),colors=np.asarray(linecolors,np.uint8),line_width=1.3))
        if normal_lines:add(server.scene.add_line_segments('/normals',points=np.asarray(normal_lines),colors=np.asarray(normal_colors,np.uint8),line_width=2.5))
        if case['mode']=='cached' and show_object.value:
            gid=case['object_geometry'];pose=np.array(case['object_pose']);r=Rotation.from_quat(pose[3:],scalar_first=True).as_matrix()
            ov=geometry[gid+'_vertices']*case['entry']['scene_scale']@r.T+pose[:3];of=geometry[gid+'_faces']
            for side,delta in [('shadow',left),('mano',right)]:add(server.scene.add_mesh_simple('/'+side+'/object',vertices=ov+delta,faces=of,color=(90,130,180),opacity=.4,side='double'))
        notes.content='**相同编号、相同颜色是一对固定映射；法向短线用于检查正反面。**\n\n'+('原始 grasp 实际几何接触数：'+str(case['source_actual_geometric_contacts'])+'。\n\n' if case['mode']=='cached' else '')+'这些源点来自抓型模板标注，不是数据集逐样本接触真值。MANO 当前是骨架初始化，尚未执行物体 CEM；黄色线是到物体的最近表面投影，不表示已经接触。\n\n'+'\n\n'.join(distances)
    for widget in [selection,mode,finger,contact,show_links,show_normals,show_object]:widget.on_update(update)
    @server.on_client_connect
    def camera(client):
        client.camera.position=(.55,-.6,.45);client.camera.look_at=(0,0,0);client.camera.up_direction=(0,0,1)
    update()
    write(folder/'server.json',dict(pid=os.getpid(),host=host,port=port,url=f'http://{host}:{port}/'))
    print('MAPPING_VIEWER',host,port,flush=True)
    while True:time.sleep(1)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',type=Path,default=ROOT/'output/dexonomy_shadow_mano_mapping_20261103')
    p.add_argument('--prepare',action='store_true');p.add_argument('--serve',action='store_true');p.add_argument('--device',default='cuda:1')
    p.add_argument('--host',default='192.168.0.32');p.add_argument('--port',type=int,default=8099)
    a=p.parse_args();torch.set_num_threads(1)
    if a.prepare:prepare(a.folder,a.device)
    if a.serve:serve(a.folder,a.host,a.port)
