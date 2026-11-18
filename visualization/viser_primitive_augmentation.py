"""Browse saved Wuji2 poses, clouds, contact neighbourhoods and surface normals."""
from pathlib import Path
from functools import lru_cache
import argparse
import json
import sys
import time
import threading
import h5py
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from retarget.env.grasp.primitive_augmentation.storage import Dataset,load_geometry
from retarget.env.grasp.primitive_augmentation.templates import matrix

VOLUME_COLOR=np.array([80,195,140],np.uint8)
SURFACE_COLOR=np.array([90,190,240],np.uint8)
CONTACT_COLOR=np.array([255,145,35],np.uint8)


def contact_clouds(mesh,record,radius=.003,surface_count=4096,seed=0):
    """Display-only surface resampling and neighbourhoods, in object coordinates.

    The saved cloud is volumetric and has no per-point contact annotation. Only
    points in a radius of a valid witness's surface projection are highlighted;
    volume points additionally must lie in the outer 1mm layer. Never mark a
    distant nearest volume point merely to force a visible contact label.
    """
    if radius<=0 or surface_count<0:raise ValueError('Invalid contact display settings')
    rng=np.random.default_rng(seed)
    faces=rng.choice(len(mesh.faces),surface_count,p=mesh.area_faces/mesh.area)
    uv=rng.random((surface_count,2));flip=uv.sum(1)>1;uv[flip]=1-uv[flip]
    surface=(mesh.triangles[faces]*np.c_[1-uv.sum(1),uv][...,None]).sum(1)
    hit=np.asarray(record['patch_hit'],bool)
    pose=record['object_pose_world'];rot=matrix(pose)
    hand=(np.asarray(record['contact_position_world'])[hit]-pose[:3])@rot
    if len(hand):
        anchors,_,ids=trimesh.proximity.closest_point(mesh,hand)
        normals=np.asarray(mesh.face_normals)[ids]
    else:anchors=np.empty((0,3));normals=np.empty((0,3))
    volume=np.asarray(record['volume_points_m'])
    def nearby(points):
        if not len(anchors):return np.zeros(len(points),bool)
        return np.min(np.sum((points[:,None]-anchors[None])**2,axis=-1),axis=1)<=radius**2
    volume_hit=nearby(volume);surface_hit=nearby(surface)
    ids=np.flatnonzero(volume_hit)
    if len(ids):
        _,distance,_=trimesh.proximity.closest_point(mesh,volume[ids])
        volume_hit[ids] &= distance<=.001
    volume_colors=np.tile(VOLUME_COLOR,(len(volume),1));volume_colors[volume_hit]=CONTACT_COLOR
    surface_colors=np.tile(SURFACE_COLOR,(len(surface),1));surface_colors[surface_hit]=CONTACT_COLOR
    return dict(volume=volume,surface=surface,volume_colors=volume_colors,surface_colors=surface_colors,
                volume_hit=volume_hit,surface_hit=surface_hit,anchors=anchors,normals=normals,hand=hand)


def arrow_mesh(origins,normals,length=.008):
    """True 3D arrows with cylindrical shafts and cone heads."""
    if length<=0:raise ValueError('Arrow length must be positive')
    pieces=[]
    for origin,normal in zip(origins,normals):
        norm=np.linalg.norm(normal)
        if not np.isfinite(norm) or norm<1e-10:continue
        normal=np.asarray(normal)/norm
        shaft=trimesh.creation.cylinder(radius=length*.025,height=length*.7,sections=8)
        shaft.apply_translation([0,0,length*.35])
        head=trimesh.creation.cone(radius=length*.10,height=length*.3,sections=12)
        head.apply_translation([0,0,length*.7])
        arrow=trimesh.util.concatenate((shaft,head))
        transform=trimesh.geometry.align_vectors([0,0,1],normal)
        transform[:3,3]=origin;arrow.apply_transform(transform);pieces.append(arrow)
    return trimesh.util.concatenate(pieces) if pieces else trimesh.Trimesh(vertices=np.empty((0,3)),faces=np.empty((0,3),int))


def run(root,host='0.0.0.0',port=8767,check=False):
    from moxian_web.wuji_surface import WujiSurface
    import mujoco
    ds=Dataset(Path(root).resolve());surface=WujiSurface();model=surface.model;data=mujoco.MjData(model)
    if not ds.shards:raise ValueError('No committed augmentation shards yet')
    index={};coverage=[];colors=[]
    for path in ds.shards:
        with h5py.File(path,'r') as f:
            n=len(f['qpos']);rows=f['source_row'][:];masks=f['mask_fraction'][:];modes=f['augmentation'][:].astype(str)
            index.setdefault(path.parent.name,[]).extend((path,i,int(rows[i]),float(masks[i]),str(modes[i])) for i in range(n))
            step=max(1,n//32);coverage.append(f['qpos'][::step]);colors.extend([int(f['grasp_type'][0])]*len(f['qpos'][::step]))
    if check:
        for name,records in index.items():
            path,i,*_=records[0];record=ds.sample(path,i)
            shape=load_geometry(ds.root,record['geometry_id'])
            assert shape.mesh.is_watertight
            assert record['qpos'].shape==(20,) and record['volume_points_m'].shape[1]==3
            assert np.isfinite(record['contact_position_world']).all()
            display=contact_clouds(shape.mesh,record,surface_count=512)
            assert len(display['anchors'])==record['patch_hit'].sum()
            assert np.isfinite(display['normals']).all()
            arrows=arrow_mesh(display['anchors'],display['normals'])
            assert np.isfinite(arrows.vertices).all()
            print(f'CHECK augmentation {name} samples={len(records)}',flush=True)
        return
    import viser
    server=viser.ViserServer(host=host,port=port,label='Wuji2 primitive augmentation')
    server.scene.set_up_direction('+z');lock=threading.RLock()
    names=sorted(index,key=lambda n:int(n.split('_')[0]))
    initial='14_Tripod' if '14_Tripod' in names else names[0]
    status_path=ds.root/'status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {}
    server.gui.add_markdown(f'**已保存 {sum(len(r) for r in index.values()):,} 条 pose · {len(names)} 类**\n\n'
                            f'任务状态：`{status.get("phase","unknown")}`')
    kind=server.gui.add_dropdown('抓型',names,initial_value=initial)
    augmentation=server.gui.add_dropdown('增强方式',('全部','shape','noise','slide'))
    size=server.gui.add_number('尺寸行（-1 为全部）',-1,step=1)
    mask_min=server.gui.add_slider('最小 mask',0.,.5,.01,0.)
    mask_max=server.gui.add_slider('最大 mask',0.,.5,.01,.5)
    sample=server.gui.add_slider('样本',0,max(1,len(index[initial])-1),1,0)
    previous=server.gui.add_button('上一条');following=server.gui.add_button('下一条')
    random_pose=server.gui.add_button('随机 pose')
    play=server.gui.add_checkbox('回放',False)
    cloud=server.gui.add_dropdown('点云',('表面＋体积','仅表面','仅体积','隐藏'))
    object_visible=server.gui.add_checkbox('显示物体网格',False)
    contacts=server.gui.add_checkbox('表面接触点（红）',True)
    normal_visible=server.gui.add_checkbox('物体外法向（黄）',True)
    hand_normal_visible=server.gui.add_checkbox('手部法向（紫）',False)
    radius=server.gui.add_slider('接触邻域半径 / mm',.5,10.,.5,3.)
    normal_length=server.gui.add_slider('法向箭头长度 / mm',2.,20.,1.,8.)
    opacity=server.gui.add_slider('物体透明度',.05,1.,.05,.2)
    hand_opacity=server.gui.add_slider('手部透明度',.1,1.,.05,.7)
    info=server.gui.add_markdown('')
    coverage_enabled=server.gui.add_checkbox('显示关节覆盖 PCA',False)
    server.gui.add_markdown('覆盖图仅作查看；20 维覆盖与连通性以 report.json 为准。')
    server.gui.add_markdown('青色：展示用表面点 · 绿色：保存的体积点 · 橙色：接触邻域。\n\n'
                            '红色标记和黄色箭头位于真实物体表面；邻域着色为可视化估计，不是新增训练标签。')
    handles=[];state={'filtered':index[initial],'updating':False};select_rng=np.random.default_rng()
    @lru_cache(maxsize=32)
    def geometry(gid):return load_geometry(ds.root,gid)
    q=np.concatenate(coverage);domain=ds.config['identity']['domain'];lo=np.array(domain['lower']);hi=np.array(domain['upper'])
    normalized=(q-lo)/(hi-lo);center=normalized.mean(0)
    _,_,v=np.linalg.svd(normalized-center,full_matrices=False)
    reduced=(normalized-center)@v[:3].T
    xyz=np.pad(reduced,((0,0),(0,3-reduced.shape[1])))*.12+np.array([.22,0,0])
    rng=np.random.default_rng(12);palette=rng.integers(50,240,(34,3),dtype=np.uint8)
    cover=server.scene.add_point_cloud('/coverage',xyz.astype(np.float32),palette[np.array(colors)],point_size=.0015,visible=False)
    def render(_=None):
        with lock:
            if state['updating']:return
            for h in handles:h.remove()
            handles.clear();cover.visible=coverage_enabled.value
            selected=state['filtered']
            if not selected:info.content='筛选后没有样本。';return
            path,i,*_=selected[min(int(sample.value),len(selected)-1)]
            record=ds.sample(path,i);base=record['target_base_pose'];obj=record['object_pose_world']
            # Translate display by the object's saved origin; retain orientation.
            offset=obj[:3]
            data.qpos[:6]=np.r_[base[:3],Rotation.from_matrix(matrix(base)).as_euler('XYZ')]
            data.qpos[6:]=record['qpos'];mujoco.mj_kinematics(model,data)
            for owner,mesh in enumerate(surface.meshes):
                body=surface.body_ids[owner]
                vertices=np.asarray(mesh.vertices)@data.xmat[body].reshape(3,3).T+data.xpos[body]-offset
                handles.append(server.scene.add_mesh_simple(f'/hand/{owner}',vertices,mesh.faces,color=(145,188,225),opacity=hand_opacity.value))
            shape=geometry(record['geometry_id']);rot=matrix(obj)
            display=contact_clouds(shape.mesh,record,radius.value/1000,seed=int(record.get('cloud_seed',0)))
            if object_visible.value:
                vertices=shape.mesh.vertices@rot.T
                handles.append(server.scene.add_mesh_simple('/object',vertices,shape.mesh.faces,color=(90,200,150),opacity=opacity.value))
            if cloud.value in ('表面＋体积','仅体积'):
                handles.append(server.scene.add_point_cloud('/volume',display['volume']@rot.T,
                    display['volume_colors'],point_size=.0009,point_shape='circle',precision='float32'))
            if cloud.value in ('表面＋体积','仅表面'):
                handles.append(server.scene.add_point_cloud('/surface',display['surface']@rot.T,
                    display['surface_colors'],point_size=.0007,point_shape='circle',precision='float32'))
            if contacts.value:
                handles.append(server.scene.add_point_cloud('/contacts',display['anchors']@rot.T,
                    np.array([255,65,55],np.uint8),point_size=.002,point_shape='circle',precision='float32'))
            if normal_visible.value:
                arrows=arrow_mesh(display['anchors']@rot.T,display['normals']@rot.T,normal_length.value/1000)
                if len(arrows.faces):handles.append(server.scene.add_mesh_simple('/normals',arrows.vertices,arrows.faces,color=(255,220,45)))
            if hand_normal_visible.value:
                hit=record['patch_hit'].astype(bool)
                arrows=arrow_mesh(record['contact_position_world'][hit]-offset,record['contact_normal_world'][hit],normal_length.value/1000)
                if len(arrows.faces):handles.append(server.scene.add_mesh_simple('/hand_normals',arrows.vertices,arrows.faces,color=(220,100,255)))
            info.content=(f'**{kind.value}** · {record["augmentation"]} · {int(sample.value)+1}/{len(selected)}\n\n尺寸行 {record["source_row"]} · '
                          f'mask {100*record["mask_fraction"]:.1f}% · patch {record["patch_hit"].sum()}/{len(record["patch_hit"])}\n\n'
                          f'手物穿透 {record["object_penetration_m"]*1000:.3f} mm · '
                          f'自穿透 {record["self_penetration_m"]*1000:.3f} mm\n\n'
                          f'接触邻域：表面点 {display["surface_hit"].sum()} · 体积点 {display["volume_hit"].sum()}\n\n'
                          f'文件：`{path.name}` / 行 {i}')
    def filter_rows(_=None):
        with lock:
            state['updating']=True
            state['filtered']=[r for r in index[kind.value] if (size.value<0 or r[2]==int(size.value))
                               and mask_min.value<=r[3]<=mask_max.value and (augmentation.value=='全部' or r[4]==augmentation.value)]
            sample.max=max(1,len(state['filtered'])-1);sample.value=0
            state['updating']=False
        render()
    for widget in (kind,augmentation,size,mask_min,mask_max):widget.on_update(filter_rows)
    for widget in (sample,cloud,object_visible,contacts,normal_visible,hand_normal_visible,radius,normal_length,opacity,hand_opacity,coverage_enabled):widget.on_update(render)
    @previous.on_click
    def _(_event):sample.value=max(0,int(sample.value)-1)
    @following.on_click
    def _(_event):sample.value=min(max(0,len(state['filtered'])-1),int(sample.value)+1)
    @random_pose.on_click
    def _(_event):
        if state['filtered']:sample.value=int(select_rng.integers(len(state['filtered'])))
    @server.on_client_connect
    def _(client):
        client.camera.position=(.22,-.28,.20);client.camera.look_at=(0.,0.,.01);client.camera.up=(0.,0.,1.)
    render();print(f'VISER_READY http://127.0.0.1:{server.get_port()}',flush=True)
    try:
        while True:
            time.sleep(.2)
            if play.value and state['filtered']:sample.value=(int(sample.value)+1)%len(state['filtered'])
    except KeyboardInterrupt:server.stop()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/datasets/wuji2_primitive_aug_pilot_v1')
    parser.add_argument('--host',default='0.0.0.0');parser.add_argument('--port',type=int,default=8782)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args();run(args.data,args.host,args.port,args.check)
