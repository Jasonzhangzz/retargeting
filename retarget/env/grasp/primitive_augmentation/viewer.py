"""Viser integration for committed augmentation shards; CPU-only reading."""
from pathlib import Path
import json
import time
import threading
import h5py
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from .storage import Dataset,load_geometry
from .templates import matrix


def run(root,host='0.0.0.0',port=8767,check=False):
    from moxian_web.wuji_surface import WujiSurface
    import mujoco
    ds=Dataset(root);surface=WujiSurface();model=surface.model;data=mujoco.MjData(model)
    if not ds.shards:raise ValueError('No committed augmentation shards yet')
    index={};coverage=[];colors=[]
    for path in ds.shards:
        with h5py.File(path,'r') as f:
            n=len(f['qpos']);rows=f['source_row'][:];masks=f['mask_fraction'][:]
            index.setdefault(path.parent.name,[]).extend((path,i,int(rows[i]),float(masks[i])) for i in range(n))
            step=max(1,n//32);coverage.append(f['qpos'][::step]);colors.extend([int(f['grasp_type'][0])]*len(f['qpos'][::step]))
    if check:
        for name,records in index.items():
            path,i,_,_=records[0];record=ds.sample(path,i,world=True)
            shape=load_geometry(ds.root,record['geometry_id'])
            assert shape.mesh.is_watertight
            assert record['qpos'].shape==(20,) and record['volume_points_m'].shape[1]==3
            assert np.isfinite(record['contact_position_world']).all()
            print(f'CHECK augmentation {name} samples={len(records)}',flush=True)
        return
    import viser
    server=viser.ViserServer(host=host,port=port,label='Wuji2 primitive augmentation')
    server.scene.set_up_direction('+z');lock=threading.RLock()
    names=sorted(index,key=lambda n:int(n.split('_')[0]))
    kind=server.gui.add_dropdown('抓型',names)
    size=server.gui.add_number('尺寸行（-1 为全部）',-1,step=1)
    mask_min=server.gui.add_slider('最小 mask',0.,.5,.01,0.)
    mask_max=server.gui.add_slider('最大 mask',0.,.5,.01,.5)
    sample=server.gui.add_slider('样本',0,max(1,len(index[names[0]])-1),1,0)
    play=server.gui.add_checkbox('回放',False)
    cloud=server.gui.add_checkbox('体积点云',True)
    contacts=server.gui.add_checkbox('接触见证点',True)
    opacity=server.gui.add_slider('物体透明度',.05,1.,.05,.25)
    info=server.gui.add_markdown('')
    coverage_enabled=server.gui.add_checkbox('显示关节覆盖 PCA',False)
    server.gui.add_markdown('覆盖图仅作查看；20 维覆盖与连通性以 report.json 为准。')
    handles=[];state={'filtered':index[names[0]],'updating':False}
    q=np.concatenate(coverage);domain=ds.config['identity']['domain'];lo=np.array(domain['lower']);hi=np.array(domain['upper'])
    normalized=(q-lo)/(hi-lo);center=normalized.mean(0)
    _,_,v=np.linalg.svd(normalized-center,full_matrices=False)
    xyz=(normalized-center)@v[:3].T*.12+np.array([.22,0,0])
    rng=np.random.default_rng(12);palette=rng.integers(50,240,(34,3),dtype=np.uint8)
    cover=server.scene.add_point_cloud('/coverage',xyz.astype(np.float32),palette[np.array(colors)],point_size=.0015,visible=False)
    def render(_=None):
        with lock:
            if state['updating']:return
            for h in handles:h.remove()
            handles.clear();cover.visible=coverage_enabled.value
            selected=state['filtered']
            if not selected:info.content='筛选后没有样本。';return
            path,i,_,_=selected[min(int(sample.value),len(selected)-1)]
            record=ds.sample(path,i,world=True);base=record['target_base_pose'];obj=record['object_pose_world']
            # Translate display by the object's saved origin; retain orientation.
            offset=obj[:3]
            data.qpos[:6]=np.r_[base[:3],Rotation.from_matrix(matrix(base)).as_euler('XYZ')]
            data.qpos[6:]=record['qpos'];mujoco.mj_kinematics(model,data)
            for owner,mesh in enumerate(surface.meshes):
                body=surface.body_ids[owner]
                vertices=np.asarray(mesh.vertices)@data.xmat[body].reshape(3,3).T+data.xpos[body]-offset
                handles.append(server.scene.add_mesh_simple(f'/hand/{owner}',vertices,mesh.faces,color=(145,188,225)))
            shape=load_geometry(ds.root,record['geometry_id'])
            vertices=shape.mesh.vertices@matrix(obj).T+obj[:3]-offset
            handles.append(server.scene.add_mesh_simple('/object',vertices,shape.mesh.faces,color=(90,200,150),opacity=opacity.value))
            if cloud.value:
                handles.append(server.scene.add_point_cloud('/volume',record['volume_points_m']-offset,
                    np.array([80,220,160],np.uint8),point_size=.0007))
            if contacts.value:
                hit=record['patch_hit'].astype(bool)
                c=np.where(hit[:,None],np.array([255,110,60]),np.array([90,90,90])).astype(np.uint8)
                handles.append(server.scene.add_point_cloud('/contacts',record['contact_position_world']-offset,c,point_size=.002))
            info.content=(f'**{kind.value}** · {record["augmentation"]}\n\n尺寸行 {record["source_row"]} · '
                          f'mask {100*record["mask_fraction"]:.1f}% · patch {record["patch_hit"].sum()}/{len(record["patch_hit"])}\n\n'
                          f'手物穿透 {record["object_penetration_m"]*1000:.3f} mm · '
                          f'自穿透 {record["self_penetration_m"]*1000:.3f} mm')
    def filter_rows(_=None):
        with lock:
            state['updating']=True
            state['filtered']=[r for r in index[kind.value] if (size.value<0 or r[2]==int(size.value))
                               and mask_min.value<=r[3]<=mask_max.value]
            sample.max=max(1,len(state['filtered'])-1);sample.value=0
            state['updating']=False
        render()
    for widget in (kind,size,mask_min,mask_max):widget.on_update(filter_rows)
    for widget in (sample,cloud,contacts,opacity,coverage_enabled):widget.on_update(render)
    render();print(f'VISER_READY http://127.0.0.1:{server.get_port()}',flush=True)
    try:
        while True:
            time.sleep(.2)
            if play.value and state['filtered']:sample.value=(int(sample.value)+1)%len(state['filtered'])
    except KeyboardInterrupt:server.stop()
