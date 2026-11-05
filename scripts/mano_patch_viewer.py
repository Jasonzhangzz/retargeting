"""Real MANO mesh and patch traces for the current offline review page."""
import copy,json
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from retarget.env.grasp.mano_solver import FixedContacts

def mesh(vertices,faces,name,color,layer):
    return dict(type='mesh3d',name=name,**dict(zip('xyz',vertices.T.tolist())),
        **dict(zip('ijk',faces.T.tolist())),color=color,opacity=.9,
        meta=dict(role='surface',layer=layer),hoverinfo='skip',flatshading=False,
        lighting=dict(ambient=.65,diffuse=.7,specular=.15,roughness=.8))


def points(vertices,name,color,ids=None,layer='mapped'):
    t=dict(type='scatter3d',name=name,mode='markers'+('+text' if ids else ''),
        **dict(zip('xyz',vertices.T.tolist())),marker=dict(size=4,color=color),
        meta=dict(layer=layer),textposition='top center')
    if ids:t.update(customdata=ids,text=[i[1:] for i in ids],hovertemplate='槽 %{text}<br>x=%{x:.5f} m<br>y=%{y:.5f} m<br>z=%{z:.5f} m<extra>%{fullData.name}</extra>')
    return t


def lines(starts,ends,name,color,role=None,layer=None):
    coords=np.stack((starts,ends,np.full_like(starts,np.nan)),axis=1).reshape(-1,3)
    t=dict(type='scatter3d',name=name,mode='lines',line=dict(width=3,color=color),hoverinfo='skip',
        **{k:[None if np.isnan(x) else float(x) for x in v] for k,v in zip('xyz',coords.T)})
    t['meta']=dict(role=role,layer=layer)
    return t


def read_page(path):
    page=path.read_text();pos=page.index('const DATA=')+len('const DATA=')
    data,_=json.JSONDecoder().raw_decode(page[pos:]);assets=data.pop('assets',{})
    for c in data['cases']:
        for v in c['variants']:
            for f in v['figures']:f['data']=[copy.deepcopy(assets[t['asset']]) if 'asset' in t else t for t in f['data']]
    return data,page


def variant(method,label,base,a,m,records,decoder,provenance):
    q=torch.as_tensor(a['variables'],device='cuda:0',dtype=torch.float32)
    with torch.no_grad():v,j,_=decoder(q,full=True);hp,_=FixedContacts(decoder,records).evaluate(v,full=True)
    np.testing.assert_allclose(j.cpu().numpy(),a['mano_joints'],atol=4.e-7,rtol=1.e-5)
    np.testing.assert_allclose(hp.cpu().numpy(),a['mapped_hand_points'],atol=4.e-7,rtol=1.e-5)
    v=v[0].cpu().numpy();faces=np.asarray(decoder.base.faces)
    pose=base['object_pose_world'][0];r=Rotation.from_quat(pose[3:],scalar_first=True).as_matrix()
    ov=base['object_vertices']*float(base['scene_scale'][0]);ov=ov@r.T+pose[:3];of=base['object_faces']
    rr=a['rollout_rotation'][0];origin=a['object_final_origin'][0]
    move=lambda x:(x-pose[:3])@r@rr.T+origin
    hp=a['mapped_hand_points'][0];anchors=a['object_template_points'][0]
    ph=a['patch_hand_points'][0];po=a['patch_object_points'][0];fit=a['patch_fitted'][0].astype(bool)
    ids=['p'+str(i+1) for i in range(len(fit))];colors=['#00a080' if b else '#dc8b09' for b in fit]
    patch_vertices=[]
    for face_ids,bary in zip(provenance['patch_clipped_faces'],provenance['patch_clipped_barycentric']):
        patch_vertices.append(np.einsum('tij,tjk->tik',np.asarray(bary),v[faces[face_ids]]).reshape(-1,3))
    patch_vertices=np.concatenate(patch_vertices);patch_faces=np.arange(len(patch_vertices)).reshape(-1,3)
    n=int(a['collision_contact_count'][0]);figures=[]
    for side in range(2):
        target=anchors if side==0 else move(anchors);contact=po if side==0 else move(po)
        patch=mesh(patch_vertices,patch_faces,'模板手面 patch','#a84bce','mapped');patch['opacity']=.18
        traces=[mesh(v,faces,'MANO 皮肤','#c4b2a2','hand'),mesh(ov if side==0 else move(ov),of,'物体','#87a5af','object'),patch,
            points(hp,'手面固定模板点','#b38148',ids),points(target,'固定物体模板目标','#0077c8',ids),
            points(ph,'patch 手面接触点',colors,ids),points(contact,'patch 物面接触点',colors,ids),
            lines(hp,target,'优化的模板点距离','#b38148',layer='mapped'),
            lines(ph,contact,'验收的 patch 表面间距','#00a080',layer='mapped'),
            lines(target,contact,'验收的物面目标邻近距离','#0077c8',layer='mapped'),
            points(a['contact_hand_points'][0,:n],'原生凸包手接触','#a84bce',layer='native'),
            points(a['contact_object_points'][0,:n] if side==0 else move(a['contact_object_points'][0,:n]),'原生凸包物接触','#db5c61',layer='native'),
            lines(a['mano_joints'][0][decoder.edges[:,0]],a['mano_joints'][0][decoder.edges[:,1]],'MANO 骨架','#775c35',role='bones')]
        figures.append(dict(data=traces,layout=dict(showlegend=False,scene=dict(xaxis=dict(title='x / m'),yaxis=dict(title='y / m'),zaxis=dict(title='z / m')))))
    rows=[dict(id=key,slot=i+1,finger=records[i]['finger'],distance=f'{a["patch_gap_m"][0,i]*1000:.3f}',
        angle='不验收',fit='通过' if fit[i] else '未通过',face=int(a['patch_face_ids'][0,i]),
        anchor_offset=f'{a["patch_anchor_offset_m"][0,i]*1000:.3f}',fixed_anchor=f'{a["anchor_distance_m"][0,i]*1000:.3f}') for i,key in enumerate(ids)]
    vals=[('五步短时稳定','通过' if m['stable'] else '未通过'),('新联合通过','通过' if m['passed'] else '未通过'),
        ('patch 几何通过','通过' if m['geometry'] else '未通过'),('patch 覆盖',f'{m["coverage"]:.1%}'),
        ('手物 / 自穿透 mm',f'{m["penetration_mm"][0]:.3f} / {m["penetration_mm"][1]:.3f}'),
        ('固定目标 RMS mm',f'{m["fixed_anchor_rmse_mm"]:.3f}'),('rollout cost',f'{m["rollout_cost"]:.3f}'),
        ('平移 mm / 旋转 °',f'{m["displacement_mm"]:.3f} / {m["rotation_deg"]:.3f}')]
    for name,x in m['fingers'].items():vals.append((name+' patch 通过','是' if x['passed'] else '否'))
    if 'iterations' in m:vals.append(('迭代 / 提前停止',f'{m["iterations"]} / {m.get("early_stopped",False)}'))
    note=label+'。失败：'+('；'.join(m['failure_reasons']) or '无')+'。'
    note+='新规则：实际 patch 表面点间距≤2mm，且同一物面点距固定目标≤5mm；手物和自穿透均≤5mm。法向只在旧阶段参与验收。'
    note+='右侧仅显示五步物体运动，手固定；表格对应左侧。'
    return dict(id='v3_'+method,label=label,note=note,metrics=[dict(label=k,value=v) for k,v in vals],rows=rows,figures=figures)
