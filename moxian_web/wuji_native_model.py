"""Wuji 21-DoF URDF and deterministic Qnbot display mapping, never an IK fit."""
from __future__ import annotations
import copy,hashlib,json,xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .wuji_native_abi import FINGERS

METHOD='wuji_native'
REVISION='wuji_studio_calibration_2026_7_1_v1'
SCHEMA='moxian_wuji_native21_v1'
SOURCE_FINGERS=('thumb','index_finger','middle_finger','ring_finger','pinky')


class NativeURDF:
    def __init__(self,text):
        if not isinstance(text,str) or not 100<len(text)<10000000:raise ValueError('原生URDF长度无效')
        root=ET.fromstring(text);self.names=[];self.joints=[];self.limits=[]
        pending=list(root.findall('joint'));known={'wrist'}
        while pending:
            progress=False
            for j in pending[:]:
                parent=j.find('parent').get('link');child=j.find('child').get('link')
                if parent not in known:continue
                o=j.find('origin');t=np.eye(4)
                if o is not None:
                    t[:3,3]=list(map(float,o.get('xyz','0 0 0').split()));t[:3,:3]=Rotation.from_euler('xyz',list(map(float,o.get('rpy','0 0 0').split()))).as_matrix()
                index=None;axis=None
                if j.get('type')=='revolute':
                    index=len(self.names);self.names.append(j.get('name'));axis=np.array(list(map(float,j.find('axis').get('xyz').split())))
                    limit=j.find('limit');self.limits.append([float(limit.get('lower')),float(limit.get('upper'))])
                elif j.get('type')!='fixed':raise ValueError('原生URDF含未知关节类型')
                if not np.isfinite(t).all():raise ValueError('原生URDF含无效变换')
                self.joints.append((parent,child,t,index,axis));known.add(child);pending.remove(j);progress=True
            if not progress:raise ValueError('原生URDF不以wrist为根或存在环')
        if len(self.names)!=21:raise ValueError('Wuji原生手模型必须为21自由度')
        self.limits=np.asarray(self.limits);self.zero=self.forward(np.zeros(21))
        self.points=['wrist','thumb_metacarpal','thumb_middle','thumb_distal','thumb_tip']
        for f in SOURCE_FINGERS[1:]:self.points.extend(f+'_'+part for part in ('metacarpal','middle','distal','tip'))
        if not set(self.points+['emf_tx_base']+[f+'_rx_coil' for f in FINGERS]).issubset(known):raise ValueError('原生URDF解剖/EMF link缺失')

    def forward(self,q):
        q=np.asarray(q,float)
        if q.shape!=(21,) or not np.isfinite(q).all():raise ValueError('需要21个原生关节角')
        result={'wrist':np.eye(4)}
        for parent,child,origin,index,axis in self.joints:
            t=origin
            if index is not None:
                t=origin.copy();t[:3,:3]=t[:3,:3]@Rotation.from_rotvec(axis*q[index]).as_matrix()
            result[child]=result[parent]@t
        return result

    def lengths(self):
        result={}
        for f,source in zip(FINGERS,SOURCE_FINGERS):
            names=[source+'_'+part for part in ('metacarpal','middle','distal','tip')]
            positions=np.array([self.zero[n][:3,3] for n in names]);values=np.linalg.norm(np.diff(positions,axis=0),axis=-1)
            keys=('cmc_mcp','mcp_ip','ip_tip') if f=='thumb' else ('mcp_pip','pip_dip','dip_tip')
            result[f]=dict(zip(keys,map(float,values)))
        return result

    def extrinsics(self):
        mounts=[]
        for source,f in zip(SOURCE_FINGERS,FINGERS):mounts.append((np.linalg.inv(self.zero[source+'_distal'])@self.zero[f+'_rx_coil']).tolist())
        return dict(T_wrist_TX=self.zero['emf_tx_base'].tolist(),T_distal_RX=mounts,bone_lengths_m=self.lengths())


def native_model_hash(candidate):
    from .bone_extrinsics import digest
    keys=('wuji_native','display_adapter','geometry','registration','asset_hash','topology_hash','material_binding','serial','side','calibration_fingerprint')
    return digest({k:candidate.get(k) for k in keys})


def build_candidate(profiles,profile,mapping,source_hash):
    import torch
    from .qnbot_surface import QnbotSurface
    from .bone_extrinsics import definition_from_surface,segment_indices
    model=QnbotSurface(profile['side']).double();model.bind_mapping(mapping)
    if profile.get('material_binding'):model.install_binding(profile['material_binding'])
    native=NativeURDF(profiles['wujihand']['wrist_urdf']);lengths=native.lengths();base=definition_from_surface(model)['bone_lengths_m']
    scales=[np.log(lengths[f][key]/base[f][key]) for f in FINGERS for key in lengths[f]]
    if not np.isfinite(scales).all():raise ValueError('原生骨长不能映射到显示模型')
    with torch.no_grad():
        model.geometry['length'][segment_indices(model)]=torch.tensor(scales,dtype=model.q0.dtype)
        rr,pp=model.fk(model.q0[None]);wr=rr[0,model.wrist_body];wp=pp[0,model.wrist_body]
        r=(wr.T@rr[0]).numpy();p=((pp[0]-wp)@wr).numpy()
    native_tips=np.array([native.zero[f+'_tip'][:3,3] for f in SOURCE_FINGERS])
    display_tips=p[model.tip_bodies]
    c=Rotation.align_vectors(display_tips[1:],native_tips[1:])[0].as_matrix()
    aliases={profile['side']+'_wrist':'wrist',profile['side']+'_thumb_metacarpal':'thumb_proximal',
        profile['side']+'_thumb_proximal':'thumb_middle',profile['side']+'_thumb_distal':'thumb_distal',profile['side']+'_thumb_tip':'thumb_tip'}
    for target,source in zip(('index','middle','ring','little'),SOURCE_FINGERS[1:]):
        for part,native_part in (('proximal','proximal'),('intermediate','middle'),('distal','distal'),('tip','tip')):
            aliases[profile['side']+'_'+target+'_'+part]=source+'_'+native_part
    basis=[];sources=[]
    for i,name in enumerate(model.body_names):
        source=aliases.get(name);sources.append(source)
        basis.append((native.zero[source][:3,:3].T@c.T@r[i]).tolist() if source else np.eye(3).tolist())
    tx=native.zero['emf_tx_base']
    with torch.no_grad():model.align0.copy_(torch.tensor(c@tx[:3,:3]));model.trans0.copy_(torch.tensor(c@tx[:3,3]))
    candidate=model.artifact();candidate.update(serial=profile['serial'],side=profile['side'],mapping=copy.deepcopy(mapping),
        calibration_fingerprint=profile['calibration_fingerprint'],source_calibration_fingerprint=profile['calibration_fingerprint'],
        skeleton_runtime_revision=profile.get('skeleton_runtime_revision'),capture_session_id=profile.get('capture_session_id'),
        source_capture_hash=source_hash,delta_policy=None,
        wuji_native=dict(schema_revision=SCHEMA,sdk_version='2026.7.1',active_hand_profile='wujihand',profiles=profiles,solver_dof=21),
        display_adapter=dict(revision='qnbot_skin_from_wuji_fk_v1',rotation=c.tolist(),body_names=model.body_names,
            sources=sources,basis=basis,rest_rotations=r.tolist(),rest_positions=p.tolist(),derived_geometry_only=True))
    candidate['model_hash']=native_model_hash(candidate);validate_native_candidate(candidate);return candidate


def validate_native_candidate(candidate):
    n=candidate.get('wuji_native') or {}
    if n.get('schema_revision')!=SCHEMA or n.get('sdk_version')!='2026.7.1' or n.get('solver_dof')!=21:raise ValueError('不是固定版本Wuji原生21自由度模型')
    if candidate.get('model_hash')!=native_model_hash(candidate):raise ValueError('Wuji个体模型指纹不匹配')
    if n.get('active_hand_profile')!='wujihand' or set(n.get('profiles',{}))!={'wujihand','wujihand2'}:raise ValueError('原生双profile结果不完整')
    for p in n['profiles'].values():
        model=NativeURDF(p['wrist_urdf'])
        if ET.fromstring(p['wrist_urdf']).get('name')!=candidate['side']+'_hand':raise ValueError('原生URDF手侧不匹配')
    for key in ('width','depth','root','axis','residual'):
        if not np.isfinite(candidate['geometry'][key]).all() or np.any(np.asarray(candidate['geometry'][key])!=0):raise ValueError('原生模式禁止额外mesh优化变量')
    if candidate.get('delta_policy'):raise ValueError('原生模式禁止Delta策略')
    a=candidate['display_adapter'];r=np.asarray(a['rotation']);basis=np.asarray(a['basis'])
    if r.shape!=(3,3) or not np.isfinite(r).all() or not np.allclose(r.T@r,np.eye(3),atol=1e-8) or np.linalg.det(r)<.999:raise ValueError('显示坐标变换无效')
    if basis.shape!=(len(a['body_names']),3,3) or not np.isfinite(basis).all():raise ValueError('显示绑定变换无效')
    return candidate


def display_pose(candidate,kinematics):
    a=candidate['display_adapter'];c=np.asarray(a['rotation']);r=np.asarray(a['rest_rotations']).copy();p=np.asarray(a['rest_positions']).copy()
    for i,source in enumerate(a['sources']):
        if source:
            t=kinematics[source];r[i]=c@t[:3,:3]@a['basis'][i];p[i]=c@t[:3,3]
    return r,p
