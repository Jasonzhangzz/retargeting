"""Portable, metre/radian native hand model and atomic runtime selection.

This module intentionally depends only on NumPy/SciPy, so the isolated SDK
worker and the Torch skin worker can consume exactly the same model.
"""
from __future__ import annotations
import hashlib
import json
import os
import uuid
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

SCHEMA = 'moxian_bone_extrinsics_v1'
METHOD = 'bone_extrinsics'
FIT_REVISION = 'moxian_static_bone_extrinsics_v1'
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
SOURCE_FINGERS = ('thumb', 'index', 'middle', 'ring', 'little')
# Installation is initialized from only two holds. Its uncertainty is much
# larger than a small correction to an already registered receiver frame.
INSTALLATION_BOUNDS_REVISION = 'six_pose_installation_v2'
INSTALLATION_BOUNDS = dict(mount_pos=5., mount_rot=1.5, alignment=.5, translation=5.)
SOLVER_DEFAULTS = dict(v_max=2*np.pi, lm_damping=1e-4, global_damping=1e-8,
    posture_weight=1., thumb_posture_scale=.25, root_twist_posture_scale=.25, history_weight=20.,
    thumb_palm_posture_weight=8., thumb_palm_history_weight=80.,
    limit_gain=1., position_sigma_m=.003, rotation_sigma_rad=.15,
    max_iterations=40, retry_iterations=36, retry_position_m=.005,
    retry_rotation_rad=float(np.deg2rad(5)))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
        separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('w') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
            stream.flush(); os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def segment_indices(model):
    names = [f'{model.side}_{f}_{part}' for f in SOURCE_FINGERS
             for part in (('metacarpal', 'proximal', 'distal') if f=='thumb'
                          else ('proximal', 'intermediate', 'distal'))]
    return [model.bone_names.index(name) for name in names]


def transform(r, p):
    result = np.eye(4); result[:3, :3] = r; result[:3, 3] = p
    return result.tolist()


def definition_from_surface(model, solver=None):
    def array(name): return getattr(model, name).detach().cpu().numpy()
    with __import__('torch').no_grad():
        align, trans = model.alignment()
        mp = model.mount_p0+.01*model.geometry['mount_pos']
        from .qnbot_surface import exp_so3
        mr = model.mount_r0@exp_so3(model.geometry['mount_rot'])
    indices = segment_indices(model); lengths = []
    for bone in indices:
        children = [i for i in range(1,len(model.parents))
                    if model.body_bone[model.parents[i]]==bone and np.linalg.norm(array('body_p0')[i])>1e-6]
        if len(children)!=1: raise ValueError('骨段必须对应唯一非零平移')
        lengths.append(float(np.linalg.norm(array('body_p0')[children[0]])*
                             model.geometry['length'][bone].detach().exp().cpu()))
    rig = dict(side=model.side, joint_names=model.joint_names, body_names=model.body_names,
        parents=model.parents, body_joints=model.body_joints, body_bone=model.body_bone,
        bone_body=model.bone_body, wrist_body=model.wrist_body,
        distal_bodies=model.distal_bodies, tip_bodies=model.tip_bodies)
    for name in ('body_p0','body_r0','joint_axis0','joint_p0','limits','q0'):
        rig[name] = array(name).tolist()
    # Rest angles include the original static joint rotations.
    rest = array('q0').copy()
    for name, angle in model.thumb_straight.items(): rest[model.joint_names.index(name)] = angle
    for i,name in enumerate(model.joint_names):
        if name.endswith(('_pip_flexion','_dip_flexion')):
            body = next(b for b,js in enumerate(model.body_joints) if i in js)
            rest[i] = -Rotation.from_matrix(array('body_r0')[body]).as_rotvec()[2]
        if name in model.finger_straight:rest[i]=model.finger_straight[name]
    rest = np.clip(rest, array('limits')[:,0], array('limits')[:,1])
    return dict(schema_revision=SCHEMA, rig=rig, segment_bone_indices=indices,
        bone_lengths_m={f:dict(zip(('cmc_mcp','mcp_ip','ip_tip') if f=='thumb'
            else ('mcp_pip','pip_dip','dip_tip'), lengths[3*i:3*i+3])) for i,f in enumerate(FINGERS)},
        length_log_scales=model.geometry['length'].detach().cpu().tolist(),
        T_wrist_TX=transform(align.cpu().numpy(),trans.cpu().numpy()),
        T_distal_RX=[transform(r,p) for r,p in zip(mr.cpu().numpy(),mp.cpu().numpy())],
        rest_angles_rad=rest.tolist(),solver_config={**SOLVER_DEFAULTS,**(solver or {})},
        joint_limit_policy=__import__('copy').deepcopy(model.joint_limit_policy),mesh_policy='bone_length_driven_longitudinal_skin_v1')


def model_hash(candidate):
    if candidate.get('wuji_native'):
        from .wuji_native_model import native_model_hash
        return native_model_hash(candidate)
    return digest({key:candidate.get(key) for key in ('bone_extrinsics','geometry','registration',
        'asset_hash','topology_hash','material_binding','serial','side','calibration_fingerprint')})


def validate_candidate(candidate):
    if candidate.get('wuji_native'):
        from .wuji_native_model import validate_native_candidate
        return validate_native_candidate(candidate)
    definition = candidate.get('bone_extrinsics', {})
    if definition.get('schema_revision')!=SCHEMA: raise ValueError('不是骨长与外参模型')
    if candidate.get('model_hash')!=model_hash(candidate): raise ValueError('个体模型指纹不匹配')
    geometry = candidate['geometry']; ids = definition['segment_bone_indices']
    if len(ids)!=15 or len(set(ids))!=15: raise ValueError('需要15个独立骨段')
    lengths = np.asarray(geometry['length'],float)
    if lengths.shape!=(20,) or not np.isfinite(lengths).all() or np.max(abs(lengths))>.150001:
        raise ValueError('骨长参数越界')
    frozen = np.ones(20,bool); frozen[ids] = False
    if np.any(lengths[frozen]!=0): raise ValueError('非目标掌骨长度必须固定')
    for key in ('width','depth','root','axis','residual'):
        if np.any(np.asarray(geometry[key])!=0): raise ValueError('标定不允许修改 '+key)
    if not np.allclose(lengths,definition['length_log_scales'],atol=1e-8): raise ValueError('骨长表示不一致')
    for matrix in [definition['T_wrist_TX'],*definition['T_distal_RX']]:
        t=np.asarray(matrix,float)
        if (t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1])
            or not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-5)
            or not np.isclose(np.linalg.det(t[:3,:3]),1,atol=1e-5)):
            raise ValueError('外参不是刚体变换')
    if candidate.get('delta_policy'): raise ValueError('该模型不允许Delta策略')
    bounds_revision=definition.get('calibration_bounds_revision')
    if bounds_revision not in (None,INSTALLATION_BOUNDS_REVISION):raise ValueError('未知安装外参范围')
    bounds=INSTALLATION_BOUNDS if bounds_revision else dict(mount_pos=2.,mount_rot=.3,alignment=.2,translation=2.)
    for key,bound in bounds.items():
        if not np.isfinite(geometry[key]).all() or np.max(np.abs(geometry[key]))>bound+1e-6:raise ValueError('外参增量越界 '+key)
    registration=candidate['registration']
    expected=[transform(Rotation.from_rotvec(geometry['alignment']).as_matrix()@np.asarray(registration['align0']),
        np.asarray(registration['trans0'])+.01*np.asarray(geometry['translation']))]
    for i in range(5):
        expected.append(transform(np.asarray(registration['mount_r0'])[i]@Rotation.from_rotvec(geometry['mount_rot'][i]).as_matrix(),
            np.asarray(registration['mount_p0'])[i]+.01*np.asarray(geometry['mount_pos'])[i]))
    actual=[definition['T_wrist_TX'],*definition['T_distal_RX']]
    if len(actual)!=6 or not np.allclose(actual,expected,atol=2e-6):raise ValueError('FK与蒙皮的外参表示不一致')
    rig=definition['rig']
    if candidate.get('joint_limits_rad') is not None and not np.allclose(candidate['joint_limits_rad'],rig['limits'],atol=1e-8):raise ValueError('关节限位序列化不一致')
    dof=len(rig['joint_names'])
    if dof not in (25,29) or np.asarray(rig['limits']).shape!=(dof,2):raise ValueError('需要原生25关节或25＋4侧旋模型')
    if dof==29 and rig['joint_names'][25:]!=[rig['side']+'_'+f+'_mcp_twist' for f in SOURCE_FINGERS[1:]]:raise ValueError('四指侧旋关节名称不匹配')
    if np.asarray(rig['joint_axis0']).shape!=(dof,3) or np.asarray(geometry['axis']).shape!=(dof,3) or sorted(j for js in rig['body_joints'] for j in js)!=list(range(dof)):raise ValueError('关节自由度与FK轴不一致')
    named=np.array([[definition['bone_lengths_m'][f][name] for name in
        (('cmc_mcp','mcp_ip','ip_tip') if f=='thumb' else ('mcp_pip','pip_dip','dip_tip'))] for f in FINGERS])
    reference=[]
    for bone in ids:
        edges=[i for i in range(1,len(rig['parents'])) if rig['body_bone'][rig['parents'][i]]==bone and np.linalg.norm(rig['body_p0'][i])>1e-6]
        if len(edges)!=1:raise ValueError('无效骨段平移')
        reference.append(np.linalg.norm(rig['body_p0'][edges[0]])*np.exp(lengths[bone]))
    if named.shape!=(5,3) or not np.allclose(named.ravel(),reference,atol=1e-7):raise ValueError('命名骨长与FK不一致')
    config=definition['solver_config']
    required=set(SOLVER_DEFAULTS)-{'root_twist_posture_scale','thumb_palm_posture_weight','thumb_palm_history_weight'}
    if not required<=set(config)<=set(SOLVER_DEFAULTS) or any(not np.isfinite(v) or v<0 for v in config.values()):raise ValueError('无效QP配置')
    for key in ('position_sigma_m','rotation_sigma_rad','global_damping','limit_gain'):
        if config[key]<=0:raise ValueError('无效QP配置 '+key)
    for key in ('max_iterations','retry_iterations'):
        if type(config[key]) is not int or not 1<=config[key]<=40:raise ValueError('无效QP迭代数')
    return candidate


def publish_active(home, candidate):
    """Immutable content, then a single atomic pointer; neither consumer guesses."""
    home=Path(home)
    if candidate is None:
        atomic_json(home/'active.json',dict(schema_revision=SCHEMA,artifact=None)); return
    validate_candidate(candidate)
    checksum=digest(candidate); name=f'models/{checksum}.json'
    atomic_json(home/name,candidate)
    atomic_json(home/'active.json',dict(schema_revision=SCHEMA,artifact=name,
        artifact_sha256=checksum,model_hash=candidate['model_hash']))


def read_active(home, serial=None, side=None, fingerprint=None):
    home=Path(home); path=home/'active.json'
    if not path.exists(): return None
    pointer=json.loads(path.read_text())
    if pointer.get('schema_revision')!=SCHEMA: raise ValueError('活动模型版本无效')
    if pointer.get('artifact') is None: return None
    name=pointer['artifact']
    if not isinstance(name,str) or Path(name).parts!=('models',Path(name).name): raise ValueError('活动模型路径无效')
    candidate=json.loads((home/name).read_text())
    if digest(candidate)!=pointer.get('artifact_sha256'): raise ValueError('活动模型文件校验失败')
    validate_candidate(candidate)
    if candidate['model_hash']!=pointer.get('model_hash'): raise ValueError('活动模型指纹不一致')
    for key,value in (('serial',serial),('side',side),('calibration_fingerprint',fingerprint)):
        if value is not None and candidate.get(key)!=value: raise ValueError('活动模型来源不匹配：'+key)
    return candidate


class BoneKinematics:
    def __init__(self, definition):
        self.definition=definition; self.rig=definition['rig']; r=self.rig
        for key in ('body_p0','body_r0','joint_axis0','joint_p0','limits','q0'):
            setattr(self,key,np.asarray(r[key],float))
        self.scales=np.exp(definition['length_log_scales'])
        self.mounts=np.asarray(definition['T_distal_RX'],float)
        self.alignment=np.asarray(definition['T_wrist_TX'],float)
        self.joint_names=r['joint_names']; self.body_names=r['body_names']
        self.chains=[]; self.finger_indices=[]
        for distal in r['distal_bodies']:
            chain=[]; body=distal
            while body!=r['wrist_body']:
                chain.append(body); body=r['parents'][body]
            chain.reverse(); self.chains.append(chain)
            self.finger_indices.append(np.array([j for b in chain for j in r['body_joints'][b]],int))

    def _body(self, body, q, rotation, position, axes=None, centers=None):
        r=self.rig; p=self.body_p0[body]*self.scales[r['body_bone'][r['parents'][body]]]
        position=position+rotation@p; rotation=rotation@self.body_r0[body]
        for j in r['body_joints'][body]:
            jp=self.joint_p0[j]
            if axes is not None:
                axes.append(rotation@self.joint_axis0[j]); centers.append(position+rotation@jp)
            jr=Rotation.from_rotvec(self.joint_axis0[j]*q[j]).as_matrix()
            position=position+rotation@(jp-jr@jp); rotation=rotation@jr
        return rotation,position

    def forward(self,q):
        r=self.rig; rotations=[np.eye(3)]; positions=[np.zeros(3)]
        for body in range(1,len(r['parents'])):
            rr,pp=self._body(body,q,rotations[r['parents'][body]],positions[r['parents'][body]])
            rotations.append(rr); positions.append(pp)
        rotations=np.array(rotations); positions=np.array(positions); w=r['wrist_body']
        return rotations[w].T@rotations,(positions-positions[w])@rotations[w]

    def sensor(self,f,q,jacobian=False):
        rot=np.eye(3); pos=np.zeros(3); axes=[]; centers=[]
        for body in self.chains[f]: rot,pos=self._body(body,q,rot,pos,axes,centers)
        mount=self.mounts[f]; p=pos+rot@mount[:3,3]; r=rot@mount[:3,:3]
        if not jacobian: return r,p
        angular=np.array(axes).T
        linear=np.cross(np.array(axes),p-np.array(centers)).T
        return r,p,linear,angular
