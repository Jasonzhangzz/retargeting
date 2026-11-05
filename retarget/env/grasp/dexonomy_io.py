"""Read released Shadow records and reconstruct their physical scenes."""
from __future__ import annotations
import copy,hashlib,json,pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[3]

class NumpyUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if module in {'numpy','numpy.core.multiarray','numpy._core.multiarray'} and name in {'_reconstruct','ndarray','dtype','scalar'}:
            return super().find_class(module,name)
        raise ValueError(f'Unsupported pickle class: {module}.{name}')


def load_record(path):
    with Path(path).open('rb') as f:
        version=np.lib.format.read_magic(f)
        (np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0)(f)
        data=NumpyUnpickler(f).load().item()
    if not isinstance(data,dict):raise ValueError('Expected a dictionary record')
    return data


def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def relative_pose(hand,objects):
    from scipy.spatial.transform import Rotation as R
    h=R.from_quat(np.asarray(hand)[3:7],scalar_first=True)
    o=R.from_quat(np.asarray(objects)[3:7],scalar_first=True)
    return np.r_[h.inv().apply(np.asarray(objects)[:3]-np.asarray(hand)[:3]),(h.inv()*o).as_quat(scalar_first=True)]


def resolve_scene(dataset,record,row):
    extracted=Path(dataset)/'extracted'
    name=str(record['scene_path']).replace('\\','/')
    if 'assets/object/' in name:name=name.split('assets/object/',1)[1]
    path=(extracted/name).resolve()
    if not path.is_relative_to(extracted.resolve()):raise ValueError('Invalid scene path')
    scene=copy.deepcopy(load_record(path))
    factor=float(np.asarray(record.get('scene_scale',np.ones(len(record['grasp_qpos'])))).reshape(-1)[row])
    if not np.isfinite(factor) or factor<=0:raise ValueError('Invalid scene_scale')
    for cfg in scene['scene'].values():
        for key,value in list(cfg.items()):
            if key.endswith('_path') and isinstance(value,str):
                target=(path.parent/value).resolve()
                if not target.is_relative_to(extracted.resolve()) or not target.is_file():raise FileNotFoundError(target)
                cfg[key]=str(target)
        if cfg['type']=='rigid_object':
            cfg['scale']=np.asarray(cfg['scale'],float)*factor
            cfg['pose']=np.asarray(cfg['pose'],float).copy();cfg['pose'][:3]*=factor
            info=json.loads(Path(cfg['info_path']).read_text())
            volume=info['mass']/(info['density']*float(info['scale'])**3)
            cfg['density']=.1/(volume*float(np.prod(cfg['scale'])))
    return scene,path,factor


def make_env(dataset,scene):
    import sys
    sys.path.insert(0,str(ROOT/'third_party/Dexonomy'))
    from dexonomy.sim import MuJoCo_EvalEnv,MuJoCo_EvalCfg,HandCfg
    return MuJoCo_EvalEnv(HandCfg(str((Path(dataset)/'shadow_assets/right.xml').resolve()),freejoint=True),
        scene_cfg=scene,sim_cfg=MuJoCo_EvalCfg(miu_coef=[.6,.02]),debug_view=False)


def contacts(model,data,object_body):
    """Contact normal points out of object towards hand; forces act on object.

    MuJoCo stores the midpoint and signed separation. Endpoints are reconstructed
    symmetrically; each hand endpoint is also saved in its actual link frame.
    """
    import mujoco
    object_bodies={object_body}
    for b in range(object_body+1,model.nbody):
        if int(model.body_parentid[b]) in object_bodies:object_bodies.add(b)
    rows=defaultdict(list);self_depth=0.;object_depth=0.
    for cid,c in enumerate(data.contact):
        b1,b2=map(int,model.geom_bodyid[[c.geom1,c.geom2]])
        o1,o2=b1 in object_bodies,b2 in object_bodies
        if o1==o2:
            if not o1 and b1>0 and b2>0 and model.body_mocapid[b1]<0 and model.body_mocapid[b2]<0:self_depth=max(self_depth,-float(c.dist))
            continue
        hand=b2 if o1 else b1
        if hand==0 or model.body_mocapid[hand]>=0:continue
        normal=np.asarray(c.frame[:3]).copy()*(1 if o1 else -1)
        point=np.asarray(c.pos).copy();hp=point+.5*float(c.dist)*normal;op=point-.5*float(c.dist)*normal
        wrench=np.zeros(6)
        mujoco.mj_contactForce(model,data,cid,wrench)
        frame=np.asarray(c.frame).reshape(3,3);sign=-1 if o1 else 1
        rows['points'].append(point);rows['normals'].append(normal);rows['hand_points'].append(hp);rows['object_points'].append(op)
        rows['hand_local_normals'].append(np.asarray(data.xmat[hand]).reshape(3,3).T@normal)
        rows['hand_local_points'].append(np.asarray(data.xmat[hand]).reshape(3,3).T@(hp-data.xpos[hand]))
        rows['object_local_points'].append(np.asarray(data.xmat[object_body]).reshape(3,3).T@(op-data.xpos[object_body]))
        rows['hand_body_ids'].append(hand);rows['distances'].append(float(c.dist));rows['force_on_object_world'].append(sign*(frame.T@wrench[:3]));rows['torque_on_object_world'].append(sign*(frame.T@wrench[3:]))
        rows['hand_geom_ids'].append(int(c.geom2 if o1 else c.geom1));rows['object_geom_ids'].append(int(c.geom1 if o1 else c.geom2))
        object_depth=max(object_depth,-float(c.dist))
    vectors={'points','normals','hand_points','object_points','hand_local_points','hand_local_normals','object_local_points','force_on_object_world','torque_on_object_world'}
    result={k:np.asarray(rows[k],float).reshape(-1,3) for k in vectors}
    result.update({k:np.asarray(rows[k],int if k.endswith('_ids') else float) for k in ('hand_body_ids','hand_geom_ids','object_geom_ids','distances')})
    result['object_penetration_m']=object_depth;result['self_penetration_m']=self_depth
    return result
