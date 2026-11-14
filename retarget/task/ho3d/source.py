"""Read HO-3D train annotations without joining gaps or changing object frames."""
from pathlib import Path
import io
import pickle
import zipfile
from functools import lru_cache
import numpy as np
from scipy.spatial.transform import Rotation

# Official HO-3D setup_mano.py appends these vertices to the 16 MANO joints.
HO3D_TIPS = [744, 333, 444, 555, 672]
SIMPLE_ORDER = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
CAMERA_TO_ZUP = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


class AnnotationUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ('numpy', 'numpy.core.multiarray', 'numpy._core.multiarray') and name in ('ndarray', 'dtype', '_reconstruct', 'scalar'):
            return getattr(__import__(module, fromlist=[name]), name)
        if module == '_codecs' and name == 'encode':
            import codecs
            return codecs.encode
        raise pickle.UnpicklingError(f'Unexpected annotation class: {module}.{name}')


class Source:
    def __init__(self, path):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(path) if self.path.is_file() else None
        # A wrapper folder is allowed, but never guess another split.
        names = self.zip.namelist() if self.zip else []
        matches = [n for n in names if n == 'train.txt' or n.endswith('/train.txt')]
        if self.zip and len(matches) != 1:
            raise ValueError('Archive must contain one train.txt')
        self.prefix = matches[0][:-len('train.txt')] if self.zip else ''

    def read(self, name):
        return self.zip.read(self.prefix + name) if self.zip else (self.path / name).read_bytes()

    def sequences(self):
        rows = self.read('train.txt').decode().splitlines()
        return sorted({r.strip().split('/')[0] for r in rows if r.strip()})

    def annotations(self, sequence):
        if '/' in sequence or sequence in ('.', '..'):
            raise ValueError('Expected a sequence name')
        rows = [r.strip() for r in self.read('train.txt').decode().splitlines() if r.strip().startswith(sequence + '/')]
        rows = sorted(set(rows), key=lambda x: int(x.split('/')[1]))
        for row in rows:
            seq, frame = row.split('/')
            annotation = AnnotationUnpickler(io.BytesIO(self.read(f'train/{seq}/meta/{frame}.pkl')), encoding='latin1').load()
            yield int(frame), annotation

    def close(self):
        if self.zip: self.zip.close()


def valid_annotation(a):
    for key, size in [('handPose', 48), ('handTrans', 3), ('handBeta', 10), ('objRot', 3), ('objTrans', 3)]:
        x = np.asarray(a.get(key), dtype=float)
        if x.size != size or not np.isfinite(x).all(): return False
    return isinstance(a.get('objName'), str) and bool(a['objName'])


def contiguous_segments(rows):
    segment = []
    for frame, anno in rows:
        if not valid_annotation(anno):
            if segment: yield segment
            segment = []; continue
        if segment and (frame != segment[-1][0] + 1 or anno['objName'] != segment[-1][1]['objName']):
            yield segment; segment = []
        segment.append((frame, anno))
    if segment: yield segment


def recover_mano(model_path, poses, translations, betas):
    """Exact flat-mean MANO LBS and official HO-3D 21-joint ordering.

    handTrans is MANO's translation parameter, not the wrist joint position.
    Preserve the shape-dependent wrist offset when rotating the mesh.
    """
    import torch
    from retarget.learning.mano_contact.geometry import ManoGeometry
    from retarget.learning.mano_contact.rotations import axis_angle_to_matrix
    m = ManoGeometry(model_path)
    q = torch.as_tensor(np.c_[translations, poses], dtype=torch.float32)
    beta = torch.as_tensor(betas, dtype=torch.float32)
    vertices, physical = m(q, beta)
    rest = m.v_template + torch.einsum('vck,bk->bvc', m.shapedirs, beta)
    joints = torch.einsum('jv,bvc->bjc', m.J_regressor, rest)
    local = axis_angle_to_matrix(torch.as_tensor(poses, dtype=torch.float32).reshape(-1,16,3))
    rotations, positions = [], []
    for j, parent in enumerate(m.parents):
        if parent < 0:
            rotations.append(local[:,j]); positions.append(joints[:,j] + q[:,:3])
        else:
            rotations.append(rotations[parent] @ local[:,j])
            positions.append(positions[parent] + (rotations[parent] @ (joints[:,j]-joints[:,parent])[...,None]).squeeze(-1))
    native = torch.stack(positions,1).numpy()
    points = np.concatenate([native, vertices[:, HO3D_TIPS]], axis=1)[:, SIMPLE_ORDER]
    return vertices, points, physical, m.faces


@lru_cache(maxsize=2)
def vertex_finger_owners(model_path):
    from retarget.learning.mano_contact.geometry import ManoGeometry
    w=ManoGeometry(model_path).weights.numpy()
    groups=((13,14,15),(1,2,3),(4,5,6),(10,11,12),(7,8,9))
    return np.stack([w[:,0],*[w[:,ids].sum(1) for ids in groups]],axis=1).argmax(1)-1


def recover_segment(rows, mano_path, fps=30., *, camera_to_world=CAMERA_TO_ZUP):
    if not rows or fps <= 0: raise ValueError('Expected a nonempty motion and positive fps')
    ids = np.array([r[0] for r in rows], np.int64)
    if len(ids) > 1 and np.any(np.diff(ids) != 1): raise ValueError('Motion is not contiguous')
    a = [r[1] for r in rows]
    pose = np.stack([x['handPose'] for x in a]).reshape(-1,48)
    trans = np.stack([x['handTrans'] for x in a]).reshape(-1,3)
    betas = np.stack([x['handBeta'] for x in a]).reshape(-1,10)
    v, j, physical, faces = recover_mano(mano_path, pose, trans, betas)
    obj_t = np.stack([x['objTrans'] for x in a]).reshape(-1,3)
    obj_r = Rotation.from_rotvec(np.stack([x['objRot'] for x in a]).reshape(-1,3)).as_matrix()
    r = np.asarray(camera_to_world, float)
    if r.shape != (3,3) or not np.allclose(r.T@r,np.eye(3),atol=1e-7) or np.linalg.det(r)<0: raise ValueError('Expected a proper camera-to-world rotation')
    # One constant translation for the whole scene, never per-frame recentering.
    offset = -(r @ obj_t[0])
    offset[2] += .3
    object_world = obj_t @ r.T + offset
    root_world = r[None] @ Rotation.from_rotvec(physical[:,3:6]).as_matrix()
    pose_world = np.c_[physical[:,:3] @ r.T + offset, Rotation.from_matrix(root_world).as_rotvec(), physical[:,6:]]
    result = dict(frame_ids=ids, time_s=(ids-ids[0])/fps, mano_pose51=pose_world.astype(np.float32),
                  mano_pose48_source=pose.astype(np.float32), mano_trans_source=trans.astype(np.float32), mano_beta=betas.astype(np.float32),
                  mano_vertices=(v@r.T+offset).astype(np.float32), mano_joints=(j@r.T+offset).astype(np.float32), mano_faces=faces,
                  object_pose=np.c_[object_world,Rotation.from_matrix(r[None]@obj_r).as_quat(scalar_first=True)].astype(np.float32),
                  camera_to_world=r, world_offset=offset, object_name=np.array(a[0]['objName']), fps=np.array(fps))
    labelled = [x.get('handJoints3D') for x in a]
    if all(x is not None and np.asarray(x).shape==(21,3) for x in labelled):
        # Source annotations store native MANO joints + five appended tips.
        target=np.stack(labelled)[:, SIMPLE_ORDER]
        result['fk_joint_rmse_m']=np.array(np.sqrt(np.mean((j-target)**2)))
    for key, name, dtype in [('handVertContact','vertex_contact',bool),('handVertDist','vertex_distance_m',np.float32)]:
        arrays=[x.get(key) for x in a]
        valid=np.array([x is not None and np.asarray(x).size==778 for x in arrays])
        result[name+'_valid']=valid
        result[name]=np.stack([np.asarray(x,dtype=dtype).reshape(778) if ok else np.zeros(778,dtype=dtype) for x,ok in zip(arrays,valid)])
    owner=vertex_finger_owners(str(mano_path))
    contact=result['vertex_contact']
    fingers=np.stack([(contact[:,owner==f].sum(1)>=3) for f in range(5)],axis=1)
    result['finger_contact']=fingers & result['vertex_contact_valid'][:,None]
    result['grasp_phase']=(result['finger_contact'].sum(1)>=2).astype(np.float32)
    return result
