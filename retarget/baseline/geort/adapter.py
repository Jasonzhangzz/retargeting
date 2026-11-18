"""Shared Moxian/GeoRT coordinates and exact, fixed-wrist Wuji2 FK.

No optimization, tactile correction, scaling, or learned FK in this adapter.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

UPSTREAM = 'fb2075e2a3272be372016580c0b6ebd6120228f0'
REVISION = 'moxian_wuji2_geort_v2_mcp'
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
ROOT = Path(__file__).resolve().parents[3]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def fk_cache_key(config):
    key=config.get('fk_cache_key',digest(config['robot']))
    if not isinstance(key,str) or not key or len(key)!=64 or any(c not in '0123456789abcdef' for c in key):
        raise ValueError('无效 FK 缓存指纹')
    return key


def unit(x):
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x)
    if not np.isfinite(n) or n < 1e-8:
        raise ValueError('退化的手掌坐标系')
    return x / n


def palm_basis(normal, forward):
    z = unit(forward)
    x = unit(np.asarray(normal) - z * np.dot(normal, z))
    return np.stack((x, unit(np.cross(z, x)), z), axis=1)


def anatomical_basis(landmarks):
    """Official GeoRT canonical axes: wrist, middle MCP, index/ring MCPs.

    Palm display normals can follow bent distal pads and must not define the
    retargeting frame. Fixed MCP landmarks make this independent of finger curl.
    """
    p = np.asarray(landmarks, dtype=float)
    z = unit(p[9]-p[0])
    x = unit(np.cross(unit(p[5]-p[13]), z))
    return np.stack((x, unit(np.cross(z, x)), z), axis=1)


class SourceAdapter:
    def __init__(self, topology):
        if topology.get('surface_backend') != 'qnbot' or topology.get('coordinate_frame') != 'wrist_local':
            raise ValueError('GeoRT 需要 Moxian 标定后的腕部局部坐标')
        self.topology = topology
        chains = topology['joint_chains']
        self.indices = [chains[0][0]] + [int(i) for c in chains for i in c[-4:]]
        self.mirror = np.eye(3)
        view = topology['palm_view']
        if topology['hand_side'] == 'left':
            lateral = unit(np.cross(view['forward'], view['normal']))
            self.mirror -= 2 * np.outer(lateral, lateral)
        if 'rest_joints_m' in topology:
            rest = np.asarray(topology['rest_joints_m'])[self.indices] @ self.mirror.T
            self.basis = anatomical_basis(rest)
        else:
            # Minimal synthetic topologies used by API clients and unit fixtures.
            self.basis = palm_basis(np.asarray(view['normal']) @ self.mirror.T,
                                    np.asarray(view['forward']) @ self.mirror.T)
        self.transform = self.mirror.T @ self.basis
        self.spec = dict(revision=REVISION, upstream=UPSTREAM, units='m', finger_order=list(FINGERS),
                         hand_side=topology['hand_side'], geometry_version=topology['geometry_version'],
                         model_hash=topology.get('model_hash'), transform=self.transform.tolist(), indices=self.indices)

    def points(self, joints):
        j = np.asarray(joints, dtype=float)
        if j.ndim != 2 or j.shape[1] != 3 or len(j) <= max(self.indices) or not np.isfinite(j).all():
            raise ValueError('无效的 Moxian 骨架')
        landmarks = j[self.indices]
        return ((landmarks[[4, 8, 12, 16, 20]] - landmarks[0]) @ self.transform).astype(np.float32)

    def identity(self, serial, frame):
        if (frame.get('geometry_version') != self.spec['geometry_version'] or
                frame.get('hand_side') != self.spec['hand_side'] or
                frame.get('model_hash') != self.spec['model_hash']):
            raise ValueError('Moxian 手侧或标定几何已变化')
        return dict(self.spec, serial=serial, calibration_fingerprint=frame.get('calibration_fingerprint'),
                    skeleton_runtime_revision=frame.get('skeleton_runtime_revision'))


class RobotAdapter:
    def __init__(self):
        import mujoco
        import xml.etree.ElementTree as ET
        from retarget.env.grasp.hand import hand_tree, HAND_BODIES, asset_root
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(hand_tree('wuji2'), encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        self.joint_names = [self.model.joint(i).name for i in range(6, self.model.njnt)]
        if self.model.nq != 26 or len(self.joint_names) != 20:
            raise ValueError('GeoRT 需要 Wuji2 五指20自由度模型')
        self.lower = self.model.jnt_range[6:, 0].astype(np.float32)
        self.upper = self.model.jnt_range[6:, 1].astype(np.float32)
        self.tip_ids = [self.model.site('right_' + f + '_tip').id for f in FINGERS]
        self.body_ids = [self.model.body(n).id for n in HAND_BODIES]
        mujoco.mj_kinematics(self.model, self.data)
        palm = self.data.xpos[self.body_ids[0]]
        landmarks = np.zeros((21,3));landmarks[0] = palm
        for i in range(5):landmarks[1+4*i:5+4*i] = self.data.xpos[self.body_ids[1+4*i:5+4*i]]
        self.basis = anatomical_basis(landmarks)
        h = hashlib.sha256()
        for p in sorted(asset_root('wuji2').rglob('*')):
            if p.is_file() and p.suffix.lower() in ('.xml', '.stl', '.json'):
                h.update(str(p.relative_to(asset_root('wuji2'))).encode()); h.update(p.read_bytes())
        self.spec = dict(revision=REVISION, asset_sha256=h.hexdigest(), joint_names=self.joint_names,
                         lower=self.lower.tolist(), upper=self.upper.tolist(), basis=self.basis.tolist(),
                         keypoint_joints=[list(range(i*4, i*4+4)) for i in range(5)])
        self.key = digest(self.spec)

    def fk(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (20,) or not np.isfinite(q).all():
            raise ValueError('需要20个有限关节角')
        self.data.qpos[:6] = 0
        self.data.qpos[6:] = q
        self.mj.mj_kinematics(self.model, self.data)
        palm = self.data.xpos[self.body_ids[0]]
        return ((self.data.site_xpos[self.tip_ids] - palm) @ self.basis).astype(np.float32)

    def normalize(self, q):
        return ((q-self.lower)/(self.upper-self.lower)-.5)*2

    def unnormalize(self, q):
        return (q/2+.5)*(self.upper-self.lower)+self.lower
