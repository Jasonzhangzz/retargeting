"""Select calibrated Moxian landmarks; leave all solver preprocessing upstream."""
import numpy as np

from retarget.baseline.wuji_retargeting.reference import REFERENCE_ROOT  # Initialize pinned imports.
from retarget_reference.preprocess import estimate_frame, OPERATOR2MANO_RIGHT, OPERATOR2MANO_LEFT
from retarget_reference._numeric import ordered_matmul

INPUT_REVISION = 'moxian_calibrated_mediapipe21_v1'


class MoxianSkeleton:
    def __init__(self, topology):
        if topology.get('surface_backend') != 'qnbot' or topology.get('coordinate_frame') != 'wrist_local':
            raise ValueError('wuji-retargeting 需要当前已标定的 Moxian 腕部局部骨架')
        chains = topology['joint_chains']
        if (len(chains) != 5 or any(len(c) < 5 for c in chains)
                or len({c[0] for c in chains}) != 1 or topology['hand_side'] not in ('left', 'right')):
            raise ValueError('无效的 Moxian 五指骨架拓扑')
        # MediaPipe order: wrist, then four joints per finger. Moxian's extra
        # non-thumb metacarpal node is omitted; no mirroring/scaling/alignment.
        self.indices = [int(chains[0][0])] + [int(i) for c in chains for i in c[-4:]]
        if min(self.indices) < 0 or len(set(self.indices)) != 21:
            raise ValueError('Moxian 骨架节点不完整')
        self.topology = topology

    def read(self, frame):
        if frame.get('calibrated') is not True or frame.get('preview'):
            raise ValueError('wuji-retargeting 仅接收当前已标定骨架，不能使用预览模型')
        for key in ('hand_side', 'geometry_version', 'binding_version', 'model_hash', 'surface_backend'):
            if frame.get(key) != self.topology.get(key):
                raise ValueError('Moxian 手侧或标定骨架身份已变化: ' + key)
        stamp = float(frame['timestamp_ms'])
        skeleton_stamp = float(frame['skeleton_timestamp_ms'])
        if not np.isfinite([stamp, skeleton_stamp]).all() or abs(stamp-skeleton_stamp) > 25.:
            raise ValueError('Moxian 骨架与源帧时间不匹配')
        joints = np.asarray(frame['joints_m'], dtype=float)
        vertices = np.asarray(frame['vertices_m'], dtype=float)
        if (joints.ndim != 2 or joints.shape[1] != 3 or len(joints) <= max(self.indices)
                or not np.isfinite(joints).all()):
            raise ValueError('无效的 Moxian 骨架')
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError('无效的 Moxian 曲面')
        points = joints[self.indices]
        # Compute the inverse rigid transform for drawing BEFORE advancing the
        # session. Degenerate palms cannot mutate original warm/filter state.
        root_rotation = display_transform(points, self.topology['hand_side'])
        return points, root_rotation, joints, vertices


def display_transform(points, side):
    """Inverse of original WujiHand2 rigid preprocessing, for display only.

    Match the upstream float32 cast, wrist centering and palm-frame routine.
    Pinch only changes target tips and has no inverse rigid display transform.
    """
    p = np.asarray(points).astype(np.float32).astype(np.float64).reshape(21, 3)
    p -= p[0].copy()
    if not np.isfinite(p).all():
        raise ValueError('无效的 Moxian 骨架')
    operator = OPERATOR2MANO_RIGHT if side == 'right' else OPERATOR2MANO_LEFT
    cx, cy, cz = np.cos(np.deg2rad([0., 0., -90.]))
    sx, sy, sz = np.sin(np.deg2rad([0., 0., -90.]))
    rx = np.array([[1., 0., 0.], [0., cx, -sx], [0., sx, cx]])
    ry = np.array([[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]])
    rz = np.array([[cz, -sz, 0.], [sz, cz, 0.], [0., 0., 1.]])
    return ordered_matmul(ordered_matmul(estimate_frame(p), operator),
                          ordered_matmul(ordered_matmul(rz, ry), rx).T)
