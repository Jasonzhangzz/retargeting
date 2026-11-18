"""Fixed orientation coordinates for applying Wuji pose gates to Moxian RXs.

Only the collector uses this adapter. Native fitting and IK always receive the
original measurements, without these gesture-coordinate transforms.
"""
from __future__ import annotations
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from .wuji_native_abi import FINGERS

REVISION='moxian_rx_gesture_frame_v1'


def mean_rotation(frames,finger):
    q=np.asarray([frame['payload']['fingertip_local'][finger]['quaternion_xyzw'] for frame in frames],float)
    q/=np.linalg.norm(q,axis=1,keepdims=True)
    _,vectors=np.linalg.eigh(q.T@q)
    return Rotation.from_quat(vectors[:,-1]).as_matrix()


def from_capture(document,fingerprint,side):
    payload=document['payload']
    if payload['side']!=side:raise ValueError('动作方向参考手侧不匹配')
    groups={g['stage_id']:g['frames'] for g in payload['stage_samples']}
    if not groups.get('open') or not groups.get('mcp90'):raise ValueError('缺少展平或弯曲方向参考')
    flat=[];basis=[];angles=[]
    target=np.array([-1. if side=='right' else 1.,0.,0.])
    for finger in FINGERS[1:]:
        f=mean_rotation(groups['open'],finger);b=mean_rotation(groups['mcp90'],finger)
        delta=Rotation.from_matrix(b@f.T).as_rotvec();angle=np.linalg.norm(delta)
        if not np.isfinite(angle) or angle<np.deg2rad(15):raise ValueError('八动作弯曲参考不足，无法确定接收器弯曲轴')
        # Deterministic minimal rotation mapping the measured bend axis to the
        # original Wuji roll axis. No angle is rescaled to 90 degrees.
        axis=delta/angle;cross=np.cross(axis,target);dot=float(np.clip(axis@target,-1.,1.))
        if dot<-.999999999:
            helper=np.eye(3)[np.argmin(np.abs(axis))];cross=np.cross(axis,helper);cross/=np.linalg.norm(cross)
            c=Rotation.from_rotvec(np.pi*cross).as_matrix()
        elif np.linalg.norm(cross)<1e-12:c=np.eye(3)
        else:c=Rotation.from_rotvec(np.arctan2(np.linalg.norm(cross),dot)*cross/np.linalg.norm(cross)).as_matrix()
        flat.append(f.tolist());basis.append(c.tolist());angles.append(float(np.rad2deg(angle)))
    return dict(revision=REVISION,calibration_fingerprint=fingerprint,source_payload_sha256=document['payload_sha256'],
                side=side,flat_rotations=flat,roll_basis=basis,reference_bend_deg=angles,
                scope='pose_gate_only_native_fit_and_ik_use_raw_emf')


def from_official(home,fingerprint,side):
    from .shared_capture import official_capture_document
    path,document=official_capture_document(home,fingerprint)
    result=from_capture(document,fingerprint,side);result['source_file']=str(path);return result


def orient_for_gate(fingers,reference):
    if reference.get('revision')!=REVISION:raise ValueError('动作方向参考版本无效')
    flat=np.asarray(reference['flat_rotations']);basis=np.asarray(reference['roll_basis'])
    if flat.shape!=(4,3,3) or basis.shape!=(4,3,3) or not np.isfinite(np.r_[flat.ravel(),basis.ravel()]).all():raise ValueError('动作方向参考无效')
    result=copy.deepcopy(fingers)
    for i,finger in enumerate(FINGERS[1:]):
        current=Rotation.from_quat(fingers[finger]['quaternion_xyzw']).as_matrix()
        canonical=basis[i]@(current@flat[i].T)@basis[i].T
        result[finger]['quaternion_xyzw']=Rotation.from_matrix(canonical).as_quat().tolist()
    return result
