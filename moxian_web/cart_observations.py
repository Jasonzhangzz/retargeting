"""Synchronized measured EMF and material taxels; shared train/runtime packing."""
import numpy as np
import torch
from .cart_protocol import FINGERS


def capture_emf(frame, model):
    """Preserve raw measurements AND their calibrated wrist transform."""
    from .qnbot_reference import observations
    with torch.no_grad():
        p, r, valid = observations([frame], model.device)
        supplied=[all(k in (frame.get('fingers') or {}).get(f,{}) for k in ('position','quaternion_xyzw')) for f in FINGERS]
        valid=valid.bool() & torch.tensor([supplied],device=valid.device)
        a, t = model.alignment()
        wp, wr = p@a.T+t, a@r
    return dict(timestamp_ms=float(frame['timestamp_ms']), coordinate_frame='wrist_local',
        positions_m=wp[0].cpu().tolist(), rotations=wr[0].cpu().tolist(), valid=valid[0].bool().cpu().tolist(),
        raw_positions_m=p[0].cpu().tolist(), raw_rotations=r[0].cpu().tolist(),
        calibration_fingerprint=frame.get('calibration_fingerprint'),
        skeleton_runtime_revision=frame.get('skeleton_runtime_revision'))


def canonical_emf(observation, adapter, root, stamp):
    if not observation:
        raise ValueError('缺少原始 EMF 完整位姿；请关联同会话原始录制或重新采集')
    if observation.get('coordinate_frame') != 'wrist_local' or abs(float(observation['timestamp_ms'])-stamp) > 25:
        raise ValueError('EMF 与骨架不同步或坐标系错误')
    p = np.asarray(observation['positions_m'], np.float32)
    r = np.asarray(observation['rotations'], np.float32)
    valid = np.asarray(observation['valid'], bool)
    rawp = np.asarray(observation.get('raw_positions_m'), np.float32)
    rawr = np.asarray(observation.get('raw_rotations'), np.float32)
    if p.shape != (5,3) or r.shape != (5,3,3) or valid.shape != (5,) or rawp.shape != (5,3) or rawr.shape != (5,3,3):
        raise ValueError('EMF 必须包含五指实测位置、旋转和有效性')
    if not np.isfinite(p[valid]).all() or not np.isfinite(r[valid]).all() or not np.isfinite(rawp[valid]).all() or not np.isfinite(rawr[valid]).all():
        raise ValueError('Nonfinite EMF measurement')
    if valid.any() and (not np.allclose(r[valid].transpose(0,2,1)@r[valid],np.eye(3),atol=1e-3) or not np.allclose(np.linalg.det(r[valid]),1,atol=1e-3)):
        raise ValueError('Invalid EMF rotation')
    a = adapter.transform
    p = (np.where(valid[:,None],p,0)-root)@a
    # Change both world and receiver reference axes. Conjugation preserves
    # SO(3), including the left-hand reflection, rather than encoding det=-1.
    r = a.T@np.where(valid[:,None,None],r,np.eye(3))@a
    pose = np.concatenate((p,r[:,:,:2].transpose(0,2,1).reshape(5,6)), -1)
    return np.where(valid[:,None],pose,0.).astype(np.float32),valid


def tactile_input(frame, positions, topology):
    from .cart_capture_schema import binary_taxels
    state,valid,_=binary_taxels(frame,topology)
    return np.asarray(positions,np.float32),state.astype(np.float32),valid
