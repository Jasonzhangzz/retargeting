"""Unmodified official SDK human IK with explicit confidence and measured-EMF FK checks."""
from functools import lru_cache
from pathlib import Path
import time
import h5py
import numpy as np
from build_wuji_real_sdk_dataset import digest, stopped
from retarget.learning.wuji_ik.features import wrist_emf_features,wrist_tx_transform,urdf_limits
from retarget.learning.wuji_ik.quality_data import sdk_quality_mask

NAMES=('wrist','thumb_cmc','thumb_mcp','thumb_ip','thumb_tip')+tuple(
    f+'_'+j for f in ('index_finger','middle_finger','ring_finger','pinky') for j in ('mcp','pip','dip','tip'))
QUALITY_KEYS=('ik_confidence','skeleton_confidence','fk_position_error_m','fk_orientation_error_rad')


@lru_cache(maxsize=64)
def human_model(side,path,sha):
    from human_ik_reference.kinematics import HumanHandKinematics
    if digest(path)!=sha:raise ValueError('Calibration changed')
    return HumanHandKinematics(side,path,sdk_tx_rounding=True)


def solve(root,w,retarget=False):
    import wuji_sdk as sdk
    from select_wuji_clean_trajectories import receiver_fk_batch,quaternion_matrix
    if sdk.__version__!='2026.8.31':raise ValueError('Unexpected SDK version')
    sdk.set_log_level('error');stopped(root);side=w['side'];cal=w['calibration'];rows=slice(w['start'],w['end'])
    with h5py.File(w['source_h5'],'r',locking=False) as h:
        g=h[side];d=dict(position_tx_m=g['emf_poses/position'][rows],orientation_xyzw=g['emf_poses/orientation_xyzw'][rows],
            confidence=g['emf_poses/confidence'][rows],timestamp_ns=g['source_timestamp_ns'][rows],source_seq=g['source_seq'][rows])
    pipe=sdk.WujiGlove.offline_pipeline(sn=cal['sn'],hand_side=side,urdf_path=cal['urdf'] if cal['override'] else None)
    if pipe.urdf_source!=('override' if cal['override'] else 'builtin_default'):raise ValueError('Unexpected SDK calibration source')
    session=sdk.RetargetSession.for_hand(sdk.HandModel.WujiHand2,side=sdk.Handedness.Left if side=='left' else sdk.Handedness.Right) if retarget else None
    human=[];ik=[];sk=[];points=[];commands=[]
    for i in range(len(d['confidence'])):
        if i%128==0:stopped(root)
        frame=sdk.EmfPoseArray(header=sdk.FrameHeader(seq=int(d['source_seq'][i]),timestamp_us=int(d['timestamp_ns'][i]//1000),frame_id=side[0]+'_hand_emf_tx'),
            poses=[sdk.EmfPose(pose=sdk.Pose(position=d['position_tx_m'][i,f].tolist(),orientation=sdk.Quaternion(*d['orientation_xyzw'][i,f].tolist())),confidence=float(d['confidence'][i,f])) for f in range(5)])
        skeleton=pipe.hand_skeleton().compute(frame);by_name={j.name:j for j in skeleton.joints}
        pts=np.asarray([by_name[n].pose.position for n in NAMES],dtype=np.float32);points.append(pts)
        if session is not None:commands.append(session.step(pts))
        angles=pipe.hand_joint_angles().compute(frame)
        human.append(np.concatenate([np.asarray(f.angles)[:5 if k==0 else 4] for k,f in enumerate(angles.fingers)]))
        ik.append([f.confidence for f in angles.fingers]);sk.append([by_name[n].confidence for n in NAMES])
    human=np.asarray(human);points=np.asarray(points)
    if not np.isfinite(human).all() or not np.isfinite(points).all():raise ValueError('Non-finite official SDK output')
    model=human_model(side,cal['urdf'],digest(cal['urdf']));p,r=receiver_fk_batch(model,human)
    pe=np.linalg.norm(p-d['position_tx_m'],axis=-1);relative=r.swapaxes(-1,-2)@quaternion_matrix(d['orientation_xyzw'])
    re=np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)*.5,-1,1))
    for i in sorted({0,len(human)//2,len(human)-1}):
        if abs(model.forward(human[i])['skeleton'][:,:3,3]-points[i]).max()>2e-6:raise ValueError('Independent FK differs from SDK geometry')
    d.update(human_q_rad=human,ik_confidence=np.asarray(ik),skeleton_confidence=np.asarray(sk),fk_position_error_m=pe,fk_orientation_error_rad=re,skeleton_sdk_m=points,
        features=wrist_emf_features(d['position_tx_m'],d['orientation_xyzw'],wrist_tx_transform(cal['urdf'])),frame_id=np.arange(w['start'],w['end'],dtype=np.int64))
    if session is not None:d['q_sdk_rad']=np.asarray(commands)
    return d


def valid(d):
    return sdk_quality_mask(*(d[k] for k in QUALITY_KEYS)) & (d['confidence']>=.8).all(1)


def retarget_prefix(root,w,d):
    """Prefix of a fresh certified human trajectory retains exactly the same history."""
    import wuji_sdk as sdk
    n=w['end']-w['start'];d={k:np.asarray(v[:n]) for k,v in d.items()}
    if not valid(d).all():raise ValueError('Uncertified teacher input')
    side=w['side'];session=sdk.RetargetSession.for_hand(sdk.HandModel.WujiHand2,side=sdk.Handedness.Left if side=='left' else sdk.Handedness.Right)
    commands=[]
    for i,points in enumerate(d['skeleton_sdk_m']):
        if i%128==0:stopped(root)
        commands.append(session.step(points))
    q=np.asarray(commands);limits=urdf_limits(root/'assets'/f'robot_{side}.urdf',side)
    if not np.isfinite(q).all() or (q<limits[:,0]-1e-5).any() or (q>limits[:,1]+1e-5).any():raise ValueError('Invalid SDK q; labels are never clipped')
    d.update(q_sdk_rad=q,q_normalized=(2*(q-limits[:,0])/(limits[:,1]-limits[:,0])-1).astype(np.float32))
    return d


def retarget_history(root,w,d):
    """Retarget all real EMF history, then retain only qualified supervision rows."""
    import wuji_sdk as sdk
    side=w['side'];session=sdk.RetargetSession.for_hand(sdk.HandModel.WujiHand2,side=sdk.Handedness.Left if side=='left' else sdk.Handedness.Right)
    begin=w['start']-w['context_start'];end=w['end']-w['context_start'];commands=[]
    for i,points in enumerate(d['skeleton_sdk_m'][:end]):
        if i%128==0:stopped(root)
        commands.append(session.step(points))
    q=np.asarray(commands)[begin:end];selected={k:v[begin:end] for k,v in d.items()}
    if not valid(selected).all():raise ValueError('Unqualified supervision row')
    limits=urdf_limits(root/'assets'/f'robot_{side}.urdf',side)
    if not np.isfinite(q).all() or (q<limits[:,0]-1e-5).any() or (q>limits[:,1]+1e-5).any():raise ValueError('Invalid SDK q')
    selected.update(q_sdk_rad=q,q_normalized=(2*(q-limits[:,0])/(limits[:,1]-limits[:,0])-1).astype(np.float32),context_features=d['features'][:begin])
    return selected
