"""Explicit adapter for recorded SDK measurements; absent world data stays absent.

SDK IK is derived from EMF and must never be treated as an independent sensor.
Positions are metres, orientations XYZW, and timestamps seconds in source clock.
"""
from dataclasses import dataclass
import numpy as np
from .qnbot_surface import FINGERS


@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    quaternion_xyzw: np.ndarray
    valid: bool


@dataclass(frozen=True)
class Frame:
    timestamp: float
    T_world_wrist: np.ndarray | None
    emf_tip_pose: dict[str,Pose]
    joint_measurements: dict | None
    tactile_values: np.ndarray
    object_pose: np.ndarray | None
    side: str
    tactile_valid: np.ndarray
    tactile_timestamp: float | None
    coordinate_frame: str
    joint_measurements_source: str


def transform(value):
    if value is None:return None
    t=np.asarray(value,dtype=float)
    if t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1]):raise ValueError('无效刚体变换')
    r=t[:3,:3]
    if not np.allclose(r.T@r,np.eye(3),atol=1e-4) or not np.isclose(np.linalg.det(r),1,atol=1e-4):raise ValueError('变换旋转必须正交且右手')
    return t.copy()


def adapt_frame(row):
    stamp=row.get('timestamp_ms');side=row.get('hand_side')
    if not isinstance(stamp,(int,float)) or not np.isfinite(stamp):raise ValueError('记录缺少有效EMF时间戳')
    if side not in ('left','right'):raise ValueError('记录缺少明确手侧')
    tips={}
    for finger in FINGERS:
        node=(row.get('fingers') or {}).get(finger) or {}
        p=np.asarray(node.get('position',[0,0,0]),float);q=np.asarray(node.get('quaternion_xyzw',[0,0,0,1]),float)
        valid=bool(p.shape==(3,) and q.shape==(4,) and np.isfinite(p).all() and np.isfinite(q).all() and np.linalg.norm(q)>.1 and node.get('valid') is not False and node.get('confidence',1)>0 and bool(node))
        tips[finger]=Pose(p.copy() if valid else np.zeros(3),q/np.linalg.norm(q) if valid else np.array([0.,0,0,1]),valid)
    contact=row.get('contact') or {};values=np.asarray(contact.get('normalized') or [],dtype=float)
    if values.ndim!=1:raise ValueError('触觉数据必须是一维taxel数组')
    tstamp=row.get('tactile_timestamp_ms');tstamp=float(tstamp)/1000 if isinstance(tstamp,(int,float)) and np.isfinite(tstamp) else None
    fresh=contact.get('ready') is True and tstamp is not None and abs(tstamp-stamp/1000)<=.025
    valid=np.isfinite(values)&fresh;values=np.where(np.isfinite(values),values,0.)
    skeleton=row.get('skeleton') or {};names=skeleton.get('joint_names');angles=skeleton.get('joint_positions_rad')
    joints=None
    if names and angles is not None:
        if len(names)!=len(angles) or not np.isfinite(angles).all():raise ValueError('骨架关节记录无效')
        joints=dict(zip(names,map(float,angles)))
    return Frame(float(stamp)/1000,transform(row.get('T_world_wrist')),tips,joints,values,transform(row.get('object_pose')),side,valid,tstamp,
                 row.get('emf_coordinate_frame','sdk_emf_measurement'), 'emf_derived_ik' if joints else 'unavailable')


def evidence_summary(rows):
    frames=[adapt_frame(r) for r in rows]
    dt=np.diff([f.timestamp for f in frames]);dt=dt[(dt>0)&(dt<.25)]
    return dict(frame_count=len(frames),position_unit='metre',orientation='quaternion_xyzw',timestamps='source_timestamp_ms_converted_to_seconds',
                emf_coordinate_frames=sorted({f.coordinate_frame for f in frames}),synchronization_tolerance_ms=25,
                taxel_counts=sorted({len(f.tactile_values) for f in frames}),frames_with_fresh_tactile=sum(bool(f.tactile_valid.any()) for f in frames),
                frames_with_world_wrist=sum(f.T_world_wrist is not None for f in frames),frames_with_object_pose=sum(f.object_pose is not None for f in frames),
                emf_valid_counts={finger:sum(f.emf_tip_pose[finger].valid for f in frames) for finger in FINGERS},
                median_recorded_frame_interval_ms=float(np.median(dt)*1000) if len(dt) else None,
                joint_measurements_source='EMF-derived IK, pose seed only; not independent joint supervision',
                tactile_binding='fixed anatomical initialization, not manufacturer measured taxel coordinates')
