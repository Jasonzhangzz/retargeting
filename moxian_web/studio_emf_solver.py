"""Compiled Studio-style bounded QP on the unchanged Moxian 25/29-DoF rig.

This is a native implementation of the recovered Studio algorithm adapted to
Moxian geometry/weights, not the official fixed-21-DoF SDK binary. The saved
model, calibration, frame-level motion bounds and previous-pose state remain
the same. All iterative FK, Jacobians and QP operations execute in C++.
"""
from __future__ import annotations
import ctypes as C
import hashlib
from pathlib import Path
import subprocess
import time
import uuid

import numpy as np
from scipy.spatial.transform import Rotation
from .bone_extrinsics import BoneKinematics,FINGERS,SCHEMA,validate_candidate

REVISION='moxian_studio_native_qp_v1'
DP=C.POINTER(C.c_double);IP=C.POINTER(C.c_int)


def native_library():
    root=Path(__file__).resolve().parents[1];source=root/'moxian_web/native/studio_emf_qp.cpp'
    checksum=hashlib.sha256(source.read_bytes()).hexdigest()
    folder=root/'.cache/moxian-studio-qp';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(checksum[:24]+'.so')
    if not path.exists():
        temporary=folder/(checksum[:24]+'.'+uuid.uuid4().hex+'.so')
        try:
            result=subprocess.run(['g++','-O3','-DNDEBUG','-DEIGEN_DONT_PARALLELIZE','-std=c++17','-shared','-fPIC',
                '-I/usr/include/eigen3',str(source),'-o',str(temporary)],capture_output=True,text=True,timeout=90)
            if result.returncode:raise RuntimeError('Native Moxian solver build failed: '+result.stderr[-5000:])
            temporary.replace(path)
        finally:temporary.unlink(missing_ok=True)
    lib=C.CDLL(str(path))
    lib.studio_qp_create.argtypes=[C.c_int,C.c_int,C.c_int,IP,IP,DP,DP,DP,DP,IP,DP,DP,DP,DP,DP,IP,DP]
    lib.studio_qp_create.restype=C.c_void_p
    lib.studio_qp_destroy.argtypes=[C.c_void_p]
    lib.studio_qp_step.argtypes=[C.c_void_p,DP,DP,C.c_int,C.c_double,DP,DP,DP,DP];lib.studio_qp_step.restype=C.c_int
    lib.studio_qp_forward.argtypes=[C.c_void_p,DP,DP,DP];lib.studio_qp_forward.restype=C.c_int
    lib.studio_qp_sensor.argtypes=[C.c_void_p,C.c_int,DP,DP,DP,DP,DP];lib.studio_qp_sensor.restype=C.c_int
    return lib,checksum,path


class NativeStudioQP:
    def __init__(self,definition):
        self.kinematics=BoneKinematics(definition);k=self.kinematics;r=k.rig;c=definition['solver_config']
        self.lib,self.source_sha256,self.path=native_library();self.nq=len(k.joint_names);self.nb=len(k.body_names)
        jointbody=np.empty(self.nq,np.int32)
        for b,joints in enumerate(r['body_joints']):jointbody[joints]=b
        pos=k.body_p0.copy()
        for b in range(1,self.nb):pos[b]*=k.scales[r['body_bone'][r['parents'][b]]]
        periodic=set(definition.get('joint_limit_policy',{}).get('finger_axial_twist',{}).get('joint_names',[])) if definition.get('joint_limit_policy',{}).get('finger_axial_twist',{}).get('periodic') else set()
        posture=np.zeros(self.nq);temporal=np.full(self.nq,c['history_weight'])
        for f,ids in enumerate(k.finger_indices):
            posture[ids]=c['posture_weight']*(c['thumb_posture_scale'] if f==0 else 1.)
            for j in ids:
                if k.joint_names[j].endswith('_mcp_twist'):posture[j]*=c.get('root_twist_posture_scale',.01)
                if k.joint_names[j].endswith(('_thumb_cmc_flexion','_thumb_mcp_abduction')):
                    posture[j]=c.get('thumb_palm_posture_weight',c['posture_weight']*c['thumb_posture_scale'])
                    temporal[j]=c.get('thumb_palm_history_weight',c['history_weight'])
        def d(value):return np.ascontiguousarray(value,dtype=np.float64)
        def i(value):return np.ascontiguousarray(value,dtype=np.int32)
        arrays=[i(r['parents']),i(jointbody),d(pos),d(k.body_r0),d(k.joint_axis0),d(k.joint_p0),
            i(r['distal_bodies']),d(k.mounts),d(k.limits),d(definition['rest_angles_rad']),d(posture),d(temporal),
            i([name in periodic for name in k.joint_names]),
            d([c['v_max'],c['lm_damping'],c['global_damping'],c['limit_gain'],0.,c['position_sigma_m'],
               c['rotation_sigma_rad'],c['max_iterations'],c['retry_iterations'],c['retry_position_m'],c['retry_rotation_rad']])]
        pointers=[a.ctypes.data_as(IP if a.dtype==np.int32 else DP) for a in arrays]
        self.handle=self.lib.studio_qp_create(self.nb,self.nq,r['wrist_body'],*pointers)
        if not self.handle:raise RuntimeError('Native Moxian model initialization failed')

    def solve(self,initial,positions,rotations,dt,history):
        if not self.handle:raise RuntimeError('Native Moxian solver is closed')
        q=np.ascontiguousarray(initial,dtype=np.float64);p=np.ascontiguousarray(positions,dtype=np.float64);r=np.ascontiguousarray(rotations,dtype=np.float64)
        if q.shape!=(self.nq,) or p.shape!=(5,3) or r.shape!=(5,3,3) or not np.isfinite(np.r_[q,p.ravel(),r.ravel()]).all():raise ValueError('Invalid native Moxian observation')
        result=np.empty(self.nq);reports=np.empty((5,5))
        status=self.lib.studio_qp_step(self.handle,q.ctypes.data_as(DP),q.ctypes.data_as(DP),int(history),dt,
            p.ctypes.data_as(DP),r.ctypes.data_as(DP),result.ctypes.data_as(DP),reports.ctypes.data_as(DP))
        if status!=0:raise RuntimeError('Native Moxian QP failed: '+str(status))
        return result,reports

    def forward(self,q):
        if not self.handle:raise RuntimeError('Native Moxian solver is closed')
        q=np.ascontiguousarray(q,dtype=np.float64);p=np.empty((self.nb,3));r=np.empty((self.nb,3,3))
        if q.shape!=(self.nq,) or not np.isfinite(q).all():raise ValueError('Invalid native Moxian angles')
        status=self.lib.studio_qp_forward(self.handle,q.ctypes.data_as(DP),p.ctypes.data_as(DP),r.ctypes.data_as(DP))
        if status:raise RuntimeError('Native Moxian FK failed')
        return r,p

    def sensor(self,f,q):
        if not self.handle:raise RuntimeError('Native Moxian solver is closed')
        q=np.ascontiguousarray(q,dtype=np.float64);n=len(self.kinematics.finger_indices[f])
        p=np.empty(3);r=np.empty((3,3));jp=np.empty((3,n));jw=np.empty((3,n))
        status=self.lib.studio_qp_sensor(self.handle,f,q.ctypes.data_as(DP),p.ctypes.data_as(DP),r.ctypes.data_as(DP),jp.ctypes.data_as(DP),jw.ctypes.data_as(DP))
        if status:raise RuntimeError('Native Moxian sensor FK failed')
        return r,p,jp,jw

    def close(self):
        if getattr(self,'handle',None):self.lib.studio_qp_destroy(self.handle);self.handle=None

    def __del__(self):self.close()


class StudioQPSkeleton:
    input_registration=None
    def __init__(self,candidate):
        validate_candidate(candidate);self.candidate=candidate;self.definition=candidate['bone_extrinsics']
        self.native=NativeStudioQP(self.definition);self.kinematics=self.native.kinematics;self.config=self.definition['solver_config']
        self.model_hash=candidate['model_hash'];self.personalized=dict(geometry_hash=self.model_hash);self.reset()

    def reset(self):self.previous=None;self.previous_timestamp=None;self.frames=0

    def close(self):self.native.close()

    def policy(self,original=None):
        return dict(original or {},solver=REVISION,solver_parameters=self.config,solver_dof=self.native.nq,
            mode='moxian_compiled_studio_style',revision=REVISION,local_ik_modifications=True,joint_smoother=False,open_error_reseed=False,
            personalized_geometry_hash=self.model_hash,runtime_policy_sha256=self.model_hash,
            native_solver=True,official_sdk_binary=False,implementation='compiled_studio_style_bounded_qp',
            reference='Wuji Studio 2026.7.1 per-finger GN / active-set Cholesky box QP',
            source_sha256=self.native.source_sha256,shared_frame_motion_budget=True,
            native_angles_used=False,physics_enabled=False,schema_revision=SCHEMA,
            joint_limit_policy=self.definition.get('joint_limit_policy'))

    def step(self,frame):
        began=time.perf_counter();stamp=float(frame['timestamp_ms'])
        if not np.isfinite(stamp):raise ValueError('无效时间戳')
        interval=None if self.previous_timestamp is None else (stamp-self.previous_timestamp)/1000
        if interval is not None and interval<=0:raise ValueError('时间戳必须递增')
        if interval is not None and interval>.25:self.reset();interval=None
        points=[];rotations=[]
        for f in FINGERS:
            node=frame['fingers'][f];p=np.asarray(node['position'],float);q=np.asarray(node['quaternion_xyzw'],float)
            if p.shape!=(3,) or q.shape!=(4,) or not np.isfinite(np.r_[p,q]).all() or np.linalg.norm(q)<1e-8 or node.get('valid') is False or node.get('confidence',1)<=0:raise ValueError('无效EMF观测')
            points.append(p);rotations.append(Rotation.from_quat(q).as_matrix())
        k=self.kinematics;align=k.alignment;points=np.asarray(points)@align[:3,:3].T+align[:3,3];rotations=align[:3,:3]@np.asarray(rotations)
        first=self.previous is None;start=np.asarray(self.definition['rest_angles_rad']).copy() if first else self.previous.copy()
        solved_at=time.perf_counter();q,raw=self.native.solve(start,points,rotations,1/60 if interval is None else min(interval,1/15),not first)
        finished=time.perf_counter();r,p=self.native.forward(q)
        from .skeleton import JOINT_NAMES
        ids=[k.body_names.index(k.rig['side']+'_'+name) for name in JOINT_NAMES]
        reports=[dict(finger=f,position_error_mm=float(v[0]*1000),rotation_error_deg=float(np.rad2deg(v[1])),
            iterations=int(v[2]),cost=float(v[3]),success=bool(v[4]),tracking_satisfied=bool(v[0]<=.015 and v[1]<=np.deg2rad(15))) for f,v in zip(FINGERS,raw)]
        pose=dict(timestamp_ms=stamp,hand_side=k.rig['side'],coordinate_frame='wrist_local',names=list(JOINT_NAMES),
            joint_names=k.joint_names,joint_positions_rad=q.tolist(),positions_local=p[ids].tolist(),orientations_local_xyzw=Rotation.from_matrix(r[ids]).as_quat().tolist(),
            ik_solver=REVISION,personalized_geometry_hash=self.model_hash,model_hash=self.model_hash,
            tracking_errors={v['finger']:dict(position_error_mm=v['position_error_mm'],rotation_error_deg=v['rotation_error_deg'],satisfied=v['tracking_satisfied']) for v in reports})
        maximum=0. if first else float(np.rad2deg(abs(q-start).max()));self.previous=q.copy();self.previous_timestamp=stamp;self.frames+=1
        from .native_skeleton import joint_limit_diagnostics
        periodic=self.definition.get('joint_limit_policy',{}).get('finger_axial_twist',{})
        excluded=set(periodic.get('joint_names',[])) if periodic.get('periodic') else set()
        limits={n:v for n,v in zip(k.joint_names,k.limits.tolist()) if n not in excluded}
        return dict(pose=pose,targets=dict(timestamp_ms=stamp,hand_side=k.rig['side'],coordinate_frame='wrist_local',names=list(FINGERS),
            positions_local=points.tolist(),rotations_local=rotations.reshape(5,9).tolist(),quaternions_local_xyzw=Rotation.from_matrix(rotations).as_quat().tolist(),orientation_convention='EMF_RX',registration=REVISION),
            physics=dict(enabled=False,steps=0),joint_limit_diagnostics=joint_limit_diagnostics(k.joint_names,q,limits),
            continuity=dict(engine_frames=self.frames,reused_engine=not first,interval_ms=None if interval is None else interval*1000,max_joint_step_deg=maximum),
            ik_diagnostics=dict(algorithm=REVISION,model_hash=self.model_hash,native_solver=True,official_sdk_binary=False,
                previous_pose_initial_guess=not first,native_solve_ms=(finished-solved_at)*1000,fingers=reports,processing_ms=(time.perf_counter()-began)*1000))
