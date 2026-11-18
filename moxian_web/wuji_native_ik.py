"""Actual Wuji SDK IK (21 DoF) with a deterministic display-only adapter."""
from __future__ import annotations
import json,os,select,subprocess,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .wuji_native_abi import FINGERS
from .wuji_native_model import NativeURDF,display_pose,validate_native_candidate
from .bone_extrinsics import atomic_json


class WujiNativeSkeleton:
    input_registration=None
    def __init__(self,candidate):
        validate_native_candidate(candidate);self.candidate=candidate;self.model_hash=candidate['model_hash'];self.personalized=dict(geometry_hash=self.model_hash)
        self.urdf=NativeURDF(candidate['wuji_native']['profiles']['wujihand']['wrist_urdf']);self.c=np.asarray(candidate['display_adapter']['rotation']);self.previous_timestamp=None;self.frames=0
        root=Path(__file__).resolve().parents[1];home=root/'output/runtime/wuji-offline-models'/self.model_hash;home.mkdir(parents=True,exist_ok=True)
        self.path=home/'hand.urdf'
        if not self.path.exists():
            temp=home/'hand.pending';temp.write_text(candidate['wuji_native']['profiles']['wujihand']['wrist_urdf']);temp.replace(self.path)
        read,write=os.pipe()
        try:
            self.proc=subprocess.Popen(['/usr/bin/python3.12','-u',str(root/'moxian_web/wuji_offline_child.py'),'--event-fd',str(write)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,text=True,bufsize=1,pass_fds=(write,))
        finally:os.close(write)
        self.reader=os.fdopen(read,'r',buffering=1);self.reset()

    def request(self,data,timeout=5.):
        if self.proc.poll() is not None:raise RuntimeError('Wuji原生IK子进程已退出')
        self.proc.stdin.write(json.dumps(data,allow_nan=False)+'\n');self.proc.stdin.flush()
        if not select.select([self.reader],[],[],timeout)[0]:raise TimeoutError('Wuji原生IK响应超时')
        result=json.loads(self.reader.readline())
        if result.get('error'):raise ValueError('Wuji原生IK：'+result['error'])
        return result

    def reset(self):
        self.request(dict(action='load',serial='MOXIAN-'+self.model_hash[:20],side=self.candidate['side'],urdf_path=str(self.path)),timeout=15.)
        self.previous_timestamp=None;self.frames=0

    def policy(self,original=None):
        return dict(original or {},solver='wuji_sdk_native21',solver_dof=21,native_solver=True,
            personalized_geometry_hash=self.model_hash,runtime_policy_sha256=self.model_hash,
            physics_enabled=False,native_angles_used=True,urdf_source='override',sdk_version='2026.7.1')

    def step(self,frame):
        start=time.perf_counter();stamp=float(frame['timestamp_ms'])
        if not np.isfinite(stamp):raise ValueError('无效EMF时间戳')
        if self.previous_timestamp is not None:
            if stamp<=self.previous_timestamp:raise ValueError('时间戳必须递增')
            if stamp-self.previous_timestamp>250:self.reset()
        nodes=[frame['fingers'][f] for f in FINGERS]
        for n in nodes:
            q=np.asarray(n['quaternion_xyzw']);p=np.asarray(n['position'])
            if q.shape!=(4,) or p.shape!=(3,) or not np.isfinite(np.r_[q,p]).all() or np.linalg.norm(q)<1e-8 or n.get('valid') is False:
                self.reset();raise ValueError('原生IK需要五指有效EMF观测')
        result=self.request(dict(timestamp_ms=stamp,poses=nodes))
        if len(result['angles'])!=5 or any(len(v)!=5 for v in result['angles']) or any(abs(v[4])>1e-12 for v in result['angles'][1:]):raise ValueError('固定SDK的21自由度打包格式发生变化')
        angles=np.concatenate([result['angles'][0],*[v[:4] for v in result['angles'][1:]]])
        native=self.urdf.forward(angles);r,p=display_pose(self.candidate,native);a=self.candidate['display_adapter']
        from .skeleton import JOINT_NAMES
        ids=[a['body_names'].index(self.candidate['side']+'_'+name) for name in JOINT_NAMES]
        tx=native['emf_tx_base'];points=np.array([n['position'] for n in nodes]);rot=Rotation.from_quat([n['quaternion_xyzw'] for n in nodes]).as_matrix()
        target_native=points@tx[:3,:3].T+tx[:3,3];target_rot=tx[:3,:3]@rot
        predicted=np.array([native[f+'_rx_coil'][:3,3] for f in FINGERS]);pred_r=np.array([native[f+'_rx_coil'][:3,:3] for f in FINGERS])
        errors=np.linalg.norm(predicted-target_native,axis=-1)*1000;rotation_errors=np.rad2deg(Rotation.from_matrix(pred_r.transpose(0,2,1)@target_rot).magnitude())
        reports=[dict(finger=f,position_error_mm=float(errors[i]),rotation_error_deg=float(rotation_errors[i]),confidence=result['confidence'][i],tracking_satisfied=result['confidence'][i]>0) for i,f in enumerate(FINGERS)]
        pose=dict(timestamp_ms=stamp,hand_side=self.candidate['side'],coordinate_frame='wrist_local',names=list(JOINT_NAMES),
            positions_local=p[ids].tolist(),orientations_local_xyzw=Rotation.from_matrix(r[ids]).as_quat().tolist(),
            joint_names=self.urdf.names,joint_positions_rad=angles.tolist(),native_dof=21,
            model_hash=self.model_hash,personalized_geometry_hash=self.model_hash,ik_solver='wuji_sdk_native21',
            display_body_positions=p.tolist(),display_body_rotations=r.tolist(),display_rx_positions=(predicted@self.c.T).tolist(),
            native_skeleton_points=[native[n][:3,3].tolist() for n in self.urdf.points],
            tracking_errors={v['finger']:dict(position_error_mm=v['position_error_mm'],rotation_error_deg=v['rotation_error_deg'],satisfied=v['tracking_satisfied']) for v in reports})
        self.previous_timestamp=stamp;self.frames+=1
        return dict(pose=pose,targets=dict(timestamp_ms=stamp,hand_side=self.candidate['side'],coordinate_frame='wrist_local',names=list(FINGERS),positions_local=(target_native@self.c.T).tolist(),rotations_local=(self.c@target_rot).reshape(5,9).tolist(),orientation_convention='EMF_RX'),
            physics=dict(enabled=False,steps=0),joint_limit_diagnostics=[],continuity=dict(engine_frames=self.frames,reused_engine=self.frames>1),
            ik_diagnostics=dict(solver='wuji_sdk_native21',fingers=reports,processing_ms=(time.perf_counter()-start)*1000))

    def close(self):
        if getattr(self,'proc',None) is not None:
            self.proc.terminate()
            try:self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait()
            self.reader.close();self.proc.stdin.close();self.proc=None

    def __del__(self):
        try:self.close()
        except Exception:pass
