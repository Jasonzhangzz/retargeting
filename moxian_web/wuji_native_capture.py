"""Wuji Studio 2026.7.1 six-gesture collection; no Moxian contact objective."""
from __future__ import annotations
import copy,uuid
from collections import deque
import numpy as np
from .motion_capture import MotionCapture
from .wuji_native_abi import NativeCalibrationABI,SDK_POSES,FINGERS,LABELS
from .static_capture import aggregate

PROTOCOL='wuji_studio_native_v1'
REVISION='wuji_studio_2026_7_1_six_poses'


class WujiNativeCapture(MotionCapture):
    def __init__(self,frame=None,side='right',abi=None,gesture_frame=None):
        frame=frame or {};self.abi=abi or NativeCalibrationABI();self.rows=[];self.aggregates=[];self.pose_frames={};self.complete_groups=[];self.index=0
        self.metadata=dict(protocol=PROTOCOL,schedule_revision=REVISION,motion_session_id=uuid.uuid4().hex,
            side=side,calibration_fingerprint=frame.get('calibration_fingerprint'),mapping=copy.deepcopy(frame.get('mapping')),
            source='wuji_native_static',sdk_version='2026.7.1',native_poses=copy.deepcopy(self.abi.poses),
            hold_seconds=1.,collect_seconds=1.,variance_threshold_m2=.0002,movement_threshold_m2=.00002,window_frames=20)
        if gesture_frame:self.install_gesture_frame(gesture_frame)
        self._ready()

    def install_gesture_frame(self,reference):
        if reference.get('calibration_fingerprint')!=self.metadata['calibration_fingerprint'] or reference.get('side')!=self.metadata['side']:raise ValueError('动作方向参考不匹配当前初始化')
        self.metadata['gesture_frame']=copy.deepcopy(reference)

    def _reset(self):
        self.window=deque(maxlen=20);self.collected=[];self.hold_since=None;self.collect_since=None;self.variance=None

    def _ready(self):
        self.status='ready';self.request_id=uuid.uuid4().hex;self.error=None;self.started=None;self.last_stamp=None;self.elapsed=0.;self.stage_start=len(self.rows)
        self.phase='waiting_movement';self.movement_seen=False;self.pose_validation=dict(accepted=False,checks=[],failures=[]);self._reset()

    @property
    def stage_id(self):return SDK_POSES[min(self.index,5)]
    @property
    def stage(self):return dict(id=self.stage_id)

    def confirm(self,request_id):
        if request_id!=self.request_id or self.status not in ('ready','paused'):raise ValueError('采集状态已更新')
        self._ready();self.status='collecting'

    def check_stage(self,stage_id):
        if stage_id!=self.stage_id or self.status=='complete':raise ValueError('采集动作已更新')

    def cancel(self):self.status='cancelled';self.request_id=None;self._reset()
    def pause_cycle(self,message):self.status='paused';self.error=message;self.request_id=uuid.uuid4().hex;self._reset()
    def retry(self):
        if self.status=='complete':raise ValueError('请开始新的六动作标定')
        self._ready()

    def feed(self,frame):
        if self.status!='collecting':return False
        stamp=frame.get('timestamp_ms')
        if not isinstance(stamp,(int,float)) or not np.isfinite(stamp):self._reset();return False
        if self.last_stamp is not None and stamp<=self.last_stamp:return False
        if self.started is None:self.started=stamp
        self.elapsed=(stamp-self.started)/1000
        if self.elapsed>120:self.pause_cycle('本动作采集超时，请继续当前动作');return True
        if self.metadata.get('calibration_fingerprint')!=frame.get('calibration_fingerprint') or self.metadata.get('mapping',{}).get('assignments')!=frame.get('mapping',{}).get('assignments'):
            self.cancel();self.error='输入初始化或SID分区改变，请重新开始';return True
        try:
            p=np.asarray([frame['fingers'][f]['position'] for f in FINGERS],np.float32).astype(float)
            q=np.asarray([frame['fingers'][f]['quaternion_xyzw'] for f in FINGERS],float)
            if p.shape!=(5,3) or q.shape!=(5,4) or not np.isfinite(np.r_[p.ravel(),q.ravel()]).all() or (np.linalg.norm(q,axis=-1)<1e-8).any() or any(frame['fingers'][f].get('valid') is False for f in FINGERS):raise ValueError('EMF观测无效')
        except (KeyError,TypeError,ValueError):
            self._reset();self.pose_validation=dict(accepted=False,checks=[],failures=['五指EMF观测无效']);return False
        if self.last_stamp is not None and stamp-self.last_stamp>250:self._reset()
        self.last_stamp=stamp;self.window.append(p)
        self.variance=float(np.var(self.window,axis=0).max()) if len(self.window)==20 else None
        gate_fingers=frame['fingers'];reference=self.metadata.get('gesture_frame')
        if reference and self.stage_id in ('four_finger_bend_90','flat_open'):
            from .wuji_gesture_frame import orient_for_gate
            gate_fingers=orient_for_gate(gate_fingers,reference)
        self.pose_validation=self.abi.check_pose(self.stage_id,gate_fingers,self.metadata['side'])
        if reference:self.pose_validation['orientation_frame']='moxian_initialized_wrist_bend_axis'
        row=dict(copy.deepcopy(frame),stage=self.stage_id,capture_protocol=PROTOCOL,partition='calibration',initialization=False)
        self.rows.append(row)
        if not self.movement_seen:
            self.phase='waiting_movement'
            if self.variance is None or self.variance<=.00002:return False
            self.movement_seen=True;self._reset();self.phase='waiting_stable';return False
        stable=self.variance is not None and self.variance<=.0002
        if not stable or not self.pose_validation['accepted']:
            self.hold_since=None;self.collect_since=None;self.collected=[];self.phase='waiting_stable' if self.pose_validation['accepted'] else 'waiting_pose';return False
        if self.hold_since is None:self.hold_since=stamp
        self.phase='hold'
        if stamp-self.hold_since<1000:return False
        self.phase='collect'
        if self.collect_since is None:self.collect_since=stamp
        self.collected.append(row)
        if stamp-self.collect_since<1000:return False
        name=self.stage_id;self.pose_frames[name]=[[copy.deepcopy(r['fingers'][f]) for f in FINGERS] for r in self.collected]
        representative=aggregate(self.collected);representative['static_pose_validation']=copy.deepcopy(self.pose_validation)
        self.aggregates.append(representative);self.complete_groups.append(name);self.index+=1;self._ready()
        if self.index==6:self.status='complete';self.request_id=None;self.phase='done'
        else:self.status='collecting'
        return True

    def snapshot(self):
        done=self.status=='complete';elapsed=0. if self.hold_since is None or self.last_stamp is None else (self.last_stamp-self.hold_since)/1000
        return dict(state=self.status,protocol=PROTOCOL,source='wuji_native_static',stage_id=None if done else self.stage_id,
            prompt='六动作采集完成，点击开始 Wuji 原生标定' if done else LABELS[self.stage_id]+'；先变换手势，姿态满足且稳定保持1秒，再采集1秒。',
            request_id=self.request_id,index=self.index,total=6,actions_per_pass=6,action_index=min(self.index,5),capture_pass='calibration',
            samples=len(self.rows),aggregate_count=len(self.aggregates),completed_train=self.index,completed_validation=0,
            reference_ready=True,can_fit=done,window_seconds=2,seconds_remaining=max(0.,2-elapsed),elapsed_s=self.elapsed,
            progress=self.index/6,motion_phase=self.phase,variance_m2=self.variance,stability_frames=len(self.window),stability_frames_required=20,
            constraints_ok=self.pose_validation['accepted'],constraint_failures=self.pose_validation.get('failures',[]),pose_checks=self.pose_validation.get('checks',[]),
            auto_advance=True,last_completed_key=self.complete_groups[-1] if self.complete_groups else None,completed_groups=list(self.complete_groups),error=self.error,metadata=self.metadata)

    def document(self):
        return dict(copy.deepcopy(self.metadata),schema_version=1,rows=self.rows,aggregates=self.aggregates,
            pose_frames=copy.deepcopy(self.pose_frames),completed_groups=self.complete_groups,state=self.snapshot())

    @classmethod
    def restore(cls,doc,allow_partial=False):
        if doc.get('protocol')!=PROTOCOL or doc.get('schedule_revision')!=REVISION:raise ValueError('不是Wuji原生六动作采集')
        index=len(doc.get('completed_groups',[]))
        if doc.get('completed_groups')!=list(SDK_POSES[:index]) or index>6:raise ValueError('原生动作记录不完整')
        if index<6 and not allow_partial:raise ValueError('请先完成六动作采集')
        obj=cls(dict(calibration_fingerprint=doc.get('calibration_fingerprint'),mapping=doc.get('mapping')),doc['side'])
        obj.rows=copy.deepcopy(doc['rows']);obj.aggregates=copy.deepcopy(doc['aggregates']);obj.pose_frames=copy.deepcopy(doc['pose_frames']);obj.complete_groups=list(doc['completed_groups']);obj.index=index
        obj.metadata={k:copy.deepcopy(v) for k,v in doc.items() if k not in ('rows','aggregates','pose_frames','completed_groups','state','schema_version')};obj._ready()
        obj.status='complete' if index==6 else 'paused'
        if index==6:obj.request_id=None;obj.phase='done'
        return obj
