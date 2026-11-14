"""Seventeen instructed actions, isolated static train and holdout passes."""
from __future__ import annotations
import copy
import uuid
from collections import deque
from scipy.spatial.transform import Rotation
import numpy as np
from .motion_capture import MotionCapture
from .contact_protocol import schedule,STAGE_BY_ID
from .calibration_targets import target
from .bone_extrinsics import FINGERS
from .static_pose import StaticPoseValidator,REVISION as POSE_REVISION,LIMITS as POSE_LIMITS

PROTOCOL='contact_static_v5'
REVISION='moxian_static_pose_stable_window_v3'
LEGACY_REVISION='wuji_static_pose_hold1_collect1_v2'


def static_schedule():
    return [(p,s,endpoint) for p,s in schedule() for endpoint in
            (('spread','together') if s=='motion_spread_together' else ('target',))]


def aggregate(rows):
    if not rows: raise ValueError('没有静态观测')
    result=copy.deepcopy(rows[len(rows)//2]); fingers={}
    for f in FINGERS:
        p=np.array([r['fingers'][f]['position'] for r in rows],float)
        q=np.array([r['fingers'][f]['quaternion_xyzw'] for r in rows],float)
        q=q/np.linalg.norm(q,axis=1,keepdims=True)
        _,vectors=np.linalg.eigh(q.T@q); mean=vectors[:,-1]
        if mean[-1]<0: mean=-mean
        fingers[f]=dict(position=p.mean(0).tolist(),quaternion_xyzw=mean.tolist(),valid=True)
    result.update(fingers=fingers,static_aggregate=True,static_sample_count=len(rows),
        static_start_ms=rows[0]['timestamp_ms'],static_end_ms=rows[-1]['timestamp_ms'])
    return result


class StaticHandCapture(MotionCapture):
    def __init__(self,initial,reference=None,pose_validator=None):
        self.rows=[dict(copy.deepcopy(r),initialization=True,partition='initialization',
            capture_protocol=PROTOCOL,contact_confirmed=False) for r in initial['rows']]
        self.metadata={k:copy.deepcopy(v) for k,v in initial.items() if k not in ('rows','state','completed_groups')}
        self.metadata.update(protocol=PROTOCOL,schedule_revision=REVISION,motion_session_id=uuid.uuid4().hex,
            source='static_holds',motion_schedule=static_schedule(),window_seconds=0.,hold_seconds=0.,
            collect_seconds=0.,stability_frames=20,timeout_seconds=120.,variance_threshold_m2=.0002,completed_motion_stages=[])
        self.reference=None;self.pose_validator=pose_validator
        if pose_validator is not None and hasattr(pose_validator,'spec'):
            self.metadata['static_pose_validator']=copy.deepcopy(pose_validator.spec)
        if reference:self.install_reference(reference,initial)
        self.complete_groups=[]; self.aggregates=[]; self.index=0; self._ready()

    def install_reference(self,reference,initial):
        self.reference=reference;self.metadata['contact_reference']=reference.spec
        saved=self.metadata.get('static_pose_validator')
        # Resume unfinished sessions with current collection tolerances.
        # Accepted aggregates retain their original validation evidence.
        if saved and getattr(self,'status',None)!='complete':
            saved=copy.deepcopy(saved);saved['limits']=copy.deepcopy(POSE_LIMITS)
        self.pose_validator=StaticPoseValidator(saved) if saved else StaticPoseValidator.from_reference(reference,initial)
        self.metadata['static_pose_validator']=copy.deepcopy(self.pose_validator.spec)

    @property
    def schedule(self): return static_schedule()
    @property
    def entry(self): return self.schedule[min(self.index,len(self.schedule)-1)]
    @property
    def stage(self): return STAGE_BY_ID[self.entry[1]]
    @property
    def partition(self): return self.entry[0]

    def _reset_window(self):
        self.window=deque(maxlen=20); self.stable_since=None; self.collect_since=None; self.collected=[]
        self.variance=None; self.phase='waiting_stable'

    def _ready(self):
        self.status='ready'; self.request_id=uuid.uuid4().hex; self.error=None; self.started=None
        self.last_stamp=None; self.elapsed=0.; self.stage_start=len(self.rows); self._reset_window()
        self.pose_validation=dict(accepted=False,failures=['请摆好当前目标姿态'])
        self.transition_reference=None

    def confirm(self,request_id):
        if request_id!=self.request_id or self.status not in ('ready','paused','fit_ready'): raise ValueError('采集请求已更新')
        if self.pose_validator is None:raise ValueError('静态动作检验参考尚未准备完成')
        self._ready(); self.status='collecting'

    def check_stage(self,stage_id):
        if self.status=='complete' or stage_id!=self.stage['id']: raise ValueError('采集动作已更新')

    def frame_usable(self,frame):
        try:
            if not np.isfinite(frame['timestamp_ms']):return False
            for f in FINGERS:
                n=frame['fingers'][f]; p=np.asarray(n['position']); q=np.asarray(n['quaternion_xyzw'])
                if n.get('valid') is False or n.get('confidence',1)<=0 or p.shape!=(3,) or q.shape!=(4,) or not np.isfinite(np.r_[p,q]).all() or np.linalg.norm(q)<1e-8:return False
            return True
        except (KeyError,TypeError,ValueError):return False

    def feed(self,frame):
        if self.status!='collecting':return False
        stamp=frame.get('timestamp_ms')
        if isinstance(stamp,(int,float)) and np.isfinite(stamp):
            if self.started is None:self.started=stamp
            self.elapsed=(stamp-self.started)/1000
            if self.elapsed>120:self.pause_cycle('目标姿态采集超时，请重新就位');return True
        if not self.frame_usable(frame):
            self._reset_window();self.pose_validation=dict(accepted=False,failures=['五指EMF观测无效，请检查定位']);self.phase='waiting_pose';return False
        if self.last_stamp is not None and stamp<=self.last_stamp:return False
        if self.last_stamp is not None and stamp-self.last_stamp>250:self._reset_window()
        self.last_stamp=stamp
        if (frame.get('calibration_fingerprint')!=self.metadata.get('calibration_fingerprint') or
            self.rows and frame.get('mapping',{}).get('assignments')!=self.rows[0].get('mapping',{}).get('assignments')):
            self.cancel();self.error='八动作初始化或SID分区改变，请重新采集';return True
        part,stage,endpoint=self.entry; row=copy.deepcopy(frame); spec=target(stage)
        row.update(capture_protocol=PROTOCOL,partition=part,capture_pass=part,stage=stage,
            static_endpoint=endpoint,initialization=False,training_eligible=True,motion_phase=self.phase,
            observation_weights=[float(f in spec['active_fingers']) for f in FINGERS],
            contact_confirmed=False,contact_constraint_source='instructed_static_hold')
        row['expected_contact_regions']=dict(copy.deepcopy(spec),equality_phase='accepted_static_hold',seconds=0,stability_frames=20,repetitions=1)
        try:self.pose_validation=self.pose_validator(row)
        except (KeyError,ValueError,TypeError) as exc:self.pose_validation=dict(accepted=False,failures=['静态动作检验失败：'+str(exc)])
        if self.transition_reference is not None:
            active=spec['active_fingers']
            distance=max(np.linalg.norm(np.asarray(row['fingers'][f]['position'])-self.transition_reference['fingers'][f]['position']) for f in active)
            angle=max((Rotation.from_quat(row['fingers'][f]['quaternion_xyzw'])*Rotation.from_quat(self.transition_reference['fingers'][f]['quaternion_xyzw']).inv()).magnitude() for f in active)
            if distance<.005 and angle<.1:
                self.pose_validation=dict(accepted=False,failures=['上一动作已通过，请变换到图示的下一姿态'])
            else:self.transition_reference=None
        row['static_pose_validation']=copy.deepcopy(self.pose_validation);self.rows.append(row)
        if not self.pose_validation.get('accepted'):
            self._reset_window();self.phase='waiting_pose';return False
        self.window.append(np.array([frame['fingers'][f]['position'] for f in FINGERS]))
        self.collected.append(row);self.collected=self.collected[-20:]
        if len(self.window)<20:return False
        self.variance=float(np.var(np.array(self.window),axis=0).max())
        if self.variance>.0002:self._reset_window();return False
        self.phase='collect'
        representative=aggregate(self.collected)
        representative.update(contact_confirmed=spec['kind']!='noncontact_motion',motion_phase='hold',
            static_key=f'{part}:{stage}:{endpoint}')
        accepted={r['timestamp_ms'] for r in self.collected}
        for i in range(self.stage_start,len(self.rows)):
            if self.rows[i]['timestamp_ms'] in accepted:
                self.rows[i]=dict(self.rows[i],static_accepted=True,static_key=representative['static_key'])
        self.aggregates.append(representative);self.complete_groups.append(representative['static_key'])
        if endpoint!='spread':self.metadata['completed_motion_stages'].append(part+':'+stage)
        self.index+=1;self._ready()
        if self.index==len(self.schedule):self.status='complete';self.request_id=None
        elif self.index==18:
            self.status='fit_ready';self.phase='fit_ready'
        else:
            self.status='collecting';self.transition_reference=representative;self.phase='waiting_pose'
            self.pose_validation=dict(accepted=False,failures=['上一动作已通过，请变换到图示的下一姿态'])
        return True

    def pause_cycle(self,message):
        self.status='paused';self.error=message;self.request_id=uuid.uuid4().hex;self._reset_window()

    def retry(self):
        if self.status=='complete':raise ValueError('请开始新的标定')
        for i in range(self.stage_start,len(self.rows)):self.rows[i]=dict(self.rows[i],training_eligible=False)
        self._ready()

    def snapshot(self):
        done=self.status=='complete';part,stage,endpoint=self.entry;completed=self.metadata['completed_motion_stages']
        prompt=target(stage)['label'].split('两次')[0]
        if endpoint!='target':prompt='五指伸直'+('张开' if endpoint=='spread' else '并拢')
        return dict(state=self.status,protocol=PROTOCOL,source='static_holds',stage_id=None if done else stage,
            static_endpoint=endpoint,prompt=prompt+'；姿态通过且最近20帧稳定后自动采集，无额外保持倒计时。',
            request_id=self.request_id,index=self.index,total=len(self.schedule),actions_per_pass=17,
            action_index=min(self.index%18,16),
            capture_pass=part,samples=len(self.rows),aggregate_count=len(self.aggregates),progress=self.index/len(self.schedule),
            completed_train=sum(k.startswith('train:') for k in completed),completed_validation=sum(k.startswith('validation:') for k in completed),
            reference_ready=self.pose_validator is not None,can_fit=done or self.status=='fit_ready',fit_data_ready=self.index>=18,window_seconds=0,elapsed_s=self.elapsed,seconds_remaining=None,stability_frames=len(self.window),stability_frames_required=20,
            estimated_remaining_s=None,
            auto_advance=True,last_completed_key=self.complete_groups[-1] if self.complete_groups else None,
            motion_phase=self.phase,variance_m2=self.variance,constraints_ok=self.pose_validation.get('accepted',False),
            constraint_failures=self.pose_validation.get('failures',[]),pose_checks=self.pose_validation.get('checks',[]),cycle=1,required_repetitions=1,detected_repetitions=0,
            missing_actions=[dict(capture_pass=p,stage_id=s,label=('五指伸直张开' if e=='spread' else '五指伸直并拢' if e=='together' else target(s)['label']),endpoint=e) for p,s,e in self.schedule[self.index:]],
            completed_groups=list(self.complete_groups),error=self.error,metadata=self.metadata)

    def document(self):
        return dict(copy.deepcopy(self.metadata),schema_version=5,rows=list(self.rows),
            aggregates=copy.deepcopy(self.aggregates),completed_groups=list(self.complete_groups),state=self.snapshot())

    @classmethod
    def restore(cls,document,allow_partial=False):
        if document.get('protocol')!=PROTOCOL:raise ValueError('需要新的静态17动作记录')
        if document.get('schedule_revision') not in (REVISION,LEGACY_REVISION):raise ValueError('静态动作检验已更新，请重新采集17动作')
        done=document.get('state',{}).get('state')=='complete'
        if not done and not allow_partial:raise ValueError('静态采集尚未完成')
        obj=cls.__new__(cls);obj.rows=copy.deepcopy(document['rows']);obj.aggregates=copy.deepcopy(document.get('aggregates',[]))
        obj.complete_groups=list(document['completed_groups']);obj.reference=None;obj.pose_validator=None
        obj.metadata={k:copy.deepcopy(v) for k,v in document.items() if k not in ('rows','state','aggregates','schema_version','completed_groups')}
        obj.index=len(obj.aggregates);obj._ready()
        if done:
            if [r.get('static_key') for r in obj.aggregates]!=[':'.join(x) for x in static_schedule()]:raise ValueError('静态训练／复测不完整')
            if any(r.get('static_pose_validation',{}).get('revision')!=POSE_REVISION or not r['static_pose_validation'].get('accepted') for r in obj.aggregates):raise ValueError('记录缺少静态动作检验，请重新采集')
            obj.status='complete';obj.request_id=None
        else:
            obj.metadata.update(schedule_revision=REVISION,window_seconds=0.,hold_seconds=0.,collect_seconds=0.,stability_frames=20)
            obj.status='fit_ready' if obj.index>=18 else 'paused'
            obj.error=None if obj.status=='fit_ready' else '已保存完整保持片段，请重新采集当前姿态'
        return obj
