"""Six-second minimum, two fully flat-to-contact-to-flat trajectories."""
from __future__ import annotations
import copy,math,uuid
from .motion_capture import MotionCapture
from .contact_protocol import schedule,STAGE_BY_ID
from .calibration_targets import target
from .contact import FINGERS

PROTOCOL='contact_motion_v4'
REVISION='flat_return_trajectory_v1'


class FlatCycleDetector:
    """Incremental detector; no history rescans or duration/velocity gates."""
    def __init__(self):self.reset()
    def reset(self):
        self.phase='flat';self.flat_count=0;self.anchor=None;self.peak=None;self.minimum=None;self.last=None;self.noise=[]
    def feed(self,row,index):
        stamp=row['timestamp_ms'];gap=row.get('reference_gap_m');events=[]
        if self.last is not None and stamp-self.last>250:self.reset()
        self.last=stamp
        if not isinstance(gap,(float,int)) or not math.isfinite(gap) or row.get('reference_error'):
            self.reset();return events
        self.flat_count=self.flat_count+1 if row.get('flat_valid') else 0 if row.get('flat_exit',True) else self.flat_count
        flat=self.flat_count>=3
        if self.anchor is None:
            if flat:self.anchor=index;self.peak=gap;self.phase='apart'
            return events
        if self.phase=='apart':
            self.peak=max(self.peak,gap)
            if self.peak-gap>=.002 and not row.get('flat_valid'):self.phase='approach';self.minimum=(gap,index,stamp)
        elif self.phase in ('approach','release'):
            if gap<self.minimum[0]:self.minimum=(gap,index,stamp)
            if gap-self.minimum[0]>=max(.002,.35*(self.peak-self.minimum[0])):self.phase='release'
            if flat and self.phase=='release':
                low,i,t=self.minimum
                events.append(dict(index=i,timestamp_ms=t,gap_m=low,flat_start_index=self.anchor,flat_end_index=index,source='trajectory_assumed_contact'))
                self.anchor=index;self.peak=gap;self.minimum=None;self.phase='apart'
        return events


class ContactTrajectoryCapture(MotionCapture):
    def __init__(self,initial,reference=None):
        self.rows=[dict(copy.deepcopy(r),capture_protocol=PROTOCOL,initialization=True,partition='initialization',contact_confirmed=False,guided_contact=False) for r in initial['rows']]
        self.metadata={k:copy.deepcopy(v) for k,v in initial.items() if k not in ('rows','state','completed_groups')}
        self.metadata.update(protocol=PROTOCOL,schedule_revision=REVISION,motion_schedule=schedule(),window_seconds=6.,required_repetitions=2,
            validation_scope='separate_complete_contact_pass',source='continuous_motion',completed_motion_stages=[],motion_session_id=uuid.uuid4().hex)
        self.complete_groups=list(initial.get('completed_groups',[]));self.reference=reference
        if reference:self.metadata['contact_reference']=reference.spec
        self.index=0;self.events=[];self.recovered_cycles=0;self.stage_start=len(self.rows);self.window_start=len(self.rows);self._ready()

    def _ready(self):
        self.status='ready';self.request_id=uuid.uuid4().hex;self.last_stamp=None;self.started=None;self.elapsed=0.;self.error=None;self.window_gap_count=0;self.detector=FlatCycleDetector()

    @property
    def schedule(self):return self.metadata['motion_schedule']
    @property
    def partition(self):return self.schedule[min(self.index,len(self.schedule)-1)][0]
    @property
    def stage(self):return STAGE_BY_ID[self.schedule[min(self.index,len(self.schedule)-1)][1]]

    def check_stage(self,stage_id):
        if stage_id!=self.stage['id'] or self.status=='complete':raise ValueError('采集动作已更新')

    def confirm(self,request_id):
        if request_id!=self.request_id or self.status not in ('ready','paused'):raise ValueError('采集请求已更新')
        if self.reference is None:raise ValueError('冻结官方参考模型尚未准备完成')
        self.status='collecting';self.started=None;self.last_stamp=None;self.elapsed=0.;self.error=None;self.window_start=len(self.rows);self.detector=FlatCycleDetector()

    def frame_usable(self,frame):
        s=frame.get('skeleton') or {};t=frame.get('timestamp_ms')
        return isinstance(t,(float,int)) and math.isfinite(t) and isinstance(s.get('timestamp_ms'),(float,int)) and abs(s['timestamp_ms']-t)<=25

    def _finish(self):
        if len(self.events)<2:return False
        key=self.partition+':'+self.stage['id'];self.complete_groups.append(key);self.metadata['completed_motion_stages'].append(key)
        self.index+=1;self.events=[];self.stage_start=len(self.rows);self.window_start=len(self.rows);self._ready()
        if self.index>=len(self.schedule):self.status='complete';self.request_id=None
        return True

    def feed(self,frame):
        if self.status!='collecting':return False
        if frame.get('calibration_fingerprint')!=self.metadata.get('calibration_fingerprint') or frame.get('mapping',{}).get('assignments')!=self.rows[0].get('mapping',{}).get('assignments'):
            self.cancel();self.error='初始化或SID分区改变，请重新采集';return True
        if frame.get('skeleton_runtime_revision')!=self.rows[0].get('skeleton_runtime_revision'):
            self.cancel();self.error='骨架运行版本改变，请重新采集';return True
        if not self.frame_usable(frame):return False
        stamp=frame['timestamp_ms']
        if self.last_stamp is not None and stamp-self.last_stamp<19:return False
        if self.last_stamp is not None and stamp-self.last_stamp>250:self.window_gap_count+=1
        self.last_stamp=stamp
        if self.started is None:self.started=stamp
        self.elapsed=(stamp-self.started)/1000
        active=target(self.stage['id'])['active_fingers'];row=copy.deepcopy(frame)
        row.update(stage=self.stage['id'],capture_protocol=PROTOCOL,partition=self.partition,capture_pass=self.partition,initialization=False,
            window_id=f'{self.index}:{self.recovered_cycles}:{self.window_start}',cycle=len(self.events),motion_phase=self.detector.phase,
            contact_confirmed=False,guided_contact=False,training_eligible=True,observation_weights=[float(f in active) for f in FINGERS],
            expected_contact_regions=copy.deepcopy(target(self.stage['id'])))
        try:row.update(self.reference(row))
        except (ValueError,KeyError) as exc:row['reference_error']=str(exc);row['flat_valid']=False
        i=len(self.rows);self.rows.append(row)
        for e in self.detector.feed(row,i):
            if len(self.events)>=2:break
            rep=len(self.events);motion=target(self.stage['id'])['kind']=='noncontact_motion'
            e.update(row_index=e['index'],repetition=rep+1,source='trajectory_assumed_adduction' if motion else 'trajectory_assumed_contact')
            event_id=f'{self.metadata["motion_session_id"]}:{self.index}:{rep}'
            # Replace dictionaries: background checkpoints retain immutable rows.
            for k in range(e['flat_start_index'],e['flat_end_index']+1):
                old=self.rows[k];phase='apart' if k==e['flat_start_index'] else 'touch' if k==e['index'] else 'approach' if k<e['index'] else 'release'
                self.rows[k]=dict(old,cycle=rep,motion_phase=phase,trajectory_flat_start_index=e['flat_start_index'],cycle_event_id=event_id)
            self.rows[e['index']]=dict(self.rows[e['index']],contact_confirmed=not motion,guided_contact=not motion,motion_event_confirmed=motion,
                contact_event_id=event_id,contact_constraint_source=None if motion else 'trajectory_assumed_contact',motion_constraint_source='trajectory_assumed_adduction' if motion else None)
            self.events.append(e)
        if self.elapsed>=6 and self._finish():return True
        return False

    def pause_cycle(self,message):
        if self.elapsed>=6 and self._finish():return
        self.status='paused';self.error=message;self.recovered_cycles+=1;self.request_id=uuid.uuid4().hex;self.detector.reset()

    def retry(self):
        if self.status=='complete':raise ValueError('请开始新的标定')
        for i in range(self.stage_start,len(self.rows)):self.rows[i]=dict(self.rows[i],training_eligible=False,contact_confirmed=False,guided_contact=False,motion_event_confirmed=False)
        self.events=[];self.stage_start=len(self.rows);self.window_start=len(self.rows);self._ready()

    def snapshot(self):
        total=len(self.schedule);done=self.status=='complete';n=total//2;completed=self.metadata['completed_motion_stages'];duration=total*6.
        phase=self.detector.phase;latest=self.rows[-1] if len(self.rows)>self.window_start else {}
        points=self.rows[max(self.window_start,len(self.rows)-300):]
        return dict(state=self.status,protocol=PROTOCOL,stage_id=None if done else self.stage['id'],prompt=self.stage['prompt'].split('；')[0]+'；展平→贴合→松开展平，完成两次；6秒不足时自动延长。',
            request_id=self.request_id,reference_ready=self.reference is not None,index=self.index,total=total,actions_per_pass=n,action_index=min(self.index,total-1)%n,
            capture_pass=self.partition,samples=len(self.rows),progress=1. if done else min((self.index*6+min(self.elapsed,6))/duration,.999),
            completed_train=sum(x.startswith('train:') for x in completed),completed_validation=sum(x.startswith('validation:') for x in completed),
            can_fit=done,missing_actions=[dict(capture_pass=p,stage_id=s,label=STAGE_BY_ID[s]['prompt'],seconds=6) for p,s in self.schedule if p+':'+s not in completed],
            window_seconds=6,elapsed_s=self.elapsed,seconds_remaining=max(0,6-self.elapsed),extended=self.status=='collecting' and self.elapsed>=6,
            estimated_remaining_s=max(0,duration-self.index*6-min(self.elapsed,6)),motion_phase=phase,cycle=min(len(self.events)+1,2),cycle_seconds=6,cycle_remaining_s=max(0,6-self.elapsed),
            flat_valid=latest.get('flat_valid',False),flat_evidence=latest.get('flat_evidence'),waiting_for='展平回位' if phase in ('flat','release') else '贴合后松开展平',
            detected_repetitions=len(self.events),required_repetitions=2,contact_events=copy.deepcopy(self.events),retained_cycles=len(self.events),recovered_cycles=self.recovered_cycles,
            trajectory=[dict(t_ms=r['timestamp_ms'],gap_m=r['reference_gap_m'],flat=r.get('flat_valid',False),patches=r.get('patch_trajectory',{})) for r in points if 'reference_gap_m' in r],
            window_gap_count=self.window_gap_count,error=self.error,completed_groups=list(self.complete_groups),source='continuous_motion',metadata=self.metadata)

    def document(self):return dict(copy.deepcopy(self.metadata),schema_version=4,rows=list(self.rows),completed_groups=list(self.complete_groups),state=self.snapshot())

    @classmethod
    def restore(cls,document,allow_partial=False):
        if document.get('protocol')!=PROTOCOL:raise ValueError('非v4采集记录')
        complete=document.get('state',{}).get('state')=='complete'
        if complete:validate_document(document)
        elif not allow_partial:raise ValueError('轨迹采集尚未完成')
        obj=cls.__new__(cls);obj.rows=copy.deepcopy(document['rows']);obj.complete_groups=list(document['completed_groups'])
        obj.metadata={k:copy.deepcopy(v) for k,v in document.items() if k not in ('rows','state','schema_version','completed_groups')};obj.reference=None
        state=document['state'];obj.index=state['index'];obj.events=copy.deepcopy(state.get('contact_events',[]));obj.recovered_cycles=state.get('recovered_cycles',0)+1;obj.stage_start=len(obj.rows);obj.window_start=len(obj.rows);obj._ready()
        if complete:obj.status='complete';obj.request_id=None
        else:obj.status='paused';obj.error='已确认循环保留，请展平后补录缺少的次数'
        return obj


def validate_document(document):
    if document.get('protocol')!=PROTOCOL or document.get('state',{}).get('state')!='complete':raise ValueError('v4轨迹采集尚未完成')
    rows=document['rows'];stamps=[r['timestamp_ms'] for r in rows]
    if len(set(stamps))!=len(stamps):raise ValueError('采集时间戳重复')
    for part,stage in document['motion_schedule']:
        group=[r for r in rows if r.get('partition')==part and r.get('stage')==stage and not r.get('initialization') and r.get('training_eligible',True)]
        events=[r for r in group if r.get('contact_confirmed') or r.get('motion_event_confirmed')]
        if len(events)!=2:raise ValueError('缺少两次展平回位循环 '+part+':'+stage)
        for event in events:
            ids=[r for r in group if r.get('cycle_event_id')==event.get('cycle_event_id')]
            if not ids or not ids[-1].get('flat_valid') or any(r.get('capture_pass')!=part for r in ids):raise ValueError('展平回位证据或分区无效')
    refs={r.get('contact_reference_id') for r in rows if not r.get('initialization') and r.get('training_eligible',True) and not r.get('reference_error')}
    if len(refs)!=1 or None in refs:raise ValueError('冻结参考不一致')
    return document
