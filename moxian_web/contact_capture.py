"""Five-second natural contact windows with a separate complete holdout pass."""
import copy,math,uuid
from .motion_capture import MotionCapture
from .contact_protocol import PROTOCOL,STAGES,EXTENDED_STAGES,STAGE_BY_ID,SCHEDULE_REVISION,schedule,extract_events,observation_mask
from .calibration_targets import target


class ContactMotionCapture(MotionCapture):
    def __init__(self,initial,reference=None):
        self.rows=copy.deepcopy(initial['rows'])
        for row in self.rows:
            row.update(capture_protocol=PROTOCOL,initialization=True,partition='initialization',contact_confirmed=False,guided_contact=False)
        self.metadata={k:copy.deepcopy(v) for k,v in initial.items() if k not in ('rows','state')}
        self.metadata.update(protocol=PROTOCOL,validation_scope='separate_complete_contact_pass',source='continuous_motion',
                             completed_motion_stages=[],motion_session_id=uuid.uuid4().hex)
        self.metadata.setdefault('motion_schedule',schedule())
        self.metadata.setdefault('schedule_revision',SCHEDULE_REVISION)
        self.complete_groups=list(initial['completed_groups']);self.reference=reference
        if reference:self.metadata['contact_reference']=reference.spec
        self.index=0;self.events=[];self.recovered_cycles=0;self.stage_start=len(self.rows);self.window_start=len(self.rows)
        self._ready()

    def _ready(self):
        self.status='ready';self.request_id=uuid.uuid4().hex;self.started=None;self.last_stamp=None;self.elapsed=0.;self.error=None;self.window_gap_count=0

    @property
    def schedule(self):return self.metadata.get('motion_schedule') or schedule(False)
    @property
    def partition(self):return self.schedule[min(self.index,len(self.schedule)-1)][0]
    @property
    def stage(self):return STAGE_BY_ID[self.schedule[min(self.index,len(self.schedule)-1)][1]]

    def extend_spread_schedule(self):
        """Append to an unfinished legacy session; never renumber its data."""
        if self.status=='complete' or any(s=='motion_spread_together' for _,s in self.schedule):return False
        self.metadata['motion_schedule']=self.schedule+[[p,'motion_spread_together'] for p in ('train','validation')]
        self.metadata['schedule_revision']=SCHEDULE_REVISION
        return True

    def check_stage(self,stage_id):
        if self.index>=len(self.schedule) or stage_id!=self.stage['id']:raise ValueError('采集动作已更新，请按最新提示操作')

    def confirm(self,request_id):
        if self.status not in ('ready','paused') or request_id!=self.request_id:raise ValueError('采集请求已更新')
        if self.reference is None:raise ValueError('冻结参考模型尚未恢复，请重启模型后继续')
        if self._finish_stage():return
        self.status='collecting';self.started=None;self.last_stamp=None;self.elapsed=0.;self.error=None;self.window_start=len(self.rows);self.window_gap_count=0

    def retry(self):
        if self.status=='complete':raise ValueError('采集已完成，请开始新的标定')
        # Preserve raw observations, but remove superseded samples from fitting.
        self.rows[self.stage_start:]=[dict(row,training_eligible=False,contact_confirmed=False,guided_contact=False,motion_event_confirmed=False) for row in self.rows[self.stage_start:]]
        self.events=[];self.stage_start=len(self.rows);self.window_start=len(self.rows);self._ready()

    def pause_cycle(self,message):
        self._label_events()
        if self._finish_stage():return
        self.status='paused';self.started=None;self.last_stamp=None;self.elapsed=0.
        self.error=message;self.recovered_cycles+=1;self.request_id=uuid.uuid4().hex

    def _finish_stage(self):
        """Closed contact events survive a later stream interruption/restart."""
        if len(self.events)!=2:return False
        key=self.partition+':'+self.stage['id']
        if key not in self.complete_groups:self.complete_groups.append(key)
        if key not in self.metadata['completed_motion_stages']:self.metadata['completed_motion_stages'].append(key)
        self.index+=1;self.stage_start=len(self.rows);self.window_start=len(self.rows);self.events=[];self._ready()
        if self.index==len(self.schedule):self.status='complete';self.request_id=None
        return True

    def _label_events(self):
        group=self.rows[self.window_start:]
        existing={e['timestamp_ms'] for e in self.events}
        motion=target(self.stage['id'])['kind']=='noncontact_motion'
        for event in extract_events(group,allow_terminal=motion and self.elapsed>=5):
            if len(self.events)>=2 or event['timestamp_ms'] in existing:continue
            row=group[event['index']]
            event={**event,'row_index':self.window_start+event['index'],'repetition':len(self.events)+1}
            if motion:event['source']='trajectory_assumed_adduction'
            contact=target(self.stage['id'])['kind']!='noncontact_motion'
            self.rows[event['row_index']]=dict(row,contact_confirmed=contact,guided_contact=contact,motion_event_confirmed=not contact,motion_phase='touch' if contact else 'together',cycle=len(self.events),
                       contact_event_id=f'{self.metadata["motion_session_id"]}:{self.index}:{len(self.events)}',
                       contact_constraint_source='trajectory_assumed_contact' if contact else None,
                       motion_constraint_source=None if contact else 'trajectory_assumed_adduction',load_role='onset')
            self.events.append(event)

    def frame_usable(self,frame):
        skeleton=frame.get('skeleton') or {}
        return bool(frame.get('hand_side') in ('left','right') and isinstance(frame.get('timestamp_ms'),(int,float)) and math.isfinite(frame['timestamp_ms']) and isinstance(skeleton.get('timestamp_ms'),(int,float)) and abs(skeleton['timestamp_ms']-frame['timestamp_ms'])<=25)

    def feed(self,frame):
        if self.status!='collecting':return False
        if frame.get('calibration_fingerprint')!=self.metadata['calibration_fingerprint'] or frame.get('mapping',{}).get('assignments')!=self.rows[0]['mapping']['assignments']:
            self.cancel();self.error='初始化或 SID 映射已改变，请重新开始标定';return True
        if self.metadata.get('contact_reference',{}).get('skeleton_runtime_revision') and frame.get('skeleton_runtime_revision')!=self.metadata['contact_reference']['skeleton_runtime_revision']:
            self.cancel();self.error='IK 运行时已改变，请重新开始标定';return True
        stamp=frame['timestamp_ms']
        if not math.isfinite(stamp) or self.last_stamp is not None and stamp-self.last_stamp<19:return False
        gap_before=stamp-self.last_stamp if self.last_stamp is not None else 0.
        if gap_before>250:self.window_gap_count+=1
        self.last_stamp=stamp
        if self.started is None:self.started=stamp
        self.elapsed=(stamp-self.started)/1000
        row=copy.deepcopy(frame)
        row.update(stage=self.stage['id'],capture_protocol=PROTOCOL,initialization=False,partition=self.partition,
                   capture_pass=self.partition,window_id=f'{self.index}:{self.recovered_cycles}:{self.window_start}',cycle=-1,
                   motion_phase='trajectory',guided_contact=False,contact_confirmed=False,training_eligible=True,
                   contact_constraint_source=None,expected_contact_regions=copy.deepcopy(target(self.stage['id'])))
        row['observation_weights']=observation_mask(row)
        # Keep the remainder of the five-second window. The detector splits
        # at the original >250ms timestamp gap, never interpolating a release.
        if gap_before>250:row['capture_gap_before_ms']=gap_before
        active=target(self.stage['id'])['active_fingers']
        valid=all(f in row.get('fingers',{}) and row['fingers'][f].get('valid') is not False and row['fingers'][f].get('confidence',1)>0 for f in active)
        if valid:
            try:row.update(self.reference(row))
            except (ValueError,KeyError) as exc:row['reference_error']=str(exc)
        else:row['reference_error']='目标手指 EMF 缺失'
        self.rows.append(row)
        if self.elapsed>=5:
            self._label_events()
            if self._finish_stage():return True
            gaps=f'；窗口内有 {self.window_gap_count} 处采样缺口，缺口两侧未拼接识别' if self.window_gap_count else ''
            self.pause_cycle(f'已识别 {len(self.events)}/2 次完整接触—松开，缺少 {2-len(self.events)} 次{gaps}；已确认事件保留，请补录缺失次数')
            return True
        if len(self.rows[self.window_start:])%5==0:self._label_events()
        return False

    def snapshot(self):
        stage=self.stage;duration=len(self.schedule)*5.;done=min(self.index*5+self.elapsed,duration)
        stage_ids=list(dict.fromkeys(s for _,s in self.schedule));n=len(stage_ids)
        completed=self.metadata['completed_motion_stages']
        missing=[dict(capture_pass=p,stage_id=s,label=STAGE_BY_ID[s]['prompt'],seconds=5) for p,s in self.schedule if p+':'+s not in completed]
        return dict(state=self.status,protocol=PROTOCOL,reference_ready=self.reference is not None,stage_id=stage['id'] if self.status!='complete' else None,
                    prompt=stage['prompt'],request_id=self.request_id,index=self.index,total=len(self.schedule),action_index=stage_ids.index(stage['id']),
                    actions_per_pass=n,capture_pass=self.partition,samples=len(self.rows),progress=done/duration,
                    completed_train=sum('train:'+s in completed for s in stage_ids),completed_validation=sum('validation:'+s in completed for s in stage_ids),
                    can_fit=self.status=='complete' and not missing,missing_actions=missing,
                    seconds_remaining=max(0.,5-self.elapsed),estimated_remaining_s=max(0.,duration-done),
                    motion_phase='trajectory',cycle=min(len(self.events)+1,2),cycle_seconds=5,cycle_remaining_s=max(0.,5-self.elapsed),
                    window_gap_count=self.window_gap_count,
                    detected_repetitions=len(self.events),required_repetitions=2,contact_events=copy.deepcopy(self.events),
                    trajectory=[dict(t_ms=r['timestamp_ms'],gap_m=r.get('reference_gap_m')) for r in self.rows[self.window_start:] if 'reference_gap_m' in r],
                    retained_cycles=len(self.events),recovered_cycles=self.recovered_cycles,error=self.error,
                    completed_groups=self.complete_groups,source='continuous_motion',metadata=self.metadata)

    def document(self):return {**copy.deepcopy(self.metadata),'schema_version':3,'rows':list(self.rows),'completed_groups':list(self.complete_groups),'state':copy.deepcopy(self.snapshot())}

    @classmethod
    def restore(cls,document,allow_partial=False):
        if document.get('protocol')!=PROTOCOL:raise ValueError('不匹配的接触标定协议')
        complete=document['state']['state']=='complete'
        if complete:validate_document(document)
        elif not allow_partial:raise ValueError('接触标定尚未完成')
        obj=cls.__new__(cls);obj.rows=copy.deepcopy(document['rows']);obj.complete_groups=list(document['completed_groups'])
        obj.metadata={k:copy.deepcopy(v) for k,v in document.items() if k not in ('rows','state','completed_groups','schema_version')}
        state=document['state'];obj.index=state['index'];obj.events=copy.deepcopy(state.get('contact_events',[]));obj.reference=None
        obj.stage_start=next((i for i,r in enumerate(obj.rows) if r.get('stage')==obj.stage['id'] and r.get('partition')==obj.partition and not r.get('initialization') and r.get('training_eligible',True)),len(obj.rows)) if not complete else len(obj.rows)
        obj.window_start=len(obj.rows);obj.recovered_cycles=state.get('recovered_cycles',0);obj._ready()
        if complete:obj.status='complete';obj.request_id=None
        elif not obj._finish_stage():obj.status='paused';obj.error='采集已恢复；已确认事件保留，请补录缺失次数'
        return obj


def validate_document(document):
    if document.get('protocol')!=PROTOCOL or document.get('state',{}).get('state')!='complete':raise ValueError('接触标定尚未完成')
    rows=document['rows'];stamps=[r['timestamp_ms'] for r in rows]
    if len(stamps)!=len(set(stamps)):raise ValueError('重复采集时间戳')
    refs={r.get('contact_reference_id') for r in rows if not r.get('initialization') and 'reference_gap_m' in r}
    if len(refs)!=1 or None in refs:raise ValueError('采集参考模型不一致')
    for partition in ('train','validation'):
        stages=EXTENDED_STAGES if any(r.get('stage')=='motion_spread_together' for r in rows) or any(s=='motion_spread_together' for _,s in document.get('motion_schedule',[])) else STAGES
        for stage in stages:
            group=[r for r in rows if not r.get('initialization') and r.get('training_eligible',True) and r.get('stage')==stage['id'] and r.get('partition')==partition]
            motion=target(stage['id'])['kind']=='noncontact_motion'
            events=[r for r in group if r.get('motion_event_confirmed' if motion else 'contact_confirmed')]
            if len(events)!=2 or len({r.get('contact_event_id') for r in events})!=2:raise ValueError('动作缺少两次完整接触：'+partition+':'+stage['id'])
            if any(r.get('capture_pass')!=partition for r in group):raise ValueError('训练与复测分区不一致')
            if motion:
                if any(r.get('motion_constraint_source')!='trajectory_assumed_adduction' or r.get('contact_confirmed') for r in events):raise ValueError('张开并拢事件不应作为指尖接触')
            elif any(r.get('contact_constraint_source')!='trajectory_assumed_contact' for r in events):raise ValueError('接触证据来源不一致')
    return document
