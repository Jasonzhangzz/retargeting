"""Continuous, repeated calibration motions; independent complete-cycle holdout.

Raw observations are copied, never reconstructed from a fitted hand. Timing
labels describe instructions, while contact labels require fresh bilateral ADC.
"""
from __future__ import annotations

import copy
import math
import uuid
from .calibration_targets import target as contact_target

PROTOCOL = 'geort_motion_v2'
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
NAMES = ('拇指', '食指', '中指', '无名指', '小指')
STAGES = [
    *[dict(id='range_'+f, seconds=9., cycles=3,
           prompt=f'{n}缓慢伸直→弯曲→伸直，兼顾侧向活动；四指加入钩状变化，重复三轮。')
      for f, n in zip(FINGERS, NAMES)],
    dict(id='thumb_opposition', seconds=12., cycles=3,
         prompt='拇指外展→扫向掌心和各指→展开，舒适地重复三轮。'),
    *[dict(id='pinch_'+f, seconds=15., cycles=3,
           prompt=f'拇指与{n}分开→接近→轻触→释放；跟随倒计时，每轮五秒，共三轮。')
      for f, n in zip(FINGERS[1:], NAMES[1:])],
    dict(id='open_repeat', seconds=4., cycles=1, prompt='再次自然张开，保持四秒；本组仅作独立验证。'),
    dict(id='together_repeat', seconds=4., cycles=1, prompt='四指伸直并拢，保持四秒；本组仅作独立验证。'),
]


def phase_at(stage, elapsed):
    cycle_seconds=stage['seconds']/stage['cycles'];t=max(0.,elapsed)%cycle_seconds
    if stage['id'].startswith('pinch_'):
        return ('apart' if t<1 else 'approach' if t<2 else 'touch' if t<3.5 else 'release')
    if stage['id'].endswith('_repeat'):return 'hold'
    if stage['id']=='thumb_opposition':return ('spread' if t<cycle_seconds/3 else 'oppose' if t<2*cycle_seconds/3 else 'return')
    return 'extend' if t<cycle_seconds/3 else 'curl' if t<2*cycle_seconds/3 else 'return'


def confirmed_touch(row):
    """Only the instructed hold interval may carry a contact equality."""
    if row.get('capture_protocol')=='contact_motion_v3':
        return bool(not row.get('initialization') and row.get('training_eligible',True) and row.get('contact_confirmed') and row.get('contact_constraint_source')=='trajectory_assumed_contact')
    if not row.get('stage', '').startswith('pinch_'):
        return False
    if row.get('capture_protocol') != PROTOCOL:
        return bool(row.get('guided_contact'))
    if row.get('initialization'):return bool(row.get('guided_contact'))
    return row.get('motion_phase') == 'touch' and row.get('contact_confirmed') is True


def bilateral(row, partner):
    contact = row.get('contact') or {}
    stamp, tactile = row.get('timestamp_ms'), row.get('tactile_timestamp_ms')
    if not contact.get('ready') or not isinstance(tactile, (float, int)):
        return False
    if not isinstance(stamp, (float, int)) or abs(stamp-tactile) > 25:
        return False
    if abs(stamp-contact.get('timestamp_ms', tactile)) > 25:
        return False
    patches = {p['id']: p for p in contact.get('patches', [])}
    return bool(patches.get('thumb_distal', {}).get('contact') and
                patches.get(partner+'_distal', {}).get('contact'))


class MotionCapture:
    @classmethod
    def restore(cls,document,allow_partial=False):
        if document.get('protocol')=='contact_static_v5':
            from .static_capture import StaticHandCapture
            return StaticHandCapture.restore(document,allow_partial)
        if document.get('protocol')=='contact_motion_v4':
            from .contact_capture_v4 import ContactTrajectoryCapture
            return ContactTrajectoryCapture.restore(document,allow_partial)
        if document.get('protocol')=='contact_motion_v3':
            from .contact_capture import ContactMotionCapture
            return ContactMotionCapture.restore(document,allow_partial)
        complete=document.get('state',{}).get('state')=='complete'
        if complete:validate_document(document)
        elif not allow_partial or document.get('protocol')!=PROTOCOL:raise ValueError('连续标定尚未完成')
        obj=cls.__new__(cls)
        obj.rows=copy.deepcopy(document['rows'])
        obj.metadata={k:copy.deepcopy(v) for k,v in document.items() if k not in ('rows','state','completed_groups','schema_version')}
        obj.complete_groups=list(document['completed_groups']);obj.status='complete' if complete else 'ready'
        obj.index=len(STAGES) if complete else len(document.get('completed_motion_stages',[]))
        if not complete and [s['id'] for s in STAGES[:obj.index]]!=document.get('completed_motion_stages',[]):raise ValueError('保存的动作顺序无效')
        obj.error=None;obj.request_id=None;obj.started=None;obj.last_stamp=None;obj.stage_start=len(obj.rows);obj.elapsed=0.
        obj.segment_offset=0.;obj.recovered_cycles=0
        if not complete:
            obj.stage_start=next((i for i,r in enumerate(obj.rows) if not r.get('initialization') and r['stage']==STAGES[obj.index]['id']),len(obj.rows))
            obj.elapsed=max(0.,STAGES[obj.index]['seconds']-document['state'].get('seconds_remaining',STAGES[obj.index]['seconds']))
            obj.pause_cycle('已恢复采集记录；已完成轮次保留，点击继续当前一轮')
        return obj

    def __init__(self, initial):
        self.rows = copy.deepcopy(initial['rows'])
        for stage in dict.fromkeys(r['stage'] for r in self.rows):
            group = sorted((r for r in self.rows if r['stage'] == stage), key=lambda r:r['timestamp_ms'])
            for i, row in enumerate(group):
                row.update(capture_protocol=PROTOCOL, initialization=True,
                           partition='train' if i < len(group)//2 else 'validation')
        self.metadata = {k:copy.deepcopy(v) for k,v in initial.items() if k != 'rows'}
        self.metadata.update(protocol=PROTOCOL, validation_scope='complete_third_cycles_and_final_repeats',
                             source='continuous_motion', completed_motion_stages=[],motion_session_id=uuid.uuid4().hex)
        self.complete_groups = list(initial['completed_groups'])
        self.status = 'ready'; self.index = 0; self.error = None
        self.request_id = uuid.uuid4().hex; self.started = None; self.last_stamp = None
        self.stage_start = len(self.rows); self.elapsed = 0.
        self.segment_offset=0.;self.recovered_cycles=0

    def confirm(self, request_id):
        if self.status not in ('ready','paused') or request_id != self.request_id:
            raise ValueError('采集请求已更新，请按最新动作提示确认')
        self.segment_offset=self.elapsed if self.status=='paused' else 0.
        self.status = 'collecting'; self.started = None; self.last_stamp=None;self.elapsed=self.segment_offset; self.error = None

    def check_stage(self,stage_id):
        if self.index>=len(STAGES) or stage_id!=STAGES[self.index]['id']:
            raise ValueError('采集阶段已更新，请按最新提示操作')

    def cancel(self):
        self.status = 'cancelled'; self.request_id = None

    def retry(self):
        if self.status not in ('collecting', 'ready', 'paused', 'cancelled'):
            raise ValueError('当前动作不能重试')
        self.rows = self.rows[:self.stage_start]
        self.status = 'ready'; self.started = None; self.last_stamp = None; self.elapsed = 0.
        self.error = None; self.request_id = uuid.uuid4().hex
        self.segment_offset=0.

    def pause_cycle(self,message):
        stage=STAGES[self.index];duration=stage['seconds']/stage['cycles']
        cycle=min(int(self.elapsed/duration),stage['cycles']-1)
        self.rows=self.rows[:self.stage_start]+[r for r in self.rows[self.stage_start:] if r.get('cycle',0)<cycle]
        self.elapsed=cycle*duration;self.segment_offset=self.elapsed
        self.started=None;self.last_stamp=None;self.status='paused';self.error=message
        self.recovered_cycles+=1;self.request_id=uuid.uuid4().hex

    def cycle_error(self,group,stage):
        duration=stage['seconds']/stage['cycles']
        if len(group)<duration*15:return '本轮有效采样不足；请继续本轮，已完成轮次保留'
        if stage['id'].startswith('pinch_'):
            if sum(r.get('contact_confirmed',False) for r in group)<3:return '本轮未记录到足够的双侧轻触；请检查触觉去皮并重做本轮'
        elif stage['id'].startswith('range_') or stage['id']=='thumb_opposition':
            finger=stage['id'][6:] if stage['id'].startswith('range_') else 'thumb'
            samples=[r['fingers'].get(finger) for r in group]
            if not samples or not all(samples):return '本轮活动范围观测缺失，请继续本轮'
            first=samples[0];distance=max(math.dist(s['position'],first['position']) for s in samples)
            dot=min(abs(sum(a*b for a,b in zip(s['quaternion_xyzw'],first['quaternion_xyzw']))) for s in samples)
            angle=2*math.acos(min(1.,max(0.,dot)))
            if distance<.005 and angle<math.radians(15):return '本轮活动范围不足；请按示意图缓慢屈伸后重做本轮'
        return None

    def feed(self, frame):
        if self.status != 'collecting': return False
        if frame.get('calibration_fingerprint') != self.metadata['calibration_fingerprint']:
            self.cancel(); self.error = '骨架初始化已改变，请重新开始连续标定'; return True
        if frame.get('mapping', {}).get('assignments') != self.rows[0]['mapping']['assignments']:
            self.cancel(); self.error = '采集期间 SID 分区改变，请重新开始'; return True
        stamp = frame['timestamp_ms']
        if not math.isfinite(stamp) or self.last_stamp is not None and stamp-self.last_stamp < 19: return False
        if self.last_stamp is not None and stamp-self.last_stamp > 250:
            self.pause_cycle('原始采集流出现缺口；点击继续，仅重做当前一轮，已完成轮次保留')
            return True
        self.last_stamp = stamp
        if self.started is None: self.started = stamp
        stage=STAGES[self.index];elapsed=self.segment_offset+(stamp-self.started)/1000
        duration=stage['seconds']/stage['cycles']
        previous_cycle=min(int(self.elapsed/duration),stage['cycles']-1)
        if elapsed>=(previous_cycle+1)*duration:
            group=[r for r in self.rows[self.stage_start:] if r.get('cycle')==previous_cycle]
            message=self.cycle_error(group,stage)
            if message:
                self.pause_cycle(message);return True
        self.elapsed=elapsed
        if self.elapsed >= stage['seconds']:
            self.complete_groups.append(stage['id']); self.metadata['completed_motion_stages'].append(stage['id'])
            self.index += 1; self.stage_start = len(self.rows); self.started = None; self.last_stamp = None
            self.elapsed=0.;self.segment_offset=0.
            self.status = 'complete' if self.index == len(STAGES) else 'ready'
            self.request_id = None if self.status == 'complete' else uuid.uuid4().hex
            self.metadata['frames'] = len(self.rows)
            return True
        cycle_seconds = stage['seconds']/stage['cycles']
        cycle = min(int(self.elapsed/cycle_seconds), stage['cycles']-1)
        t = self.elapsed-cycle*cycle_seconds
        phase=phase_at(stage,self.elapsed)
        row = copy.deepcopy(frame)
        row.update(stage=stage['id'], capture_protocol=PROTOCOL, initialization=False,
                   cycle=cycle, motion_phase=phase, guided_contact=False,
                   partition='validation' if cycle == 2 or stage['id'].endswith('_repeat') else 'train')
        row['expected_contact_regions']=contact_target(stage['id'])
        row['contact_confirmed'] = phase == 'touch' and bilateral(row, stage['id'][6:])
        row['guided_contact'] = row['contact_confirmed']
        row['load_role'] = 'onset' if row['contact_confirmed'] else 'approach'
        row['contact_constraint_source'] = 'instructed_hold_and_bilateral_tactile' if row['contact_confirmed'] else None
        self.rows.append(row)
        return False

    def snapshot(self):
        stage = STAGES[min(self.index, len(STAGES)-1)]
        duration = sum(s['seconds'] for s in STAGES)
        done = sum(s['seconds'] for s in STAGES[:self.index])+min(self.elapsed,stage['seconds']) if self.status != 'complete' else duration
        return dict(state=self.status, stage_id=stage['id'] if self.status != 'complete' else None,
                    prompt=stage['prompt'], request_id=self.request_id, index=self.index,
                    total=len(STAGES), samples=len(self.rows), progress=min(done/duration,1.),
                    seconds_remaining=max(0.,stage['seconds']-self.elapsed),
                    estimated_remaining_s=max(0.,duration-done),
                    cycle=min(int(self.elapsed/(stage['seconds']/stage['cycles']))+1,stage['cycles']),
                    motion_phase=phase_at(stage,self.elapsed) if self.status in ('collecting','paused') else None,
                    cycle_seconds=stage['seconds']/stage['cycles'],cycle_remaining_s=max(0.,stage['seconds']/stage['cycles']-self.elapsed%(stage['seconds']/stage['cycles'])),
                    retained_cycles=min(int(self.elapsed/(stage['seconds']/stage['cycles'])),stage['cycles']),recovered_cycles=self.recovered_cycles,
                    error=self.error, completed_groups=self.complete_groups, source='continuous_motion', metadata=self.metadata)

    def document(self):
        return {**self.metadata,'schema_version':2,'rows':self.rows,
                'completed_groups':self.complete_groups,'state':self.snapshot()}


def validate_document(document):
    if document.get('protocol')=='contact_motion_v3':
        from .contact_capture import validate_document as validate_contact
        return validate_contact(document)
    if document.get('protocol') != PROTOCOL or document.get('state', {}).get('state') != 'complete':
        raise ValueError('连续标定尚未完成')
    if set(document.get('completed_motion_stages', [])) != {s['id'] for s in STAGES}:
        raise ValueError('连续标定缺少动作')
    rows = document['rows']
    if len({r['timestamp_ms'] for r in rows}) != len(rows): raise ValueError('重复采集时间戳')
    for stage in STAGES:
        group = [r for r in rows if r['stage'] == stage['id'] and not r.get('initialization')]
        if not group or not any(r['partition']=='validation' for r in group): raise ValueError('缺少独立留出轮次')
        if stage['id'].startswith('pinch_') and any(not any(r['cycle']==c and confirmed_touch(r) for r in group) for c in range(3)):
            raise ValueError('对指轻触证据不完整')
    return document
