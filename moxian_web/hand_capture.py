"""Independent, resumable human-confirmed MANO capture; no automatic pose confirmation."""
from __future__ import annotations
import copy,math,time,uuid
from .hand_model import STAGES,aligned_frame

class HandCapture:
    def __init__(self):
        self.status='idle';self.index=0;self.rows=[];self.samples=[];self.started=None;self.request_id=None;self.error=None;self.window=[];self.angle_window=[];self.cycle=0;self.was_touch=False;self.last_timestamp=None;self.complete_groups=[];self.pending_transition=None
    def start(self):
        self.__init__();self.status='awaiting_confirmation';self.request_id=uuid.uuid4().hex
    def snapshot(self):
        stage,prompt=STAGES[self.index]
        return {'state':self.status,'stage_id':stage,'prompt':prompt,'index':self.index,'total':len(STAGES),'request_id':self.request_id,
                'samples':len(self.samples),'progress':min(1., min(len(self.samples)/(100 if stage.startswith('pinch_') else 25),
                    max(0.,((self.last_timestamp or 0)-(self.started or self.last_timestamp or 0))/(6000 if stage.startswith('pinch_') else 1500)))),
                'cycles':self.cycle,'error':self.error,'completed_groups':list(self.complete_groups),
                'expected_patches':['thumb_distal',stage[6:]+'_distal'] if stage.startswith('pinch_') else []}
    def confirm(self,rid):
        if self.status!='awaiting_confirmation' or rid!=self.request_id:raise ValueError('手型动作确认已过期')
        self.status='collecting';self.samples=[];self.window=[];self.angle_window=[];self.started=None;self.cycle=0;self.was_touch=False;self.error=None;self.pending_transition=None;self.last_timestamp=None
    def retry(self,rid):
        if rid!=self.request_id or self.status not in ('collecting','awaiting_confirmation','failed'):raise ValueError('手型阶段已改变')
        self.samples=[];self.rows=[r for r in self.rows if r['stage']!=STAGES[self.index][0]];self.status='awaiting_confirmation';self.request_id=uuid.uuid4().hex;self.error=None
    def cancel(self):
        self.status='cancelled';self.request_id=None
    def feed(self,frame):
        if self.status!='collecting':return False
        if not aligned_frame(frame) or not frame.get('contact',{}).get('ready') or not frame.get('skeleton') or abs(frame['skeleton'].get('timestamp_ms',0)-frame['timestamp_ms'])>25:
            self.error='等待有效定位、去皮与时间匹配（≤25 ms）';return False
        t=frame['timestamp_ms']
        if self.last_timestamp is not None and t<=self.last_timestamp:return False
        if self.last_timestamp is not None and t-self.last_timestamp>250:
            gap=t-self.last_timestamp
            self.samples=[];self.window=[];self.angle_window=[];self.started=None
            self.pending_transition=None;self.cycle=0;self.was_touch=False
            if gap>2000:
                self.status='failed';self.error='有效数据中断超过 2 秒，重试当前动作；此前各组保留';return False
            self.last_timestamp=t
            self.error='短暂缺帧已恢复，正在重新取得连续稳定段';return False
        self.last_timestamp=t
        if self.started is None:self.started=t
        stage=STAGES[self.index][0];pinch=stage.startswith('pinch_')
        positions=[v for f in frame['fingers'].values() for v in f['position']]
        self.window.append(positions);self.window=self.window[-10:]
        angles=[f['quaternion_xyzw'] for f in frame['fingers'].values()]
        self.angle_window.append(angles);self.angle_window=self.angle_window[-10:]
        if not pinch and len(self.window)==10:
            columns=list(zip(*self.window));variance=max(sum((x-sum(c)/len(c))**2 for x in c)/len(c) for c in columns)
            rotation=max(2*math.acos(min(1.,abs(sum(a*b for a,b in zip(q,self.angle_window[0][f]))))) for sample in self.angle_window for f,q in enumerate(sample))
            if variance>.002**2 or rotation>math.radians(5):
                self.samples=[];self.started=t;self.error='动作仍在移动，请保持稳定';return False
        self.error=None
        if pinch:
            ps={p['id']:p for p in frame['contact']['patches']}
            touching=bool(ps.get('thumb_distal',{}).get('contact') and ps.get(stage[6:]+'_distal',{}).get('contact'))
            if touching!=self.was_touch:
                if self.pending_transition is None:self.pending_transition=t
                if t-self.pending_transition>=100:
                    if self.was_touch:self.cycle+=1
                    self.was_touch=touching;self.pending_transition=None
            else:self.pending_transition=None
        saved=copy.deepcopy(frame);saved.update(stage=stage,cycle=self.cycle)
        self.samples.append(saved)
        # Keep capture sampling near 20 Hz, independently of source frequency.
        required=6000 if pinch else 1500
        done=t-self.started>=required and len(self.samples)>=(100 if pinch else 25) and (not pinch or self.cycle>=2)
        if t-self.started>60000:
            self.status='failed';self.error='60 秒内未取得稳定段或两次接触释放，请重试';return False
        if done:
            self.rows.extend(self.samples);self.complete_groups.append(stage)
            if self.index==len(STAGES)-1:self.status='complete';self.request_id=None
            else:self.index+=1;self.status='awaiting_confirmation';self.request_id=uuid.uuid4().hex;self.samples=[]
        return done
