"""Bounded, incremental telemetry kept outside high-frequency pose messages."""
import copy,math,threading,uuid

class TrainingTrace:
    def __init__(self,capacity=4096):
        from collections import deque
        self.lock=threading.Lock();self.points=deque(maxlen=capacity)
        self.run_id=None;self.sequence=0;self.seen=None;self.status='idle';self.method=None

    def ingest(self,fit):
        if not isinstance(fit,dict):return
        run_id=fit.get('run_id')
        if not run_id:return
        with self.lock:
            if run_id!=self.run_id:
                self.points.clear();self.sequence=0;self.seen=None;self.run_id=run_id
            self.status=fit.get('state',self.status);self.method=fit.get('method',self.method)
            point=fit.get('training_sample')
            if not isinstance(point,dict) or point.get('sample_id')==self.seen:return
            def finite(value):
                if isinstance(value,float):return math.isfinite(value)
                if isinstance(value,dict):return all(finite(v) for v in value.values())
                if isinstance(value,list):return all(finite(v) for v in value)
                return isinstance(value,(str,int,bool,type(None)))
            if not finite(point):return
            self.seen=point.get('sample_id');self.sequence+=1
            self.points.append({**copy.deepcopy(point),'sequence':self.sequence})

    def snapshot(self,after=0,run_id=None):
        with self.lock:
            reset=run_id!=self.run_id or bool(self.points and after<self.points[0]['sequence']-1) or after>self.sequence
            points=[copy.deepcopy(p) for p in self.points if reset or p['sequence']>after]
            return dict(run_id=self.run_id,method=self.method,status=self.status,cursor=self.sequence,reset=reset,points=points,
                        kl_definition='KL(N(beta,I)||N(0,I)) = 0.5*||beta||²; geometry uses GEOMETRY_SIGMA; deterministic prior diagnostics, not a policy KL')
