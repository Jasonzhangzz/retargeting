"""SDK-independent MANO IPC, capture contracts and relative contact mechanics."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
import queue
import subprocess
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from .skeleton import SkeletonBridge

DEFAULT_MANO = Path('/home/zz/GraspSONIC/thirdparty/DexForge/assets/mano_v1_2/models/MANO_LEFT.pkl')
STAGES = [
    ('open','五指自然张开，保持 1.5 秒。'), ('together','手指伸直并拢，保持 1.5 秒。'),
    ('mcp90','四指根关节约 90°，其余指节伸直，保持 1.5 秒。'),
    ('hook','四指弯成钩状，指根尽量伸直，保持 1.5 秒。'),
    ('fist','自然虚握拳，不用力挤压，保持 1.5 秒。'),
    *[(f'pinch_{f}',f'拇指与{label}指腹：轻触→稍加压→释放，缓慢重复两次（至少 6 秒）。')
      for f,label in [('index','食指'),('middle','中指'),('ring','无名指'),('pinky','小指')]],
    ('open_repeat','再次自然张开，保持 1.5 秒；本组只作留出验证。')]
DEFAULT_MATERIAL = {'young_pa':30000.,'thickness_m':.003,'max_compression_m':.002}

def material_config(value):
    result={**DEFAULT_MATERIAL, **value}
    limits={'young_pa':(1000.,1000000.),'thickness_m':(.0005,.02),'max_compression_m':(0.,.002)}
    if set(result)!=set(limits):raise ValueError('未知材料参数')
    for name,(lo,hi) in limits.items():
        v=result[name]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not lo<=v<=hi:
            raise ValueError(f'{name} 超出允许范围 {lo}–{hi}')
    return result

def pair_mechanics(gap, area_a, area_b, material):
    """Quasi-static unilateral normal approximation, NOT full ComFree dynamics."""
    m=material_config(material)
    if not all(math.isfinite(x) for x in (gap,area_a,area_b)) or min(area_a,area_b)<=0:
        raise ValueError('接触面积必须为正的平方米')
    ka=m['young_pa']*area_a/m['thickness_m'];kb=m['young_pa']*area_b/m['thickness_m']
    k=ka*kb/(ka+kb);delta=max(-gap,0.)
    return {'stiffness_n_m':k,'model_force_n':k*delta,
            'compression_a_m':min(m['max_compression_m'],delta*kb/(ka+kb)),
            'compression_b_m':min(m['max_compression_m'],delta*ka/(ka+kb))}

def positive_gain(predicted, observed):
    if len(predicted)!=len(observed) or not predicted:return None
    if not all(math.isfinite(v) and v>=0 for v in (*predicted,*observed)):raise ValueError('无效的相对响应')
    denom=sum(v*v for v in predicted)
    return max(0.,sum(a*b for a,b in zip(predicted,observed))/denom) if denom>1e-12 else None

def aligned_frame(frame):
    try:
        t=float(frame['timestamp_ms']);tt=float(frame['tactile_timestamp_ms'])
        if frame.get('hand_side')!='left' or not math.isfinite(t+tt) or abs(t-tt)>25:return False
        if len(frame['raw'])!=460 or not all(math.isfinite(v) and 0<=v<=255 for v in frame['raw']):return False
        for name in ('thumb','index','middle','ring','pinky'):
            f=frame['fingers'][name];p=f['position'];q=f['quaternion_xyzw']
            if len(p)!=3 or len(q)!=4 or not all(math.isfinite(v) for v in (*p,*q)):return False
            if abs(sum(v*v for v in q)-1)>.05:return False
        return True
    except (KeyError,ValueError,TypeError):return False

class HandInputSynchronizer:
    """Join asynchronous IK to the original sensor observation, never the latest one."""
    def __init__(self):
        self.frames=deque(maxlen=180)
        self.last_sent=-float('inf')

    def push(self,frame):
        self.frames.append(frame)

    def match(self,skeleton,now_ms):
        if not skeleton or not self.frames:return None
        stamp=skeleton['timestamp_ms']
        frame=min(self.frames,key=lambda f:abs(f['timestamp_ms']-stamp))
        if (abs(frame['timestamp_ms']-stamp)>25 or
            now_ms-frame['timestamp_ms']>200 or
            frame['timestamp_ms']<=self.last_sent or not aligned_frame(frame)):
            return None
        self.last_sent=frame['timestamp_ms']
        return {**frame,'skeleton':skeleton}

def fusion_state(geometry_valid, gap, facing, gesture_allowed, touch_a, touch_b, ready=True, guided=False):
    if not ready:return 'unknown'
    near=geometry_valid and math.isfinite(gap) and gap<=(.020 if guided else .005) and facing<-.15 and gesture_allowed
    if near and (touch_a or touch_b):return 'confirmed' if touch_a and touch_b else 'candidate'
    if near:return 'candidate'
    return 'conflict' if touch_a or touch_b else 'none'

def validate_hand_output(message):
    v=message.get('vertices_m');j=message.get('joints_m')
    if message.get('hand_side')!='left' or message.get('coordinate_frame') not in ('wrist_local','mano_wrist_local'):raise ValueError('MANO 坐标系／手侧错误')
    if not isinstance(v,list) or len(v)!=778 or not isinstance(j,list) or len(j)!=21:raise ValueError('MANO 节点数错误')
    if not all(len(p)==3 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in p) for p in v+j):raise ValueError('MANO 输出含无效顶点')
    if not math.isfinite(message.get('timestamp_ms',float('nan'))):raise ValueError('MANO 缺少时间戳')
    if max(abs(x) for p in v for x in p)>.6:raise ValueError('MANO 单位异常，必须为米')

class HandModelBridge(SkeletonBridge):
    """Reuse bounded transport, with a separate interpreter and event validator."""
    def __init__(self,root,folder):
        super().__init__(root,folder)
        default_python=self.root/'.venv-contact/bin/python'
        if not default_python.exists():default_python=self.root/'.venv-moxian-mano/bin/python'
        self.python=Path(os.environ.get('MOXIAN_MANO_PYTHON',str(default_python)))
        self.model_path=Path(os.environ.get('MOXIAN_MANO_MODEL',str(DEFAULT_MANO)))
        self.topology=None
        self.last_sent=0.

    def start(self,serial):
        if self.proc and self.proc.poll() is None and self.serial==serial:return
        self.stop();self.serial=serial;self.topology=None
        if not self.python.is_file() or not self.model_path.is_file():
            with self.lock:self.state.update(status='unavailable',error='缺少 MANO_LEFT.pkl 或 CUDA 环境，请执行 setup_moxian_mano.sh')
            return
        self.folder.mkdir(parents=True,exist_ok=True)
        self.log=(self.folder/'hand-model-worker.log').open('a')
        r,w=os.pipe();key=hashlib.sha256(serial.encode()).hexdigest()[:16]
        env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME')}
        env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        args=[str(self.python),'-I',str(self.root/'moxian_web/hand_worker.py'),'--event-fd',str(w),
              '--home',str(self.folder/'hand_model'/key),'--serial',serial,'--model',str(self.model_path)]
        try:proc=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=self.log,stderr=self.log,text=True,bufsize=1,pass_fds=(w,),env=env)
        except Exception:
            os.close(r);self.log.close();self.log=None;raise
        finally:os.close(w)
        self.proc=proc;self.pose_at=0
        with self.lock:self.state={'status':'starting','error':None,'calibrated':False,'capture':None,'fit':None}
        generation=self.generation
        self.threads=[threading.Thread(target=self._read,args=(r,proc,generation),daemon=True),threading.Thread(target=self._write,args=(proc,generation),daemon=True)]
        for t in self.threads:t.start()

    def control(self,action,data):
        if not self.proc or self.proc.poll() is not None:raise ValueError('手部模型 worker 未启动')
        rid=uuid.uuid4().hex;reply=queue.Queue(maxsize=1)
        with self.lock:self.pending[rid]=reply
        try:
            self.controls.put_nowait({'type':'control','action':action,'id':rid,'data':data})
            result=reply.get(timeout=4)
            if result.get('error'):raise ValueError(result['error'])
            return result.get('result',{'ok':True})
        except (queue.Empty,queue.Full):raise ValueError('手部模型控制超时，请等待初始化完成')
        finally:
            with self.lock:self.pending.pop(rid,None)

    def _read(self,fd,proc,generation):
        try:
            with os.fdopen(fd) as stream:
                for line in stream:
                    if generation!=self.generation:return
                    m=json.loads(line)
                    if m.get('type')=='ack':
                        with self.lock:q=self.pending.get(m.get('id'))
                        if q:q.put_nowait(m)
                    elif m.get('type')=='topology':
                        with self.lock:self.topology=m['model']
                    else:
                        if 'vertices_m' in m:validate_hand_output(m)
                        with self.lock:
                            if 'vertices_m' in m:self.pose_at=time.monotonic()
                            self.state.update({k:v for k,v in m.items() if k!='type'})
        except Exception as exc:
            if generation==self.generation:
                with self.lock:self.state.update(status='error',error=str(exc))

    def model(self):
        with self.lock:return copy.deepcopy(self.topology) or {'ready':False}

    def snapshot(self,connected=True):
        with self.lock:
            r=copy.deepcopy(self.state);age=(time.monotonic()-self.pose_at)*1000 if self.pose_at else None
        if self.proc and self.proc.poll() is not None:r.update(status='error',error=r.get('error') or 'MANO worker 已退出，请重启手部模型')
        r['age_ms']=round(age,1) if age is not None else None
        r['ready']=bool(connected and r.get('vertices_m') and age is not None and age<250 and r.get('status') in ('default','calibrated','preview'))
        if not r['ready']:
            r['vertices_m']=None;r['joints_m']=None
            r['fusion']={'ready':False,'status':'unknown','pairs':[],'patches':[]}
        return r


def confirmed_contact_points(fusion):
    """Read-only display markers; no pose update, deformation, or force."""
    if not fusion.get('ready'):return []
    points=[]
    for pair in fusion.get('pairs',[]):
        if pair.get('state')!='confirmed':continue
        for side in ('a','b'):
            position=pair.get('point_'+side+'_m')
            if not isinstance(position,(list,tuple)) or len(position)!=3 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in position):continue
            points.append({'position_m':list(position),'patch':pair['patch_'+side],
                           'other_patch':pair['patch_'+('b' if side=='a' else 'a')],'state':'confirmed'})
    return points
