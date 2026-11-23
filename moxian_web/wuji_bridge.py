"""Bounded, fault-isolated transport for the virtual WUJI2 retarget worker."""
import copy
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

from .skeleton import SkeletonBridge
from .hand_model import HandModelBridge
from .wuji_transport import unpack_result_geometry
from .wuji_backends import SOLVERS, DEFAULT_SOLVER

# The input reader runs independently of CEM. Source timestamps also include
# upstream MANO/GPU latency; they are not a 500 ms transport heartbeat.
INPUT_HEARTBEAT_TIMEOUT_MS=1500
MIN_RESULT_TIMEOUT_MS=5000
MAX_RESULT_TIMEOUT_MS=15000


class WujiBridge(SkeletonBridge):
    def control(self,action,data):
        if action=='solver_set' and data.get('solver') not in SOLVERS:raise ValueError('Unknown Wuji2 solver')
        result=HandModelBridge.control(self,action,data)
        if action=='solver_set':self.backend=data['solver']
        if action=='model_set' and data.get('solver')=='cart':self.backend='cart'
        if action in ('solver_set','model_set','rebuild'):self.source_version=None
        return result

    def __init__(self,root,folder):
        super().__init__(root,folder)
        default=self.root/'.venv-contact/bin/python'
        if not default.exists():default=self.root/'.venv-moxian-mano/bin/python'
        self.python=Path(os.environ.get('MOXIAN_MANO_PYTHON',str(default)));self.topology=None
        self.source_version=None;self.last_source_stamp=None;self.last_dispatch=0.
        self.source_epoch=0;self.last_input_stamp=None
        self.backend=os.environ.get('MOXIAN_WUJI_SOLVER',DEFAULT_SOLVER)
        if self.backend not in SOLVERS:
            self.backend=DEFAULT_SOLVER

    def observe_source_input(self,stamp):
        """Observe the raw glove stream, independently of expensive solves."""
        if stamp==self.last_input_stamp:return
        if self.last_input_stamp is not None and not 0<stamp-self.last_input_stamp<=250:self.source_epoch+=1
        self.last_input_stamp=stamp

    def start(self,serial):
        if self.proc and self.proc.poll() is None and self.serial==serial:return
        self.stop();self.serial=serial;self.source_version=None;self.last_source_stamp=None;self.topology=None
        self.folder.mkdir(parents=True,exist_ok=True)
        self.log=(self.folder/'wuji-worker.log').open('a')
        read_fd,write_fd=os.pipe();key=hashlib.sha256(serial.encode()).hexdigest()[:16]
        env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME')}
        env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        try:
            proc=subprocess.Popen([str(self.python),'-I',str(self.root/'moxian_web/wuji_worker.py'),
                                   '--event-fd',str(write_fd),'--home',str(self.folder/'wuji2'/key),'--solver',self.backend,
                                   '--geort-model',getattr(self,'geort_model_path',''),
                                   '--cart-model',getattr(self,'cart_model_path','')],
                                  stdin=subprocess.PIPE,stdout=self.log,stderr=self.log,text=True,bufsize=1,pass_fds=(write_fd,),env=env)
        except Exception:
            os.close(read_fd);raise
        finally:os.close(write_fd)
        self.proc=proc
        with self.lock:self.state={'status':'starting','ready':False,'error':None,'requested_solver_backend':self.backend}
        self.threads=[threading.Thread(target=self._read,args=(read_fd,proc,self.generation),daemon=True),
                      threading.Thread(target=self._write,args=(proc,self.generation),daemon=True)]
        for t in self.threads:t.start()

    def dispatch(self,hand):
        now=time.monotonic()
        if now-self.last_dispatch<1/120:return
        self.last_dispatch=now
        state=hand.snapshot(copy_state=False,include_audit=False,mesh_transport=True)
        if not state.get('ready'):return
        stamp=state.get('timestamp_ms')
        if stamp==self.last_source_stamp:return
        version=(state.get('geometry_version'),state.get('binding_version'),state.get('model_hash'))
        if version!=self.source_version:
            topology=hand.model()
            if not topology.get('rest_vertices_m'):return
            try:self.controls.put_nowait({'type':'topology','model':topology})
            except queue.Full:return
            self.source_version=version
        fields=('emf_observation','source_joint_positions_rad','vertices_m','joints_m','timestamp_ms','skeleton_timestamp_ms','tactile_timestamp_ms','source_contact','geometry_version','binding_version','model_hash','surface_backend','quality','hand_side','object_transform','self_contact_pairs','tactile_pose_correction','pose_calibration_version','skeleton_runtime_revision','contact_registration','calibrated','sensor_mounting','calibration_fingerprint','skeleton_runtime_revision','preview')
        frame={k:state.get(k) for k in fields}
        frame['serial']=self.serial
        if self.last_input_stamp is not None:frame['input_stream_epoch']=self.source_epoch
        if state.get('surface_backend')=='qnbot' and state.get('mesh_frame'):
            payload=hand.mesh_frame(sequence=state['mesh_frame']['sequence'])
            if not payload:return
            frame['mesh_f32_b64']=base64.b64encode(payload).decode('ascii')
            frame.pop('vertices_m',None);frame.pop('joints_m',None)
        self.send({'type':'frame','frame':frame})
        self.last_source_stamp=stamp

    def _read(self,fd,proc,generation):
        try:
            with os.fdopen(fd) as stream:
                for line in stream:
                    if generation!=self.generation:return
                    m=json.loads(line)
                    with self.lock:
                        if m.get('type')=='ack':
                            reply=self.pending.get(m.get('id'))
                            if reply:reply.put_nowait(m)
                        elif m.get('type')=='topology':self.topology=m['model']
                        elif m.get('type')=='frame':
                            q=m.get('qpos',[])
                            if len(q)!=26 or not all(math.isfinite(v) for v in q):raise ValueError('invalid WUJI2 solution')
                            heartbeat={k:self.state[k] for k in ('input_received_wall_ms','input_timestamp_ms') if k in self.state}
                            self.state={**heartbeat,**{k:v for k,v in m.items() if k!='type'}};self.pose_at=time.monotonic()
                        else:self.state.update({k:v for k,v in m.items() if k!='type'})
        except Exception as exc:
            if generation==self.generation:
                with self.lock:self.state.update(status='error',ready=False,error=str(exc))

    def model(self):
        with self.lock:return copy.deepcopy(self.topology) or {'ready':False}

    def snapshot(self,connected=True,copy_state=True,mesh_transport=False):
        with self.lock:published=dict(self.state);received=self.pose_at
        r=copy.deepcopy(published) if copy_state else published
        age=(time.monotonic()-received)*1000 if received else None
        source_age=time.time()*1000-r.get('source_timestamp_ms',0) if r.get('source_timestamp_ms') else None
        if self.proc and self.proc.poll() is not None:r.update(status='error',ready=False,error='Wuji2 worker 已退出；原骨架与触觉仍可使用')
        r['age_ms']=age;r['source_age_ms']=source_age
        if r.get('solver_backend') in ('geort','cart','wuji-retargeting'):
            playback=(r.get('playback') or {}).get('active',False)
            alive=not self.proc or self.proc.poll() is None
            cart=r.get('solver_backend')=='cart'
            valid=bool(alive and r.get('status') in (('ready','solving','infeasible') if cart else ('ready','solving')) and r.get('display_available') and
                       (playback or connected and age is not None and age<(3000 if cart else 500) and source_age is not None and source_age<(5000 if cart else 1000)))
            r.update(ready=valid,display_available=valid,contact_ready=False,contact_mask=[None]*460,input_live=bool(valid and not playback))
            return r if mesh_transport else unpack_result_geometry(r)
        if 'input_received_wall_ms' in r:
            input_age=time.time()*1000-r['input_received_wall_ms']
            input_stamp=r.get('input_timestamp_ms')
            source_input_age=time.time()*1000-input_stamp if input_stamp else None
            processing=r.get('quality',{}).get('processing_ms',0)
            if not isinstance(processing,(int,float)) or not math.isfinite(processing):processing=0
            budget=min(MAX_RESULT_TIMEOUT_MS,max(MIN_RESULT_TIMEOUT_MS,3*processing))
            input_live=bool(0<=input_age<INPUT_HEARTBEAT_TIMEOUT_MS and source_input_age is not None and 0<=source_input_age<budget)
            available=bool(connected and input_live and age is not None and age<budget and r.get('status') in ('ready','solving','infeasible') and (not self.proc or self.proc.poll() is None))
            r['ready']=bool(available and r.get('solution_feasible'))
            r['display_available']=bool(available and r.get('display_available'))
            r.update(input_live=bool(connected and input_live),input_age_ms=input_age,
                     input_source_age_ms=source_input_age,result_timeout_ms=budget,
                     result_delayed=bool(source_age is not None and source_age>250))
        else:
            r['ready']=bool(connected and r.get('ready') and age is not None and age<250 and source_age is not None and source_age<250)
        if not r['ready']:r.update(contact_ready=False,contact_mask=[None]*460)
        return r if mesh_transport else unpack_result_geometry(r)
