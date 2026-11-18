"""Bounded nonblocking IPC for the CPU contact predictor; stdlib only."""
import copy,json,os,queue,subprocess,threading,time,uuid
from pathlib import Path
from .skeleton import SkeletonBridge


class ContactPredictionBridge(SkeletonBridge):
    def __init__(self,root,folder):
        super().__init__(root,folder)
        self.python=Path(os.environ.get('MOXIAN_CONTACT_PYTHON',str(self.root/'.venv-contact/bin/python')))
        self.checkpoint=Path(os.environ.get('MOXIAN_CONTACT_CHECKPOINT',str(self.root/'checkpoints/mano_contact_full_20261110/model.pt')))
        self.last_sent=None;self.history_length=8

    def start(self):
        if self.proc and self.proc.poll() is None:return
        self.stop();self.last_sent=None
        if not self.python.is_file() or not self.checkpoint.is_file():
            self.state=dict(status='unavailable',ready=False,error='缺少接触预测权重或 Python 运行环境');return
        self.folder.mkdir(parents=True,exist_ok=True);self.log=(self.folder/'contact-prediction-worker.log').open('a')
        r,w=os.pipe();env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME')}
        env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
        try:proc=subprocess.Popen([str(self.python),'-I',str(self.root/'moxian_web/contact_prediction_worker.py'),
                '--event-fd',str(w),'--checkpoint',str(self.checkpoint)],stdin=subprocess.PIPE,stdout=self.log,stderr=self.log,
                text=True,bufsize=1,pass_fds=(w,),env=env)
        except Exception:os.close(r);raise
        finally:os.close(w)
        self.proc=proc;self.state=dict(status='starting',ready=False,error=None);generation=self.generation
        self.threads=[threading.Thread(target=self._read,args=(r,proc,generation),daemon=True),
                      threading.Thread(target=self._write,args=(proc,generation),daemon=True)]
        for thread in self.threads:thread.start()

    def _read(self,fd,proc,generation):
        try:
            with os.fdopen(fd) as stream:
                for line in stream:
                    if generation!=self.generation:return
                    value=json.loads(line)
                    with self.lock:
                        if value.get('type')=='ack':
                            reply=self.pending.get(value.get('id'))
                            if reply:reply.put_nowait(value)
                        else:self.state=value;self.pose_at=time.monotonic()
        except Exception as exc:
            with self.lock:self.state=dict(status='error',ready=False,error=str(exc))

    def dispatch(self,hand):
        if not self.proc or self.proc.poll() is not None:return
        keys=('timestamp_ms','hand_side','coordinate_frame','raw_official_pose_axis_angle','source_contact',
              'prediction_reference','contact_registration','geometry_version','binding_version','skeleton_runtime_revision','pose_calibration_version')
        # Do not copy mesh, calibration reports or run calibration_audit in the
        # serial loop. The numerical input is the same raw pose/ref packet.
        with hand.lock:
            state=hand.state
            if state.get('surface_backend')=='qnbot':
                with self.lock:self.state=dict(status='unsupported_surface',ready=False,message='历史接触预测网络仅适配MANO；官方mesh上的实测触觉仍直接显示')
                return
            if not hand.pose_at or time.monotonic()-hand.pose_at>=.25 or state.get('status') not in ('default','calibrated','preview'):return
            frame=copy.deepcopy({k:state[k] for k in keys if k in state})
        stamp=frame.get('timestamp_ms')
        identity=(stamp,frame.get('geometry_version'),frame.get('binding_version'))
        if stamp is None or identity==self.last_sent:return
        with self.lock:version=self.state.get('binding_version')
        topology=hand.model() if version!=frame.get('binding_version') else None
        if topology is not None and not topology.get('ready'):return
        # Only pose and matched reference fields enter the worker. The model
        # callable itself receives strictly pose51, never these reference labels.
        self.send(dict(type='frame',frame=frame,topology=topology,history_length=self.history_length))
        self.last_sent=identity

    def reset_metrics(self):
        if not self.proc or self.proc.poll() is not None:raise ValueError('预测进程未启动')
        rid=uuid.uuid4().hex;reply=queue.Queue(maxsize=1)
        with self.lock:self.pending[rid]=reply
        try:
            self.controls.put_nowait(dict(type='control',id=rid));return reply.get(timeout=4).get('result')
        finally:
            with self.lock:self.pending.pop(rid,None)

    def snapshot(self,connected=True,copy_state=True):
        with self.lock:published=dict(self.state);received=self.pose_at
        result=copy.deepcopy(published) if copy_state else published
        age=(time.monotonic()-received)*1000 if received else None
        if self.proc and self.proc.poll() is not None:result.update(status='error',error='接触预测进程已退出')
        result['ready']=bool(connected and result.get('ready') and age is not None and age<500 and result.get('status')=='ready')
        result['age_ms']=age
        if not connected:result.update(ready=False,status='waiting_device')
        elif age is not None and age>=500:result.update(ready=False,status='stale')
        return result
