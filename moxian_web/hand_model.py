"""SDK-independent MANO IPC, capture contracts and relative contact mechanics."""
from __future__ import annotations
import copy
import base64
import struct
import sys
import array
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

def tracking_geometry_hash(saved):
    model=saved.get('skeleton_seed_model') if saved.get('runtime_backend') in ('gpu_mano_v1','fixed_mano_v1') else saved.get('skeleton_model')
    return (model or {}).get('geometry_hash')


def tracking_registration_matches(saved,frame,pending=False):
    """An unchanged capture does not imply an unchanged native IK model."""
    if not saved or saved.get('calibration_fingerprint')!=frame.get('calibration_fingerprint'):return False
    runtime=frame.get('skeleton_runtime_revision')
    if runtime and saved.get('skeleton_runtime_revision')!=runtime:return False
    geometry=tracking_geometry_hash(saved)
    return not geometry or pending or geometry==frame.get('skeleton',{}).get('personalized_geometry_hash')
from .skeleton import SkeletonBridge

DEFAULT_MANO = Path('/home/zz/GraspSONIC/thirdparty/DexForge/assets/mano_v1_2/models/MANO_LEFT.pkl')
LEGACY_STAGES = [
    ('open','五指自然张开，保持 1.5 秒。'), ('together','手指伸直并拢，保持 1.5 秒。'),
    ('mcp90','四指根关节约 90°，其余指节伸直，保持 1.5 秒。'),
    ('hook','四指弯成钩状，指根尽量伸直，保持 1.5 秒。'),
    ('fist','自然虚握拳，不用力挤压，保持 1.5 秒。'),
    *[(f'pinch_{f}',f'拇指与{label}指腹：轻触→稍加压→释放，缓慢重复两次（至少 6 秒）。')
      for f,label in [('index','食指'),('middle','中指'),('ring','无名指'),('pinky','小指')]],
    ('open_repeat','再次自然张开，保持 1.5 秒；本组只作留出验证。')]
DEFAULT_MATERIAL = {'young_pa':30000.,'thickness_m':.003,'max_compression_m':.002}
MANO_FIT_REVISION='glove_motion_joint_fit_v21'
MANO_SOLVER_REVISION='mano_contact_graph_mixed_v14'

def calibration_report_valid(report):
    """Every route to an applied mesh must satisfy the same complete gate."""
    required=('accepted','contact_validation_passed','together_validation_passed',
              'flat_validation_passed','penetration_validation_passed',
              'runtime_validation_passed','per_stage_validation_passed','holdout_validation_passed')
    return bool((not report.get('gpu_backend') or report.get('performance_passed') is True and report.get('frozen_forward') is True and report.get('online_optimization') is False and report.get('mjwarp_cross_check',{}).get('query_valid') is True) and
                (report.get('method') not in ('parameters','flashsac') or report.get('frozen_forward') is True and report.get('shape_validation',{}).get('accepted') is True) and all(report.get(k) is True for k in required) and
                report.get('fit_revision')==MANO_FIT_REVISION and
                report.get('solver_revision')==MANO_SOLVER_REVISION and
                report.get('validation_pose_source')=='runtime_mano_sensor_only' and
                report.get('capture_protocol') in ('official_eight_v1','geort_motion_v2','contact_motion_v3') and
                (report.get('capture_protocol') not in ('geort_motion_v2','contact_motion_v3') or
                 all(report.get(k) is True for k in ('independent_cycles_valid','approach_validation_passed','skeleton_validation_passed'))) and
                math.isfinite(report.get('max_penetration_mm',float('inf'))) and
                0<=report['max_penetration_mm']<=2. and
                math.isfinite(report.get('runtime_max_penetration_mm',float('inf'))) and
                0<=report['runtime_max_penetration_mm']<=2.)

def persist_calibrated_profile(home,candidate,automatic=False,previous_profile=None):
    """Commit an accepted model and preserve the previously applied version."""
    report=candidate.get('report',{})
    if not calibration_report_valid(report):
        raise ValueError('标定结果未通过独立动作检查，当前模型保留')
    home=Path(home);home.mkdir(parents=True,exist_ok=True)
    profile=copy.deepcopy(candidate);profile.update(calibrated=True,applied_at_ms=time.time()*1000,revision=uuid.uuid4().hex,automatic_application=bool(automatic))
    def write(path,value):
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2));temp.replace(path)
    if (home/'current.json').exists():write(home/'previous.json',json.loads((home/'current.json').read_text()))
    elif previous_profile is not None:write(home/'previous.json',previous_profile)
    write(home/'current.json',profile)
    return profile

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
        if frame.get('hand_side') not in ('left','right') or not math.isfinite(t+tt) or abs(t-tt)>25:return False
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
            frame['timestamp_ms']<=self.last_sent or
            skeleton.get('hand_side',frame.get('hand_side'))!=frame.get('hand_side') or not aligned_frame(frame)):
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
    if message.get('hand_side') not in ('left','right') or message.get('coordinate_frame') not in ('wrist_local','mano_wrist_local'):raise ValueError('MANO 坐标系／手侧错误')
    native=message.get('surface_backend')=='qnbot'
    if not isinstance(v,list) or len(v)!=(11598 if native else 778) or not isinstance(j,list) or (len(j)!=21 if not native else not 25<=len(j)<=40):raise ValueError('手部mesh节点数错误')
    if not all(len(p)==3 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in p) for p in v+j):raise ValueError('MANO 输出含无效顶点')
    if not math.isfinite(message.get('timestamp_ms',float('nan'))):raise ValueError('MANO 缺少时间戳')
    if max(abs(x) for p in v for x in p)>.6:raise ValueError('MANO 单位异常，必须为米')

def public_fit_report(report,include_training_sample=False):
    """Keep polling cheap; full diagnostics and snapshots stay in report.json."""
    if not isinstance(report,dict):return report
    omitted={'contact_material_records','together_material_records','together_surface_distances_mm',
             'collision_frames','runtime_solver_frames','training_sample','auxiliary_initialization_checks',
             'approach_release_geometry','preferred_thumb_contacts','expected_contact_regions','native_log'}
    if include_training_sample:omitted.remove('training_sample')
    result={key:value for key,value in report.items() if key not in omitted}
    if isinstance(result.get('contact_templates'),dict):
        template=result['contact_templates']
        result['contact_templates']={key:template[key] for key in ('revision','source','validation_used') if key in template}
        edge_fields=('required','template_gap_mm','template_normal_error_deg','template_penetration_mm')
        result['contact_templates']['actions']=[dict(stage=a['stage'],feasible=a.get('feasible',False),
            edges=[{k:e[k] for k in edge_fields if k in e} for e in a.get('edges',[])]) for a in template.get('actions',[])]
    if isinstance(result.get('solver'),dict):
        result['solver']={key:value for key,value in result['solver'].items() if key not in ('parameter_snapshot','training_trace')}
        result['solver']['phases']=[{key:value for key,value in phase.items() if key!='parameter_vector'}
                                  for phase in result['solver'].get('phases',[])]
    for key in ('baseline_fixed','runtime_validation','offline_validation'):
        if isinstance(result.get(key),dict):result[key]={k:v for k,v in result[key].items() if k!='frames'}
    return result


class HandModelBridge(SkeletonBridge):
    """Reuse bounded transport, with a separate interpreter and event validator."""
    def __init__(self,root,folder):
        super().__init__(root,folder)
        default_python=self.root/'.venv-contact/bin/python'
        if not default_python.exists():default_python=self.root/'.venv-moxian-mano/bin/python'
        self.python=Path(os.environ.get('MOXIAN_MANO_PYTHON',str(default_python)))
        self.model_path=Path(os.environ.get('MOXIAN_MANO_MODEL',str(DEFAULT_MANO)))
        self.topology=None
        self.surface_backend=os.environ.get('MOXIAN_SURFACE_BACKEND','qnbot')
        self.mesh_payload=None
        self.mesh_frames=deque(maxlen=8)
        self.last_sent=0.
        self.audit_checked=0.;self.audit_cache=None
        from .training_trace import TrainingTrace
        self.training_trace=TrainingTrace()
        self.training_history={}

    def start(self,serial,side="left",surface_backend=None):
        from .hand_side import check_side
        side=check_side(side)
        backend=surface_backend or self.surface_backend
        if backend not in ('qnbot','mano'):raise ValueError('surface_backend 须为 qnbot 或 mano')
        model_path=self.model_path if side=="left" else Path(os.environ.get("MOXIAN_MANO_MODEL_RIGHT",str(self.model_path.with_name("MANO_RIGHT.pkl"))))
        if self.proc and self.proc.poll() is None and self.serial==serial and self.side==side and self.surface_backend==backend:return
        self.surface_backend=backend;self.mesh_payload=None;self.mesh_frames.clear()
        self.stop();self.serial=serial;self.side=side;self.topology=None;self.audit_checked=0.;self.audit_cache=None
        from .training_trace import TrainingTrace
        self.training_trace=TrainingTrace()
        self.training_history={}
        trace_home=self.folder/'hand_model'/hashlib.sha256(serial.encode()).hexdigest()[:16]/'training-traces'
        if backend=='qnbot':trace_home=trace_home.parent/'qnbot'/'training-traces'
        try:
            latest=json.loads((trace_home/'latest.json').read_text())
            path=(trace_home/latest['file']).resolve()
            if path.parent==trace_home.resolve():
                for line in path.read_text().splitlines():self.training_trace.ingest(json.loads(line))
        except (OSError,ValueError,KeyError,TypeError):pass
        if not self.python.is_file() or backend=='mano' and not model_path.is_file():
            with self.lock:self.state.update(status='unavailable',error='缺少当前手侧的 MANO 权重或 CUDA 环境，请执行 setup_moxian_mano.sh')
            return
        self.folder.mkdir(parents=True,exist_ok=True)
        self.log=(self.folder/'hand-model-worker.log').open('a')
        self.log.write(f'\nMANO worker starting {time.time():.3f} side={side}\n');self.log.flush()
        r,w=os.pipe();key=hashlib.sha256(serial.encode()).hexdigest()[:16]
        env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME')}
        env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        worker='qnbot_worker.py' if backend=='qnbot' else 'hand_worker.py'
        args=[str(self.python),'-I',str(self.root/'moxian_web'/worker),'--event-fd',str(w),
              '--home',str(self.folder/'hand_model'/key),'--serial',serial,'--model',str(model_path),'--side',side]
        try:proc=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=self.log,stderr=self.log,text=True,bufsize=1,pass_fds=(w,),env=env)
        except Exception:
            os.close(r);self.log.close();self.log=None;raise
        finally:os.close(w)
        self.proc=proc;self.pose_at=0
        with self.lock:self.state={'status':'starting','error':None,'calibrated':False,'capture':None,'fit':None,'surface_backend':backend}
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
                        packed=None
                        if 'mesh_f32_b64' in m:
                            descriptor=m.get('mesh_frame') or {};payload=base64.b64decode(m.pop('mesh_f32_b64'),validate=True)
                            nv,nj=descriptor.get('vertex_count'),descriptor.get('joint_count')
                            if nv!=11598 or not isinstance(nj,int) or not 25<=nj<=40 or len(payload)!=(nv+nj)*12:raise ValueError('无效官方mesh二进制帧')
                            if m.get('hand_side')!=self.side or m.get('coordinate_frame')!='wrist_local' or not math.isfinite(m.get('timestamp_ms',float('nan'))):raise ValueError('官方mesh坐标／时间戳错误')
                            points=array.array('f');points.frombytes(payload)
                            if sys.byteorder!='little':points.byteswap()
                            if not all(math.isfinite(x) and abs(x)<=.6 for x in points):raise ValueError('官方mesh含无效顶点或单位异常')
                            descriptor={**descriptor,'header_bytes':128,'format':'qnbot-f32-v1'};m['mesh_frame']=descriptor
                            header=struct.pack('<4sIIIIId',b'QNB1',1,descriptor['sequence'],nv,nj,0,m['timestamp_ms'])
                            header+=m['geometry_version'].encode().ljust(32,b'\0')[:32]+m['binding_version'].encode().ljust(64,b'\0')[:64]
                            packed=header+payload
                        if 'vertices_m' in m:validate_hand_output(m)
                        self.training_trace.ingest(m.get('fit'))
                        if isinstance(m.get('fit'),dict):m['fit']=public_fit_report(m['fit'])
                        with self.lock:
                            if packed is not None:self.mesh_payload=packed;self.mesh_frames.append((m['mesh_frame']['sequence'],packed));self.pose_at=time.monotonic()
                            if 'vertices_m' in m:self.pose_at=time.monotonic()
                            self.state.update({k:v for k,v in m.items() if k!='type'})
        except Exception as exc:
            if generation==self.generation:
                with self.lock:self.state.update(status='error',error=str(exc))

    def model(self):
        with self.lock:return copy.deepcopy(self.topology) or {'ready':False}

    def mesh_frame(self,after=None,sequence=None):
        with self.lock:
            payload=self.mesh_payload
            if not payload:return None
            if sequence is not None:
                for seq,value in reversed(self.mesh_frames):
                    if seq==sequence:return value
            if after is not None and after==struct.unpack_from('<I',payload,8)[0]:return b''
            return payload

    def training_history_snapshot(self,after=0,run_id=None,method=None):
        if not method or method==self.training_trace.method:return self.training_trace.snapshot(after,run_id)
        if method not in ('moxian_wuji','wuji_native','bone_extrinsics','parameters','flashsac','delta','least_squares'):raise ValueError('无效的训练曲线方法')
        from .training_trace import TrainingTrace
        key=hashlib.sha256((self.serial or '').encode()).hexdigest()[:16]
        home=self.folder/'hand_model'/key/'training-traces'
        if self.surface_backend=='qnbot':home=home.parent/'qnbot'/'training-traces'
        try:
            index=home/f'{method}.json';stamp=index.stat().st_mtime_ns
            cached=self.training_history.get(method)
            if not cached or cached[0]!=stamp:
                record=json.loads(index.read_text());path=(home/record['file']).resolve()
                if path.parent!=home.resolve():raise ValueError('无效的曲线记录路径')
                trace=TrainingTrace()
                for line in path.read_text().splitlines():trace.ingest(json.loads(line))
                self.training_history[method]=(stamp,trace)
            return self.training_history[method][1].snapshot(after,run_id)
        except FileNotFoundError:return TrainingTrace().snapshot(after,run_id)

    def calibration_audit(self):
        if self.state.get('surface_backend')=='qnbot':return None
        now=time.monotonic()
        if now-self.audit_checked>=1:
            self.audit_checked=now
            paths=list((self.folder/'hand_model').glob('*/calibration-audit.json'))
            if self.serial:
                key=hashlib.sha256(self.serial.encode()).hexdigest()[:16]
                paths=[p for p in paths if p.parent.name==key]
            try:
                if paths:
                    path=max(paths,key=lambda p:p.stat().st_mtime);audit=json.loads(path.read_text())
                    candidate=path.parent/'candidate.json';current=path.parent/'current.json'
                    if candidate.exists():
                        p=json.loads(candidate.read_text())
                        if p.get('dimension_mode')=='official_mano_betas_v1':
                            audit['candidate_report']=public_fit_report(p.get('report'));audit['candidate_dimensions_mm']=p.get('report',{}).get('dimensions_after_mm')
                        else:
                            audit['candidate_report']=None;audit['candidate_dimensions_mm']=None
                    if current.exists():
                        p=json.loads(current.read_text())
                        if p.get('dimension_mode')!='official_mano_betas_v1':
                            p={};audit['active_dimensions_mm']=None
                        audit.update(dimensions_applied=bool(p.get('calibrated') and calibration_report_valid(p.get('report',{}))),applied_revision=p.get('revision'),
                            active_dimension_parameters=p.get('dimension_parameters'),active_dimensions_mm=p.get('report',{}).get('dimensions_after_mm',audit.get('active_dimensions_mm')))
                    audit['candidate_report']=public_fit_report(audit.get('candidate_report'));self.audit_cache=audit
            except (OSError,ValueError):pass
        return copy.deepcopy(self.audit_cache)

    def snapshot(self,connected=True,copy_state=True,include_audit=True,mesh_transport=False):
        with self.lock:
            published={**self.state,'fit':public_fit_report(self.state.get('fit'))};received=self.pose_at;payload=getattr(self,'mesh_payload',None)
        r=copy.deepcopy(published) if copy_state else published
        age=(time.monotonic()-received)*1000 if received else None
        if self.proc and self.proc.poll() is not None:r.update(status='error',error=r.get('error') or 'MANO worker 已退出，请重启手部模型')
        r['age_ms']=round(age,1) if age is not None else None
        r['ready']=bool(connected and (r.get('vertices_m') or r.get('mesh_frame')) and age is not None and age<250 and r.get('status') in ('default','calibrated','preview'))
        if r['ready'] and r.get('surface_backend')=='qnbot' and payload and not mesh_transport:
            # Legacy/offline recording consumers receive full same-frame data;
            # the interactive page fetches packed Float32 geometry separately.
            import array
            nv,nj=struct.unpack_from('<II',payload,12);points=array.array('f');points.frombytes(payload[128:])
            if sys.byteorder!='little':points.byteswap()
            values=list(points);r['vertices_m']=[values[i:i+3] for i in range(0,nv*3,3)];r['joints_m']=[values[i:i+3] for i in range(nv*3,(nv+nj)*3,3)]
        if not r['ready']:
            r['vertices_m']=None;r['joints_m']=None
            r['fusion']={'ready':False,'status':'unknown','pairs':[],'patches':[]}
        audit=self.calibration_audit() if include_audit else None
        if audit:
            r['calibration_audit']=audit
            if not self.proc or self.proc.poll() is not None:
                r.update(fit=audit.get('candidate_report'),calibrated=audit['dimensions_applied'],
                         dimensions_mm=audit['active_dimensions_mm'],penetration_policy='soft_penalty_only')
                r['calibration_process']={'capture_complete':audit['capture_complete'],'dimensions_applied':audit['dimensions_applied'],
                                          'fit_accepted':bool((audit.get('candidate_report') or {}).get('accepted'))}
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
