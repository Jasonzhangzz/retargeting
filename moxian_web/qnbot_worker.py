"""Native Qnbot fixed-skin display, capture and private offline fitting worker."""
from __future__ import annotations
import argparse,base64,copy,hashlib,json,os,queue,signal,sys,threading,time,traceback,uuid
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moxian_web.qnbot_surface import QnbotSurface,REVISION
from moxian_web.emf_qp import REVISION as QP_REVISION
from moxian_web.qnbot_calibration import fit_capture,load_policy,tactile_features,FIT_REVISION,apply_delta
from moxian_web.qnbot_reference import QnbotReference,observations,initialize_registration
from moxian_web.contact_capture_v4 import ContactTrajectoryCapture
from moxian_web.static_capture import StaticHandCapture
from moxian_web.wuji_native_capture import WujiNativeCapture,PROTOCOL as WUJI_PROTOCOL
from moxian_web.wuji_native_model import validate_native_candidate
from moxian_web.moxian_six_calibration import METHOD as WUJI_METHOD,REVISION as WUJI_FIT_REVISION
ACTIVE_REVISIONS={'moxian_native25_wuji_six_v1','moxian_wuji_six_v2_distal_twist',WUJI_FIT_REVISION}
from moxian_web.bone_extrinsics import METHOD as BONE_METHOD,FIT_REVISION as BONE_FIT_REVISION,publish_active,read_active,validate_candidate
from moxian_web.motion_capture import MotionCapture
from moxian_web.shared_capture import SharedHandCapture,load_official_capture
from moxian_web.background_io import CaptureCheckpoints
from moxian_web.recording import SnapshotRecorder
from moxian_web.hand_model import public_fit_report


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')));tmp.replace(path)


class StreamingModel:
    def __init__(self,model):self.model=model;self.policy=None;self.artifact=None;self.previous=None;self.graph=None;self.graph_error=None;self.warmup_s=0.;self.model_hash=None;self.wuji=None
    def load(self,candidate):
        old=self.model.artifact();artifact=candidate.get('delta_policy')
        # Captures retain the joint buffers' storage; a 25↔29-DoF switch must
        # invalidate the graph even if a proposed artifact subsequently fails.
        self.graph=None;self.graph_error=None
        try:
            if candidate.get('wuji_native'):validate_native_candidate(candidate)
            if candidate.get('bone_extrinsics'):validate_candidate(candidate)
            self.model.load_artifact(candidate)
            if artifact and artifact.get('geometry_hash')!=self.model.version():raise ValueError('Delta与几何指纹不匹配')
            policy=load_policy(artifact,self.model) if artifact else None
            if artifact:
                mean=torch.tensor(artifact['mean'],device=self.model.device);scale=torch.tensor(artifact['scale'],device=self.model.device)
                if mean.shape!=(146,) or scale.shape!=(146,) or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale<=0).any():raise ValueError('Delta输入归一化无效')
        except Exception:
            self.model.load_artifact(old);raise
        self.policy=policy;self.artifact=artifact;self.previous=None;self.graph=None;self.graph_error=None;self.model_hash=candidate.get('model_hash') if candidate.get('bone_extrinsics') or candidate.get('wuji_native') else None
        self.wuji=candidate.get('wuji_native');self.wuji_rest=None
        if artifact:self.mean=mean;self.scale=scale
    def render_forward(self,q):
        m=self.model;v,j,r=m(q);mp=m.mount_p0+.01*m.geometry['mount_pos']
        pred=j[:,m.distal_ids]+(r[:,m.distal_ids]@mp[...,None]).squeeze(-1)
        return v,j,r,pred
    @torch.no_grad()
    def warmup(self):
        m=self.model
        if m.device.type!='cuda':return
        # A later SID rebinding clears model._rest_cache. CUDA Graph does not
        # own input tensor storage: retain these tensors for its whole lifetime.
        began=time.monotonic();self.graph_rest=m.rest();self.static_q=m.q0[None].clone();s=torch.cuda.Stream(device=m.device);s.wait_stream(torch.cuda.current_stream(m.device))
        with torch.cuda.stream(s):
            for _ in range(3):self.render_forward(self.static_q)
        torch.cuda.current_stream(m.device).wait_stream(s);s.synchronize()
        try:
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=s):self.static_outputs=self.render_forward(self.static_q)
            self.graph=graph
        except RuntimeError as exc:self.graph_error=str(exc);self.graph=None
        self.warmup_s=time.monotonic()-began
    @torch.no_grad()
    def infer(self,frame):
        if self.model_hash and (frame.get('skeleton') or {}).get('model_hash')!=self.model_hash:raise ValueError('骨骼与mesh模型版本尚未一致')
        if self.wuji:return self.infer_wuji(frame)
        m=self.model;q=m.angles([frame]).to(m.q0.dtype);rawp,rawr,valid=(value.to(m.q0.dtype) for value in observations([frame],m.device));stamp=frame['timestamp_ms'];dt=(stamp-self.previous[0])/1000 if self.previous else 0.
        if not 0<dt<=.25:self.previous=None;dt=0.
        if self.policy:
            pred,_=m.sensor_prediction(q);align,trans=m.alignment();p=rawp@align.T+trans;r=align@rawr;touch,mask=tactile_features([frame],m.device)
            features=torch.cat((q,p.flatten(1),r.flatten(1),(pred-p).flatten(1),touch,mask,valid,torch.tensor([[dt]],device=m.device)),1)
            raw=self.policy((features-self.mean)/self.scale);delta=raw
            if self.previous:
                limit=dt*np.deg2rad(25);previous=self.previous[1];delta=previous+(raw-previous).clamp(-limit,limit)
            self.previous=(stamp,raw.detach());q=apply_delta(q,delta,m.limits)
        else:self.previous=(stamp,None)
        if m.device.type=='cuda' and self.graph is None and self.graph_error is None:self.warmup()
        if self.graph is not None:self.static_q.copy_(q);self.graph.replay();vertices,joints,rot,pred=self.static_outputs
        else:vertices,joints,rot,pred=self.render_forward(q)
        align,trans=m.alignment();error=((pred-(rawp@align.T+trans)).square().sum(-1)*valid).sum()/valid.sum().clamp_min(1)
        return vertices,joints,rot,q,pred,float(error.sqrt())*1000

    @torch.no_grad()
    def infer_wuji(self,frame):
        m=self.model;s=frame['skeleton']
        if s.get('native_dof')!=21 or abs(s['timestamp_ms']-frame['timestamp_ms'])>25:raise ValueError('等待同模型同时间戳的Wuji21自由度结果')
        if self.wuji_rest is None:
            rest,br,bp=m.rest();rr,pp=m.fk(m.q0[None]);wr=rr[0,m.wrist_body];wp=pp[0,m.wrist_body]
            self.wuji_rest=((rest-wp)@wr,wr.T@br,(bp-wp)@wr)
        rest,br,bp=self.wuji_rest
        r=torch.as_tensor(s['display_body_rotations'],device=m.device,dtype=m.q0.dtype)[None]
        p=torch.as_tensor(s['display_body_positions'],device=m.device,dtype=m.q0.dtype)[None]
        if r.shape!=(1,len(m.body_names),3,3) or p.shape!=(1,len(m.body_names),3):raise ValueError('Wuji显示适配骨骼数量不匹配')
        relative=r[:,m.bone_ids]@br.transpose(-1,-2);out=torch.zeros((1,len(rest),3),device=m.device,dtype=m.q0.dtype)
        for k in range(20):out+=((rest-bp[k])@relative[:,k].transpose(-1,-2)+p[:,m.bone_body[k],None])*m.weights[:,k,None]
        predicted=torch.as_tensor(s['display_rx_positions'],device=m.device,dtype=m.q0.dtype)[None]
        raw,_,valid=observations([frame],m.device);a,t=m.alignment();error=(((predicted-(raw@a.T+t)).square().sum(-1)*valid).sum()/valid.sum().clamp_min(1)).sqrt()
        q=torch.as_tensor(s['joint_positions_rad'],device=m.device,dtype=m.q0.dtype)[None]
        return out,p,r,q,predicted,float(error)*1000


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--event-fd',type=int,required=True);ap.add_argument('--home',type=Path,required=True);ap.add_argument('--serial',required=True);ap.add_argument('--side',choices=('left','right'),default='right');ap.add_argument('--model');args=ap.parse_args()
    home=args.home;native=home/'qnbot';native.mkdir(parents=True,exist_ok=True);stopping=threading.Event();cancel=threading.Event();lock=threading.Lock()
    stream=os.fdopen(args.event_fd,'w',buffering=1)
    def emit(value):
        if 'fit' in value:value={**value,'fit':public_fit_report(value['fit'],include_training_sample=value.get('type')=='training')}
        with lock:stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n');stream.flush()
    def stop(*_):stopping.set();cancel.set()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    frames=queue.Queue(maxsize=1024);controls=queue.Queue();jobs=queue.Queue()
    def reader():
        for line in sys.stdin:
            try:
                value=json.loads(line)
                if value.get('type')=='control':controls.put(value)
                elif value.get('type')=='frame':
                    try:frames.put_nowait(value)
                    except queue.Full:jobs.put(('overflow',None))
            except Exception:traceback.print_exc()
        stop()
    threading.Thread(target=reader,daemon=True).start();torch.set_num_threads(1)
    emit(dict(type='state',status='starting',surface_backend='qnbot',message='初始化官方Qnbot骨架和连续mesh；不加载MANO'))
    if not torch.cuda.is_available():raise ValueError('Qnbot GPU前向需要CUDA')
    model=QnbotSurface(args.side,'cuda:0',finger_twist=True);model.fixed=True;runtime=StreamingModel(model);runtime.warmup()
    capture=SharedHandCapture();fit=dict(state='idle');profile=None;candidate=None;preview=None;baseline=None;last_frame=None;initial=None;reference=None;mapping=None;preparing=False;fitting=None
    saves=CaptureCheckpoints();raw_journal=SnapshotRecorder(native/'raw-captures',capacity=512);geometry_version=model.version();sequence=0;last_output=0.;last_state=0.;attempted=False;topology_key=None
    try:pending=json.loads((native/'current.json').read_text())
    except (OSError,ValueError):pending=None
    if not pending or pending.get('report',{}).get('fit_revision') not in ACTIVE_REVISIONS:publish_active(native,None)
    try:
        candidate=json.loads((native/'candidate.json').read_text())
        if candidate.get('serial')!=args.serial or candidate.get('side')!=args.side or candidate.get('asset_hash')!=model.asset_hash:candidate=None
        elif pending is None or candidate.get('report',{}).get('validation_pending'):fit=candidate.get('report',fit)
    except (OSError,ValueError):pass
    default=model.artifact();shown_artifact=default;trace_path=None

    def state():
        emit(dict(type='state',surface_backend='qnbot',capture=capture.snapshot(),fit=public_fit_report(fit),capture_storage=saves.snapshot(),capture_storage_error=saves.error,
            preview=preview is not None,checkpoint_preview=preview,calibrated=bool(profile and preview is None),penetration_tolerance_m=.002,solver_dof=len(model.joint_names),joint_limit_policy=model.joint_limit_policy,
            calibration_method=WUJI_METHOD,calibration_fit_revision=WUJI_FIT_REVISION,calibration_version='Moxian25＋四指侧旋 / 末节反折 / Wuji L-BFGS',
            gpu_warmup=dict(state='completed' if runtime.graph_error is None else 'failed',seconds=runtime.warmup_s,cuda_graph=runtime.graph is not None,message='官方mesh固定前向CUDA Graph已预热',error=runtime.graph_error),display_options=dict(compression=False,physics=False,contact_points=True)))
    def checkpoint(history=False):
        if not isinstance(capture,MotionCapture):return
        # Historical captures may be restored for read-only inspection. A
        # startup status/request-id refresh must not rewrite their files.
        if not history and not isinstance(capture,WujiNativeCapture):return
        document=dict(capture.document(),serial=args.serial,side=args.side)
        paths=[native/'motion-capture.json']
        if isinstance(capture,WujiNativeCapture):paths=[native/'wuji-calibration-capture.json']
        if isinstance(capture,ContactTrajectoryCapture):paths.append(native/'contact-motion-v4.json')
        if isinstance(capture,StaticHandCapture):paths.append(native/'contact-static-v5.json')
        if history:paths=[native/'capture-history'/str(time.time_ns())/'motion-capture.json']
        saves.submit(paths,document,history=history)
    def check(candidate):
        if candidate.get('wuji_native'):raise ValueError('Wuji内置人手模型已停用；请用Moxian骨架拟合，原文件仅供诊断')
        if candidate.get('surface_backend')!='qnbot':raise ValueError('该权重使用MANO，请切换模型基础为历史MANO后预览')
        if candidate.get('serial')!=args.serial or candidate.get('side')!=args.side:raise ValueError('权重设备／手侧不匹配')
        if not last_frame or candidate.get('calibration_fingerprint')!=last_frame.get('calibration_fingerprint'):raise ValueError('权重与当前骨架初始化不匹配')
        if not candidate.get('bone_extrinsics') and not candidate.get('wuji_native') and candidate.get('skeleton_runtime_revision')!=last_frame.get('skeleton_runtime_revision'):raise ValueError('历史权重与骨架版本不匹配')
        if candidate.get('mapping',{}).get('assignments')!=(mapping or {}).get('assignments'):raise ValueError('SID分区不匹配')
    def load(candidate):
        nonlocal geometry_version,topology_key,shown_artifact
        runtime.load(candidate)
        try:publish_active(native,candidate if candidate.get('bone_extrinsics') or candidate.get('wuji_native') else None)
        except Exception:
            runtime.load(shown_artifact);raise
        geometry_version=model.version();model._rest_cache=None;model._topology=None;topology_key=None
        shown_artifact=candidate
    def prepare(new=False):
        nonlocal preparing,default,shown_artifact,geometry_version
        if preparing:raise ValueError('官方参考正在准备')
        def native_capture_reference(capture):
            if not capture.metadata.get('gesture_frame'):
                from moxian_web.wuji_gesture_frame import from_official
                key=hashlib.sha256(args.serial.encode()).hexdigest()[:16]
                capture.install_gesture_frame(from_official(home.parents[1]/'ik'/key,last_frame['calibration_fingerprint'],args.side))
            return capture
        if new:return native_capture_reference(WujiNativeCapture(last_frame,side=args.side))
        native_capture=native/'wuji-calibration-capture.json'
        if native_capture.exists():
            doc=json.loads(native_capture.read_text())
            if (doc.get('calibration_fingerprint')==(last_frame or {}).get('calibration_fingerprint') and
                doc.get('mapping',{}).get('assignments')==(mapping or {}).get('assignments')):
                restored=native_capture_reference(WujiNativeCapture.restore(doc,allow_partial=True))
                if profile is None and preview is None:
                    from moxian_web.moxian_six_calibration import initial_rows as moxian_initial_rows
                    params=dict(serial=args.serial,side=args.side,calibration_fingerprint=last_frame['calibration_fingerprint'],skeleton_runtime_revision=last_frame.get('skeleton_runtime_revision'))
                    source=moxian_initial_rows(params,mapping,stopping,lambda v:emit(dict(type='state',message=v.get('message'))))
                    initialize_registration(model,source);default=model.artifact();shown_artifact=default;geometry_version=model.version()
                return restored
        preparing=True;snapshot=copy.deepcopy(last_frame);saved=capture;session=uuid.uuid4().hex
        binding=dict(assignments=model.mapping['assignments'],face_ids=model.sid_faces.tolist(),barycentric=model.sid_bc.tolist())
        if new:
            saved=StaticHandCapture(dict(rows=[],completed_groups=[],calibration_fingerprint=snapshot['calibration_fingerprint'],side=args.side));saved.metadata['motion_session_id']=session
        saved.status='preparing';saved.request_id=None
        def work():
            try:
                key=hashlib.sha256(args.serial.encode()).hexdigest()[:16]
                source=load_official_capture(home.parents[1]/'ik'/key,snapshot['calibration_fingerprint'],snapshot['mapping'],side=args.side)
                from moxian_web.mano_calibration import prepare_rows
                parameters=dict(artifact_home=str(home),serial=args.serial,side=args.side,calibration_fingerprint=snapshot['calibration_fingerprint'],skeleton_runtime_revision=snapshot.get('skeleton_runtime_revision'))
                rows,_=prepare_rows(parameters,source['rows'],lambda v:emit(dict(type='state',capture=dict(saved.snapshot(),state='preparing',error=v.get('message')))),stopping)
                source=dict(source,rows=rows)
                spec=saved.metadata.get('contact_reference') if isinstance(saved,(ContactTrajectoryCapture,StaticHandCapture)) else None
                ref=QnbotReference(args.side,snapshot['mapping'],source,spec=spec,material_binding=binding)
                restored=None
                if not new and not isinstance(saved,(ContactTrajectoryCapture,StaticHandCapture)):
                    path=native/'motion-capture.json'
                    if not path.exists():path=home/'motion-capture.json'
                    doc=json.loads(path.read_text()) if path.exists() else None
                    if doc and doc.get('calibration_fingerprint')==snapshot.get('calibration_fingerprint') and doc.get('rows',[{}])[0].get('mapping',{}).get('assignments')==snapshot['mapping']['assignments']:
                        restored=MotionCapture.restore(doc,allow_partial=True)
                        if isinstance(restored,(ContactTrajectoryCapture,StaticHandCapture)):
                            restored.reference=QnbotReference(args.side,snapshot['mapping'],source,spec=restored.metadata['contact_reference'])
                            if isinstance(restored,StaticHandCapture):restored.install_reference(restored.reference,source)
                jobs.put(('prepared',(source,ref,new,saved,restored)))
            except Exception as exc:traceback.print_exc();jobs.put(('prepare_error',str(exc)))
        threading.Thread(target=work,daemon=True).start()
        return saved
    def start_moxian_fit(data):
        nonlocal fitting,fit,trace_path
        if fitting and fitting.is_alive():raise ValueError('已有Moxian拟合任务')
        if preview:raise ValueError('请先结束历史预览')
        if not isinstance(capture,WujiNativeCapture) or capture.status!='complete':raise ValueError('请重新采集Wuji原生六动作；旧17动作仅供查看')
        if capture.metadata.get('calibration_fingerprint')!=(last_frame or {}).get('calibration_fingerprint'):raise ValueError('输入初始化变化，请重新采集六动作')
        from moxian_web.moxian_six_calibration import fit_capture as fit_native_capture
        cancel.clear();run_id=uuid.uuid4().hex;began=time.monotonic();trace_home=native/'training-traces';trace_home.mkdir(parents=True,exist_ok=True);trace_path=trace_home/f'{run_id}.jsonl'
        for name in ('latest',WUJI_METHOD):save(trace_home/f'{name}.json',dict(file=trace_path.name,method=WUJI_METHOD,surface_backend='qnbot'))
        fit=dict(state='running',phase='native_fit',run_id=run_id,method=WUJI_METHOD,surface_backend='qnbot',progress=None,message='准备Moxian骨长与EMF外参六动作拟合')
        doc=capture.document();parameters=dict(serial=args.serial,side=args.side,calibration_fingerprint=doc['calibration_fingerprint'],capture_session_id=doc['motion_session_id'],skeleton_runtime_revision=(last_frame or {}).get('skeleton_runtime_revision'),material_binding=model.artifact().get('material_binding'))
        map_copy=copy.deepcopy(mapping)
        def progress(value):
            nonlocal fit
            fit=dict(fit,**value,elapsed_s=time.monotonic()-began)
            if 'training_sample' in value:
                with trace_path.open('a') as stream:stream.write(json.dumps(fit,allow_nan=False)+'\n')
            emit(dict(type='training' if 'training_sample' in value else 'state',fit=fit))
        def work():
            try:
                result,report=fit_native_capture(parameters,map_copy,doc,progress,cancel);report['run_id']=run_id
                jobs.put(('fit_success',(result,report)))
            except Exception as exc:traceback.print_exc();jobs.put(('fit_error',str(exc)))
        fitting=threading.Thread(target=work,daemon=True);fitting.start()
    def start_fit(data):
        nonlocal fitting,fit,trace_path
        if data.get('method',WUJI_METHOD)==WUJI_METHOD:return start_moxian_fit(data)
        raise ValueError('当前只允许Moxian原生25自由度六动作标定；Wuji内置手型已停用')

    while not stopping.is_set():
        try:
            while True:
                kind,value=jobs.get_nowait()
                if kind=='prepared':
                    initial,reference,new,saved,restored=value;preparing=False
                    if initial.get('calibration_fingerprint')!=(last_frame or {}).get('calibration_fingerprint'):attempted=False;continue
                    if saved.status=='cancelled':continue
                    if new:capture=StaticHandCapture(initial,reference)
                    elif isinstance(capture,(ContactTrajectoryCapture,StaticHandCapture)):
                        if isinstance(capture,StaticHandCapture):capture.install_reference(reference,initial)
                        else:capture.reference=reference
                        capture.status='fit_ready' if isinstance(capture,StaticHandCapture) and capture.index>=18 else 'paused';capture.request_id=uuid.uuid4().hex
                    else:
                        # Prefer a native capture; leave the original MANO raw
                        # records untouched and available for explicit replay.
                        if restored is not None:capture=restored
                        else:capture=SharedHandCapture();capture.install(initial)
                    if profile is None and preview is None:
                        with torch.no_grad():
                            for k in ('align0','trans0','mount_p0','mount_r0'):getattr(model,k).copy_(getattr(reference.model,k))
                        geometry_version=model.version();model._topology=None;topology_key=None
                    checkpoint();state()
                elif kind=='prepare_error':preparing=False;capture.status='paused';capture.error=value;state()
                elif kind=='fit_success':
                    if cancel.is_set():
                        fit=dict(fit,state='cancelled',message='原生标定已取消，当前模型保留');state();continue
                    candidate,fit=value;candidate['report']=fit;method=fit['method'];folder=native/'methods'/method
                    path=folder/'candidate.json'
                    if path.exists():save(native/'candidate-history'/f'{time.time_ns()}.json',json.loads(path.read_text()))
                    save(path,candidate);save(folder/'report.json',fit)
                    if method!='least_squares':save(native/'candidate.json',candidate)
                    if fit.get('accepted') and method!='least_squares':
                        try:
                            check(candidate)
                            if (native/'current.json').exists():save(native/'previous.json',json.loads((native/'current.json').read_text()))
                            load(candidate);profile=candidate;fit['applied']=True;save(native/'current.json',profile);save(folder/'report.json',fit)
                        except Exception as exc:
                            fit.update(applied=False,application_error=str(exc));save(folder/'report.json',fit)
                    state()
                elif kind=='fit_error':fit=dict(fit,state='cancelled' if cancel.is_set() else 'failed',error=value);state()
                elif kind=='overflow':
                    if isinstance(capture,MotionCapture) and capture.status=='collecting':capture.pause_cycle('采集队列溢出，已停止当前片段；完整循环保留');checkpoint();state()
        except queue.Empty:pass
        try:
            control=controls.get_nowait();action=control['action'];data=control.get('data') or {}
            try:
                if action=='capture_start':
                    if fitting and fitting.is_alive() or preview:raise ValueError('请先结束拟合或权重预览')
                    if last_frame is None or mapping is None:raise ValueError('等待官方骨架与SID分区')
                    if isinstance(capture,MotionCapture):checkpoint(history=True)
                    raw_journal.stop();capture=prepare(new=True)
                elif action in ('capture_confirm','capture_resume','capture_retry','capture_cancel'):
                    if not isinstance(capture,WujiNativeCapture):raise ValueError('历史记录只读；新采集请点击开始Wuji六动作')
                    if action in ('capture_confirm','capture_resume'):
                        if not raw_journal.active:
                            journal=raw_journal.start();capture.metadata.setdefault('raw_journals',[]).append(dict(file=str(raw_journal.path),source='immutable_static_raw_rows'))
                        capture.confirm(data.get('request_id'))
                    elif action=='capture_retry':capture.check_stage(data.get('stage_id'));capture.retry()
                    else:capture.cancel();raw_journal.stop()
                    checkpoint()
                elif action=='fit':start_fit(data)
                elif action=='fit_cancel':cancel.set();fit=dict(fit,state='cancelling' if fitting and fitting.is_alive() else 'cancelled')
                elif action=='checkpoint_preview':
                    if fitting and fitting.is_alive() or capture.status=='collecting':raise ValueError('请先结束采集或训练')
                    proposed=data['checkpoint'];check(proposed)
                    if baseline is None:
                        baseline=copy.deepcopy(profile) if profile else model.artifact();baseline['delta_policy']=runtime.artifact
                    load(proposed);preview=data['info']
                elif action=='checkpoint_end':
                    if baseline:load(baseline)
                    baseline=None;preview=None
                elif action=='apply':
                    if not candidate or not candidate['report'].get('accepted') or not candidate['report'].get('validation_complete'):raise ValueError('候选尚未完成原生标定与结果检查')
                    if candidate['report'].get('fit_revision')!=WUJI_FIT_REVISION:raise ValueError('目标函数已更新，请重新执行Wuji原生标定')
                    check(candidate)
                    if (native/'current.json').exists():save(native/'previous.json',json.loads((native/'current.json').read_text()))
                    load(candidate);profile=candidate;save(native/'current.json',profile);preview=None
                elif action=='restore':
                    old=json.loads((native/'previous.json').read_text())
                    if old.get('bone_extrinsics') and old.get('report',{}).get('fit_revision') in ACTIVE_REVISIONS:
                        check(old);save(native/'previous.json',profile or default);load(old);profile=old
                    else:
                        fallback=reference.model.artifact() if reference else default
                        save(native/'previous.json',profile or default);load(fallback);profile=None
                    save(native/'current.json',profile or default);preview=None;baseline=None
                    fit=dict(fit,applied=bool(profile and candidate and profile.get('model_hash')==candidate.get('model_hash')))
                elif action=='fit_options':
                    if float(data.get('penetration_tolerance_mm',2))!=2:raise ValueError('穿透验收固定为2mm')
                elif action=='compression_display':
                    if data.get('enabled'):raise ValueError('官方mesh仅离线学习，禁止实时压缩')
                elif action=='registration_get':
                    emit(dict(type='ack',id=control['id'],result=dict(status='anatomical_initialization_not_measured',artifact=None)));continue
                elif action=='bind_sid':
                    if fitting and fitting.is_alive() or capture.status=='collecting' or preview:raise ValueError('请先结束采集、训练或预览')
                    face=data.get('face');bc=data.get('barycentric');sids=data.get('sids',[])
                    if type(face) is not int or not 0<=face<len(model.faces) or len(bc or [])!=3 or not np.isfinite(bc).all() or min(bc)<0 or abs(sum(bc)-1)>1e-5:raise ValueError('无效材料坐标')
                    from moxian_web.contact import PATCH_IDS
                    for sid in sids:
                        if type(sid) is not int or not 0<=sid<460 or face not in model.regions[PATCH_IDS[mapping['assignments'][sid]]]:raise ValueError('绑定必须位于SID对应区域')
                    for sid in sids:model.sid_faces[sid]=face;model.sid_bc[sid]=bc
                    model.install_binding(dict(assignments=mapping['assignments'],face_ids=model.sid_faces.tolist(),barycentric=model.sid_bc.tolist(),normal_offset_m=model.sid_normal_offset.tolist(),source='manual_material_binding'));topology_key=None
                    save(native/'manual-bindings.json',dict(asset_hash=model.asset_hash,material_binding=model.artifact()['material_binding']))
                else:raise ValueError('官方mesh暂不支持该旧MANO操作：'+action)
                emit(dict(type='ack',id=control['id'],result=dict(ok=True)));state()
            except Exception as exc:emit(dict(type='ack',id=control['id'],error=str(exc)))
        except queue.Empty:pass
        batch=[]
        try:
            for _ in range(16):batch.append(frames.get_nowait())
        except queue.Empty:pass
        if not batch:stopping.wait(.005);continue
        for frame in batch:
            if isinstance(capture,WujiNativeCapture) and capture.status=='collecting':
                previous_rows=len(capture.rows);changed=capture.feed(frame)
                if len(capture.rows)>previous_rows and raw_journal.active:
                    row=dict(capture.rows[-1],capture_session_id=capture.metadata['motion_session_id'],capture_row_index=len(capture.rows)-1)
                    if not raw_journal.submit(row):capture.pause_cycle('原始轨迹写入队列已满，已暂停并保留完整循环');changed=True
                if changed:
                    if capture.status=='complete':raw_journal.stop()
                    checkpoint();state()
        frame=batch[-1]
        if frame['timestamp_ms']<=((last_frame or {}).get('timestamp_ms',-1)):continue
        last_frame=frame
        if not frame.get('skeleton') or abs(frame['skeleton'].get('timestamp_ms',-1)-frame['timestamp_ms'])>25:continue
        try:
            incoming=frame.get('mapping')
            shown=shown_artifact
            if shown.get('calibration_fingerprint') and (shown.get('calibration_fingerprint')!=frame.get('calibration_fingerprint') or not shown.get('bone_extrinsics') and not shown.get('wuji_native') and shown.get('skeleton_runtime_revision')!=frame.get('skeleton_runtime_revision')):
                cancel.set();load(default);profile=None;preview=None;baseline=None;attempted=False;reference=None
                if isinstance(capture,MotionCapture) and capture.status=='collecting':capture.cancel();checkpoint()
                emit(dict(type='state',message='骨架初始化已更新，恢复默认官方mesh并重建冻结参考'))
            if incoming and (mapping is None or incoming.get('version')!=mapping.get('version')):
                if fitting and fitting.is_alive():cancel.set()
                if mapping is not None and incoming.get('assignments')!=mapping.get('assignments'):
                    load(default);profile=None;preview=None;baseline=None;reference=None;attempted=False
                    if isinstance(capture,MotionCapture) and capture.status=='collecting':capture.cancel();raw_journal.stop();checkpoint()
                mapping=copy.deepcopy(incoming);model.bind_mapping(mapping);topology_key=None
                try:
                    manual=json.loads((native/'manual-bindings.json').read_text())
                    if manual.get('asset_hash')==model.asset_hash:model.install_binding(manual['material_binding'])
                except (FileNotFoundError,ValueError,KeyError):pass
            if mapping is None:continue
            if pending is not None:
                saved=pending;pending=None
                try:
                    check(saved)
                    if saved.get('report',{}).get('fit_revision') not in ACTIVE_REVISIONS:raise ValueError('默认管线已切换为Wuji原生模型，旧结果只读保留')
                    load(saved);profile=saved;fit=dict(saved.get('report',fit),applied=True)
                except Exception as exc:
                    publish_active(native,None)
                    emit(dict(type='state',message='候选未加载，保留官方初始化：'+str(exc)))
            if not attempted and frame.get('calibration_fingerprint'):
                attempted=True;capture=prepare()
            if time.monotonic()-last_output<1/90:continue
            if frame['skeleton'].get('model_hash')!=runtime.model_hash:
                emit(dict(type='state',ready=False,status='waiting_model',message='等待骨骼与mesh使用同一模型'));continue
            began=time.monotonic();v,j,r,q,pred,error=runtime.infer(frame);sequence+=1
            key=(geometry_version,model.binding_hash)
            if topology_key!=key:
                topo=dict(model.topology(),geometry_version=geometry_version,model_hash=runtime.model_hash);emit(dict(type='topology',model=topo));topology_key=key
            vertices=v[0].cpu().numpy().astype('<f4');joints=j[0].cpu().numpy().astype('<f4');payload=vertices.tobytes()+joints.tobytes();last_output=time.monotonic()
            emit(dict(type='frame',status='preview' if preview else 'calibrated' if profile else 'default',ready=True,error=None,surface_backend='qnbot',runtime_backend='wuji_native21_qnbot_display_v1' if runtime.wuji else REVISION,
                hand_side=args.side,coordinate_frame='wrist_local',timestamp_ms=frame['timestamp_ms'],skeleton_timestamp_ms=frame['skeleton']['timestamp_ms'],geometry_version=geometry_version,binding_version=model.binding_hash,
                source_contact=frame.get('contact'),tactile_timestamp_ms=frame.get('tactile_timestamp_ms'),
                mesh_frame=dict(sequence=sequence,vertex_count=len(vertices),joint_count=len(joints)),mesh_f32_b64=base64.b64encode(payload).decode(),
                tracking_source='wuji_sdk_native21' if runtime.wuji else frame['skeleton'].get('ik_solver',QP_REVISION) if runtime.model_hash else 'qnbot_native25_fixed_forward',model_hash=runtime.model_hash,model_implementation='Wuji native21 FK → Qnbot display LBS' if runtime.wuji else 'Qnbot SKN + bone-length-driven skin',calibrated=bool(profile and preview is None),preview=preview is not None,
                sensor_positions_m=pred[0].cpu().tolist(),tip_positions_m=j[0,model.tip_bodies].cpu().tolist(),root_rotation=np.eye(3).tolist(),
                input_targets=dict(timestamp_ms=frame['timestamp_ms'],positions_local=(observations([frame],model.device)[0][0]@model.alignment()[0].T+model.alignment()[1]).cpu().tolist()),
                sensor_mounting=dict(status='multi_pose_calibrated' if profile else 'multi_pose_candidate' if reference else 'position_prior',flat_reference_id=reference.fingerprint if reference else None),
                mano_solver=dict(enabled=runtime.model_hash is not None,delta_policy=dict(enabled=runtime.policy is not None),online_optimization=runtime.model_hash is not None),fusion=dict(ready=False,status='unknown',pairs=[],patches=[]),
                quality=dict(processing_ms=(time.monotonic()-began)*1000,forward_ms=(time.monotonic()-began)*1000,position_rmse_mm=error,
                    source_ik_solver=frame['skeleton'].get('ik_solver'),source_ik_processing_ms=(frame.get('source_solver_diagnostics') or {}).get('processing_ms'),
                    emf_rotation_error_deg=float(np.mean([v['rotation_error_deg'] for v in frame['skeleton'].get('tracking_errors',{}).values()])) if frame['skeleton'].get('tracking_errors') else None,
                    source_age_ms=time.time()*1000-frame['timestamp_ms']),
                contact_registration=dict(status='anatomical_initialization_not_measured'),penetration_tolerance_m=.002))
            if last_output-last_state>.2:last_state=last_output;state()
        except Exception as exc:
            traceback.print_exc();emit(dict(type='state',status='input_invalid',error=str(exc)))
    cancel.set()
    if fitting:fitting.join(timeout=5)
    saves.close();raw_journal.close();stream.close()


if __name__=='__main__':main()
