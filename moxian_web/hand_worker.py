"""Isolated CUDA MANO process. Never opens a serial port or imports the glove SDK."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,queue,signal,sys,threading,time,traceback,uuid,subprocess
from pathlib import Path
from collections import deque
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moxian_web.hand_model import aligned_frame,material_config,DEFAULT_MATERIAL,confirmed_contact_points,persist_calibrated_profile,tracking_registration_matches
from moxian_web.official_mano import OFFICIAL_MANO_MODE, OFFICIAL_REFERENCE_POLICY, reference_from_frame
from moxian_web.shared_capture import SharedHandCapture,load_official_capture,PROTOCOL
from moxian_web.motion_capture import MotionCapture,PROTOCOL as MOTION_PROTOCOL
from moxian_web.contact_capture import ContactMotionCapture
from moxian_web.contact_templates import FrozenContactReference
from moxian_web.contact_protocol import MOTION_PROTOCOLS
from moxian_web.mano_runtime import ManoSurface,CollisionSurface,fuse_contacts,deform_surface
from moxian_web.mano_fit import FIT_REVISION
from moxian_web.mano_calibration import fit_capture,METHODS
from moxian_web.checkpoints import comparison_profile as make_comparison_profile
from moxian_web.hand_anatomy import GEOMETRY_REVISION
from moxian_web.hand_dimensions import DIMENSION_MODE,DIMENSION_COUNT,PENETRATION_TOLERANCE_M
from moxian_web.mano_collision import HARD_PENETRATION_M
from moxian_web.mano_objective import SOLVER_REVISION
from moxian_web.hand_model import calibration_report_valid,public_fit_report,tracking_geometry_hash
import numpy as np
import torch

DIMENSION_MODE=OFFICIAL_MANO_MODE


def write_json(path,value,compact=False):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=None if compact else 2,separators=(',',':') if compact else None));tmp.replace(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--event-fd',type=int,required=True);ap.add_argument('--home',type=Path,required=True);ap.add_argument('--model',type=Path,required=True);ap.add_argument('--serial',required=True);ap.add_argument('--side',choices=('left','right'),default='left');args=ap.parse_args()
    home=args.home;home.mkdir(parents=True,exist_ok=True)
    stream=os.fdopen(args.event_fd,'w',buffering=1);event_lock=threading.Lock();stopping=threading.Event();cancel_fit=threading.Event();capture_preparation_cancel=threading.Event()
    def emit(value):
        if 'fit' in value:value={**value,'fit':public_fit_report(value['fit'],include_training_sample=value.get('type')=='training')}
        with event_lock:stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n');stream.flush()
    def finish(_s=None,_f=None):stopping.set();cancel_fit.set();capture_preparation_cancel.set()
    signal.signal(signal.SIGTERM,finish);signal.signal(signal.SIGINT,finish)
    frames=queue.Queue(maxsize=1);controls=queue.Queue(maxsize=32)
    capture_frames=queue.Queue(maxsize=512);capture_overflow=threading.Event()
    def clear_capture_frames():
        while True:
            try:capture_frames.get_nowait()
            except queue.Empty:break
        capture_overflow.clear()
    def reader():
        for line in sys.stdin:
            try:
                m=json.loads(line)
                if m.get('type')=='control':controls.put(m)
                elif m.get('type')=='frame':
                    try:capture_frames.put_nowait(m)
                    except queue.Full:
                        capture_overflow.set()
                        try:capture_frames.get_nowait()
                        except queue.Empty:pass
                        capture_frames.put_nowait(m)
                    try:frames.put_nowait(m)
                    except queue.Full:
                        try:frames.get_nowait()
                        except queue.Empty:pass
                        frames.put_nowait(m)
            except Exception:traceback.print_exc()
        finish()
    threading.Thread(target=reader,daemon=True).start()
    torch.set_num_threads(1)
    emit({'type':'state','status':'starting','error':None,'message':'初始化 MANO 与 GPU 碰撞结构'})
    if not torch.cuda.is_available():raise RuntimeError('GPU 不可用；MANO 已停止，原骨架与触觉服务不受影响')
    model=ManoSurface(args.model,side=args.side);model.enable_official_fk();model.set_dimensions(np.zeros(DIMENSION_COUNT));capture=SharedHandCapture();fit_status={'state':'idle'};candidate=None;fit_results=queue.Queue();fitting=None;preview=False;preview_base=None;comparison=None;comparison_info=None;compression_display=False
    model.registration_serial=args.serial
    model.solver_enabled=False;model.contact_pose_enabled=False;model.fixed_forward=True
    try:model.contact_registration=json.loads((home/'contact-registration.json').read_text())
    except (OSError,ValueError):pass
    profile={'schema_version':1,'serial':args.serial,'side':args.side,'operator':'local','weights_hash':model.weights_hash,
             'dimension_mode':DIMENSION_MODE,'dimension_parameters':model.dimension_parameters.tolist(),'proximal_lateral_scales':model.proximal_lateral_scales.tolist(),'penetration_tolerance_m':HARD_PENETRATION_M,'surface':'gloved_effective','geometry_revision':GEOMETRY_REVISION,'calibrated':False,'shape':model.shape.tolist(),'material':dict(DEFAULT_MATERIAL),
             'runtime_backend':'fixed_mano_v1','raw_rotation':model.raw_rotation.tolist(),'raw_translation':model.raw_translation.tolist(),
             'bone_reference':None,'mount_pos':model.mount_pos.tolist(),'mount_rot':model.mount_rot.tolist(),'response_gains':{},'binding_overrides':{}}
    def validate_profile(p):
        from moxian_web.mano_local_geometry import validate_rest_residual
        validate_rest_residual(p.get('rest_vertex_residual_m'))
        if p.get('schema_version')!=1 or p.get('side')!=args.side or p.get('serial')!=args.serial or p.get('weights_hash')!=model.weights_hash:raise ValueError('手型记录与设备、手侧或 MANO 权重不兼容')
        if p.get('dimension_mode')!=OFFICIAL_MANO_MODE:raise ValueError('旧定制手型已停用；原文件保留，请重新拟合官方 MANO betas')
        if p.get('calibrated') and p.get('report',{}).get('fit_revision')!=FIT_REVISION:raise ValueError('标定流程已更新，请使用保留的八动作重新拟合官方 betas')
        if p.get('calibrated') and not p.get('report',{}).get('contact_validation_passed'):raise ValueError('旧手型缺少四组有效对指检查，请使用保留记录重新拟合')
        if p.get('calibrated') and p.get('report',{}).get('validation_pose_source') !='runtime_mano_sensor_only':raise ValueError('旧手型使用了触觉贴合验证，请用保留记录重新拟合并做独立验证')
        if p.get('calibrated') and p.get('geometry_revision')!=GEOMETRY_REVISION:raise ValueError('几何算法已更新，请用保留的八动作记录重新拟合；原手型文件保留')
        if p.get('calibrated') and not calibration_report_valid(p.get('report',{})):raise ValueError('旧标定缺少完整留出与2 mm碰撞验收')
        if p.get('calibrated') and model.last_frame and p.get('calibration_fingerprint')!=model.last_frame.get('calibration_fingerprint'):raise ValueError('官方骨架标定已改变，旧手型暂不兼容；请使用保留记录重新拟合')
        sensor=p.get('sensor_joint_position')
        zeros=np.asarray(p.get('calibration_pose_offsets',np.zeros((15,3))))
        if zeros.shape!=(15,3) or not np.isfinite(zeros).all() or np.max(np.linalg.norm(zeros,axis=-1))>1.20001:raise ValueError('无效的官方 MANO 姿态零位参数')
        if sensor is not None and (np.asarray(sensor).shape!=(5,3) or not np.isfinite(sensor).all() or np.max(np.linalg.norm(sensor,axis=1))>.1):raise ValueError('无效的末节到 EMF 安装位置')
        if p.get('observation_space')=='raw_emf_sensor' and p.get('mounting_calibrated') and sensor is None:raise ValueError('已标定 EMF 记录缺少独立安装变换')
        dimensions=np.asarray(p.get('dimension_parameters',np.zeros(DIMENSION_COUNT)),dtype=float)
        if dimensions.shape!=(DIMENSION_COUNT,) or not np.isfinite(dimensions).all() or np.any(dimensions):raise ValueError('固定手指尺寸参数无效')
        material_config(p['material'])
        if len(p.get('shape',[]))!=41 or not np.isfinite(p['shape']).all():raise ValueError('手型尺寸记录无效')
        for name,shape in [('raw_rotation',(3,3)),('raw_translation',(3,)),('mount_pos',(5,3)),('mount_rot',(5,3,3))]:
            value=np.asarray(p[name])
            if value.shape!=shape or not np.isfinite(value).all():raise ValueError('无效的传感器安装记录')
        if p.get('bone_reference') is not None:
            lengths=np.asarray(p['bone_reference'])
            if lengths.shape!=(5,3) or not np.isfinite(lengths).all() or np.any(lengths<.003) or np.any(lengths>.12):raise ValueError('无效的骨长参考')
        if p.get('glove_parameters') is not None:
            from moxian_web.glove_geometry import validate,validate_payload
            validate(p['glove_parameters'])
            if p.get('calibrated'):validate_payload(p['skeleton_model'],args.side,p['glove_parameters'])
    def load(p):
        validate_profile(p);model.material=material_config(p['material']);model.binding_overrides=copy.deepcopy(p.get('binding_overrides',{}))
        model.set_rest_vertex_residual(p.get('rest_vertex_residual_m'))
        model.bone_reference=np.array(p['bone_reference']) if p.get('bone_reference') is not None else None
        model.dimension_parameters=np.zeros(DIMENSION_COUNT)
        model.proximal_lateral_scales=np.array(p.get('proximal_lateral_scales',[0.]*4),dtype=float)
        model.rigid_finger_fk=bool(p.get('rigid_finger_fk',False));model.finger_base_offsets_m=np.array(p.get('finger_base_offsets_m',[0.]*4),dtype=float);model.finger_axis_yaw_rad=np.array(p.get('finger_axis_yaw_rad',[0.]*4),dtype=float)
        if p.get('glove_parameters') is not None:model.set_glove_geometry(p['glove_parameters'])
        else:
            model.model.glove_parameters=None
            if hasattr(model,'glove_parameters'):del model.glove_parameters
        model.set_shape(p['shape']);model.raw_rotation=np.array(p['raw_rotation']);model.raw_translation=np.array(p['raw_translation']);model.raw_aligned=bool(p.get('calibrated') or p.get('report') or p.get('flat_reference'))
        model.calibration_pose_offsets=np.zeros((15,3),np.float32)
        model.contact_pose_delta=None
        model.gpu_pose_enabled=False;model.solver_enabled=False;model.contact_pose_enabled=False;model.fixed_forward=True
        model._gpu_previous=None;model.delta_policy=None;model.delta_artifact=None
        if p.get('delta_policy') is not None:
            from moxian_web.mano_calibrate_delta import load_policy
            artifact=p['delta_policy']
            if artifact.get('side')!=args.side or artifact.get('serial')!=args.serial or artifact.get('weights_hash')!=model.weights_hash or artifact.get('geometry_hash')!=p.get('skeleton_model',{}).get('geometry_hash') or artifact.get('calibration_fingerprint')!=p.get('calibration_fingerprint'):raise ValueError('Delta policy 与几何／设备版本不匹配')
            model.delta_policy=load_policy(artifact,model.device);model.delta_artifact=artifact
        model.mount_pos=np.array(p['mount_pos']);model.mount_rot=np.array(p['mount_rot']);model.response_gains=p.get('response_gains',{});model.compression[:]=0
        model.mounting_calibrated=bool(p.get('mounting_calibrated',p.get('calibrated') or p.get('report',{}).get('accepted')));model.mounting_estimated=p.get('report',{}).get('validation_pose_source')=='runtime_emf_only'
        model.has_wrist_targets=p.get('observation_space') in ('sdk_wrist_targets','raw_emf_sensor')
        model.flat_reference=copy.deepcopy(p.get('flat_reference'));model.sensor_joint_position=np.array(p['sensor_joint_position']) if p.get('sensor_joint_position') is not None else None;model.reference_translation=np.zeros(3);model.reference_root=np.eye(3,dtype=np.float32) if model.flat_reference else None
        if model.flat_reference:model.flat_seed=np.array(model.flat_reference.get('flat_seed',np.zeros((15,3))),dtype=np.float32)
    try:
        if (home/'current.json').exists():
            saved=json.loads((home/'current.json').read_text());validate_profile(saved);profile=saved;load(profile)
    except Exception as exc:emit({'type':'state','error':'旧手型未加载：'+str(exc)})
    for filename,field in [('material.json','material'),('bindings.json','overrides')]:
        try:
            saved=json.loads((home/filename).read_text())
            if saved.get('weights_hash')==model.weights_hash and not profile.get('calibrated'):
                if field=='material':model.material=material_config(saved[field]);profile['material']=model.material
                else:model.binding_overrides=saved[field];profile['binding_overrides']=saved[field]
        except (OSError,ValueError,KeyError):pass
    from moxian_web.background_io import CaptureCheckpoints
    capture_saves=CaptureCheckpoints()
    initial_capture_cache=None
    def capture_document():
        if isinstance(capture,MotionCapture):doc=capture.document()
        else:doc={**copy.deepcopy(capture.metadata),'schema_version':1,'rows':list(capture.rows),'completed_groups':list(capture.complete_groups),'state':copy.deepcopy(capture.snapshot())}
        frozen=copy.deepcopy({k:v for k,v in doc.items() if k!='rows'});frozen['rows']=list(capture.rows)
        return dict(frozen,weights_hash=model.weights_hash,serial=args.serial,side=args.side)
    def checkpoint():
        if isinstance(capture,ContactMotionCapture) and not capture.rows:return
        doc=capture_document()
        paths=[home/'shared-capture.json']
        if isinstance(capture,MotionCapture):paths.insert(0,home/'motion-capture.json')
        if isinstance(capture,ContactMotionCapture):paths.append(home/'contact-motion-v3.json')
        capture_saves.submit(paths,doc)
    try:
        settings=json.loads((home/'fit-settings.json').read_text());tolerance=float(settings['penetration_tolerance_m'])
        profile['penetration_tolerance_m']=HARD_PENETRATION_M
    except (OSError,ValueError,KeyError):pass
    try:
        saved_candidate=json.loads((home/'candidate.json').read_text())
        if saved_candidate.get('geometry_revision')==GEOMETRY_REVISION and saved_candidate.get('dimension_mode')==DIMENSION_MODE:
            validate_profile(saved_candidate);candidate=saved_candidate;fit_status=copy.deepcopy(candidate['report'])
            if any(profile.get(k,default)!=candidate.get(k,default) for k,default in (('rigid_finger_fk',False),('finger_base_offsets_m',[0.]*4),('finger_axis_yaw_rad',[0.]*4),('proximal_lateral_scales',[0.]*4))):
                candidate=None;fit_status={'state':'superseded','accepted':False,'message':'已应用固定指根标定；旧候选保留。新拟合会以当前手型继续优化。'}
            if candidate and profile.get('calibrated') and profile.get('shape')==candidate.get('shape') and profile.get('report',{}).get('source_capture_hash')==candidate['report'].get('source_capture_hash'):fit_status.update(applied=True,automatic_application=bool(profile.get('automatic_application')),message='已标定：当前尺寸已生效')
    except (OSError,ValueError,KeyError):pass
    pending_reference=None;reference_samples=deque(maxlen=120)
    try:
        if (home/'sensor-reference.json').exists():pending_reference=json.loads((home/'sensor-reference.json').read_text())
    except (OSError,ValueError):pass
    def current_parameters():
        p=copy.deepcopy(profile);p.update(rigid_finger_fk=model.rigid_finger_fk,finger_base_offsets_m=model.finger_base_offsets_m.tolist(),finger_axis_yaw_rad=model.finger_axis_yaw_rad.tolist(),proximal_lateral_scales=model.proximal_lateral_scales.tolist(),tracking_source='official_qnbot_skeleton_fk_v1',dimension_mode=DIMENSION_MODE,dimension_parameters=model.dimension_parameters.tolist(),penetration_tolerance_m=HARD_PENETRATION_M,sensor_joint_position=model.sensor_offsets().tolist() if model.flat_reference is not None else None,mounting_calibrated=model.mounting_calibrated,mounting_estimated=model.mounting_estimated,flat_reference=model.flat_reference,observation_space='raw_emf_sensor' if model.flat_reference is not None else 'sdk_wrist_targets' if model.has_wrist_targets else 'legacy_rigid_fit',calibration_fingerprint=(model.last_frame or {}).get('calibration_fingerprint'),skeleton_runtime_revision=(model.last_frame or {}).get('skeleton_runtime_revision'),shape=model.shape.tolist(),bone_reference=model.bone_reference.tolist() if model.bone_reference is not None else None,material=model.material,raw_rotation=model.raw_rotation.tolist(),raw_translation=model.raw_translation.tolist(),mount_pos=model.mount_pos.tolist(),mount_rot=model.mount_rot.tolist(),binding_overrides=model.binding_overrides)
        p['contact_registration']=copy.deepcopy(model.contact_registration)
        p['calibration_pose_offsets']=model.calibration_pose_offsets.tolist()
        p['glove_parameters']=model.glove_parameters.tolist() if hasattr(model,'glove_parameters') else None
        p['rest_vertex_residual_m']=model.model.rest_vertex_residual.detach().cpu().tolist()
        p['capture_session_id']=capture.metadata.get('motion_session_id')
        return p
    capture_preparations=queue.Queue()
    def prepare_capture_reference(target,initial=None,new_session=False):
        nonlocal capture_preparation_cancel
        capture_preparation_cancel.set();capture_preparation_cancel=threading.Event()
        cancel=capture_preparation_cancel;session=target.metadata['motion_session_id']
        parameters=dict(artifact_home=str(home),serial=args.serial,side=args.side,calibration_fingerprint=(model.last_frame or {}).get('calibration_fingerprint'),skeleton_runtime_revision=(model.last_frame or {}).get('skeleton_runtime_revision'))
        mapping=copy.deepcopy(model.mapping)
        target.status='preparing';target.request_id=None
        target.error='正在回放 SDK 初始化并冻结标准 MANO 参考；完成后可开始动作'
        def work():
            try:
                source=initial
                if source is None:
                    key=hashlib.sha256(args.serial.encode()).hexdigest()[:16]
                    source=load_official_capture(home.parents[1]/'ik'/key,parameters['calibration_fingerprint'],mapping,side=args.side)
                from moxian_web.mano_calibration import prepare_rows
                def progress(value):
                    if not cancel.is_set():capture_preparations.put((session,'progress',value))
                if target.metadata.get('contact_reference'):
                    prepared=source['rows']
                else:prepared,_=prepare_rows(parameters,source['rows'],progress,cancel)
                if cancel.is_set() or stopping.is_set():return
                reference=FrozenContactReference(args.model,args.side,mapping,dict(source,rows=prepared),target.metadata.get('contact_reference'))
                if new_session:
                    ready=ContactMotionCapture(source,reference);ready.metadata['motion_session_id']=session
                    result=(ready,source)
                else:result=reference
                if not cancel.is_set() and not stopping.is_set():capture_preparations.put((session,'ready',result))
            except Exception as exc:
                if not cancel.is_set() and not stopping.is_set():capture_preparations.put((session,'error',str(exc)))
        threading.Thread(target=work,daemon=True).start()
    warmup_thread=None;warmup_state={'state':'pending'};warmup_done=threading.Event();warmup_cancel=threading.Event()
    def start_warmup(frame):
        nonlocal warmup_thread,warmup_state
        # Restoration reads the file once. Never reparse a growing capture on
        # every render frame, especially while an incomplete capture is paused.
        if not isinstance(capture,MotionCapture) or capture.status!='complete':return
        if warmup_thread is not None:
            if warmup_done.is_set() and warmup_state.get('state')=='paused':warmup_thread=None
            else:return
        saved=list(capture.rows)
        if not saved or any(r.get('calibration_fingerprint')!=frame.get('calibration_fingerprint') for r in saved):return
        warmup_cancel.clear();warmup_done.clear()
        parameters=current_parameters();parameters.update(artifact_home=str(home),calibration_fingerprint=frame.get('calibration_fingerprint'),skeleton_runtime_revision=frame.get('skeleton_runtime_revision'));mapping=copy.deepcopy(model.mapping)
        def work():
            nonlocal warmup_state
            began=time.monotonic()
            def update(v):
                nonlocal warmup_state
                warmup_state={**v,'state':'running','elapsed_s':time.monotonic()-began}
                emit({'type':'state','gpu_warmup':warmup_state})
            try:
                from moxian_web.mano_gpu_warmup import warmup
                from moxian_web.mano_calibration import prepare_rows
                prepared,_=prepare_rows(parameters,saved,update,warmup_cancel)
                warmup_state=warmup(args.model,parameters,mapping,prepared,update,warmup_cancel)
            except Exception as exc:
                if warmup_cancel.is_set():warmup_state={'state':'paused','message':'采集优先，GPU 预热已暂停'}
                else:traceback.print_exc();warmup_state={'state':'failed','error':str(exc)}
            finally:
                warmup_done.set();emit({'type':'state','gpu_warmup':warmup_state})
        warmup_thread=threading.Thread(target=work,daemon=True);warmup_thread.start()
    def start_fit(method='parameters',training_options=None,resume=False,resume_run_id=None):
        if method not in METHODS:raise ValueError('未知标定方法')
        if method=='least_squares':raise ValueError('最小二乘分支仅支持Qnbot模型基础')
        from moxian_web.training_options import options,validate_optimizer
        training_options=options(training_options)
        validate_optimizer(method,training_options)
        nonlocal fitting,fit_status
        if fitting and fitting.is_alive():raise ValueError('尺寸拟合正在运行')
        if preview:raise ValueError('请先返回当前模型，再开始拟合')
        if capture.status!='complete':
            state=capture.snapshot();pending=state.get('missing_actions') or []
            next_step=('独立复测：' if pending[0]['capture_pass']=='validation' else '训练采集：')+pending[0]['label'] if pending else state.get('prompt','当前动作')
            raise ValueError('尚未开始数据处理；请完成剩余采集。下一项是'+next_step)
        if not model.mapping:raise ValueError('等待 SID 分区映射')
        initial=None
        if type(resume) is not bool:raise ValueError('resume 必须是布尔值')
        if resume:
            if method!='delta' or not candidate or candidate.get('report',{}).get('method')!='delta' or not candidate.get('delta_policy'):
                raise ValueError('请先完成一版 MANO＋Delta 联合训练')
            if resume_run_id!=candidate['report'].get('run_id'):raise ValueError('续训来源已变化，请刷新后重试')
            if candidate.get('capture_session_id')!=capture.metadata.get('motion_session_id') or candidate.get('mapping',{}).get('assignments')!=model.mapping['assignments']:
                raise ValueError('续训 checkpoint 与当前采集或 SID 分区不匹配')
            initial=copy.deepcopy(candidate)
        data=list(capture.rows);mapping=copy.deepcopy(model.mapping);parameters=current_parameters();parameters['artifact_home']=str(home);cancel_fit.clear()
        run_id=uuid.uuid4().hex
        trace_home=home/'training-traces';trace_home.mkdir(exist_ok=True)
        trace_path=trace_home/f'{run_id}.jsonl'
        if initial is not None:write_json(trace_home/f'{run_id}.parent.json',initial,compact=True)
        write_json(trace_home/'latest.json',{'file':trace_path.name,'run_id':run_id})
        write_json(trace_home/f'{method}.json',{'file':trace_path.name,'run_id':run_id})
        fit_status={'state':'running','phase':'copy','progress':0,'completed':0,'total':len(data),'message':'准备训练数据快照','run_id':run_id,'training_options':training_options,'method':method}
        if initial is not None:fit_status['resumed_from_run_id']=initial['report']['run_id']
        write_json(home/'shared-fit-attempt.json',{'calibration_fingerprint':parameters.get('calibration_fingerprint'),'skeleton_runtime_revision':(model.last_frame or {}).get('skeleton_runtime_revision'),'source_payload_sha256':capture.metadata.get('source_payload_sha256'),'fit_revision':FIT_REVISION})
        fit_started=time.monotonic();fit_started_at_ms=time.time()*1000
        def report(v):
            nonlocal fit_status
            if 'training_sample' in v:v={**fit_status,**v}
            v=dict(v);v['method']=method;v['started_at_ms']=fit_started_at_ms;v['elapsed_s']=time.monotonic()-fit_started
            v['run_id']=run_id;v['training_options']=training_options
            if 'estimated_remaining_s' not in v:
                p=v.get('progress',0)
                v['estimated_remaining_s']=v['elapsed_s']*(1-p)/p if p>=.08 else None
            fit_status=v
            if 'training_sample' in v or v.get('state') not in ('running','cancelling'):
                with trace_path.open('a') as log:log.write(json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n')
            emit({'type':'training' if 'training_sample' in v else 'state','fit':fit_status})
        def run():
            try:
                from moxian_web.fit_preparation import copy_rows
                data_started=time.monotonic();prepared_data=copy_rows(data,cancel_fit,report);data_copy_s=time.monotonic()-data_started
                if warmup_thread is not None and not warmup_done.is_set():
                    report(dict(state='running',phase='compile',progress=0,message='等待 GPU 首次编译预热完成',estimated_remaining_s=None))
                    while not warmup_done.wait(.2):
                        if cancel_fit.is_set():raise InterruptedError('标定已取消')
                fingerprint=(model.last_frame or {}).get('calibration_fingerprint')
                if not fingerprint:raise ValueError('尚无官方腕部坐标映射，请等待骨架就绪')
                runtime_revision=(model.last_frame or {}).get('skeleton_runtime_revision')
                result,summary=fit_capture(args.model,parameters,mapping,prepared_data,report,cancel_fit,surface_backend='mano',training_options=training_options,method=method,initial_candidate=initial)
                summary.update(run_id=run_id,training_options=training_options,data_copy_s=data_copy_s,request_to_completion_s=time.monotonic()-fit_started)
                report({'state':'completed','message':'训练及完整留出验收已完成','accepted':summary.get('accepted')})
                fit_results.put(('success',(result,summary)))
            except Exception as exc:
                traceback.print_exc();report({'state':'cancelled' if cancel_fit.is_set() else 'failed','error':str(exc)});fit_results.put(('error',str(exc)))
        fitting=threading.Thread(target=run,daemon=True);fitting.start()
    def apply_candidate(automatic=False):
        nonlocal profile,preview,preview_base,fit_status
        if comparison is not None:raise ValueError('对比预览不能直接应用；请先返回当前模型')
        report=(candidate or {}).get('report',{})
        if not calibration_report_valid(report) or report.get('fit_revision')!=FIT_REVISION:raise ValueError('候选未通过完整留出及2 mm碰撞检查，当前模型保留')
        if not model.mapping or candidate.get('mapping',{}).get('assignments')!=model.mapping['assignments']:raise ValueError('SID 分区已改变，请重新拟合')
        if candidate['material']!=model.material:raise ValueError('材料先验已改变，请重新拟合')
        if candidate.get('capture_session_id')!=capture.metadata.get('motion_session_id'):raise ValueError('采集记录已改变，请使用当前完整记录拟合')
        if candidate.get('penetration_tolerance_m')!=HARD_PENETRATION_M:raise ValueError('候选必须采用2 mm硬验收')
        validate_profile(candidate)
        if model.last_frame and candidate.get('calibration_fingerprint')!=model.last_frame.get('calibration_fingerprint'):raise ValueError('官方骨架版本已改变，请重新拟合')
        if model.last_frame and not tracking_registration_matches(candidate,model.last_frame,pending=True):raise ValueError('骨架运动学／安装变换已更新，请重新验证候选')
        profile=persist_calibrated_profile(home,candidate,automatic,previous_profile=preview_base if preview and preview_base is not None else current_parameters());load(profile);preview=False;preview_base=None
        fit_status=copy.deepcopy(candidate['report']);fit_status.update(applied=True,automatic_application=automatic,message='已标定：模型已自动应用' if automatic else '已标定：模型已应用')
        emit({'type':'state','calibrated':True,'preview':False,'applied_revision':profile['revision'],'fit':fit_status})
    pending_auto_apply=bool(candidate and not (home/'current.json').exists() and candidate.get('report',{}).get('fit_revision')==FIT_REVISION and candidate.get('report',{}).get('accepted'))
    flat_pose_calibration=None;flat_pose_active=False
    try:flat_pose_calibration=json.loads((home/'flat-pose-calibration.json').read_text())
    except (OSError,ValueError):pass
    attempted_import=None
    last_output=0.;last_capture=0.;last_storage_error=None;topology_version=None;last_input=None;processing=[];frame_count=0;collision_at=0.;collision_signature=None;cached_fusion=None;cached_geom=None
    while not stopping.is_set():
        try:
            while True:
                session,outcome,value=capture_preparations.get_nowait()
                if not isinstance(capture,ContactMotionCapture) or session!=capture.metadata.get('motion_session_id') or capture.status=='cancelled':continue
                if outcome=='ready':
                    if isinstance(value,tuple):capture,initial_capture_cache=value
                    else:capture.reference=value;capture.metadata['contact_reference']=value.spec;capture._ready()
                    checkpoint()
                elif outcome=='error':
                    capture.status='paused';capture.error='冻结参考初始化失败：'+value;checkpoint()
                else:capture.error=value.get('message',capture.error)
                emit({'type':'state','capture':capture.snapshot()})
        except queue.Empty:pass
        try:
            while True:
                outcome,value=fit_results.get_nowait()
                if outcome=='flat_pose':
                    flat_pose_calibration=value;write_json(home/'flat-pose-calibration.json',value)
                    emit({'type':'state','flat_pose_calibration':value});continue
                if outcome=='success':
                    candidate,fit_status=value;candidate['report']=fit_status
                    if (home/'candidate.json').exists():
                        history=home/'candidate-history';history.mkdir(exist_ok=True)
                        previous=json.loads((home/'candidate.json').read_text())
                        write_json(history/f'{time.time_ns()}.json',previous)
                    method_home=home/'methods'/fit_status.get('method','legacy');method_home.mkdir(parents=True,exist_ok=True)
                    write_json(method_home/'candidate.json',candidate);write_json(method_home/'report.json',fit_status)
                    write_json(home/'candidate.json',candidate)
                    np.savez(home/'candidate.npz',dimension_parameters=candidate.get('dimension_parameters',[]),sensor_joint_position=candidate.get('sensor_joint_position',[]),shape=candidate['shape'],rest_vertex_residual_m=candidate.get('rest_vertex_residual_m',np.zeros((778,3))),bone_reference=candidate['bone_reference'],mount_pos=candidate['mount_pos'],mount_rot=candidate['mount_rot'],raw_rotation=candidate['raw_rotation'],raw_translation=candidate['raw_translation'])
                    if fit_status.get('accepted'):
                        try:apply_candidate(automatic=True)
                        except Exception as exc:fit_status={**fit_status,'application_error':str(exc),'message':'自动应用未完成：'+str(exc)}
                else:fit_status={'state':'cancelled' if cancel_fit.is_set() else 'failed','error':value}
                write_json(home/'shared-fit-report.json',{'calibration_fingerprint':capture.metadata.get('calibration_fingerprint'),'fit':fit_status})
                emit({'type':'state','fit':fit_status})
        except queue.Empty:pass
        try:
            control=controls.get_nowait();action=control['action'];data=control.get('data') or {}
            try:
                if comparison is not None and action not in ('checkpoint_preview','checkpoint_end','fit_cancel'):
                    raise ValueError('当前为权重对比预览，请先返回当前模型')
                if action=='capture_start':
                    if capture.status=='preparing':raise ValueError('参考模型正在后台准备，请稍候')
                    if fitting and fitting.is_alive() or preview:raise ValueError('请先结束拟合或预览')
                    warmup_cancel.set()
                    fingerprint=(model.last_frame or {}).get('calibration_fingerprint')
                    if not fingerprint or not model.mapping:raise ValueError('等待骨架初始化与 SID 映射')
                    if capture.rows:
                        history=home/'capture-history'/str(time.time_ns())/'motion-capture.json'
                        capture_saves.submit([history],capture_document(),history=True)
                    initial=initial_capture_cache
                    if initial and (initial.get('calibration_fingerprint')!=fingerprint or initial['rows'][0]['mapping']['assignments']!=model.mapping['assignments']):initial=None
                    capture=ContactMotionCapture(dict(rows=[],completed_groups=[],calibration_fingerprint=fingerprint,side=args.side))
                    prepare_capture_reference(capture,initial,new_session=True);clear_capture_frames()
                elif action in ('capture_confirm','capture_resume','capture_retry','capture_cancel'):
                    if not isinstance(capture,MotionCapture):raise ValueError('请先开始连续标定')
                    if fitting and fitting.is_alive():raise ValueError('请先取消拟合')
                    if action in ('capture_confirm','capture_resume'):
                        clear_capture_frames();capture.confirm(data.get('request_id'))
                    else:
                        capture.check_stage(data.get('stage_id'))
                        if action=='capture_retry':
                            capture.retry()
                            if isinstance(capture,ContactMotionCapture) and capture.reference is None:
                                prepare_capture_reference(capture,initial_capture_cache,new_session=not bool(capture.rows))
                        else:
                            capture.cancel();capture_preparation_cancel.set()
                    checkpoint()
                elif action=='fit':start_fit(data.get('method','parameters'),data.get('training_options'),data.get('resume',False),data.get('resume_run_id'))
                elif action=='fit_cancel':
                    cancel_fit.set()
                    if fitting and fitting.is_alive():
                        fit_status={**fit_status,'state':'cancelling','message':'已请求取消；当前计算或首次编译结束后停止，当前模型保留'}
                        emit({'type':'state','fit':fit_status})
                elif action=='checkpoint_preview':
                    if fitting and fitting.is_alive() or capture.status=='collecting':raise ValueError('请先结束拟合或采集，再对比权重')
                    if preview and comparison is None:raise ValueError('请先结束原候选预览')
                    proposed=make_comparison_profile(data['checkpoint'],model,model.last_frame,model.mapping,args.serial,args.side)
                    validate_profile(proposed)
                    baseline=preview_base if comparison is not None else current_parameters()
                    base_dimensions=(comparison_info or {}).get('base_dimensions_mm') if comparison is not None else model.dimension_summary()
                    try:load(proposed)
                    except Exception:
                        load(comparison if comparison is not None else baseline)
                        raise
                    if comparison is None:preview_base=baseline
                    comparison=proposed;comparison_info={**copy.deepcopy(data['info']),'base_dimensions_mm':base_dimensions};preview=True
                    cached_fusion=None;cached_geom=None;collision_signature=None
                    emit({'type':'state','status':'switching','preview':True,'checkpoint_preview':comparison_info})
                elif action=='checkpoint_end':
                    if comparison is not None:
                        load(preview_base if preview_base is not None else profile)
                        comparison=None;comparison_info=None;preview_base=None;preview=False
                        cached_fusion=None;cached_geom=None;collision_signature=None
                    emit({'type':'state','status':'switching','preview':preview,'checkpoint_preview':None})
                elif action=='preview':
                    if not candidate:raise ValueError('没有候选手型')
                    if not calibration_report_valid(candidate.get('report',{})):raise ValueError('候选未通过完整验收；失败候选仅保留离线诊断，不应用到实时预览')
                    if candidate.get('skeleton_model'):raise ValueError('联合候选须整体应用，不能单独预览 MANO')
                    enabled=bool(data.get('enabled',True))
                    if enabled!=preview:
                        if enabled:
                            preview_base=current_parameters();load(candidate)
                        else:
                            load(preview_base if preview_base is not None else profile);preview_base=None
                        preview=enabled
                elif action=='apply':
                    apply_candidate(automatic=False)
                elif action=='restore':
                    if not (home/'previous.json').exists():raise ValueError('尚无上一个已应用版本')
                    old=json.loads((home/'previous.json').read_text());validate_profile(old)
                    write_json(home/'previous.json',profile);profile=old;write_json(home/'current.json',profile);load(profile);preview=False;preview_base=None;fit_status={**fit_status,'message':'已恢复上一版本','applied':False}
                elif action=='fit_options':
                    if fitting and fitting.is_alive():raise ValueError('请先结束拟合，再修改容差')
                    tolerance=float(data.get('penetration_tolerance_mm',2))/1000
                    if not np.isfinite(tolerance) or abs(tolerance-HARD_PENETRATION_M)>1e-9:raise ValueError('穿透硬验收固定为2 mm')
                    profile['penetration_tolerance_m']=HARD_PENETRATION_M;write_json(home/'fit-settings.json',{'penetration_tolerance_m':HARD_PENETRATION_M})
                elif action=='flat_reference':
                    if fitting and fitting.is_alive() or capture.status=='collecting' or preview:raise ValueError('请先结束采集、拟合或候选预览，再校正平摊参考')
                    reference=data.get('reference')
                    if reference is None:
                        samples=[r for r in reference_samples if r['timestamp_ms']>=reference_samples[-1]['timestamp_ms']-900] if reference_samples else []
                        if len(samples)<15 or samples[-1]['timestamp_ms']-samples[0]['timestamp_ms']<600:raise ValueError('请佩戴并平摊，稳定保持约一秒后重试')
                        positions=np.array([[r['fingers'][f]['position'] for f in ('thumb','index','middle','ring','pinky')] for r in samples])
                        rotations=[Rotation.from_quat([r['fingers'][f]['quaternion_xyzw'] for r in samples]) for f in ('thumb','index','middle','ring','pinky')]
                        means=[r.mean() for r in rotations]
                        if np.max(np.linalg.norm(positions-np.median(positions,axis=0),axis=-1))>.002 or max(np.max((mean.inv()*r).magnitude()) for mean,r in zip(means,rotations))>np.deg2rad(4):raise ValueError('姿态尚不稳定，请平摊保持约一秒')
                        reference={'pose':'flat_open','hand_side':args.side,'observation_space':'raw_emf_sensor','positions_local':np.median(positions,axis=0).tolist(),'rotations_local':[r.as_matrix().ravel().tolist() for r in means],'timestamp_ms':samples[-1]['timestamp_ms'],'calibration_fingerprint':samples[-1].get('calibration_fingerprint'),'source':'user_confirmed_stable_raw_frames','reference_id':uuid.uuid4().hex}
                    if reference.get('calibration_fingerprint')!=(model.last_frame or {}).get('calibration_fingerprint'):raise ValueError('平摊参考与当前官方标定不匹配')
                    if model.official_fk is not None:
                        reference=reference_from_frame(reference_samples[-1],model.weights_hash,args.serial,uuid.uuid4().hex)
                    reference={**reference,'weights_hash':model.weights_hash,'serial':args.serial}
                    model.install_flat_reference(reference);model.mounting_calibrated=False
                    if (home/'sensor-reference.json').exists():write_json(home/'sensor-reference.previous.json',json.loads((home/'sensor-reference.json').read_text()))
                    write_json(home/'sensor-reference.json',reference)
                elif action=='material':
                    model.material=material_config(data.get('material',{}));profile['material']=model.material
                    write_json(home/'material.json',{'weights_hash':model.weights_hash,'material':model.material})
                    if profile.get('calibrated'):write_json(home/'current.json',profile)
                elif action=='compression_display':
                    if type(data.get('enabled')) is not bool:raise ValueError('压缩显示需要 enabled 布尔值')
                    if data['enabled']:raise ValueError('实时压缩显示已停用；仅显示绿色接触点，不改变曲面或关节')
                    compression_display=False;model.compression[:]=0;model.last_time=None
                elif action=='registration_get':
                    emit({'type':'ack','id':control['id'],'result':{'status':model.contact_registration_status,'artifact':model.contact_registration}})
                    continue
                elif action in ('registration_import','registration_restore'):
                    if fitting and fitting.is_alive() or preview:raise ValueError('请先结束拟合或预览，再导入实测接触注册')
                    artifact=(json.loads((home/'contact-registration.previous.json').read_text())
                              if action=='registration_restore' else data.get('artifact'))
                    if not isinstance(artifact,dict):raise ValueError('缺少实测材料注册文件')
                    # Validate before persisting or changing any active points.
                    from moxian_web.contact_registration import validate_registration
                    validate_registration(artifact,model.registration_topology(),args.serial)
                    path=home/'contact-registration.json'
                    if path.exists():write_json(home/'contact-registration.previous.json',json.loads(path.read_text()))
                    # Importing measured coordinates supersedes older manual
                    # clicks for these SIDs. Unmeasured SIDs retain their edits.
                    for sid in artifact['records']:model.binding_overrides.pop(sid,None)
                    profile['binding_overrides']=copy.deepcopy(model.binding_overrides)
                    write_json(home/'bindings.json',{'weights_hash':model.weights_hash,'overrides':model.binding_overrides})
                    write_json(path,artifact);model.install_contact_registration(artifact,args.serial)
                    if profile.get('calibrated'):write_json(home/'current.json',profile)
                elif action=='bind_sid':
                    sids=data.get('sids');face=data.get('face');bary=data.get('barycentric')
                    if not isinstance(sids,list) or not sids or any(type(s) is not int or not 0<=s<460 for s in sids):raise ValueError('请选择有效 SID')
                    if type(face) is not int or not 0<=face<len(model.faces) or len(bary or [])!=3 or not np.isfinite(bary).all() or min(bary)<0 or abs(sum(bary)-1)>1e-5:raise ValueError('无效的曲面定位')
                    if any(model.mapping['assignments'][sid]!=model.face_patches[face] for sid in sids):raise ValueError('绑定点必须位于所选 SID 的解剖 patch 内')
                    for sid in sids:model.binding_overrides[str(sid)]={'face':face,'barycentric':bary}
                    model.build_binding(model.mapping);profile['binding_overrides']=model.binding_overrides;write_json(home/'bindings.json',{'weights_hash':model.weights_hash,'overrides':model.binding_overrides})
                    if profile.get('calibrated'):write_json(home/'current.json',profile)
                else:raise ValueError('未知 MANO 操作')
                emit({'type':'ack','id':control['id'],'result':{'ok':True}})
                emit({'type':'state','capture':capture.snapshot(),'fit':fit_status,'gpu_warmup':warmup_state,'material':model.material,'preview':preview,'checkpoint_preview':comparison_info,'calibrated':bool(profile.get('calibrated')) and comparison is None,'display_options':{'compression':False,'physics':False,'contact_points':True}})
            except Exception as exc:emit({'type':'ack','id':control['id'],'error':str(exc)})
        except queue.Empty:pass
        if capture_saves.error!=last_storage_error:
            last_storage_error=capture_saves.error
            emit({'type':'state','capture_storage_error':last_storage_error})
        # Consume every synchronized observation independently of expensive
        # MANO inference. The rendering queue deliberately retains only latest.
        capture_changed=False
        if isinstance(capture,MotionCapture) and capture.status=='collecting' and capture_overflow.is_set():
            capture.pause_cycle('原始采集缓冲已满；继续当前一轮，已完成轮次保留');capture_changed=True
        capture_batch_started=time.monotonic()
        for _ in range(32):
            if time.monotonic()-capture_batch_started>.012:break
            try:observation=capture_frames.get_nowait()
            except queue.Empty:break
            if isinstance(capture,MotionCapture) and capture.status=='collecting' and (capture.frame_usable(observation) if isinstance(capture,ContactMotionCapture) else aligned_frame(observation)):
                capture_changed=capture.feed(observation) or capture_changed
        if isinstance(capture,ContactMotionCapture) and capture.status=='collecting' and time.monotonic()-last_capture>.15:
            last_capture=time.monotonic();emit({'type':'state','capture':capture.snapshot()})
        if capture_changed:
            if capture.status=='complete':fit_status={'state':'idle','accepted':False,'message':'训练与独立复测已采集完成，请点击“开始拟合”。'}
            checkpoint();emit({'type':'state','capture':capture.snapshot(),'fit':fit_status})
        try:frame=frames.get(timeout=.01)
        except queue.Empty:continue
        if not aligned_frame(frame):
            emit({'type':'state','status':'waiting_input','error':'定位／触觉时间差超过 25 ms 或输入无效'});continue
        if not frame.get('skeleton_targets'):
            emit({'type':'state','status':'waiting_input','error':'等待官方腕部坐标指尖目标，避免使用单帧刚性猜测'});continue
        if last_input is not None and frame['timestamp_ms']<=last_input:continue
        last_input=frame['timestamp_ms'];reference_samples.append(frame)
        try:
            # Historical independent flat offsets are diagnostic-only.
            flat_pose_active=False
            model.calibration_pose_offsets[:]=0
            if pending_reference is not None:
                saved_reference=pending_reference;pending_reference=None
                compatible=saved_reference.get('reference_policy')==OFFICIAL_REFERENCE_POLICY and saved_reference.get('weights_hash')==model.weights_hash and saved_reference.get('serial')==args.serial and saved_reference.get('calibration_fingerprint')==frame.get('calibration_fingerprint')
                if compatible and not (model.mounting_calibrated and model.flat_reference and model.flat_reference.get('reference_id')==saved_reference.get('reference_id')):
                    model.observations(frame);model.last_frame=frame;model.install_flat_reference(saved_reference)
            # Rest dimensions remain valid FK geometry after a new official
            # calibration. Only the old observation registration is stale;
            # never hide the mesh while waiting for a shared-data refit.
            registration_matches=tracking_registration_matches(comparison if comparison is not None else profile,frame) and bool(model.flat_reference)
            mapping=frame.get('mapping')
            if mapping and (not model.mapping or model.mapping['version']!=mapping['version']):
                if fitting and fitting.is_alive() and model.mapping and mapping['assignments']!=model.mapping['assignments']:cancel_fit.set()
                model.build_binding(mapping)
            if not model.mapping:continue
            if not (fitting and fitting.is_alive()):start_warmup(frame)
            fingerprint=frame.get('calibration_fingerprint')
            if fitting and fitting.is_alive() and capture.metadata.get('calibration_fingerprint')!=fingerprint:cancel_fit.set()
            if fingerprint and attempted_import!=fingerprint and not isinstance(capture,MotionCapture) and not (fitting and fitting.is_alive()):
                attempted_import=fingerprint
                try:
                    key=hashlib.sha256(args.serial.encode()).hexdigest()[:16]
                    shared=load_official_capture(home.parents[1]/'ik'/key,fingerprint,model.mapping,side=args.side)
                    initial_capture_cache=shared
                    saved_motion=json.loads((home/'motion-capture.json').read_text()) if (home/'motion-capture.json').exists() else None
                    if saved_motion and saved_motion.get('calibration_fingerprint')==fingerprint and saved_motion.get('source_payload_sha256')==shared['source_payload_sha256'] and saved_motion['rows'][0]['mapping']['assignments']==model.mapping['assignments']:
                        capture=MotionCapture.restore(saved_motion,allow_partial=True)
                        if isinstance(capture,ContactMotionCapture) and capture.status!='complete':
                            capture.extend_spread_schedule()
                            prepare_capture_reference(capture,shared)
                    else:capture.install(shared)
                    checkpoint()
                    # MANO joints and geometry remain entirely inside the official model.
                    # Build EMF registration on the private fitting model after
                    # raw captures have been replayed through the official SDK.
                    # Live geometry never waits for or invents a flat-pose IK.
                    model.last_frame=frame
                    attempt={'calibration_fingerprint':fingerprint,'skeleton_runtime_revision':frame.get('skeleton_runtime_revision'),'source_payload_sha256':shared['source_payload_sha256'],'fit_revision':FIT_REVISION}
                    previous=json.loads((home/'shared-fit-attempt.json').read_text()) if (home/'shared-fit-attempt.json').exists() else None
                    if previous!=attempt and capture.metadata.get('protocol') in MOTION_PROTOCOLS and capture.status=='complete':fit_status={'state':'idle','accepted':False,'message':'完整采集记录已恢复；训练版本更新，请选择方法并点击重新拟合。显示仅使用固定模型前向。'}
                    elif isinstance(capture,MotionCapture) and capture.status!='complete':fit_status={'state':'idle','message':'连续采集记录已恢复；请继续当前一轮。'}
                    elif (home/'shared-fit-report.json').exists():
                        saved=json.loads((home/'shared-fit-report.json').read_text())
                        if saved.get('calibration_fingerprint')==fingerprint:fit_status=saved['fit']
                    else:fit_status={'state':'idle','message':'八动作初始化已复用；请开始连续联合标定，约 2–3 分钟。'}
                except Exception as exc:
                    capture.error=str(exc);fit_status={'state':'failed','error':'八动作复用／拟合初始化：'+str(exc)}
                registration_matches=tracking_registration_matches(comparison if comparison is not None else profile,frame) and bool(model.flat_reference)
            if pending_auto_apply:
                pending_auto_apply=False;model.last_frame=frame;apply_candidate(automatic=True)
            shown_profile=comparison if comparison is not None else profile
            active_hash=tracking_geometry_hash(shown_profile) if (comparison is not None or profile.get('calibrated')) and shown_profile.get('calibration_fingerprint')==fingerprint else None
            incoming_hash=frame['skeleton'].get('personalized_geometry_hash')
            if (active_hash or incoming_hash) and incoming_hash!=active_hash:
                emit({'type':'state','status':'waiting_input','error':'等待骨架与 MANO 同步个体标定版本'});continue
            if time.monotonic()-last_output<1/60:continue
            began=time.perf_counter();v,j,rotations,ordered,rmse=model.track(frame,steps=1)
            forward_ms=(time.perf_counter()-began)*1000;collision_began=time.perf_counter()
            guide=capture.snapshot()['stage_id'] if capture.status=='collecting' else None
            signature=(model.shape_version,model.binding_version,frame['contact'].get('ready'),tuple(p.get('contact') for p in frame['contact'].get('patches',[])),guide)
            if cached_fusion is None or signature!=collision_signature or time.monotonic()-collision_at>=.04:
                if model.geometry is None:model.geometry=CollisionSurface(model)
                cached_geom=model.geometry.query(v,j,rotations)
                cached_fusion=fuse_contacts(model,frame,cached_geom,guide)
                cached_fusion['timestamp_ms']=frame['skeleton']['timestamp_ms'];collision_signature=signature;collision_at=time.monotonic()
            geom=cached_geom;fusion=cached_fusion
            collision_ms=(time.perf_counter()-collision_began)*1000
            deformed,deformation=deform_surface(model,v,fusion,frame['timestamp_ms'],enabled=False)
            current_version=(model.shape_version,model.binding_version)
            if topology_version!=current_version:
                emit({'type':'topology','model':model.topology()});topology_version=current_version;dimensions=model.dimension_summary()
            elapsed=(time.perf_counter()-began)*1000;processing.append(elapsed);processing=processing[-100:];frame_count+=1;last_output=began
            emit({'type':'frame','status':'preview' if preview else 'calibrated' if profile.get('calibrated') else 'default',
                  'error':None,'hand_side':args.side,'coordinate_frame':'mano_wrist_local','surface':'gloved_effective','units':'m',
                  'skeleton_timestamp_ms':frame['skeleton']['timestamp_ms'],'vertices_m':np.round(deformed,7).tolist(),'joints_m':np.round(ordered,7).tolist(),'pose_axis_angle':model.last_pose.tolist(),'joint_base_offsets_m':model.last_base_offsets.tolist() if model.last_base_offsets is not None else None,'joint_centres_m':model.last_joint_centres.tolist() if model.last_joint_centres is not None else None,'tracking_source':'official_qnbot_skeleton_fk_v1','official_wrist_to_mano_rotation':model.official_fk.sdk_to_mano.tolist(),
                  'root_rotation':model.last_root.tolist(),'tip_positions_m':deformed[[744,320,443,554,671]].tolist(),
                  'input_targets':{'timestamp_ms':frame['timestamp_ms'],'hand_side':args.side,'coordinate_frame':'mano_wrist_local','positions_local':(np.array([frame['fingers'][f]['position'] for f in ('thumb','index','middle','ring','pinky')])@model.raw_rotation.T+model.raw_translation).tolist()} if registration_matches else None,
                  'input_sensors':frame['fingers'],'sensor_positions_m':model.sensor_predictions.tolist(),'sensor_mounting':{'status':'candidate_preview' if comparison is not None else 'stale' if model.flat_reference and not registration_matches else 'multi_pose_calibrated' if model.mounting_calibrated else 'multi_pose_candidate' if model.mounting_estimated else 'position_prior' if model.flat_reference else 'unregistered','distal_to_emf_position_m':model.sensor_offsets().tolist(),'distal_to_emf_rotation':model.mount_rot.tolist(),'flat_reference_id':model.flat_reference.get('reference_id') if model.flat_reference else None,'emf_to_mano_rotation':model.raw_rotation.tolist(),'emf_to_mano_translation_m':model.raw_translation.tolist(),'surface_tip_is_sensor':False},
                  'timestamp_ms':frame['skeleton']['timestamp_ms'],'input_timestamp_ms':frame['timestamp_ms'],'tactile_timestamp_ms':frame['tactile_timestamp_ms'],
                  'source_contact':{k:frame['contact'].get(k) for k in ('ready','status','cell_contact','normalized','timestamp_ms','mapping_version')},
                  'prediction_reference':{'timestamp_ms':frame['timestamp_ms'],'raw_adc':list(frame['raw'])},
                  'contact_registration':model.contact_registration_status,
                  'self_contact_pairs':model.retarget_contacts,'tactile_pose_correction':model.contact_correction,
                  'raw_official_pose_axis_angle':model.raw_official_pose.tolist(),'calibration_pose_offsets':model.calibration_pose_offsets.tolist(),
                  'flat_pose_calibration':{'applied':flat_pose_active,'validation':(flat_pose_calibration or {}).get('validation'),'revision':(flat_pose_calibration or {}).get('revision')},
                  'pose_calibration_version':hashlib.sha256(model.calibration_pose_offsets.tobytes()).hexdigest()[:16],
                  'skeleton_runtime_revision':frame.get('skeleton_runtime_revision'),
                  'geometry_version':model.shape_version,'geometry_revision':GEOMETRY_REVISION,'dimension_mode':DIMENSION_MODE,'dimension_parameters':model.dimension_parameters.tolist(),'proximal_lateral_scales':model.proximal_lateral_scales.tolist(),'pose_correctives':True,'model_implementation':'smplx.MANO+fixed_glove_geometry' if hasattr(model,'glove_parameters') else 'smplx.MANO','shape_parameter_count':10,'fixed_geometry_parameter_count':43 if hasattr(model,'glove_parameters') else 0,'betas':model.shape[:10].tolist(),'penetration_tolerance_m':profile.get('penetration_tolerance_m',PENETRATION_TOLERANCE_M),'binding_version':model.binding_version,'calibrated':bool(profile.get('calibrated')) and comparison is None,
                  'runtime_backend':shown_profile.get('runtime_backend','legacy'),'checkpoint_preview':comparison_info,
                  'shared_skeleton_pose':{'timestamp_ms':frame['timestamp_ms'],'hand_side':args.side,'coordinate_frame':'wrist_local','names':['wrist']+[f'{f}_{k}' for f in ('thumb','index','middle','ring','pinky') for k in range(4)],'positions_local':(ordered@model.official_fk.mano_to_sdk.T).tolist(),'mano_pose_axis_angle':model.last_pose.tolist()},
                  'dimensions_mm':dimensions,'penetration_policy':'hard_calibration_gate_runtime_best_finite_with_warning','mano_solver':model.solver_report,'constraints_satisfied':model.solver_report.get('constraints_satisfied',False),'applied_revision':profile.get('revision'),'fit_revision':FIT_REVISION,'calibration_warnings':profile.get('report',{}).get('quality_warnings',[])+(['当前几何沿用旧标定；旧独立 IK 的验收数字不代表官方 FK 驱动质量'] if profile.get('calibrated') and profile.get('report',{}).get('fit_revision')!=FIT_REVISION else []),'calibration_matches_tracking':profile.get('report',{}).get('fit_revision')==FIT_REVISION and tracking_registration_matches(profile,frame),'registration_matches_tracking':registration_matches,'calibration_pair_gaps_mm':profile.get('report',{}).get('together_pair_gaps_mm'),
                  'surface_refinement':profile.get('report',{}).get('surface_refinement'),'rigid_finger_fk':model.rigid_finger_fk,'finger_base_offsets_m':model.finger_base_offsets_m.tolist(),'finger_axis_yaw_rad':model.finger_axis_yaw_rad.tolist(),'rigid_spacing_report':profile.get('rigid_spacing_report'),
                  'display_options':{'compression':False,'physics':False,'contact_points':True},
                  'contact_binary':[p['value'] for p in fusion['patches']],'contact_points':confirmed_contact_points(fusion),'simulation':{'dynamics':False,'contact_forces':False,'collision_checking':True},
                  'calibration_process':{'capture_complete':capture.status=='complete','fit_accepted':bool(fit_status.get('accepted')),'dimensions_applied':bool(profile.get('calibrated')) and comparison is None},'preview':preview,'capture':capture.snapshot(),'capture_storage':capture_saves.snapshot(),'fit':fit_status,'gpu_warmup':warmup_state,'material':model.material,'fusion':fusion,'deformation':deformation,
                  'quality':{'position_rmse_mm':rmse if registration_matches else None,'processing_ms':elapsed,'mean_processing_ms':float(np.mean(processing)),
                    'forward_ms':forward_ms,'collision_ms':collision_ms,
                    'skeleton_delta_ms':abs(frame['timestamp_ms']-frame['skeleton']['timestamp_ms']) if frame.get('skeleton') else None,
                    'tactile_delta_ms':abs(frame['timestamp_ms']-frame['tactile_timestamp_ms']),
                    'tracking':'official_qnbot_skeleton_fk_v1','skeleton_solver':frame['skeleton'].get('ik_solver','qnbot_native'),'joint_limits':True,'joint_limits_source':'bounded_anatomical_MANO_solver','additional_joint_projection':True,'contact_affects_joints':bool(model.contact_correction.get('applied')),'contact_force_feedback':False,'sensor_contact_pose_constraint':True,'thumb':model.thumb_quality,'alignment':model.alignment_quality,
                    'collision_age_ms':frame['skeleton']['timestamp_ms']-fusion['timestamp_ms'],'source_age_ms':time.time()*1000-frame['timestamp_ms'],'dispatch_age_ms':(frame.get('dispatch_timestamp_ms',frame['timestamp_ms'])-frame['timestamp_ms']),'native_contacts':geom['native_count'],'collision_overflow':geom['overflow'],
                             'frame_count':frame_count,'force_calibrated':False,'unobserved_patches':[p['id'] for p in frame['contact'].get('patches',[]) if p['id'] not in model.mapping.get('confirmed',[])]}})
        except Exception as exc:
            traceback.print_exc();emit({'type':'state','status':'waiting_input','error':str(exc),'capture':capture.snapshot(),'fit':fit_status});time.sleep(.05)
    checkpoint();cancel_fit.set();capture_preparation_cancel.set();warmup_cancel.set();capture_saves.close()

if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise
