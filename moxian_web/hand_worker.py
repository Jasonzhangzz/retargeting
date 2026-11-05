"""Isolated CUDA MANO process. Never opens a serial port or imports the glove SDK."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,queue,signal,sys,threading,time,traceback,uuid,subprocess
from pathlib import Path
from collections import deque
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moxian_web.hand_model import aligned_frame,material_config,DEFAULT_MATERIAL,confirmed_contact_points,persist_calibrated_profile
from moxian_web.hand_capture import HandCapture
from moxian_web.mano_runtime import ManoSurface,CollisionSurface,fuse_contacts,deform_surface
from moxian_web.mano_fit import fit_capture,FIT_REVISION
from moxian_web.hand_anatomy import GEOMETRY_REVISION
from moxian_web.hand_dimensions import DIMENSION_MODE,DIMENSION_COUNT,PENETRATION_TOLERANCE_M
import numpy as np
import torch


def write_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2));tmp.replace(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--event-fd',type=int,required=True);ap.add_argument('--home',type=Path,required=True);ap.add_argument('--model',type=Path,required=True);ap.add_argument('--serial',required=True);args=ap.parse_args()
    home=args.home;home.mkdir(parents=True,exist_ok=True)
    stream=os.fdopen(args.event_fd,'w',buffering=1);event_lock=threading.Lock();stopping=threading.Event();cancel_fit=threading.Event()
    def emit(value):
        with event_lock:stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n');stream.flush()
    def finish(_s=None,_f=None):stopping.set();cancel_fit.set()
    signal.signal(signal.SIGTERM,finish);signal.signal(signal.SIGINT,finish)
    frames=queue.Queue(maxsize=1);controls=queue.Queue(maxsize=32)
    def reader():
        for line in sys.stdin:
            try:
                m=json.loads(line)
                if m.get('type')=='control':controls.put(m)
                elif m.get('type')=='frame':
                    try:frames.put_nowait(m)
                    except queue.Full:
                        try:frames.get_nowait()
                        except queue.Empty:pass
                        frames.put_nowait(m)
            except Exception:traceback.print_exc()
        finish()
    threading.Thread(target=reader,daemon=True).start()
    torch.set_num_threads(1)
    emit({'type':'state','status':'starting','error':None,'message':'初始化左手 MANO 与 GPU 碰撞结构'})
    if not torch.cuda.is_available():raise RuntimeError('GPU 不可用；MANO 已停止，原骨架与触觉服务不受影响')
    model=ManoSurface(args.model);model.enable_official_fk();model.set_dimensions(np.zeros(DIMENSION_COUNT));capture=HandCapture();fit_status={'state':'idle'};candidate=None;fit_results=queue.Queue();fitting=None;preview=False;preview_base=None;compression_display=False
    profile={'schema_version':1,'serial':args.serial,'side':'left','operator':'local','weights_hash':model.weights_hash,
             'dimension_mode':DIMENSION_MODE,'dimension_parameters':model.dimension_parameters.tolist(),'proximal_lateral_scales':model.proximal_lateral_scales.tolist(),'penetration_tolerance_m':PENETRATION_TOLERANCE_M,'surface':'gloved_effective','geometry_revision':GEOMETRY_REVISION,'calibrated':False,'shape':model.shape.tolist(),'material':dict(DEFAULT_MATERIAL),
             'raw_rotation':model.raw_rotation.tolist(),'raw_translation':model.raw_translation.tolist(),
             'bone_reference':None,'mount_pos':model.mount_pos.tolist(),'mount_rot':model.mount_rot.tolist(),'response_gains':{},'binding_overrides':{}}
    def validate_profile(p):
        if p.get('schema_version')!=1 or p.get('side')!='left' or p.get('serial')!=args.serial or p.get('weights_hash')!=model.weights_hash:raise ValueError('手型记录与设备、左手或 MANO 权重不兼容')
        if p.get('calibrated') and not p.get('report',{}).get('contact_validation_passed'):raise ValueError('旧手型缺少四组有效对指检查，请使用保留记录重新拟合')
        if p.get('calibrated') and p.get('report',{}).get('validation_pose_source') not in ('runtime_emf_only','official_skeleton_fk'):raise ValueError('旧手型使用了触觉贴合验证，请用保留记录重新拟合并做独立验证')
        if p.get('calibrated') and p.get('geometry_revision')!=GEOMETRY_REVISION:raise ValueError('几何算法已更新，请用保留的十动作记录重新拟合；原手型文件保留')
        if p.get('calibrated') and model.last_frame and p.get('calibration_fingerprint')!=model.last_frame.get('calibration_fingerprint'):raise ValueError('官方骨架标定已改变，旧手型暂不兼容；请使用保留记录重新拟合')
        sensor=p.get('sensor_joint_position')
        if sensor is not None and (np.asarray(sensor).shape!=(5,3) or not np.isfinite(sensor).all() or np.max(np.linalg.norm(sensor,axis=1))>.1):raise ValueError('无效的末节到 EMF 安装位置')
        if p.get('observation_space')=='raw_emf_sensor' and p.get('mounting_calibrated') and sensor is None:raise ValueError('已标定 EMF 记录缺少独立安装变换')
        dimensions=np.asarray(p.get('dimension_parameters',np.zeros(DIMENSION_COUNT)),dtype=float)
        if dimensions.shape!=(DIMENSION_COUNT,) or not np.isfinite(dimensions).all():raise ValueError('固定手指尺寸参数无效')
        material_config(p['material'])
        if len(p.get('shape',[]))!=41 or not np.isfinite(p['shape']).all():raise ValueError('手型尺寸记录无效')
        for name,shape in [('raw_rotation',(3,3)),('raw_translation',(3,)),('mount_pos',(5,3)),('mount_rot',(5,3,3))]:
            value=np.asarray(p[name])
            if value.shape!=shape or not np.isfinite(value).all():raise ValueError('无效的传感器安装记录')
        if p.get('bone_reference') is not None:
            lengths=np.asarray(p['bone_reference'])
            if lengths.shape!=(5,3) or not np.isfinite(lengths).all() or np.any(lengths<.003) or np.any(lengths>.12):raise ValueError('无效的骨长参考')
    def load(p):
        validate_profile(p);model.material=material_config(p['material']);model.binding_overrides=copy.deepcopy(p.get('binding_overrides',{}))
        model.bone_reference=np.array(p['bone_reference']) if p.get('bone_reference') is not None else None
        model.dimension_parameters=np.array(p.get('dimension_parameters',np.zeros(DIMENSION_COUNT)),dtype=float)
        model.proximal_lateral_scales=np.array(p.get('proximal_lateral_scales',[0.]*4),dtype=float)
        model.set_shape(p['shape']);model.raw_rotation=np.array(p['raw_rotation']);model.raw_translation=np.array(p['raw_translation']);model.raw_aligned=bool(p.get('calibrated') or p.get('report') or p.get('flat_reference'))
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
    try:
        if (home/'capture.json').exists():
            saved=json.loads((home/'capture.json').read_text())
            if saved.get('weights_hash')==model.weights_hash:
                capture.rows=saved.get('rows',[]);capture.complete_groups=saved.get('completed_groups',[])
                if len(capture.complete_groups)==10:capture.status='complete';capture.index=9
                elif saved.get('state',{}).get('state') in ('awaiting_confirmation','collecting','failed'):
                    capture.index=len(capture.complete_groups);capture.status='awaiting_confirmation';capture.request_id=uuid.uuid4().hex
                    capture.error='采集已恢复，请摆好当前动作后确认'
    except Exception:traceback.print_exc()
    def checkpoint():
        write_json(home/'capture.json',{'schema_version':1,'weights_hash':model.weights_hash,'serial':args.serial,'side':'left','rows':capture.rows,'completed_groups':capture.complete_groups,'state':capture.snapshot()})
    try:
        settings=json.loads((home/'fit-settings.json').read_text());tolerance=float(settings['penetration_tolerance_m'])
        if 0<=tolerance<=.01:profile['penetration_tolerance_m']=tolerance
    except (OSError,ValueError,KeyError):pass
    try:
        saved_candidate=json.loads((home/'candidate.json').read_text())
        if saved_candidate.get('geometry_revision')==GEOMETRY_REVISION and saved_candidate.get('dimension_mode')==DIMENSION_MODE:
            validate_profile(saved_candidate);candidate=saved_candidate;fit_status=copy.deepcopy(candidate['report'])
            if profile.get('proximal_lateral_scales',[0.]*4)!=candidate.get('proximal_lateral_scales',[0.]*4):
                candidate=None;fit_status={'state':'superseded','accepted':False,'message':'已应用近端形状修正；旧候选保留。新拟合会以当前手型继续优化。'}
            if candidate and profile.get('calibrated') and profile.get('dimension_parameters')==candidate.get('dimension_parameters') and profile.get('report',{}).get('source_capture_hash')==candidate['report'].get('source_capture_hash'):fit_status.update(applied=True,automatic_application=bool(profile.get('automatic_application')),message='已标定：当前尺寸已生效')
    except (OSError,ValueError,KeyError):pass
    pending_reference=None;reference_samples=deque(maxlen=120)
    try:
        if (home/'sensor-reference.json').exists():pending_reference=json.loads((home/'sensor-reference.json').read_text())
    except (OSError,ValueError):pass
    def current_parameters():
        p=copy.deepcopy(profile);p.update(tracking_source='official_qnbot_skeleton_fk_v1',dimension_mode=DIMENSION_MODE,dimension_parameters=model.dimension_parameters.tolist(),penetration_tolerance_m=profile.get('penetration_tolerance_m',PENETRATION_TOLERANCE_M),sensor_joint_position=model.sensor_offsets().tolist() if model.flat_reference is not None else None,mounting_calibrated=model.mounting_calibrated,mounting_estimated=model.mounting_estimated,flat_reference=model.flat_reference,observation_space='raw_emf_sensor' if model.flat_reference is not None else 'sdk_wrist_targets' if model.has_wrist_targets else 'legacy_rigid_fit',calibration_fingerprint=(model.last_frame or {}).get('calibration_fingerprint'),shape=model.shape.tolist(),bone_reference=model.bone_reference.tolist() if model.bone_reference is not None else None,material=model.material,raw_rotation=model.raw_rotation.tolist(),raw_translation=model.raw_translation.tolist(),mount_pos=model.mount_pos.tolist(),mount_rot=model.mount_rot.tolist(),binding_overrides=model.binding_overrides)
        return p
    def start_fit():
        nonlocal fitting,fit_status
        if fitting and fitting.is_alive():raise ValueError('尺寸拟合正在运行')
        if capture.status!='complete':raise ValueError('请先完成十组手型采集')
        if not model.mapping:raise ValueError('等待 SID 分区映射')
        data=copy.deepcopy(capture.rows);mapping=copy.deepcopy(model.mapping);parameters=current_parameters();cancel_fit.clear()
        fit_status={'state':'running','progress':0,'message':'准备离线拟合'}
        def report(v):
            nonlocal fit_status
            fit_status=v
            emit({'type':'state','fit':fit_status})
        def run():
            try:
                fingerprint=(model.last_frame or {}).get('calibration_fingerprint')
                if not fingerprint:raise ValueError('尚无官方腕部坐标映射，请等待骨架就绪')
                runtime_revision=(model.last_frame or {}).get('skeleton_runtime_revision')
                if any(r.get('calibration_fingerprint')!=fingerprint or r.get('skeleton_runtime_revision')!=runtime_revision or not r.get('skeleton_targets') for r in data):
                    report({'state':'running','progress':.01,'message':'将保存的原始观测转换到官方腕部坐标；原记录保留'})
                    source=home/'fit-source.json';prepared=home/'fit-wrist-targets.json';write_json(source,{'rows':data})
                    root=Path(__file__).resolve().parents[1];key=hashlib.sha256(args.serial.encode()).hexdigest()[:16]
                    config=home.parents[1]/'ik'/key/'runtime'/f'kinematic-{fingerprint[:16]}.yaml'
                    subprocess.run([str(root/'.venv-moxian-ik/bin/python'),'-I',str(root/'moxian_web/prepare_wrist_targets.py'),
                                    '--input',str(source),'--output',str(prepared),'--config',str(config),'--fingerprint',fingerprint,'--runtime-revision',runtime_revision or ''],check=True,timeout=45,stdout=subprocess.DEVNULL)
                    prepared_rows=json.loads(prepared.read_text())['rows']
                else:prepared_rows=data
                fit_results.put(('success',fit_capture(args.model,parameters,mapping,prepared_rows,report,cancel_fit)))
            except Exception as exc:traceback.print_exc();fit_results.put(('error',str(exc)))
        fitting=threading.Thread(target=run,daemon=True);fitting.start()
    def apply_candidate(automatic=False):
        nonlocal profile,preview,preview_base,fit_status
        report=(candidate or {}).get('report',{})
        if not report.get('accepted') or not report.get('contact_validation_passed') or not report.get('together_validation_passed') or report.get('validation_pose_source')!='official_skeleton_fk' or report.get('fit_revision')!=FIT_REVISION:raise ValueError('候选未通过当前骨架绑定的独立标定检查，当前模型保留')
        if not model.mapping or candidate.get('mapping',{}).get('assignments')!=model.mapping['assignments']:raise ValueError('SID 分区已改变，请重新拟合')
        if candidate['material']!=model.material:raise ValueError('材料先验已改变，请重新拟合')
        if candidate.get('penetration_tolerance_m',PENETRATION_TOLERANCE_M)!=profile.get('penetration_tolerance_m',PENETRATION_TOLERANCE_M):raise ValueError('软惩罚参数已改变，请重新验证候选')
        validate_profile(candidate)
        if model.last_frame and candidate.get('calibration_fingerprint')!=model.last_frame.get('calibration_fingerprint'):raise ValueError('官方骨架版本已改变，请重新拟合')
        profile=persist_calibrated_profile(home,candidate,automatic,previous_profile=preview_base if preview and preview_base is not None else current_parameters());load(profile);preview=False;preview_base=None
        fit_status=copy.deepcopy(candidate['report']);fit_status.update(applied=True,automatic_application=automatic,message='已标定：模型已自动应用' if automatic else '已标定：模型已应用')
        emit({'type':'state','calibrated':True,'preview':False,'applied_revision':profile['revision'],'fit':fit_status})
    pending_auto_apply=bool(candidate and not (home/'current.json').exists() and candidate.get('report',{}).get('fit_revision')==FIT_REVISION and candidate.get('report',{}).get('accepted'))
    last_output=0.;last_capture=0.;topology_version=None;last_input=None;processing=[];frame_count=0;collision_at=0.;collision_signature=None;cached_fusion=None;cached_geom=None
    while not stopping.is_set():
        try:
            while True:
                outcome,value=fit_results.get_nowait()
                if outcome=='success':
                    candidate,fit_status=value;candidate['report']=fit_status;write_json(home/'candidate.json',candidate)
                    np.savez(home/'candidate.npz',dimension_parameters=candidate.get('dimension_parameters',[]),sensor_joint_position=candidate.get('sensor_joint_position',[]),shape=candidate['shape'],bone_reference=candidate['bone_reference'],mount_pos=candidate['mount_pos'],mount_rot=candidate['mount_rot'],raw_rotation=candidate['raw_rotation'],raw_translation=candidate['raw_translation'])
                    if fit_status.get('accepted'):
                        try:apply_candidate(automatic=True)
                        except Exception as exc:fit_status={**fit_status,'application_error':str(exc),'message':'自动应用未完成：'+str(exc)}
                else:fit_status={'state':'cancelled' if cancel_fit.is_set() else 'failed','error':value}
                emit({'type':'state','fit':fit_status})
        except queue.Empty:pass
        try:
            control=controls.get_nowait();action=control['action'];data=control.get('data') or {}
            try:
                if action=='capture_start':
                    if fitting and fitting.is_alive():raise ValueError('请先取消当前拟合')
                    if preview:raise ValueError('请先结束候选预览')
                    capture.start();checkpoint()
                elif action=='capture_confirm':capture.confirm(data.get('request_id'))
                elif action=='capture_resume':
                    if capture.status=='complete' or not capture.complete_groups:raise ValueError('没有可恢复的未完成采集')
                    capture.index=len(capture.complete_groups);capture.samples=[];capture.status='awaiting_confirmation';capture.request_id=uuid.uuid4().hex;capture.error=None
                elif action=='capture_retry':capture.retry(data.get('request_id'))
                elif action=='capture_cancel':capture.cancel();checkpoint()
                elif action=='fit':start_fit()
                elif action=='fit_cancel':cancel_fit.set()
                elif action=='preview':
                    if not candidate:raise ValueError('没有候选手型')
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
                    tolerance=float(data.get('penetration_tolerance_mm',5))/1000
                    if not np.isfinite(tolerance) or not 0<=tolerance<=.01:raise ValueError('穿透软惩罚起点范围为 0–10 mm；不作为硬性拒绝门槛')
                    profile['penetration_tolerance_m']=tolerance;write_json(home/'fit-settings.json',{'penetration_tolerance_m':tolerance})
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
                        reference={'pose':'flat_open','hand_side':'left','observation_space':'raw_emf_sensor','positions_local':np.median(positions,axis=0).tolist(),'rotations_local':[r.as_matrix().ravel().tolist() for r in means],'timestamp_ms':samples[-1]['timestamp_ms'],'calibration_fingerprint':samples[-1].get('calibration_fingerprint'),'source':'user_confirmed_stable_raw_frames','reference_id':uuid.uuid4().hex}
                    if reference.get('calibration_fingerprint')!=(model.last_frame or {}).get('calibration_fingerprint'):raise ValueError('平摊参考与当前官方标定不匹配')
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
                emit({'type':'state','capture':capture.snapshot(),'fit':fit_status,'material':model.material,'preview':preview,'calibrated':bool(profile.get('calibrated')),'display_options':{'compression':False,'physics':False,'contact_points':True}})
            except Exception as exc:emit({'type':'ack','id':control['id'],'error':str(exc)})
        except queue.Empty:pass
        try:frame=frames.get(timeout=.01)
        except queue.Empty:continue
        if not aligned_frame(frame):
            emit({'type':'state','status':'waiting_input','error':'定位／触觉时间差超过 25 ms 或输入无效'});continue
        if not frame.get('skeleton_targets'):
            emit({'type':'state','status':'waiting_input','error':'等待官方腕部坐标指尖目标，避免使用单帧刚性猜测'});continue
        if last_input is not None and frame['timestamp_ms']<=last_input:continue
        last_input=frame['timestamp_ms'];reference_samples.append(frame)
        try:
            if pending_reference is not None:
                saved_reference=pending_reference;pending_reference=None
                if saved_reference.get('weights_hash')!=model.weights_hash or saved_reference.get('serial')!=args.serial or saved_reference.get('calibration_fingerprint')!=frame.get('calibration_fingerprint'):raise ValueError('保存的 EMF 平摊参考不兼容，请重新校正平摊参考')
                if not (model.mounting_calibrated and model.flat_reference and model.flat_reference.get('reference_id')==saved_reference.get('reference_id')):
                    model.observations(frame);model.last_frame=frame;model.install_flat_reference(saved_reference)
            if (profile.get('calibrated') or preview) and (candidate if preview else profile).get('calibration_fingerprint')!=frame.get('calibration_fingerprint'):raise ValueError('官方骨架标定与手型版本不一致，请恢复兼容记录或重新拟合')
            mapping=frame.get('mapping')
            if mapping and (not model.mapping or model.mapping['version']!=mapping['version']):
                if fitting and fitting.is_alive() and model.mapping and mapping['assignments']!=model.mapping['assignments']:cancel_fit.set()
                model.build_binding(mapping)
            if not model.mapping:continue
            if pending_auto_apply:
                pending_auto_apply=False;model.last_frame=frame;apply_candidate(automatic=True)
            if capture.status=='collecting' and frame['timestamp_ms']-last_capture>=45:
                if capture.feed(frame):
                    checkpoint()
                    if capture.status=='complete':start_fit()
                last_capture=frame['timestamp_ms']
            if time.monotonic()-last_output<1/60:continue
            began=time.perf_counter();v,j,rotations,ordered,rmse=model.track(frame,steps=1)
            guide=capture.snapshot()['stage_id'] if capture.status=='collecting' else None
            signature=(model.shape_version,model.binding_version,frame['contact'].get('ready'),tuple(p.get('contact') for p in frame['contact'].get('patches',[])),guide)
            if cached_fusion is None or signature!=collision_signature or time.monotonic()-collision_at>=.04:
                if model.geometry is None:model.geometry=CollisionSurface(model)
                cached_geom=model.geometry.query(v,j,rotations)
                cached_fusion=fuse_contacts(model,frame,cached_geom,guide)
                cached_fusion['timestamp_ms']=frame['skeleton']['timestamp_ms'];collision_signature=signature;collision_at=time.monotonic()
            geom=cached_geom;fusion=cached_fusion
            deformed,deformation=deform_surface(model,v,fusion,frame['timestamp_ms'],enabled=False)
            current_version=(model.shape_version,model.binding_version)
            if topology_version!=current_version:
                emit({'type':'topology','model':model.topology()});topology_version=current_version;dimensions=model.dimension_summary()
            elapsed=(time.perf_counter()-began)*1000;processing.append(elapsed);processing=processing[-100:];frame_count+=1;last_output=began
            emit({'type':'frame','status':'preview' if preview else 'calibrated' if profile.get('calibrated') else 'default',
                  'error':None,'hand_side':'left','coordinate_frame':'mano_wrist_local','surface':'gloved_effective','units':'m',
                  'skeleton_timestamp_ms':frame['skeleton']['timestamp_ms'],'vertices_m':np.round(deformed,7).tolist(),'joints_m':np.round(ordered,7).tolist(),'pose_axis_angle':model.last_pose.tolist(),'joint_base_offsets_m':model.last_base_offsets.tolist() if model.last_base_offsets is not None else None,'joint_centres_m':model.last_joint_centres.tolist() if model.last_joint_centres is not None else None,'tracking_source':'official_qnbot_skeleton_fk_v1','official_wrist_to_mano_rotation':model.official_fk.sdk_to_mano.tolist(),
                  'root_rotation':model.last_root.tolist(),'tip_positions_m':deformed[[744,320,443,554,671]].tolist(),
                  'input_targets':{'timestamp_ms':frame['timestamp_ms'],'hand_side':'left','coordinate_frame':'mano_wrist_local','positions_local':(np.array([frame['fingers'][f]['position'] for f in ('thumb','index','middle','ring','pinky')])@model.raw_rotation.T+model.raw_translation).tolist()} if model.flat_reference is not None else None,
                  'input_sensors':frame['fingers'],'sensor_positions_m':model.sensor_predictions.tolist(),'sensor_mounting':{'status':'multi_pose_calibrated' if model.mounting_calibrated else 'multi_pose_candidate' if model.mounting_estimated else 'position_prior' if model.flat_reference else 'unregistered','distal_to_emf_position_m':model.sensor_offsets().tolist(),'distal_to_emf_rotation':model.mount_rot.tolist(),'flat_reference_id':model.flat_reference.get('reference_id') if model.flat_reference else None,'emf_to_mano_rotation':model.raw_rotation.tolist(),'emf_to_mano_translation_m':model.raw_translation.tolist(),'surface_tip_is_sensor':False},
                  'timestamp_ms':frame['skeleton']['timestamp_ms'],'input_timestamp_ms':frame['timestamp_ms'],'tactile_timestamp_ms':frame['tactile_timestamp_ms'],
                  'geometry_version':model.shape_version,'geometry_revision':GEOMETRY_REVISION,'dimension_mode':DIMENSION_MODE,'dimension_parameters':model.dimension_parameters.tolist(),'proximal_lateral_scales':model.proximal_lateral_scales.tolist(),'pose_correctives':False,'penetration_tolerance_m':profile.get('penetration_tolerance_m',PENETRATION_TOLERANCE_M),'binding_version':model.binding_version,'calibrated':bool(profile.get('calibrated')),
                  'dimensions_mm':dimensions,'penetration_policy':'soft_penalty_only','applied_revision':profile.get('revision'),'fit_revision':FIT_REVISION,'calibration_warnings':profile.get('report',{}).get('quality_warnings',[])+(['当前几何沿用旧标定；旧独立 IK 的验收数字不代表官方 FK 驱动质量'] if profile.get('calibrated') and profile.get('report',{}).get('fit_revision')!=FIT_REVISION else []),'calibration_matches_tracking':profile.get('report',{}).get('fit_revision')==FIT_REVISION,'calibration_pair_gaps_mm':profile.get('report',{}).get('together_pair_gaps_mm'),
                  'surface_refinement':profile.get('report',{}).get('surface_refinement'),
                  'display_options':{'compression':False,'physics':False,'contact_points':True},
                  'contact_binary':[p['value'] for p in fusion['patches']],'contact_points':confirmed_contact_points(fusion),'simulation':{'dynamics':False,'contact_forces':False,'collision_checking':True},
                  'calibration_process':{'capture_complete':capture.status=='complete','fit_accepted':bool(fit_status.get('accepted')),'dimensions_applied':bool(profile.get('calibrated'))},'preview':preview,'capture':capture.snapshot(),'fit':fit_status,'material':model.material,'fusion':fusion,'deformation':deformation,
                  'quality':{'position_rmse_mm':rmse,'processing_ms':elapsed,'mean_processing_ms':float(np.mean(processing)),
                    'skeleton_delta_ms':abs(frame['timestamp_ms']-frame['skeleton']['timestamp_ms']) if frame.get('skeleton') else None,
                    'tactile_delta_ms':abs(frame['timestamp_ms']-frame['tactile_timestamp_ms']),
                    'tracking':'official_qnbot_skeleton_fk_v1','joint_limits':True,'joint_limits_source':'official_ik','additional_joint_projection':False,'contact_affects_joints':False,'contact_force_feedback':False,'sensor_contact_pose_constraint':False,'thumb':model.thumb_quality,'alignment':model.alignment_quality,
                    'collision_age_ms':frame['skeleton']['timestamp_ms']-fusion['timestamp_ms'],'source_age_ms':time.time()*1000-frame['timestamp_ms'],'dispatch_age_ms':(frame.get('dispatch_timestamp_ms',frame['timestamp_ms'])-frame['timestamp_ms']),'native_contacts':geom['native_count'],'collision_overflow':geom['overflow'],
                             'frame_count':frame_count,'force_calibrated':False,'unobserved_patches':[p['id'] for p in frame['contact'].get('patches',[]) if p['id'] not in model.mapping.get('confirmed',[])]}})
        except Exception as exc:
            traceback.print_exc();emit({'type':'state','status':'waiting_input','error':str(exc),'capture':capture.snapshot(),'fit':fit_status});time.sleep(.05)
    checkpoint();cancel_fit.set()

if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise
