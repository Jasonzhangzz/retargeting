"""Public GPU calibration entrypoint, content-addressed preprocessing and timing."""
import copy
import hashlib
import json
import time
from pathlib import Path

METHODS=('bone_extrinsics','parameters','delta','flashsac','least_squares')


def prepare_rows(profile,rows,progress,cancel,use_cache=True):
    if cancel.is_set():raise InterruptedError('GPU 标定已取消')
    progress(dict(state='running',phase='prepare',progress=.01,completed=0,total=len(rows),message='检查原始观测与骨架版本',estimated_remaining_s=None))
    fingerprint=profile.get('calibration_fingerprint') or rows[0].get('calibration_fingerprint')
    runtime=profile.get('skeleton_runtime_revision') or rows[-1].get('skeleton_runtime_revision')
    needed=[i for i,r in enumerate(rows) if not r.get('skeleton') or not r.get('skeleton_targets') or r.get('calibration_fingerprint')!=fingerprint or r.get('skeleton_runtime_revision')!=runtime]
    if not needed:
        progress(dict(state='running',phase='prepare',progress=.03,completed=len(rows),total=len(rows),message='全部骨架观测已就绪',estimated_remaining_s=0.))
        return rows,dict(cache_hit=False,replayed_frames=0)
    home=Path(profile['artifact_home']);root=Path(__file__).resolve().parents[1]
    key=hashlib.sha256(profile['serial'].encode()).hexdigest()[:16]
    config=home.parents[1]/'ik'/key/'runtime'/f'kinematic-{fingerprint[:16]}.yaml'
    progress(dict(state='running',phase='prepare',progress=.01,message='计算预处理指纹并检查缓存',estimated_remaining_s=None))
    raw=json.dumps(rows,sort_keys=True,separators=(',',':')).encode()
    from .emf_slsqp import REVISION as IK_REVISION
    digest=hashlib.sha256(raw+config.read_bytes()+json.dumps([runtime,fingerprint,profile.get('side'), 'gpu_prepare_v2_sequence_resets',IK_REVISION]).encode()).hexdigest()
    folder=home/'gpu-cache'/digest;folder.mkdir(parents=True,exist_ok=True)
    target=folder/'prepared.json';index=folder/'index.json'
    if use_cache and target.exists() and index.exists():
        try:
            content=target.read_bytes();record=json.loads(index.read_text())
            if record.get('sha256')==hashlib.sha256(content).hexdigest():
                result=json.loads(content)['rows']
                if len(result)==len(rows) and all(r.get('skeleton') and r.get('skeleton_targets') and r.get('calibration_fingerprint')==fingerprint and r.get('skeleton_runtime_revision')==runtime for r in result):return result,dict(cache_hit=True,replayed_frames=0,key=digest)
        except (OSError,ValueError,KeyError,TypeError):pass
    source=folder/'missing-source.json';converted=folder/'missing-targets.json'
    source.write_text(json.dumps({'rows':[rows[i] for i in needed]},allow_nan=False))
    from .fit_process import run_replay
    command=[str(root/'.venv-moxian-ik/bin/python'),'-I',str(root/'moxian_web/prepare_wrist_targets.py'),'--input',str(source),'--output',str(converted),'--config',str(config),'--fingerprint',fingerprint,'--runtime-revision',runtime]
    run_replay(command,cancel,lambda v:progress(dict(state='running',phase='prepare',progress=.01+.02*v['completed']/v['total'],completed=v['completed'],total=v['total'],message='转换缺失或过期骨架',estimated_remaining_s=v['estimated_remaining_s'],eta_scope='stage')))
    result=copy.deepcopy(rows)
    for i,row in zip(needed,json.loads(converted.read_text())['rows']):result[i]=row
    content=json.dumps({'rows':result},ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
    temporary=target.with_suffix('.tmp');temporary.write_bytes(content);temporary.replace(target)
    index.write_text(json.dumps(dict(key=digest,sha256=hashlib.sha256(content).hexdigest())))
    return result,dict(cache_hit=False,replayed_frames=len(needed),key=digest)


def run_backend(method,model_path,profile,mapping,rows,progress,cancel,use_cache=True,**kwargs):
    from .mano_fit import fit_capture as fit
    kwargs.setdefault('max_seconds',45.)
    selected_options=kwargs.pop('training_options',None)
    if selected_options is not None:
        from .training_options import options,validate_optimizer
        from .mano_objective import SolverSettings
        from dataclasses import replace
        selected_options=options(selected_options)
        validate_optimizer(method,selected_options)
        kwargs['max_seconds']=selected_options['max_seconds']
        kwargs['training_batch_size']=selected_options['batch_size']
        kwargs['solver_settings']=replace(kwargs.get('solver_settings') or SolverSettings(),adam_steps=selected_options['adam_steps'],optimizer=selected_options['optimizer'])
    from .gpu_memory import DeviceMemoryMonitor
    began=time.monotonic();data,preparation=prepare_rows(profile,rows,progress,cancel,use_cache)
    prep_s=time.monotonic()-began
    with DeviceMemoryMonitor() as memory:
        result,report=fit(model_path,profile,mapping,data,progress,cancel,gpu_method=method,**kwargs)
    report['peak_device_memory_bytes']=memory.peak
    report['device_memory_scope']='whole_device_including_live_rendering_and_Warp'
    duration=time.monotonic()-began
    report.update(method=method,preparation=preparation,preparation_s=prep_s,end_to_end_s=duration,
                  performance_passed=duration<=60)
    report['time_target_met']=report['performance_passed']
    report['training_options']=selected_options
    if not report['performance_passed']:
        report.update(accepted=False,holdout_validation_passed=False)
        if '端到端性能未达标' not in report['rejection_reasons']:report['rejection_reasons'].append('端到端性能未达标')
        report['reason']='；'.join(report['rejection_reasons'])+'；当前模型保留'
    result['report']=report
    return result,report


def fit_capture(model_path,profile,mapping,rows,progress,cancel,method='bone_extrinsics',**kwargs):
    if method not in METHODS:raise ValueError('未知标定方法')
    legacy=profile.get('runtime_backend') in ('fixed_mano_v1','gpu_mano_v1')
    backend=kwargs.pop('surface_backend',profile.get('surface_backend','mano' if legacy else 'qnbot'))
    if backend=='qnbot':
        from .qnbot_calibration import fit_capture as fit
        return fit(model_path,profile,mapping,rows,progress,cancel,method=method,**kwargs)
    if backend!='mano':raise ValueError('surface_backend 须为 qnbot 或 mano')
    if method in ('least_squares','bone_extrinsics'):raise ValueError('骨长外参与最小二乘分支仅支持Qnbot模型基础')
    if method=='parameters':from .mano_calibrate_parameters import fit_capture as fit
    elif method=='delta':from .mano_calibrate_delta import fit_capture as fit
    else:from .mano_calibrate_flashsac import fit_capture as fit
    return fit(model_path,profile,mapping,rows,progress,cancel,**kwargs)
