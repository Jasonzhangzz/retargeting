"""Default six-gesture calibration: actual Wuji native objective and solver."""
from __future__ import annotations
import hashlib,json,queue,re,subprocess,sys,tempfile,threading,time
from pathlib import Path
from .wuji_native_model import METHOD,REVISION,build_candidate
from .wuji_native_capture import PROTOCOL
from .wuji_native_abi import SDK_POSES,FULL_FIT_RVA
from .wuji_calibration_solver import ELF_SHA256


def fit_native_capture(profile,mapping,document,progress,cancel):
    if document.get('protocol')!=PROTOCOL or document.get('completed_groups')!=list(SDK_POSES) or document.get('state',{}).get('state')!='complete':raise ValueError('请重新完成Wuji原生六动作采集；历史17动作只读保留')
    if profile.get('side')!=document.get('side') or profile.get('calibration_fingerprint')!=document.get('calibration_fingerprint'):raise ValueError('原生采集来源不匹配')
    if document.get('mapping',{}).get('assignments')!=mapping.get('assignments'):raise ValueError('原生采集SID分区已变化')
    if any(not document.get('pose_frames',{}).get(name) for name in SDK_POSES):raise ValueError('原生六动作观测不完整')
    root=Path(__file__).resolve().parents[1];began=time.monotonic();logs=[];native_results={};current=None;proc=None
    with tempfile.TemporaryDirectory(prefix='moxian-wuji-calibration-') as tmp:
        source=Path(tmp)/'poses.json';destination=Path(tmp)/'result.json'
        source.write_text(json.dumps(dict(side=profile['side'],pose_frames={k:document['pose_frames'][k] for k in SDK_POSES}),allow_nan=False))
        source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
        try:
            proc=subprocess.Popen([sys.executable,'-u',str(root/'moxian_web/wuji_native_fit_child.py'),str(source),str(destination)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            progress(dict(state='running',phase='native_fit',progress=None,message='调用Wuji完整原生calibration：六动作、21自由度、原生objective与L-BFGS',native_solver=True,native_objective=True))
            lines=queue.Queue()
            def read_logs():
                try:
                    for line in proc.stdout:lines.put(line)
                finally:lines.put(None)
            reader=threading.Thread(target=read_logs,daemon=True);reader.start()
            while True:
                if cancel.is_set():raise InterruptedError('原生标定已取消')
                try:line=lines.get(timeout=.1)
                except queue.Empty:continue
                if line is None:break
                if line:
                    line=line.rstrip();logs.append(line)
                    if line.startswith('{'):
                        try:
                            event=json.loads(line)
                            if event.get('event')=='profile':current=event['profile']
                        except ValueError:pass
                    event=re.search(r'\[optimize\] eval #(\d+), cost = ([0-9.eE+-]+)',line)
                    done=re.search(r'\[optimize\] done: (\d+) evals in ([0-9.]+)s, final cost = ([0-9.eE+-]+)',line)
                    if done:
                        native_results[current]=dict(evaluations=int(done[1]),seconds=float(done[2]),final_cost=float(done[3]))
                    if event or done:
                        count=int(event[1] if event else done[1]);cost=float(event[2] if event else done[3])
                        progress(dict(state='running',phase='native_fit',progress=None,completed=count,total=None,native_profile=current,objective_value=cost,
                            message=f'Wuji原生 {current} · 评估{count}次 · 目标值{cost:.6g}',training_sample=dict(sample_id=f'{current}:{count}',step=count,phase=current,scope=current,source='train_only',raw=dict(loss=cost,optimizer_name='Wuji native calibration'))))
            if proc.wait()!=0 or not destination.exists():raise RuntimeError('Wuji原生标定失败：'+'\n'.join(logs[-8:]))
            result=json.loads(destination.read_text())
        finally:
            if proc is not None:
                if proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                if 'reader' in locals():reader.join(timeout=2)
                proc.stdout.close()
    if cancel.is_set():raise InterruptedError('原生标定已取消')
    candidate=build_candidate(result['profiles'],profile,mapping,source_hash)
    from .wuji_native_ik import WujiNativeSkeleton
    # Verify that the actual online SDK can load the generated active URDF
    # before this candidate can replace a working model. No synthetic pose is
    # used here, and this does not publish an active pointer.
    online=WujiNativeSkeleton(candidate)
    online.close()
    if cancel.is_set():raise InterruptedError('原生标定已取消')
    from .wuji_native_model import NativeURDF
    native=NativeURDF(result['profiles']['wujihand']['wrist_urdf']);extrinsics=native.extrinsics()
    report=dict(method=METHOD,fit_revision=REVISION,surface_backend='qnbot',state='completed',accepted=True,
        native_calibration_succeeded=True,native_ik_load_verified=True,validation_complete=True,validation_scope='wuji_native_result_and_urdf_integrity',validation_pending=False,
        reason='Wuji原生标定成功',capture_protocol=PROTOCOL,calibration_poses=6,solver_dof=21,
        generated_hand_profiles=['wujihand','wujihand2'],active_hand_profile='wujihand',independent_geometry_parameters=51,
        end_to_end_s=time.monotonic()-began,capture_session_id=profile.get('capture_session_id'),**extrinsics,
        online_optimization=True,delta_policy=False,source_capture_hash=source_hash,
        collection_orientation_frame='moxian_initialized_wrist_bend_axis' if document.get('gesture_frame') else 'raw_emf_tx',
        gesture_frame_source_sha256=(document.get('gesture_frame') or {}).get('source_payload_sha256'),
        solver=dict(optimizer_name='Wuji native calibration / LD_LBFGS',native_solver_reused=True,native_caliber_objective_reused=True,
            sdk_version='2026.7.1',elf_sha256=ELF_SHA256,entry_point=hex(FULL_FIT_RVA),full_dimension=187,optimized_dimension=177,
            ftol_rel=1e-10,maxeval=20000,profiles=native_results,native_result='Ok'),native_log=logs)
    if cancel.is_set():raise InterruptedError('原生标定已取消')
    candidate['report']=report;return candidate,report
