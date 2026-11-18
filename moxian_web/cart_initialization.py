"""Cached original wuji-retargeting IK for reference-state initialization only."""
import fcntl
import json
import time
from pathlib import Path
import numpy as np
import torch
from .cart_data import file_hash,manifest
from retarget.baseline.geort.adapter import RobotAdapter,digest
from retarget.baseline.geort.pose_domain import extension_domain
from retarget.baseline.geort.training import atomic_json
from retarget.baseline.wuji_retargeting.reference import REFERENCE_ROOT
from retarget_reference import RetargetSession

IK_VERSION='cart_reference_state_ik_v1'
DEFAULT_INITIAL_NOISE_RAD=.1


def identity(folder,robot,session):
    return digest(dict(version=IK_VERSION,inputs={n:file_hash(folder/n) for n in
        ('cart_targets.jsonl','cart_preprocess.json','topology.json','profile.json')},
        implementation=file_hash(__file__),upstream={str(p.relative_to(REFERENCE_ROOT)):file_hash(p)
            for p in sorted(REFERENCE_ROOT.rglob('*.py'))},urdf=file_hash(session.kinematics.urdf_path),
        robot=robot.spec,domain=extension_domain(robot.lower,robot.upper),
        input='canonical_right_mediapipe21',output='filtered_upstream_angles_projected_to_cart_domain'))


def _save_npz(path,**values):
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as stream:np.savez(stream,**values)
    temp.replace(path)


def prepare_ik(root,key,check=lambda:None,progress=None):
    folder,meta=manifest(root,key)
    with (folder/'cart_ik.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('该会话已有离线 IK 初始化任务')
        robot=RobotAdapter();session=RetargetSession.for_hand('WujiHand2','right')
        if session.kinematics.joint_names!=robot.joint_names:raise ValueError('离线 IK 与 CART 关节顺序不一致')
        fingerprint=identity(folder,robot,session);info_path=folder/'cart_ik.json';path=folder/'cart_ik.npz'
        if info_path.exists() and path.exists():
            info=json.loads(info_path.read_text())
            if info.get('fingerprint')==fingerprint and info.get('sha256')==file_hash(path):return info
        n=meta['frames'];domain=extension_domain(robot.lower,robot.upper)
        low,high=np.asarray(domain['lower']),np.asarray(domain['upper'])
        q=np.full((n,20),np.nan,np.float32);raw=q.copy();valid=np.zeros(n,bool)
        stamps=np.full(n,np.nan,np.float64);fallback=np.zeros(n,bool);limited=np.zeros(n,bool)
        pending=folder/'cart_ik.pending.npz';cursor=0;previous_stamp=None;previous_epoch=None;started=time.perf_counter()
        if pending.exists():
            with np.load(pending,allow_pickle=False) as saved:
                if str(saved['fingerprint'])==fingerprint:
                    cursor=int(saved['next_frame'])
                    if not 0<=cursor<=n:raise ValueError('离线 IK 恢复位置无效')
                    for name,array in [('q',q),('raw',raw),('valid',valid),('stamps',stamps),('fallback',fallback),('limited',limited)]:
                        if saved[name].shape!=array.shape:raise ValueError('离线 IK 恢复数据形状变化')
                        array[:cursor]=saved[name][:cursor]
                    if saved['warm'].size:session._warm=saved['warm'].copy()
                    if saved['filtered'].size:session._filtered=saved['filtered'].copy()
        def save_pending():
            _save_npz(pending,q=q,raw=raw,valid=valid,stamps=stamps,fallback=fallback,limited=limited,
                fingerprint=fingerprint,next_frame=cursor,
                warm=session._warm if session._warm is not None else np.empty(0),
                filtered=session._filtered if session._filtered is not None else np.empty(0,np.float32))
        try:
            with (folder/'cart_targets.jsonl').open() as stream:
                count=0
                for i,line in enumerate(stream):
                    check();row=json.loads(line);stamp=float(row['timestamp_ms']);epoch=row.get('stream_epoch')
                    if i>=n:raise ValueError('离线 IK 与录制帧数不一致')
                    if i<cursor:
                        if stamps[i]!=stamp:raise ValueError('离线 IK 恢复时间轴变化')
                        previous_stamp,previous_epoch=stamp,epoch;count=i+1;continue
                    if previous_stamp is None or not 0<stamp-previous_stamp<=250 or epoch!=previous_epoch:session.reset()
                    points=np.asarray(row['joints_m'],np.float64)
                    if points.shape!=(21,3) or not np.isfinite(points).all():raise ValueError('离线 IK 骨架无效')
                    try:
                        raw[i]=session.step(points)
                        valid[i]=np.isfinite(raw[i]).all()
                        q[i]=np.clip(raw[i],low,high)
                        limited[i]=bool(np.any(np.abs(q[i]-raw[i])>1.e-6))
                        fallback[i]=any(s['fallback'] for s in session.last_trace['stages'])
                    except (ValueError,RuntimeError):
                        # A bad palm/frame cannot seed the next window or retain
                        # a stale solver state across a failed observation.
                        session.reset();valid[i]=False
                    stamps[i]=stamp;previous_stamp,previous_epoch=stamp,epoch;cursor=i+1;count=cursor
                    if cursor%100==0 or cursor==n:
                        save_pending()
                        status=dict(status='running',phase='offline_ik',dataset=key,completed=cursor,total=n,
                            valid_frames=int(valid[:cursor].sum()),elapsed_s=time.perf_counter()-started)
                        atomic_json(folder/'cart_ik_status.json',status)
                        if progress:progress('offline_ik',**{k:v for k,v in status.items() if k not in ('status','phase','elapsed_s')})
                if count!=n:raise ValueError('离线 IK 与录制帧数不一致')
            if not valid.any():raise ValueError('离线 IK 没有有效参考状态')
            _save_npz(path,q=q,raw=raw,valid=valid,timestamps_ms=stamps,fallback=fallback,domain_limited=limited)
            info=dict(version=IK_VERSION,fingerprint=fingerprint,sha256=file_hash(path),frames=n,valid_frames=int(valid.sum()),
                fallback_frames=int(fallback.sum()),domain_limited_frames=int(limited.sum()),joint_names=robot.joint_names,
                source_side=meta['source']['hand_side'],solver_side='right',coordinate_frame='canonical_source',
                initialization_only=True,trajectory_supervision=False,tube_required=False)
            atomic_json(info_path,info);pending.unlink(missing_ok=True)
            atomic_json(folder/'cart_ik_status.json',dict(status='completed',dataset=key,completed=n,total=n))
            return info
        except BaseException as exc:
            save_pending();atomic_json(folder/'cart_ik_status.json',dict(status='cancelled' if isinstance(exc,InterruptedError) else 'failed',
                dataset=key,completed=cursor,total=n,error=str(exc)))
            raise


def attach_ik(root,sessions):
    """Join by exact timestamp, leaving the raw/prepared sensor files untouched."""
    reports=[]
    for meta,rows,_ in sessions:
        folder=Path(root)/'recordings'/meta['id'];info=json.loads((folder/'cart_ik.json').read_text())
        if info['version']!=IK_VERSION or info['sha256']!=file_hash(folder/'cart_ik.npz'):raise ValueError('离线 IK 缓存无效')
        with np.load(folder/'cart_ik.npz',allow_pickle=False) as data:
            if len(rows)!=len(data['q']) or not np.array_equal(data['timestamps_ms'],[r['timestamp_ms'] for r in rows]):
                raise ValueError('离线 IK 与观测时间轴不一致')
            for i,row in enumerate(rows):
                row['ik_valid']=bool(data['valid'][i])
                if row['ik_valid']:row['ik_q_rad']=data['q'][i].tolist()
        reports.append(dict(id=meta['id'],fingerprint=info['fingerprint'],valid_frames=info['valid_frames']))
    return reports


def perturb_initial_states(batch,hand,noise_rad):
    if isinstance(noise_rad,bool) or not np.isfinite(noise_rad) or not 0<=noise_rad<=.5:
        raise ValueError('IK 初始状态噪声幅度须为 0–0.5 rad')
    if not noise_rad:return batch
    if not all('initial_q' in w for w in batch):raise ValueError('IK 初始状态缺失')
    initial=torch.stack([hand.tensor(w['initial_q']) if not isinstance(w['initial_q'],torch.Tensor) else w['initial_q'] for w in batch])
    initial=(initial+(torch.rand_like(initial)*2-1)*noise_rad).clamp(hand.low,hand.high)
    return [dict(w,initial_q=q) for w,q in zip(batch,initial)]
