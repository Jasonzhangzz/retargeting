"""Immutable CART recordings, wrist geometry snapshots and tube-free training data."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .cart_protocol import PROTOCOL, LEGACY_PROTOCOL, SUPPORTED_RECORDINGS
from retarget.baseline.geort.adapter import SourceAdapter, digest
from retarget.baseline.geort.training import atomic_json


def file_hash(path):
    # Complete wrist journals can be several GB; hashing must stay bounded.
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def contact_fingerprint(folder):
    """Contact extraction cache is independent of the subsequent CEM version."""
    import inspect
    folder=Path(folder);root=Path(__file__).parent
    return digest(dict(source={name:file_hash(folder/name) if (folder/name).exists() else None for name in
        ('motion.npz','timestamps.npy','profile.json','topology.json','frames.jsonl','points.npy')},
        adapter=inspect.getsource(SourceGeometry),implementation={name:file_hash(root/name) for name in
        ('cart_contacts.py','cart_contact_witness.py','qnbot_surface.py','hand_anatomy.py','wuji_contact_source.py','tactile_surface.py')},
        source_adapter=file_hash(root.parent/'retarget/baseline/geort/adapter.py')))


def implementation_identity():
    from .cart_contacts import CONTACT_POLICY
    root=Path(__file__).parent
    names=('cart_contacts.py','cart_contact_witness.py','cart_objective.py','cart_data.py','cart_network.py',
           'cart_observations.py','cart_collision.py','cart_training.py','cart_capture.py','cart_capture_schema.py')
    deps=('retarget/env/grasp/pose_gpu.py','retarget/baseline/card/collision.py',
          'retarget/baseline/geort/adapter.py','retarget/baseline/geort/pose_domain.py',
          'moxian_web/wuji_surface.py','moxian_web/tactile_surface.py','moxian_web/qnbot_surface.py',
          'moxian_web/hand_anatomy.py','moxian_web/wuji_contact_source.py','moxian_web/cart_losses.py',
          'retarget/baseline/wuji_retargeting/upstream/retarget_reference/preprocess.py',
          'retarget/baseline/wuji_retargeting/upstream/retarget_reference/objectives.py')
    return dict(protocol=PROTOCOL,contact=CONTACT_POLICY,
        files={name:file_hash(root/name) for name in names},
        dependencies={name:file_hash(root.parent/name) for name in deps})


class SourceGeometry:
    def __init__(self,topology,profile,device='cpu'):
        import torch
        from .qnbot_surface import QnbotSurface
        from .cart_contacts import SourceContacts
        self.torch=torch;self.topology=topology;self.adapter=SourceAdapter(topology)
        self.model=QnbotSurface(topology['hand_side'],device,finger_twist=True)
        self.model.load_artifact(profile);self.model.fixed=True
        self.detector=SourceContacts(topology,self.model,device)

    def reconstruct(self,q):
        q=np.asarray(q,np.float32)
        if len(q)==25 and len(self.model.q0)==29:q=np.r_[q,np.zeros(4,np.float32)]
        with self.torch.inference_mode():v,j,_=self.model(self.torch.tensor(q,device=self.model.device)[None])
        return v[0].cpu().numpy(),j[0].cpu().numpy()

    def canonical(self,vertices,joints):
        j=np.asarray(joints)[self.adapter.indices];root=j[0]
        return ((np.asarray(vertices)-root)@self.adapter.transform,(j-root)@self.adapter.transform)

    def frame(self,q,stamp,observation=None,raw=None):
        import time
        from .cart_contacts import transform_contacts
        started=time.perf_counter();v,j=self.reconstruct(q) if raw is None else raw
        _,cj=self.canonical(v,j)
        observation=dict(observation or {},timestamp_ms=float(stamp))
        observed=self.detector.detect(v,j,observation)
        contacts=transform_contacts(observed,self.adapter.transform,j[self.adapter.indices[0]])
        row=dict(timestamp_ms=float(stamp),joints_m=cj.tolist(),points=cj[[4,8,12,16,20]].tolist(),
                 contacts=contacts,processing_ms=(time.perf_counter()-started)*1000)
        from .cart_observations import canonical_emf,tactile_input
        taxels,_=self.detector.taxels.evaluate(v)
        tp,ts,tv=tactile_input(observation,(taxels-j[self.adapter.indices[0]])@self.adapter.transform,self.topology)
        row.update(tactile_positions_m=tp.tolist(),tactile_state=ts.tolist(),tactile_valid=tv.tolist())
        if observation.get('emf_observation'):
            emf,valid=canonical_emf(observation['emf_observation'],self.adapter,j[self.adapter.indices[0]],float(stamp))
            row.update(emf_pose=emf.tolist(),emf_valid=valid.tolist())
        return row,v,j


    def capture_frame(self,q,stamp,observation,raw=None):
        """Persist physical, unmirrored wrist coordinates before policy adaptation."""
        import copy
        import time
        from .cart_contacts import transform_contacts
        from .cart_capture_schema import CAPTURE_SCHEMA,binary_taxels
        from .cart_observations import canonical_emf
        started=time.perf_counter();v,j=self.reconstruct(q) if raw is None else raw
        root=np.asarray(j)[self.adapter.indices[0]]
        observed=self.detector.detect(v,j,dict(observation,timestamp_ms=float(stamp)),include_all=True)
        points,normals=self.detector.taxels.evaluate(v)
        state,valid,bound=binary_taxels(observation,self.topology)
        points=np.asarray(points)-root
        active=np.flatnonzero((state==1)&valid)
        emf=copy.deepcopy(observation.get('emf_observation'))
        if not emf:raise ValueError('录制缺少原始 EMF 完整位姿，请关联原始录制或重新采集')
        # Validate the measured pose before serializing an explicitly wrist-rooted copy.
        canonical_emf(emf,self.adapter,root,float(stamp))
        emf['positions_m']=(np.asarray(emf['positions_m'])-root).tolist()
        return dict(schema=CAPTURE_SCHEMA,coordinate_frame='wrist_local',units='m',hand_side=self.topology['hand_side'],
            timestamp_ms=float(stamp),tactile_timestamp_ms=observation.get('tactile_timestamp_ms'),
            stream_epoch=observation.get('input_stream_epoch'),emf_observation=emf,
            source_wrist_origin_m=root.tolist(),joint_order='wrist_then_thumb_index_middle_ring_pinky_four_each',
            joints_m=(np.asarray(j)[self.adapter.indices]-root).tolist(),
            tactile_state=state.tolist(),tactile_valid=valid.tolist(),tactile_position_valid=bound.tolist(),
            tactile_positions_m=points.tolist(),tactile_normals=np.asarray(normals).tolist(),
            tactile_signal_sids=active.tolist(),tactile_signal_positions_m=points[active].tolist(),
            tactile_signal_normals=np.asarray(normals)[active].tolist(),
            contacts=transform_contacts(observed,np.eye(3),root),
            processing_ms=(time.perf_counter()-started)*1000)

    @staticmethod
    def row_from_capture(capture,adapter):
        from .cart_contacts import transform_contacts
        from .cart_capture_schema import CAPTURE_SCHEMA
        from .cart_observations import canonical_emf
        if capture.get('schema')!=CAPTURE_SCHEMA or capture.get('coordinate_frame')!='wrist_local' or capture.get('units')!='m':
            raise ValueError('录制的几何坐标协议不兼容')
        stamp=capture['timestamp_ms'];joints=np.asarray(capture['joints_m'])@adapter.transform
        # The complete point set stays in the streamed capture journal. Policy
        # training uses the existing link-pair/contact summaries, so thousands
        # of unused source witnesses per frame never accumulate in RAM.
        observed={k:v for k,v in capture['contacts'].items() if k!='self_penetrations'}
        emf,valid=canonical_emf(capture['emf_observation'],adapter,np.zeros(3),stamp)
        return dict(timestamp_ms=stamp,stream_epoch=capture.get('stream_epoch'),joints_m=joints.tolist(),
            points=joints[[4,8,12,16,20]].tolist(),emf_pose=emf.tolist(),emf_valid=valid.tolist(),
            tactile_positions_m=(np.asarray(capture['tactile_positions_m'])@adapter.transform).tolist(),
            tactile_state=capture['tactile_state'],tactile_valid=capture['tactile_valid'],
            contacts=transform_contacts(observed,adapter.transform,np.zeros(3)),
            processing_ms=capture['processing_ms'])


def manifest(root,key):
    from .geort_service import safe_id
    folder=Path(root)/'recordings'/safe_id(key);m=json.loads((folder/'manifest.json').read_text())
    if m.get('protocol') not in SUPPORTED_RECORDINGS:raise ValueError('不是受支持的 CART 录制协议')
    if not m.get('complete') or m.get('frames',0)<2:raise ValueError('录制未保存或有效帧不足')
    from .source_ik_initialization import validate_recorded_source
    validate_recorded_source(json.loads((folder/'profile.json').read_text()),m['source'])
    if m.get('split') not in ('train','validation'):raise ValueError('CART 录制缺少训练／验证分区')
    return folder,m


def input_digest(folder,m,full):
    files=('motion.npz','timestamps.npy','profile.json','topology.json','points.npy','frames.jsonl','cart_emf.jsonl','cart_capture.jsonl','cart_capture_meta.json')
    return digest(dict(protocol=PROTOCOL,source=m['source'],full=full,
        contact=contact_fingerprint(folder),observation=file_hash(Path(__file__).with_name('cart_observations.py')),
        adapter=file_hash(__file__),files={n:file_hash(folder/n) if (folder/n).exists() else None for n in files}))


def preprocessing_current(folder,m,full=True):
    try:
        info=json.loads((folder/'cart_preprocess.json').read_text())
        return (info['protocol']==PROTOCOL and info['profile']==('full' if full else 'smoke_test')
            and info['fingerprint']==input_digest(folder,m,full)
            and info['targets_sha256']==file_hash(folder/'cart_targets.jsonl'))
    except (OSError,ValueError,KeyError):return False


def _preprocess(root,key,full=True,progress=None,check=lambda:None):
    import torch
    folder,m=manifest(root,key)
    if file_hash(folder/'points.npy')!=m['points_sha256']:raise ValueError('原始录制指纹不一致')
    stamps=np.load(folder/'timestamps.npy',allow_pickle=False)
    motion=np.load(folder/'motion.npz',allow_pickle=False)
    if len(stamps)!=m['frames'] or len(motion['qpos'])!=len(stamps):raise ValueError('CART 原始帧数不一致')
    if not (folder/'frames.jsonl').exists():raise ValueError('缺少同步 EMF／触觉原始记录，请重新采集')
    observations=[json.loads(line) for line in (folder/'frames.jsonl').open()]
    if len(observations)!=len(stamps):raise ValueError('CART 原始传感器帧数不一致')
    if (folder/'cart_emf.jsonl').exists():
        linked=[json.loads(line) for line in (folder/'cart_emf.jsonl').open()]
        if len(linked)!=len(stamps):raise ValueError('关联 EMF 帧数不一致')
        for obs,item,stamp in zip(observations,linked,stamps):
            if item['timestamp_ms']!=stamp:raise ValueError('关联 EMF 时间轴不一致')
            obs['emf_observation']=item['emf_observation']
    if any(not obs.get('emf_observation') for obs in observations):
        raise ValueError('录制缺少原始 EMF 完整位姿，请用 import-emf 关联同会话原始录制或重新采集')
    from .cart_capture import capture_recording
    capture_recording(root,key,check=check,progress=progress)
    topology=json.loads((folder/'topology.json').read_text());adapter=SourceAdapter(topology)
    temp=folder/'cart_targets.tmp';rows=[]
    try:
        with temp.open('w') as stream,(folder/'cart_capture.jsonl').open() as captures:
            for i,line in enumerate(captures):
                check();capture=json.loads(line)
                if i>=len(stamps) or capture['timestamp_ms']!=stamps[i]:raise ValueError('wrist 几何与原始录制时间轴不一致')
                row=SourceGeometry.row_from_capture(capture,adapter);rows.append(row)
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                if progress and (i%10==0 or i+1==len(stamps)):progress('observations',dataset=key,completed=i+1,total=len(stamps))
            if len(rows)!=len(stamps):raise ValueError('wrist 几何录制缺帧')
        temp.replace(folder/'cart_targets.jsonl')
    finally:
        if temp.exists():temp.unlink()
    info=dict(protocol=PROTOCOL,fingerprint=input_digest(folder,m,full),source=m['source'],frames=len(rows),
        profile='full' if full else 'smoke_test',targets_sha256=file_hash(folder/'cart_targets.jsonl'),
        tactile_frames=sum(r['contacts']['tactile_ready'] for r in rows),
        contact_frames=sum(bool(r['contacts']['self_pairs'] or r['contacts']['object_contacts']) for r in rows),
        emf_frames=sum(any(r['emf_valid']) for r in rows),tube_required=False,capture_schema='cart_capture_wrist_v1')
    atomic_json(folder/'cart_preprocess.json',info)
    return info


def preprocess(root,key,full=True,progress=None,check=lambda:None):
    import fcntl
    from .geort_service import safe_id
    folder=Path(root)/'recordings'/safe_id(key)
    with (folder/'cart_preprocess.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('该会话已有预处理任务')
        return _preprocess(root,key,full,progress,check)


def load_sessions(root,ids,split,require_complete=True):
    if not ids or len(set(ids))!=len(ids):raise ValueError('请选择不重复的 CART 会话')
    result=[];identity=None;binding=None
    for key in ids:
        folder,m=manifest(root,key)
        if m['split']!=split:raise ValueError('训练和验证必须使用独立会话')
        if identity is not None and identity!=m['source']:raise ValueError('设备、标定或坐标不兼容')
        identity=m['source'];info=json.loads((folder/'cart_preprocess.json').read_text())
        current_binding=json.loads((folder/'topology.json').read_text())['binding_version']
        if binding is not None and binding!=current_binding:raise ValueError('CART 会话触觉材料绑定不兼容')
        binding=current_binding
        if not preprocessing_current(folder,m,info['profile']=='full'):raise ValueError('CART 观测预处理已过期，请重建')
        if require_complete and info['profile']!='full':raise ValueError('测试预处理不能用于正式训练')
        rows=[json.loads(line) for line in (folder/'cart_targets.jsonl').open()]
        if len(rows)!=m['frames']:raise ValueError('CART 预处理帧数不一致')
        result.append((m,rows,info))
    return result,identity


def sequence_indices(rows,length=16,history=10):
    """Warmup + current state + loss steps + true future tail. No robot labels."""
    segments=[];start=0
    for end in range(1,len(rows)+1):
        if end==len(rows) or not 0<rows[end]['timestamp_ms']-rows[end-1]['timestamp_ms']<=250 or rows[end].get('stream_epoch')!=rows[end-1].get('stream_epoch'):
            for i in range(start,end-length-9,max(1,length//2)):
                segments.append(list(range(max(start,i-history),i+length+10)))
            start=end
    return segments


def import_emf(root,key,raw_path):
    """Explicit, identity-checked association. Raw captures are never rewritten."""
    from .cart_observations import capture_emf
    folder,m=manifest(root,key);raw_path=Path(raw_path).resolve()
    source=m['source'];stamps=np.load(folder/'timestamps.npy',allow_pickle=False)
    # Device identity may be encoded by the existing device-specific journal path.
    device_hash=hashlib.sha256(source['serial'].encode()).hexdigest()[:16]
    profile=json.loads((folder/'profile.json').read_text());topology=json.loads((folder/'topology.json').read_text())
    geometry=SourceGeometry(topology,profile,'cpu');raw=[]
    with raw_path.open() as stream:
        for line in stream:
            row=json.loads(line)
            if row.get('serial')!=source['serial'] and device_hash not in raw_path.parts:
                raise ValueError('原始 EMF 缺少匹配的设备身份')
            if row.get('hand_side')!=source['hand_side'] or row.get('calibration_fingerprint')!=source['calibration_fingerprint'] or row.get('skeleton_runtime_revision')!=source['skeleton_runtime_revision']:
                raise ValueError('原始 EMF 手侧、标定或源骨架版本不匹配')
            raw.append(row)
    times=np.asarray([r['timestamp_ms'] for r in raw])
    if not len(times) or not np.isfinite(times).all() or (np.diff(times)<=0).any():raise ValueError('原始 EMF 时间轴无效')
    chosen=[];last=-1
    for stamp in stamps:
        right=int(np.searchsorted(times,stamp));candidates=[i for i in (right-1,right) if 0<=i<len(times)]
        idx=min(candidates,key=lambda i:abs(times[i]-stamp))
        if idx<=last or abs(times[idx]-stamp)>25:raise ValueError('原始 EMF 无法逐帧唯一同步；请补采')
        last=idx;chosen.append(dict(timestamp_ms=float(stamp),emf_observation=capture_emf(raw[idx],geometry.model)))
    target=folder/'cart_emf.jsonl'
    if target.exists():raise ValueError('已有关联 EMF，拒绝覆盖')
    temporary=folder/'cart_emf.tmp'
    try:
        with temporary.open('x') as stream:
            for item in chosen:stream.write(json.dumps(item,allow_nan=False)+'\n')
        temporary.replace(target)
    finally:
        if temporary.exists():temporary.unlink()
    atomic_json(folder/'cart_emf_source.json',dict(raw_file=str(raw_path),sha256=file_hash(raw_path),frames=len(chosen),source=source))
    return dict(frames=len(chosen),raw_file=str(raw_path))
