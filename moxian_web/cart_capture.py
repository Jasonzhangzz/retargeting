"""Lossless per-frame wrist geometry follower; independent of the preview mailbox."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
from .cart_capture_schema import CAPTURE_SCHEMA
from retarget.baseline.geort.adapter import SourceAdapter,digest
from retarget.baseline.geort.training import atomic_json


def capture_identity(folder):
    from .cart_data import file_hash
    here=Path(__file__).parent
    return digest(dict(schema=CAPTURE_SCHEMA,
        source={name:file_hash(folder/name) for name in ('profile.json','topology.json')},
        implementation={name:file_hash(here/name) for name in ('cart_capture.py','cart_capture_schema.py','cart_data.py',
            'cart_contacts.py','cart_contact_witness.py','cart_observations.py','qnbot_surface.py','hand_anatomy.py','tactile_surface.py')},
        linked_emf=file_hash(folder/'cart_emf.jsonl') if (folder/'cart_emf.jsonl').exists() else None))


def capture_current(folder,meta=None):
    from .cart_data import file_hash
    try:
        folder=Path(folder);meta=meta or json.loads((folder/'manifest.json').read_text())
        info=json.loads((folder/'cart_capture_meta.json').read_text())
        return (meta.get('complete') is True and info['schema']==CAPTURE_SCHEMA and info['frames']==meta['frames']
            and info['identity']==capture_identity(folder) and info['raw_sha256']==file_hash(folder/'frames.jsonl')
            and info['sha256']==file_hash(folder/'cart_capture.jsonl'))
    except (OSError,ValueError,KeyError):return False


def capture_recording(root,key,*,follow=False,device=None,check=lambda:None,progress=None):
    """Append every complete raw row, resume a torn tail, publish only when complete.

    Raw capture never waits for geometry. The worker catches up on disk, including
    after stop, without dropping frames when inference/preview runs more slowly.
    """
    from .geort_service import safe_id
    from .cart_protocol import SUPPORTED_RECORDINGS
    from .cart_data import SourceGeometry,file_hash
    from .source_ik_initialization import validate_recorded_source
    import torch
    folder=Path(root)/'recordings'/safe_id(key)
    manifest_path=folder/'manifest.json';meta=json.loads(manifest_path.read_text())
    if meta.get('protocol') not in SUPPORTED_RECORDINGS:raise ValueError('不是受支持的 CART 录制')
    if not follow and not meta.get('complete'):raise ValueError('录制尚未保存')
    lock=(folder/'cart_capture.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close();raise ValueError('该录制的几何数据正在后台保存，请等待完成')
    count=0;geometry=None;raw=None;out=None;identity=None;started=time.perf_counter()
    status_path=folder/'cart_capture_status.json';pending=folder/'cart_capture.pending.jsonl'
    def status(state_name,**values):
        state=dict(status=state_name,id=key,pid=os.getpid(),schema=CAPTURE_SCHEMA,frames=count,
            raw_frames=meta.get('frames',0) if meta.get('complete') else None,elapsed_s=time.perf_counter()-started,**values)
        state['identity']=identity
        atomic_json(status_path,state)
        if progress:
            # CLI jobs and training own their id/pid/phase/elapsed/status.
            # Never overwrite them or pass duplicate keyword arguments.
            reserved={'id','pid','phase','elapsed_s','status'}
            payload={('capture_'+k if k in reserved else k):v for k,v in state.items()}
            progress('capture_geometry',**payload)
    try:
        if capture_current(folder,meta):
            info=json.loads((folder/'cart_capture_meta.json').read_text());count=info['frames'];status('completed')
            return info
        identity=capture_identity(folder);profile=json.loads((folder/'profile.json').read_text())
        topology=json.loads((folder/'topology.json').read_text())
        validate_recorded_source(profile,meta['source'])
        try:
            previous=json.loads(status_path.read_text())
            if previous.get('identity')!=identity and pending.exists():pending.unlink()
        except (OSError,ValueError):
            if pending.exists():pending.unlink()
        status('running',phase='initializing',identity=identity)
        geometry=SourceGeometry(topology,profile,device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        linked=None
        if (folder/'cart_emf.jsonl').exists():linked=[json.loads(line) for line in (folder/'cart_emf.jsonl').open()]
        while not (folder/'frames.jsonl').exists():
            check();meta=json.loads(manifest_path.read_text())
            if meta.get('complete'):raise ValueError('录制未保存原始帧')
            time.sleep(.05)
        raw=(folder/'frames.jsonl').open()
        penetration_count=0;contact_count=0
        # Resume only complete, timestamp-matched rows of the same implementation.
        if pending.exists():
            offset=0
            with pending.open('rb+') as stream:
                while True:
                    line=stream.readline()
                    if not line:break
                    try:
                        if not line.endswith(b'\n'):raise ValueError('torn geometry row')
                        row=json.loads(line);source=json.loads(raw.readline())
                        if row['timestamp_ms']!=source['timestamp_ms'] or row['schema']!=CAPTURE_SCHEMA:raise ValueError('geometry prefix differs')
                    except (ValueError,KeyError):
                        stream.truncate(offset);raw.seek(0)
                        for _ in range(count):raw.readline()
                        break
                    count+=1;offset=stream.tell()
                    penetration_count+=row['contacts']['self_penetrations']['count']
                    contact_count+=len(row['tactile_signal_sids'])
        resumed_prefix=count;out=pending.open('a');last_update=0.
        while True:
            check();position=raw.tell();line=raw.readline()
            if not line or not line.endswith('\n'):
                raw.seek(position);meta=json.loads(manifest_path.read_text())
                if meta.get('complete'):
                    if line:raise ValueError('原始录制尾帧不完整')
                    if count!=meta['frames']:raise ValueError('原始录制与几何帧数不一致')
                    break
                if not follow:raise ValueError('原始录制尚未完成')
                time.sleep(.05);continue
            observation=json.loads(line);stamp=float(observation['timestamp_ms'])
            if linked is not None:
                if count>=len(linked) or linked[count]['timestamp_ms']!=stamp:raise ValueError('关联 EMF 时间轴不一致')
                observation['emf_observation']=linked[count]['emf_observation']
            emf=observation.get('emf_observation') or {}
            if emf.get('calibration_fingerprint')!=meta['source']['calibration_fingerprint'] or emf.get('skeleton_runtime_revision')!=meta['source']['skeleton_runtime_revision']:
                raise ValueError('录制 EMF 的设备标定或骨架版本不匹配；请补采或关联原始录制')
            row=geometry.capture_frame(observation['qpos'],stamp,observation)
            # The saved q and fixed profile must reproduce the same raw skeleton.
            original=(np.asarray(observation['joints_m'])[geometry.adapter.indices]-row['source_wrist_origin_m'])
            if not np.allclose(row['joints_m'],original,atol=1.e-5,rtol=1.e-4):raise ValueError('录制关节角与同帧 wrist 骨架不一致')
            out.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n');out.flush();count+=1
            penetration_count+=row['contacts']['self_penetrations']['count']
            contact_count+=len(row['tactile_signal_sids'])
            if time.perf_counter()-last_update>.25:
                meta=json.loads(manifest_path.read_text());status('running',phase='geometry',identity=identity)
                last_update=time.perf_counter()
        out.flush();os.fsync(out.fileno());out.close();out=None
        pending.replace(folder/'cart_capture.jsonl')
        info=dict(schema=CAPTURE_SCHEMA,identity=identity,frames=count,coordinate_frame='wrist_local',units='m',
            sha256=file_hash(folder/'cart_capture.jsonl'),raw_sha256=file_hash(folder/'frames.jsonl'),
            all_sampled_penetrations=True,all_taxels=True,all_confirmed_contact_points=True,
            penetration_samples_processed=penetration_count,signal_points_processed=contact_count,
            resumed_prefix_frames=resumed_prefix)
        atomic_json(folder/'cart_capture_meta.json',info);status('completed',identity=identity)
        return info
    except BaseException as exc:
        status('cancelled' if isinstance(exc,InterruptedError) else 'failed',error=str(exc))
        raise
    finally:
        if raw:raw.close()
        if out:out.close()
        lock.close()


def follow(root,key,device=None):
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    def check():
        if stopped:raise InterruptedError('录制几何保存已暂停，原始帧和已处理前缀保留')
    old={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
    try:return capture_recording(root,key,follow=True,device=device,check=check)
    finally:
        for s,handler in old.items():signal.signal(s,handler)
