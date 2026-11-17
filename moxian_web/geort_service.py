"""GeoRT jobs and lossless small-frame recording, independent of UI polling."""
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import uuid
import numpy as np
from retarget.baseline.geort.adapter import ROOT, SourceAdapter, UPSTREAM, digest


def write_json(path, value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2));tmp.replace(path)


def safe_id(value):
    if not isinstance(value,str) or not value or value.startswith('.') or Path(value).name!=value:
        raise ValueError('无效 GeoRT ID')
    return value


class GeoRTService:
    def __init__(self, root=None, workspace=False):
        self.root=Path(root or ROOT/'output/moxian/geort')
        self.lock=threading.RLock();self.latest=None;self.latest_at=0.;self.capture=None
        self.recording=dict(active=False,saving=False,frames=0,invalid_frames=0,error=None)
        self.proc=None;self.job_id=None;self.job_log=None;self.selected=None
        self.published_run=None;self.on_model_published=None;self.model_update=None
        try:
            active=json.loads((self.root/'active.json').read_text());self.selected=active['model'];self.published_run=active.get('run_id')
        except (OSError,ValueError,KeyError):pass
        self.replay_active=False;self.replay_topology=None
        self.coverage_base=None;self.coverage_total=None;self.coverage_segment=None;self.coverage_ids=[];self.coverage_current_id=None
        self.coverage_notice=None;self.coverage_cache=None;self.coverage_cache_at=0.
        self.workspace_enabled=workspace;self.workspace_source=None;self.workspace_folder=None
        self.workspace_proc=None;self.workspace_reference=None;self.workspace_status=dict(status='waiting_input')
        self.workspace_lock=threading.Lock();self.workspace_spec=None;self.workspace_spec_signature=None
        self.closed=False
        self.monitor=threading.Thread(target=self._monitor,daemon=True);self.monitor.start()

    def observe(self, frame, topology, joints, serial, cart_hull=None):
        # Called on each completed hand-worker frame, never an HTTP snapshot.
        try:
            if not frame.get('ready') or frame.get('preview') or not frame.get('calibrated'):
                with self.lock:
                    if self.capture:self.recording['invalid_frames']+=1
                return
            if not topology or topology.get('geometry_version')!=frame.get('geometry_version'):return
            adapter=SourceAdapter(topology)
            source=adapter.identity(serial,frame)
            points=adapter.points(joints)
            stamp=float(frame['timestamp_ms'])
            if not np.isfinite(stamp) or abs(stamp-float(frame.get('skeleton_timestamp_ms',stamp)))>25:
                raise ValueError('骨架与手部模型时间戳不一致')
            q=np.asarray(frame.get('source_joint_positions_rad'),dtype=np.float32)
            if q.ndim!=1 or len(q) not in (25,29) or not np.isfinite(q).all():
                raise ValueError('缺少同帧 Moxian 关节角，请重启手部模型')
            item=dict(source=source,points=points.tolist(),timestamp_ms=stamp,
                      joints_m=np.asarray(joints).tolist(),qpos=q.tolist())
            if frame.get('cart_label'):item['cart_label']=frame['cart_label']
            with self.lock:
                self.latest=(item,topology);self.latest_at=time.monotonic()
                capture=self.capture
                if not capture:return
                if source!=capture['source']:
                    self._stop_locked('标定、设备或手侧已变化，已停止录制');return
                if stamp==capture['last_stamp']:return
                if stamp<capture['last_stamp']:
                    self._stop_locked('源时间戳回退，已停止录制');return
                capture['last_stamp']=stamp
                if capture['meta'].get('capture_mode')=='cart':
                    if not cart_hull or cart_hull.get('timestamp_ms')!=stamp:
                        self.recording['invalid_frames']+=1;return
                    item['cart_hull']=cart_hull
                try:capture['queue'].put_nowait(item)
                except queue.Full:self._stop_locked('写入队列已满，已停止并保留已排队帧')
        except Exception as exc:
            with self.lock:
                if self.capture:self.recording['invalid_frames']+=1;self.recording['last_invalid_reason']=str(exc)

    def _monitor(self):
        while not self.closed:
            with self.lock:
                if self.capture and time.monotonic()-self.latest_at>1:
                    self._stop_locked('手套有效数据已中断，已停止录制')
            self.refresh_published()
            if self.workspace_enabled:
                try:self.poll_workspace()
                except Exception as exc:
                    self.workspace_status=dict(status='failed',error=str(exc))
            time.sleep(.1)

    def poll_workspace(self,force=False):
        with self.workspace_lock:return self._poll_workspace(force)

    def _poll_workspace(self,force=False):
        from retarget.baseline.geort.workspace import WorkspaceReference,REVISION,training_reference_spec
        with self.lock:latest=self.latest
        if not latest:return
        source=latest[0]['source']
        try:
            config_path=self.root/'models/current/config.json'
            signature=[(config_path.stat().st_mtime_ns,config_path.stat().st_size)]
            if self.workspace_spec and self.workspace_spec_signature and signature[0]==self.workspace_spec_signature[0]:
                labels=Path(self.workspace_spec['labels'])
                signature.append((labels.stat().st_mtime_ns,labels.stat().st_size))
            if signature!=self.workspace_spec_signature or force:
                self.workspace_spec=training_reference_spec(self.root)
                labels=Path(self.workspace_spec['labels'])
                self.workspace_spec_signature=[signature[0],(labels.stat().st_mtime_ns,labels.stat().st_size)]
            spec=self.workspace_spec
        except FileNotFoundError:
            self.workspace_spec=None;self.workspace_spec_signature=None
            with self.lock:
                self.workspace_reference=None;self.coverage_cache=None
                self.workspace_status=dict(status='waiting_training_samples',error='需要当前 GeoRT 模型及其原始训练 FK 样本')
            return
        except (KeyError,ValueError):
            with self.lock:self.workspace_reference=None;self.coverage_cache=None
            raise
        key=digest([source,REVISION,spec])
        if source!=self.workspace_source or self.workspace_folder!=self.root/'workspace'/key or force:
            if self.workspace_proc and self.workspace_proc.poll() is None:self.workspace_proc.terminate()
            folder=self.root/'workspace'/key;folder.mkdir(parents=True,exist_ok=True)
            self.workspace_source=source;self.workspace_folder=folder;self.workspace_reference=None;self.coverage_cache=None
            try:cached=json.loads((folder/'status.json').read_text())
            except (OSError,ValueError):cached={}
            adopted=False
            if cached.get('status')=='running' and not force:
                try:
                    command=Path('/proc')/str(int(cached['pid']))/'cmdline'
                    args=command.read_bytes().decode().split('\0')
                    adopted='retarget.baseline.geort.cli' in args and 'workspace' in args and str(folder) in args
                except (OSError,ValueError,KeyError):pass
            if (cached.get('status')!='completed' or force) and not adopted:
                write_json(folder/'source.json',source);write_json(folder/'reference_spec.json',spec)
                write_json(folder/'status.json',dict(status='starting'))
                with (folder/'process.log').open('a') as log:
                    self.workspace_proc=subprocess.Popen([os.environ.get('MOXIAN_GEORT_PYTHON',str(ROOT/'.venv-contact/bin/python')),
                        '-m','retarget.baseline.geort.cli','--root',str(self.root),'workspace','--folder',str(folder)],cwd=ROOT,
                        stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        if not self.workspace_folder:return
        try:status=json.loads((self.workspace_folder/'status.json').read_text())
        except (OSError,ValueError):return
        self.workspace_status={k:v for k,v in status.items() if k not in ('source','joint_limits')}
        if status.get('status')=='completed' and self.workspace_reference is None:
            reference=WorkspaceReference.load(self.workspace_folder)
            with self.lock:self.workspace_reference=reference;self.coverage_cache=None
        elif status.get('status') in ('running','starting') and self.workspace_proc and self.workspace_proc.poll() is not None:
            self.workspace_status=dict(status='failed',error='可达空间估计进程已退出，请查看 process.log')

    def refresh_published(self):
        try:
            active=json.loads((self.root/'active.json').read_text())
            run=active.get('run_id')
            if active.get('model')!='current' or not run or run==self.published_run:return
            with self.lock:
                if run==self.published_run:return
                self.selected='current';self.published_run=run
                self.model_update=dict(model='current',run_id=run,status='published')
            callback=self.on_model_published
            if callback:callback('current')
        except (OSError,ValueError,KeyError) as exc:
            with self.lock:self.model_update=dict(status='reload_failed',error=str(exc))

    def _job_folder(self,key):
        folder=self.root/'training'/key
        return folder if folder.exists() else self.root/'models'/key

    def start_recording(self, label='', capture_mode='geort', metadata=None):
        with self.lock:
            if capture_mode not in ('geort','cart','cart_v2','cart_v3'):raise ValueError('无效动作录制模式')
            if self.capture or self.recording.get('saving'):raise ValueError('上一段录制尚未保存')
            if not self.latest or time.monotonic()-self.latest_at>.5:
                raise ValueError('请先连接手套并完成六动作标定，等待有效的标定后手部模型')
            item,topology=self.latest;key=time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'_'+uuid.uuid4().hex[:8]
            folder=self.root/'recordings'/key;folder.mkdir(parents=True,exist_ok=False)
            source=item['source'];serial=source['serial'];device_hash=hashlib.sha256(serial.encode()).hexdigest()[:16]
            profile_path=ROOT/'output/moxian/hand_model'/device_hash/'qnbot/current.json'
            if not profile_path.is_file():raise ValueError('缺少已应用的手部模型配置')
            profile=json.loads(profile_path.read_text())
            if profile.get('model_hash')!=source['model_hash']:raise ValueError('已应用模型与实时输入不一致')
            from .source_ik_initialization import validate_recorded_source
            validate_recorded_source(profile,source)
            write_json(folder/'profile.json',profile);write_json(folder/'topology.json',topology)
            meta=dict(id=key,label=str(label)[:120],source=source,upstream=UPSTREAM,complete=False,
                      frames=0,created_at_ms=time.time()*1000,capture_mode=capture_mode,
                      cart_algorithm='cart_current_five_fingertips_v1' if capture_mode=='cart' else None)
            if metadata:meta.update(metadata)
            write_json(folder/'manifest.json',meta)
            self.capture=dict(id=key,source=source,folder=folder,queue=queue.Queue(maxsize=256),
                              stop=threading.Event(),last_stamp=-float('inf'),meta=meta)
            self.recording=dict(active=True,saving=False,id=key,frames=0,invalid_frames=0,error=None,
                                started_at=time.time(),capture_mode=capture_mode)
            from retarget.baseline.geort.coverage import WorkspaceCoverage
            # A new capture always starts a fresh distribution, including when
            # saved recordings were explicitly selected for comparison before it.
            self._clear_coverage_locked()
            self.coverage_segment=WorkspaceCoverage(source)
            self.coverage_current_id=key
            self.coverage_total=WorkspaceCoverage(source)
            self.coverage_cache=None
            self.capture['coverage']=self.coverage_segment
            threading.Thread(target=self._write,args=(self.capture,),daemon=True).start()
            return dict(self.recording)

    def _stop_locked(self, reason=None):
        if self.capture:
            self.capture['stop'].set();self.capture=None
            self.recording.update(active=False,saving=True,error=reason,seconds=round(time.time()-self.recording['started_at'],1))
        return dict(self.recording)

    def stop_recording(self, reason=None):
        with self.lock:return self._stop_locked(reason)

    def _write(self, capture):
        folder=capture['folder'];error=None;count=0;dimensions=None;temporary=[]
        try:
            import contextlib
            with contextlib.ExitStack() as stack:
                stream=stack.enter_context((folder/'frames.jsonl').open('x'))
                arrays={}
                for name in ('points','timestamps','qpos','joints'):
                    path=folder/(name+'.raw.tmp');temporary.append(path)
                    arrays[name]=stack.enter_context(path.open('xb'))
                while not capture['stop'].is_set() or not capture['queue'].empty():
                    try:row=capture['queue'].get(timeout=.05)
                    except queue.Empty:continue
                    stream.write(json.dumps({k:v for k,v in row.items() if k!='source'},allow_nan=False)+'\n')
                    if dimensions is None:dimensions=(len(row['qpos']),len(row['joints_m']))
                    for name,value,dtype in (('points',row['points'],'<f4'),('timestamps',[row['timestamp_ms']],'<f8'),
                                             ('qpos',row['qpos'],'<f4'),('joints',row['joints_m'],'<f4')):
                        arrays[name].write(np.asarray(value,dtype=dtype).tobytes())
                    count+=1
                    with self.lock:
                        capture['coverage'].add(row['points'],row['timestamp_ms'])
                        self.coverage_total.add(row['points'],row['timestamp_ms'])
                        self.recording['frames']=count
                stream.flush();os.fsync(stream.fileno())
            if count:
                points=np.memmap(folder/'points.raw.tmp',dtype='<f4',mode='r',shape=(count,5,3))
                stamps=np.memmap(folder/'timestamps.raw.tmp',dtype='<f8',mode='r',shape=(count,))
                angles=np.memmap(folder/'qpos.raw.tmp',dtype='<f4',mode='r',shape=(count,dimensions[0]))
                joints=np.memmap(folder/'joints.raw.tmp',dtype='<f4',mode='r',shape=(count,dimensions[1],3))
                np.save(folder/'points.npy',points);np.save(folder/'timestamps.npy',stamps)
                np.savez(folder/'motion.npz',qpos=angles,joints=joints)
                del points,stamps,angles,joints
            else:
                np.save(folder/'points.npy',np.empty((0,5,3),np.float32));np.save(folder/'timestamps.npy',np.empty(0))
            with self.lock:stats=dict(self.recording)
            capture['meta'].update(complete=True,frames=count,invalid_frames=stats['invalid_frames'],
                                   stop_reason=stats.get('error'),coverage=capture['coverage'].summary(),
                                   points_sha256=hashlib.sha256((folder/'points.npy').read_bytes()).hexdigest())
            write_json(folder/'manifest.json',capture['meta'])
            for path in temporary:path.unlink()
        except Exception as exc:error='录制保存失败：'+str(exc)
        finally:
            with self.lock:
                self.recording.update(active=False,saving=False)
                self.coverage_cache=None
                if error:self.recording['error']=error
                if self.capture is capture:self.capture=None

    def catalog(self, kind):
        if kind not in ('recordings','models'):raise ValueError('无效目录')
        items=[]
        for folder in sorted((self.root/kind).glob('*'),reverse=True):
            if not folder.is_dir():continue
            try:
                if kind=='recordings':item=json.loads((folder/'manifest.json').read_text())
                else:
                    try:config=json.loads((folder/'config.json').read_text())
                    except (OSError,ValueError):config={}
                    try:status=json.loads((folder/'status.json').read_text())
                    except (OSError,ValueError):status=dict(status='incomplete')
                    item=dict(status,source=config.get('source'),datasets=config.get('datasets',[]),
                              reproduction_profile=config.get('reproduction_profile'),training_mode=config.get('training_mode','official'))
                    item['id']=folder.name
                    if (folder/'evaluation.json').exists():item['evaluation']=json.loads((folder/'evaluation.json').read_text())
                items.append(item)
            except (OSError,ValueError,KeyError):continue
        return items

    def _clear_coverage_locked(self, notice=None):
        self.coverage_base=None;self.coverage_total=None;self.coverage_segment=None;self.coverage_ids=[]
        self.coverage_current_id=None
        self.coverage_cache=None;self.coverage_notice=notice

    def select_coverage(self, ids):
        """Analyze saved compatible data off the hand-frame ingestion thread."""
        from retarget.baseline.geort.training import load_recordings
        from retarget.baseline.geort.coverage import WorkspaceCoverage
        from .saved_assets import asset_lock
        if not isinstance(ids,list):raise ValueError('覆盖基线需要录制ID列表')
        with self.lock:
            if self.capture or self.recording.get('saving'):raise ValueError('请先停止并保存，再切换覆盖基线')
        with asset_lock(self.root):
            if ids:
                all_points,source,_=load_recordings(self.root,ids)
                base=WorkspaceCoverage(source)
                offset=0
                for key in ids:
                    folder=self.root/'recordings'/safe_id(key)
                    stamps=np.load(folder/'timestamps.npy',allow_pickle=False,mmap_mode='r')
                    points=all_points[offset:offset+len(stamps)];offset+=len(stamps)
                    if stamps.shape!=(len(points),) or not np.isfinite(stamps).all() or np.any(np.diff(stamps)<=0):
                        raise ValueError('录制时间戳无效')
                    base.reset_time()
                    for point,stamp in zip(points,stamps):base.add(point,float(stamp))
                base.reset_time();base.recent.clear()
            else:base=None
            with self.lock:
                if self.capture or self.recording.get('saving'):raise ValueError('录制已开始，请停止后再切换覆盖基线')
                if base and self.latest and base.source!=self.latest[0]['source']:
                    raise ValueError('覆盖基线与当前设备、标定或坐标系不兼容')
                self.coverage_base=base;self.coverage_total=base.clone() if base else None
                self.coverage_segment=None;self.coverage_current_id=None;self.coverage_ids=list(ids);self.coverage_cache=None;self.coverage_notice=None
        return self.coverage_snapshot()

    def coverage_snapshot(self, include_cloud=False, finger='thumb'):
        with self.lock:
            now=time.monotonic()
            current=self.coverage_total
            compatible=not current or not self.latest or current.source==self.latest[0]['source']
            if (not include_cloud and self.coverage_cache is not None and now-self.coverage_cache_at<.25
                    and self.coverage_cache['compatible']==compatible):
                return self.coverage_cache
            result=dict(ready=current is not None,compatible=compatible,baseline_ids=list(self.coverage_ids),
                        recording=bool(self.capture),notice=self.coverage_notice,
                        scope='selected_baseline_plus_current_recording',
                        source_fingerprint=digest(current.source) if current else None)
            if current:
                result.update(current.summary(include_cloud,now))
                result['segment']=self.coverage_segment.summary(now=now) if self.coverage_segment else None
            reference=self.workspace_reference
            expected=current.source if current else self.latest[0]['source'] if self.latest else None
            if reference and reference.source==expected:
                if current is None:
                    from retarget.baseline.geort.coverage import WorkspaceCoverage
                    result.update(WorkspaceCoverage(expected).summary());result['ready']=True
                result['workspace']=reference.evaluate(current.coarse_counts if current else None,finger if include_cloud else None)
            else:result['workspace']=self.workspace_status if self.workspace_enabled else dict(status='not_enabled')
            if not compatible:result['notice']='当前设备或标定与覆盖数据不同，请重新设置基线并录制'
            if not include_cloud:self.coverage_cache=result;self.coverage_cache_at=now
            return result

    def workspace_preview(self,reference_id,sample_id=None,finger='thumb'):
        with self.lock:
            reference=self.workspace_reference;folder=self.workspace_folder
            expected=self.latest[0]['source'] if self.latest else None
            if (reference is None or folder is None or folder.name!=reference_id or
                    reference.source!=expected):
                raise ValueError('姿态参考已更新，请刷新散点后重新选择')
            if sample_id is None:
                if not reference.preview_available:raise ValueError('旧参考缺少对应姿态，请重建姿态参考')
                return json.loads((folder/'robot_topology.json').read_text())
            current=self.coverage_total
            counts=current.coarse_counts if current and current.source==reference.source else None
            return reference.pose(sample_id,finger,counts)

    def delete_assets(self, kind, ids, before_delete=None):
        from .saved_assets import delete_saved
        with self.lock:
            if self.proc and self.proc.poll() is None:raise ValueError('后台训练或评估正在运行，请先取消或等待完成')
            if kind=='recordings' and self.replay_active:raise ValueError('请先返回实时输入再删除录制数据')
            if kind=='recordings' and (self.capture or self.recording.get('saving')):
                raise ValueError('请先停止并保存录制，再删除数据')
            result=delete_saved(self.root,kind,ids,before_delete=before_delete)
            if kind=='models':
                if self.selected in result['deleted']:
                    self.selected=None;(self.root/'active.json').unlink(missing_ok=True)
                    result['unloaded']=True
                if self.job_id in result['deleted']:self.job_id=None
                if getattr(self,'evaluation_model',None) in result['deleted']:self.evaluation_model=None
            elif set(result['deleted'])&set(self.coverage_ids):
                self._clear_coverage_locked('覆盖基线包含已删除数据，已清空；可重新选择剩余数据')
            elif kind=='recordings' and self.coverage_current_id in result['deleted']:
                self.coverage_segment=None;self.coverage_current_id=None
                self.coverage_total=self.coverage_base.clone() if self.coverage_base else None
                self.coverage_cache=None;self.coverage_notice='当前段录制已删除，覆盖仅保留所选基线'
            if kind=='recordings' and self.recording.get('id') in result['deleted']:
                self.recording=dict(active=False,saving=False,frames=0,invalid_frames=0,error=None)
            return result

    def start_training(self, ids, epochs=200, fk_epochs=200, mode='official'):
        from retarget.baseline.geort.training import load_recordings
        for value in (epochs,fk_epochs):
            if isinstance(value,bool) or not isinstance(value,int) or not 1<=value<=2000:raise ValueError('训练轮数必须为1–2000')
        if mode not in ('official','contact_v1'):raise ValueError('未知的 GeoRT 训练模式')
        with self.lock:
            if self.proc and self.proc.poll() is None:raise ValueError('已有 GeoRT 任务正在运行')
            import fcntl
            self.root.mkdir(parents=True,exist_ok=True)
            with (self.root/'training.lock').open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:raise ValueError('已有 GeoRT 命令行训练正在运行')
            load_recordings(self.root,ids)
            key=time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'_'+uuid.uuid4().hex[:8]
            folder=self.root/'training'/key;folder.mkdir(parents=True)
            write_json(folder/'status.json',dict(run_id=key,status='starting'))
            self.job_id=key
            self.job_log=(folder/'process.log').open('a')
            try:
                self.proc=subprocess.Popen([os.environ.get('MOXIAN_GEORT_PYTHON',str(ROOT/'.venv-contact/bin/python')),'-m','retarget.baseline.geort.cli',
                    '--root',str(self.root),'train','--run-id',key,'--epochs',str(epochs),'--fk-epochs',str(fk_epochs),
                    '--mode',mode,'--datasets',*ids],cwd=ROOT,stdout=self.job_log,stderr=subprocess.STDOUT,
                    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
            finally:self.job_log.close()
            return dict(id=key,status='starting')

    def cancel_training(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                if self.job_id:
                    path=self._job_folder(self.job_id)/'status.json'
                    try:
                        status=json.loads(path.read_text());status.update(status='cancelling',cancel_requested=True);write_json(path,status)
                    except (OSError,ValueError):pass
                self.proc.terminate()
            else:
                job=self._latest_job()
                if job and job.get('status') in ('starting','running'):
                    # A CLI-started job can outlive the WebUI. Only signal our
                    # CLI with this exact run ID, never a PID from an API request.
                    for entry in Path('/proc').glob('[0-9]*'):
                        try:
                            args=(entry/'cmdline').read_bytes().split(b'\0')
                            if b'retarget.baseline.geort.cli' in args and b'train' in args and job['run_id'].encode() in args:
                                os.kill(int(entry.name),15)
                        except (OSError,ValueError):continue
        return dict(cancellation_requested=True)

    def _latest_job(self):
        candidates=[]
        for path in [*(self.root/'models').glob('*/status.json'),*(self.root/'training').glob('*/status.json')]:
            try:
                item=json.loads(path.read_text());candidates.append((path.stat().st_mtime,item))
            except (OSError,ValueError):continue
        return max(candidates,key=lambda v:v[0])[1] if candidates else None

    def activate(self, key):
        self.model_folder(key);self.root.mkdir(parents=True,exist_ok=True)
        active=dict(model=key)
        if key=='current':
            config=json.loads((self.root/'models/current/config.json').read_text())
            if config.get('run_id'):active['run_id']=config['run_id'];self.published_run=config['run_id']
        write_json(self.root/'active.json',active);self.selected=key

    def start_evaluation(self, model_id, ids):
        from retarget.baseline.geort.training import load_recordings
        folder=self.model_folder(model_id)
        _,source,_=load_recordings(self.root,ids)
        config=json.loads((folder/'config.json').read_text())
        if source!=config['source']:raise ValueError('评估数据与模型不兼容')
        if set(ids)&{d['id'] for d in config['datasets']}:raise ValueError('请另录一段独立动作进行评估')
        with self.lock:
            if self.proc and self.proc.poll() is None:raise ValueError('已有 GeoRT 任务正在运行')
            with (folder/'evaluation-process.log').open('a') as log:
                self.proc=subprocess.Popen([os.environ.get('MOXIAN_GEORT_PYTHON',str(ROOT/'.venv-contact/bin/python')),'-m','retarget.baseline.geort.cli',
                    '--root',str(self.root),'evaluate','--model',model_id,'--datasets',*ids],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            self.evaluation_model=model_id
        return dict(status='starting',model=model_id)

    def model_folder(self, key):
        folder=self.root/'models'/safe_id(key)
        status=json.loads((folder/'status.json').read_text())
        if status.get('status')!='completed':raise ValueError('请选择训练成功的模型')
        return folder

    def snapshot(self):
        with self.lock:
            result=dict(recording=dict(self.recording),selected_model=self.selected,replay_active=self.replay_active,model_update=self.model_update)
            if self.recording.get('active'):
                result['recording']['seconds']=round(time.time()-self.recording['started_at'],1)
            key=self.job_id;proc=self.proc
        if getattr(self,'evaluation_model',None):
            try:result['evaluation']=json.loads((self.root/'models'/self.evaluation_model/'evaluation_status.json').read_text())
            except (OSError,ValueError):result['evaluation']={'status':'starting'}
        if key:
            try:result['training']=json.loads((self._job_folder(key)/'status.json').read_text())
            except (OSError,ValueError):result['training']=dict(status='starting',run_id=key)
            if proc and proc.poll() is not None and result['training'].get('status') in ('running','starting','cancelling'):
                cancelled=result['training'].get('cancel_requested',False)
                result['training'].update(status='cancelled' if cancelled else 'failed',error='训练已取消' if cancelled else f'训练进程退出 ({proc.returncode})，请查看 process.log')
                write_json(self._job_folder(key)/'status.json',result['training'])
        else:
            job=self._latest_job()
            if job:result['training']=job
        result['coverage']=self.coverage_snapshot()
        if hasattr(self,'diagnostics'):result['diagnostics']=self.diagnostics.snapshot()
        return result

    def close(self):
        # Training jobs write to their own logs and survive a WebUI restart.
        self.closed=True;self.stop_recording('服务停止')
        if hasattr(self,'diagnostics'):self.diagnostics.close()
