"""CART control plane; no torch/SciPy/mesh operations on the SDK thread."""
import hashlib,json,os,queue,subprocess,threading,time,uuid,signal
import numpy as np
from pathlib import Path
from retarget.baseline.geort.adapter import ROOT
from .geort_service import GeoRTService,safe_id,write_json
from .cart_protocol import PROTOCOL,SUPPORTED_RECORDINGS

class CartService:
    def __init__(self,root=None):
        self.root=Path(root or ROOT/'output/moxian/cart');self.recorder=GeoRTService(self.root)
        from .cart_ik_view_service import IKViewService
        self.ik_view=IKViewService(self.root)
        self.lock=threading.RLock();self.stage_id='free_motion';self.split='train';self.session_id=uuid.uuid4().hex
        self.preview=None;self.preview_proc=None;self.preview_key=None;self.preview_at=0.;self.last_sent=0.;self.preview_topology=None
        self.preview_queue=queue.Queue(maxsize=1);self.closed=False;self.manual='auto';self.selected=None
        self.generation=0;self.capture_proc=None;self.capture_key=None;self.capture_pid=None
        self.motion={}
        self.job=None;self.job_id=None;self.job_action=None;self.replay_active=False;self.replay_topology=None
        try:self.selected=json.loads((self.root/'active.json').read_text())['model']
        except (OSError,ValueError,KeyError):pass
        self.error=None;self.latest=None
        self._recover_recordings();self._recover_geometry()
        try:
            config=json.loads((self.root/'protocol_state.json').read_text())
            self.split=config['split'];self.session_id=config['session_id']
        except (OSError,ValueError,KeyError):pass
    @property
    def python(self):return os.environ.get('MOXIAN_CART_PYTHON',str(ROOT/'.venv-contact/bin/python'))
    def _recover_recordings(self):
        """Finalize only orphaned raw captures after a service interruption."""
        for path in (self.root/'recordings').glob('*/manifest.json'):
            try:
                meta=json.loads(path.read_text())
                if meta.get('complete') or meta.get('protocol')!=PROTOCOL:continue
                folder=path.parent;rows=[]
                with (folder/'frames.jsonl').open() as stream:
                    for line in stream:
                        try:rows.append(json.loads(line))
                        except ValueError:break
                if rows:
                    np.save(folder/'points.npy',np.asarray([r['points'] for r in rows],np.float32))
                    np.save(folder/'timestamps.npy',np.asarray([r['timestamp_ms'] for r in rows],np.float64))
                    np.savez(folder/'motion.npz',qpos=np.asarray([r['qpos'] for r in rows],np.float32),
                             joints=np.asarray([r['joints_m'] for r in rows],np.float32))
                    # Truncate a torn tail before making the capture complete.
                    (folder/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                    meta.update(complete=True,frames=len(rows),recovered=True,stop_reason='服务中断，已恢复完整落盘帧',
                        points_sha256=hashlib.sha256((folder/'points.npy').read_bytes()).hexdigest())
                    write_json(path,meta)
            except (OSError,ValueError,KeyError):continue
    def _geometry_running(self):
        if self.capture_proc and self.capture_proc.poll() is None:return True
        if self.capture_pid and self.capture_key:
            try:
                command=Path('/proc',str(self.capture_pid),'cmdline').read_bytes().split(b'\0')
                return (b'moxian_web.cart_cli' in command and b'capture' in command and
                        self.capture_key.encode() in command and str(self.root).encode() in command)
            except OSError:pass
        return False

    def _start_geometry_capture(self,key):
        with self.lock:
            if self._geometry_running():raise ValueError('上一段 wrist 几何仍在保存')
            folder=self.root/'recordings'/key
            log=(folder/'capture.log').open('a')
            try:
                self.capture_proc=subprocess.Popen([self.python,'-u','-m','moxian_web.cart_cli','--root',str(self.root),
                    'capture','--dataset',key,'--follow'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
                self.capture_key=key;self.capture_pid=self.capture_proc.pid
            finally:log.close()

    def _recover_geometry(self):
        if self.closed:return
        if self.job and self.job.poll() is None:return
        if self._geometry_running():return
        from .cart_capture_schema import CAPTURE_SCHEMA
        for path in sorted((self.root/'recordings').glob('*/manifest.json'),reverse=True):
            try:
                meta=json.loads(path.read_text())
                if not meta.get('complete') or meta.get('capture_geometry_schema')!=CAPTURE_SCHEMA:continue
                status_path=path.parent/'cart_capture_status.json'
                status=json.loads(status_path.read_text()) if status_path.exists() else {}
                # Explicit cancellation is persistent; only a user-requested
                # preparation action may resume it.
                if status.get('status') in ('completed','failed','cancelled'):continue
                pid=status.get('pid')
                if isinstance(pid,int) and Path('/proc',str(pid),'cmdline').exists():
                    command=Path('/proc',str(pid),'cmdline').read_bytes().split(b'\0')
                    if b'moxian_web.cart_cli' in command and b'capture' in command and path.parent.name.encode() in command:
                        self.capture_pid=pid;self.capture_key=path.parent.name;return
                self._start_geometry_capture(path.parent.name);return
            except (OSError,ValueError):continue

    def observe(self,frame,topology,joints,serial):
        label=dict(protocol=PROTOCOL,stage=self.stage_id,split=self.split,session_id=self.session_id)
        self.recorder.observe({**frame,'cart_label':label},topology,joints,serial)
        self.latest=(frame,topology,serial)
        # A bounded mailbox only drops preview work, never recorded frames.
        if not frame.get('ready') or not frame.get('calibrated') or frame.get('preview'):return
        if time.monotonic()-self.last_sent<.1:return
        self.last_sent=time.monotonic()
        if not self.preview_proc:return
        try:
            if self.preview_key!=(frame.get('geometry_version'),frame.get('binding_version'),frame.get('model_hash'),frame.get('calibration_fingerprint')):
                self.stop_preview();return
            item=dict(type='frame',qpos=frame.get('source_joint_positions_rad'),timestamp_ms=frame['timestamp_ms'],
                source_contact=frame.get('source_contact'),tactile_timestamp_ms=frame.get('tactile_timestamp_ms'),generation=self.generation)
            try:self.preview_queue.get_nowait()
            except queue.Empty:pass
            self.preview_queue.put_nowait(item)
        except Exception as exc:self.error=str(exc)
    def start_preview(self):
        if self.preview_proc and self.preview_proc.poll() is None:return
        if not self.latest:raise ValueError('等待有效标定后的手部模型')
        frame,topology,serial=self.latest
        path=ROOT/'output/moxian/hand_model'/hashlib.sha256(serial.encode()).hexdigest()[:16]/'qnbot/current.json'
        profile=json.loads(path.read_text())
        self.root.mkdir(parents=True,exist_ok=True)
        log=(self.root/'preview.log').open('a')
        proc=subprocess.Popen([self.python,'-u','-m','moxian_web.cart_cli','preview'],cwd=ROOT,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,
            env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        log.close();self.preview_proc=proc;self.preview_topology=topology;self.preview_key=(frame['geometry_version'],frame['binding_version'],frame.get('model_hash'),frame.get('calibration_fingerprint'));self.error=None
        proc.stdin.write(json.dumps(dict(type='init',topology=topology,profile=profile))+'\n');proc.stdin.flush()
        def writer():
            try:
                while proc.poll() is None and not self.closed:
                    try:item=self.preview_queue.get(timeout=.2)
                    except queue.Empty:continue
                    proc.stdin.write(json.dumps(item)+'\n');proc.stdin.flush()
            except (OSError,ValueError) as exc:
                if self.preview_proc is proc:self.error=str(exc)
        def reader():
            try:
                for line in proc.stdout:
                    result=json.loads(line)
                    if self.preview_proc is not proc:return
                    if result.get('generation',self.generation)!=self.generation:continue
                    with self.lock:
                        if result.get('error'):self.error=result['error']
                        else:self.preview=result;self.preview_at=time.monotonic();self.error=None
            except (OSError,ValueError) as exc:
                if self.preview_proc is proc:self.error=str(exc)
        for action in (writer,reader):threading.Thread(target=action,daemon=True).start()
    def stop_preview(self):
        proc=self.preview_proc;self.preview_proc=None;self.preview=None
        if proc and proc.poll() is None:proc.terminate()
    def start_recording(self,data):
        if self.replay_active:raise ValueError('请先退出回放')
        if self.recorder.recording.get('active'):raise ValueError('请先停止当前动作录制')
        if not self.recorder.latest or not self.recorder.latest[0].get('emf_observation'):
            raise ValueError('缺少原始 EMF 完整位姿，请重启源手套服务后录制')
        if self._geometry_running():raise ValueError('上一段 wrist 几何仍在保存，请等待完成')
        from .cart_capture_schema import CAPTURE_SCHEMA
        self.configure(data);self.start_preview()
        result=self.recorder.start_recording(data.get('label') or '任意动作',capture_mode='cart_v5',metadata=dict(
            protocol=PROTOCOL,split=self.split,session_id=self.session_id,tactile_recorded=True,emf_recorded=True,
            capture_geometry_schema=CAPTURE_SCHEMA,coordinate_frame='wrist_local',units='m'))
        try:self._start_geometry_capture(result['id'])
        except Exception:
            self.recorder.stop_recording('wrist 几何进程启动失败；原始数据已保留，可用准备观测补算');raise
        return result
    def configure(self,data):
        if self.recorder.recording.get('active') or self.recorder.recording.get('saving'):raise ValueError('录制期间不能切换动作或分区')
        if data.get('stage') not in (None,'free_motion'):raise ValueError('CART 已改为任意动作采集，无需指定动作')
        if 'split' in data:
            if data['split'] not in ('train','validation'):raise ValueError('无效训练／验证分区')
            if data['split']!=self.split:self.session_id=uuid.uuid4().hex
            self.split=data['split']
        self.preview=None
        self.motion={}
        self.generation+=1;self.root.mkdir(parents=True,exist_ok=True)
        write_json(self.root/'protocol_state.json',dict(stage=self.stage_id,split=self.split,session_id=self.session_id))
        return dict(stage=self.stage_id,split=self.split)
    def catalog(self,kind):
        if kind=='recordings':
            result=[m for m in self.recorder.catalog('recordings') if m.get('protocol') in SUPPORTED_RECORDINGS]
            for m in result:
                try:m['ik']=json.loads((self.root/'recordings'/m['id']/'cart_ik.json').read_text())
                except (OSError,ValueError):pass
                try:m['ik_progress']=json.loads((self.root/'recordings'/m['id']/'cart_ik_status.json').read_text())
                except (OSError,ValueError):pass
                try:m['capture_geometry']=json.loads((self.root/'recordings'/m['id']/'cart_capture_status.json').read_text())
                except (OSError,ValueError):pass
                try:m['preprocessing']=json.loads((self.root/'recordings'/m['id']/'cart_preprocess.json').read_text())
                except (OSError,ValueError):pass
            return result
        items=[]
        for folder in sorted((self.root/'models').glob('*'),reverse=True):
            try:
                status=json.loads((folder/'status.json').read_text());status['id']=folder.name
                deployment=folder/'deployment.json'
                if deployment.exists():
                    imported=json.loads(deployment.read_text())
                    status.update(imported=True,experimental=imported.get('experimental',False))
                items.append(status)
            except (OSError,ValueError):pass
        return items
    def start_job(self,action,data):
        from .cart_data import manifest
        if self._geometry_running():raise ValueError('wrist 几何正在保存，请等待完成')
        if self.job and self.job.poll() is None:raise ValueError('已有 CART 后台任务')
        ids=data.get('datasets',[])
        if not ids:raise ValueError('请选择 CART 数据')
        for key in ids:manifest(self.root,key)
        if action=='train':
            metadata=[manifest(self.root,key)[1] for key in ids]
            if any(not any(m['split']==split for m in metadata) for split in ('train','validation')):
                raise ValueError('请选择独立的训练和验证会话；无需指定动作类别')
        key=time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'_'+uuid.uuid4().hex[:8]
        folder=self.root/'jobs'/key;folder.mkdir(parents=True)
        command=[self.python,'-u','-m','moxian_web.cart_cli','--root',str(self.root),action,'--datasets',*ids,'--job-id',key]
        if action=='train':
            epochs=data.get('epochs',200)
            if isinstance(epochs,bool) or not isinstance(epochs,int) or not 1<=epochs<=2000:raise ValueError('轮数必须为1–2000')
            architecture=data.get('architecture','attention')
            if architecture not in ('attention','fsq'):raise ValueError('无效 CART 网络结构')
            batch=data.get('batch_size',0);micro=data.get('micro_batch_size',0)
            if isinstance(batch,bool) or not isinstance(batch,int) or not 0<=batch<=65536:raise ValueError('并行窗口数必须为 0–65536；0 表示按显存自动选择')
            if isinstance(micro,bool) or not isinstance(micro,int) or micro<0:raise ValueError('GPU 分块必须为非负整数')
            noise=data.get('initial_noise_rad',.1)
            if isinstance(noise,bool) or not isinstance(noise,(int,float)) or not np.isfinite(noise) or not 0<=noise<=.5:raise ValueError('IK 初始噪声幅度须为 0–0.5 rad')
            command+=['--epochs',str(epochs),'--architecture',architecture,'--batch-size',str(batch),'--micro-batch-size',str(micro)]
            command+=['--initial-noise-rad',str(noise)]
            stride=data.get('window_stride',0)
            if isinstance(stride,bool) or not isinstance(stride,int) or not 0<=stride<=65536:raise ValueError('窗口步长必须为 0–65536；0 表示 length//2')
            command+=['--window-stride',str(stride)]
            if data.get('initialize_from'):command+=['--initialize-from',safe_id(data['initialize_from'])]
            if data.get('initialize_geometry_from'):command+=['--initialize-geometry-from',safe_id(data['initialize_geometry_from'])]
            if data.get('initialize_sdf_from'):command+=['--initialize-sdf-from',str(data['initialize_sdf_from'])]
            for field in ('wandb_project','wandb_entity','wandb_base_url'):
                if data.get(field):command+=['--'+field.replace('_','-'),str(data[field])]
            self.stop_preview()
        if action=='evaluate':command+=['--model',safe_id(data.get('model'))]
        write_json(folder/'status.json',dict(status='starting',action=action,id=key))
        log=(folder/'process.log').open('a')
        self.job=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'});log.close()
        self.job_id=key;self.job_action=action
        return dict(id=key,status='starting')
    def cancel(self):
        if self.job and self.job.poll() is None:self.job.terminate()
        else:
            # Recover only this exact CLI/job after a WebUI restart.
            for entry in Path('/proc').glob('[0-9]*'):
                try:
                    args=(entry/'cmdline').read_bytes().split(b'\0')
                    if b'moxian_web.cart_cli' in args and self.job_id and self.job_id.encode() in args:os.kill(int(entry.name),15)
                except (OSError,ValueError):pass
        if self._geometry_running():
            try:os.kill(self.capture_pid,signal.SIGTERM)
            except ProcessLookupError:pass
        return dict(cancellation_requested=True)
    def model_folder(self,key):
        folder=self.root/'models'/safe_id(key)
        from .cart_deployment import validate_package
        if validate_package(folder) is None:
            from .cart_training import validate_model_geometry
            validate_model_geometry(json.loads((folder/'config.json').read_text()))
        return folder
    def activate(self,key):
        self.model_folder(key);write_json(self.root/'active.json',dict(model=key));self.selected=key
    def delete_assets(self,kind,ids,before_delete=None):
        with self.lock:
            if kind=='recordings' and self._geometry_running():raise ValueError('wrist 几何正在保存，请等待完成')
            if self.job and self.job.poll() is None:raise ValueError('CART 后台任务正在运行，请先取消或等待完成')
            if kind=='recordings' and self.replay_active:raise ValueError('请先返回实时输入再删除录制数据')
            self.recorder.selected=self.selected
            result=self.recorder.delete_assets(kind,ids,before_delete)
            self.selected=self.recorder.selected
            return result
    def preview_packet(self):
        """Independent collection viewer, including when the Wuji solver is idle."""
        import base64
        data=self.snapshot(connected=time.monotonic()-self.recorder.latest_at<1,include_mesh=True)
        frame=data.get('preview_frame')
        if not data['ready'] or not frame:
            return dict(ready=False,error=data.get('error'),preview_age_ms=data.get('preview_age_ms'))
        vertices=np.asarray(frame['vertices_m'],dtype='<f4');joints=np.asarray(frame['joints_m'],dtype='<f4')
        if (vertices.ndim!=2 or joints.ndim!=2 or vertices.shape[1]!=3 or joints.shape[1]!=3
                or not np.isfinite(vertices).all() or not np.isfinite(joints).all()):
            return dict(ready=False,error='CART 预览网格无效')
        return dict(ready=True,hull=data['hull'],preview_age_ms=data['preview_age_ms'],
            timestamp_ms=frame['timestamp_ms'],geometry_version=frame['geometry_version'],binding_version=frame['binding_version'],
            mesh_f32=dict(vertices=len(vertices),joints=len(joints),
                          data=base64.b64encode(vertices.tobytes()+joints.tobytes()).decode()))
    def snapshot(self,connected=True,include_mesh=False):
        with self.lock:
            preview=dict(self.preview or {});age=(time.monotonic()-self.preview_at)*1000 if self.preview_at else None
        result=dict(protocol=PROTOCOL,split=self.split,
            recording=self.recorder.snapshot()['recording'],selected_model=self.selected,replay_active=self.replay_active,
            preview_age_ms=age,error=self.error,ready=bool(connected and age is not None and age<3000),
            hull=preview.get('hull'),motion=self.motion,geometric_motion=preview.get('motion'),session_id=self.session_id)
        self._recover_geometry()
        if self.capture_key:
            result['capture_geometry']=dict(status='starting',id=self.capture_key,frames=0)
            try:result['capture_geometry']=json.loads((self.root/'recordings'/self.capture_key/'cart_capture_status.json').read_text())
            except (OSError,ValueError):pass
            if not result['recording'].get('active') and self._geometry_running():
                result['recording']=dict(result['recording'],saving=True)
        if not result['ready']:result['hull']=None
        if include_mesh and result['ready']:result['preview_frame']=preview.get('preview_frame')
        if not self.job_id:
            paths=sorted((self.root/'jobs').glob('*/status.json'),key=lambda p:p.stat().st_mtime)
            if paths:self.job_id=paths[-1].parent.name
        if self.job_id:
            try:result['job']=json.loads((self.root/'jobs'/self.job_id/'status.json').read_text())
            except (OSError,ValueError):pass
            model_status=self.root/'models'/self.job_id/'status.json'
            if model_status.exists():
                try:result['job']={**result.get('job',{}),**json.loads(model_status.read_text())}
                except (OSError,ValueError):pass
            if self.job and self.job.poll() is not None and result.get('job',{}).get('status') in ('starting','running'):
                result['job'].update(status='failed',error='后台进程已退出，请查看 process.log')
        return result
    def close(self):
        self.closed=True;self.stop_preview();self.recorder.close();self.ik_view.close()
        if self._geometry_running():
            try:os.kill(self.capture_pid,signal.SIGTERM)
            except ProcessLookupError:pass
        deadline=time.monotonic()+5
        while self.recorder.recording.get('saving') and time.monotonic()<deadline:time.sleep(.02)
