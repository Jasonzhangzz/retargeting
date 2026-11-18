"""CART control plane; no torch/SciPy/mesh operations on the SDK thread."""
import hashlib,json,os,queue,subprocess,threading,time,uuid
import numpy as np
from pathlib import Path
from retarget.baseline.geort.adapter import ROOT
from .geort_service import GeoRTService,safe_id,write_json
from .cart_protocol import PROTOCOL,STAGES,stage,MotionTracker

class CartService:
    def __init__(self,root=None):
        self.root=Path(root or ROOT/'output/moxian/cart');self.recorder=GeoRTService(self.root)
        self.lock=threading.RLock();self.stage_id=STAGES[0]['id'];self.split='train';self.session_id=uuid.uuid4().hex
        self.preview=None;self.preview_proc=None;self.preview_key=None;self.preview_at=0.;self.last_sent=0.;self.preview_topology=None
        self.preview_queue=queue.Queue(maxsize=1);self.closed=False;self.manual='auto';self.selected=None
        self.generation=0
        self.progress_tracker=MotionTracker();self.motion={}
        self.job=None;self.job_id=None;self.job_action=None;self.replay_active=False;self.replay_topology=None
        try:self.selected=json.loads((self.root/'active.json').read_text())['model']
        except (OSError,ValueError,KeyError):pass
        self.error=None;self.latest=None
        self._recover_recordings()
        try:
            config=json.loads((self.root/'protocol_state.json').read_text());stage(config['stage'])
            self.stage_id=config['stage'];self.split=config['split'];self.session_id=config['session_id']
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
    def observe(self,frame,topology,joints,serial):
        label=dict(protocol=PROTOCOL,stage=self.stage_id,split=self.split,session_id=self.session_id)
        self.recorder.observe({**frame,'cart_label':label},topology,joints,serial)
        self.latest=(frame,topology,serial)
        # A bounded mailbox only drops preview work, never recorded frames.
        if not frame.get('ready') or not frame.get('calibrated') or frame.get('preview'):return
        # Guidance uses every raw frame, even when expensive geometric preview
        # skips work. This counter is a motion proxy, never contact evidence.
        spec=stage(self.stage_id);j=np.asarray(joints);chains=topology['joint_chains']
        thumb=j[chains[0][-1]];tips=j[[c[-1] for c in chains]]
        if spec['category']==1:gap=np.linalg.norm(thumb-tips[spec['fingers'][-1]])
        elif spec['category']==2:gap=np.mean([np.linalg.norm(thumb-tips[f]) for f in range(1,5)])
        else:gap=np.mean([np.linalg.norm(tips[f]-j[chains[f][-4]]) for f in range(1,5)])
        self.motion={**self.progress_tracker.update(frame['timestamp_ms'],gap), 'evidence':'bone_motion_proxy'}
        if time.monotonic()-self.last_sent<.1:return
        self.last_sent=time.monotonic()
        if not self.preview_proc:return
        try:
            if self.preview_key!=(frame.get('geometry_version'),frame.get('binding_version'),frame.get('model_hash'),frame.get('calibration_fingerprint')):
                self.stop_preview();return
            item=dict(type='frame',qpos=frame.get('source_joint_positions_rad'),timestamp_ms=frame['timestamp_ms'],stage=self.stage_id,generation=self.generation)
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
        self.configure(data);self.start_preview()
        spec=stage(self.stage_id)
        return self.recorder.start_recording(data.get('label') or spec['label'],capture_mode='cart_v3',metadata=dict(
            protocol=PROTOCOL,stage=self.stage_id,category=spec['category'],category_name=spec['category_name'],split=self.split,session_id=self.session_id))
    def configure(self,data):
        if self.recorder.recording.get('active') or self.recorder.recording.get('saving'):raise ValueError('录制期间不能切换动作或分区')
        if data.get('stage'):stage(data['stage']);self.stage_id=data['stage']
        if 'split' in data:
            if data['split'] not in ('train','validation'):raise ValueError('无效训练／验证分区')
            if data['split']!=self.split:self.session_id=uuid.uuid4().hex
            self.split=data['split']
        self.preview=None
        self.progress_tracker.reset();self.motion={}
        self.generation+=1;self.root.mkdir(parents=True,exist_ok=True)
        write_json(self.root/'protocol_state.json',dict(stage=self.stage_id,split=self.split,session_id=self.session_id))
        return dict(stage=self.stage_id,split=self.split)
    def catalog(self,kind):
        if kind=='recordings':
            result=[m for m in self.recorder.catalog('recordings') if m.get('protocol')==PROTOCOL]
            for m in result:
                try:m['preprocessing']=json.loads((self.root/'recordings'/m['id']/'cart_preprocess.json').read_text())
                except (OSError,ValueError):pass
            return result
        items=[]
        for folder in sorted((self.root/'models').glob('*'),reverse=True):
            try:
                status=json.loads((folder/'status.json').read_text());status['id']=folder.name
                items.append(status)
            except (OSError,ValueError):pass
        return items
    def start_job(self,action,data):
        from .cart_data import manifest
        if self.job and self.job.poll() is None:raise ValueError('已有 CART 后台任务')
        ids=data.get('datasets',[])
        if not ids:raise ValueError('请选择 CART 数据')
        for key in ids:manifest(self.root,key)
        if action=='train':
            metadata=[manifest(self.root,key)[1] for key in ids]
            for split in ('train','validation'):
                if {m['stage'] for m in metadata if m['split']==split}!={s['id'] for s in STAGES}:
                    raise ValueError('训练和独立验证各需全部8个子动作，请先完成五类动作采集')
        key=time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'_'+uuid.uuid4().hex[:8]
        folder=self.root/'jobs'/key;folder.mkdir(parents=True)
        command=[self.python,'-u','-m','moxian_web.cart_cli','--root',str(self.root),action,'--datasets',*ids,'--job-id',key]
        if action=='train':
            epochs=data.get('epochs',200)
            if isinstance(epochs,bool) or not isinstance(epochs,int) or not 1<=epochs<=2000:raise ValueError('轮数必须为1–2000')
            command+=['--epochs',str(epochs)]
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
        return dict(cancellation_requested=True)
    def model_folder(self,key):
        folder=self.root/'models'/safe_id(key)
        status=json.loads((folder/'status.json').read_text())
        if status['status']!='completed':raise ValueError('CART 模型尚未完整训练成功')
        from .cart_training import validate_model_geometry
        validate_model_geometry(json.loads((folder/'config.json').read_text()))
        return folder
    def activate(self,key):
        self.model_folder(key);write_json(self.root/'active.json',dict(model=key));self.selected=key
    def delete_assets(self,kind,ids,before_delete=None):
        with self.lock:
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
        result=dict(protocol=PROTOCOL,stages=STAGES,stage=self.stage_id,split=self.split,manual=self.manual,
            recording=self.recorder.snapshot()['recording'],selected_model=self.selected,replay_active=self.replay_active,
            preview_age_ms=age,error=self.error,ready=bool(connected and age is not None and age<3000),
            hull=preview.get('hull'),motion=self.motion,geometric_motion=preview.get('motion'),session_id=self.session_id)
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
        self.closed=True;self.stop_preview();self.recorder.close()
        deadline=time.monotonic()+5
        while self.recorder.recording.get('saving') and time.monotonic()<deadline:time.sleep(.02)
