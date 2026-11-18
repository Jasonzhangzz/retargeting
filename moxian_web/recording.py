"""Bounded asynchronous recording of immutable same-instant snapshots.

The device thread only publishes shallow snapshots. Encoding and disk writes
never run in the serial owner or its command queue. Overflow stops explicitly
and preserves all already queued frames instead of silently dropping samples.
"""
import json,queue,threading,time
from datetime import datetime,timezone
from pathlib import Path
from .background_io import write_json_chunks


class SnapshotRecorder:
    def __init__(self,folder,capacity=128):
        self.folder=Path(folder);self.capacity=capacity;self.lock=threading.Lock()
        self.active=False;self.saving=False;self.frames=0;self.accepted=0;self.dropped=0
        self.path=None;self.started=0.;self.elapsed=0.;self.error=None;self.thread=None;self.pending=None

    def start(self):
        with self.lock:
            if self.active:return self._snapshot()
            if self.saving:raise ValueError('上一段录制正在后台保存，请稍候')
            self.folder.mkdir(parents=True,exist_ok=True)
            path=self.folder/(datetime.now(timezone.utc).strftime('glove_%Y%m%dT%H%M%S_%fZ')+'.jsonl')
            stream=path.open('x',encoding='utf-8',buffering=1024*1024)
            self.path=path;self.started=time.monotonic();self.elapsed=0.;self.frames=0;self.accepted=0;self.dropped=0;self.error=None
            self.active=True;self.saving=False;self.pending=queue.Queue(maxsize=self.capacity)
            self.thread=threading.Thread(target=self._write,args=(stream,self.pending),daemon=True,name='glove-recording');self.thread.start()
            return self._snapshot()

    def submit(self,payload):
        with self.lock:
            if not self.active:return False
            try:self.pending.put_nowait(payload);self.accepted+=1;return True
            except queue.Full:
                self.dropped+=1;self.active=False;self.saving=True;self.elapsed=time.monotonic()-self.started
                self.error='录制写入跟不上采样，已停止并保留已排队数据；1帧未写入'
                return False

    def stop(self):
        with self.lock:
            if self.active:self.elapsed=time.monotonic()-self.started;self.active=False;self.saving=True
            return self._snapshot()

    def close(self):
        self.stop()
        if self.thread:self.thread.join(timeout=30)
        if self.thread and self.thread.is_alive():raise TimeoutError('录制仍在后台保存')

    def _snapshot(self):
        return dict(active=self.active,saving=self.saving,frames=self.frames,queued_frames=self.accepted-self.frames,
                    dropped_frames=self.dropped,seconds=round(time.monotonic()-self.started if self.active else self.elapsed,1),
                    file=self.path.name if self.path else None,error=self.error)

    def snapshot(self):
        with self.lock:return self._snapshot()

    def _write(self,stream,pending):
        mano_version=wuji_version=None
        try:
            while True:
                try:payload=pending.get(timeout=.03)
                except queue.Empty:
                    with self.lock:
                        if not self.active:break
                    continue
                # Topology objects are immutable references, included once per
                # version in this session. The source timestamp stays unchanged.
                row=dict(payload)
                hand=row.get('hand_model') or {};version=(hand.get('geometry_version'),hand.get('binding_version'))
                if hand.get('_mesh_f32') is not None:
                    # Decode the immutable same-frame binary on this writer,
                    # never inside the serial/device owner's dispatch loop.
                    import array,struct,sys
                    hand=dict(hand);packed=hand.pop('_mesh_f32');nv,nj=struct.unpack_from('<II',packed,12)
                    a=array.array('f');a.frombytes(packed[128:])
                    if sys.byteorder!='little':a.byteswap()
                    values=list(a);hand['vertices_m']=[values[i:i+3] for i in range(0,nv*3,3)];hand['joints_m']=[values[i:i+3] for i in range(nv*3,(nv+nj)*3,3)];row['hand_model']=hand
                if not hand.get('ready') or version==mano_version:row['hand_model_topology']=None
                elif (row.get('hand_model_topology') or {}).get('ready'):mano_version=version
                wuji=row.get('wuji2') or {};version=wuji.get('topology_version',wuji.get('mapping_version'))
                if not version or version==wuji_version:row['wuji2_topology']=None
                elif (row.get('wuji2_topology') or {}).get('ready'):wuji_version=version
                write_json_chunks(stream,row,newline=True)
                with self.lock:self.frames+=1
                pending.task_done()
                if self.frames%50==0:stream.flush()
        except Exception as exc:
            with self.lock:self.error='录制保存失败：'+str(exc);self.active=False
        finally:
            try:stream.close()
            except Exception as exc:
                with self.lock:self.error='录制保存失败：'+str(exc)
            with self.lock:self.active=False;self.saving=False
            # Persistent failure/overflow evidence travels with the data file.
            if self.error and self.path:
                self.path.with_suffix('.status.json').write_text(json.dumps(self.snapshot(),ensure_ascii=False))
