"""HTTP-side serialized RPC to an independent, read-only ik viewer process."""
import json
import os
from pathlib import Path
import queue
import subprocess
import threading

from retarget.baseline.geort.adapter import ROOT
from .geort_service import safe_id


class IKViewService:
    def __init__(self,root):
        self.root=Path(root);self.lock=threading.RLock();self.proc=None;self.key=None;self.signature=None

    def close(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:self.proc.terminate()
            self.proc=None;self.key=None;self.signature=None

    def request(self,key,index=None):
        key=safe_id(key)
        folder=self.root/'recordings'/key
        if folder.is_symlink() or folder.resolve().parent!=(self.root/'recordings').resolve():
            raise ValueError('无效的 IK 会话路径')
        manifest=folder/'manifest.json'
        if not manifest.is_file() or not json.loads(manifest.read_text()).get('complete'):
            raise ValueError('请先停止并保存该录制会话')
        if index is not None and (isinstance(index,bool) or not isinstance(index,int) or index<0):
            raise ValueError('定位帧号必须为非负整数')
        # Reload only when the immutable IK cache or source identity changes.
        if not (folder/'cart_ik.json').exists():raise ValueError('请先准备观测、触点与 IK')
        signature=tuple((folder/name).stat().st_mtime_ns for name in ('manifest.json','profile.json','topology.json','cart_ik.json','cart_ik.npz'))
        with self.lock:
            if self.key!=key or self.signature!=signature or not self.proc or self.proc.poll() is not None:
                self.close();self.root.mkdir(parents=True,exist_ok=True)
                log=(self.root/'ik-view.log').open('a')
                python=os.environ.get('MOXIAN_CART_PYTHON',str(ROOT/'.venv-contact/bin/python'))
                self.proc=subprocess.Popen([python,'-u','-m','moxian_web.cart_ik_view','--folder',str(folder)],
                    cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,
                    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
                log.close();self.key=key;self.signature=signature;self.responses=queue.Queue()
                proc=self.proc;responses=self.responses
                def reader():
                    try:
                        for line in proc.stdout:responses.put(json.loads(line))
                    except (OSError,ValueError) as exc:responses.put(dict(error=str(exc)))
                    finally:responses.put(dict(error='IK 回放进程已退出，请查看 ik-view.log'))
                threading.Thread(target=reader,daemon=True).start()
            try:
                self.proc.stdin.write(json.dumps(dict(topology=index is None,index=index))+'\n');self.proc.stdin.flush()
                result=self.responses.get(timeout=45)
            except (OSError,queue.Empty) as exc:
                self.close();raise ValueError('IK 回放进程无响应，请查看 ik-view.log') from exc
            if result.get('error'):raise ValueError(result['error'])
            return result
