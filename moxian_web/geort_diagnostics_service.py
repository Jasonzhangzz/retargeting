"""Bounded read-only diagnostic subprocess, independent of capture/inference."""
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from retarget.baseline.geort.adapter import ROOT


class DiagnosticService:
    def __init__(self,provider,root):
        self.provider=provider;self.root=Path(root);self.lock=threading.RLock()
        self.enabled=False;self.closed=threading.Event();self.proc=None;self.pending=None
        self.context_key=None;self.completed_key=None;self.sequence=0;self.result=None
        self.semantic_key=None
        self.received=0.;self.input_kind='live';self.reason=None;self.failure=None
        self.thread=threading.Thread(target=self._loop,daemon=True);self.thread.start()

    def configure(self,enabled):
        if not isinstance(enabled,bool):raise ValueError('enabled必须为布尔值')
        with self.lock:
            self.enabled=enabled;self.result=None;self.failure=None;self.reason=None
            self.completed_key=None
            if not enabled:self._stop_locked()
        return self.snapshot()

    def _stop_locked(self):
        proc=self.proc;self.proc=None;self.pending=None;self.context_key=None
        if proc and proc.poll() is None:proc.terminate()

    def _start(self):
        self.root.mkdir(parents=True,exist_ok=True)
        with (self.root/'diagnostics.log').open('a') as log:
            proc=subprocess.Popen([os.environ.get('MOXIAN_GEORT_PYTHON',str(ROOT/'.venv-contact/bin/python')),
                '-u','-m','retarget.baseline.geort.diagnostics','--worker'],cwd=ROOT,stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,
                env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        self.proc=proc;self.context_key=None
        threading.Thread(target=self._read,args=(proc,),daemon=True).start()
        return proc

    def _read(self,proc):
        try:
            for line in proc.stdout:
                value=json.loads(line)
                with self.lock:
                    if self.proc is not proc or not self.enabled:return
                    pending=self.pending
                    if not pending or value.get('request_id')!=pending['id']:continue
                    if pending['semantic']!=self.semantic_key:self.pending=None;continue
                    value['latency_ms']=(time.monotonic()-pending['started'])*1000
                    self.result=value;self.received=time.monotonic();self.completed_key=pending['key'];self.pending=None
                    if value.get('ready'):self.failure=None
                    else:self.failure=value.get('error','诊断失败');self._stop_locked()
        except (OSError,ValueError) as exc:
            with self.lock:
                if self.proc is proc:self.failure=str(exc);self._stop_locked()
        finally:
            with self.lock:
                if self.proc is proc and self.enabled:
                    self.failure=self.failure or '诊断进程已退出，请查看 diagnostics.log';self._stop_locked()
            try:proc.wait(timeout=2)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            proc.stdout.close();proc.stdin.close()

    def _loop(self):
        while not self.closed.wait(.25):
            with self.lock:
                if not self.enabled or self.failure:continue
            try:
                packet=self.provider()
                with self.lock:
                    if not self.enabled:continue
                    if not packet:
                        self.reason='等待有效标定后的源网格';continue
                    self.reason=None;self.input_kind=packet.get('input_kind','live')
                    semantic=(packet['context_key'],packet['frame'].get('model_run_id'),bool(packet.get('robot')),self.input_kind)
                    if semantic!=self.semantic_key:self.semantic_key=semantic;self.result=None
                    if self.pending:
                        if time.monotonic()-self.pending['started']>60:
                            self.failure='诊断进程超时，请关闭后重新启动';self._stop_locked()
                        continue
                    key=(packet['context_key'],packet['timestamp_ms'],packet['frame'].get('model_run_id'),bool(packet.get('robot')))
                    if self.completed_key==key:continue
                    if packet['context_key']!=self.context_key:self.result=None
                    proc=self.proc
                    if proc and proc.poll() is not None:
                        self.failure='诊断进程已退出，请查看 diagnostics.log';self._stop_locked();continue
                    if proc is None:proc=self._start()
                    self.sequence+=1;packet=dict(packet,request_id=self.sequence)
                    if packet['context_key']==self.context_key:packet.pop('context',None)
                    self.pending=dict(id=self.sequence,key=key,semantic=semantic,started=time.monotonic())
                    self.context_key=packet['context_key']
                # Only this dedicated thread writes; one request in flight.
                proc.stdin.write(json.dumps(packet,ensure_ascii=False,allow_nan=False)+'\n');proc.stdin.flush()
            except Exception as exc:
                with self.lock:
                    if self.enabled:self.failure=str(exc);self._stop_locked()

    def snapshot(self):
        with self.lock:
            if not self.enabled:return dict(enabled=False,ready=False,status='disabled')
            age=(time.monotonic()-self.received)*1000 if self.received else None
            result=dict(self.result or {},enabled=True,age_ms=age)
            if self.failure:result.update(ready=False,status='error',error=self.failure)
            elif self.reason:result.update(ready=False,status='waiting_input',error=self.reason)
            elif not self.result:result.update(ready=False,status='starting')
            elif result.get('input_kind')=='live' and (age is None or age>1500 or time.time()*1000-result.get('timestamp_ms',0)>2000):
                result.update(ready=False,status='stale',error='诊断帧已过期')
            if not result.get('ready'):result.update(pairs=[],source=None,robot=None)
            return result

    def close(self):
        self.closed.set()
        with self.lock:self.enabled=False;self._stop_locked()
