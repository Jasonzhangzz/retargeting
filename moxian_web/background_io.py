"""Bounded C encoding and ordered, atomic capture checkpoints.

Encode one immutable row at a time and yield between bounded blocks, avoiding
both a whole-session GIL hold and Python work on every individual scalar.
"""
import json,shutil,threading,time
from collections import deque
from pathlib import Path


def write_json_chunks(stream,value,newline=False,progress=None):
    # The C encoder is fast; bound each call to one row/field so a growing
    # multi-minute capture never holds the GIL for the entire document.
    encode=lambda v:json.dumps(v,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    def field(v,tracked=False):
        if isinstance(v,list) and len(v)>64:
            if tracked and progress:progress(0,len(v))
            yield '['
            for i,row in enumerate(v):
                if i:yield ','
                yield encode(row)
                if tracked and progress and ((i+1)%64==0 or i+1==len(v)):progress(i+1,len(v))
            yield ']'
        else:
            yield encode(v)
            if tracked and progress and isinstance(v,list):progress(len(v),len(v))
    def chunks():
        if isinstance(value,dict):
            yield '{'
            for i,(key,v) in enumerate(value.items()):
                if i:yield ','
                yield encode(key)+':'
                yield from field(v,key=='rows')
            yield '}'
        else:yield from field(value)
    size=0
    for part in chunks():
        stream.write(part);size+=len(part)
        if size>=262144:time.sleep(0);size=0
    if newline:stream.write('\n')


class CaptureCheckpoints:
    """One writer; coalesce pending current files, never discard history jobs."""
    def __init__(self):
        self.condition=threading.Condition();self.jobs=deque();self.active=False
        self.closed=False;self.error=None;self.saved=0;self.last_saved=None;self.current=None;self.thread=threading.Thread(target=self._run,daemon=True,name='capture-save');self.thread.start()

    def snapshot(self):
        with self.condition:
            current=dict(self.current) if self.current else None
            if current and self.active:current['elapsed_s']=time.monotonic()-current.pop('_started')
            elif current:current.pop('_started',None)
            return dict(active=self.active,pending=len(self.jobs),saved=self.saved,error=self.error,last_saved=self.last_saved,current=current)

    def submit(self,paths,document,history=False):
        paths=tuple(Path(p) for p in paths)
        with self.condition:
            if self.closed:raise RuntimeError('采集保存已关闭')
            if not history:
                self.jobs=deque(job for job in self.jobs if job[2] or job[0]!=paths)
            self.jobs.append((paths,document,history));self.condition.notify_all()

    def flush(self,timeout=30):
        end=time.monotonic()+timeout
        with self.condition:
            while self.jobs or self.active:
                left=end-time.monotonic()
                if left<=0:raise TimeoutError('采集记录仍在后台保存')
                self.condition.wait(min(left,.1))
            if self.error:raise OSError(self.error)

    def close(self):
        try:self.flush()
        finally:
            with self.condition:self.closed=True;self.condition.notify_all()
            self.thread.join(timeout=2)

    def _run(self):
        while True:
            with self.condition:
                while not self.jobs and not self.closed:self.condition.wait()
                if not self.jobs:return
                paths,document,history=self.jobs.popleft();self.active=True
                self.current=dict(phase='encoding',completed=0,total=len(document.get('rows',[])),files_completed=0,files_total=len(paths),history=history,_started=time.monotonic(),elapsed_s=0.,estimated_remaining_s=None)
            temporary=paths[0].with_suffix('.tmp')
            try:
                def update(done,total):
                    with self.condition:
                        elapsed=time.monotonic()-self.current['_started']
                        self.current.update(completed=done,total=total,estimated_remaining_s=elapsed*(total-done)/done if done else None)
                paths[0].parent.mkdir(parents=True,exist_ok=True)
                with temporary.open('w',encoding='utf-8',buffering=1024*1024) as stream:write_json_chunks(stream,document,progress=update)
                temporary.replace(paths[0])
                with self.condition:self.current.update(phase='copying',files_completed=1,estimated_remaining_s=None)
                # Encode once; aliases retain byte-identical consistent snapshots.
                for path in paths[1:]:
                    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
                    shutil.copyfile(paths[0],tmp);tmp.replace(path)
                    with self.condition:self.current['files_completed']+=1
                with self.condition:
                    self.saved+=1;self.error=None
                    if not history:self.last_saved=dict(paths=[str(p) for p in paths],samples=len(document.get('rows',[])),saved_at_ms=time.time()*1000)
            except Exception as exc:
                with self.condition:self.error=str(exc)
                temporary.unlink(missing_ok=True)
            finally:
                with self.condition:
                    self.current.update(phase='failed' if self.error else 'saved',elapsed_s=time.monotonic()-self.current['_started'],estimated_remaining_s=0.)
                    self.active=False;self.condition.notify_all()
