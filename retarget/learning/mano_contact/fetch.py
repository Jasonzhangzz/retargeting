"""Fetch exact HACO auxiliary matrices using resumable HTTP range downloads."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import os
import tarfile
import threading
import time
import urllib.request
from .data import digest, write_json

URL='https://huggingface.co/datasets/dqj5182/haco-data/resolve/main/train/data/base_data.tar.gz'
SIZE=6133035809


def fetch_regressors(assets, cache, workers=8):
    assets=Path(assets);assets.mkdir(parents=True,exist_ok=True)
    wanted={'V_regressor_336.npy','V_regressor_84.npy'}
    if all((assets/n).exists() for n in wanted):return
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    archive=cache/'base_data.tar.gz.partial';fd=os.open(archive,os.O_RDWR|os.O_CREAT,0o600)
    os.ftruncate(fd,SIZE);chunk=16*1024*1024
    stop=threading.Event()
    def fetch(i):
        start=i*chunk;end=min(SIZE,start+chunk)-1;done=cache/f'{i:04d}.done'
        if done.exists():return
        offset=start
        for attempt in range(5):
            try:
                if stop.is_set():return
                request=urllib.request.Request(URL+f'?download=true&contact_part={i}&offset={offset}',headers={'Range':f'bytes={offset}-{end}'})
                with urllib.request.urlopen(request,timeout=60) as r:
                    expected=f'bytes {offset}-{end}/{SIZE}'
                    if r.status!=206 or r.headers.get('Content-Range')!=expected:
                        raise ValueError('Server did not honor byte range')
                    while offset<=end:
                        if stop.is_set():return
                        b=r.read(min(1024*1024,end-offset+1))
                        if not b:raise IOError('Truncated range')
                        written=0
                        while written<len(b):written+=os.pwrite(fd,b[written:],offset+written)
                        offset+=len(b)
                done.write_text('complete\n');return
            except Exception as error:
                print('range retry',i,attempt,type(error).__name__,str(error).splitlines()[0][:120],flush=True)
                if attempt==4:raise
                time.sleep(min(2**attempt,8))
    count=(SIZE+chunk-1)//chunk
    class Prefix:
        def __init__(self,limit):self.file=open(archive,'rb');self.remaining=limit
        def read(self,n=-1):
            n=self.remaining if n<0 else min(n,self.remaining)
            value=self.file.read(n);self.remaining-=len(value);return value
    def extract_prefix(limit):
        prefix=Prefix(limit)
        try:
            with tarfile.open(fileobj=prefix,mode='r|gz') as t:
                for member in t:
                    name=Path(member.name).name
                    if name in wanted:
                        value=t.extractfile(member).read()
                        if len(value)==member.size:
                            (assets/name).write_bytes(value);wanted.remove(name)
                            print('extracted exact HACO asset',name,flush=True)
                        if not wanted:return
        except (tarfile.ReadError,EOFError,OSError):pass
        finally:prefix.file.close()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(fetch,i) for i in range(count)];last_prefix=0
            for n,future in enumerate(as_completed(futures),1):
                try:future.result()
                except Exception:
                    stop.set()
                    for f in futures:f.cancel()
                    raise
                if n%10==0:print('HACO asset archive',n,count,flush=True)
                contiguous=0
                while (cache/f'{contiguous:04d}.done').exists():contiguous+=1
                limit=min(SIZE,contiguous*chunk)
                if limit-last_prefix>=32*1024*1024 or limit==SIZE:
                    extract_prefix(limit);last_prefix=limit
                if not wanted:
                    stop.set()
                    for f in futures:f.cancel()
                    break
    finally:os.close(fd)
    if wanted:raise ValueError(f'Missing upstream assets: {wanted}')
    write_json(assets/'haco_assets.json',dict(url=URL,archive_size=SIZE,files={n:digest(assets/n) for n in ['V_regressor_336.npy','V_regressor_84.npy']}))
    archive.unlink()
    for p in cache.glob('*.done'):p.unlink()
