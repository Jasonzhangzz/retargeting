#!/usr/bin/env python3
"""Fetch the exact official CUDA 12.1 wheel with bounded parallel HTTP ranges."""
import concurrent.futures,hashlib,json,os,re,sys,time,urllib.request
from pathlib import Path
from html.parser import HTMLParser
filename='torch-2.5.1%2Bcu121-cp311-cp311-linux_x86_64.whl'
class Links(HTMLParser):
    links=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key=='href' and filename in value:self.links.append(value)
index='https://download.pytorch.org/whl/cu121/torch/'
p=Links();p.feed(urllib.request.urlopen(index,timeout=60).read().decode())
assert len(p.links)==1,p.links
url,fragment=p.links[0].split('#');expected=fragment.removeprefix('sha256=')
root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
output=root/urllib.parse.unquote(filename)
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while data:=f.read(8*1024*1024):h.update(data)
    return h.hexdigest()
if output.exists() and digest(output)==expected:sys.exit(0)
req=urllib.request.Request(url,headers={'Range':'bytes=0-0'})
with urllib.request.urlopen(req,timeout=60) as r:
    total=int(r.headers['Content-Range'].split('/')[-1])
partial=output.with_suffix('.partial');fd=os.open(partial,os.O_CREAT|os.O_RDWR,0o600);os.ftruncate(fd,total)
size=16*1024*1024
chunks=[(start,min(total-1,start+size-1)) for start in range(0,total,size)]
def fetch(bounds):
    start,end=bounds
    for attempt in range(3):
        try:
            request=urllib.request.Request(url+'?range_start='+str(start),headers={'Range':f'bytes={start}-{end}'})
            with urllib.request.urlopen(request,timeout=120) as response:
                assert response.headers.get('Content-Range')==f'bytes {start}-{end}/{total}',dict(response.headers)
                pos=start
                while block:=response.read(256*1024):
                    os.pwrite(fd,block,pos);pos+=len(block)
                assert pos==end+1,(pos,end)
            return end-start+1
        except Exception:
            if attempt==2:raise
            time.sleep(1)
try:
    completed=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for count in pool.map(fetch,chunks):
            completed+=count;print('downloaded',completed,total,flush=True)
finally:os.close(fd)
actual=digest(partial)
if actual!=expected:raise RuntimeError(f'Wheel checksum mismatch: {actual} != {expected}')
partial.replace(output)
(root/'torch_receipt.json').write_text(json.dumps(dict(url=url,index=index,sha256=actual,bytes=total),indent=2))
print('verified',str(output),actual,flush=True)
