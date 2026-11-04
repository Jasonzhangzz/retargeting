#!/usr/bin/env python3
"""Download exact pinned wheels from PyPI, checking official SHA256 receipts."""
from pathlib import Path
import concurrent.futures,email,hashlib,json,os,re,sys,time,urllib.request,zipfile
try:
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename
except ImportError:
    from pip._vendor.packaging.tags import sys_tags
    from pip._vendor.packaging.utils import parse_wheel_filename
compatible=set(sys_tags())
root=Path(sys.argv[1]);wheels=root/'wheels';wheels.mkdir(exist_ok=True)
requirements={}
for line in (root/'requirements-graspadmm.txt').read_text().splitlines():
    match=re.match(r'([\w-]+)(?:\[[^\]]+\])?==([^\s]+)',line)
    if match and match[1]!='torch':requirements[match[1]]=match[2]
torch=next(wheels.glob('torch-*.whl'))
with zipfile.ZipFile(torch) as archive:
    name=next(n for n in archive.namelist() if n.endswith('.dist-info/METADATA'))
    metadata=email.message_from_bytes(archive.read(name))
    for requirement in metadata.get_all('Requires-Dist'):
        match=re.match(r'([\w-]+)\s*\(?==([^\s;)]+)',requirement)
        if match and (match[1].startswith('nvidia-') or match[1]=='triton'):requirements[match[1]]=match[2]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as file:
        while data:=file.read(8*1024*1024):h.update(data)
    return h.hexdigest()
def metadata_for(item):
    name,version=item
    url=f'https://pypi.org/pypi/{name}/{version}/json'
    data=json.load(urllib.request.urlopen(url,timeout=60))
    candidates=[]
    for entry in data['urls']:
        fn=entry['filename']
        if entry['packagetype']!='bdist_wheel':continue
        if not parse_wheel_filename(fn)[3].intersection(compatible):continue
        candidates.append(entry)
    if not candidates:raise RuntimeError('No compatible wheel for '+name)
    entry=sorted(candidates,key=lambda x:x['filename'])[0]
    entry['metadata_url']=url
    entry['download_url']=entry['url'].replace('https://files.pythonhosted.org/','https://mirrors.aliyun.com/pypi/')
    return entry
entries=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
    entries=list(pool.map(metadata_for,requirements.items()))
requests=[];handles={};pending=[]
for entry in entries:
    path=wheels/entry['filename'];expected=entry['digests']['sha256']
    if path.exists() and digest(path)==expected:continue
    partial=path.with_suffix('.partial');fd=os.open(partial,os.O_CREAT|os.O_RDWR,0o600);os.ftruncate(fd,entry['size'])
    handles[path]=fd;pending.append((entry,path,partial))
    step=8*1024*1024
    for start in range(0,entry['size'],step):requests.append((entry,path,start,min(entry['size']-1,start+step-1)))
def fetch(job):
    entry,path,start,end=job;url=entry['download_url']
    for attempt in range(3):
        try:
            request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
            with urllib.request.urlopen(request,timeout=120) as response:
                if response.status==200:
                    if start!=0 or end!=entry['size']-1:raise RuntimeError('HTTP range unsupported: '+entry['filename'])
                else:
                    assert response.headers.get('Content-Range')==f'bytes {start}-{end}/{entry["size"]}'
                position=start
                while block:=response.read(256*1024):
                    if position+len(block)>end+1:raise RuntimeError('Range overflow')
                    os.pwrite(handles[path],block,position);position+=len(block)
                assert position==end+1
            return end-start+1
        except Exception:
            if attempt==2:raise
            time.sleep(1)
try:
    count=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        futures=[pool.submit(fetch,job) for job in requests]
        for future in concurrent.futures.as_completed(futures):
            done=future.result()
            count+=done
            if count//(64*1024*1024)!=(count-done)//(64*1024*1024):print('dependency_bytes',count,flush=True)
finally:
    for fd in handles.values():os.close(fd)
for entry,path,partial in pending:
    actual=digest(partial)
    if actual!=entry['digests']['sha256']:raise RuntimeError('Hash mismatch: '+str(path))
    partial.replace(path);print('verified',path.name,flush=True)
(wheels/'dependency_receipts.json').write_text(json.dumps(entries,indent=2))
