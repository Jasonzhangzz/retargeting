import concurrent.futures,json,os,tarfile,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'data/dexonomy';out=root/'extracted';out.mkdir(exist_ok=True)
def extract(name):
 marker=out/(name+'.complete.json')
 if marker.exists():return
 n=0;start=time.time()
 with tarfile.open(root/name,'r|gz') as t:
  for m in t:
   target=(out/m.name).resolve()
   if not target.is_relative_to(out.resolve()) or m.issym() or m.islnk():raise ValueError('Unsafe archive member: '+m.name)
   if not (m.isdir() or m.isfile()):continue
   if m.isfile() and target.exists() and target.stat().st_size==m.size:continue
   if m.isdir():target.mkdir(parents=True,exist_ok=True)
   else:
    target.parent.mkdir(parents=True,exist_ok=True)
    with t.extractfile(m) as src,target.with_suffix(target.suffix+'.pending').open('wb') as dst:
     import shutil;shutil.copyfileobj(src,dst)
    target.with_suffix(target.suffix+'.pending').replace(target)
   n+=1
   if n%5000==0:print(name,n,round(time.time()-start),flush=True)
 marker.write_text(json.dumps({'archive':name,'members_written':n,'seconds':time.time()-start}))
 print('COMPLETE',name,n,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(extract,['Dexonomy_GRASP_shadow.tar.gz','DGN_5k_processed.tar.gz','objaverse_5k_processed.tar.gz']))
