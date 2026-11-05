#!/usr/bin/env python3
"""Extract a deterministic type/object/size-stratified subset from release archives."""
import concurrent.futures,hashlib,json,os,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'data/dexonomy';pilot=source/'pilot_dataset_stratified';target=pilot/'extracted';target.mkdir(parents=True,exist_ok=True)
for name in ['download_manifest.json','shadow_assets']:
 p=pilot/name
 if not p.exists():p.symlink_to(source/name,target_is_directory=(name=='shadow_assets'))
chosen={};start=time.time();seen=0
with tarfile.open(source/'Dexonomy_GRASP_shadow.tar.gz','r|gz') as t:
 for m in t:
  t.members.clear()
  if not m.isfile() or not m.name.endswith('.npy') or '/succ_collect/' not in m.name:continue
  parts=Path(m.name).parts;idx=parts.index('succ_collect');typ,obj=parts[idx+1:idx+3]
  group=chosen.setdefault(typ,{})
  if obj not in group:
   if len(group)>=2:
    worst=max(group,key=lambda x:hashlib.sha256(('7/'+typ+'/'+x).encode()).digest())
    if hashlib.sha256(('7/'+typ+'/'+obj).encode()).digest()>=hashlib.sha256(('7/'+typ+'/'+worst).encode()).digest():continue
    del group[worst]
   group[obj]={}
  candidates=group[obj]
  # Retain the smallest/largest available named scene per selected object.
  if len(candidates)>=2 and min(candidates)<=m.name<=max(candidates):continue
  candidates[m.name]=t.extractfile(m).read()
  if len(candidates)>2:
   mid=sorted(candidates)[1];del candidates[mid]
  seen+=1
  if seen%500==0:print('selected_updates',seen,'types_seen',len(chosen),'seconds',round(time.time()-start),flush=True)
print('scanned_shadow','types',len(chosen),'seconds',round(time.time()-start),flush=True)
# Decode only the small final selection using the restricted NumPy loader.
import sys
sys.path.insert(0,str(ROOT))
from retarget.env.grasp.dexonomy_io import load_record,write_json
needed_scenes=set();needed_objects=set();selected=[]
for typ,objects in chosen.items():
 for obj,files in objects.items():
  for name,data in files.items():
   path=target/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
   record=load_record(path);scene=str(record['scene_path']).split('assets/object/',1)[-1]
   needed_scenes.add(scene);needed_objects.add((Path(scene).parts[0],obj));selected.append({'archive_member':name,'sha256':hashlib.sha256(data).hexdigest()})
write_json(pilot/'selection.json',{'grasp_types':len(chosen),'records':selected,'seed':7,'objects':sorted(needed_objects)})

def extract_objects(archive):
 count=0
 with tarfile.open(source/archive,'r|gz') as t:
  for m in t:
   t.members.clear()
   if not m.isfile():continue
   parts=Path(m.name).parts
   take=m.name in needed_scenes or (len(parts)>2 and parts[1]=='processed_data' and (parts[0],parts[2]) in needed_objects)
   if not take:continue
   path=(target/m.name).resolve()
   if not path.is_relative_to(target.resolve()) or m.issym() or m.islnk():raise ValueError('Unsafe tar member')
   path.parent.mkdir(parents=True,exist_ok=True)
   with t.extractfile(m) as src,path.open('wb') as dst:
    import shutil;shutil.copyfileobj(src,dst)
   count+=1
 print('assets',archive,count,'seconds',round(time.time()-start),flush=True)
with concurrent.futures.ThreadPoolExecutor(2) as pool:list(pool.map(extract_objects,['DGN_5k_processed.tar.gz','objaverse_5k_processed.tar.gz']))
for scene in needed_scenes:
 if not (target/scene).is_file():raise FileNotFoundError(scene)
write_json(pilot/'ready.json',{'ready':True,'types':len(chosen),'scenes':len(needed_scenes),'seconds':time.time()-start})
print('PILOT_ASSETS_READY',flush=True)
