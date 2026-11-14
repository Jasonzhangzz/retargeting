#!/usr/bin/env python3
"""Freeze source/asset hashes and choose disjoint continuous held-out records."""
import argparse,hashlib,json,shutil,random
from pathlib import Path
import numpy as np
import h5py


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True);a=p.parse_args();r=a.root
 source_hash={}
 for name in ['retarget_reference','human_ik_reference','emf_retarget']:
  dest=r/'reference'/name
  if not dest.exists():shutil.copytree(a.source/name,dest,ignore=shutil.ignore_patterns('__pycache__'))
  for file in (a.source/name).rglob('*'):
   if file.is_file() and '__pycache__' not in str(file):source_hash[str(file.relative_to(a.source))]=sha(file)
 for name in ['assets','sdk831']:
  dest=r/name
  if not dest.exists():dest.symlink_to((a.previous/name).resolve(),target_is_directory=True)
 old=json.loads((a.previous/'selection.json').read_text())['trajectories'];records=[]
 for item in old:
  records.append(dict(index=item['index'],split='development',source_h5=item['source_h5'],cached_h5=item['cached_h5'],source_sha256=item['h5_sha256'],collection=item['collection'],sides=['left','right'],start=0,end=None,metadata=item['recording_metadata']))
 prepared=json.loads((a.prepared/'manifest.json').read_text());selection=json.loads(Path(prepared['source_metadata']).read_text());segments=selection['segments']
 train_hashes={s['source_sha256'] for s in segments if s['split']=='train'};train_paths={s['source_h5'] for s in segments if s['split']=='train'}
 excluded={v['source_sha256'] for v in records};pool={}
 for s in segments:
  if s['split']!='val' or s['source_sha256'] in train_hashes|excluded or s['source_h5'] in train_paths:continue
  pool.setdefault(s['source_sha256'],s)
 candidates=list(pool.values());rng=random.Random(42);rng.shuffle(candidates)
 devices=sorted({s['device'] for s in candidates});chosen=[];collections=set();used=set()
 # Round-robin dataset, side and length quartile; do not inspect output errors.
 lengths=sorted(s['end']-s['context_start'] for s in candidates);cuts=np.quantile(lengths,[.25,.5,.75])
 for cycle in range(40):
  for device in devices:
   for side in ['right','left']:
    bucket=cycle%4
    matches=[s for s in candidates if s['device']==device and s['side']==side and s['collection_id'] not in collections and s['source_sha256'] not in used and int(np.searchsorted(cuts,s['end']-s['context_start']))==bucket]
    if matches:
     s=matches[0];chosen.append(s);collections.add(s['collection_id']);used.add(s['source_sha256'])
     if len(chosen)==20:break
   if len(chosen)==20:break
  if len(chosen)==20:break
 if len(chosen)<20:
  for s in candidates:
   if s['source_sha256'] not in used:chosen.append(s);used.add(s['source_sha256'])
   if len(chosen)==20:break
 if len(chosen)<20:
  # There are only 19 original validation source files. Add an unused sibling
  # recording, selected solely by finite/confident continuous EMF, never q error.
  for anchor in sorted(candidates,key=lambda v:v['source_h5']):
   for path in sorted(Path(anchor['source_h5']).parent.parent.glob('*/wuji_glove_offline.h5')):
    if str(path) in train_paths or any(str(path)==v['source_h5'] for v in chosen+records):continue
    digest=sha(path)
    if digest in train_hashes|excluded|used:continue
    side='left' if sum(v['side']=='left' for v in chosen)<sum(v['side']=='right' for v in chosen) else 'right'
    with h5py.File(path,'r',locking=False) as h:
     g=h[side];pos=g['emf_poses/position'][:];rot=g['emf_poses/orientation_xyzw'][:];conf=g['emf_poses/confidence'][:];ts=g['source_timestamp_ns'][:]
    valid=np.isfinite(pos).all((1,2))&np.isfinite(rot).all((1,2))&(np.linalg.norm(rot,axis=-1)>1e-8).all(1)&(conf>=.8).all(1)
    cuts_bad=np.r_[True,np.diff(ts)<=0];valid &= ~cuts_bad
    starts=np.flatnonzero(valid & ~np.r_[False,valid[:-1]]);ends=np.flatnonzero(valid & ~np.r_[valid[1:],False])+1
    if not len(starts) or max(ends-starts)<16:continue
    k=int(np.argmax(ends-starts));extra=dict(anchor,source_h5=str(path),source_sha256=digest,side=side,context_start=int(starts[k]),start=int(starts[k]),end=int(ends[k]),additional_unseen_record=True)
    chosen.append(extra);used.add(digest);break
   if len(chosen)==20:break
 if len(chosen)!=20:raise RuntimeError('Insufficient distinct held-out records: '+str(len(chosen)))
 for s in chosen:
  records.append(dict(index=len(records),split='heldout',source_h5=s['source_h5'],source_sha256=s['source_sha256'],collection=s['collection'],dataset=s['device'],sides=[s['side']],start=s['context_start'],end=s['end'],source_row_start=s['start'],sdk_reset_at='context_start',additional_unseen_record=s.get('additional_unseen_record',False),sn=s.get('calibration',{}).get('sn')))
 out=r/'inputs';out.mkdir(exist_ok=True)
 for item in records:
  path=Path(item.get('cached_h5',item['source_h5']));source_sha=sha(path)
  if source_sha!=item['source_sha256']:raise ValueError('Source hash changed '+str(path))
  for side in item['sides']:
   dest=out/f'trajectory_{item["index"]}_{side}.npz'
   if dest.exists():continue
   with h5py.File(path,'r',locking=False) as f:
    g=f[side];sl=slice(item['start'],item['end']);position=g['emf_poses/position'][sl];quaternion=g['emf_poses/orientation_xyzw'][sl];timestamps=g['source_timestamp_ns'][sl];confidence=g['emf_poses/confidence'][sl];seq=g['source_seq'][sl]
   if len(position)==0 or not np.isfinite(position).all() or not np.isfinite(quaternion).all() or np.any(np.linalg.norm(quaternion,axis=-1)<1e-8) or np.any(np.diff(timestamps)<=0):raise ValueError('Invalid continuous source '+str(path))
   deltas,counts=np.unique(np.diff(seq),return_counts=True);nominal=deltas[np.argmax(counts)]
   breaks=np.flatnonzero((np.diff(seq)!=nominal)|(np.diff(timestamps)>1.5*np.median(np.diff(timestamps))))+1
   segment_offsets=np.r_[0,breaks,len(position)]
   np.savez_compressed(dest,segment_offsets=segment_offsets,position=position,quaternion=quaternion,confidence=confidence,timestamp_ns=timestamps,source_seq=seq)
  item['verified_source_sha256']=source_sha
 assets={p.name:sha(p) for p in (r/'assets').glob('*.urdf')}
 report=dict(complete=True,seed=42,development_records=5,heldout_records=20,records=records,source_hashes=source_hash,asset_hashes=assets,training_hash_overlap=False,prepared_signature=prepared['signature'],policy='20 independent original held-out records, continuous context-start to segment-end; no frame or spike removal; not an unseen-model-validation claim')
 (r/'selection.json').write_text(json.dumps(report,indent=2));print('PREPARED',len(records),flush=True)


if __name__=='__main__':main()
