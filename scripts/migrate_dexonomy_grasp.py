#!/usr/bin/env python3
"""Reuse only original grasp targets from verified three-target static shards."""
import sys,json,gzip,time,concurrent.futures,multiprocessing
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from retarget.env.grasp.dexonomy_grasp_static import write_shard,write_json

def migrate(args):
    old,new=args;old=Path(old);manifest=json.loads((old/'manifest.json').read_text())
    typ=manifest['cases'][0]['grasp_type'];part=int(old.name)
    target=Path(new)/typ/'shards'/old.name
    if (target/'verification.json').exists():
        return {'type':typ,'part':part,**json.loads((target/'summary.json').read_text()),'path':str(target)}
    with np.load(old/'poses.npz',allow_pickle=False) as raw:poses={k:raw[k] for k in raw.files}
    with np.load(old/'contacts.npz',allow_pickle=False) as raw:contact={k:raw[k] for k in raw.files}
    metrics=json.loads(gzip.decompress((old/'metrics.json.gz').read_bytes()))
    refs=json.loads((old/'source_records.json').read_text());failures=json.loads((old/'failures.json').read_text())
    dataset=manifest['dataset_root'];samples=[]
    for i,entry in enumerate(manifest['cases']):
        a,b=map(int,contact['offsets'][3*i+1:3*i+3])
        obs={k:v[a:b] for k,v in contact.items() if k not in {'offsets','stage_names','body_names'}}
        samples.append({'source_stage_hand_qpos':poses['source_stage_hand_qpos'][i,1:2],
            'stage_qpos':poses['stage_qpos'][i,1:2],'object_pose_world':poses['object_pose_world'][i],
            'object_pose_in_hand':poses['object_pose_in_hand'][i],'object_scale':poses['object_scale'][i],
            'scene_scale':float(poses['scene_scale'][i]),'contacts':[obs],
            'penetration_m':np.asarray(metrics[i]['stage_penetration_m'][1:2]),
            'joint_limit_violation_rad':np.asarray(metrics[i]['stage_joint_limit_violation_rad'][1:2]),
            'body_names':contact['body_names'],'scene_path':entry['scene_path'],'dataset':dataset,
            'source_ref':{'archive_member':entry['object']['source_record'],'row':entry['object']['source_row']}})
    result=write_shard(new,typ,part,samples,failures,refs)
    result['reused_geometry_without_recomputation']=True
    return result

def main():
    old=ROOT/'results/dexonomy_shadow_static_full';new=ROOT/'results/dexonomy_shadow_grasp_full';new.mkdir(exist_ok=True)
    shards=[p.parent for p in sorted(old.glob('*/shards/*/verification.json'))]
    status={'state':'migrating_completed_grasp_targets','samples_cached':0,'shards_completed':0,'errors':0,'by_type':{},'stage_names':['grasp'],'source_shards':len(shards)}
    started=time.time()
    with (new/'shards.jsonl').open('w') as journal,concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        for result in pool.map(migrate,[(str(p),str(new)) for p in shards]):
            journal.write(json.dumps(result)+'\n');journal.flush()
            status['samples_cached']+=result['samples'];status['shards_completed']+=1;status['errors']+=result['errors']
            d=status['by_type'].setdefault(result['type'],{'samples':0,'shards':0,'errors':0});d['samples']+=result['samples'];d['shards']+=1;d['errors']+=result['errors']
            status['elapsed_seconds']=time.time()-started;write_json(new/'status.json',status)
            if status['shards_completed']%25==0:print(json.dumps(status),flush=True)
    status['state']='migration_completed';write_json(new/'migration.json',status)
    write_json(new/'status.json',status);print(json.dumps(status),flush=True)
if __name__=='__main__':main()
