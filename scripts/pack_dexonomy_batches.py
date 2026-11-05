#!/usr/bin/env python3
import argparse
import fcntl
import json
from pathlib import Path
import sqlite3
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.mano_campaign_inputs import source_groups,read,write,stamp,load_npz,save_npz


def pack(db,typ,count,output):
    pending=db.execute('select path,cursor,n from sources where typ=? and cursor<n order by rowid',(typ,)).fetchall()
    remaining=count;ranges=[]
    for path,start,n in pending:
        stop=min(n,start+remaining);ranges.append((path,start,stop));remaining-=stop-start
        if not remaining:break
    assert remaining==0
    part=db.execute('select count(*) from batches where typ=?',(typ,)).fetchone()[0]
    dest=output/typ/'shards'/f'{part:07d}';dest.mkdir(parents=True,exist_ok=True)
    entries=[];arrays={key:[] for key in ('grasp_qpos','object_pose_world','object_scale','scene_scale')}
    for path,start,stop in ranges:
        path=Path(path);cases=read(path/'manifest.json')['cases'][start:stop]
        for row,entry in enumerate(cases,start):
            entries.append(dict(entry,source_raw_manifest=str(path/'manifest.json'),source_raw_case=row))
        archive=load_npz(path/'poses.npz')
        for key in arrays:
            value=archive[key][start:stop] if key in archive else np.array([e[key] for e in cases])
            if key=='object_pose_world' and value.ndim==3:value=value[:,0]
            arrays[key].append(value)
    temporary=dest/'poses.tmp.npz';save_npz(temporary,**{k:np.concatenate(v) for k,v in arrays.items()});temporary.replace(dest/'poses.npz')
    write(dest/'manifest.json',dict(hand='shadow',task='grasp',cases=entries,source_ranges=ranges))
    group=dict(type=typ,part=part,path=str(dest.resolve()),samples=count)
    with db:
        for path,start,stop in ranges:db.execute('update sources set cursor=? where path=?',(stop,path))
        db.execute('insert into batches values(?,?,?,?)',(typ,part,count,json.dumps(group)))
    with (output/'shards.jsonl').open('a') as stream:stream.write(json.dumps(group)+'\n')
    total,batches=db.execute('select coalesce(sum(n),0),count(*) from batches').fetchone()
    write(output/'status.json',dict(state='running',samples=total,shards=batches,updated_utc=stamp()))
    print(json.dumps(group),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--batch-size',type=int,default=2048);parser.add_argument('--max-batches',type=int,default=0)
    a=parser.parse_args();a.source=a.source.resolve();a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    db=sqlite3.connect(a.output/'packing.sqlite')
    db.execute('create table if not exists sources(path text primary key,typ text,n integer,cursor integer default 0)')
    db.execute('create table if not exists batches(typ text,part integer,n integer,payload text,primary key(typ,part))')
    (a.output/'shards.jsonl').write_text(''.join(row[0]+'\n' for row in db.execute('select payload from batches order by rowid')))
    write(a.output/'configuration.json',dict(source=str(a.source),batch_size=a.batch_size,stage='input_packing_no_optimization'))
    while not (a.output/'STOP').exists():
        groups=source_groups(a.source)
        with db:
            db.executemany('insert or ignore into sources(path,typ,n) values(?,?,?)',[(str(g['path']),g['type'],g['samples']) for g in groups])
        producer=read(a.source/'status.json')
        closed=producer['state']=='completed'
        pending=db.execute('select typ,sum(n-cursor) from sources group by typ order by min(rowid)').fetchall()
        for typ,total in pending:
            while total>=a.batch_size or closed and total:
                size=min(total,a.batch_size);pack(db,typ,size,a.output);total-=size
                if a.max_batches and db.execute('select count(*) from batches').fetchone()[0]>=a.max_batches:
                    closed=True;break
            if a.max_batches and closed:break
        samples,batches=db.execute('select coalesce(sum(n),0),count(*) from batches').fetchone()
        write(a.output/'status.json',dict(state='completed' if closed else 'running',samples=samples,shards=batches,raw_state=producer['state'],updated_utc=stamp()))
        if closed:return
        time.sleep(20)
    write(a.output/'status.json',dict(state='stopped',updated_utc=stamp()))
if __name__=='__main__':main()
