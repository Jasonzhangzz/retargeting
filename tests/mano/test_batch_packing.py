import sqlite3
import numpy as np
from scripts.pack_dexonomy_batches import pack
from scripts.mano_campaign_inputs import write,read


def test_packing_crosses_shards_without_changing_or_repeating_poses(tmp_path):
    db=sqlite3.connect(':memory:')
    db.execute('create table sources(path text primary key,typ text,n integer,cursor integer default 0)')
    db.execute('create table batches(typ text,part integer,n integer,payload text,primary key(typ,part))')
    expected=[]
    for shard in range(2):
        path=tmp_path/f'raw{shard}';path.mkdir()
        q=np.arange(shard*9,(shard+1)*9).reshape(3,3)
        expected.extend(q)
        np.savez(path/'poses.npz',grasp_qpos=q,object_pose_world=np.zeros((3,1,7)),object_scale=np.ones((3,3)),scene_scale=np.ones(3))
        write(path/'manifest.json',dict(cases=[dict(name=f'{shard}-{i}') for i in range(3)]))
        db.execute('insert into sources(path,typ,n) values(?,?,?)',(str(path),'7_Test',3))
    pack(db,'7_Test',4,tmp_path/'packed')
    pack(db,'7_Test',2,tmp_path/'packed')
    outputs=[];names=[]
    for shard in range(2):
        path=tmp_path/'packed/7_Test/shards'/f'{shard:07d}'
        with np.load(path/'poses.npz') as archive:outputs.extend(archive['grasp_qpos'])
        names.extend(e['name'] for e in read(path/'manifest.json')['cases'])
    np.testing.assert_array_equal(outputs,expected)
    assert len(names)==len(set(names))==6
    assert db.execute('select sum(n-cursor) from sources').fetchone()[0]==0
