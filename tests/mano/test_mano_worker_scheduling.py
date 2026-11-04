"""Multi-process ownership and progress isolation without CUDA initialization."""
import pytest
from scripts.retarget_dexonomy_mano import worker_layout,shard_owner,active_path


def test_four_workers_per_gpu_cover_shards_and_resume_once(tmp_path):
    devices,layout=worker_layout('cuda:0,cuda:1,cuda:2,cuda:3',4)
    assert len(layout)==16
    keys=[f'7_Prismatic_3_Finger/{i}' for i in range(2000)]
    reused=set(keys[:700])
    assignments=[{k for k in keys if shard_owner(k,len(layout))==i} for i in range(len(layout))]
    assert set.union(*assignments)==set(keys)
    assert sum(map(len,assignments))==len(keys)
    assert all(assignments)
    assert sum(len(a&reused) for a in assignments)==len(reused)
    assert sum(len(a-reused) for a in assignments)==len(keys)-len(reused)
    paths=[active_path(tmp_path,i) for i in range(len(layout))]
    assert len(set(paths))==16
    for i,path in enumerate(paths):
        path.parent.mkdir(exist_ok=True);path.write_text(str(i))
    assert [p.read_text() for p in paths]==list(map(str,range(16)))
    assert all(layout.count(d)==4 for d in devices)


@pytest.mark.parametrize('devices,count',[('cuda:0',0),('cuda:0',-1),('cuda:0,cuda:00',4),('',4),('cpu',4)])
def test_invalid_worker_layout(devices,count):
    with pytest.raises(ValueError):worker_layout(devices,count)


def test_dynamic_queue_skips_reused_and_duplicate_journal_entries():
    from queue import Queue,Empty
    from scripts.retarget_dexonomy_mano import schedule_groups
    q=Queue();scheduled=set();reused={'T/0'}
    groups=[{'type':'T','part':i} for i in range(10)]
    schedule_groups(groups+groups,reused,scheduled,q)
    schedule_groups(groups+[{'type':'T','part':10}],reused,scheduled,q)
    claimed=[]
    while True:
        try:claimed.append(q.get_nowait()['part'])
        except Empty:break
    assert claimed==list(range(1,11))
    assert scheduled=={'T/'+str(i) for i in range(1,11)}
