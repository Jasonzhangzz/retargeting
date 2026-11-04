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


def test_round_robin_interleaves_types_and_preserves_every_unique_batch():
    from scripts.retarget_dexonomy_mano import round_robin_groups
    groups = [dict(type=typ, part=part, samples=2048)
              for typ, count in [('14_Tripod', 2), ('7_Prismatic', 4), ('2_Small', 3)]
              for part in reversed(range(count))]
    ordered = round_robin_groups(groups + groups[:1])
    assert [(g['type'], g['part']) for g in ordered] == [
        ('2_Small', 0), ('7_Prismatic', 0), ('14_Tripod', 0),
        ('2_Small', 1), ('7_Prismatic', 1), ('14_Tripod', 1),
        ('2_Small', 2), ('7_Prismatic', 2), ('7_Prismatic', 3)]
    assert sum(g['samples'] for g in ordered) == 9 * 2048
    assert round_robin_groups([]) == []


def test_round_robin_excludes_committed_and_queued_batches_before_balancing():
    from queue import Queue
    from scripts.retarget_dexonomy_mano import schedule_groups
    q = Queue(); scheduled = {'7_A/0'}; reused = {'7_A/1', '14_B/0'}
    groups = [dict(type=typ, part=part) for typ in ('7_A', '14_B', '22_C') for part in range(3)]
    schedule_groups(groups, reused, scheduled, q)
    claimed = [q.get_nowait() for _ in range(q.qsize())]
    assert [(g['type'], g['part']) for g in claimed] == [
        ('7_A', 2), ('14_B', 1), ('22_C', 0), ('14_B', 2), ('22_C', 1), ('22_C', 2)]
    schedule_groups(groups, reused, scheduled, q)
    assert q.empty()
