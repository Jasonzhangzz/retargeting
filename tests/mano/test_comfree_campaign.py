from scripts.retarget_mano_comfree_campaign import make_jobs,key


def test_batches_interleave_types_and_preserve_every_source_row():
    groups=[dict(type='7_A',part=0,path='/a',samples=129),dict(type='1_B',part=0,path='/b',samples=70),
            dict(type='7_A',part=1,path='/c',samples=4)]
    jobs=list(make_jobs(groups,64))
    assert [j['type'] for j in jobs]==['1_B','7_A','1_B','7_A','7_A','7_A']
    assert sum(j['samples'] for j in jobs)==203
    assert len({key(j) for j in jobs})==len(jobs)
    for source in groups:
        selected=[j for j in jobs if j['path']==source['path']]
        rows=[r for j in selected for r in range(j['offset'],j['offset']+j['samples'])]
        assert rows==list(range(source['samples']))
