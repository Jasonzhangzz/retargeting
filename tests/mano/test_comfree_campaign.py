from scripts.retarget_mano_comfree_campaign import make_jobs,key
import pytest


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


def test_larger_batches_cover_only_remaining_rows_and_keep_types_interleaved():
    groups=[dict(type='1_A',part=0,path='/a',samples=8),
            dict(type='2_B',part=0,path='/b',samples=8),
            dict(type='1_A',part=1,path='/c',samples=8)]
    completed=[dict(groups[0],offset=0,samples=2),dict(groups[0],offset=4,samples=2),
               dict(groups[1],offset=0,samples=4)]
    jobs=list(make_jobs(groups,4,completed))
    assert [j['type'] for j in jobs[:2]]==['1_A','2_B']
    assert [j['samples'] for j in jobs[:2]]==[4,4]
    assert max(j['samples'] for j in jobs)==4
    for source in groups:
        rows=[row for job in [*completed,*jobs] if job['path']==source['path']
              for row in range(job['offset'],job['offset']+job['samples'])]
        assert sorted(rows)==list(range(source['samples']))
        assert len(rows)==len(set(rows))


def test_completed_intervals_cannot_overlap_or_change_source():
    group=dict(type='1_A',part=0,path='/a',samples=8)
    with pytest.raises(ValueError,match='Overlapping'):
        list(make_jobs([group],4,[dict(group,offset=0,samples=4),dict(group,offset=2,samples=4)]))
    with pytest.raises(ValueError,match='mismatch'):
        list(make_jobs([group],4,[dict(group,path='/other',offset=0,samples=4)]))
    assert list(make_jobs([group],4,[dict(group,offset=0,samples=8)]))==[]


def test_inherited_receipts_keep_real_row_counts_and_detect_changed_results(tmp_path):
    import json
    from scripts.retarget_mano_comfree_campaign import inherited_results
    from scripts.mano_campaign_inputs import write,sha
    prior=tmp_path/'old';new=tmp_path/'new';prior.mkdir()
    config=dict(output=str(new),source='/source',model='/model',mapping='/mapping',
                shadow_xml='/shadow',dataset='/dataset',seed=7,rollout={'gravity':(0.,0.,-9.81)},model_sha256='model',
                mapping_sha256='mapping',algorithm_parameters={'resources':{'batch_size':4},'sampling':{'iterations':24,'scale':(1.,2.)}},
                code_sha256={'scripts/retarget_mano_comfree_campaign.py':'new','score.py':'same'})
    old=dict(config,output=str(prior),config_sha256='old',algorithm_parameters={**config['algorithm_parameters'],'resources':{'batch_size':2}})
    write(prior/'configuration.json',old);write(prior/'status.json',{'state':'stopped'})
    folder=prior/'1_A/0000000/0000000';folder.mkdir(parents=True)
    job=dict(type='1_A',part=0,path='/source/shard',offset=0,samples=2)
    write(folder/'MANIFEST.json',{'job':job});(folder/'final.npz').write_bytes(b'final')
    write(folder/'verification.json',dict(success=True,config_sha256='old',samples=2,stable=1,geometry_accepted=0,
        files={name:sha(folder/name) for name in ('MANIFEST.json','final.npz')}))
    result=inherited_results(prior,config)
    assert result[key(job)]['samples']==2
    assert result[key(job)]['job']==job
    incompatible={**config,'rollout':{'horizon':6}}
    with pytest.raises(ValueError,match='rollout'):inherited_results(prior,incompatible)
    write(prior/'status.json',{'state':'running'})
    with pytest.raises(ValueError,match='not stopped'):inherited_results(prior,config)
    write(prior/'status.json',{'state':'stopped'});(folder/'final.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Changed completed'):inherited_results(prior,config)
