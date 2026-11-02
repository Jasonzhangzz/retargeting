"""Whole-type replacement retains every seed and never chooses better outcomes."""
import copy
import json
from pathlib import Path

import pytest

from retarget.common.results import write_json
from retarget.env.grasp.primitives import CATALOG,size_specifications
from scripts import merge_graspadmm_batches as merge


def make_source(path,jobs,shared,*,policy='upstream',success=True):
    runtime=path/'runtime';vendor=runtime/'third_party/Dexonomy/core.py'
    vendor.parent.mkdir(parents=True);vendor.write_text('original vendor')
    adapter=runtime/'retarget/bridge.py';adapter.parent.mkdir();adapter.write_text(policy)
    plan=dict(runtime=str(runtime),protocol=merge.PROTOCOL,workers=6,physical_model='wuji2',
        numerical_stack=dict(python='3.12',torch='fixture'),input_policy=policy,
        source_hashes={'third_party/Dexonomy/core.py':merge.digest(vendor),
                       'retarget/bridge.py':merge.digest(adapter)},
        input_hashes={str(shared):merge.digest(shared)},jobs=jobs,skipped=[dict(type_id=19),dict(type_id=21)],
        trial_policy=dict(epoch=1,n_final=1,template_feedback=False,failure_resampling=False))
    write_json(path/'plan.json',plan)
    for job in jobs:
        work=path/'jobs'/job['id']/'attempt_000';case=work/'solve/case'
        write_json(work.parent/'status.json',dict(state='completed',work=str(work),wall_seconds=2.))
        write_json(work/'solve/manifest.json',dict(cases=[dict(name='case',method='graspadmm',task='grasp',hand='wuji2',
            template=job['template'],summary='case/summary.json',model='case/scene.xml',trajectory='case/trajectory.npz')]))
        write_json(case/'summary.json',dict(task_success=success,evaluation_protocol=merge.PROTOCOL,
            init_seconds=.5,solve_seconds=.5,total_wall_seconds=1.,seed=job['seed']))
        (case/'scene.xml').write_text('<mujoco/>');(case/'trajectory.npz').write_bytes(b'fixture')
    return path


@pytest.fixture(scope='module')
def source_pair(tmp_path_factory):
    root=tmp_path_factory.mktemp('whole_type_merge');shared=root/'hand.xml';shared.write_text('same hand')
    jobs=[]
    for type_id,families in CATALOG.items():
        for index,p in enumerate(size_specifications(type_id,families[0][0])):
            for seed in range(10):
                jobs.append(dict(id=f't{type_id:02d}_{p["kind"]}_s{index:02d}_seed{seed:02d}',
                    type_id=type_id,size_index=index,seed=seed,template=f'{type_id}_fixture',primitive=p))
    base=make_source(root/'base',jobs,shared)
    replacement=make_source(root/'replacement',[j for j in jobs if j['type_id']==9],shared,
        policy='taxonomy',success=False)
    return base,replacement


def test_complete_replacement_is_used_even_when_all_new_outcomes_are_worse(source_pair,tmp_path):
    base,replacement=source_pair
    before={p:p.read_bytes() for source in source_pair for p in source.rglob('*.json')}
    output=tmp_path/'combined'
    summary=merge.merge_batches(base,replacement,output)
    assert summary['planned']==summary['finished']==1610
    assert summary['successes']==1560 and summary['success_rate']==pytest.approx(1560/1610)
    assert len(summary['by_type_size'])==161 and all(row['trials']==10 for row in summary['by_type_size'])
    for row in summary['jobs']:
        assert Path(row['work']).is_absolute()
        assert row['source_batch']==str(replacement if row['type_id']==9 else base)
        assert row['task_success'] is (row['type_id']!=9)
    manifest=json.loads((output/'manifest.json').read_text())
    assert len(manifest['cases'])==1610
    assert all(Path(c[field]).is_absolute() for c in manifest['cases'] for field in ('summary','model','trajectory'))
    plan=json.loads((output/'plan.json').read_text())
    assert plan['combined_experiment'] and plan['resumable'] is False and plan['workers']==6
    assert [s['input_policy'] for s in plan['source_runs']]==['upstream','taxonomy']
    assert [s['selected_trial_count'] for s in plan['source_runs']]==[1560,50]
    assert [s['timings_all_source_trials']['trials'] for s in plan['source_runs']]==[1610,50]
    assert [s['timings_selected_trials']['trials'] for s in plan['source_runs']]==[1560,50]
    assert all(p.read_bytes()==content for p,content in before.items())
    with pytest.raises(ValueError,match='new directory'):
        merge.merge_batches(base,replacement,output)


@pytest.fixture(scope='module')
def loaded_pair(source_pair):
    return tuple(merge.load_source(path) for path in source_pair)


def test_missing_replacement_seed_rejects_entire_merge(loaded_pair):
    base,replacement=loaded_pair;replacement=copy.deepcopy(replacement)
    replacement['rows'].pop(next(iter(replacement['rows'])))
    with pytest.raises(ValueError,match='every size/seed'):
        merge.validate_pair(base,replacement)


@pytest.mark.parametrize('field,value,message',[
    ('protocol','other','graspadmm_native_v1'),('numerical_stack',{'torch':'other'},'numerical stacks'),
    ('physical_model','wuji1','wuji2')])
def test_protocol_stack_and_hand_mismatch_rejected(loaded_pair,field,value,message):
    base,replacement=loaded_pair;replacement=copy.deepcopy(replacement);replacement['plan'][field]=value
    with pytest.raises(ValueError,match=message):merge.validate_pair(base,replacement)


def test_vendor_or_geometry_mismatch_rejected(loaded_pair):
    base,replacement=loaded_pair;changed=copy.deepcopy(replacement)
    changed['plan']['source_hashes']['third_party/Dexonomy/core.py']='other'
    with pytest.raises(ValueError,match='vendor hashes'):merge.validate_pair(base,changed)
    changed=copy.deepcopy(replacement);key=next(iter(changed['rows']))
    changed['rows'][key]['primitive']['height_m']=.2
    with pytest.raises(ValueError,match='geometry/template'):merge.validate_pair(base,changed)


def test_duplicate_and_nonterminal_trials_rejected_before_output(tmp_path):
    shared=tmp_path/'hand.xml';shared.write_text('hand');p=size_specifications(9)[0]
    job=dict(id='trial',type_id=9,size_index=0,seed=0,template='9_fixture',primitive=p)
    source=make_source(tmp_path/'single',[job],shared)
    plan=merge.read(source/'plan.json');plan['jobs'].append(job);write_json(source/'plan.json',plan)
    with pytest.raises(ValueError,match='Duplicate'):merge.load_source(source)
    plan['jobs']=[job];write_json(source/'plan.json',plan)
    status=source/'jobs/trial/status.json';row=merge.read(status);row['state']='running';write_json(status,row)
    with pytest.raises(ValueError,match='Nonterminal'):merge.load_source(source)


def test_timeout_and_failed_trials_are_kept_in_denominator(tmp_path):
    shared=tmp_path/'hand.xml';shared.write_text('hand');p=size_specifications(9)[0]
    jobs=[dict(id=f'trial{s}',type_id=9,size_index=0,seed=s,template='9_fixture',primitive=p) for s in range(2)]
    source=make_source(tmp_path/'failures',jobs,shared)
    for job,state in zip(jobs,['timeout','failed']):
        path=source/'jobs'/job['id']/'status.json';status=merge.read(path);status['state']=state;write_json(path,status)
    loaded=merge.load_source(source);summary=merge.aggregate(list(loaded['rows'].values()))
    assert summary['planned']==summary['finished']==2 and summary['successes']==0
    assert summary['by_type_size'][0]['trials']==2


def test_incomplete_main_sweep_is_not_accepted(loaded_pair):
    base=copy.deepcopy(loaded_pair[0]);base['rows'].pop(next(iter(base['rows'])))
    with pytest.raises(ValueError,match='complete 1610-trial'):merge.require_main_inventory(base)


def test_input_policy_is_explicit_and_mismatched_declaration_is_rejected(loaded_pair):
    source=copy.deepcopy(loaded_pair[1]);source['plan']['input_policy']='upstream'
    next(iter(source['rows'].values()))['primitive_seed_policy']='taxonomy'
    with pytest.raises(ValueError,match='input policy declaration conflicts'):
        merge.source_policy(source)


def test_legacy_policy_is_derived_from_verified_frozen_runner(tmp_path):
    runner=tmp_path/'runtime/retarget/env/grasp/grasp_admm_runner.py'
    runner.parent.mkdir(parents=True);runner.write_text('initial=initialize_one(hand)\n')
    source=dict(plan=dict(runtime=str(tmp_path/'runtime')),rows={})
    policy,evidence=merge.source_policy(source)
    assert policy=='upstream' and 'verified legacy frozen runner' in evidence
