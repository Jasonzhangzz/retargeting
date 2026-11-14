import json
from pathlib import Path

import pytest

from scripts.grasp_campaign import parse_devices, run_campaign


def job_key(job):
    return job['id']


def run(tmp_path, monkeypatch, jobs, devices, **kw):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    config=dict(output=str(tmp_path),batch_size=4,candidate_chunk=2,gate_count=len(devices))
    return run_campaign(config=config,jobs=jobs,devices=devices,
        backend='campaign_test_backend:FakeWorker',job_key=job_key,**kw)


def test_real_spawn_assigns_every_pending_batch_once_and_counts_inherited(tmp_path,monkeypatch):
    jobs=[dict(id=f'{i:03d}',type='1_A',samples=4) for i in range(12)]
    inherited={'000':dict(type='1_A',success=True,samples=2,stable=1,geometry_accepted=0)}
    jobs[0]['samples']=2
    completed,failed=run(tmp_path,monkeypatch,jobs,['cuda:1','cuda:3'],completed=inherited)
    assert not failed and len(completed)==12
    files=list(tmp_path.glob('*/result.json'))
    assert len(files)==11 and not (tmp_path/'000').exists()
    saved=[json.loads(p.read_text()) for p in files]
    assert {x['device'] for x in saved}=={'cuda:1','cuda:3'}
    assert len({x['pid'] for x in saved})==2
    status=json.loads((tmp_path/'status.json').read_text())
    assert status['state']=='complete' and status['samples']==status['total_inputs']==46
    assert status['stable']==34 and status['geometry_accepted']==11
    assert status['by_type']=={'1_A':46}


def test_stop_saves_current_batch_without_starting_queued_batches(tmp_path,monkeypatch):
    jobs=[dict(id=str(i),type='1_A',samples=4,stop=i==0) for i in range(5)]
    completed,failed=run(tmp_path,monkeypatch,jobs,['cuda:1'])
    assert not failed and list(completed)==['0']
    assert len(list(tmp_path.glob('*/result.json')))==1
    status=json.loads((tmp_path/'status.json').read_text())
    assert status['state']=='stopped' and status['samples']==4 and status['total_inputs']==20


def test_abrupt_worker_death_is_failed_not_a_false_completion(tmp_path,monkeypatch):
    jobs=[dict(id='crash',type='1_A',samples=4,crash=True)]
    completed,failed=run(tmp_path,monkeypatch,jobs,['cuda:2'])
    assert failed and not completed
    status=json.loads((tmp_path/'status.json').read_text())
    assert status['state']=='failed' and status['samples']==0


def test_duplicate_device_or_job_is_rejected_before_work(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match='Distinct'):
        parse_devices('cuda:1,cuda:1')
    with pytest.raises(ValueError,match='Duplicate campaign job'):
        run(tmp_path,monkeypatch,[dict(id='x',type='1_A',samples=4)]*2,['cuda:1'])
