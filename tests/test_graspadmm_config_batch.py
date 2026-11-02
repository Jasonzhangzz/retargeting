import json

import pytest
import yaml

from retarget.config.run import load
from retarget.solver.base import Settings, solve
from scripts import run_graspadmm_batch as batch


def test_native_defaults_and_dedicated_runner():
    values = Settings(type="graspadmm").graspadmm
    assert values["step"] == 50 and values["substep"] == 10
    assert values["rho_admm"] == 1000 and values["lr_obj"] == .0002
    assert values["init_epoch"] == values["init_n_final"] == 1
    with pytest.raises(ValueError, match="dedicated"):
        solve(None, Settings(type="graspadmm"))


@pytest.mark.parametrize("values", [{"step": True}, {"lr_obj": float("nan")},
    {"fix_hand_local_cp": 1}, {"opt_miu_coef": [-1, 0]}, {"unknown": 3}, {"device": "cuda:0"}])
def test_invalid_native_settings(values):
    with pytest.raises(ValueError):
        Settings(type="graspadmm", graspadmm=values)


def test_config_requires_grasp_wuji2_and_native_dynamics(tmp_path):
    raw = dict(task=dict(type="grasp", data="inputs"), solver=dict(type="graspadmm"),
               environment=dict(dynamics="graspadmm"))
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert load(path).solver.type == "graspadmm"
    for task, env in [(dict(type="dex", data="inputs"), raw["environment"]),
                      (dict(type="grasp", data="inputs", options=dict(hand_model="wuji1")), raw["environment"]),
                      (raw["task"], {})]:
        path.write_text(yaml.safe_dump(dict(raw, task=task, environment=env)))
        with pytest.raises(ValueError):
            load(path)


def test_full_inventory_has_1610_independent_single_candidate_trials():
    jobs, skipped = batch.sample_jobs()
    assert len(jobs) == len({j["id"] for j in jobs}) == 1610
    assert {r["type_id"] for r in skipped} == {19, 21}
    assert len({(j["type_id"], j["size_index"]) for j in jobs}) == 161
    assert {j["seed"] for j in jobs} == set(range(10))
    assert len([j for j in jobs if j["type_id"] == 10]) == 110


def test_failures_and_timeouts_stay_in_denominator(tmp_path):
    jobs, skipped = batch.sample_jobs(seeds=[0, 1, 2], types=[11])
    jobs = jobs[:3]
    batch.write_json(tmp_path/"plan.json", dict(jobs=jobs, skipped=skipped))
    for job, state in zip(jobs, ["completed", "failed", "timeout"]):
        batch.write_json(tmp_path/"jobs"/job["id"]/"status.json", dict(state=state))
    work = tmp_path/"jobs"/jobs[0]["id"]/"solve"
    batch.write_json(work/"case/summary.json", dict(task_success=True, evaluation_protocol=batch.PROTOCOL))
    batch.write_json(work/"manifest.json", dict(cases=[dict(name="case", summary="case/summary.json",
        trajectory="case/trajectory.npz", model="case/scene.xml")]))
    report = batch.refresh(tmp_path)
    assert report["planned"] == report["finished"] == 3
    assert report["successes"] == 1 and report["success_rate"] == pytest.approx(1/3)
    assert report["by_type_size"][0]["trials"] == 3
    assert json.loads((tmp_path/"manifest.json").read_text())["cases"]


def test_seed_iterators_are_materialized_for_every_size():
    jobs, _ = batch.sample_jobs(seeds=iter([2, 4]), types=[16])
    assert len(jobs) == 10
    assert all({j['seed'] for j in jobs if j['size_index'] == i} == {2,4} for i in range(5))


@pytest.mark.parametrize('seeds', [[], [0,0], [-1], [True]])
def test_invalid_seed_inventory_rejected(seeds):
    with pytest.raises(ValueError, match='seeds'):
        batch.sample_jobs(seeds=seeds)


def test_unknown_types_rejected_before_batch_creation():
    with pytest.raises(ValueError, match='Unknown grasp types'):
        batch.sample_jobs(types=[34])


@pytest.mark.parametrize('state,protocol,expected', [
    ('completed', batch.PROTOCOL, True), ('completed', None, False),
    ('running', batch.PROTOCOL, False), ('timeout', batch.PROTOCOL, False),
    ('failed', batch.PROTOCOL, False)])
def test_only_terminal_completed_shared_protocol_success_can_enter_numerator(tmp_path,state,protocol,expected):
    jobs, skipped = batch.sample_jobs(seeds=[0],types=[16]); job=jobs[0]
    batch.write_json(tmp_path/'plan.json',dict(jobs=[job],skipped=skipped))
    work=tmp_path/'jobs'/job['id']
    batch.write_json(work/'status.json',dict(state=state))
    batch.write_json(work/'solve/manifest.json',dict(cases=[dict(name='case',summary='case/summary.json')]))
    batch.write_json(work/'solve/case/summary.json',dict(task_success=True,evaluation_protocol=protocol))
    result=batch.refresh(tmp_path)
    assert result['successes']==int(expected) and result['success_rate']==int(expected)
    assert result['planned']==1


def test_multiple_candidate_manifest_is_not_best_of_success(tmp_path):
    jobs, skipped = batch.sample_jobs(seeds=[0],types=[16]); job=jobs[0]
    batch.write_json(tmp_path/'plan.json',dict(jobs=[job],skipped=skipped))
    work=tmp_path/'jobs'/job['id']
    batch.write_json(work/'status.json',dict(state='completed'))
    batch.write_json(work/'solve/manifest.json',dict(cases=[dict(summary='a.json'),dict(summary='b.json')]))
    for name in ('a','b'):
        batch.write_json(work/f'solve/{name}.json',dict(task_success=True,evaluation_protocol=batch.PROTOCOL))
    report=batch.refresh(tmp_path)
    assert report['successes']==0 and report['jobs'][0]['failure_stage']=='single_candidate_contract'


def test_snapshot_drift_is_rejected(tmp_path):
    source=tmp_path/'runtime/source.py';source.parent.mkdir();source.write_text('original')
    asset=tmp_path/'hand.xml';asset.write_text('hand')
    plan=dict(runtime=str(source.parent),source_hashes={'source.py':batch.digest(source)},
        input_hashes={str(asset):batch.digest(asset)})
    batch.verify_snapshot(plan)
    source.write_text('changed')
    with pytest.raises(ValueError,match='Frozen runtime'):
        batch.verify_snapshot(plan)
    source.write_text('original');asset.write_text('different hand')
    with pytest.raises(ValueError,match='Shared experiment asset'):
        batch.verify_snapshot(plan)


def test_timeout_terminates_process_and_remains_a_trial_failure(tmp_path):
    import sys
    runtime=tmp_path/'runtime';script=runtime/'scripts'/batch.Path(batch.__file__).name
    script.parent.mkdir(parents=True);script.write_text('import time\ntime.sleep(60)\n')
    job=dict(id='trial')
    trial=tmp_path/'jobs/trial';batch.write_json(trial/'status.json',dict(state='queued'))
    plan=dict(runtime=str(runtime),python=sys.executable,timeout_seconds=.05)
    batch.execute_job(tmp_path,job,plan)
    result=batch.read(trial/'status.json')
    assert result['state']=='timeout' and result['reason']=='trial_timeout'
    assert batch.Path(result['work']).name=='attempt_000'
    assert result['child_pid'] is None
    # Terminal failures are never retried automatically.
    batch.execute_job(tmp_path,job,plan)
    assert len(list(trial.glob('attempt_*')))==1


def test_resume_preserves_interrupted_output_and_uses_new_attempt(tmp_path):
    import sys
    runtime=tmp_path/'runtime';script=runtime/'scripts'/batch.Path(batch.__file__).name
    script.parent.mkdir(parents=True);script.write_text('pass\n')
    job=dict(id='trial');trial=tmp_path/'jobs/trial';old=trial/'attempt_000'
    old.mkdir(parents=True);(old/'partial.txt').write_text('keep')
    batch.write_json(trial/'status.json',dict(state='running',work=str(old),child_pid=None))
    plan=dict(runtime=str(runtime),python=sys.executable,timeout_seconds=2,jobs=[job])
    batch.recover_interrupted(tmp_path,plan)
    batch.execute_job(tmp_path,job,plan)
    assert (old/'partial.txt').read_text()=='keep'
    assert batch.Path(batch.read(trial/'status.json')['work']).name=='attempt_001'


def test_stopped_supervisor_cancels_pending_and_refreshes_only_once(tmp_path,monkeypatch):
    import concurrent.futures
    jobs=[dict(id=f'trial{i}') for i in range(10)]
    batch.write_json(tmp_path/'plan.json',dict(runtime=str(tmp_path/'runtime'),workers=2,jobs=jobs))
    monkeypatch.setattr(batch,'verify_snapshot',lambda _:None)
    monkeypatch.setattr(batch,'recover_interrupted',lambda *a:None)
    calls=[];futures=[]
    class FakeFuture:
        def result(self):
            calls.append('result');batch.STOP.set()
        def cancel(self):
            calls.append('cancel')
    class FakePool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,*args):
            future=FakeFuture();futures.append(future);return future
    monkeypatch.setattr(batch,'ThreadPoolExecutor',FakePool)
    monkeypatch.setattr(concurrent.futures,'as_completed',lambda rows:iter(rows))
    report=dict(finished=0,planned=10,successes=0)
    monkeypatch.setattr(batch,'refresh',lambda path:(calls.append('refresh') or report))
    try:
        assert batch.run(tmp_path)==report
        assert calls.count('result')==1
        assert calls.count('cancel')==10
        assert calls.count('refresh')==1
    finally:
        batch.STOP.clear()


def test_supervisor_throttles_normal_refresh_and_always_flushes_final(tmp_path,monkeypatch):
    import concurrent.futures
    jobs=[dict(id=f'trial{i}') for i in range(10)]
    batch.write_json(tmp_path/'plan.json',dict(runtime=str(tmp_path/'runtime'),workers=2,jobs=jobs))
    monkeypatch.setattr(batch,'verify_snapshot',lambda _:None)
    monkeypatch.setattr(batch,'recover_interrupted',lambda *a:None)
    refreshes=[]
    class FakeFuture:
        def result(self):pass
    class FakePool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,*args):return FakeFuture()
    monkeypatch.setattr(batch,'ThreadPoolExecutor',FakePool)
    monkeypatch.setattr(concurrent.futures,'as_completed',lambda rows:iter(rows))
    times=iter([0,1,2,3,4,5,6,7,8,9,10])
    monkeypatch.setattr(batch.time,'monotonic',lambda:next(times))
    report=dict(finished=10,planned=10,successes=0)
    monkeypatch.setattr(batch,'refresh',lambda path:(refreshes.append(path) or report))
    assert batch.run(tmp_path)==report
    assert len(refreshes)==3 # 5s, 10s, and an unconditional final flush.


def test_numerical_stack_uses_worker_metadata_without_importing_cuda(monkeypatch):
    from types import SimpleNamespace
    calls=[]
    def execute(command,**kwargs):
        calls.append((command,kwargs));return SimpleNamespace(stdout='{"python":"3.12.3","torch":"2.5.1"}')
    monkeypatch.setattr(batch.subprocess,'run',execute)
    assert batch.numerical_stack('/frozen/.venv/bin/python')['torch']=='2.5.1'
    command,kwargs=calls[0]
    assert command[0]=='/frozen/.venv/bin/python'
    assert 'importlib.metadata' in command[2] and 'import torch' not in command[2]
    assert kwargs['timeout']==30.


def test_resume_rejects_numerical_stack_version_drift(tmp_path,monkeypatch):
    plan=dict(runtime=str(tmp_path),source_hashes={},input_hashes={},
              python='/frozen/python',numerical_stack={'python':'3.12.3','mujoco':'3.3.3'})
    monkeypatch.setattr(batch,'numerical_stack',lambda python:{'python':'3.12.3','mujoco':'3.3.7'})
    with pytest.raises(ValueError,match='numerical stack changed: mujoco'):
        batch.verify_snapshot(plan)
