"""Batch inventory and subprocess failure handling, with no grasp fabrication."""

import json
from pathlib import Path

import pytest

from scripts import run_spider_grasp_batch as batch


def test_full_inventory_uses_shared_per_type_dimensions():
    from retarget.env.grasp.primitives import CATALOG, size_specifications
    jobs, skipped = batch.sample_jobs()
    assert len(jobs) == 161 and len({j["id"] for j in jobs}) == 161
    assert {row["type_id"] for row in skipped} == {19, 21}
    assert len({j["type_id"] for j in jobs}) == 31
    for type_id in {j["type_id"] for j in jobs}:
        actual = sorted((j for j in jobs if j["type_id"] == type_id), key=lambda j:j["size_index"])
        assert [j["primitive"] for j in actual] == size_specifications(type_id, CATALOG[type_id][0][0])


def test_explicit_legacy_batch_grid_remains_available():
    jobs, _ = batch.sample_jobs([.5,.75,1.,1.25,1.5])
    assert len(jobs) == 155
    assert all(j["primitive"]["height_m"] == .18 for j in jobs if j["type_id"] <= 6)


@pytest.mark.parametrize("scales", [[1]*5, [1, 2], [0, .75, 1, 1.25, 1.5], [.5, .75, 1, 1.25, float("nan")]])
def test_invalid_size_grids_fail_before_queue_creation(scales):
    with pytest.raises(ValueError, match="five distinct"):
        batch.sample_jobs(scales)


def test_existing_output_is_never_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    monkeypatch.setattr(batch, "sample_jobs", lambda *args: ([], []))
    target = tmp_path/"output/batch"
    target.mkdir(parents=True)
    (target/"keep").write_text("original")
    with pytest.raises(ValueError, match="new direct child"):
        batch.make_batch(target)
    assert (target/"keep").read_text() == "original"


@pytest.mark.parametrize("code, delay, timeout, expected", [(0, .01, 5., 0), (3, .01, 5., 3), (0, 30., .05, "timeout")])
def test_child_exit_and_timeout_are_reported_without_starting_real_solver(tmp_path, code, delay, timeout, expected):
    runtime = tmp_path/"runtime"
    script = runtime/"scripts"/Path(batch.__file__).name
    script.parent.mkdir(parents=True)
    script.write_text(f"import time,sys\nprint('child-start',flush=True)\ntime.sleep({delay})\nsys.exit({code})\n")
    batch.write(tmp_path/"plan.json", dict(runtime=str(runtime)))
    work = tmp_path/"work"; work.mkdir()
    events = []
    result = batch.run_child(tmp_path, {"id": "t01"}, work, "prepare", timeout, lambda **event: events.append(event))
    assert result == expected and events[0]["state"] == "preparing_ca"
    assert events[0]["child_pid"] > 0
    assert (work/"prepare.log").is_file()


def test_ca_failure_never_runs_mppi_and_remains_in_summary(tmp_path, monkeypatch):
    job = dict(id="t01", type_id=1, template="1_Large_Diameter", primitive={})
    status = tmp_path/"jobs/t01/status.json"
    batch.write(status, dict(state="queued"))
    calls = []
    def child(root, row, work, mode, timeout, update):
        calls.append(mode)
        batch.write(work/"prepared.json", dict(state="ca_failed", reason="strict_CA_refinement_not_accepted"))
        return 3
    monkeypatch.setattr(batch, "run_child", child)
    batch.execute_job(tmp_path, job, 10.)
    assert calls == ["prepare"]
    assert batch.read(status)["state"] == "ca_failed"
    plan = dict(jobs=[job], workers=2, skipped=[], radius_scales=[.5,.75,1,1.25,1.5], long_cylinder_height_m=.18)
    batch.refresh(tmp_path, plan, final=True)
    report = batch.read(tmp_path/"summary.json")
    assert report["state"] == "completed_with_failures"
    assert report["failed"] == 1 and report["completed"] == 0


def test_viewer_manifest_accepts_pending_jobs_without_fabricating_poses(tmp_path):
    from scripts.prepare_spider_batch_viewer import prepare
    batch.write(tmp_path/"summary.json",dict(jobs=[dict(id="t01_cylinder_s02",state="queued")]))
    report=prepare(tmp_path)
    assert report["available"]==0 and report["pending"]==["t01_cylinder_s02"]
    assert batch.read(tmp_path/"manifest.json")["cases"]==[]


def test_finite_failed_ca_proposal_reaches_mppi_in_seed_mode(tmp_path,monkeypatch):
    job=dict(id='t01',type_id=1,template='1_Large_Diameter',primitive={})
    batch.write(tmp_path/'plan.json',dict(ca_is_seed=True))
    batch.write(tmp_path/'jobs/t01/status.json',dict(state='queued'))
    calls=[]
    def child(root,row,work,mode,timeout,update):
        calls.append(mode)
        if mode=='prepare':
            batch.write(work/'prepared.json',dict(state='seed_ready',ca_success=False))
        else:
            batch.write(work/'solve/summary.json',dict(cases=[dict(task_success=False,
                native_hold_success=False,independent_mujoco_hold_success=False,
                acquisition=dict(success=False),timing=dict(optimization_seconds=1.))]))
        return 0
    monkeypatch.setattr(batch,'run_child',child)
    batch.execute_job(tmp_path,job,10.)
    assert calls==['prepare','solve']
    assert batch.read(tmp_path/'jobs/t01/status.json')['state']=='completed'


def test_full_sweep_gate_uses_shared_protocol_and_rejects_historical_success(tmp_path):
    jobs,_=batch.sample_jobs()
    selected=set()
    for type_id in batch.PILOT_TYPES:
        indices=sorted(j['size_index'] for j in jobs if j['type_id']==type_id)
        selected.update((type_id,i) for i in (indices[0],indices[len(indices)//2],indices[-1]))
    jobs=[j for j in jobs if (j['type_id'],j['size_index']) in selected]
    assert len(jobs)==24
    batch.write(tmp_path/'plan.json',dict(jobs=jobs))
    for j in jobs:
        work=tmp_path/'jobs'/j['id']/'attempt_000';case=work/'solve'/('000_'+j['id'])
        batch.write(work.parent/'status.json',dict(state='completed',work=str(work)))
        batch.write(work/'solve/manifest.json',dict(cases=[dict(summary=str(case.relative_to(work/'solve')/'summary.json'))]))
        batch.write(case/'summary.json',dict(task_success=True,
            evaluation_protocol=None if j['primitive']['kind']=='sphere' else 'graspadmm_native_v1',
            acquisition=dict(success=False),native_hold_success=False))
    gate=batch.promotion_gate(tmp_path)
    assert not gate['eligible'] and gate['missing_families']==['sphere']
    sphere=next(j for j in jobs if j['primitive']['kind']=='sphere')
    certificate=tmp_path/'jobs'/sphere['id']/'attempt_000/solve'/('000_'+sphere['id'])/'summary.json'
    result=batch.read(certificate);result['evaluation_protocol']='graspadmm_native_v1';batch.write(certificate,result)
    assert batch.promotion_gate(tmp_path)['eligible']
    batch.write(tmp_path/'jobs'/jobs[-1]['id']/'status.json',dict(state='queued'))
    assert not batch.promotion_gate(tmp_path)['eligible']


def test_nonfinite_seed_remains_visible_as_object_only_failure(tmp_path):
    import numpy as np
    from scripts.prepare_spider_batch_viewer import prepare
    row=dict(id='t01_cylinder_s02',type_id=1,template='1_Large_Diameter',state='input_invalid',
        primitive=dict(kind='cylinder',radius_m=.035,height_m=.18,radial_scale=1.),work=str(tmp_path/'work'))
    seed=tmp_path/'work/spider_seed';seed.mkdir(parents=True)
    np.savez_compressed(seed/'spider_seed.npz',qpos=np.full(33,np.nan))
    batch.write(seed/'spider_seed.json',dict(scene='missing.xml'))
    batch.write(tmp_path/'summary.json',dict(jobs=[row]))
    result=prepare(tmp_path);cases=batch.read(tmp_path/'manifest.json')['cases']
    assert result['object_only_failures']==1 and len(cases)==1
    assert cases[0]['stage']=='失败信息' and not cases[0]['has_hand_pose']
