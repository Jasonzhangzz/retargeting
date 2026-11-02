"""Final-grasp extraction and a single native acceptance source for all methods."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from retarget.env.grasp import grasp_evaluation as evaluation
from scripts import reevaluate_grasps as history


def test_static_grasp_is_not_replaced_by_predicted_last_state(tmp_path):
    np.savez(tmp_path/"grasp.npz",qpos=np.arange(33))
    np.savez(tmp_path/"trajectory.npz",qpos=np.full((3,33),99.))
    qpos,selected=evaluation.select_grasp(tmp_path,"mppi")
    np.testing.assert_array_equal(qpos,np.arange(33))
    assert selected["kind"]=="optimized_static_grasp"


def test_dynamic_grasp_uses_last_actual_state_without_searching_eval_frames(tmp_path):
    qpos=np.arange(99).reshape(3,33)
    np.savez(tmp_path/"trajectory.npz",qpos=qpos)
    final,selected=evaluation.select_grasp(tmp_path,"dexforge")
    np.testing.assert_array_equal(final,qpos[-1])
    assert selected["kind"]=="final_executed_state"


def test_failed_acquisition_still_has_a_real_mppi_output(tmp_path):
    (tmp_path/"acquisition").mkdir()
    (tmp_path/"acquisition/acquisition.json").write_text(json.dumps(dict(success=False,hand_qpos=list(range(26)))))
    np.savez(tmp_path/"trajectory.npz",qpos=np.zeros((1,33)))
    qpos,selected=evaluation.select_grasp(tmp_path,"spider_mppi")
    assert qpos.shape==(26,) and selected["kind"]=="optimized_mppi_acquisition"


@pytest.mark.parametrize("reason",["initialization_failed","contactaware_reference_failed","no_feasible_candidate"])
def test_failure_preview_does_not_become_a_solver_output(tmp_path,reason):
    np.savez(tmp_path/"trajectory.npz",qpos=np.zeros((1,33)))
    (tmp_path/"summary.json").write_text(json.dumps(dict(reason=reason)))
    with pytest.raises(ValueError,match="No solver output"):
        evaluation.select_grasp(tmp_path,"mppi")


@pytest.mark.parametrize("old,new",[(True,False),(False,True)])
def test_common_verdict_replaces_method_acceptance_without_adding_legacy_gates(tmp_path,monkeypatch,old,new):
    summary=dict(task_success=old,acceptance=dict(max_displacement_mm=2.))
    monkeypatch.setattr(evaluation,"evaluate_grasp",lambda *a,**kw:dict(success=new,six_direction_tested=True))
    evaluation.apply_evaluation(summary,"unused.xml",np.zeros(33),tmp_path,method="mppi")
    assert summary["task_success"] is new
    assert summary["method_original_acceptance"]["task_success"] is old
    assert summary["evaluation_protocol"]==evaluation.PROTOCOL
    assert summary["acceptance"]["translation_threshold_m"]==.05


def test_native_subprocess_timeout_remains_failed_and_visible(tmp_path,monkeypatch):
    import subprocess
    def timeout(*a,**kw):raise subprocess.TimeoutExpired("native",1.)
    monkeypatch.setattr(evaluation.subprocess,"run",timeout)
    report=evaluation.evaluate_candidate(tmp_path,timeout=1.)
    assert not report["task_success"] and report["failure_stage"]=="evaluation_runtime"
    assert (tmp_path/"evaluation.json").is_file()


def test_historical_manifest_aliases_merge_and_other_tasks_are_untouched(tmp_path):
    original=tmp_path/"old";original.mkdir()
    case=dict(task="grasp",method="mppi",name="pose",model="scene.xml",trajectory="trajectory.npz",summary="summary.json")
    document=dict(task="grasp",cases=[case,case])
    (original/"manifest.json").write_text(json.dumps(document))
    other=tmp_path/"humanoid";other.mkdir()
    (other/"manifest.json").write_text(json.dumps(dict(task="humanoid",cases=[dict(case,task="humanoid")])))
    before={p:p.read_bytes() for p in tmp_path.rglob("*.json")}
    found=history.discover_cases(tmp_path)
    assert len(found)==1 and len(found[0]["aliases"])==2
    assert all(p.read_bytes()==value for p,value in before.items())


def test_historical_groups_use_actual_dimensions_and_keep_failed_inputs_in_denominator(tmp_path):
    rows=[dict(id="a",state="completed",method="mppi",template="9_Lateral",task_success=True,
               primitive=dict(kind="disk",radius_m=.018,height_m=.008)),
          dict(id="b",state="input_unavailable",method="mppi",template="9_Lateral",task_success=False,
               primitive=dict(kind="disk",radius_m=.018,height_m=.008)),
          dict(id="c",state="completed",method="mppi",template="9_Lateral",task_success=False,
               primitive=dict(kind="cylinder",radius_m=.018,height_m=.12))]
    import time
    result=history.refresh(tmp_path,rows,time.perf_counter(),complete=True)
    assert len(result["groups"])==2
    disk=next(g for g in result["groups"] if g["primitive"]=="disk")
    assert disk["success_rate"]==.5 and disk["trials"]==2
    assert (tmp_path/"success_rates.csv").is_file()
    assert json.loads((tmp_path/"manifest.json").read_text())["historical_reevaluation"]


def test_candidate_bridge_preserves_actual_object_local_geometry_and_relative_palm_pose(tmp_path):
    import mujoco
    import trimesh
    from retarget.env.grasp.inputs import build_scene
    mesh=tmp_path/"source.obj";trimesh.creation.box(extents=[.04,.03,.02]).export(mesh)
    model,_,xml,_=build_scene(mesh,hand_model="wuji2")
    scene=tmp_path/"scene.xml";scene.write_text(xml)
    q=model.qpos0.copy();q[:3]=[.03,.02,.01]
    metadata=evaluation.prepare_candidate(scene,q,tmp_path/"candidate",method="cem",
        selection=dict(required_cbody=[["right_finger1_link4"],["right_palm_link"]]))
    with np.load(tmp_path/"candidate/candidate.npz") as raw:
        grasp=raw["grasp_qpos"]
    np.testing.assert_allclose(grasp[:3],q[:3],atol=1e-9)
    assert grasp.shape==(27,) and metadata["required_cbody"][1]==["right_palm_link"]
    object_mesh=trimesh.load_mesh(tmp_path/"candidate/object.obj",process=False)
    np.testing.assert_allclose(object_mesh.extents,[.04,.03,.02],atol=1e-8)
    assert scene.read_text()==xml


def test_contactaware_uses_final_failed_optimizer_pose_before_preview(tmp_path):
    qpos=np.arange(26,dtype=float)
    (tmp_path/"summary.json").write_text(json.dumps(dict(contactaware=dict(success=False,reference_qpos=qpos.tolist()))))
    np.savez(tmp_path/"trajectory.npz",qpos=np.zeros((1,26)))
    selected,details=evaluation.select_grasp(tmp_path,"contactaware")
    np.testing.assert_array_equal(selected,qpos)
    assert details["kind"]=="optimized_contactaware_grasp"


def test_historical_fa_rejection_missing_from_manifest_still_counts(tmp_path):
    case=tmp_path/"failed_fa/000_sample";case.mkdir(parents=True)
    (case/"summary.json").write_text(json.dumps(dict(name="000_sample",method="dexforge",
        solver_completed=False,task_success=False,failed_stage="static_actuator_preload")))
    rows=history.discover_cases(tmp_path)
    assert len(rows)==1 and rows[0]["model"] is None and rows[0]["method"]=="dexforge"


def test_recording_observes_original_six_axis_checks_without_extra_steps():
    pytest.importorskip("qpsolvers")
    from retarget.env.grasp.graspadmm_source import require_upstream
    source=require_upstream()
    target=np.array([0.,0.,0.,1.,0.,0.,0.])
    class FakeEnv:
        def __init__(self):
            self._model=SimpleNamespace(opt=SimpleNamespace(timestep=.004))
            self._data=SimpleNamespace(qpos=np.zeros(34),ctrl=np.zeros(20))
            self.steps=[];self.resets=0
        def reset_qpos(self,*args,**kwargs):self.resets+=1
        def run_trajectory(self,*args):return None,None
        def set_active_obj_ext_fdir(self,direction):pass
        def get_active_obj_pose(self):return target.copy()
        def step_sim(self,substep):self.steps.append(substep)
    env=FakeEnv();records=evaluation.record_native_evaluation(env,target)
    success,_,_=source.sim.MuJoCo_EvalEnv.eval_fc(env,[np.zeros(27)],["ee_pose"],[None],target,.05,15.)
    assert success and env.resets==6 and env.steps==[50]*60
    assert len(records["qpos"])==66
    assert all(row["tested"] and row["success"] and len(row["checks"])==10 for row in records["directions"])


def test_cylinder_rim_vertices_are_not_misclassified_as_a_sphere():
    import trimesh
    cylinder=trimesh.creation.cylinder(radius=.035,height=.12,sections=48)
    sphere=trimesh.creation.icosphere(radius=.035)
    disk=trimesh.creation.cylinder(radius=.035,height=.01,sections=48)
    assert evaluation.measure_primitive(cylinder.vertices)["kind"]=="cylinder"
    assert evaluation.measure_primitive(sphere.vertices)["kind"]=="sphere"
    assert evaluation.measure_primitive(disk.vertices)["kind"]=="disk"
    assert evaluation.measure_primitive(cylinder.vertices)["radius_m"]==.035


def test_persistent_worker_reuses_interpreter_and_preserves_candidate_identity(tmp_path):
    import sys
    helper=tmp_path/"worker.py"
    helper.write_text('''import json,os,sys
for line in sys.stdin:
 request=json.loads(line)
 print("GRASPADMM_RESULT "+json.dumps(dict(candidate=request["candidate"],report=dict(state="completed",success=False,task_success=False,worker_pid=os.getpid()))),flush=True)
''')
    first=tmp_path/"first";second=tmp_path/"second";first.mkdir();second.mkdir()
    with evaluation.NativeEvaluationWorker(tmp_path/"worker.log",command=[sys.executable,str(helper)]) as worker:
        a=worker.evaluate(first,timeout=5.)
        b=worker.evaluate(second,timeout=5.)
    assert a["worker_pid"]==b["worker_pid"]
    assert json.loads((first/"evaluation.json").read_text())["task_success"] is False
    assert worker.process is None


def test_resume_refreshes_misclassified_old_metadata_without_repeating_physics(tmp_path,monkeypatch):
    import trimesh
    case=tmp_path/"case";native=case/"evaluation";native.mkdir(parents=True)
    mesh=native/"object.obj";trimesh.creation.cylinder(radius=.035,height=.12,sections=48).export(mesh)
    (native/"candidate.json").write_text(json.dumps(dict(mesh=str(mesh),source_model="unused.xml")))
    monkeypatch.setattr(history,"infer_template",lambda *a:"1_Large_Diameter")
    row=dict(primitive=dict(kind="sphere",radius_m=.06946,height_m=.12,dimensions_source="actual source object-local mesh bounds"))
    history.refresh_candidate_metadata(case,row)
    assert row["primitive"]["kind"]=="cylinder" and row["primitive"]["radius_m"]==pytest.approx(.035,abs=1e-8)
    assert row["template"]=="1_Large_Diameter"


def test_historical_method_selection_is_saved_in_plan_and_report(tmp_path):
    source=tmp_path/"source";source.mkdir()
    cases=[dict(task="grasp",method=method,name=method,model="missing.xml",trajectory=f"{method}/trajectory.npz",
                summary=f"{method}/summary.json") for method in ("mppi","graspadmm")]
    (source/"manifest.json").write_text(json.dumps(dict(task="grasp",cases=cases)))
    result=history.run(source,tmp_path/"evaluation",methods=["mppi"])
    plan=json.loads((tmp_path/"evaluation/plan.json").read_text())
    assert result["total"]==1 and result["methods"]==["mppi"]
    assert plan["methods"]==["mppi"] and plan["cases"][0]["method"]=="mppi"
    assert result["cases"][0]["state"]=="input_unavailable"


def test_metadata_resume_preserves_initial_evaluation_timing(tmp_path):
    source=tmp_path/"source";source.mkdir()
    (source/"manifest.json").write_text(json.dumps(dict(task="grasp",cases=[])))
    output=tmp_path/"evaluation"
    first=history.run(source,output)
    second=history.run(source,output,resume=True)
    assert second["evaluation_run_wall_seconds"]==first["evaluation_run_wall_seconds"]
    assert second["wall_seconds"]==first["evaluation_run_wall_seconds"]
    assert second["metadata_refresh_wall_seconds"]>=0.


def test_native_fastpath_is_limited_to_graspadmm_in_a_verified_environment(tmp_path,monkeypatch):
    mesh=tmp_path/"mesh.obj";mesh.write_text("copied by the bridge; parser is mocked\n")
    q=np.r_[np.zeros(3),1.,np.zeros(23)]
    monkeypatch.setattr(evaluation,"pinned_native_environment",lambda:True)
    monkeypatch.setattr(evaluation,"native_evaluate",lambda *a:dict(state="completed",success=True,task_success=True))
    calls=[]
    monkeypatch.setattr(evaluation,"evaluate_candidate",lambda *a,**k:calls.append(a) or dict(state="completed",success=False,runtime_mode="isolated_subprocess"))
    native=evaluation.evaluate_native_grasp(q,mesh,tmp_path/"native",method="graspadmm")
    external=evaluation.evaluate_native_grasp(q,mesh,tmp_path/"external",method="mppi")
    assert native["runtime_mode"]=="in_process_pinned" and native["success"]
    assert external["runtime_mode"]=="isolated_subprocess" and len(calls)==1


def test_native_fastpath_records_runtime_errors_in_the_same_result_contract(tmp_path,monkeypatch):
    mesh=tmp_path/"mesh.obj";mesh.write_text("mock mesh\n")
    q=np.r_[np.zeros(3),1.,np.zeros(23)]
    monkeypatch.setattr(evaluation,"pinned_native_environment",lambda:True)
    def fail(*args):raise RuntimeError("native failure")
    monkeypatch.setattr(evaluation,"native_evaluate",fail)
    result=evaluation.evaluate_native_grasp(q,mesh,tmp_path/"native")
    assert result["runtime_mode"]=="in_process_pinned" and result["state"]=="failed"
    assert not result["task_success"] and "native failure" in result["error"]
    assert "native failure" in (tmp_path/"native/evaluation.log").read_text()


def test_pinned_environment_rejects_a_different_interpreter_prefix(monkeypatch,tmp_path):
    evaluation.pinned_native_environment.cache_clear()
    monkeypatch.setattr(evaluation.sys,"prefix",str(tmp_path/"another-environment"))
    try:assert not evaluation.pinned_native_environment()
    finally:evaluation.pinned_native_environment.cache_clear()


def test_direct_and_subprocess_native_protocol_match_on_a_real_successful_grasp(tmp_path):
    if not evaluation.pinned_native_environment():
        pytest.skip("Run in .venv-graspadmm to compare the exact pinned native stack")
    import trimesh
    fixture=Path(__file__).parent/"fixtures/graspadmm_eval_sphere.npz"
    with np.load(fixture,allow_pickle=False) as raw:
        q=raw["grasp_qpos"].copy();required=json.loads(str(raw["required_cbody_json"]))
        mesh=trimesh.Trimesh(raw["vertices"],raw["faces"],process=False)
    source=tmp_path/"sphere.obj";mesh.export(source,digits=17)
    direct=evaluation.evaluate_native_grasp(q,source,tmp_path/"direct",required_cbody=required)
    # Force the public isolated route while retaining an identical candidate.
    original=evaluation.pinned_native_environment
    try:
        evaluation.pinned_native_environment=lambda:False
        isolated=evaluation.evaluate_native_grasp(q,source,tmp_path/"isolated",required_cbody=required)
    finally:evaluation.pinned_native_environment=original
    assert direct["success"] is True and isolated["success"] is True
    assert direct["runtime_mode"]=="in_process_pinned"
    assert isolated["runtime_mode"]=="isolated_subprocess"
    for key in ("state","success","task_success","criterion","generation","directions","upstream_commit"):
        assert direct[key]==isolated[key]
    with np.load(tmp_path/"direct/evaluation_trajectory.npz",allow_pickle=False) as a, np.load(tmp_path/"isolated/evaluation_trajectory.npz",allow_pickle=False) as b:
        assert a.files==b.files
        for key in a.files:np.testing.assert_array_equal(a[key],b[key])
        assert np.all(np.diff(a["time"])>0)
        assert np.count_nonzero(np.diff(a["sim_time"])<0)==5
