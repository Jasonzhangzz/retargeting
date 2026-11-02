"""Runner contracts without running a GPU optimizer or certifying a grasp."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import trimesh

from retarget.env.grasp import template_mppi_runner as runner
from retarget.solver.base import Settings


def _config(tmp_path, **runtime):
    from retarget.config.run import ROOT

    mesh = tmp_path / "offcenter.obj"
    object_mesh = trimesh.creation.box(extents=[.02, .03, .04])
    object_mesh.apply_translation([.11, -.07, .03])
    object_mesh.export(mesh)
    return SimpleNamespace(task=dict(type="grasp", data=str(ROOT / "data/grasp"),
                                    object=str(mesh), templates=[24], options={}),
                           environment={"template_mppi": runtime, "dynamics": "mjwarp"},
                           solver=Settings(type="mppi", population=32, worlds=4, iterations=1),
                           device="cuda:0", seed=13)


def test_new_config_uses_q_only_runner_and_no_admm_input():
    from retarget.config.run import ROOT, load

    config = load(ROOT / "configs/grasp_template_mppi.yaml")
    args = runner.validate(config)
    assert config.solver.type == "mppi"
    assert args.settings.objective.stages == 2
    assert args.settings.initialization == dict(points=24, inplane=8, iterations=60)
    assert args.settings.evaluate_native is False
    assert args.templates == ["1_Large_Diameter"]
    assert set(config.task["options"]) == {"hand_model", "primary_shape_only"}


@pytest.mark.parametrize("options", [
    {"initialization": {"points": True}}, {"initialization": {"iterations": 0}},
    {"initialization": {"inplane": 1.5}}, {"initialization": {"unknown": 1}},
    {"initialization": []}, {"object_mass": 0}, {"object_mass": float("nan")},
    {"evaluate_native": 1}, {"evaluate_native": True}, {"objective": []}, {"unknown": 1},
])
def test_invalid_options_fail_before_cuda_allocation(tmp_path, options):
    with pytest.raises(ValueError):
        runner.validate(_config(tmp_path, **options))


def test_old_contact_fitting_options_cannot_enter_the_template_runner(tmp_path):
    config = _config(tmp_path)
    config.task["options"]["reference_method"] = "dexforge_contactaware"
    with pytest.raises(ValueError, match="Unknown template MPPI input"):
        runner.validate(config)


def test_root_candidates_are_scored_at_their_actual_poses_without_a_contact_gate():
    candidates = np.zeros((3, 26))
    candidates[:, 0] = [.1, -.2, .3]
    calls = []

    class Objective:
        def bind(self, template, q):
            calls.append(q.copy())

        def evaluate(self, controls):
            torch.testing.assert_close(controls[:, 0, 0], torch.tensor([.1, -.2, .3]))
            return controls[:, 0, 0].square()

    q, metadata = runner._choose_root(Objective(), None, candidates, "cpu")
    np.testing.assert_array_equal(q, candidates[0])
    assert len(calls) == 1
    assert metadata["actual_contact_or_force_closure_required"] is False
    assert metadata["selection_objective"] == runner.OBJECTIVE


def test_nonfinite_root_scores_cannot_be_selected():
    class Objective:
        def bind(self, *args):
            pass

        def evaluate(self, controls):
            return torch.full((len(controls),), torch.inf)

    with pytest.raises(FloatingPointError, match="No finite root"):
        runner._choose_root(Objective(), None, np.zeros((2, 26)), "cpu")


def test_centered_object_fixed_contacts_and_final_grasp_are_exported(tmp_path, monkeypatch):
    import warp as wp
    from retarget.env.grasp import template_mppi as objective_module
    from retarget.env.grasp import template_mppi_inputs as inputs_module
    from retarget.env.grasp import grasp_evaluation

    config = _config(tmp_path, evaluate_native=False)
    output = tmp_path / "results"
    local = np.array([[.001, -.002, .003]])
    normals = np.array([[0., 1., 0.]])
    fingers = np.full(20, .1)
    template = SimpleNamespace(body_names=("right_palm_link",), local_points=local,
                               local_normals=normals, finger_qpos=fingers,
                               required_cbody=(("right_palm_link",),),
                               to_dict=lambda: dict(local_points=local.tolist(),
                                                    local_normals=normals.tolist()))
    observed = {}

    def load_template(name, checkout, model, **kwargs):
        assert name == "24_Tip_Pinch"
        return template

    def initialize(hand, supplied, **kwargs):
        assert supplied is template
        np.testing.assert_array_equal(hand.local, local)
        np.testing.assert_array_equal(hand.normals, normals)
        candidates = np.zeros((2, 26))
        candidates[:, 0] = [.1, -.12]
        candidates[:, 6:] = fingers
        return candidates, dict(root_only=True, loss_dist=[[.01], [0.]],
            loss_normal=[[0.], [0.]], obj_cp_w=[[[0., 0., 0.]], [[.01, 0., 0.]]],
            obj_cn_w=[[[0., -1., 0.]], [[0., -1., 0.]]])

    class Objective:
        def __init__(self, mesh, settings, device, object_mass):
            centered = trimesh.load_mesh(mesh, process=False)
            np.testing.assert_allclose(centered.bounds.mean(0), 0., atol=1e-8)
            np.testing.assert_allclose(centered.extents, [.02, .03, .04], atol=1e-8)
            assert object_mass == .1
            observed["mesh"] = Path(mesh)
            self.xml = "<mujoco/>"

        def bind(self, supplied, q):
            assert supplied is template

        def evaluate(self, controls):
            return (controls[:, 0, 0] + .12).square()

        def optimize(self, initial, solver, seed):
            assert initial[0] == -.12 and seed == 13
            q = initial.copy()
            q[0] += .001
            observed["q"] = q
            return q, dict(trajectory=np.stack([initial, q]), initial_cost=2., final_cost=1.)

        def diagnostics(self, q):
            np.testing.assert_array_equal(q, observed["q"])
            return dict(projected_closure_success=True, geometry_success=False,
                        soft_contact_success=True, physical_contact_required=False)

    def evaluate(*args, **kwargs):
        pytest.fail("Template MPPI must never invoke native physical evaluation")

    monkeypatch.setattr(inputs_module, "load_template_contacts", load_template)
    monkeypatch.setattr(inputs_module, "initialize_root_candidates", initialize)
    monkeypatch.setattr(objective_module, "TemplatePoseObjective", Objective)
    monkeypatch.setattr(grasp_evaluation, "evaluate_native_grasp", evaluate)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(wp, "init", lambda: None)
    monkeypatch.setattr(wp, "set_device", lambda device: None)
    choose = runner._choose_root
    monkeypatch.setattr(runner, "_choose_root", lambda objective, template, qs, device:
                        choose(objective, template, qs, "cpu"))
    manifest = runner.run(config, output)
    case = output / "000_24_Tip_Pinch"
    summary = json.loads((case / "summary.json").read_text())
    assert summary["solver_completed"] is True
    assert summary["task_success"] is True
    assert summary["native_evaluated"] is False
    assert summary["admm_reference_used"] is False
    initialization = json.loads((case / "initialization.json").read_text())
    assert initialization["policy"] == "template_1x_object_in_wrist_v1"
    assert initialization["pose_reused_without_resampling"] is True
    assert initialization["translation_scaled"] is False
    frame = json.loads((case / "object_frame.json").read_text())
    np.testing.assert_allclose(frame["recenter_offset_m"], [.11, -.07, .03], atol=1e-7)
    assert json.loads((case / "template_contacts.json").read_text())["local_points"] == local.tolist()
    with np.load(case / "grasp.npz") as values:
        np.testing.assert_array_equal(values["qpos"], observed["q"])
        assert values["grasp_qpos"].shape == (27,)
    with np.load(case / "trajectory.npz") as values:
        np.testing.assert_array_equal(values["qpos"][-1], observed["q"])
    assert not (case / "native_evaluation").exists()
    assert not (case / "native_candidate.npz").exists()
    assert not (case / "native_candidate.json").exists()
    assert len(manifest) == 1
    assert manifest[0]["success_criterion"] == "projected_soft_contact_closure_v2"
    assert all(row["method"] == "mppi" and row["objective"] == runner.OBJECTIVE for row in manifest)
    aggregate = json.loads((output / "summary.json").read_text())
    assert aggregate["planned"] == aggregate["finished"] == 1
    assert aggregate["successes"] == 1


def test_missing_cuda_does_not_create_a_success_or_run_initialization(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no GPU solve"):
        runner.run(_config(tmp_path), tmp_path / "results")
    assert not (tmp_path / "results/summary.json").exists()


def test_input_failure_keeps_trial_and_full_traceback(tmp_path, monkeypatch):
    import warp as wp
    from retarget.env.grasp import template_mppi_inputs as inputs_module

    def reject(*args, **kwargs):
        raise ValueError("template identity mismatch")

    monkeypatch.setattr(inputs_module, "load_template_contacts", reject)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(wp, "init", lambda: None)
    monkeypatch.setattr(wp, "set_device", lambda device: None)
    output = tmp_path / "results"
    manifest = runner.run(_config(tmp_path), output)
    assert manifest == []
    report = json.loads((output / "summary.json").read_text())
    assert report["planned"] == report["finished"] == 1
    assert report["successes"] == report["solver_completed"] == 0
    failure = report["cases"][0]
    assert failure["failure_stage"] == "template_input"
    assert failure["task_success"] is False
    trace = (output / failure["error_log"]).read_text()
    assert "Traceback" in trace and "template identity mismatch" in trace
