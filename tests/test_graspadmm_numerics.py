"""Numerical comparisons against the pinned, unmodified original operation.

Run in .venv-graspadmm. No mock success verdicts or existing output files are
used: the frozen fixture supplies only geometry and an initialization input.
"""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import trimesh

pytest.importorskip("qpsolvers")
pytest.importorskip("rtree")

from retarget.env.grasp.graspadmm_source import require_upstream
from retarget.env.grasp.graspadmm_scene import (write_scene, WujiOptEnv,
    project_to_hand, to_upstream_qpos, from_upstream_qpos)
from retarget.env.grasp.graspadmm_optimizer import optimize, operation_config


@pytest.fixture
def native_case(tmp_path):
    record = json.loads((Path(__file__).parent/"fixtures/graspadmm_reference.json").read_text())
    mesh = tmp_path/"input.obj"
    trimesh.Trimesh(record["vertices"], record["faces"], process=False).export(mesh)
    scene = write_scene(tmp_path/"scene", mesh)
    data = record["data"]
    for key in ("grasp_qpos", "hand_cpn_w", "obj_cpn_w", "ext_wrench", "ext_center", "n_evo"):
        data[key] = np.asarray(data[key], np.float32)
    data["scene_path"] = str(scene.scene_path)
    return data, scene


def original_result(data, scene, directory, settings, monkeypatch):
    source = require_upstream()
    core = source.core
    cfg = operation_config(scene, directory, settings)
    path = Path(cfg.init_dir)/"candidate.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, data)
    function = core._single_grasp.__wrapped__
    captured = {}

    def inspect_return(frame, event, arg):
        if frame.f_code is function.__code__:
            if event == "return":
                captured.update(frame.f_locals)
            return inspect_return
        return None

    monkeypatch.setattr(core, "MuJoCo_OptEnv", WujiOptEnv)
    monkeypatch.setattr(core, "project_to_hand", project_to_hand)
    previous = sys.gettrace()
    try:
        sys.settrace(inspect_return)
        function((str(path), cfg))
    finally:
        sys.settrace(previous)
    return captured


@pytest.mark.parametrize("steps", [1, 4])
def test_recorded_adapter_matches_original_state_and_dual(native_case, tmp_path, monkeypatch, steps):
    data, scene = native_case
    settings = dict(step=steps)
    expected = original_result(copy.deepcopy(data), scene, tmp_path/"original", settings, monkeypatch)
    actual, status, trace = optimize(copy.deepcopy(data), scene, tmp_path/"adapted", settings)
    assert len(trace) == steps and not status["fallback_used"]
    np.testing.assert_allclose(actual["grasp_qpos"], expected["grasp_data"]["grasp_qpos"], rtol=0, atol=1e-7)
    for name, key in (("cp_h", "cp_h"), ("cp_o", "cp_o"), ("dual", "lam")):
        np.testing.assert_allclose(trace[-1][name], expected[key].detach().numpy(), rtol=0, atol=1e-7)
    np.testing.assert_allclose(trace[-1]["grasp_metric_cost"], float(expected["res"].detach()), rtol=0, atol=1e-7)


def test_original_qp_failure_fallback_is_retained(native_case, tmp_path, monkeypatch):
    data, scene = native_case
    source = require_upstream()
    monkeypatch.setattr(source.qp.ContactQPTorch, "solve", lambda *a, **kw: None)
    expected = original_result(copy.deepcopy(data), scene, tmp_path/"original", dict(step=4), monkeypatch)
    actual, status, trace = optimize(copy.deepcopy(data), scene, tmp_path/"adapted", dict(step=4))
    assert expected["terminated"] is True and status["fallback_used"] is True
    np.testing.assert_allclose(actual["grasp_qpos"], expected["grasp_data"]["grasp_qpos"], rtol=0, atol=1e-7)
    with np.load(tmp_path/"adapted/admm_trace.npz", allow_pickle=False) as saved:
        for key in saved.files:
            assert saved[key].dtype != object
        assert np.isnan(saved["grasp_metric_cost"]).all()
        assert not saved["object_metric_available"].any()
    with np.load(tmp_path/"adapted/simulation_trace.npz", allow_pickle=False) as saved:
        for key in saved.files:
            assert saved[key].dtype != object
        assert "fallback" in saved["stage"]


def test_qp_envelope_gradient_matches_value_finite_difference():
    source = require_upstream()
    qp = source.qp.ContactQPTorch([.1, 0.])
    points = torch.tensor([[.04, .01, .003], [.01, .035, -.004]], dtype=torch.float64, requires_grad=True)
    # The original mesh interpolation supplies float64 normals; the upstream
    # tangent-frame routine explicitly uses a float64 tangent constant.
    normals = torch.tensor([[1., 0., 0.], [0., 1., 0.]], dtype=torch.float64)
    wrench, center = np.asarray([.1, -.2, .3, .02, -.01, .005]), np.zeros(3)
    value = qp.solve(points, normals, wrench, center)
    assert value is not None and torch.isfinite(value) and value > .1
    value.backward()
    numerical = np.zeros((2, 3))
    epsilon = .001
    for i in range(2):
        for j in range(3):
            plus, minus = points.detach().clone(), points.detach().clone()
            plus[i, j] += epsilon; minus[i, j] -= epsilon
            numerical[i, j] = (qp.solve(plus, normals, wrench, center)-qp.solve(minus, normals, wrench, center))/(2*epsilon)
    np.testing.assert_allclose(points.grad.numpy(), numerical, rtol=.015, atol=5e-5)
    assert np.all(qp.h_matrix[:12] == -.01)


def test_root_rotation_roundtrip_preserves_physical_transform():
    from scipy.spatial.transform import Rotation
    q = np.r_[.01, -.02, .03, .3, -.7, 2.1, np.linspace(-.1, .2, 20)]
    recovered = from_upstream_qpos(to_upstream_qpos(q))
    np.testing.assert_allclose(recovered, q, atol=1e-12)
    np.testing.assert_allclose(Rotation.from_euler("XYZ", recovered[3:6]).as_matrix(),
                               Rotation.from_euler("XYZ", q[3:6]).as_matrix(), atol=1e-12)
