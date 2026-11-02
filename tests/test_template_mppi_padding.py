"""Padding is used for native FK, not for closure or candidate diagnostics."""

import numpy as np
import pytest
import torch
import trimesh
import warp as wp

wp.config.kernel_cache_dir = "/tmp/retarget-template-mppi-warp"

from retarget.env.grasp.template_mppi import TemplateMPPISettings, TemplatePoseObjective
from retarget.env.grasp.template_mppi_inputs import TemplateContacts
from retarget.solver.base import Settings


def _objective(tmp_path, device):
    path = tmp_path / "object.obj"
    if not path.exists():
        trimesh.creation.icosphere(subdivisions=1, radius=.03).export(path)
    settings = TemplateMPPISettings(stages=1, closure={
        "gn_iterations": 1, "equilibrium_iterations": 2,
        "equilibrium_line_search_steps": 1, "cuda_graph": False})
    objective = TemplatePoseObjective(path, settings, device=device)
    body = "right_palm_link"
    template = TemplateContacts((body, body), [[0., .02, -.05], [.01, .02, -.05]],
        [[0., 1., 0.], [0., 1., 0.]], np.zeros(20), ((body,),), {})
    q = objective.cpu.qpos0.copy()
    objective.bind(template, q)
    return objective, q


def test_padded_geometry_never_enters_closure_or_parts(tmp_path, monkeypatch):
    reference, q = _objective(tmp_path, "cpu")
    objective, _ = _objective(tmp_path, "cpu")
    objective.worlds = 4
    geometry = objective._geometry

    def padded_geometry(values):
        parts = geometry(values)
        padded = []
        for value in parts:
            shape = (objective.worlds - len(values),) + value.shape[1:]
            # Poison the extra rows to detect any downstream use of them.
            fill = float("nan") if value.is_floating_point() else True
            pad = torch.full(shape, fill, dtype=value.dtype, device=value.device)
            padded.append(torch.cat((value, pad)))
        return tuple(padded)

    batches = []
    solve = objective.closure_solver.solve

    def record_solve(points, normals):
        batches.append(len(points))
        assert bool(torch.isfinite(points).all() & torch.isfinite(normals).all())
        return solve(points, normals)

    monkeypatch.setattr(objective, "_geometry", padded_geometry)
    monkeypatch.setattr(objective.closure_solver, "solve", record_solve)
    candidates = np.tile(q, (7, 1))
    candidates[:, 0] = np.linspace(0., .15, len(candidates))
    candidates[5, 0] = np.nan
    controls = torch.tensor(candidates, dtype=torch.float32)[:, None]
    expected = reference.evaluate(controls)
    actual = objective.evaluate(controls)
    assert batches == [4, 3]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert objective.parts.keys() == reference.parts.keys()
    for key, value in objective.parts.items():
        assert value.shape[0] == len(candidates), key
        torch.testing.assert_close(value, reference.parts[key], rtol=1e-5, atol=1e-5)
    objective.diagnostics(q)
    assert batches == [4, 3, 1]
    assert all(value.shape[0] == 1 for value in objective.parts.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native batched FK requires CUDA")
def test_cuda_tail_guard_and_diagnostics_only_solve_real_candidates(tmp_path, monkeypatch):
    objective, q = _objective(tmp_path, "cuda:0")
    objective.set_batch_size(4)
    solve = objective.closure_solver.solve
    batches = []

    def record_solve(points, normals):
        batches.append(len(points))
        return solve(points, normals)

    monkeypatch.setattr(objective.closure_solver, "solve", record_solve)
    candidates = np.tile(q, (3, 1))
    candidates[:, 0] = [0., .05, .15]
    controls = torch.tensor(candidates, dtype=torch.float32, device="cuda:0")[:, None]
    costs = objective.evaluate(controls)
    batched_parts = {key: value.clone() for key, value in objective.parts.items()}
    singleton = objective.evaluate(controls[:1])
    assert batches == [3, 1]
    torch.testing.assert_close(singleton, costs[:1], rtol=1e-5, atol=1e-5)
    for key, value in objective.parts.items():
        torch.testing.assert_close(value, batched_parts[key][:1], rtol=1e-5, atol=1e-5)
    batches.clear()
    solver = Settings(type="mppi", population=20, worlds=4, iterations=1,
                      improvement_check_steps=2)
    _, report = objective.optimize(q, solver, 17)
    # Initial/final diagnostics and the mean guard are singletons, while
    # ordinary population chunks retain the configured native batch size.
    assert 1 in batches and 4 in batches
    assert set(batches) == {1, 4}
    assert report["history"][0]["mean_guard_evaluated"] == 1
    assert not bool(wp.to_torch(objective.physics.data.time).any())
