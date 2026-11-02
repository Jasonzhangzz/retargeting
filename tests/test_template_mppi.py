"""Fixed-slot scoring, full-hand collisions and q-only MPPI regression tests."""
from dataclasses import replace
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
import trimesh
import warp as wp
import yaml

wp.config.kernel_cache_dir = "/tmp/retarget-template-mppi-warp"

from retarget.config.run import ROOT, load
from retarget.env.grasp.audit import GraspAudit
from retarget.env.geometry import ObjectMeshSurface
from retarget.env.grasp.template_mppi import TemplateMPPISettings, TemplatePoseObjective, contact_terms, force_contact_terms
from retarget.env.grasp.template_mppi_inputs import TemplateContacts
from retarget.solver.base import Settings


def test_optional_two_contact_antipodal_guidance_detects_axial_sliding():
    from retarget.env.grasp.template_mppi import antipodal_contact_term
    projected = torch.tensor([[[-.000025, 0, 0], [.000025, 0, 0]],
                              [[-.000025, 0, .003], [.000025, 0, 0]]])
    normals = torch.tensor([[[-1., 0, 0], [1., 0, 0]]]).expand_as(projected)
    result = antipodal_contact_term(projected, normals, .001)
    torch.testing.assert_close(result, torch.tensor([0., 9.]))
    # Rigid transforms and exchanging the two immutable slots cannot change
    # the score. Radial diameter does not mask tangential separation.
    rotated = projected[..., [2, 0, 1]] + torch.tensor([2., 3., 4.])
    torch.testing.assert_close(antipodal_contact_term(rotated, normals[..., [2, 0, 1]], .001),
                               result, atol=.002, rtol=1.e-4)
    torch.testing.assert_close(antipodal_contact_term(projected.flip(1), normals.flip(1), .001), result)
    with pytest.raises(ValueError, match='two fixed contact slots'):
        antipodal_contact_term(projected[:, :1], normals[:, :1], .001)
    assert TemplateMPPISettings().antipodal_weight == 0.


def test_coverage_selects_joint_distance_normal_slots_and_keeps_outlier_guidance():
    settings = TemplateMPPISettings()
    gap = torch.tensor([[0., 0., 0., 0., .02]])
    obj = torch.tensor([[[1., 0., 0.]]]).expand(1, 5, 3)
    normal = -obj.clone()
    normal[:, 0] *= -1
    value = contact_terms(gap, normal, obj, settings)
    assert set(value["selected"][0].tolist()) == {0, 1, 2, 3}
    assert value["coverage"].item() == pytest.approx(.6)
    assert value["distance"].item() > 0  # The distant fifth slot still guides.
    assert value["normal"].item() > 0  # The same selected slots carry this term.


def test_acceptance_angle_and_distance_have_the_same_cost_scale():
    import math
    settings = TemplateMPPISettings(outlier_weight=0.)
    theta = math.radians(settings.normal_tolerance_deg)
    outward = torch.tensor([[[-1., 0., 0.]]])
    hand = torch.tensor([[[math.cos(theta), math.sin(theta), 0.]]])
    angular = contact_terms(torch.zeros(1, 1), hand, outward, settings)
    separated = contact_terms(torch.full((1, 1), settings.contact_tolerance_m), -outward, outward, settings)
    assert angular["normal"].item() == pytest.approx(1., abs=1.e-6)
    assert separated["distance"].item() == pytest.approx(1., abs=1.e-6)


def test_force_on_an_unfitted_outlier_cannot_hide_in_the_topk_cost():
    slot_error = torch.tensor([[0., 0., 100.]])
    fitted = torch.tensor([[True, True, False]])
    force = torch.zeros(1, 13, 3, 4)
    force[:, :, 2, 0] = 1.
    cost, coverage = force_contact_terms(slot_error, fitted, force)
    assert cost.item() == pytest.approx(125.)
    assert coverage.max() == 0.
    force[:, :, 2, 0] = 0.
    force[:, :, 0, 0] = 2.
    cost, coverage = force_contact_terms(slot_error, fitted, force)
    assert cost == 0.
    assert coverage.min() == 1.


def test_cpu_projection_is_independent_of_contact_and_preserves_outward_normals():
    mesh = trimesh.creation.box(extents=[.02, .04, .06])
    audit = GraspAudit(ObjectMeshSurface(mesh.vertices, mesh.faces), "cpu")
    points = torch.tensor([[.04, 0., 0.], [.07, 0., 0.], [.005, 0., 0.]])
    projected, normal, distance = audit.query(points)
    torch.testing.assert_close(projected[:, 0], torch.full((3,), .01), atol=1e-7, rtol=0)
    torch.testing.assert_close(normal, torch.tensor([[1., 0., 0.]]).expand(3, -1))
    torch.testing.assert_close(distance, torch.tensor([.03, .06, -.005]), atol=1e-7, rtol=0)


@pytest.fixture(scope="module")
def mesh_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("template_objective") / "sphere.obj"
    trimesh.creation.icosphere(subdivisions=2, radius=.03).export(path)
    return path


def tiny_settings():
    return TemplateMPPISettings(stages=1, closure={"horizon": 8, "tail_steps": 4, "gn_iterations": 1})


def bind_synthetic(objective):
    body = "right_palm_link"
    template = TemplateContacts((body, body), [[0., .02, -.05], [.01, .02, -.05]],
        [[0., 1., 0.], [0., 1., 0.]], np.zeros(20), ((body,),), {"test": "slot identity"})
    q = objective.cpu.qpos0.copy()
    objective.bind(template, q)
    return template, q


def test_fixed_slots_follow_fk_without_body_deduplication_or_pose_clipping(mesh_path):
    objective = TemplatePoseObjective(mesh_path, tiny_settings(), device="cpu")
    template, initial = bind_synthetic(objective)
    q = np.tile(initial, (2, 1))
    q[1, :3] = [.15, .07, .03]  # Outside the initial local search box.
    q[1, 3:6] = [.2, -.3, .4]
    original = template.local_points.copy()
    values = objective.evaluate(torch.tensor(q, dtype=torch.float32)[:, None])
    assert bool(torch.isfinite(values).all())
    assert objective.parts["hand_points"].shape == (2, 2, 3)
    for i, pose in enumerate(q):
        data = mujoco.MjData(objective.cpu)
        data.qpos[:] = pose
        mujoco.mj_kinematics(objective.cpu, data)
        body = objective.cpu.body("right_palm_link").id
        expected = data.xpos[body] + template.local_points @ data.xmat[body].reshape(3, 3).T
        np.testing.assert_allclose(objective.parts["hand_points"][i], expected, atol=2e-7)
    np.testing.assert_array_equal(template.local_points, original)
    np.testing.assert_array_equal(objective.local, original.astype(np.float32))
    assert np.linalg.norm(objective.parts["hand_points"][1] - objective.parts["hand_points"][0]) > .1


def test_closed_mesh_with_duplicated_cap_seams_retains_its_geometry(tmp_path):
    mesh = trimesh.creation.cylinder(radius=.02, height=.1, sections=16)
    mesh.unmerge_vertices()
    assert not mesh.is_watertight
    path = tmp_path / "seamed.obj"
    mesh.export(path)
    objective = TemplatePoseObjective(path, tiny_settings(), device="cpu")
    np.testing.assert_allclose(objective.center, [0., 0., 0.], atol=1e-8)
    assert objective.radius == pytest.approx(np.sqrt(.02 ** 2 + .05 ** 2), abs=1e-8)


def test_sampler_updates_and_scores_the_same_bounded_candidates(mesh_path, monkeypatch):
    from retarget.env.grasp import template_mppi
    objective = TemplatePoseObjective(mesh_path, tiny_settings(), device="cpu")
    _, q = bind_synthetic(objective)

    def sample(mean, evaluate, settings, **kwargs):
        sampled = mean[None].expand(2, -1, -1).clone()
        sampled[1, 0, 6] = 100.
        sampled[1, 0, 0] = 2.
        evaluate(sampled)
        # The sampler's own storage must now hold the configurations scored.
        assert sampled[1, 0, 6] == objective.qhi[6]
        assert sampled[1, 0, 0] == objective.qhi[0]
        return mean, dict(sampled=2)

    monkeypatch.setattr(template_mppi, "sample_mppi", sample)
    objective.optimize(q, Settings(type="mppi", population=20, worlds=2, iterations=1), 0)


def test_full_hand_penetration_is_penalized_even_away_from_template_points(mesh_path):
    objective = TemplatePoseObjective(mesh_path, tiny_settings(), device="cpu")
    _, q = bind_synthetic(objective)
    candidates = np.tile(q, (2, 1))
    candidates[1, 0] = .4
    objective.evaluate(torch.tensor(candidates, dtype=torch.float32)[:, None])
    depth = objective.parts["penetration_depth_m"]
    assert depth[0, 0] > .001
    assert depth[1, 0] == 0
    assert objective.parts["penetration"][0] > objective.parts["penetration"][1]
    # All guided palm points themselves lie outside the sphere here.
    assert bool((objective.parts["signed_distance_m"][0] > 0).all())


def test_required_palm_still_has_a_cost_when_topk_can_discard_it(mesh_path):
    objective = TemplatePoseObjective(mesh_path, tiny_settings(), device="cpu")
    q = objective.cpu.qpos0.copy()
    q[0] = .4  # Keep the actual hand away; this test isolates template scoring.
    data = mujoco.MjData(objective.cpu)
    data.qpos[:] = q
    mujoco.mj_kinematics(objective.cpu, data)
    names = ("right_finger2_link4",) * 9 + ("right_palm_link",)
    desired = np.array([[.03, 0., 0.]] * 9 + [[0., 0., .1]])
    outward = np.array([[1., 0., 0.]] * 9 + [[0., 0., 1.]])
    points, normals = [], []
    for name, point, normal in zip(names, desired, outward):
        body = objective.cpu.body(name).id
        rotation = data.xmat[body].reshape(3, 3)
        points.append((point - data.xpos[body]) @ rotation)
        normals.append(-normal @ rotation)
    template = TemplateContacts(names, points, normals, np.zeros(20),
        (("right_finger2_link4",), ("right_palm_link",)), {"test": "required palm"})
    objective.bind(template, q)
    objective.evaluate(torch.tensor(q, dtype=torch.float32)[None, None])
    assert 9 not in objective.parts["selected_slots"][0].tolist()
    assert objective.parts["required_group"][0] > 100.
    assert objective.parts["required_groups_fitted"][0].tolist() == [True, False]


def test_invalid_candidates_do_not_feed_nan_to_fk_or_become_finite(mesh_path):
    objective = TemplatePoseObjective(mesh_path, tiny_settings(), device="cpu")
    _, q = bind_synthetic(objective)
    candidates = np.tile(q, (3, 1))
    candidates[1, 0] = np.nan
    candidates[2, 6] = 100.
    costs = objective.evaluate(torch.tensor(candidates, dtype=torch.float32)[:, None])
    assert torch.isfinite(costs[0])
    assert bool(torch.isinf(costs[1:]).all())


def test_hand_hand_collision_is_penalized_with_the_object_far_away(mesh_path):
    objective = TemplatePoseObjective(mesh_path, tiny_settings(), device="cpu")
    _, q = bind_synthetic(objective)
    q[0] = .4
    q[6:] = [.956, .382, 1.075, -.363, -.845, .623, .881, -1.040,
              1.336, .677, -.148, 1.082, -.831, -.086, 1.521, .023,
              .308, -.535, 1.510, .256]
    objective.evaluate(torch.tensor(q, dtype=torch.float32)[None, None])
    assert objective.parts["penetration_depth_m"][0, 0] == 0
    assert objective.parts["penetration_depth_m"][0, 1] > .005


def test_diagnostics_separate_nominal_preload_response_and_physical_acceptance(mesh_path):
    objective = TemplatePoseObjective(mesh_path, tiny_settings(), device="cpu")
    _, q = bind_synthetic(objective)
    detail = objective.diagnostics(q)
    assert detail["search_dimension"] == 26 and detail["preload_dimension"] == 2
    assert len(detail["scenario_residuals"]) == 13
    assert len(detail["nominal_preloads_n"]) == 2
    assert len(detail["response_normal_forces_n"]) == 13
    assert np.shape(detail["response_contact_forces_n"]) == (13, 2, 3)
    assert np.shape(detail["response_intrinsic_torques_nm"]) == (13, 2, 3)
    assert np.shape(detail["response_torsional_torques_nm"]) == (13, 2)
    assert np.all(np.abs(detail["response_torsional_torques_nm"])
                  <= detail["torsional_friction_m"] * np.asarray(detail["response_normal_forces_n"]) + 1.e-6)
    assert detail["actual_contact_required_for_projection"] is False
    assert detail["actual_object_integrated"] is False
    assert "no native physical evaluation" in detail["acceptance"]
    json.dumps(detail)


def test_final_preload_polish_leaves_q_and_fast_solver_unchanged(mesh_path):
    objective = TemplatePoseObjective(mesh_path, tiny_settings(), device="cpu")
    _, q = bind_synthetic(objective)
    before = q.copy()
    original = objective.closure_solver
    result = objective.final_diagnostics(q)
    np.testing.assert_array_equal(q, before)
    assert objective.closure_solver is original
    assert result["preload_polish"]["q_changed"] is False
    assert result["preload_polish"]["final_gn_iterations"] == 8
    assert result["valid"]


def test_final_preload_polish_never_discards_an_existing_certificate(mesh_path, monkeypatch):
    objective = TemplatePoseObjective(mesh_path, tiny_settings(), device="cpu")
    _, q = bind_synthetic(objective)
    original = objective.closure_solver
    def diagnose(value):
        return dict(projected_closure=objective.closure_solver is original,
                    valid=True, cost=2. if objective.closure_solver is original else 1.)
    monkeypatch.setattr(objective, "diagnostics", diagnose)
    result = objective.final_diagnostics(q)
    assert result["projected_closure"]
    assert result["preload_polish"]["certificate_preserved"]
    assert objective.closure_solver is original


@pytest.mark.parametrize("bad", [
    {"stages": True}, {"contact_fraction": 1.1}, {"normal_tolerance_deg": 180},
    {"penetration_weight": float("nan")}, {"closure": []},
])
def test_invalid_objective_settings_fail_before_allocating_physics(bad):
    with pytest.raises(ValueError):
        TemplateMPPISettings(**bad)


def test_template_route_validation_and_results_are_not_overwritten(tmp_path, monkeypatch):
    from retarget.common.results import prepare_output
    from retarget.config import run as configuration
    raw = yaml.safe_load((ROOT / "configs/grasp_template_mppi.yaml").read_text())
    raw["task"]["data"] = str(ROOT / "data/grasp")
    for environment in ({"dynamics": "mjwarp", "template_mppi": {}, "spider": {}},
                        {"template_mppi": {}},
                        {"dynamics": "mjwarp", "template_mppi": []}):
        raw["environment"] = environment
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match="template_mppi"):
            load(path)
    (tmp_path / "output/keep").mkdir(parents=True)
    original = tmp_path / "output/keep/existing.txt"
    original.write_text("existing output")
    monkeypatch.setattr(configuration, "ROOT", tmp_path)
    config = SimpleNamespace(destination=original.parent, environment={"template_mppi": {}})
    with pytest.raises(FileExistsError, match="choose a new output.name"):
        prepare_output(config)
    assert original.read_text() == "existing output"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native batched FK requires CUDA")
def test_cuda_fk_collision_padding_seeded_mppi_and_cpu_agree(mesh_path):
    settings = tiny_settings()
    cpu = TemplatePoseObjective(mesh_path, settings, device="cpu")
    gpu = TemplatePoseObjective(mesh_path, settings, device="cuda:0")
    _, q = bind_synthetic(cpu)
    template, _ = bind_synthetic(gpu)
    gpu.set_batch_size(4)
    poses = np.tile(q, (3, 1))
    poses[:, 0] = [0., .15, -.01]
    expected = cpu.evaluate(torch.tensor(poses, dtype=torch.float32)[:, None])
    actual = gpu.evaluate(torch.tensor(poses, dtype=torch.float32, device="cuda:0")[:, None])
    np.testing.assert_allclose(gpu.parts["hand_points"].cpu(), cpu.parts["hand_points"], atol=2e-7)
    np.testing.assert_allclose(gpu.parts["clearance_deficit_m"].cpu(), cpu.parts["clearance_deficit_m"], atol=2e-5)
    assert bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all())
    repeated = gpu.evaluate(torch.tensor(poses[:1], dtype=torch.float32, device="cuda:0")[:, None])
    torch.testing.assert_close(actual[:1], repeated, rtol=1e-5, atol=1e-5)
    solver = Settings(type="mppi", population=20, worlds=4, iterations=1,
                      improvement_check_steps=2)
    a, first = gpu.optimize(q, solver, 17)
    b, second = gpu.optimize(q, solver, 17)
    np.testing.assert_array_equal(a, b)
    assert first["final_cost"] <= first["initial_cost"]
    assert first["sampled"] == second["sampled"] == 20
    assert not bool(wp.to_torch(gpu.physics.data.time).any())
    np.testing.assert_array_equal(template.local_points.astype(np.float32), gpu.local.cpu())


def test_coincident_parallel_projections_cannot_fake_antipodal_alignment():
    from retarget.env.grasp.template_mppi import antipodal_contact_term
    p=torch.zeros(1,2,3)
    n=torch.tensor([[[1.,0.,0.],[1.,0.,0.]]])
    assert antipodal_contact_term(p,n,.002).item()==pytest.approx(20.)
    n[:,1]*=-1
    assert antipodal_contact_term(p,n,.002).item()==0
