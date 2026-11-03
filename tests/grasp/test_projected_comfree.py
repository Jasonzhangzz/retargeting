"""Projected closure uses fixed virtual contacts and actual wrench witnesses."""

from tests._paths import ROOT as _PROJECT_ROOT, TEST_ROOT as _TEST_ROOT

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from retarget.env.grasp.projected_comfree import ClosureSettings, ProjectedComFreeClosure


def sphere(dtype=torch.float64, device="cpu"):
    normals = torch.cat((torch.eye(3), -torch.eye(3))).to(device=device, dtype=dtype)
    return (normals * .05)[None], normals[None]


def solver(settings=None):
    # These regression fixtures exercise the original finite-horizon model.
    settings = replace(settings or ClosureSettings(), response_mode="finite_horizon", torsional_friction_m=0.)
    return ProjectedComFreeClosure(settings, center=[0., 0., 0.], radius=.05,
                                   object_mass=.1, object_inertia=[.0001] * 3)


def static_solver(settings=None):
    settings = replace(settings or ClosureSettings(), torsional_friction_m=0.)
    return ProjectedComFreeClosure(settings, center=[0., 0., 0.], radius=.05,
                                   object_mass=.1, object_inertia=[.0001] * 3)


def regression_case(kind, *, dtype=torch.float64, device="cpu", settings=None, torsional_friction_m=0.):
    raw = json.loads((_TEST_ROOT / "fixtures/projected_comfree_regression.json").read_text())[str(kind)]
    settings = replace(settings or ClosureSettings(), torsional_friction_m=torsional_friction_m)
    closure = ProjectedComFreeClosure(settings, **{name: raw[name] for name in
        ("center", "radius", "object_mass", "object_inertia")})
    return closure, torch.tensor(raw["points"], dtype=dtype, device=device)[None], torch.tensor(raw["outward_normals"], dtype=dtype, device=device)[None]


def test_symmetric_contacts_support_all_axes_and_same_side_does_not():
    points, normal = sphere()
    closure = solver()
    good = closure.probe(points, normal, torch.ones(1, 6, dtype=points.dtype))
    bad = closure.probe(points, normal.new_tensor([[[1., 0., 0.]]]).expand_as(normal), torch.ones(1, 6, dtype=points.dtype))
    assert good.valid.all() and good.certified.all()
    assert good.scenario_rms[:, 1:].max() < .05
    assert not bad.certified.any()
    assert good.cost.item() < bad.cost.item()
    assert good.certified_margin.item() > 0
    assert good.row_forces.shape == (1, 13, 6, 4)
    assert good.metadata["row_mass"] == "effective_ray_mass_approximation"


def test_two_contacts_cannot_support_torque_about_their_common_axis():
    points, normal = sphere()
    index = torch.tensor([0, 3])
    result = solver().solve(points[:, index], normal[:, index])
    assert result.valid.all()
    assert not result.certified.any()
    assert result.scenario_rms[0, 4] >= .99  # positive x torque


def test_zero_preload_does_not_certify_from_zero_baseline_alone():
    points, normal = sphere()
    result = solver().probe(points, normal, torch.zeros(1, 6, dtype=points.dtype))
    assert result.scenario_rms[0, 0] == 0
    assert result.scenario_rms[0, 1:].max() > .05
    assert not result.certified.any()


def test_gn_actually_improves_underloaded_contacts():
    points, normals = sphere()
    result = solver(ClosureSettings(initial_preload_n=.01, gn_iterations=3)).solve(points, normals)
    assert result.cost.item() < result.objective_history[0, 0].item() * .1
    assert (result.preloads > .01).any()
    assert (result.objective_history.diff(dim=1) <= 0).all()


def test_exact_forward_sensitivity_matches_autograd_and_finite_differences():
    points, normal = sphere()
    # Asymmetric preload stays away from the clamp boundaries of this short probe.
    settings = ClosureSettings(horizon=9, tail_steps=3, gn_iterations=1)
    closure = solver(settings)
    preload = torch.tensor([[.8, 1.1, 1.4, 1.2, .9, 1.3]], dtype=torch.float64, requires_grad=True)
    geometry = closure._geometry(points, normal)
    residual, analytic, *_ = closure._rollout(preload, geometry, jacobian=True)
    automatic = torch.autograd.functional.jacobian(lambda value: closure._rollout(value, geometry)[0], preload)
    torch.testing.assert_close(analytic, automatic[:, :, 0], atol=1.e-8, rtol=2.e-7)
    assert torch.autograd.gradcheck(lambda value: closure.probe(points, normal, value).cost,
                                    (preload,), atol=1.e-5, rtol=1.e-4)
    assert residual.isfinite().all()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batch_equals_individual_and_gn_is_bounded_and_monotone(device, dtype):
    if device != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    points, normals = sphere(device=device, dtype=dtype)
    points = torch.cat((points, points * .9))
    normals = normals.expand_as(points)
    settings = ClosureSettings(gn_iterations=3, max_preload_n=1.5, initial_preload_n=2.)
    closure = solver(settings)
    batch = closure.solve(points, normals)
    individuals = [closure.solve(points[i:i+1], normals[i:i+1]) for i in range(2)]
    torch.testing.assert_close(batch.cost, torch.cat([result.cost for result in individuals]))
    torch.testing.assert_close(batch.preloads, torch.cat([result.preloads for result in individuals]))
    assert (batch.preloads >= 0).all() and (batch.preloads <= settings.max_preload_n).all()
    assert (batch.objective_history.diff(dim=1) <= 1.e-12).all()
    assert batch.valid.all()


def test_nonfinite_sample_is_rejected_without_contaminating_the_batch():
    points, normals = sphere()
    points = points.expand(2, -1, -1).clone()
    normals = normals.expand_as(points).clone()
    points[1, 0, 0] = torch.nan
    closure = solver(ClosureSettings(gn_iterations=1))
    result = closure.solve(points, normals)
    assert result.valid.tolist() == [True, False]
    assert result.cost[1] == closure.settings.invalid_cost
    assert not result.certified[1]
    assert torch.isfinite(result.cost).all()


def test_certificate_agrees_with_actual_tail_mean_ray_wrenches():
    points, normals = sphere()
    closure = solver()
    result = closure.probe(points, normals, torch.ones(1, 6, dtype=points.dtype))
    A, _, _, _, _, loads, _ = closure._geometry(points, normals)
    error = (torch.einsum("brj,bsr->bsj", A, result.row_forces.flatten(2)) + loads) / closure.test_force_n
    torch.testing.assert_close(error, result.scenario_residuals)
    epsilon = error[:, 1:].norm(dim=-1).amax(1)
    torch.testing.assert_close(result.certified_margin, closure.test_force_n * (1 / 6 ** .5 - epsilon))
    # A convex combination of valid nonnegative friction rays remains physical.
    assert (result.row_forces >= 0).all()


def test_default_static_equilibrium_corrects_the_old_dynamic_false_negative():
    closure, points, normals = regression_case(11)
    preloads = torch.full(points.shape[:2], 2., dtype=points.dtype)
    static = closure.probe(points, normals, preloads)
    old, _, _ = regression_case(11, settings=ClosureSettings(response_mode="finite_horizon"))
    dynamic = old.probe(points, normals, preloads)
    assert static.valid.all() and static.certified.all()
    assert static.scenario_rms.max() < 1.e-5
    assert not dynamic.certified.any()
    assert dynamic.scenario_rms.max() > .05
    assert static.metadata["response_version"] == 2
    assert dynamic.metadata["response_version"] == 1
    assert static.metadata["row_force_statistic"] == "static equilibrium response"


def test_static_force_effort_gives_a_useful_preload_objective():
    points, normals = sphere()
    closure = static_solver(ClosureSettings(cuda_graph=False, gn_iterations=4))
    result = closure.solve(points, normals)
    assert result.valid.all() and result.certified.all()
    assert result.cost.item() < result.objective_history[0, 0].item() * .8
    assert (result.objective_history.diff(dim=1) <= 0).all()


@pytest.mark.parametrize("kind", [1, 6, 9, 10, 11])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_static_regression_matches_independent_friction_ray_linear_programs(kind, dtype):
    from scipy.optimize import linprog

    closure, points, normals = regression_case(kind, dtype=dtype, settings=ClosureSettings(cuda_graph=False))
    geometry = closure._geometry(points, normals)
    rays, _, _, _, _, loads, _ = geometry
    feasible = all(linprog(torch.ones(rays.shape[1]).numpy(), A_eq=rays[0].double().numpy().T,
                          b_eq=-load.double().numpy(), bounds=(0., None), method="highs").success
                   for load in loads[1:])
    result = closure.solve(points, normals)
    assert feasible == (kind != 9)
    assert result.valid.all()
    assert result.certified.item() == feasible
    assert (result.objective_history.diff(dim=1) <= 0).all()
    if feasible:
        assert result.scenario_rms.max() < 1.e-4
    else:
        assert result.scenario_rms.max() > .05


def test_zero_nominal_preload_requires_nonzero_balancing_forces_in_every_load_scenario():
    points, normals = sphere()
    result = static_solver().probe(points, normals, torch.zeros(1, 6, dtype=points.dtype))
    assert result.valid.all() and result.certified.all()
    assert result.row_forces[:, 1:].sum(-1).sum(-1).min() > 0
    assert result.scenario_rms.max() < 1.e-5
    # Zero baseline alone still cannot certify a one-sided contact arrangement.
    one_side = normals.new_tensor([1., 0., 0.]).expand_as(normals)
    bad = static_solver().probe(points, one_side, torch.zeros(1, 6, dtype=points.dtype))
    assert bad.valid.all() and not bad.certified.any()


def test_static_forward_sensitivity_matches_autograd_with_force_effort():
    points, normals = sphere()
    closure = static_solver(ClosureSettings(equilibrium_iterations=8))
    preload = torch.tensor([[.8, 1.1, 1.4, 1.2, .9, 1.3]], dtype=torch.float64, requires_grad=True)
    geometry = closure._geometry(points, normals)
    _, analytic, *_ = closure._rollout(preload, geometry, jacobian=True)
    automatic = torch.autograd.functional.jacobian(lambda value: closure._rollout(value, geometry)[0], preload)
    torch.testing.assert_close(analytic, automatic[:, :, 0], atol=1.e-8, rtol=2.e-6)
    assert torch.autograd.gradcheck(lambda value: closure.probe(points, normals, value).cost,
                                    (preload,), atol=1.e-5, rtol=1.e-4)


def test_soft_finger_torsion_closes_the_actual_two_contact_pinch():
    soft, points, normals = regression_case(9, torsional_friction_m=.02)
    result = soft.solve(points, normals)
    assert result.valid.all() and result.certified.all()
    assert result.scenario_rms.max() < 1.e-5
    assert result.preloads.shape == (1, 2)
    assert result.row_forces.shape == (1, 13, 2, 6)
    assert result.metadata["response_version"] == 3
    assert result.metadata["ray_count_per_contact"] == 6
    assert result.metadata["ray_order"][4:] == ["+normal_spin", "-normal_spin"]
    assert result.metadata["torsional_friction_m"] == .02
    legacy, _, _ = regression_case(9, torsional_friction_m=0.)
    old = legacy.solve(points, normals)
    assert old.valid.all() and not old.certified.any()
    assert old.row_forces.shape[-1] == 4 and old.metadata["response_version"] == 2


def test_decoded_soft_contact_wrenches_obey_coupled_friction_and_rebuild_grasp_wrench():
    points, normals = sphere()
    closure = ProjectedComFreeClosure(center=[.001, -.002, .003], radius=.05,
        object_mass=.1, object_inertia=[.0001] * 3)
    generator = torch.Generator().manual_seed(517)
    weights = torch.rand(1, 7, 6, 6, dtype=points.dtype, generator=generator)
    weights[:, :6] = torch.eye(6, dtype=points.dtype)[None, :, None, :]
    force, intrinsic = closure.contact_wrenches(weights, normals)
    inward = -normals[:, None]
    normal_force = (force * inward).sum(-1)
    tangential = force - normal_force[..., None] * inward
    spin = (intrinsic * inward).sum(-1)
    torch.testing.assert_close(normal_force, weights.sum(-1))
    assert (tangential.norm(dim=-1) / closure.settings.friction +
            spin.abs() / closure.settings.torsional_friction_m <= normal_force + 1.e-12).all()
    torch.testing.assert_close(torch.linalg.cross(intrinsic, inward.expand_as(intrinsic)), torch.zeros_like(intrinsic))
    torch.testing.assert_close(spin[:, 4], torch.full_like(spin[:, 4], .02))
    torch.testing.assert_close(spin[:, 5], torch.full_like(spin[:, 5], -.02))
    A, *_ = closure._geometry(points, normals)
    rebuilt = torch.cat((force.sum(2), (torch.linalg.cross(points[:, None] - closure.center.to(points), force) + intrinsic).sum(2) / closure.radius), -1)
    torch.testing.assert_close(rebuilt, torch.einsum("brj,bsr->bsj", A, weights.flatten(2)))
    one_force, one_torque = closure.contact_wrenches(weights[:, 0], normals)
    torch.testing.assert_close(one_force, force[:, 0])
    torch.testing.assert_close(one_torque, intrinsic[:, 0])


@pytest.mark.parametrize("response_mode", ["equilibrium", "finite_horizon"])
def test_six_ray_response_sensitivity_and_gradcheck(response_mode):
    points, normals = sphere()
    closure = ProjectedComFreeClosure(ClosureSettings(response_mode=response_mode, horizon=9, tail_steps=3),
        center=[0., 0., 0.], radius=.05, object_mass=.1, object_inertia=[.0001] * 3)
    preload = torch.tensor([[.8, 1.1, 1.4, 1.2, .9, 1.3]], dtype=torch.float64, requires_grad=True)
    geometry = closure._geometry(points, normals)
    _, analytic, *_ = closure._rollout(preload, geometry, jacobian=True)
    automatic = torch.autograd.functional.jacobian(lambda value: closure._rollout(value, geometry)[0], preload)
    torch.testing.assert_close(analytic, automatic[:, :, 0], atol=1.e-8, rtol=2.e-6)
    assert torch.autograd.gradcheck(lambda value: closure.probe(points, normals, value).cost,
                                    (preload,), atol=1.e-5, rtol=1.e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("response_mode", ["equilibrium", "finite_horizon"])
def test_six_ray_cuda_graph_matches_eager(response_mode):
    closure, points, normals = regression_case(9, dtype=torch.float32, device="cuda",
        settings=ClosureSettings(response_mode=response_mode, gn_iterations=2, horizon=12, tail_steps=4),
        torsional_friction_m=.02)
    points, normals = points.expand(3, -1, -1), normals.expand(3, -1, -1)
    graph = closure.solve(points, normals)
    eager = closure.solve_eager(points, normals)
    for name in ("cost", "preloads", "row_forces", "scenario_residuals", "scenario_rms", "certified_margin", "objective_history"):
        torch.testing.assert_close(getattr(graph, name), getattr(eager, name), atol=2.e-6, rtol=2.e-5)
    assert torch.equal(graph.certified, eager.certified)
    assert graph.row_forces.shape == (3, 13, 2, 6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("contacts", [2, 6, 16])
@pytest.mark.parametrize("response_mode", ["equilibrium", "finite_horizon"])
def test_cuda_graph_matches_eager_and_replays_new_inputs_without_mutating_outputs(contacts, response_mode):
    settings = ClosureSettings(response_mode=response_mode, initial_preload_n=.01, gn_iterations=2, horizon=12, tail_steps=4)
    closure = static_solver(settings)
    generator = torch.Generator().manual_seed(1782)
    normals = torch.randn(3, contacts, 3, generator=generator).cuda()
    normals = torch.nn.functional.normalize(normals, dim=-1)
    points = normals * .05
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    graph = closure.solve(points, normals)
    eager = closure.solve_eager(points, normals)
    for name in ("cost", "preloads", "row_forces", "scenario_residuals", "scenario_rms",
                 "certified_margin", "objective_history"):
        torch.testing.assert_close(getattr(graph, name), getattr(eager, name), atol=2.e-6, rtol=2.e-5)
    for name in ("valid", "converged", "certified"):
        assert torch.equal(getattr(graph, name), getattr(eager, name))
    snapshot = graph.cost.clone()
    shifted = points.clone()
    shifted[0, 0, 0] = torch.nan
    updated = closure.solve(shifted, normals)
    expected = closure.solve_eager(shifted, normals)
    torch.testing.assert_close(updated.cost, expected.cost, atol=2.e-6, rtol=2.e-5)
    assert not updated.valid[0]
    assert graph.cost.data_ptr() != updated.cost.data_ptr()
    torch.testing.assert_close(graph.cost, snapshot)
    assert len(closure._graph_cache) == 1
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    assert graph.metadata["execution"] == "cuda_graph"
    assert graph.metadata["cuda_graph_capture_seconds"] > 0
    closure.clear_cuda_graph_cache()
    assert not closure._graph_cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_graph_can_be_disabled_and_probe_remains_differentiable():
    closure = solver(ClosureSettings(cuda_graph=False, gn_iterations=1, horizon=4, tail_steps=2))
    points, normals = sphere(device="cuda")
    assert closure.solve(points, normals).valid.all()
    assert not closure._graph_cache
    preload = torch.ones(1, 6, device="cuda", dtype=points.dtype, requires_grad=True)
    cost = closure.probe(points, normals, preload).cost
    assert cost.requires_grad
    gradient, = torch.autograd.grad(cost.sum(), preload)
    assert gradient.isfinite().all()


@pytest.mark.parametrize("setting", [dict(timestep=0), dict(friction=-1), dict(horizon=0),
                                     dict(tail_steps=65), dict(impedance=1), dict(test_force_n=0), dict(cuda_graph="yes"),
                                     dict(response_mode="unknown"), dict(equilibrium_iterations=0),
                                     dict(torsional_friction_m=-.01)])
def test_settings_reject_invalid_parameters(setting):
    with pytest.raises(ValueError):
        ClosureSettings(**setting)
