"""SCSP checks: finite contact CEM, force derivatives, ADMM and IPOPT."""

import numpy as np
import pytest
import torch

from retarget.solver.base import NoValidCandidate
from retarget.solver.scsp import (
    ObjectEvaluation, ObjectSelection, RobotEvaluation, SCSPProblem,
    SCSPSettings, _TorchObjective, _augmented, _force_optimize, _object_cem, solve,
)

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))]


def _problem(device="cpu", seen=None):
    mean = torch.zeros(1, dtype=torch.float64, device=device)
    points = mean.new_tensor([-1.0, 1.0, 3.0])

    def object_evaluate(indices, forces):
        if seen is not None:
            seen.append((indices.detach().clone(), forces.detach().clone(), torch.is_grad_enabled()))
        point = points[indices[:, 0]]
        consensus = torch.stack((point, forces[:, 0]), -1)
        cost = (point - 1).square() + (forces[:, 0] - 2).square()
        return ObjectEvaluation(cost, consensus, {"point": point, "force": forces})

    def robot_evaluate(action, selection):
        return RobotEvaluation(
            0.1 * (action[0] - 1).square(), torch.cat((action, 2 * action))
        )

    return SCSPProblem(
        mean=mean, candidate_counts=torch.tensor([3], device=device),
        force_mean=mean.new_tensor([2.0]), initial_consensus=mean.new_zeros(2),
        object_evaluate=object_evaluate, robot_evaluate=robot_evaluate,
        force_project=lambda indices, force: force.clamp(0, 4),
        lower=mean.new_tensor([-4.0]), upper=mean.new_tensor([4.0]),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_only_contact_combinations_are_sampled_and_catalog_is_deduplicated(device, monkeypatch):
    problem = _problem(device)
    problem.candidate_counts = torch.tensor([11, 11], device=device)

    def evaluate(indices, forces):
        target = indices.new_tensor([7, 9])
        contact_cost = (indices - target).square().sum(-1).to(forces)
        cost = contact_cost + (forces[:, 0] - 2).square()
        return ObjectEvaluation(cost, torch.stack((forces[:, 0], forces[:, 0]), -1))

    def forbidden_noise(*args, **kwargs):
        raise AssertionError("SCSP must never sample force noise")

    monkeypatch.setattr(torch, "randn", forbidden_noise)
    monkeypatch.setattr(torch, "randn_like", forbidden_noise)
    problem.object_evaluate = evaluate
    settings = SCSPSettings(candidate_population=24, candidate_iterations=6, force_iterations=2)
    target = problem.mean.new_tensor([2.0, 2.0])

    def search():
        with torch.no_grad():
            return _object_cem(
                problem, settings, problem.candidate_counts, target, torch.zeros_like(target),
                torch.Generator(device=device).manual_seed(8), None,
            )

    first, sampled, evaluated = search()
    visited = problem._object_search_statistics["visited_indices"].clone()
    second, _, _ = search()
    assert first.indices.tolist() == [7, 9]
    assert first.cost == 0
    torch.testing.assert_close(first.indices, second.indices, atol=0, rtol=0)
    torch.testing.assert_close(first.forces, second.forces, atol=0, rtol=0)
    assert sampled == 121 and len(torch.unique(visited, dim=0)) == 121
    assert len(visited) == 121 and evaluated > sampled
    assert problem._object_search_statistics["contact_universe"] == 121


@pytest.mark.parametrize("device", DEVICES)
def test_projected_force_gradients_solve_batched_quadratics_monotonically(device):
    problem = _problem(device)
    problem.force_mean.zero_()
    targets = problem.mean.new_tensor([0.7, 1.4, 2.1])
    gradients_seen = []

    def evaluate(indices, force):
        target = targets[indices[:, 0]]
        cost = (force[:, 0] - target).square()
        if torch.is_grad_enabled():
            gradients_seen.append((indices[:, 0].detach().clone(), cost.detach().clone()))
        return ObjectEvaluation(cost, torch.zeros_like(force).expand(-1, 2))

    problem.object_evaluate = evaluate
    settings = SCSPSettings(force_iterations=60, force_step_size=0.2, force_backtracks=10, force_tolerance=1e-7)
    with torch.no_grad():
        force, score, metrics = _force_optimize(
            problem, settings, torch.tensor([[0], [1], [2]], device=device),
            problem.initial_consensus, torch.zeros_like(problem.initial_consensus),
        )
    torch.testing.assert_close(force[:, 0], targets, atol=2e-3, rtol=0)
    assert bool((score < 4e-6).all())
    assert metrics["force_gradient_steps"] > 0
    for contact in range(3):
        costs = [float(cost[ids == contact][0]) for ids, cost in gradients_seen if bool((ids == contact).any())]
        assert all(after <= before + 1e-12 for before, after in zip(costs, costs[1:]))


def test_nan_projected_trial_is_rejected_even_if_callback_ignores_bad_force():
    problem = _problem()
    problem.force_mean.zero_()
    seen_invalid = []

    def project(indices, force):
        value = torch.where(force > 1.5, torch.nan, force.clamp_min(0))
        seen_invalid.append(bool(torch.isnan(value).any()))
        return value

    def evaluate(indices, force):
        safe = torch.nan_to_num(force[:, 0], nan=1.4)
        cost = (safe - 1.4).square()
        return ObjectEvaluation(cost, torch.stack((safe * 0, safe * 0), -1))

    problem.force_project = project
    problem.object_evaluate = evaluate
    settings = SCSPSettings(force_iterations=30, force_step_size=2.0, force_backtracks=12)
    with torch.no_grad():
        force, scores, _ = _force_optimize(
            problem, settings, torch.tensor([[0], [1]]),
            problem.initial_consensus, torch.zeros_like(problem.initial_consensus),
        )
    assert any(seen_invalid)
    assert bool(torch.isfinite(force).all()) and bool(torch.isfinite(scores).all())
    torch.testing.assert_close(force[:, 0], problem.mean.new_full((2,), 1.4), atol=0.01, rtol=0)


def test_force_solve_requires_derivatives_connected_to_forces():
    problem = _problem()
    problem.object_evaluate = lambda indices, force: ObjectEvaluation(
        torch.zeros(len(indices), dtype=force.dtype), torch.zeros(len(indices), 2, dtype=force.dtype)
    )
    with pytest.raises(ValueError, match="differentiable object_evaluate"):
        _force_optimize(
            problem, SCSPSettings(), torch.tensor([[0]]),
            problem.initial_consensus, torch.zeros_like(problem.initial_consensus),
        )


def test_ipopt_gradient_matches_central_difference():
    problem = _problem()

    def nonlinear(action, selection):
        return RobotEvaluation(
            (action[0] ** 3 - 0.3).square(),
            torch.stack((action[0].sin(), action[0].exp())),
        )

    problem.robot_evaluate = nonlinear
    selection = ObjectSelection(torch.tensor([1]), torch.tensor([2.0]), torch.tensor([0.2, 1.2]), 0.0)
    objective = _TorchObjective(problem, selection, torch.tensor([0.1, -0.1]), 3.0)
    point = np.array([0.45])
    gradient = objective.gradient(point).copy()
    step = 1e-6
    finite_difference = (objective.objective(point + step) - objective.objective(point - step)) / (2 * step)
    np.testing.assert_allclose(gradient, [finite_difference], atol=1e-8, rtol=1e-6)


def _masked_problem():
    problem = _problem()
    problem.candidate_counts = torch.tensor([1])
    problem.initial_consensus = torch.tensor([float("nan"), 1e9, 0.0], dtype=torch.float64)
    problem.consensus_mask = torch.tensor([False, False, True])
    problem.force_mean.fill_(1.0)

    def object_evaluate(indices, forces):
        inactive = torch.full_like(forces[:, 0], float("nan"))
        consensus = torch.stack((inactive, forces[:, 0] * 0 - 1e9, forces[:, 0]), -1)
        return ObjectEvaluation((forces[:, 0] - 1).square(), consensus)

    def robot_evaluate(actions, selection):
        consensus = torch.stack((actions.new_tensor(float("nan")), actions[0] * 1e12, actions[0]))
        return RobotEvaluation((actions[0] - 1).square(), consensus)

    problem.object_evaluate, problem.robot_evaluate = object_evaluate, robot_evaluate
    return problem


def test_inactive_position_and_nan_do_not_enter_objective_or_action_gradient():
    problem = _masked_problem()
    selection = ObjectSelection(torch.tensor([0]), torch.tensor([1.0]), torch.tensor([float("nan"), -1e9, 0.5]), 0.0)
    dual = torch.tensor([float("nan"), 1e20, 0.1], dtype=torch.float64)
    objective = _TorchObjective(problem, selection, dual, rho=3.0)
    point = np.array([0.2])
    expected = (0.2 - 1) ** 2 + 1.5 * (0.5 - 0.2 + 0.1) ** 2
    derivative = 2 * (0.2 - 1) - 3 * (0.5 - 0.2 + 0.1)
    np.testing.assert_allclose(objective.objective(point), expected, atol=1e-7)
    np.testing.assert_allclose(objective.gradient(point), [derivative], atol=1e-7)
    cost = _augmented(torch.tensor([2.0]), selection.consensus[None],
                      torch.tensor([float("nan"), 1e9, 0.2]), dual, 3.0,
                      problem.consensus_mask)
    assert torch.isfinite(cost).all()
    np.testing.assert_allclose(cost.numpy(), [2 + 1.5 * 0.4**2], atol=1e-7)


def test_inactive_fixed_position_does_not_block_admm_convergence():
    pytest.importorskip("cyipopt")
    result = solve(_masked_problem(), SCSPSettings(
        admm_iterations=24, rho=2.0, candidate_population=1, candidate_iterations=1,
        force_iterations=40, force_step_size=0.2, force_backtracks=10,
        abs_tol=0.01, rel_tol=0.01, ipopt_tol=1e-8,
    ))
    assert result.converged
    assert result.history[-1]["constraints_active"] == 1
    assert result.history[-1]["primal_residual"] < 0.03
    torch.testing.assert_close(result.parameters, torch.tensor([1.0], dtype=torch.float64), atol=0.03, rtol=0)
    torch.testing.assert_close(result.dual[:2], torch.zeros(2, dtype=torch.float64), atol=0, rtol=0)


def test_no_active_consensus_still_optimizes_robot_base_cost_before_stopping():
    pytest.importorskip("cyipopt")
    problem = _masked_problem()
    problem.consensus_mask.zero_()

    def robot_evaluate(actions, selection):
        return RobotEvaluation((actions[0] - 2).square(), actions.new_full((3,), float("nan")))

    problem.robot_evaluate = robot_evaluate
    result = solve(problem, SCSPSettings(
        admm_iterations=2, candidate_population=1, candidate_iterations=1, force_iterations=2,
    ))
    assert result.converged and len(result.history) == 1
    assert result.history[0]["ipopt_evaluated"] > 1
    assert result.history[0]["constraints_active"] == 0
    assert result.history[0]["primal_residual"] == result.history[0]["dual_residual"] == 0
    assert result.history[0]["primal_tolerance"] == result.history[0]["dual_tolerance"] == 0
    torch.testing.assert_close(result.parameters, torch.tensor([2.0], dtype=torch.float64), atol=1e-4, rtol=0)
    assert bool((result.dual == 0).all())


def test_prepare_iteration_freezes_mask_per_round_and_resets_changed_multipliers(monkeypatch):
    problem = _problem()
    problem.candidate_counts = torch.tensor([1])
    masks = [torch.tensor([True, True]), torch.tensor([True, False]), torch.tensor([False, True])]
    prepared = []
    object_duals = []

    def prepare(actions):
        assert not torch.is_grad_enabled()
        prepared.append(actions.clone())
        return torch.zeros(2, dtype=actions.dtype), masks[len(prepared) - 1]

    def evaluate(indices, forces):
        return ObjectEvaluation(forces.square().sum(1), forces.new_tensor([1.0, 2.0]).expand(len(forces), -1))

    problem.object_evaluate = evaluate
    problem.prepare_iteration = prepare
    from retarget.solver import scsp
    original = scsp._object_cem

    def record(*args, **kwargs):
        object_duals.append(args[4].clone())
        return original(*args, **kwargs)

    def robot(problem, selection, actions, dual, settings, cyipopt):
        # Keep a deliberate nonzero active discrepancy so all three ADMM
        # rounds execute; this isolates contact activation from IPOPT behavior.
        return actions, RobotEvaluation(actions.new_zeros(()), actions.new_zeros(2)), {
            "ipopt_status": 0, "ipopt_success": True, "ipopt_evaluated": 1,
        }, 1

    monkeypatch.setattr(scsp, "_load_ipopt", lambda: object())
    monkeypatch.setattr(scsp, "_object_cem", record)
    monkeypatch.setattr(scsp, "_robot_ipopt", robot)
    result = solve(problem, SCSPSettings(
        admm_iterations=3, candidate_population=1, candidate_iterations=1, force_iterations=1,
    ))
    assert len(prepared) == 3
    torch.testing.assert_close(object_duals[0], torch.tensor([0.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(object_duals[1], torch.tensor([1.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(object_duals[2], torch.tensor([0.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(result.dual, torch.tensor([0.0, 2.0], dtype=torch.float64))
    assert [row["constraints_active"] for row in result.history] == [2, 1, 1]
    assert [row["prepare_evaluated"] for row in result.history] == [1, 1, 1]


@pytest.mark.parametrize("robot, mask, message", [
    (torch.zeros(2), torch.zeros(2), "Boolean"),
    (torch.zeros(2), torch.ones(3, dtype=torch.bool), "shape/device"),
    (torch.zeros(3), torch.ones(2, dtype=torch.bool), "shape/device"),
    (torch.tensor([float("nan"), 0.0]), torch.ones(2, dtype=torch.bool), "must be finite"),
])
def test_prepare_iteration_validates_active_consensus_and_mask(robot, mask, message, monkeypatch):
    from retarget.solver import scsp
    problem = _problem()
    problem.prepare_iteration = lambda actions: (robot, mask)
    monkeypatch.setattr(scsp, "_load_ipopt", lambda: object())
    with pytest.raises(ValueError, match=message):
        solve(problem, SCSPSettings())


def test_real_ipopt_and_admm_reach_shared_contact_force():
    pytest.importorskip("cyipopt")
    settings = SCSPSettings(
        admm_iterations=24, rho=2.0, abs_tol=0.01, rel_tol=0.01,
        candidate_population=3, candidate_iterations=2, force_iterations=30,
        force_step_size=0.2, force_backtracks=10, ipopt_tol=1e-8,
    )
    with torch.no_grad():
        result = solve(_problem(), settings, seed=42)
    assert result.selection.indices.tolist() == [1]
    assert result.cost < 0.01
    torch.testing.assert_close(result.parameters, torch.tensor([1.0], dtype=torch.float64), atol=0.04, rtol=0)
    assert result.history[-1]["ipopt_success"] and result.converged
    assert result.history[-1]["primal_residual"] < 0.04
    assert result.sampled == 3 < result.evaluated
    assert all(row["contact_combinations"] == 3 and row["contact_universe"] == 3 for row in result.history)
    assert all(row["force_evaluations"] > row["contact_combinations"] for row in result.history)
    assert not result.parameters.requires_grad


def test_ipopt_respects_action_bounds():
    pytest.importorskip("cyipopt")
    problem = _problem()
    problem.upper = torch.tensor([0.5], dtype=torch.float64)
    settings = SCSPSettings(admm_iterations=2, candidate_population=3, candidate_iterations=1, force_iterations=4)
    result = solve(problem, settings, seed=0)
    assert 0.49 < result.parameters.item() <= 0.5
    assert not result.converged


def test_ipopt_cpu_budget_retains_finite_iterate_without_claiming_convergence():
    pytest.importorskip("cyipopt")
    settings = SCSPSettings(
        admm_iterations=1, candidate_population=2, candidate_iterations=1,
        force_iterations=1, ipopt_max_cpu_time=1e-6,
    )
    result = solve(_problem(), settings, seed=0)
    assert result.history[0]["ipopt_status"] == -4
    assert not result.history[0]["ipopt_success"] and not result.converged
    assert torch.isfinite(result.parameters).all() and np.isfinite(result.cost)


def test_invalid_contact_population_cannot_export_a_selection():
    problem = _problem()
    problem.object_evaluate = lambda indices, force: ObjectEvaluation(
        torch.full((len(indices),), torch.inf), torch.zeros(len(indices), 2)
    )
    settings = SCSPSettings(candidate_population=3, candidate_iterations=1, force_iterations=1)
    with torch.no_grad(), pytest.raises(NoValidCandidate):
        _object_cem(
            problem, settings, problem.candidate_counts, problem.initial_consensus,
            torch.zeros_like(problem.initial_consensus), torch.Generator().manual_seed(0), None,
        )


def test_contact_catalog_larger_than_121_rejected():
    problem = _problem()
    problem.candidate_counts = torch.tensor([11, 11, 11])
    with pytest.raises(ValueError, match="at most 121"):
        solve(problem, SCSPSettings())


@pytest.mark.parametrize("options", [
    {"rho": 0}, {"force_iterations": 0}, {"admm_iterations": True},
    {"smoothing": 1.1}, {"force_step_size": float("nan")}, {"ipopt_max_iter": 1.5},
    {"ipopt_max_cpu_time": 0.0}, {"force_backtracks": 0}, {"force_tolerance": 0},
])
def test_invalid_scsp_settings(options):
    with pytest.raises(ValueError):
        SCSPSettings(**options)
