import math
import numpy as np
import pytest
import torch
from retarget.solver.base import Settings, Problem, Evaluation, NoValidCandidate, solve
from retarget.solver.cem import elite_mean
from retarget.solver.mppi import weights, refine_mean


def test_card_update_independent_population_std_and_ceil():
    x = torch.arange(15, dtype=torch.float64).reshape(5, 3)
    cost = torch.tensor([4.0, 2.0, 1.0, 3.0, 5.0], dtype=x.dtype)
    c = np.array([1.0, 2.0])
    w = np.exp(-(c - c.min()) / (c.std() * 0.1))
    w /= w.sum()
    np.testing.assert_allclose(elite_mean(x, cost, 0.21, 0.1), w @ x[[2, 1]].numpy())
    torch.testing.assert_close(elite_mean(x, torch.ones(5), 0.21, 0.1), x[:2].mean(0))
    assert elite_mean(x, torch.full((5,), torch.nan), 0.2, 0.1) is None


def test_mppi_top_decile_uses_sample_std_plus_epsilon():
    costs = torch.arange(20, dtype=torch.float64)
    ids, w = weights(costs, 0.6)
    np.testing.assert_array_equal(ids, [0, 1])
    values = np.array([0.0, 1.0])
    expected = np.exp(-(values - values.mean()) / ((values.std(ddof=1) + 0.01) * 0.6))
    expected /= expected.sum()
    np.testing.assert_allclose(w, expected)


@pytest.mark.parametrize("method", ["cem", "mppi"])
def test_schedule_reproducibility_and_only_finite_hard_candidates(method):
    seen = []

    def score(x):
        seen.append(x.clone())
        c = ((x - 2) ** 2).sum(1)
        c[x[:, 0] < 0] = torch.inf
        return c

    s = Settings(type=method, population=20, iterations=4, improvement_threshold=0.0)
    p = Problem(torch.zeros(3), torch.ones(3), score)
    a = solve(p, s, 4)
    b = solve(p, s, 4)
    torch.testing.assert_close(a.parameters, b.parameters, rtol=0, atol=0)
    assert a.sampled == 80 and a.evaluated == 81 and math.isfinite(a.cost)
    assert [h["noise_ratio"] for h in a.history] == [
        s.noise_ratio ** (i / 4) for i in range(4)
    ]
    assert all(
        h["best_cost"] >= k["best_cost"] for h, k in zip(a.history, a.history[1:])
    )
    assert len(seen[0]) == 20


@pytest.mark.parametrize("method", ["cem", "mppi"])
def test_restoration_cannot_export_hard_invalid_candidate(method):
    def score(x):
        return Evaluation(torch.full((len(x),), torch.inf), x.square().sum(1))

    with pytest.raises(NoValidCandidate):
        solve(
            Problem(torch.zeros(2), torch.ones(2), score),
            Settings(type=method, population=20, iterations=2),
        )


def test_mppi_early_stop_counts_actual_budget():
    r = solve(
        Problem(torch.zeros(2), torch.ones(2), lambda x: torch.zeros(len(x))),
        Settings(type="mppi", population=20, iterations=16),
        0,
    )
    assert len(r.history) == 1 and r.sampled == 20 and r.evaluated == 21


def test_gradient_mppi_improves_same_objective_and_counts_all_evaluations():
    seen = []

    def score(x):
        seen.append((len(x), torch.is_grad_enabled()))
        return (x - 1).square().sum(1)

    problem = Problem(torch.zeros(2), torch.full((2,), 0.01), score)
    common = dict(type="mppi", population=20, iterations=3, improvement_threshold=0.0)
    baseline = solve(problem, Settings(**common), 4)
    assert all(not enabled for _, enabled in seen)
    seen.clear()
    # solve() normally runs under no_grad. Only mean-gradient evaluations need
    # an autograd graph; population scoring and backtracking remain no_grad.
    with torch.no_grad():
        guided = solve(
            problem,
            Settings(**common, gradient_steps=2, gradient_step_size=40),
            4,
        )
    assert guided.cost < baseline.cost * 0.1
    assert guided.sampled == baseline.sampled == 60
    assert guided.evaluated == sum(count for count, _ in seen)
    assert sum(enabled for _, enabled in seen) > 0
    assert all(count == 1 for count, enabled in seen if enabled)
    assert guided.evaluated == guided.sampled + 1 + sum(
        row["gradient_evaluated"] for row in guided.history
    )
    assert all(
        row["gradient_final_cost"] <= row["gradient_initial_cost"]
        for row in guided.history
    )
    assert not guided.parameters.requires_grad


def test_gradient_refinement_projects_and_backtracks_invalid_trials():
    def score(x):
        cost = (x - 0.5).square().sum(1)
        return torch.where((x.abs() > 0.75).any(1), torch.inf, cost)

    settings = Settings(
        type="mppi", gradient_steps=1, gradient_step_size=4, gradient_backtracks=4
    )
    problem = Problem(torch.zeros(1), torch.ones(1), score)
    with torch.no_grad():
        mean, info = refine_mean(problem, problem.mean, settings)
    torch.testing.assert_close(mean, torch.tensor([0.5]))
    assert info["gradient_accepted_steps"] == 1
    assert info["gradient_evaluated"] == 5  # one gradient and four trials
    assert info["gradient_final_cost"] == 0

    projected = Problem(
        torch.zeros(1), torch.ones(1), lambda x: (x - 1).square().sum(1),
        project=lambda x: x.clamp(-0.2, 0.2),
    )
    with torch.no_grad():
        mean, info = refine_mean(projected, projected.mean, settings)
    torch.testing.assert_close(mean, torch.tensor([0.2]))
    assert info["gradient_final_cost"] < info["gradient_initial_cost"]


@pytest.mark.parametrize(
    "score",
    [
        lambda x: x.square().sum(1),  # zero gradient
        lambda x: x.sqrt().sum(1),  # finite cost, nonfinite derivative
        lambda x: x.sum(1) * 0 + torch.inf,  # hard-invalid mean
    ],
)
def test_gradient_refinement_keeps_mean_on_zero_or_nonfinite_gradient(score):
    problem = Problem(torch.zeros(2), torch.ones(2), score)
    with torch.no_grad():
        mean, info = refine_mean(
            problem, problem.mean, Settings(type="mppi", gradient_steps=3)
        )
    torch.testing.assert_close(mean, problem.mean)
    assert info["gradient_accepted_steps"] == 0
    assert info["gradient_evaluated"] == 1


def test_gradient_refinement_requires_differentiable_objective():
    with pytest.raises(ValueError, match="differentiable"):
        solve(
            Problem(torch.zeros(2), torch.ones(2), lambda x: torch.zeros(len(x))),
            Settings(type="mppi", population=20, gradient_steps=1),
        )


@pytest.mark.parametrize(
    "options",
    [
        dict(gradient_steps=-1),
        dict(gradient_steps=True),
        dict(gradient_steps=1.5),
        dict(gradient_step_size=0),
        dict(gradient_step_size=float("nan")),
        dict(gradient_backtracks=0),
        dict(gradient_backtracks=1.5),
        dict(type="cem", gradient_steps=1),
    ],
)
def test_invalid_gradient_settings_fail_before_solving(options):
    with pytest.raises(ValueError):
        Settings(**{**dict(type="mppi"), **options})
