import math
import numpy as np
import pytest
import torch
from retarget.solver.base import Settings, Problem, Evaluation, NoValidCandidate, solve
from retarget.solver.cem import elite_mean
from retarget.solver.mppi import weights


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
