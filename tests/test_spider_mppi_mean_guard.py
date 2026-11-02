"""Opt-in sampling centers must use the costs of their actual configurations."""

import pytest
import torch

from retarget.solver.base import Settings
from retarget.solver.spider_mppi import optimize, reward_weights


def _settings(iterations=2):
    return Settings(type="mppi", population=20, worlds=4, iterations=iterations,
                    improvement_check_steps=iterations + 1)


@pytest.mark.parametrize("mean_cost", [100., float("inf"), float("nan"), -float("inf")])
def test_guard_rejects_bad_mean_before_it_becomes_the_next_center(mean_cost):
    settings = _settings()
    populations, winners = [], []
    checks = []

    def evaluate(values):
        if len(values) == settings.population:
            populations.append(values.clone())
            costs = values.new_full((len(values),), torch.inf)
            costs[7], costs[11] = 1., 1.
            ids, _ = reward_weights(-costs, settings.temp)
            winners.append(values[ids[0]].clone())
            return costs
        checks.append(values.clone())
        # The last singleton is the existing final-candidate recheck.
        cost = mean_cost if len(checks) <= settings.iterations else 1.
        return values.new_tensor([cost])

    actual, report = optimize(torch.zeros(1, 2), evaluate, settings,
        knots=1, knot_steps=1, channel_noise=torch.ones(2), seed=81,
        guard_mean=True, return_best=True)
    assert len(populations) == 2 and len(checks) == 3
    torch.testing.assert_close(populations[1][0], winners[0], rtol=0, atol=0)
    # Equal-cost winners may replace each other as the current center.
    torch.testing.assert_close(actual, winners[-1], rtol=0, atol=0)
    assert report["mean_guard"] is True
    assert report["mean_fallbacks"] == report["mean_guard_evaluated"] == 2
    assert report["evaluated"] == 2 * settings.population + 2 + 1
    assert report["sampled"] == 2 * settings.population
    assert all(row["mean_fallback"] for row in report["history"])
    assert all(row["iteration_best_cost"] == 1. for row in report["history"])
    assert report["cost"] == 1.


def test_a_better_evaluated_mean_is_retained_as_the_best_output():
    settings = _settings()
    means = []

    def evaluate(values):
        if len(values) == settings.population:
            # The first weighted mean is better than either round's samples.
            costs = values.new_full((len(values),), torch.inf)
            costs[7], costs[11] = 1., 1.
            return costs
        means.append(values.clone())
        return values.new_tensor([0. if len(means) == 1 else .5])

    actual, report = optimize(torch.zeros(1, 2), evaluate, settings,
        knots=1, knot_steps=1, channel_noise=torch.ones(2), seed=81,
        guard_mean=True, return_best=True)
    torch.testing.assert_close(actual, means[0][0], rtol=0, atol=0)
    assert report["cost"] == 0.
    assert report["mean_fallbacks"] == 0
    assert report["mean_guard_evaluated"] == 2
    assert all(row["best_cost"] == 0. for row in report["history"])
    assert not any(row["mean_fallback"] for row in report["history"])


def test_guard_disabled_preserves_the_default_sampling_path_and_call_count():
    settings = _settings()
    results = []
    for options in ({}, {"guard_mean": False}):
        calls = []

        def evaluate(values):
            calls.append(values.clone())
            return values.square().sum((1, 2))

        actual, report = optimize(torch.ones(1, 2), evaluate, settings,
            knots=1, knot_steps=1, channel_noise=torch.ones(2), seed=81, **options)
        results.append((actual, report, calls))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for left, right in zip(results[0][2], results[1][2]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert len(results[0][2]) == len(results[1][2]) == 3
    assert results[0][1]["evaluated"] == results[1][1]["evaluated"] == 41
    assert results[0][1]["history"] == results[1][1]["history"]
    assert "mean_guard" not in results[0][1]


def test_guard_requires_a_boolean():
    with pytest.raises(ValueError, match="guard_mean must be boolean"):
        optimize(torch.zeros(1, 2), lambda values: values.square().sum((1, 2)), _settings(),
                 knots=1, knot_steps=1, channel_noise=torch.ones(2), seed=81, guard_mean=1)


def test_nonconvex_feasible_regions_do_not_seed_from_their_colliding_average():
    settings = Settings(type="mppi", population=100, worlds=4, iterations=3,
                        improvement_check_steps=4)

    def objective(values):
        x = values[:, 0, 0]
        # The two safe wells surround a costly forbidden interval. Averages
        # of individually good samples can lie inside that interval.
        return 100 * torch.relu(.75 - x.abs()).square() + (x.abs() - 1).square()

    actual, report = optimize(torch.tensor([[.9]]), objective, settings,
        knots=1, knot_steps=1, channel_noise=torch.tensor([2.]), seed=1,
        first_noise=1., last_noise=1., guard_mean=True, return_best=True)
    first = report["history"][0]
    assert first["mean_cost"] > 1. and first["iteration_best_cost"] < .01
    assert first["mean_fallback"] is True
    assert report["mean_fallbacks"] >= 1
    assert actual.abs().item() >= .75
    assert report["cost"] == pytest.approx(objective(actual[None]).item())
    assert report["cost"] < .01


def test_guard_keeps_sample_cost_when_the_evaluator_reuses_its_output_buffer():
    settings = _settings(iterations=1)
    workspace = torch.full((settings.population,), torch.inf)
    sampled = []

    def evaluate(values):
        if len(values) == settings.population:
            sampled.append(values.clone())
            workspace[7], workspace[11] = 1., 1.
            return workspace
        # A cached rollout workspace can be overwritten by the singleton
        # evaluation. That must not change the saved round-winner cost.
        workspace.fill_(100.)
        return workspace[:1]

    actual, report = optimize(torch.zeros(1, 2), evaluate, settings,
        knots=1, knot_steps=1, channel_noise=torch.ones(2), seed=81,
        guard_mean=True, return_best=True)
    assert report["mean_fallbacks"] == 1
    assert report["history"][0]["iteration_best_cost"] == 1.
    assert report["history"][0]["invalid"] == settings.population - 2
    assert report["cost"] == 1.
    expected_costs = torch.full((settings.population,), torch.inf)
    expected_costs[7], expected_costs[11] = 1., 1.
    ids, _ = reward_weights(-expected_costs, settings.temp)
    torch.testing.assert_close(actual, sampled[0][ids[0]], rtol=0, atol=0)
