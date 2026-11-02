"""CPU parity with pinned SPIDER sampling sources and explicit safety fixes."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from retarget.solver.base import Settings
from retarget.solver.spider_mppi import noise_scale, optimize, reward_weights, shift_controls


UPSTREAM = Path(__file__).resolve().parents[1]/"third_party/SPIDER/dex/spider"


def source_function(relative, name, **extra):
    """Execute one original function, without importing its simulator/package."""
    path = UPSTREAM/relative
    node = next(item for item in ast.parse(path.read_text()).body
                if isinstance(item, ast.FunctionDef) and item.name == name)
    namespace = dict(torch=torch, F=F, Config=SimpleNamespace, **extra)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@pytest.fixture(scope="module")
def original():
    interpolation = source_function("interp.py", "interp")
    return SimpleNamespace(
        interp=interpolation,
        weights=source_function("optimizers/sampling.py", "_compute_weights_impl"),
        sample=source_function("optimizers/sampling.py", "_sample_ctrls_impl", interp=interpolation),
        scale=source_function("config.py", "get_noise_scale"),
    )


def noise_config(population=100, knots=4, channels=26):
    return SimpleNamespace(first_ctrl_noise_scale=.5, last_ctrl_noise_scale=1.,
        horizon=float(knots), knot_dt=1., num_samples=population, nu=channels,
        embodiment_type="right", object_action_dims=0, pos_noise_scale=.03,
        rot_noise_scale=.01, joint_noise_scale=.15, contact_guidance=False,
        object_actuator_ids=[], exploit_ratio=.01, exploit_noise_scale=.01,
        device="cpu")


def channel_noise(config):
    return torch.tensor([config.pos_noise_scale]*3+[config.rot_noise_scale]*3
                        +[config.joint_noise_scale]*(config.nu-6))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("temperature", [.1, 1.])
def test_reward_weights_match_original_on_finite_population(original, dtype, temperature):
    rewards = torch.randn(100, dtype=dtype, generator=torch.Generator().manual_seed(44717007))
    expected, invalid = original.weights(rewards, 100, temperature)
    ids, weights = reward_weights(rewards, temperature)
    actual = torch.zeros_like(rewards).scatter(0, ids, weights)
    assert not invalid.any()
    assert len(ids) == 10
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)


def test_dense_interpolation_preserves_original_global_grid(original):
    from retarget.env.humanoid.interp import interp
    knots = torch.tensor([[[0., 2.], [2., -1.], [1., 5.]],
                          [[4., 8.], [1., 7.], [-3., 6.]]])
    actual = interp(knots, 4)
    expected = original.interp(knots, 4)
    assert actual.shape == (2, 12, 2)
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
    torch.testing.assert_close(actual[:, 0], knots[:, 0])
    torch.testing.assert_close(actual[:, -1], knots[:, -1])
    # Original aligns the full sequence endpoints; a per-segment hold has a
    # different time grid even though both schemes pass through the knots.
    assert actual[0, 4, 0] == pytest.approx(16/11)


def test_noise_scale_matches_original_at_population_100(original):
    config = noise_config()
    actual = noise_scale(config.num_samples, 4, channel_noise(config))
    torch.testing.assert_close(actual, original.scale(config), rtol=0., atol=0.)
    assert torch.equal(actual[0], torch.zeros_like(actual[0]))
    torch.testing.assert_close(actual[-1], actual[1]*.01, rtol=0., atol=0.)


def test_small_population_fixes_original_negative_zero_exploit_slice(original):
    config = noise_config(population=64)
    actual = noise_scale(64, 4, channel_noise(config))
    source = original.scale(config)
    # int(64*.01)==0; the original [-0:] accidentally shrinks every sample.
    torch.testing.assert_close(source, actual*.01, rtol=0., atol=0.)
    torch.testing.assert_close(actual[1, 0], channel_noise(config)*.5)
    torch.testing.assert_close(actual[-1], actual[1], rtol=0., atol=0.)


def test_one_iteration_dense_weighted_mean_matches_original_source(original):
    config = noise_config(knots=3)
    config.noise_scale, config.knot_steps = original.scale(config), 4
    mean = torch.linspace(-.2, .4, 12*26).reshape(12, 26)
    target = mean.flip(0)+.02
    settings = Settings(type="mppi", population=100, iterations=1,
                        temperature=.3, improvement_check_steps=2)
    evaluate = lambda candidates: ((candidates-target)**2).sum(dim=(1, 2))
    seed = 1226
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        candidates = original.sample(config, mean, {"global_noise_scale": 1.})
    costs = evaluate(candidates)
    weights, _ = original.weights(-costs, 100, settings.temp)
    expected = (weights[:, None, None]*candidates).sum(0)
    seen = []

    def record(values):
        seen.append(values.clone())
        return evaluate(values)

    actual, result = optimize(mean, record, settings, knots=3, knot_steps=4,
        channel_noise=channel_noise(config), seed=seed)
    torch.testing.assert_close(seen[0], candidates, rtol=0., atol=0.)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(seen[0][0], mean, rtol=0., atol=0.)
    assert not torch.allclose(actual, candidates[costs.argmin()], atol=1e-5, rtol=1e-5)
    assert result["iterations"] == 1 and result["sampled"] == 100
    assert result["evaluated"] == 101
    assert result["cost"] == pytest.approx(evaluate(actual[None]).item())


def test_annealed_dense_updates_match_original_noise_schedule(original):
    config = noise_config(knots=3)
    config.noise_scale, config.knot_steps = original.scale(config), 2
    settings = Settings(type="mppi", population=100, iterations=3,
        temperature=.6, final_noise=.1, improvement_check_steps=4)
    initial = torch.linspace(-.1, .15, 6*26).reshape(6, 26)
    evaluate = lambda candidates: (candidates.square().sum((1, 2)))
    expected, seed = initial.clone(), 17007
    beta = settings.noise_ratio**(1/settings.iterations)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for iteration in range(settings.iterations):
            candidates = original.sample(config, expected, {"global_noise_scale": beta**iteration})
            weights, _ = original.weights(-evaluate(candidates), 100, settings.temp)
            expected = (weights[:, None, None]*candidates).sum(0)
    actual, result = optimize(initial, evaluate, settings, knots=3, knot_steps=2,
        channel_noise=channel_noise(config), seed=seed)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    assert result["iterations"] == 3 and result["evaluated"] == 301


def test_invalid_candidates_are_excluded_even_when_top_decile_is_not_full():
    rewards = torch.full((100,), torch.nan)
    rewards[1], rewards[3], rewards[7], rewards[42] = torch.inf, -torch.inf, 2., 1.
    ids, weights = reward_weights(rewards, .3)
    assert ids.tolist() == [7, 42]
    assert torch.isfinite(weights).all()
    assert weights.sum() == pytest.approx(1.)
    # One surviving rollout also stays finite; original std() on a singleton
    # would produce NaN when top_k itself is one.
    rewards[42] = torch.nan
    ids, weights = reward_weights(rewards, .3)
    assert ids.tolist() == [7] and weights.tolist() == [1.]


@pytest.mark.parametrize("value", [torch.nan, torch.inf, -torch.inf])
def test_all_invalid_population_fails_closed(value):
    with pytest.raises(FloatingPointError, match="All SPIDER MPPI"):
        reward_weights(torch.full((100,), value), 1.)


def test_nonfinite_weighted_mean_recheck_falls_back_to_actual_finite_sample():
    settings = Settings(type="mppi", population=100, iterations=1,
                        improvement_check_steps=2)
    seen = []

    def evaluate(candidates):
        seen.append(candidates.clone())
        if len(candidates) == settings.population:
            cost = torch.full((len(candidates),), torch.inf)
            cost[17] = 3.
            return cost
        return torch.full((len(candidates),), torch.nan)

    actual, result = optimize(torch.zeros(6, 2), evaluate, settings,
        knots=3, knot_steps=2, channel_noise=torch.ones(2), seed=447)
    torch.testing.assert_close(actual, seen[0][17], rtol=0., atol=0.)
    assert result["cost"] == 3. and result["history"][0]["invalid"] == 99
    assert torch.isfinite(actual).all()


def test_warm_shift_appends_new_reference_tail_without_repeating_last_action():
    controls = torch.arange(18.).reshape(6, 3)
    reference_tail = torch.tensor([[-1., -2., -3.], [9., 8., 7.]])
    actual = shift_controls(controls, 2, reference_tail)
    torch.testing.assert_close(actual, torch.cat((controls[2:], reference_tail)), rtol=0., atol=0.)
    torch.testing.assert_close(shift_controls(controls, 6, -controls), -controls)
    with pytest.raises(ValueError, match="append one reference action"):
        shift_controls(controls, 2, reference_tail[:1])
    with pytest.raises(ValueError, match="append one reference action"):
        shift_controls(controls, 0, reference_tail[:0])
