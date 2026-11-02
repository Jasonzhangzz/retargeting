"""Numerical audit-driver tests; synthetic functions do not certify native FA."""

from dataclasses import dataclass

import numpy as np
import pytest

from retarget.env.grasp.forceaware_objective_validation import (
    ObjectiveSample, diagnostic_config, directional_checks, objective_difference,
    objective_scores, unscaled_raw_gradient,
)


def test_objective_score_accumulates_original_terms_in_float64():
    terms = np.asarray([[1e8, 3., 3.], [1e8, 4., 4.]], np.float32)
    regularization = np.asarray([3., 4.], np.float32)
    np.testing.assert_array_equal(objective_scores(terms, regularization), [100000009., 100000012.])
    assert float(terms[0].sum(dtype=np.float32)+regularization[0]) != 100000009.
    with pytest.raises(ValueError, match="Objective terms"):
        objective_scores(terms, regularization[None])


def test_raw_gradient_is_unscaled_once_for_the_unscaled_objective():
    stored = np.asarray([[2.e-6, -3.e-6]], np.float32)
    np.testing.assert_allclose(unscaled_raw_gradient(stored, gradient_scale=1e-6), [[2., -3.]], rtol=1e-7)
    with pytest.raises(ValueError, match="positive finite scale"):
        unscaled_raw_gradient(stored, gradient_scale=0.)


def _quadratic_sample(raw, *, branch="base", contact="base", actuator="base"):
    raw = np.asarray(raw, np.float32)
    return ObjectiveSample(raw=raw, trajectory_terms=np.asarray([1e20, raw[2]**2+2*raw[3]**2], np.float32),
        regularization=0., controls=raw[None, 2:].copy(), contact_signature=(contact,),
        solver_signature=(branch,), actuator_signature=(actuator,), contact_counts=(1,))


def _quadratic_check(evaluate, *, gradient=None):
    raw = np.asarray([0., 0., .2, -.3], np.float32)
    base = _quadratic_sample(raw)
    gradient = np.r_[0., 0., 2*raw[2], 4*raw[3]] if gradient is None else gradient
    return directional_checks(evaluate, base, gradient, [[0., 0., .6, .8]],
        expected_control_direction=lambda direction: direction[None, 2:],
        epsilons=(.001, .0003, .0001), rtol=.01, atol=.001)[0]


def test_directional_fd_subtracts_terms_before_large_scalar_cancellation():
    low = _quadratic_sample([0., 0., .1994, -.3008])
    high = _quadratic_sample([0., 0., .2006, -.2992])
    assert high.score == low.score  # Even float64 scalar totals lose this signal.
    assert objective_difference(high, low) != 0.
    result = _quadratic_check(_quadratic_sample)
    assert result["passed"] is True
    assert result["outcome"] == "passed"
    assert result["finite_difference"] == pytest.approx(-.72, abs=.001)


def test_large_constant_regularization_cannot_erase_trajectory_difference():
    low = _quadratic_sample([0., 0., .1994, -.3008])
    high = _quadratic_sample([0., 0., .2006, -.2992])
    low = ObjectiveSample(**{**low.__dict__, "regularization": 1e20})
    high = ObjectiveSample(**{**high.__dict__, "regularization": 1e20})
    expected = float(np.sum(high.trajectory_terms.astype(np.float64)-low.trajectory_terms.astype(np.float64)))
    assert expected != 0.
    assert objective_difference(high, low) == expected


@pytest.mark.parametrize("changed", ["branch", "contact", "actuator"])
def test_matching_derivative_is_uncertified_when_native_branch_changes(changed):
    def evaluate(raw):
        return _quadratic_sample(raw, **{changed: "perturbed"})
    result = _quadratic_check(evaluate)
    assert result["passed"] is False
    assert result["outcome"] == "inconclusive"
    assert result["selected_epsilon"] is None


def test_same_branch_stable_fd_detects_wrong_unscaled_adjoint():
    result = _quadratic_check(_quadratic_sample, gradient=[0., 0., .4, -12.])
    assert result["passed"] is False
    assert result["outcome"] == "mismatch_on_same_branch"
    assert result["selected_epsilon"] is not None


def test_unresolved_float32_controls_cannot_certify_a_derivative():
    def evaluate(raw):
        sample = _quadratic_sample(raw)
        # A quantized actuator input can lose the requested raw perturbation.
        return ObjectiveSample(**{**sample.__dict__, "controls": np.zeros((1, 2), np.float32)})
    result = _quadratic_check(evaluate)
    assert result["passed"] is False
    assert result["outcome"] == "inconclusive"
    assert all(row["control_direction_error_norm"] > .02 for row in result["trials"])


def test_fd_ladder_must_supply_distinct_decreasing_positive_scales():
    base = _quadratic_sample([0., 0., .2, -.3])
    with pytest.raises(ValueError, match="decreasing positive ladder"):
        directional_checks(_quadratic_sample, base, np.zeros(4), [[0, 0, 1, 0]],
            expected_control_direction=lambda direction: direction[None, 2:], epsilons=(.001, .003))


@dataclass(frozen=True)
class Budget:
    multistart_count: int = 64
    n_iter: int = 40
    horizon: int = 5
    lr_max: float = .0015


@dataclass(frozen=True)
class Profile:
    optimizer: object
    loss: object
    smoothing: object
    simulator: object


def test_diagnostic_budget_preserves_all_losses_and_complete_window():
    cfg = Profile(Budget(), object(), object(), object())
    diagnostic = diagnostic_config(cfg)
    assert diagnostic.optimizer.multistart_count == diagnostic.optimizer.n_iter == 1
    assert diagnostic.optimizer.horizon == cfg.optimizer.horizon == 5
    assert diagnostic.optimizer.lr_max == cfg.optimizer.lr_max
    assert diagnostic.loss is cfg.loss and diagnostic.smoothing is cfg.smoothing
    assert diagnostic.simulator is cfg.simulator
    assert cfg.optimizer.multistart_count == 64 and cfg.optimizer.n_iter == 40
