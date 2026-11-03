"""Noise identity and paired tightening contracts for the local inspector."""
import numpy as np
import pytest

from retarget.env.grasp.denoising_inspector import corruption_path, compose_stages


def test_noise_rounds_are_cumulative_independent_and_reproducible():
    clean = np.linspace(-.4, .4, 20)
    observed, noise, steps = corruption_path(clean, .05, 5, 11)
    np.testing.assert_array_equal(observed[0], clean)
    np.testing.assert_array_equal(noise[0], 0)
    np.testing.assert_allclose(observed-clean, noise, atol=1e-16)
    np.testing.assert_allclose(np.diff(noise, axis=0), steps, atol=1e-16)
    np.testing.assert_array_equal(corruption_path(clean, .05, 5, 11)[0], observed)
    assert not np.allclose(steps[0], steps[1])
    assert not np.allclose(corruption_path(clean, .05, 5, 23)[1], noise)
    for sigma in (0, -.1, np.nan):
        with pytest.raises(ValueError):
            corruption_path(clean, sigma, 5, 11)


def test_full_gaussian_is_added_to_both_predictions_without_resampling():
    rng = np.random.default_rng(8)
    truth, noise, delta, perpendicular = rng.normal(size=(4, 20))
    observed = truth+noise
    estimates = [truth+.01*rng.normal(size=20) for _ in range(2)]
    outputs = [compose_stages(estimate, delta, noise, observed, perpendicular)
               for estimate in estimates]
    for estimate, values in zip(estimates, outputs):
        np.testing.assert_allclose(values['noise']-values['tight'], noise, atol=1e-15)
        np.testing.assert_allclose(values['tight']-values['clean'], delta, atol=1e-15)
        np.testing.assert_array_equal(values['clean'], estimate)
        np.testing.assert_allclose(values['removed'], observed+delta, atol=1e-15)
        assert not np.allclose(values['noise'], values['removed'])
        assert not np.allclose(values['noise'], values['perpendicular'])
    # Different estimates remain different after the same noise and size shift.
    np.testing.assert_allclose(outputs[0]['noise']-outputs[1]['noise'],
                               estimates[0]-estimates[1], atol=1e-15)
