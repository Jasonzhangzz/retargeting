"""Normal contact accuracy checks and adaptive pressure-loop bookkeeping."""

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from retarget.dynamics.explicit import (
    contact_normal_residual, projected_response, solve_contact_response,
    store_contact_residual,
)
from retarget.dynamics.explicit_parameters import Parameters


def _array(value, device, dtype=float):
    return wp.array(np.asarray(value), dtype=dtype, device=device)


def _residual(jacobian, free, phi, pressure, inverse, device):
    wp.init()
    p = Parameters()
    p.inverse = _array(inverse, device)
    jac, displacement, gaps, force = (_array(value, device) for value in (jacobian, free, phi, pressure))
    response = wp.zeros(np.shape(free), device=device)
    result = wp.zeros(len(free), device=device)
    wp.launch(projected_response, (len(free), 32), inputs=[p, jac, force, response], device=device)
    wp.launch(contact_normal_residual, len(free), inputs=[jac, displacement, gaps, force, response, result], device=device)
    return result.numpy()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_normal_contact_residual_matches_independent_numpy(device):
    wp.init()
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA required")
    rng = np.random.default_rng(12)
    jac = rng.normal(0, .2, (3, 8, 32)).astype(np.float32)
    free = rng.normal(0, .01, (3, 32)).astype(np.float32)
    phi = rng.normal(0, .005, (3, 8)).astype(np.float32)
    force = rng.uniform(0, .1, (3, 8)).astype(np.float32)
    force[1, 4:] = 0
    inverse = rng.uniform(.001, .1, 32).astype(np.float32)
    actual = _residual(jac, free, phi, force, inverse, device)
    response = inverse * np.einsum("wrj,wr->wj", jac, force)
    gaps = (phi + np.einsum("wrj,wj->wr", jac, free + response)).reshape(3, 2, 4).mean(2)
    loaded = force.reshape(3, 2, 4).sum(2) > 1e-4
    expected = np.maximum(np.maximum(-gaps, 0), np.where(loaded, np.abs(gaps), 0)).max(1) * 1000
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-4)


def test_loaded_positive_gap_is_a_complementarity_violation_and_empty_contacts_are_valid():
    jac, free, inverse = np.zeros((2, 4, 32)), np.zeros((2, 32)), np.ones(32)
    phi = np.full((2, 4), .003)
    pressure = np.array([[.1, .1, .1, .1], [0, 0, 0, 0]])
    np.testing.assert_allclose(_residual(jac, free, phi, pressure, inverse, "cpu"), [3, 0], atol=1e-6)
    np.testing.assert_array_equal(_residual(np.zeros((2, 0, 32)), free, np.zeros((2, 0)),
        np.zeros((2, 0)), inverse, "cpu"), [0, 0])


@pytest.mark.parametrize("where", ["jac", "free", "phi", "force", "inverse"])
def test_nonfinite_contact_inputs_never_look_converged(where):
    arrays = dict(jac=np.zeros((2, 4, 32)), free=np.zeros((2, 32)),
        phi=np.full((2, 4), .01), force=np.zeros((2, 4)), inverse=np.ones(32))
    arrays[where].flat[-1] = np.nan
    result = _residual(arrays["jac"], arrays["free"], arrays["phi"], arrays["force"], arrays["inverse"], "cpu")
    assert np.isinf(result[-1])


def _scalar_contact_stage(device="cpu", depth=.001):
    wp.init()
    depth = np.atleast_1d(depth)
    n = len(depth)
    jac = np.zeros((n, 4, 32), np.float32)
    jac[:, :, 0] = 1
    p = Parameters()
    p.inverse = _array(np.ones(32), device)
    p.friction = .8
    stage = SimpleNamespace(
        phi=_array(np.repeat(-depth[:, None], 4, axis=1), device),
        jac=_array(jac, device), free=wp.zeros((n, 32), device=device),
        rate=_array(np.full(n, .5), device),
        forces=[wp.zeros((n, 4), device=device), wp.zeros((n, 4), device=device)],
        responses=[wp.zeros((n, 32), device=device)],
        projections=[wp.zeros((n, 4), device=device)],
        contact_residual=wp.zeros(n, device=device), force=None, iterations_used=0,
    )
    return p, stage


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_adaptive_stopping_uses_actual_final_buffer_and_waits_for_all_worlds(device):
    wp.init()
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA required")
    p, stage = _scalar_contact_stage(device, depth=[.001, .004])
    used = solve_contact_response(p, stage, 2, device, 9, .1, 2)
    assert used == stage.iterations_used == 6
    assert stage.force is stage.forces[0]  # maximum 9 would incorrectly pick buffer 1.
    np.testing.assert_allclose(stage.force.numpy().sum(1), np.array([.001, .004]) * (1 - .5**6), rtol=2e-6)
    np.testing.assert_allclose(stage.contact_residual.numpy(), [1/64, 4/64], atol=2e-6)
    trajectory = wp.zeros((2, 3), device=device)
    wp.launch(store_contact_residual, 2, inputs=[stage.contact_residual, 1, trajectory], device=device)
    np.testing.assert_allclose(trajectory.numpy()[:, 1], stage.contact_residual.numpy())


def test_disabled_check_keeps_legacy_iteration_count_and_odd_maximum_buffer():
    p, stage = _scalar_contact_stage()
    used = solve_contact_response(p, stage, 1, "cpu", 5, None, 2)
    assert used == stage.iterations_used == 5 and stage.force is stage.forces[1]
    np.testing.assert_allclose(stage.force.numpy().sum(), .001 * (1 - .5**5), rtol=2e-6)
    np.testing.assert_allclose(stage.contact_residual.numpy(), [1/32], atol=2e-6)
    # A strict failed solve must still expose the newest pressure buffer.
    used = solve_contact_response(p, stage, 1, "cpu", 3, 1e-9, 2)
    assert used == 3 and stage.force is stage.forces[1]
    assert stage.contact_residual.numpy()[0] > 1e-9
