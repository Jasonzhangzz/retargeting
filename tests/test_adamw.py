"""AdamW checked against an independent Torch optimizer."""
import numpy as np
import torch

from retarget.fa.adamw import AdamW


def test_adamw_matches_torch_with_decoupled_decay():
    start = np.array([[2., -.5], [.7, -1.]])
    parameter = torch.tensor(start, dtype=torch.float64, requires_grad=True)
    reference = torch.optim.AdamW([parameter], lr=.1, weight_decay=.03)
    target = torch.tensor([[.2, .3], [.1, .2]], dtype=torch.float64)
    for _ in range(5):
        reference.zero_grad()
        loss = ((parameter-target)**2).sum()
        loss.backward()
        reference.step()
    calls, last = [], []
    def evaluate(x, gradient):
        calls.append(gradient);last[:] = [x.copy()]
        delta = x-target.detach().numpy()
        return np.sum(delta**2, axis=1), 2*delta if gradient else None
    result, _, diagnostic = AdamW(learning_rate=.1, weight_decay=.03).update(
        start, evaluate, iterations=5, horizon=1, advance=0.)
    np.testing.assert_allclose(result, parameter.detach().numpy(), rtol=1.e-12, atol=1.e-12)
    np.testing.assert_array_equal(last[0], result)
    assert calls.count(True)==5 and diagnostic['gradient_evaluations']==5


def test_adamw_persistent_moments_shift_before_torch_equivalent_update():
    solver = AdamW(learning_rate=.1, weight_decay=.01, persist=True)
    solver.memory = (np.array([1., 3.]), np.array([1., 9.]), 3)
    parameter = torch.tensor([[2., 2.]], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.AdamW([parameter], lr=.1, weight_decay=.01)
    optimizer.state[parameter] = dict(step=torch.tensor(3.),
        exp_avg=torch.tensor([[2., 3.]], dtype=torch.float64),
        exp_avg_sq=torch.tensor([[5., 9.]], dtype=torch.float64))
    (parameter**2).sum().backward()
    optimizer.step()
    def evaluate(x, gradient):
        return np.sum(x*x, axis=1), 2*x if gradient else None
    result, _, diagnostic = solver.update(np.array([[2., 2.]]), evaluate,
        iterations=1, horizon=2, advance=.5)
    np.testing.assert_allclose(result, parameter.detach().numpy(), rtol=1.e-12)
    assert diagnostic['moments_restored'] and solver.memory[2]==4
    np.testing.assert_allclose(solver.memory[0], optimizer.state[parameter]['exp_avg'].numpy()[0])


def test_alternating_blocks_keep_each_candidates_moments_without_shifting_again():
    start = np.array([[2., 3.], [-1., 2.]])
    def evaluate(x, gradient):
        return np.sum(x*x, axis=1), 2*x if gradient else None
    solver = AdamW(learning_rate=.1)
    first, _, _ = solver.update(start, evaluate, iterations=2, horizon=2, advance=.5)
    actual, _, diag = solver.update(first, evaluate, iterations=3, horizon=2, advance=.5,
                                    continue_window=True)
    expected, _, _ = AdamW(learning_rate=.1).update(start, evaluate, iterations=5,
                                                  horizon=2, advance=0.)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert diag['continued_within_window'] and diag['moments_restored']
    assert solver.memory is None  # Cross-window persistence is still opt-in.
    fresh, _, diag = solver.update(start, evaluate, iterations=2, horizon=2, advance=.5)
    np.testing.assert_allclose(fresh, first, atol=1e-12)
    assert not diag['moments_restored']
