"""Optional CUDA capture matches eager costs, gradients, and cache lifetime."""

import numpy as np
import pytest
import torch

from retarget.config.run import ROOT


@pytest.mark.parametrize(
    "options",
    [
        dict(contact_graph="yes", dynamics="frozen_contact", record_predictions=False),
        dict(contact_graph=True, dynamics="mjwarp", record_predictions=False),
        dict(contact_graph=True, dynamics="frozen_contact", record_predictions=True),
    ],
)
def test_graph_settings_fail_before_simulator_setup(monkeypatch, options):
    from retarget.env.humanoid import runtime

    def unexpected_setup(*args, **kwargs):
        raise AssertionError("Invalid graph options reached simulator setup")

    monkeypatch.setattr(runtime, "make_config", unexpected_setup)
    with pytest.raises(ValueError, match="contact_graph"):
        runtime.HumanoidEnvironment(
            "move_largebox", ROOT / "data/humanoid/move_largebox", **options,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_contact_graph_matches_eager_and_preserves_outstanding_results():
    from retarget.env.humanoid.runtime import HumanoidEnvironment

    task = HumanoidEnvironment(
        "move_largebox", ROOT / "data/humanoid/move_largebox", batch=2,
        dynamics="frozen_contact", contact_graph=True, record_predictions=False,
    )
    task.config.horizon_steps = 4
    task.nominal = task.nominal[:4].clone()
    before = task.env.data_wp.qpos.numpy().copy()
    with torch.cuda.stream(task.stream):
        actions = (task.nominal[None].expand(2, -1, -1) + 0.1).clone()
        task.prepare_contact_rollout()

        def eager(values):
            task.contact_graph = False
            result = task.score_controls(values)
            task.contact_graph = True
            return result

        # Inference outputs must own their values after later graph replays.
        with torch.no_grad():
            expected = eager(actions)
            first = task.score_controls(actions)
            saved = first.clone()
            second = task.score_controls(actions + 0.2)
            torch.testing.assert_close(first, expected, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(first, saved, atol=0, rtol=0)
            torch.testing.assert_close(second, eager(actions + 0.2), atol=1e-6, rtol=1e-6)
        assert task.contact_graph_builds == 1

        variable = actions.detach().requires_grad_(True)
        expected = eager(variable)
        expected_gradient = torch.autograd.grad(expected.sum(), variable)[0]
        captured = task.score_controls(variable)
        actual_gradient = torch.autograd.grad(captured.sum(), variable)[0]
        torch.testing.assert_close(captured, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-6, rtol=2e-5)
        assert task.contact_graph_builds == 2

        # A second unconsumed differentiable forward must not overwrite the
        # first graph's saved activations. It falls back to an eager operation.
        a = actions.detach().requires_grad_(True)
        b = (actions + 0.2).detach().requires_grad_(True)
        a_cost, b_cost = task.score_controls(a), task.score_controls(b)
        ga, gb = torch.autograd.grad((a_cost + b_cost).sum(), (a, b))
        expected_a = torch.autograd.grad(eager(a).sum(), a)[0]
        expected_b = torch.autograd.grad(eager(b).sum(), b)[0]
        torch.testing.assert_close(ga, expected_a, atol=1e-6, rtol=2e-5)
        torch.testing.assert_close(gb, expected_b, atol=1e-6, rtol=2e-5)
        assert task.contact_graph_builds == 2
        assert all(not entry["pending"] for entry in task._contact_graphs.values())
        saved_gradient = ga.clone()
        torch.autograd.grad(task.score_controls(b).sum(), b)
        torch.testing.assert_close(ga, saved_gradient, atol=0, rtol=0)
        np.testing.assert_array_equal(task.env.data_wp.qpos.numpy(), before)

        task.shift(task.nominal, 1)
        assert task._contact_cache is None and not task._contact_graphs
        with torch.no_grad():
            task.score_controls(task.nominal[None])
        assert task.contact_graph_builds == 3
        assert task.contact_graph_seconds > 0
        task.execute(task.nominal[:1])
        assert task._contact_cache is None and not task._contact_graphs
    torch.cuda.synchronize(task.device)
    # The first real action is the only operation allowed to change state.
    assert not np.array_equal(task.env.data_wp.qpos.numpy(), before)
