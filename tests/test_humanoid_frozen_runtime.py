"""Cached-contact scoring is independent of MPPI rounds and candidate chunks."""

import numpy as np
import pytest
import torch

from retarget.config.run import ROOT


@pytest.mark.parametrize(
    "dynamics,options",
    [
        ("unknown", None),
        ("mjwarp", {}),
        ("frozen_contact", []),
        ("frozen_contact", {"unknown_option": 1}),
        ("frozen_contact", {"contact_weight": -1}),
    ],
)
def test_invalid_contact_settings_fail_before_simulator_setup(
    monkeypatch, dynamics, options
):
    from retarget.env.humanoid import runtime

    def unexpected_setup(*args, **kwargs):
        raise AssertionError("Invalid options reached simulator setup")

    monkeypatch.setattr(runtime, "make_config", unexpected_setup)
    with pytest.raises(ValueError):
        runtime.HumanoidEnvironment(
            "move_largebox",
            ROOT / "data/humanoid/move_largebox",
            dynamics=dynamics,
            contact_options=options,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_runtime_reuses_nominal_contacts_and_preserves_execution_state(monkeypatch):
    import mujoco_warp as mjwarp
    from mujoco_warp._src import collision_driver

    from retarget.env.humanoid.runtime import HumanoidEnvironment
    from retarget.task.humanoid import cost_contact

    task = HumanoidEnvironment(
        "move_largebox",
        ROOT / "data/humanoid/move_largebox",
        batch=2,
        dynamics="frozen_contact",
    )
    # A short horizon is sufficient to check the full scoring/caching lifecycle.
    task.config.horizon_steps = 4
    task.nominal = task.nominal[:4].clone()
    keys = ("qpos", "qvel", "qacc_warmstart", "ctrl", "time")
    before = {name: getattr(task.env.data_wp, name).numpy().copy() for name in keys}
    margins = task.env.model_wp.pair_margin.numpy().copy()
    assert task.config.num_dr == 0
    assert task.env_params == [{"pair_margin": 0.0, "xy_offset": 0.0}]

    with torch.cuda.stream(task.stream):
        cached = task.prepare_contact_rollout()
        assert cached is task.prepare_contact_rollout()
        assert task.contact_cache_builds == 1
        offsets = torch.linspace(0.0, 0.2, 5, device=task.device)[:, None, None]
        actions = (task.nominal[None] + offsets).detach().requires_grad_(True)

        def unexpected_collision(*args, **kwargs):
            raise AssertionError(
                "Candidate scoring attempted simulator/collision refresh"
            )

        with monkeypatch.context() as patched:
            for name in (
                "step",
                "forward",
                "fwd_position",
                "fwd_velocity",
                "fwd_actuation",
                "fwd_acceleration",
                "collision",
            ):
                if hasattr(mjwarp, name):
                    patched.setattr(mjwarp, name, unexpected_collision)
            patched.setattr(collision_driver, "collision", unexpected_collision)
            patched.setattr(cost_contact, "_refresh", unexpected_collision)
            # Five candidates require three world-sized chunks. A later MPPI
            # evaluation reuses the same cache as all three earlier chunks.
            scored = task.score_controls(actions)
            again = task.score_controls(actions[:2])
            assert task.contact_cache_builds == 1
            assert task.calls == 4
            assert task.trajectories == 7
            torch.testing.assert_close(scored[:2], again)
            assert bool(torch.isfinite(scored).all())
            assert float((scored - scored[0]).detach().abs().max()) > 1e-8
            scored.sum().backward()
            assert bool(torch.isfinite(actions.grad).all())
            assert float(actions.grad.abs().max()) > 0
            # Prediction buffers are for recording only; they must not retain
            # candidate autograd graphs or advance the live execution world.
            assert not task.q.requires_grad and not task.v.requires_grad
            assert task._contact_cache is cached
            task.shift(task.nominal, 1)
            assert task._contact_cache is None

    torch.cuda.synchronize(task.device)
    for name in keys:
        np.testing.assert_array_equal(
            getattr(task.env.data_wp, name).numpy(), before[name]
        )
    np.testing.assert_array_equal(task.env.model_wp.pair_margin.numpy(), margins)
    with torch.cuda.stream(task.stream):
        assert task.prepare_contact_rollout() is not cached
        assert task.contact_cache_builds == 2
        states, _, _ = task.execute(task.nominal[:1])
        assert len(states) == 1
        assert task._contact_cache is None
    torch.cuda.synchronize(task.device)
    # Cache preparation must preserve registration of the shared Warp stream.
    # A second environment captures a new simulator graph on that same stream.
    default_task = HumanoidEnvironment(
        "move_largebox",
        ROOT / "data/humanoid/move_largebox",
        batch=1,
        record_predictions=False,
    )
    assert default_task.dynamics == "mjwarp"
