"""A static load command seeds inputs without replacing CA state targets."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from retarget.env.grasp.forceaware_initialization import seeded_initial_state


@dataclass(frozen=True)
class Initial:
    qpos: object
    qvel: object
    warm: object
    previous_action: object


def fixture():
    model = SimpleNamespace(actuator_ctrlrange=np.array([[-2., 2.], [-2., 2.]]),
        actuator_ctrllimited=np.ones(2, bool), actuator_gainprm=np.array([[10.], [20.]]),
        actuator_biasprm=np.array([[0., -10., -1.], [0., -20., -2.]]),
        actuator_trnid=np.array([[0, -1], [1, -1]]), jnt_dofadr=np.arange(2),
        actuator_forcerange=np.array([[-1., 1.], [-2., 2.]]), actuator_forcelimited=np.ones(2, bool))
    index = SimpleNamespace(hand_qpos=np.arange(2), hand_ctrl=np.arange(2))
    initial = Initial(np.array([.2, .3, 9.]), np.zeros(3), np.zeros(8), np.array([.2, .3]))
    return initial, model, index


def test_preload_seeds_all_knots_and_previous_input_without_moving_state():
    initial, model, index = fixture()
    command = np.array([.25, .34])
    seeded = seeded_initial_state(initial, model, index, command, [.2, .3], 3)
    assert seeded.qpos is initial.qpos and seeded.qvel is initial.qvel
    np.testing.assert_array_equal(initial.warm, 0.)
    np.testing.assert_array_equal(seeded.warm[:2], 0.)
    np.testing.assert_allclose(seeded.warm[2:].reshape(3, 2), [[.05, .04]]*3)
    np.testing.assert_allclose(seeded.previous_action, command)


@pytest.mark.parametrize("command,reference,match", [
    ([2.1, .3], [.2, .3], "control limits"),
    ([.4, .3], [.2, .3], "force limits"),
    ([.25, .34], [.1, .3], "different ContactAware"),
    ([np.nan, .3], [.2, .3], "finite controls")])
def test_preload_cannot_hide_invalid_initial_inputs(command, reference, match):
    initial, model, index = fixture()
    with pytest.raises(ValueError, match=match):
        seeded_initial_state(initial, model, index, command, reference, 3)


def test_known_ca_state_selects_original_solver_option_without_replacing_inputs(monkeypatch):
    from retarget.env.grasp.forceaware_reference import register_wuji2_hand
    register_wuji2_hand()
    from forceaware.servo_prior import ServoPriorOptimizer
    from retarget.env.grasp.forceaware_initialization import activate_fixed_initial_state
    calls = []
    def original(self, *args, **kwargs):
        calls.append((args, kwargs))
        return "original solver result"
    monkeypatch.setattr(ServoPriorOptimizer, "solve", original)
    metadata = activate_fixed_initial_state()
    optimizer = object.__new__(ServoPriorOptimizer)
    state, warm = object(), object()
    assert optimizer.solve(state, allow_init=True, warm=warm) == "original solver result"
    assert calls[0][0][0] is state and calls[0][1]["warm"] is warm
    assert calls[0][1]["allow_init"] is False
    assert metadata["original_solve_option"] == {"allow_init": False}
    assert metadata["control_optimization_enabled"] is True
