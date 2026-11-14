import mujoco
import numpy as np
from retarget.env.contact.native import snapshot
from retarget.env.contact.native_evaluation import evaluate_candidate
from retarget.env.contact.objective import force_penalty_parameters


def test_normal_humanoid_weight_support_is_not_penalized_as_an_impact():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="pelvis" pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="50"/></body>
      </worldbody></mujoco>''')
    scale, weight = force_penalty_parameters(model)
    assert np.isclose(scale, 50 * 9.81)
    for points in (1, 2, 4):
        force = 50 * 9.81 / points
        assert weight * max(force / scale - 1, 0)**2 == 0.
    assert weight * max(2 * scale / scale - 1, 0)**2 > 0.


def test_native_extra_costs_observe_the_same_state_as_integration_forces(monkeypatch):
    from retarget.env.contact import native_evaluation
    model = mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".02"/>
      <worldbody><body pos="0 0 1"><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>''')
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    positions = []
    def observe(m, d, *args, **kwargs):
        # This geom's origin is its free root: stale post-step qpos would fail.
        np.testing.assert_allclose(d.geom_xpos[0], d.qpos[:3], atol=1e-12)
        positions.append(d.qpos.copy())
        return 0., dict(smooth=0.), []
    monkeypatch.setattr(native_evaluation, 'step_cost', observe)
    result = evaluate_candidate(model, snapshot(data), np.zeros((3, 0)), {}, [], True)
    np.testing.assert_allclose(positions[0], data.qpos)
    np.testing.assert_allclose(positions[1:], result['qpos'][1:-1])
    assert result['qpos'][-1, 2] < positions[-1][2]
