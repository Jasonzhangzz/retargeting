"""Lower robot dynamics refresh real contacts and preserve logical sites."""

from types import SimpleNamespace

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import torch
import warp as wp

from retarget.env.humanoid.real_contact import RealContactRollout


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    wp.init()
    if request.param != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA required")
    return request.param


def _environment(device, *, separated=False, floor=True, approaching=False):
    object_x = .24 if approaching else (.8 if separated else .18)
    floor_z = 0 if floor else -2
    cpu = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option timestep=".02" gravity="0 0 0" cone="pyramidal" jacobian="dense"/>
      <worldbody>
        <geom name="floor" type="plane" pos="0 0 {floor_z}" size="2 2 .1"/>
        <body name="robot" pos="0 0 .3"><freejoint/>
          <inertial pos="0 0 0" mass="10" diaginertia="1 1 1"/>
          <body name="left_hand"><joint name="push" type="slide" axis="1 0 0"/>
            <geom name="left_hand_geom" type="sphere" size=".1" mass="1" condim="3"/>
            <site name="left_palm" pos=".1 0 0"/>
          </body>
          <body name="right_hand" pos="0 1 0"><joint type="slide" axis="1 0 0"/>
            <geom type="sphere" size=".1" mass="1" condim="3"/><site name="right_palm" pos=".1 0 0"/>
          </body>
          <body name="left_foot" pos="0 .4 -.22"><joint type="slide" axis="0 0 1"/>
            <geom type="box" size=".1 .1 .1" mass="1" condim="3"/><site name="left_sole" pos="0 0 -.1"/>
          </body>
          <body name="right_foot" pos="0 -.4 -.22"><joint type="slide" axis="0 0 1"/>
            <geom type="box" size=".1 .1 .1" mass="1" condim="3"/><site name="right_sole" pos="0 0 -.1"/>
          </body>
        </body>
        <body name="object" pos="{object_x} 0 .3"><freejoint/>
          <geom name="object_geom" type="sphere" size=".1" mass="1" condim="3"/>
        </body>
      </worldbody><actuator><motor joint="push" gear="1"/></actuator>
    </mujoco>''')
    data = mujoco.MjData(cpu)
    if approaching:
        data.qvel[cpu.jnt_dofadr[cpu.joint("push").id]] = 2.0
    mujoco.mj_forward(cpu, data)
    with wp.ScopedDevice(device):
        model_wp = mjw.put_model(cpu)
        data_wp = mjw.put_data(cpu, data, nworld=2, nconmax=64, njmax=128)
    env = SimpleNamespace(model_cpu=cpu, model_wp=model_wp, data_wp=data_wp, device=device)
    config = SimpleNamespace(
        device=device, embodiment_type="humanoid_object",
        contact_site_ids=[cpu.site(name).id for name in ("left_palm", "right_palm", "left_sole", "right_sole")],
    )
    return config, env


def _state(env):
    return {name: getattr(env.data_wp, name).numpy().copy() for name in ("qpos", "qvel", "ctrl", "time", "qacc_warmstart")}


def test_real_reactions_use_native_rows_and_group_foot_manifolds(device):
    config, env = _environment(device)
    before = _state(env)
    rollout = RealContactRollout(config, env)
    controls = torch.zeros((1, 1, env.model_cpu.nu), device=device)
    result = rollout.rollout(controls)
    assert result.site_xpos.shape == (1, 1, 4, 3)
    assert result.site_jacobian.shape == (1, 1, 4, 3, env.model_cpu.nv)
    assert result.contact_force.shape == (1, 1, 4, 3)
    assert result.active_mask[0, 0].tolist() == [True, False, True, True]
    assert result.actual_contact_count[0, 0] > result.active_mask[0, 0].sum()
    assert result.constraint_count[0, 0] == result.actual_contact_count[0, 0] * 4
    assert result.contact_force[0, 0, 0, 0] > 0
    assert (result.contact_force[0, 0, 2:, 2] > 0).all()
    torch.testing.assert_close(result.contact_force[0, 0, 1], torch.zeros(3, device=device), atol=0, rtol=0)
    # Sites remain their configured palm/sole centres, independent of manifold
    # positions or the closest physical surface point of either body.
    torch.testing.assert_close(result.site_xpos[0, 0], wp.to_torch(env.data_wp.site_xpos)[0, config.contact_site_ids])
    for point, site in enumerate(config.contact_site_ids):
        jp, jr = np.zeros((3, env.model_cpu.nv)), np.zeros((3, env.model_cpu.nv))
        cpu_data = mujoco.MjData(env.model_cpu)
        cpu_data.qpos[:] = before["qpos"][0]
        mujoco.mj_forward(env.model_cpu, cpu_data)
        mujoco.mj_jacSite(env.model_cpu, cpu_data, jp, jr, site)
        np.testing.assert_allclose(result.site_jacobian[0, 0, point].cpu().numpy(), jp, rtol=1e-5, atol=1e-6)
    vadr = rollout.object_dof_adr
    object_force = (result.qvel[0, 1, vadr:vadr+3] - result.qvel[0, 0, vadr:vadr+3]) / result.timestep[0]
    torch.testing.assert_close(object_force, result.contact_force[0, 0, :2].sum(0), atol=1e-6, rtol=2e-5)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(env.data_wp, name).numpy(), value)


def test_absent_actual_hand_object_contact_has_exactly_zero_object_force(device):
    config, env = _environment(device, separated=True)
    backend = RealContactRollout(config, env)
    result = backend.rollout(torch.full((1, 2, env.model_cpu.nu), .4, device=device))
    assert not result.active_mask[..., :2].any()
    torch.testing.assert_close(result.contact_force[..., :2, :], torch.zeros_like(result.contact_force[..., :2, :]), atol=0, rtol=0)
    assert result.active_mask[..., 2:].any()


def test_new_actual_contact_is_acquired_as_the_predicted_hand_moves(device):
    config, env = _environment(device, approaching=True, floor=False)
    backend = RealContactRollout(config, env)
    result = backend.rollout(torch.zeros((1, 4, env.model_cpu.nu), device=device))
    assert not result.active_mask[0, 0, 0]
    assert result.active_mask[0, 1:, 0].any()
    assert result.contact_force[0, 1:, 0].abs().max() > 0
    assert torch.isfinite(result.qpos).all()


def test_contact_force_action_gradient_matches_finite_difference(device):
    config, env = _environment(device, floor=False)
    backend = RealContactRollout(config, env)
    action = torch.full((1, 1, env.model_cpu.nu), .4, device=device, requires_grad=True)
    result = backend.rollout(action)
    force_x = result.contact_force[0, 0, 0, 0]
    gradient = torch.autograd.grad(force_x, action)[0]
    assert torch.isfinite(gradient).all()
    assert gradient[0, 0, 0] > 0
    epsilon = .02
    plus = backend.rollout(action.detach() + epsilon).contact_force[0, 0, 0, 0]
    minus = backend.rollout(action.detach() - epsilon).contact_force[0, 0, 0, 0]
    numerical = (plus - minus) / (2 * epsilon)
    torch.testing.assert_close(gradient[0, 0, 0], numerical, atol=2e-6, rtol=.025)


def test_early_action_affects_later_actual_logical_site_and_object(device):
    config, env = _environment(device, floor=False)
    backend = RealContactRollout(config, env)
    action = torch.full((1, 3, env.model_cpu.nu), .4, device=device, requires_grad=True)
    result = backend.rollout(action)
    palm = result.site_xpos[0, 1, 0, 0]
    gradient = torch.autograd.grad(palm, action, retain_graph=True)[0]
    assert torch.isfinite(gradient).all()
    assert gradient[0, 0, 0] > 0
    assert gradient[0, 1:, 0].abs().sum() == 0
    object_x = result.qpos[0, -1, backend.object_qpos_adr]
    influence = torch.autograd.grad(object_x, action)[0]
    assert torch.isfinite(influence).all()
    assert influence[0, 0, 0].abs() > 1e-10
