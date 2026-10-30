"""Contact costs use live Warp geometry and retain action gradients."""

from contextlib import nullcontext
from types import SimpleNamespace

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import torch
import warp as wp


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    wp.init()
    if request.param != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA required")
    return request.param


def _environment(device, worlds=1, *, floor=False, jacobian="dense", actuator=None):
    """A floating robot pushes a separate free body through a motorized hand."""
    ground_z = 0 if floor else -2
    actuator = actuator or '<motor joint="push" gear="1"/>'
    cpu = mujoco.MjModel.from_xml_string(
        f"""<mujoco>
          <option timestep=".1" gravity="0 0 0" jacobian="{jacobian}"/>
          <worldbody>
            <geom name="floor" type="plane" pos="0 0 {ground_z}" size="2 2 .1"/>
            <body name="robot" pos="0 0 .09">
              <freejoint/>
              <inertial pos="0 0 0" mass="10" diaginertia="1 1 1"/>
              <body name="hand">
                <joint name="push" type="slide" axis="1 0 0"/>
                <geom name="hand_geom" type="sphere" size=".1" mass="1"/>
                <site name="palm" pos=".1 0 0" size=".001"/>
                <site name="inactive" pos="0 .5 0" size=".001"/>
              </body>
            </body>
            <body name="object" pos=".18 0 .09">
              <freejoint/>
              <geom name="object_geom" type="sphere" size=".1" mass="1"/>
            </body>
          </worldbody>
          <actuator>{actuator}</actuator>
        </mujoco>"""
    )
    data = mujoco.MjData(cpu)
    mujoco.mj_forward(cpu, data)
    with wp.ScopedDevice(device):
        model_wp = mjw.put_model(cpu)
        data_wp = mjw.put_data(cpu, data, nworld=worlds, nconmax=12, njmax=48)
    env = SimpleNamespace(
        model_cpu=cpu,
        data_cpu=data,
        model_wp=model_wp,
        data_wp=data_wp,
        device=device,
        num_worlds=worlds,
    )
    config = SimpleNamespace(
        device=device,
        embodiment_type="humanoid_object",
        nv=cpu.nv,
        nu=cpu.nu,
        nq_obj=7,
        contact_site_ids=[cpu.site("palm").id, cpu.site("inactive").id],
        contact_rew_scale=1.0,
        base_pos_rew_scale=1.0,
        base_rot_rew_scale=1.0,
        pos_rew_scale=1.0,
        rot_rew_scale=1.0,
        joint_rew_scale=1.0,
        vel_rew_scale=1.0,
        terminal_rew_scale=1.0,
    )
    return config, env


def _position(env):
    stream = (
        wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(env.device)))
        if env.device != "cpu"
        else nullcontext()
    )
    with wp.ScopedDevice(env.device), stream:
        mjw.fwd_position(env.model_wp, env.data_wp)


def _reference(env, *, contact_mask=None, contact_points=None):
    device = env.device
    if contact_mask is None:
        contact_mask = torch.tensor([1.0, 0.0], device=device)
    if contact_points is None:
        contact_points = wp.to_torch(env.data_wp.site_xpos)[0].clone()
    return (
        wp.to_torch(env.data_wp.qpos)[0].clone(),
        wp.to_torch(env.data_wp.qvel)[0].clone(),
        wp.to_torch(env.data_wp.ctrl)[0].clone(),
        contact_mask,
        contact_points,
    )


def test_two_penetration_levels_filter_floor_and_ignore_global_padding(device):
    from retarget.task.humanoid.cost_contact import get_penetration_cost

    config, env = _environment(device, worlds=4, floor=True)
    qpos = env.data_wp.qpos.numpy()
    qpos[:, -7] = [0.199, 0.194, 0.18, 0.5]
    env.data_wp.qpos.assign(qpos)
    # The helper must refresh collision geometry after a qpos update.
    actual, info = get_penetration_cost(config, env)
    depth = torch.tensor([0.001, 0.006, 0.02, 0.0], device=device)
    soft = 15.0 * torch.relu(depth - 0.002).square()
    deep = 75.0 * torch.relu(depth - 0.01).square()
    torch.testing.assert_close(actual, soft + deep, atol=2e-8, rtol=2e-5)
    torch.testing.assert_close(
        info["soft_penetration_cost"], soft, atol=2e-8, rtol=2e-5
    )
    torch.testing.assert_close(
        info["deep_penetration_cost"], deep, atol=2e-8, rtol=2e-5
    )

    contact = env.data_wp.contact
    count = int(env.data_wp.nacon.numpy()[0])
    assert count > 4  # Floor contacts exist in addition to robot-object contacts.
    assert count < len(contact.dist)
    distances = contact.dist.numpy()
    geoms = contact.geom.numpy()
    worlds = contact.worldid.numpy()
    # MJWarp contacts are globally packed; unused slots can retain old values.
    distances[count:] = np.nan
    geoms[count:] = [
        env.model_cpu.geom("hand_geom").id,
        env.model_cpu.geom("object_geom").id,
    ]
    worlds[count:] = np.arange(len(worlds) - count) % env.num_worlds
    contact.dist.assign(distances)
    contact.geom.assign(geoms)
    contact.worldid.assign(worlds)
    padded, _ = get_penetration_cost(config, env, refresh=False)
    torch.testing.assert_close(padded, actual)


def test_correspondence_follows_object_translation_rotation_and_contact_mask(device):
    from retarget.task.humanoid.cost_contact import get_contact_cost

    config, env = _environment(device)
    _position(env)
    local = torch.tensor([[0.08, 0.02, 0.0], [float("nan")] * 3], device=device)
    reference = _reference(
        env,
        contact_points=local + torch.tensor([0.18, 0.0, 0.09], device=device),
    )
    qpos = env.data_wp.qpos.numpy()
    # Rotate the object by 90 degrees around z and translate it. Move the palm
    # onto the transformed active correspondence; inactive targets are unset.
    qpos[0, -7:-4] = [0.6, 0.4, 0.5]
    qpos[0, -4:] = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
    qpos[0, :3] = [0.48, 0.48, 0.5]
    env.data_wp.qpos.assign(qpos)
    _position(env)
    inferred, _ = get_contact_cost(config, env, reference)
    explicit, _ = get_contact_cost(config, env, reference, object_local_points=local)
    torch.testing.assert_close(inferred, torch.zeros_like(inferred), atol=1e-7, rtol=0)
    torch.testing.assert_close(explicit, inferred, atol=1e-7, rtol=0)

    qpos[0, -7] += 0.2
    env.data_wp.qpos.assign(qpos)
    _position(env)
    translated, _ = get_contact_cost(config, env, reference)
    torch.testing.assert_close(
        translated, torch.tensor([0.2], device=device), atol=1e-7, rtol=1e-6
    )


def test_comfree_action_gradient_matches_finite_difference_and_requires_contact(device):
    from retarget.task.humanoid.cost_contact import object_displacement_cost

    config, env = _environment(device, worlds=2)
    qpos = env.data_wp.qpos.numpy()
    qpos[1, -7] = 0.8
    env.data_wp.qpos.assign(qpos)
    before = {
        name: getattr(env.data_wp, name).numpy().copy()
        for name in ("qpos", "qvel", "ctrl")
    }
    action = torch.full((2, 1), 0.4, device=device, requires_grad=True)
    target = torch.tensor([0.1, 0.0, 0.0], device=device)
    cost, info = object_displacement_cost(config, env, action, target)
    displacement = info["predicted_displacement"]
    influence = torch.autograd.grad(displacement[0, 0], action, retain_graph=True)[0]
    assert influence[0, 0] > 1e-7
    torch.testing.assert_close(
        influence[1], torch.zeros_like(influence[1]), atol=0, rtol=0
    )
    torch.testing.assert_close(
        displacement[1], torch.zeros(3, device=device), atol=0, rtol=0
    )
    gradient = torch.autograd.grad(cost.sum(), action)[0]
    assert gradient[0, 0].abs() > 1e-8
    torch.testing.assert_close(
        gradient[1], torch.zeros_like(gradient[1]), atol=0, rtol=0
    )

    epsilon = 0.02
    plus, _ = object_displacement_cost(config, env, action.detach() + epsilon, target)
    minus, _ = object_displacement_cost(config, env, action.detach() - epsilon, target)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    torch.testing.assert_close(gradient[:, 0], finite_difference, atol=2e-7, rtol=0.02)
    for name, expected in before.items():
        np.testing.assert_array_equal(getattr(env.data_wp, name).numpy(), expected)


def test_reward_adds_contact_penetration_and_differentiable_motion_costs(device):
    from retarget.task.humanoid import cost, cost_contact

    config, env = _environment(device)
    _position(env)
    reference = _reference(env)
    action = torch.tensor([[0.4]], device=device, requires_grad=True)
    base, _ = cost.get_reward(config, env, reference)
    reward, info = cost_contact.get_reward(
        config,
        env,
        reference,
        action=action,
        target_displacement=torch.tensor([0.1, 0.0, 0.0], device=device),
    )
    expected = (
        base
        - info["contact_cost"]
        - info["penetration_cost"]
        - info["object_motion_cost"]
    )
    torch.testing.assert_close(reward, expected)
    derivative = torch.autograd.grad(reward.sum(), action)[0]
    assert derivative.abs().sum() > 1e-8


def test_sparse_and_dense_contact_rows_produce_same_prediction_and_gradient(device):
    from retarget.task.humanoid.cost_contact import object_displacement_cost

    results = []
    for jacobian in ("dense", "sparse"):
        config, env = _environment(device, worlds=2, jacobian=jacobian)
        assert env.model_wp.is_sparse == (jacobian == "sparse")
        qpos = env.data_wp.qpos.numpy()
        qpos[1, -7] = 0.8
        env.data_wp.qpos.assign(qpos)
        action = torch.full((2, 1), 0.4, device=device, requires_grad=True)
        value, info = object_displacement_cost(
            config, env, action, torch.tensor([0.1, 0.0, 0.0], device=device)
        )
        gradient = torch.autograd.grad(value.sum(), action)[0]
        results.append(
            (value.detach(), info["predicted_displacement"].detach(), gradient)
        )
    for dense, sparse in zip(*results):
        torch.testing.assert_close(sparse, dense, atol=2e-9, rtol=2e-5)
    assert results[1][2][0, 0].abs() > 1e-8


def test_action_graph_survives_refreshing_simulator_before_backward(device):
    from retarget.task.humanoid.cost_contact import object_displacement_cost

    config, env = _environment(device)
    action = torch.tensor([[0.4]], device=device, requires_grad=True)
    target = torch.tensor([0.1, 0.0, 0.0], device=device)
    value, info = object_displacement_cost(config, env, action, target)
    expected = torch.autograd.grad(value.sum(), action, retain_graph=True)[0]
    prediction = info["predicted_displacement"].detach().clone()

    # Reuse and overwrite the same Warp buffers while the old graph is alive.
    qpos = env.data_wp.qpos.numpy()
    qpos[0, -7] = 0.8
    env.data_wp.qpos.assign(qpos)
    other_action = torch.tensor([[-0.8]], device=device, requires_grad=True)
    replacement, replacement_info = object_displacement_cost(
        config, env, other_action, target
    )
    assert int(env.data_wp.nacon.numpy()[0]) == 0
    torch.testing.assert_close(
        replacement_info["predicted_displacement"], torch.zeros_like(prediction)
    )
    torch.testing.assert_close(
        torch.autograd.grad(replacement.sum(), other_action)[0],
        torch.zeros_like(other_action),
    )
    torch.testing.assert_close(
        info["predicted_displacement"].detach(), prediction, atol=0, rtol=0
    )
    actual = torch.autograd.grad(value.sum(), action)[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.abs().sum() > 1e-8


def test_affine_position_servo_matches_motor_force_and_respects_saturation(device):
    from retarget.task.humanoid.cost_contact import object_displacement_cost

    servo = '<position joint="push" kp="20" kv="1" ctrllimited="true" ctrlrange="-.1 .1" forcelimited="true" forcerange="-1 1"/>'
    results = []
    for actuator, controls in ((servo, [[0.02], [0.2]]), (None, [[0.26], [1.0]])):
        config, env = _environment(device, worlds=2, actuator=actuator)
        qpos, qvel = env.data_wp.qpos.numpy(), env.data_wp.qvel.numpy()
        qpos[:, 7] = 0.005
        qvel[:, 6] = 0.04
        env.data_wp.qpos.assign(qpos)
        env.data_wp.qvel.assign(qvel)
        action = torch.tensor(controls, device=device, requires_grad=True)
        value, info = object_displacement_cost(
            config, env, action, torch.tensor([0.1, 0.0, 0.0], device=device)
        )
        gradient = torch.autograd.grad(value.sum(), action)[0]
        results.append(
            (value.detach(), info["predicted_displacement"].detach(), gradient)
        )
    # Unsaturated servo force is 20 * (.02 - .005) - 1 * .04 = .26.
    # The second control clamps at .1 and its force clamps again at 1 N.
    for servo_result, motor_result in zip(results[0][:2], results[1][:2]):
        torch.testing.assert_close(servo_result, motor_result, atol=2e-9, rtol=2e-5)
    servo_gradient, motor_gradient = results[0][2], results[1][2]
    torch.testing.assert_close(
        servo_gradient[0], 20.0 * motor_gradient[0], atol=2e-9, rtol=2e-5
    )
    assert servo_gradient[0].abs().sum() > 1e-8
    torch.testing.assert_close(
        servo_gradient[1], torch.zeros_like(servo_gradient[1]), atol=0, rtol=0
    )
