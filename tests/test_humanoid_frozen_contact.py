"""Frozen contact trajectories stay differentiable without an engine rollout."""

from dataclasses import fields, is_dataclass
from types import SimpleNamespace

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import torch
import warp as wp
from test_humanoid_cost_contact import _environment, _reference


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    wp.init()
    if request.param != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA required")
    return request.param


def _references(env, horizon):
    return tuple(
        value.unsqueeze(0).repeat(horizon, *([1] * value.ndim))
        for value in _reference(env)
    )


def _state(env):
    return {
        name: getattr(env.data_wp, name).numpy().copy()
        for name in ("qpos", "qvel", "ctrl", "time")
    }


def _assert_state(env, expected):
    for name, value in expected.items():
        np.testing.assert_array_equal(getattr(env.data_wp, name).numpy(), value)


def _cached_tensors(value, prefix="cache"):
    """Compare immutable cached tensors without depending on every field name."""
    if isinstance(value, torch.Tensor):
        return {prefix: value.detach().clone()}
    if is_dataclass(value):
        children = ((field.name, getattr(value, field.name)) for field in fields(value))
    elif isinstance(value, dict):
        children = value.items()
    elif isinstance(value, (tuple, list)):
        children = enumerate(value)
    else:
        return {}
    snapshots = {}
    for key, child in children:
        snapshots.update(_cached_tensors(child, f"{prefix}.{key}"))
    return snapshots


def _forbid_engine(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A cached candidate must use only Torch dynamics")

    for name in dir(mjw):
        if (
            name.startswith("fwd_")
            or name in ("step", "forward", "collision", "kinematics", "put_data")
        ) and callable(getattr(mjw, name)):
            monkeypatch.setattr(mjw, name, forbidden)
    for name in (
        "mj_forward",
        "mj_step",
        "mj_fwdPosition",
        "mj_fwdVelocity",
        "mj_collision",
        "mj_kinematics",
        "mj_jacSite",
    ):
        monkeypatch.setattr(mujoco, name, forbidden)
    for name in ("launch", "to_torch", "from_torch"):
        monkeypatch.setattr(wp, name, forbidden)


def test_nominal_cache_is_reused_for_changed_candidate_batches_and_costs(
    device, monkeypatch
):
    from retarget.env.humanoid.frozen_contact import FrozenContactRollout
    from retarget.task.humanoid.cost_contact import get_trajectory_cost

    config, env = _environment(device)
    horizon = 4
    nominal = torch.full((horizon, config.nu), 0.2, device=device)
    before = _state(env)
    cache = FrozenContactRollout.from_env(config, env, nominal)
    _assert_state(env, before)
    frames_before = _cached_tensors(cache.frames)
    assert frames_before
    refs = _references(env, horizon)
    actions = torch.full(
        (3, horizon, config.nu), 0.4, device=device, requires_grad=True
    )

    with monkeypatch.context() as patched:
        _forbid_engine(patched)
        first = cache.rollout(actions)
        cost, info = get_trajectory_cost(config, first, refs)
        second = cache.rollout(actions.detach() * -0.5)
        changed, _ = get_trajectory_cost(config, second, refs)
        assert first.qpos.shape == (3, horizon + 1, env.model_cpu.nq)
        assert first.qvel.shape == (3, horizon + 1, env.model_cpu.nv)
        assert first.site_xpos.shape == (
            3,
            horizon + 1,
            len(config.contact_site_ids),
            3,
        )
        assert first.object_displacement.shape == (3, horizon, 3)
        assert len(first.penetration_distance) == horizon
        assert torch.isfinite(cost).all()
        assert torch.isfinite(changed).all()
        assert not torch.allclose(first.qpos, second.qpos, atol=1e-7, rtol=0)
        torch.testing.assert_close(cost, info["step_cost"].mean(dim=1))
        gradient = torch.autograd.grad(cost.sum(), actions)[0]
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 1e-8

    _assert_state(env, before)
    frames_after = _cached_tensors(cache.frames)
    assert frames_before.keys() == frames_after.keys()
    for key in frames_before:
        torch.testing.assert_close(
            frames_after[key], frames_before[key], atol=0, rtol=0
        )


def test_early_action_affects_final_object_pose_and_late_action_is_causal(device):
    from retarget.env.humanoid.frozen_contact import FrozenContactRollout

    config, env = _environment(device)
    horizon = 4
    cache = FrozenContactRollout.from_current(config, env, horizon)
    actions = torch.zeros((1, horizon, config.nu), device=device, requires_grad=True)
    result = cache.rollout(actions)
    final_x = result.qpos[0, -1, -7]
    gradient = torch.autograd.grad(final_x, actions)[0]
    assert gradient[0, 0, 0] > 1e-7

    epsilon = 0.02
    delta = torch.zeros_like(actions)
    delta[0, 0, 0] = epsilon
    plus = cache.rollout(actions.detach() + delta)
    minus = cache.rollout(actions.detach() - delta)
    finite_difference = (plus.qpos[0, -1, -7] - minus.qpos[0, -1, -7]) / (2 * epsilon)
    torch.testing.assert_close(
        gradient[0, 0, 0], finite_difference, atol=3e-7, rtol=0.025
    )
    assert (plus.qpos[0, -1, -7] - result.qpos[0, -1, -7]).abs() > 1e-7

    late = torch.zeros_like(actions)
    late[0, 2, 0] = 0.8
    late_result = cache.rollout(late)
    torch.testing.assert_close(
        late_result.qpos[:, :3], result.qpos[:, :3], atol=0, rtol=0
    )
    torch.testing.assert_close(
        late_result.qvel[:, :3], result.qvel[:, :3], atol=0, rtol=0
    )
    assert (late_result.qpos[:, 3:] - result.qpos[:, 3:]).abs().sum() > 1e-6
    assert (late_result.qpos[:, 4] - result.qpos[:, 4]).abs().sum() > 1e-6


def test_nominal_mjwarp_correction_matches_engine_and_retains_gradients(
    device, monkeypatch
):
    from retarget.env.humanoid.frozen_contact import FrozenContactRollout
    from retarget.task.humanoid.cost_contact import _stream

    config, env = _environment(device)
    env.model_cpu.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    env.model_wp.opt.integrator = int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
    horizon = 4
    nominal = torch.linspace(0.2, 0.5, horizon, device=device)[:, None]
    before = _state(env)
    cache = FrozenContactRollout.from_env(
        config, env, nominal, anchor_to_mjwarp=True
    )
    _assert_state(env, before)
    assert any(frame.velocity_correction.abs().max() > 1e-5 for frame in cache.frames)
    positions = [wp.to_torch(env.data_wp.qpos)[0].clone()]
    velocities = [wp.to_torch(env.data_wp.qvel)[0].clone()]
    for control in nominal:
        with wp.ScopedDevice(device), _stream(env):
            wp.copy(env.data_wp.ctrl, wp.from_torch(control[None].contiguous()))
            mjw.step(env.model_wp, env.data_wp)
        positions.append(wp.to_torch(env.data_wp.qpos)[0].clone())
        velocities.append(wp.to_torch(env.data_wp.qvel)[0].clone())

    actions = nominal[None].detach().clone().requires_grad_(True)
    with monkeypatch.context() as patched:
        _forbid_engine(patched)
        result = cache.rollout(actions)
        torch.testing.assert_close(
            result.qpos[0], torch.stack(positions), atol=2e-6, rtol=2e-5
        )
        torch.testing.assert_close(
            result.qvel[0], torch.stack(velocities), atol=2e-6, rtol=2e-5
        )
        final_x = result.qpos[0, -1, cache.object_qpos_adr]
        gradient = torch.autograd.grad(final_x, actions)[0]
        assert torch.isfinite(gradient).all()
        assert gradient[0, 0, 0].abs() > 1e-7
        delta = torch.zeros_like(actions)
        delta[0, 0, 0] = 0.02
        plus = cache.rollout(actions.detach() + delta)
        minus = cache.rollout(actions.detach() - delta)
        numerical = (
            plus.qpos[0, -1, cache.object_qpos_adr]
            - minus.qpos[0, -1, cache.object_qpos_adr]
        ) / 0.04
        torch.testing.assert_close(gradient[0, 0, 0], numerical, atol=5e-7, rtol=0.03)


def test_terminal_site_jacobian_uses_terminal_spatial_dofs(device):
    from retarget.env.humanoid.frozen_contact import FrozenContactRollout

    config, env = _environment(device)
    qpos, qvel = env.data_wp.qpos.numpy(), env.data_wp.qvel.numpy()
    qpos[0, -7] = 0.8  # Isolate the terminal FK from contact forces.
    qvel[0, 5] = 1.1
    qvel[0, 6] = 0.3
    env.data_wp.qpos.assign(qpos)
    env.data_wp.qvel.assign(qvel)
    nominal = torch.full((3, config.nu), 0.2, device=device)
    cache = FrozenContactRollout.from_env(config, env, nominal)
    terminal = cache._geometry[-1]

    cpu_data = mujoco.MjData(env.model_cpu)
    cpu_data.qpos[:] = terminal.qpos.cpu().numpy()
    mujoco.mj_forward(env.model_cpu, cpu_data)
    expected = []
    for site in config.contact_site_ids:
        translation = np.zeros((3, env.model_cpu.nv))
        rotation = np.zeros_like(translation)
        mujoco.mj_jacSite(env.model_cpu, cpu_data, translation, rotation, site)
        expected.append(translation)
    torch.testing.assert_close(
        terminal.site_jacobian,
        torch.as_tensor(np.asarray(expected), device=device, dtype=terminal.qpos.dtype),
        atol=2e-6,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        terminal.site_xpos,
        torch.as_tensor(
            cpu_data.site_xpos[config.contact_site_ids],
            device=device,
            dtype=terminal.qpos.dtype,
        ),
        atol=2e-6,
        rtol=2e-5,
    )


def test_final_robot_root_joint_and_object_gradients_cover_action_sequence(device):
    from retarget.env.humanoid.frozen_contact import FrozenContactRollout

    config, env = _environment(device)
    horizon = 4
    cache = FrozenContactRollout.from_current(config, env, horizon)
    actions = torch.full(
        (1, horizon, config.nu), 0.4, device=device, requires_grad=True
    )
    result = cache.rollout(actions)
    # Root x, the actuated sliding joint, and object x all depend on earlier
    # controls through the coupled contact dynamics. Once contact releases,
    # later controls can correctly have zero influence on the object.
    coordinates = (0, 7, env.model_cpu.nq - 7)
    final = result.qpos[0, -1, list(coordinates)]
    derivatives = torch.stack(
        [
            torch.autograd.grad(value, actions, retain_graph=True)[0][0, :, 0]
            for value in final
        ]
    )
    assert torch.isfinite(derivatives).all()
    assert torch.all(derivatives[:, 0].abs() > 1e-8)
    assert torch.all(derivatives[:2].abs() > 1e-8)
    epsilon = 0.02
    finite_difference = []
    for step in range(horizon):
        change = torch.zeros_like(actions)
        change[0, step, 0] = epsilon
        plus = cache.rollout(actions.detach() + change)
        minus = cache.rollout(actions.detach() - change)
        finite_difference.append(
            (
                plus.qpos[0, -1, list(coordinates)]
                - minus.qpos[0, -1, list(coordinates)]
            )
            / (2 * epsilon)
        )
    torch.testing.assert_close(
        derivatives, torch.stack(finite_difference, dim=1), atol=3e-7, rtol=0.03
    )


def test_grouped_joint_integration_and_tangents_match_mujoco(device):
    from retarget.env.humanoid.frozen_contact import (
        _difference, _integrate, _joint_layout,
    )

    cpu = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody>
        <body><freejoint/><geom type="sphere" size=".1"/>
          <body><joint type="hinge" axis="0 1 0"/>
          <joint type="slide" axis="1 0 0"/>
          <geom type="sphere" size=".1"/>
            <body><joint type="ball"/><geom type="sphere" size=".1"/>
              <body><joint type="hinge" axis="0 0 1"/>
              <geom type="sphere" size=".1"/></body>
            </body>
          </body>
        </body></worldbody></mujoco>"""
    )
    layout = _joint_layout(cpu)
    assert any(width == 2 for _, _, _, width in layout)
    initial = cpu.qpos0.copy()
    warm_velocity = np.linspace(-0.2, 0.4, cpu.nv)
    mujoco.mj_integratePos(cpu, initial, warm_velocity, 0.7)
    velocity = np.linspace(-0.6, 0.3, cpu.nv)
    timestep = 0.13
    expected = initial.copy()
    mujoco.mj_integratePos(cpu, expected, velocity, timestep)
    initial_tensor = torch.as_tensor(initial, dtype=torch.float64, device=device)
    velocity_tensor = torch.tensor(
        velocity, dtype=torch.float64, device=device, requires_grad=True
    )
    actual = _integrate(initial_tensor, velocity_tensor, timestep, layout)
    torch.testing.assert_close(
        actual, torch.as_tensor(expected, device=device), atol=2e-12, rtol=2e-12
    )
    tangent = np.zeros(cpu.nv)
    mujoco.mj_differentiatePos(cpu, tangent, 1.0, initial, expected)
    torch.testing.assert_close(
        _difference(actual, initial_tensor, layout),
        torch.as_tensor(tangent, device=device),
        atol=2e-12,
        rtol=2e-12,
    )
    assert torch.isfinite(torch.autograd.grad(actual.sum(), velocity_tensor)[0]).all()


def test_mesh_collision_jacobian_couples_robot_root_joint_and_object(device):
    from retarget.env.humanoid.frozen_contact import FrozenContactRollout

    # A convex tetrahedral hand has a unique tip against the box's flat face.
    # The off-axis tip also exercises rotational columns of both body Jacobians.
    cpu = mujoco.MjModel.from_xml_string(
        """<mujoco>
        <option timestep=".05" gravity="0 0 0" jacobian="dense"/>
        <asset><mesh name="hand_mesh" vertex="
          .1 .025 .015  -.08 -.07 -.06  -.08 .07 -.06  -.08 0 .08"/></asset>
        <worldbody>
          <body name="robot" pos="0 0 .3"><freejoint/>
            <inertial pos="0 0 0" mass="10" diaginertia="1 1 1"/>
            <body name="hand"><joint name="push" type="slide" axis="1 0 0"/>
              <geom name="hand_geom" type="mesh" mesh="hand_mesh" mass="1"/>
            </body>
          </body>
          <body name="object" pos=".19 0 .3"><freejoint/>
            <geom name="object_geom" type="box" size=".1 .12 .12" mass="1"/>
          </body>
        </worldbody><actuator><motor joint="push"/></actuator>
        </mujoco>"""
    )
    initial = mujoco.MjData(cpu)
    mujoco.mj_forward(cpu, initial)
    with wp.ScopedDevice(device):
        model = mjw.put_model(cpu)
        data = mjw.put_data(cpu, initial, nworld=1, nconmax=12, njmax=48)
    env = SimpleNamespace(
        model_cpu=cpu, model_wp=model, data_wp=data, device=device, num_worlds=1
    )
    config = SimpleNamespace(
        embodiment_type="humanoid_object", contact_site_ids=[], device=device
    )
    cache = FrozenContactRollout.from_current(config, env, horizon=3)
    frame = cache.frames[0]
    assert cpu.geom_type[cpu.geom("hand_geom").id] == mujoco.mjtGeom.mjGEOM_MESH
    assert frame.contact_distance.shape == (1,)
    assert frame.contact_distance[0] < 0
    robot_root, robot_joint, object_root = 0, 6, 7
    columns = (robot_root, robot_joint, object_root)
    normal = frame.contact_jacobian[0]
    assert torch.all(normal[list(columns)].abs() > 0.9)
    assert frame.point_jacobian.shape == (1, 3, cpu.nv)
    assert torch.all(frame.point_jacobian[0, 0, list(columns)].abs() > 0.9)
    # Both free roots have nonzero angular columns at the off-axis mesh tip.
    assert normal[3:6].abs().max() > 0.01
    assert normal[10:13].abs().max() > 0.01

    epsilon = 1e-5
    finite_difference = []
    for dof in (*columns, 5, 12):
        gaps = []
        for sign in (1, -1):
            perturbed = mujoco.MjData(cpu)
            perturbed.qpos[:] = initial.qpos
            tangent = np.zeros(cpu.nv)
            tangent[dof] = sign
            mujoco.mj_integratePos(cpu, perturbed.qpos, tangent, epsilon)
            mujoco.mj_forward(cpu, perturbed)
            assert perturbed.ncon == 1
            gaps.append(perturbed.contact[0].dist)
        finite_difference.append((gaps[0] - gaps[1]) / (2 * epsilon))
    torch.testing.assert_close(
        normal[[*columns, 5, 12]],
        torch.tensor(finite_difference, dtype=normal.dtype, device=device),
        atol=3e-4,
        rtol=0.02,
    )

    actions = torch.full((1, 3, cpu.nu), 0.2, device=device, requires_grad=True)
    final_x = cache.rollout(actions).qpos[0, -1, cache.object_qpos_adr]
    gradient = torch.autograd.grad(final_x, actions)[0]
    assert torch.isfinite(gradient).all()
    assert gradient[0, 0, 0] > 1e-7
    delta = torch.zeros_like(actions)
    delta[0, 0, 0] = 0.02
    plus = cache.rollout(actions.detach() + delta).qpos[0, -1, cache.object_qpos_adr]
    minus = cache.rollout(actions.detach() - delta).qpos[0, -1, cache.object_qpos_adr]
    torch.testing.assert_close(
        gradient[0, 0, 0], (plus - minus) / 0.04, atol=5e-7, rtol=0.03
    )


def test_physical_contact_cost_has_gradients_without_annotated_sites(device, monkeypatch):
    from retarget.env.humanoid.frozen_contact import FrozenContactRollout
    from retarget.task.humanoid.cost_contact import ContactCostOptions, get_trajectory_cost

    config, env = _environment(device)
    config.contact_site_ids = []
    for name in (
        "base_pos_rew_scale", "base_rot_rew_scale", "pos_rew_scale",
        "rot_rew_scale", "joint_rew_scale", "vel_rew_scale",
    ):
        setattr(config, name, 0.0)
    qvel = env.data_wp.qvel.numpy()
    qvel[0, 8] = 0.4  # Tangential motion of the object's physical patch.
    env.data_wp.qvel.assign(qvel)
    options = ContactCostOptions(
        contact_weight=0.0,
        physical_contact_weight=1.0,
        soft_penetration_weight=0.0,
        penetration_weight=0.0,
        object_motion_weight=0.0,
    )
    horizon = 3
    cache = FrozenContactRollout.from_current(config, env, horizon, options=options)
    frame = cache.frames[0]
    assert frame.point_offset.shape == (1, 3)
    # Translating both complete bodies equally preserves their fixed patch
    # correspondence, even though their individual Jacobian blocks are nonzero.
    common = torch.zeros(env.model_cpu.nv, device=device)
    common[:3] = common[7:10] = torch.tensor([0.3, -0.2, 0.1], device=device)
    torch.testing.assert_close(
        torch.einsum("pcv,v->pc", frame.point_jacobian, common),
        torch.zeros((1, 3), device=device), atol=1e-7, rtol=0,
    )
    refs = list(_references(env, horizon))
    refs[3] = torch.zeros((horizon, 0), device=device)
    refs[4] = torch.zeros((horizon, 0, 3), device=device)
    actions = torch.full(
        (1, horizon, env.model_cpu.nu), 0.4, device=device, requires_grad=True
    )
    with monkeypatch.context() as patched:
        _forbid_engine(patched)
        result = cache.rollout(actions)
        assert all(error.shape == (1, 1, 3) for error in result.contact_point_error)
        assert result.contact_point_error[-1][0, 0, 1].abs() > 1e-6
        cost, info = get_trajectory_cost(config, result, tuple(refs), options=options)
        expected = torch.stack(
            [torch.linalg.vector_norm(error, dim=-1).sum(-1)
             for error in result.contact_point_error], dim=1
        )
        torch.testing.assert_close(info["contact_cost"], expected)
        torch.testing.assert_close(cost, expected.mean(dim=1))
        assert cost[0] > 0
        gradient = torch.autograd.grad(cost.sum(), actions)[0]
        assert torch.isfinite(gradient).all()
        assert gradient[0, 0, 0].abs() > 1e-8
        delta = torch.zeros_like(actions)
        delta[0, 0, 0] = 0.02
        plus, _ = get_trajectory_cost(
            config, cache.rollout(actions.detach() + delta), tuple(refs), options=options
        )
        minus, _ = get_trajectory_cost(
            config, cache.rollout(actions.detach() - delta), tuple(refs), options=options
        )
        torch.testing.assert_close(
            gradient[0, 0, 0], ((plus - minus) / 0.04)[0], atol=5e-7, rtol=0.03
        )


def test_separated_cached_contacts_stop_applying_force(device):
    from retarget.env.humanoid.frozen_contact import FrozenContactRollout

    config, env = _environment(device)
    qvel = env.data_wp.qvel.numpy()
    qvel[0, -6] = 0.1
    env.data_wp.qvel.assign(qvel)
    horizon = 7
    cache = FrozenContactRollout.from_current(config, env, horizon)
    result = cache.rollout(torch.zeros((1, horizon, config.nu), device=device))
    gaps = torch.stack([gap[0, 0] for gap in result.penetration_distance])
    assert gaps[0] < 0
    separated = torch.nonzero(gaps > 0).flatten()
    assert len(separated) > 1
    first = int(separated[0])
    assert first < horizon - 1
    assert result.qvel[0, 1, -6] > 0.1
    torch.testing.assert_close(
        result.qvel[:, first + 2 :],
        result.qvel[:, first + 1 : first + 2].expand(-1, horizon - first - 1, -1),
        atol=2e-7,
        rtol=2e-6,
    )
    assert torch.all(gaps[1:] > gaps[:-1])


def test_penetration_uses_physical_normal_gap_not_friction_ray_distance(device):
    from retarget.env.humanoid.frozen_contact import FrozenContactRollout

    config, env = _environment(device)
    qvel = env.data_wp.qvel.numpy()
    qvel[0, 1] = 0.2  # Tangential to the sphere-sphere contact normal.
    env.data_wp.qvel.assign(qvel)
    horizon = 3
    cache = FrozenContactRollout.from_current(config, env, horizon)
    frame = cache.frames[0]
    assert frame.contact_jacobian.shape == (1, env.model_cpu.nv)
    assert frame.jacobian.shape[0] > len(frame.contact_distance)
    assert frame.jacobian[:, 1].abs().max() > 0
    torch.testing.assert_close(
        frame.contact_jacobian[:, 1], torch.zeros(1, device=device), atol=1e-7, rtol=0
    )
    result = cache.rollout(torch.zeros((1, horizon, config.nu), device=device))
    for t, gap in enumerate(result.penetration_distance):
        assert gap.shape == (1, 1)
        # The frozen normal lies along x. Friction-ray y displacement must
        # never be mistaken for geometric penetration.
        pose = result.qpos[:, t + 1]
        expected = pose[:, -7] - pose[:, 0] - pose[:, 7] - 0.2
        torch.testing.assert_close(gap[:, 0], expected, atol=2e-6, rtol=2e-5)


def test_sparse_and_dense_cached_trajectories_and_gradients_match(device):
    from retarget.env.humanoid.frozen_contact import FrozenContactRollout
    from retarget.task.humanoid.cost_contact import get_trajectory_cost

    results = []
    horizon = 4
    for jacobian in ("dense", "sparse"):
        config, env = _environment(device, jacobian=jacobian)
        nominal = torch.full((horizon, config.nu), 0.2, device=device)
        cache = FrozenContactRollout.from_env(config, env, nominal)
        actions = torch.full(
            (2, horizon, config.nu), 0.4, device=device, requires_grad=True
        )
        trajectory = cache.rollout(actions)
        cost, _ = get_trajectory_cost(config, trajectory, _references(env, horizon))
        gradient = torch.autograd.grad(cost.sum(), actions)[0]
        results.append(
            (
                trajectory.qpos.detach(),
                trajectory.object_displacement.detach(),
                cost.detach(),
                gradient,
            )
        )
    for dense, sparse in zip(*results):
        torch.testing.assert_close(sparse, dense, atol=3e-7, rtol=3e-5)
    assert results[1][-1].abs().sum() > 1e-8


def test_trajectory_score_has_original_weights_two_barriers_and_terminal_mean(device):
    from retarget.task.humanoid.cost_contact import get_trajectory_cost

    config = SimpleNamespace(
        device=device,
        embodiment_type="humanoid_object",
        nv=13,
        nu=1,
        nq_obj=7,
        contact_site_ids=[0, 1],
        base_pos_rew_scale=0.0,
        base_rot_rew_scale=0.0,
        pos_rew_scale=0.0,
        rot_rew_scale=0.0,
        joint_rew_scale=0.0,
        vel_rew_scale=0.0,
        terminal_rew_scale=3.0,
    )
    horizon = 3
    qpos = torch.zeros((1, horizon + 1, 15), device=device)
    qpos[..., 3] = qpos[..., -4] = 1.0
    qpos[0, :, -7] = torch.tensor([0.18, 0.2, 0.3, 0.4], device=device)
    qpos.requires_grad_()
    sites = torch.zeros((1, horizon + 1, 2, 3), device=device)
    sites[..., 0, 0] = 0.1
    sites[..., 1, :] = float("nan")
    velocity = torch.zeros((1, horizon + 1, config.nv), device=device)
    gaps = tuple(
        torch.tensor([[value]], device=device) for value in (-0.001, -0.006, -0.02)
    )
    displacement = qpos[:, 1:, -7:-4] - qpos[:, :-1, -7:-4]
    result = SimpleNamespace(
        qpos=qpos,
        qvel=velocity,
        site_xpos=sites,
        penetration_distance=gaps,
        object_displacement=displacement,
        object_qpos_adr=8,
        object_dof_adr=7,
        timestep=torch.full((horizon,), 0.1, device=device),
    )
    refs = (
        qpos[0, :horizon].detach(),
        velocity[0, :horizon],
        torch.zeros((horizon, 1), device=device),
        torch.tensor([[1.0, 0.0]], device=device).repeat(horizon, 1),
        torch.zeros((horizon, 2, 3), device=device),
    )
    local = torch.tensor([[0.0, 0.0, 0.0], [float("nan")] * 3], device=device)
    target = torch.tensor([[0.05, 0.0, 0.0]], device=device).repeat(horizon, 1)
    cost, info = get_trajectory_cost(
        config, result, refs, object_local_points=local, target_displacement=target
    )
    contact = torch.tensor([[0.1, 0.2, 0.3]], device=device)
    depth = torch.tensor([[0.001, 0.006, 0.02]], device=device)
    soft = 15.0 * torch.relu(depth - 0.002).square()
    deep = 75.0 * torch.relu(depth - 0.01).square()
    motion = torch.tensor([[0.0009, 0.0025, 0.0025]], device=device)
    expected_step = (contact + soft + deep + motion) * torch.tensor(
        [1.0, 1.0, 3.0], device=device
    )
    torch.testing.assert_close(info["contact_cost"], contact)
    torch.testing.assert_close(info["soft_penetration_cost"], soft)
    torch.testing.assert_close(info["deep_penetration_cost"], deep)
    torch.testing.assert_close(info["object_motion_cost"], motion)
    torch.testing.assert_close(info["step_cost"], expected_step)
    torch.testing.assert_close(cost, expected_step.mean(dim=1))
    gradient = torch.autograd.grad(cost.sum(), qpos)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 1e-8
