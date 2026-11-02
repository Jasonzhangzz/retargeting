"""SCSP uses fixed palm candidates and actual robot collision dynamics."""

from types import SimpleNamespace

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import torch
import torch.nn.functional as F
import warp as wp

from retarget.env.humanoid.scsp import HumanoidSCSP, _object_points
from retarget.solver.scsp import ObjectSelection, SCSPSettings, solve


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    wp.init()
    if request.param != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA required")
    return request.param


def _task(device, *, gap=.002, horizon=4):
    cpu = mujoco.MjModel.from_xml_string(f"""<mujoco>
      <option timestep=".1" gravity="0 0 0" jacobian="dense"/>
      <worldbody>
        <body name="robot"><freejoint/>
          <inertial mass="10" pos="0 0 0" diaginertia="1 1 1"/>
          <body name="left_hand" pos="0 -.08 0"><joint name="push_left" type="slide" axis="1 0 0"/>
            <geom name="left_pusher" type="sphere" pos=".04 0 0" size=".06" mass="1"/>
            <site name="left_palm" pos=".08 0 0"/>
          </body>
          <body name="right_hand" pos="0 .08 0"><joint name="push_right" type="slide" axis="1 0 0"/>
            <geom name="right_pusher" type="sphere" pos=".04 0 0" size=".06" mass="1"/>
            <site name="right_palm" pos=".08 0 0"/>
          </body>
          <body name="left_foot" pos="-.1 -.1 -.2"><joint name="left_ankle" type="slide" axis="0 0 1"/><geom type="sphere" size=".04" mass=".1"/>
            <site name="left_sole" pos="0 0 -.02"/>
          </body>
          <body name="right_foot" pos="-.1 .1 -.2"><joint name="right_ankle" type="slide" axis="0 0 1"/><geom type="sphere" size=".04" mass=".1"/>
            <site name="right_sole" pos="0 0 -.02"/>
          </body>
        </body>
        <body name="object" pos="{.2 + gap} 0 0"><freejoint/>
          <geom name="box" type="box" size=".1 .15 .1" mass="2"/>
        </body>
      </worldbody>
      <actuator><motor joint="push_left" gear="1"/><motor joint="push_right" gear="1"/></actuator>
    </mujoco>""")
    data = mujoco.MjData(cpu)
    mujoco.mj_forward(cpu, data)
    with wp.ScopedDevice(device):
        model_wp = mjw.put_model(cpu)
        model_wp.opt.graph_conditional = False
        data_wp = mjw.put_data(cpu, data, nconmax=16, njmax=64)
    env = SimpleNamespace(model_cpu=cpu, data_cpu=data, model_wp=model_wp,
                          data_wp=data_wp, device=device)
    config = SimpleNamespace(
        device=device, embodiment_type="humanoid_object", nq=cpu.nq,
        nv=cpu.nv, nu=cpu.nu, nq_obj=7, horizon_steps=horizon,
        contact_site_ids=[], contact_rew_scale=0.,
        base_pos_rew_scale=1., base_rot_rew_scale=.3,
        pos_rew_scale=1., rot_rew_scale=.3, joint_rew_scale=.1,
        vel_rew_scale=.001, terminal_rew_scale=2.,
    )
    qpos = torch.tensor(data.qpos, device=device, dtype=torch.float32).repeat(horizon+1, 1)
    qpos[:, -7] += torch.linspace(0, .01, horizon+1, device=device)
    refs = (
        qpos, torch.zeros((horizon+1, cpu.nv), device=device),
        torch.zeros((horizon+1, cpu.nu), device=device),
        torch.zeros((horizon+1, 4), device=device),
        torch.zeros((horizon+1, 4, 3), device=device),
    )
    task = SimpleNamespace(config=config, env=env, refs=refs, first_step=0,
                           nominal=torch.zeros((horizon, cpu.nu), device=device),
                           knots=2, control_scale=1.)

    def controls(actions):
        knots = actions.reshape(-1, task.knots, cpu.nu)
        return task.nominal + F.interpolate(knots.transpose(1, 2), size=horizon,
                                            mode="linear", align_corners=True).transpose(1, 2)

    task.controls = controls
    return task


def _adapter(task, *, operation_mask=None, **options):
    return HumanoidSCSP(task, options={"foot_sites": ["left_sole", "right_sole"], **options},
                        operation_mask=operation_mask)


def _selection(adapter, *, index=0):
    indices = torch.full((adapter.hands,), index, device=adapter.device, dtype=torch.long)
    force = torch.zeros((adapter.knots*adapter.hands*3,), device=adapter.device)
    evaluation = adapter.object_evaluate(indices[None], force[None])
    return ObjectSelection(indices, force, evaluation.consensus[0], float(evaluation.cost[0]))


def _forbid_engine(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("The upper object callback revisited the simulation engine")
    for name in dir(mjw):
        if callable(getattr(mjw, name)) and (name.startswith("fwd_") or name in ("step", "forward", "collision", "kinematics", "com_pos", "put_data", "make_data")):
            monkeypatch.setattr(mjw, name, forbidden)
    for name in ("mj_step", "mj_forward", "mj_kinematics", "mj_jac", "mj_jacSite", "mj_collision"):
        monkeypatch.setattr(mujoco, name, forbidden)
    for name in ("launch", "to_torch", "from_torch"):
        monkeypatch.setattr(wp, name, forbidden)


def test_object_point_jacobian_matches_mujoco_and_velocity_difference(device):
    task = _task(device)
    cpu = task.env.model_cpu
    data = mujoco.MjData(cpu)
    qpos = data.qpos.copy()
    qpos[-4:] = np.array([.8, .2, -.3, .1]) / np.linalg.norm([.8, .2, -.3, .1])
    data.qpos[:] = qpos
    mujoco.mj_forward(cpu, data)
    pose = torch.tensor(qpos[-7:][None], device=device, dtype=torch.float64)
    local = pose.new_tensor([[[-.1, .03, .04], [.02, -.1, -.05]]])
    point, jacobian = _object_points(pose, local)
    for k in range(2):
        jp = np.zeros((3, cpu.nv))
        jr = np.zeros_like(jp)
        mujoco.mj_jac(cpu, data, jp, jr, point[0, k].cpu().numpy(), cpu.body("object").id)
        np.testing.assert_allclose(jacobian[0, k].cpu().numpy(), jp[:, -6:], atol=1e-12, rtol=1e-12)
    step = 1e-5
    for column in range(6):
        velocity = np.zeros(cpu.nv)
        velocity[-6 + column] = 1
        plus, minus = qpos.copy(), qpos.copy()
        mujoco.mj_integratePos(cpu, plus, velocity, step)
        mujoco.mj_integratePos(cpu, minus, velocity, -step)
        pplus, _ = _object_points(pose.new_tensor(plus[-7:][None]), local)
        pminus, _ = _object_points(pose.new_tensor(minus[-7:][None]), local)
        torch.testing.assert_close((pplus-pminus)/(2*step), jacobian[..., column], atol=2e-11, rtol=2e-9)


def test_single_initial_palm_candidate_pool_stays_fixed_over_horizon(device):
    task = _task(device)
    task.refs[0][1:, 0] = torch.linspace(.01, .04, task.config.horizon_steps, device=device)
    adapter = _adapter(task)
    assert adapter.candidates.local_points.shape == (1, 2, 11, 3)
    assert adapter.candidates.local_normals.shape == (1, 2, 11, 3)
    expected_palms = torch.tensor([[[.08, -.08, 0], [.08, .08, 0]]], device=device)
    torch.testing.assert_close(adapter.candidates.reference_hand_positions, expected_palms, atol=1e-6, rtol=0)
    sites = tuple(task.env.model_cpu.site(name).id for name in ("left_palm", "right_palm", "left_sole", "right_sole"))
    assert tuple(adapter.config.contact_site_ids) == sites
    expected_four = torch.tensor([[.08, -.08, 0], [.08, .08, 0],
                                  [-.1, -.1, -.22], [-.1, .1, -.22]], device=device)
    torch.testing.assert_close(adapter.reference_sites[0], expected_four, atol=1e-6, rtol=0)
    problem = adapter.problem(torch.zeros(task.knots*task.config.nu, device=device))
    assert problem.candidate_counts.tolist() == [11, 11]
    combinations = torch.cartesian_prod(torch.arange(11, device=device), torch.arange(11, device=device))
    assert combinations.shape == (121, 2)
    points, normals = adapter.select(combinations)
    assert points.shape == (121, task.config.horizon_steps, 2, 3)
    torch.testing.assert_close(points, points[:, :1].expand_as(points), atol=0, rtol=0)
    torch.testing.assert_close(normals, normals[:, :1].expand_as(normals), atol=0, rtol=0)


def test_force_projection_respects_unilateral_pyramid_and_force_cap(device):
    adapter = _adapter(_task(device), friction=.6, max_contact_force=3.)
    generator = torch.Generator(device=device).manual_seed(101)
    values = torch.randn((64, adapter.knots*adapter.hands*3), device=device, generator=generator) * 100
    ids = torch.zeros((64, 2), device=device, dtype=torch.long)
    projected = adapter.force_project(ids, values).reshape(64, adapter.knots, 2, 3)
    assert bool((projected[..., 0] >= 0).all())
    assert bool((projected[..., 1:].abs().sum(-1) <= .6 * projected[..., 0] + 1e-6).all())
    assert bool((projected.norm(dim=-1) <= 3.000001).all())
    dense = adapter.dense_forces(projected.flatten(1))
    assert bool((dense[..., 1:].abs().sum(-1) <= .6*dense[..., 0] + 1e-6).all())
    assert bool((dense.norm(dim=-1) <= 3.000001).all())


@pytest.mark.parametrize("gap", [.3, .002])
def test_discrete_object_targets_do_not_create_or_change_robot_contacts(device, gap):
    task = _task(device, gap=gap)
    before = {name: getattr(task.env.data_wp, name).numpy().copy() for name in ("qpos", "qvel", "ctrl", "time")}
    adapter = _adapter(task)
    action = torch.full((task.knots*task.config.nu,), .5, device=device)
    nearest = adapter.robot_evaluate(action, _selection(adapter))
    neighbor = adapter.robot_evaluate(action, _selection(adapter, index=10))
    torch.testing.assert_close(nearest.payload["qpos"], neighbor.payload["qpos"], atol=0, rtol=0)
    torch.testing.assert_close(nearest.payload["qvel"], neighbor.payload["qvel"], atol=0, rtol=0)
    torch.testing.assert_close(nearest.consensus, neighbor.consensus, atol=0, rtol=0)
    torch.testing.assert_close(nearest.payload["base_tracking_cost"], neighbor.payload["base_tracking_cost"], atol=0, rtol=0)
    if gap > .1:
        assert float(nearest.consensus[..., 3:].abs().sum()) == 0.
        object_x = nearest.payload["qpos"][..., -7]
        torch.testing.assert_close(object_x, object_x[:1].expand_as(object_x), atol=1e-7, rtol=0)
        assert not bool(nearest.payload["contact_active"].any())
    else:
        assert float(nearest.consensus[..., 3:].abs().sum()) > 0.
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(task.env.data_wp, name).numpy(), value)


def test_action_gradient_uses_contacts_discovered_during_robot_rollout(device):
    task = _task(device, gap=.002)
    adapter = _adapter(task)
    selection = _selection(adapter)
    action = torch.full((task.knots*task.config.nu,), .5, device=device, requires_grad=True)
    result = adapter.robot_evaluate(action, selection)
    assert bool(torch.isfinite(result.cost))
    final_x = result.payload["qpos"][-1, -7]
    gradient = torch.autograd.grad(final_x, action, retain_graph=True)[0]
    assert bool(torch.isfinite(gradient).all())
    assert gradient.abs().sum() > 1e-7
    assert result.consensus[..., 3:].abs().sum() > 0
    counts = result.payload["real_contact_count"]
    assert int(counts[0]) == 0 and int(counts[1:].max()) > 0
    cost_gradient = torch.autograd.grad(result.cost, action)[0]
    assert bool(torch.isfinite(cost_gradient).all())
    assert cost_gradient.abs().sum() > 1e-7


def test_upper_force_changes_object_motion_and_tracking_with_torch_only(device, monkeypatch):
    adapter = _adapter(_task(device), force_regularization=0., force_smoothness=0.)
    indices = torch.zeros((2, 2), dtype=torch.long, device=device)
    parameters = torch.zeros((2, adapter.knots*adapter.hands*3), device=device)
    parameters[1, 0::3] = .1
    with monkeypatch.context() as patched:
        _forbid_engine(patched)
        result = adapter.object_evaluate(indices, parameters)
        assert bool(torch.isfinite(result.cost).all())
        assert float(result.cost[1]) < float(result.cost[0])
        assert result.consensus[1, -1, 0, 0] > result.consensus[0, -1, 0, 0]
        variable = parameters[1:].clone().requires_grad_(True)
        differentiable = adapter.object_evaluate(indices[1:], variable)
        gradient = torch.autograd.grad(differentiable.cost.sum(), variable)[0]
        assert bool(torch.isfinite(gradient).all())
        assert gradient.abs().sum() > 1e-7


def test_adapter_executes_contact_only_cem_force_solve_and_real_ipopt(device):
    pytest.importorskip("cyipopt")
    task = _task(device, horizon=3)
    adapter = _adapter(task, action_trust_radius=.5)
    problem = adapter.problem(torch.zeros(task.knots*task.config.nu, device=device))
    settings = SCSPSettings(admm_iterations=1, candidate_population=3,
                            candidate_iterations=1, force_iterations=2,
                            force_backtracks=2, ipopt_max_iter=3)
    result = solve(problem, settings, seed=12)
    assert np.isfinite(result.cost)
    assert bool(torch.isfinite(result.parameters).all())
    assert result.history[0]["ipopt_status"] in (0, 1, 6, -1, -4)
    assert result.sampled >= 1
    assert result.evaluated > result.sampled
    assert result.selection.indices.shape == (2,)
    assert result.consensus.shape == problem.initial_consensus.shape


def test_far_hands_disable_position_consensus_and_preserve_zero_force_agreement(device):
    task = _task(device, gap=.3)
    adapter = _adapter(task)
    action = torch.zeros(task.knots*task.config.nu, device=device)
    problem = adapter.problem(action)
    robot, mask = problem.prepare_iteration(action)
    assert mask.shape == (task.config.horizon_steps, 2, 6)
    assert not bool(mask[..., :3].any())
    assert bool(mask[..., 3:].all())
    target = _selection(adapter).consensus
    assert (target[..., :3]-robot[..., :3]).abs().sum() > 1.
    torch.testing.assert_close(target[..., 3:], torch.zeros_like(target[..., 3:]), atol=0, rtol=0)
    torch.testing.assert_close(robot[..., 3:], torch.zeros_like(robot[..., 3:]), atol=0, rtol=0)
    # A distant target cannot impose an equality during the approach phase;
    # the still-active force agreement accepts exactly zero contact force.
    torch.testing.assert_close((target-robot)*mask, torch.zeros_like(target), atol=0, rtol=0)


def test_position_gate_activates_only_at_later_real_contacts_and_stays_fixed(device):
    task = _task(device, gap=.002)
    adapter = _adapter(task)
    zero = torch.zeros(task.knots*task.config.nu, device=device)
    action = torch.full_like(zero, .5)
    problem = adapter.problem(zero)
    _, approach_mask = problem.prepare_iteration(zero)
    assert not bool(approach_mask[..., :3].any())
    saved_mask = approach_mask.clone()
    moving = adapter.robot_evaluate(action, _selection(adapter))
    physical = moving.payload["contact_active"][:, :2]
    assert not bool(physical[0].any()) and bool(physical[1:].any())
    torch.testing.assert_close(approach_mask, saved_mask, atol=0, rtol=0)
    _, next_mask = problem.prepare_iteration(action)
    torch.testing.assert_close(next_mask[..., :3], physical[..., None].expand(-1, -1, 3))
    assert not bool(next_mask[0, :, :3].any())
    assert bool(next_mask[..., 3:].all())


def test_selected_object_target_does_not_toggle_actual_contact_gate(device):
    task = _task(device)
    adapter = _adapter(task)
    action = torch.full((task.knots*task.config.nu,), .5, device=device)
    problem = adapter.problem(action)
    robot_before, mask_before = problem.prepare_iteration(action)
    adapter.robot_evaluate(action, _selection(adapter, index=10))
    robot_after, mask_after = problem.prepare_iteration(action)
    torch.testing.assert_close(robot_before, robot_after, atol=0, rtol=0)
    torch.testing.assert_close(mask_before, mask_after, atol=0, rtol=0)
    assert bool(mask_after[..., :3].any())


def test_operation_schedule_disables_position_consensus_at_physical_contacts(device):
    task = _task(device)
    operation = torch.ones((task.config.horizon_steps, 2), dtype=torch.bool, device=device)
    operation[:, 0] = False
    adapter = _adapter(task, operation_mask=operation)
    action = torch.full((task.knots*task.config.nu,), .5, device=device)
    problem = adapter.problem(action)
    _, mask = problem.prepare_iteration(action)
    result = adapter.robot_evaluate(action, _selection(adapter))
    actual = result.payload["contact_active"][:, :2]
    assert bool(actual[:, 0].any()) and bool(actual[:, 1].any())
    expected = (actual & operation)[..., None].expand(-1, -1, 3)
    torch.testing.assert_close(mask[..., :3], expected)
    assert not bool(mask[:, 0, :3].any())
    assert bool(mask[:, 1, :3].any()) and bool(mask[..., 3:].all())


def test_no_operation_disables_approach_cost_and_contact_position_consensus(device):
    task = _task(device)
    operation = torch.zeros(task.config.horizon_steps, dtype=torch.bool, device=device)
    adapter = _adapter(task, operation_mask=operation)
    action = torch.full((task.knots*task.config.nu,), .5, device=device)
    problem = adapter.problem(action)
    _, mask = problem.prepare_iteration(action)
    nearest = adapter.robot_evaluate(action, _selection(adapter))
    neighbor = adapter.robot_evaluate(action, _selection(adapter, index=10))
    assert bool(nearest.payload["contact_active"][:, :2].any())
    assert not bool(mask[..., :3].any()) and bool(mask[..., 3:].all())
    assert float(nearest.payload["approach_cost"]) == 0.
    # Without an operation request, changing the desired object point cannot
    # introduce either a position equality or a target-dependent approach cost.
    torch.testing.assert_close(nearest.cost, neighbor.cost, atol=0, rtol=0)


def test_operation_intervals_use_global_task_time_after_window_shift(device):
    task = _task(device)
    task.first_step = 5
    task.refs = tuple(torch.cat((value[:1].expand(5, *value.shape[1:]), value), dim=0)
                      for value in task.refs)
    # This window starts at t=.5s. The interval therefore enables stages at
    # .6s and .7s; treating it as time relative to the window would enable none.
    adapter = _adapter(task, operation_intervals=[[.59, .71]])
    schedule = torch.tensor([False, True, True, False], device=device)
    torch.testing.assert_close(adapter.operation_mask, schedule[:, None].expand(-1, 2))
    action = torch.full((task.knots*task.config.nu,), .5, device=device)
    problem = adapter.problem(action)
    _, mask = problem.prepare_iteration(action)
    physical = adapter.robot_evaluate(action, _selection(adapter)).payload["contact_active"][:, :2]
    expected = (physical & schedule[:, None])[..., None].expand(-1, -1, 3)
    torch.testing.assert_close(mask[..., :3], expected)
    assert bool(mask[..., 3:].all())
