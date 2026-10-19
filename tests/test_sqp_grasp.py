"""Regression checks for missing thumb targets, zero tip gradients and servo units."""
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
import pytest

from forceaware.config import load_retarget_config
from forceaware.targets import (ContactGuidance, SequenceData, build_index,
                                initial_state, make_window)
from retarget.contact_plan import optimize_contact_plan
from retarget.fa.trigrasp_comfree_sqp import TrigraspComFreeSQP, servo_commands
from retarget.contact_exploration import LocalContactExplorer, ObjectMeshSurface


@pytest.fixture(scope='module')
def problem():
    root = Path(__file__).resolve().parents[1]/'data/example'
    cfg = load_retarget_config(root/'sequence', root/'forceaware.yaml',
                               contactaware_dir=root/'contactaware')
    model = mujoco.MjModel.from_xml_path(str(root/'sequence/scene/wuji_right/sugar_box_mesh.xml'))
    index = build_index(model)
    with np.load(root/'contactaware/contact_guidance.npz') as z:
        contacts = ContactGuidance(z['contact_mask'], z['contact_weight'], z['contact_pos_obj'],
                                   np.zeros_like(z['contact_mask'], int),
                                   z['hand_query_body_ids'], z['hand_query_local_pos'])
    seq = SequenceData(np.load(root/'contactaware/wuji_right_qpos.npy'),
                        np.load(root/'contactaware/object_pose_7.npy'), contacts)
    q, v = initial_state(model, index, seq, start_frame=23, ref_dt=cfg.simulator.time_grid.ref_dt)
    v[:] = 0.
    targets = make_window(seq, executed_steps=0, horizon=5, time_grid=cfg.simulator.time_grid,
                          start_frame=23, end_frame=23, hold_frame=-1)
    backend = SimpleNamespace(cpu_model=model, index=index, cfg=cfg, guidance_count=19,
                               steps=range(30))
    controller = TrigraspComFreeSQP(backend, build_solver=False)
    solver = LocalContactExplorer(ObjectMeshSurface.from_model(model, index.object_body))
    plan = optimize_contact_plan(model, index, solver, q, targets,
        friction=np.full((31, 19, 2), .6), center=model.body_ipos[index.object_body], object_radius=.1)
    params, geo = controller.parameters(q, v, targets, plan.points[0])
    state = np.r_[q[index.object_qpos:index.object_qpos+7], q[index.hand_qpos],
                   v[controller.object_dofs], v[controller.hand_dofs], q[index.hand_qpos]].astype(float)
    return model, index, controller, q, v, targets, plan, params, geo, state


def test_all_active_slots_and_thumb_reach_sqp(problem):
    model, _, controller, _, _, targets, plan, _, geo, _ = problem
    assert controller.n_contacts == 19
    active = targets.contact_mask[0] > 0
    np.testing.assert_array_equal(geo['anchors'][active], plan.points[0, active])
    assert sum(active) == 7
    names = [model.body(int(b)).name for b in geo['body'][active]]
    assert any('finger1_' in name for name in names)
    assert len({name.split('_')[1] for name in names}) == 5


def test_tip_error_has_actual_hand_jacobian_and_thumb_gradient(problem):
    _, _, controller, _, _, _, _, params, geo, state = problem
    derivative = np.asarray(controller.tip_jacobian_fn(state, params))[:, 7:controller.nq]
    np.testing.assert_allclose(derivative, geo['tip_jac'], atol=1e-12)
    active_ids = np.flatnonzero(geo['mask'] > 0)
    assert np.linalg.norm(derivative[3*active_ids[-1]:3*active_ids[-1]+3, 6:10]) > .01
    eps = 1e-6
    fd = np.empty_like(derivative)
    for j in range(controller.nu):
        plus, minus = state.copy(), state.copy()
        plus[7+j] += eps; minus[7+j] -= eps
        fd[:, j] = (np.asarray(controller.tip_error_fn(plus, params)).ravel()
                     -np.asarray(controller.tip_error_fn(minus, params)).ravel())/(2*eps)
    np.testing.assert_allclose(derivative, fd, atol=1e-8)


def test_mujoco_jacobian_is_evaluated_at_world_query_point(problem):
    model, index, controller, q, _, _, _, _, geo, _ = problem
    for slot in [0, 5, 17]:
        body, local = int(geo['body'][slot]), geo['local'][slot]
        fd = np.empty((3, controller.nu))
        eps = 1e-6
        for j, address in enumerate(index.hand_qpos):
            samples = []
            for sign in [1., -1.]:
                data = mujoco.MjData(model); data.qpos[:] = q
                data.qpos[address] += sign*eps; mujoco.mj_kinematics(model, data)
                samples.append(data.xpos[body]+data.xmat[body].reshape(3, 3)@local)
            fd[:, j] = (samples[0]-samples[1])/(2*eps)
        np.testing.assert_allclose(geo['tip_jac'][3*slot:3*slot+3], fd, atol=1e-7)
    assert np.linalg.norm(geo['jac'][:, :6]) > 0


def test_contact_gradient_can_reduce_real_fingertip_alignment_error(problem):
    model, index, controller, q, _, _, _, params, geo, state = problem
    errors = np.asarray(controller.tip_error_fn(state, params)).ravel()
    weights = np.repeat(geo['mask'], 3)
    gradient = geo['tip_jac'].T@(weights*errors)
    gradient[:6] = 0.  # Demonstrate a finger correction, not a palm translation.
    assert np.linalg.norm(gradient[6:]) > 1e-6
    delta = -1e-4*gradient/max(np.max(np.abs(gradient)), 1e-12)
    data = mujoco.MjData(model); data.qpos[:] = q
    data.qpos[index.hand_qpos] += delta
    mujoco.mj_kinematics(model, data)
    points = data.xpos[geo['body']]+np.einsum('nij,nj->ni',
                data.xmat[geo['body']].reshape(-1, 3, 3), geo['local'])
    obj = data.body(index.object_body)
    target = obj.xpos+geo['anchors']@obj.xmat.reshape(3, 3).T
    assert np.sum(weights*(points-target).ravel()**2) < np.sum(weights*errors**2)


def test_predicted_servo_commands_are_exactly_executor_commands(problem):
    _, index, controller, q, _, _, _, params, _, state = problem
    previous = q[index.hand_qpos].copy()
    previous[6:] += .01  # preload must survive the next solve
    state = state.copy(); state[-controller.nu:] = previous
    increments = np.full((3, controller.nu), .001)
    commands = servo_commands(previous, increments, controller.lower, controller.upper)
    for k, command in enumerate(commands):
        state = np.asarray(controller.prediction_fn(state, increments[k], params)).ravel()
        np.testing.assert_allclose(state[-controller.nu:], command, atol=1e-12)
    assert not np.allclose(commands[0], previous+.1+increments[0])


def test_shared_point_update_is_deterministic_on_measured_state(problem):
    model, index, _, q, _, targets, first, _, _, _ = problem
    solver = LocalContactExplorer(ObjectMeshSurface.from_model(model, index.object_body))
    second = optimize_contact_plan(model, index, solver, q, targets,
        friction=np.full((31, 19, 2), .6), center=model.body_ipos[index.object_body], object_radius=.1)
    np.testing.assert_array_equal(first.points, second.points)
    data = mujoco.MjData(model); data.qpos[:] = q; mujoco.mj_kinematics(model, data)
    bodies, local = targets.query_body[0], targets.query_local[0]
    hand = data.xpos[bodies]+np.einsum('nij,nj->ni', data.xmat[bodies].reshape(-1, 3, 3), local)
    expected = (hand-data.xpos[index.object_body])@data.xmat[index.object_body].reshape(3, 3)
    np.testing.assert_allclose(first.snapshot['x'][0], expected, atol=1e-8)
    assert np.max(np.linalg.norm(first.snapshot['c']-first.snapshot['template'], axis=-1)) <= .00500001


def test_sqp_contacts_equal_adamw_mesh_subproblem_for_equal_inputs(problem):
    model, index, _, q, _, targets, plan, _, _, _ = problem
    solver = LocalContactExplorer(ObjectMeshSurface.from_model(model, index.object_body))
    np.testing.assert_array_equal(plan.points[0], targets.contact_position_object[0].astype(np.float32))
    for stage in (1, 15, 30):
        k = max(stage-1, 0) if stage < 30 else 29
        template = targets.contact_position_object[k].astype(np.float32)
        # Call the exact solver used by AdamW _contact_block, including all slots.
        actual, _ = solver.solve(template, template, plan.snapshot['x'][max(stage-1, 0)],
            (targets.contact_mask[k]*targets.contact_weight[k]).astype(np.float32),
            np.full((19, 2), .6), center=model.body_ipos[index.object_body], object_radius=.1)
        np.testing.assert_array_equal(plan.points[stage], actual)


def test_sqp_tracks_object_linear_and_world_angular_velocity(problem):
    from scipy.spatial.transform import Rotation
    _,_,controller,_,_,_,_,params,_,state=problem
    target=np.array([.02,-.01,.03,.1,-.2,.05])
    params=params.copy();params[7:13]=target
    stationary=state.copy();stationary[controller.nq:controller.nq+6]=0.
    matching=stationary.copy();matching[controller.nq:controller.nq+3]=target[:3]
    rotation=Rotation.from_quat(state[[4,5,6,3]])
    matching[controller.nq+3:controller.nq+6]=rotation.inv().apply(target[3:])
    u=np.zeros(controller.nu)
    difference=float(controller.path_cost_fn(stationary,u,params)-controller.path_cost_fn(matching,u,params))
    weights=controller.cfg.loss
    expected=weights.reference_pose_scale*(weights.obj_vel**2*np.sum(target[:3]**2)
                                           +weights.obj_ang_vel**2*np.sum(target[3:]**2))
    np.testing.assert_allclose(difference,expected,atol=1e-8)
