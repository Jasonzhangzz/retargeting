"""GPU node isolation, replay without host access, and independent certificates."""
import numpy as np
import pytest
import torch
import warp as wp
from pathlib import Path
from types import SimpleNamespace
from scipy.spatial.transform import Rotation

from retarget.dexonomy import ROOT
from retarget.grasp import build_scene, StaticHand
from retarget.grasp_primitives import primitive_mesh
from retarget.grasp_initialization import from_dexonomy_result
from retarget.grasp_runtime import prepare_experiment, GraspRuntime
from retarget.grasp_audit import GraspAudit
from retarget.grasp_device import GraspDeviceNode, CudaNodeGraph
from retarget.grasp_cem import CardCemNode
from retarget.grasp_device_closure import closure_tensors, export_closure
from retarget.grasp_closure import force_closure

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def make_runtime(tmp_path, radius, *, rounds=1, iterations=1, search='adamw', primitive='sphere', hand_model='wuji1'):
    from retarget.grasp_hand import query_path
    model, surface, _, _ = build_scene(primitive_mesh(tmp_path/'object.obj', primitive, radius, 2*radius), hand_model=hand_model)
    bodies, local, normals = [], [], []
    for finger in (1, 2, 3):
        name = f'right_finger{finger}_link4'
        with np.load(query_path(model, name)) as asset:
            bodies.append(model.body(name).id)
            local.append(asset['local_pos'][10])
            normals.append(asset['local_normal'][10])
    hand = StaticHand(model, surface, np.array(bodies), np.array(local), np.array(normals))
    q = np.r_[np.zeros(6), np.full(20, .3)]
    hand.set(q)
    q[:3] -= hand.points(hand.bodies, hand.local).mean(0)
    hand.set(q)
    auditor = GraspAudit(surface, 'cuda:0')
    anchor, normal, _ = auditor.query(torch.tensor(hand.points(hand.bodies, hand.local), device='cuda', dtype=torch.float32))
    guess = from_dexonomy_result(dict(obj_pose=np.array([0., 0., 0., 1., 0., 0., 0.]),
        grasp_pose=np.r_[q[:3], Rotation.from_euler('XYZ', q[3:6]).as_quat(scalar_first=True)],
        grasp_qpos=q[6:], obj_cp_w=anchor.cpu().numpy(), obj_cn_w=-normal.cpu().numpy()))
    args = SimpleNamespace(solver_config=ROOT/'data/example/forceaware.yaml', object_mass=.2,
        friction=.6, iterations=iterations, multistarts=2, seed=0, init_device='cuda:0', hand_solver='adamw',
        contact_rounds=rounds, contact_radius_mm=5., contact_motion_tolerance_mm=.1,
        contact_tolerance_mm=2., penetration_tolerance_mm=2., cone_edges=16, load_conditioning=False,
        grasp_optimizer=search, cem_population=8, cem_iterations=2)
    runtime = GraspRuntime(prepare_experiment(tmp_path/'prepared', hand, guess, args), args)
    runtime.audit_context = auditor, hand, args
    return runtime


@pytest.mark.parametrize('rounds,iterations', [(1, 1), (2, 2)])
def test_device_graph_matches_shared_adamw_from_identical_seed(tmp_path, rounds, iterations):
    runtime = make_runtime(tmp_path, .025, rounds=rounds, iterations=iterations)
    # The CPU Trigrasp initializer is explicitly excluded from the GPU path.
    runtime.optimizer._trigrasp_seed = lambda *a: None
    expected, expected_snapshot, _ = runtime.solve()
    node = GraspDeviceNode(runtime)
    # Poison the warm-up output: successful replay must initialize its own state.
    # Order the poison before replay on the node's stream; an asynchronous
    # default-stream write can otherwise race with the reset being tested.
    with wp.ScopedStream(node.stream):
        node.backend.control.raw.fill_(123.)
    node.launch()
    wp.synchronize_stream(node.stream)
    actual, snapshot, _ = node.export()
    np.testing.assert_allclose(actual.raw, expected.raw, atol=2.e-6, rtol=2.e-6)
    np.testing.assert_allclose(actual.next_qpos, expected.next_qpos, atol=2.e-6, rtol=2.e-6)
    for key in ('z', 'x', 'dual', 'template'):
        np.testing.assert_allclose(snapshot[key], expected_snapshot[key], atol=2.e-6, rtol=2.e-6)
    assert actual.seconds > 1.e-4  # Guard against accidentally capturing an empty graph.


@pytest.mark.parametrize('search', ['adamw', 'cem'])
@pytest.mark.parametrize('hand_model', ['wuji1', 'wuji2'])
def test_independent_graphs_replay_without_host_reads_or_iteration_sync(tmp_path, monkeypatch, search, hand_model):
    runtimes = [make_runtime(tmp_path/f'n{i}', r, search=search, hand_model=hand_model) for i, r in enumerate((.02, .03))]
    for runtime in runtimes:
        runtime.args.cem_iterations = 2
    nodes = [(CardCemNode if search == 'cem' else GraspDeviceNode)(runtime) for runtime in runtimes]
    assert nodes[0].state.x.ptr != nodes[1].state.x.ptr
    assert nodes[0].stream.cuda_stream != nodes[1].stream.cuda_stream
    assert nodes[0].backend.compiled.base is not nodes[1].backend.compiled.base
    dag = CudaNodeGraph(nodes)
    for node in nodes:
        node.selected.fill_(-1)
        node.state.x.fill_(123.)

    def forbidden(*a, **kw):
        raise AssertionError('Host access/synchronization during GPU node replay')

    with monkeypatch.context() as patch:
        patch.setattr(wp.array, 'numpy', forbidden)
        patch.setattr(torch.Tensor, 'cpu', forbidden)
        patch.setattr(torch.Tensor, 'item', forbidden)
        patch.setattr(torch.Tensor, '__bool__', forbidden)
        patch.setattr(torch.Tensor, '__float__', forbidden)
        patch.setattr(wp, 'synchronize', forbidden)
        patch.setattr(wp, 'synchronize_device', forbidden)
        patch.setattr(wp, 'synchronize_stream', forbidden)
        dag.launch()
    wp.synchronize_stream(dag.stream)
    for node in nodes:
        plan, _, _ = node.export()
        assert np.isfinite(plan.next_qpos).all()
        report, arrays = node.runtime.device_audit_result
        active = arrays['realized_contact_mask']
        reference = force_closure(arrays['realized_object_points'][active],
            arrays['realized_object_normals'][active], friction=.6, radius=node.audit.auditor.radius)
        assert report['force_closure']['force_closure'] == reference['force_closure']
    dag.close()


@pytest.mark.parametrize('case', ['closed', 'rank_five', 'same_side', 'empty'])
def test_masked_device_closure_matches_independent_highs(case):
    points = np.array([[-.05, 0., 0.], [.05, 0., 0.], [-.05, .001, 0.], [0., 0., .03]])
    normals = np.array([[-1., 0., 0.], [1., 0., 0.], [-1., 0., 0.], [0., 0., 1.]])
    mask = np.array([True, True, True, False])
    if case == 'rank_five':
        points[2, 1] = 0.
    elif case == 'same_side':
        normals[:] = [0., 0., -1.]
    elif case == 'empty':
        mask[:] = False
    tensors = closure_tensors(torch.tensor(points, device='cuda'), torch.tensor(normals, device='cuda'),
        torch.tensor(mask, device='cuda'), friction=.6, radius=.1)
    actual = export_closure(tensors, torch.tensor(mask, device='cuda'), 16)
    expected = force_closure(points[mask], normals[mask], radius=.1)
    assert actual['force_closure'] == expected['force_closure']
    assert actual['rank'] == expected['rank']
    assert actual['contacts'] == int(mask.sum())
    if actual['force_closure']:
        assert actual['equilibrium_residual'] < 1.e-7
        assert actual['positive_margin'] > 1.e-7


@pytest.mark.parametrize('name', ['1_Large_Diameter', '6_Prismatic_4_Finger', '12_Precision_Disk'])
def test_gpu_bone_seed_fits_without_cpu_optimizer(tmp_path, monkeypatch, name):
    import retarget.mano_noise as mano
    from retarget.dexonomy import load_template, anatomical_joints
    from retarget.grasp_seed import bone_seed_cuda
    checkout = ROOT/'third_party/Dexonomy'
    if not (checkout/f'assets/hand/shadow/raw_anno/{name}.yaml').is_file():
        pytest.skip('Original annotations required')
    model, _, _, _ = build_scene(primitive_mesh(tmp_path/'seed.obj', 'cylinder', .02, .08))
    shadow, posed, _, _ = load_template(checkout, name)
    human = anatomical_joints(shadow, posed)[None]
    _, reference = mano.retarget_from_mano(model, human)
    def forbidden(*a, **kw):
        raise AssertionError('CPU least-squares must not run in GPU bone seeding')
    monkeypatch.setattr(mano, 'least_squares', forbidden)
    q, diagnostic = bone_seed_cuda(model, human, device='cuda:0')
    assert q.shape == (1, 26) and np.isfinite(q).all()
    ids = np.array([np.flatnonzero(model.jnt_qposadr == i)[0] for i in range(26)])
    assert np.all(q[0] >= model.jnt_range[ids, 0]-1.e-8)
    assert np.all(q[0] <= model.jnt_range[ids, 1]+1.e-8)
    assert diagnostic['chain_direction_rms'] <= reference['chain_direction_rms']+1.e-3


def test_scene_snapshots_preserve_each_nodes_object_mesh(tmp_path):
    import mujoco
    from retarget.contact_exploration import ObjectMeshSurface
    runtimes = [make_runtime(tmp_path/f'n{i}', radius) for i, radius in enumerate((.02, .03))]
    for i, runtime in enumerate(runtimes):
        # Both models have already been loaded. A late mj_saveLastXML would
        # incorrectly reuse the last scene's mesh file for the first node.
        assert str(tmp_path/f'n{i}/prepared/object.obj') in runtime.scene_xml
        loaded = mujoco.MjModel.from_xml_string(runtime.scene_xml)
        actual = ObjectMeshSurface.from_model(loaded, loaded.body('grasp_object').id)
        expected = ObjectMeshSurface.from_model(runtime.model, runtime.index.object_body)
        np.testing.assert_allclose(np.ptp(actual.triangles.reshape(-1, 3), axis=0),
            np.ptp(expected.triangles.reshape(-1, 3), axis=0), atol=1.e-8, rtol=1.e-6)


@pytest.mark.parametrize('hand_model', ['wuji1', 'wuji2'])
def test_card_cem_replay_is_deterministic_without_rollout_or_host_reads(tmp_path, monkeypatch, hand_model):
    import forceaware.optimizer as shared
    runtime = make_runtime(tmp_path, .025, search='cem', hand_model=hand_model)
    def no_dynamics(*a, **kw):
        raise AssertionError('CARD invoked dynamics or an adjoint')
    monkeypatch.setattr(shared, 'step_forward', no_dynamics)
    monkeypatch.setattr(shared, 'step_backward', no_dynamics)
    node = CardCemNode(runtime, capture_standalone=False)
    assert node.graph is None and node.audit.graph is None
    assert runtime.optimizer.backend.steps == ()
    dag = CudaNodeGraph([node])
    def forbidden(*a, **kw):
        raise AssertionError('CARD replay touched the host')
    with monkeypatch.context() as patch:
        patch.setattr(wp.array, 'numpy', forbidden)
        patch.setattr(torch.Tensor, 'cpu', forbidden)
        patch.setattr(torch.Tensor, 'item', forbidden)
        patch.setattr(torch.Tensor, '__bool__', forbidden)
        patch.setattr(wp, 'synchronize_stream', forbidden)
        dag.launch()
    wp.synchronize_stream(dag.stream)
    first = node.state.qpos.numpy().copy()
    plan, snapshot, output = node.export()
    diag = runtime.optimizer.last_solver_diag
    history = [x for x in diag['cem_best_loss_history'] if x is not None]
    assert np.all(np.diff(history) <= 1.e-3)
    assert diag['solver'] == 'card_annealed_cem'
    assert diag['gradient_source'] == 'none' and diag['dynamics_rollouts'] == 0
    assert not diag['contact_admm_enabled']
    assert diag['output_time_seconds'] == diag['prediction_horizon_seconds'] == 0.
    np.testing.assert_array_equal(output, first[0])
    np.testing.assert_array_equal(plan.next_qpos, plan.initial_qpos)
    assert not snapshot['admm_active']
    assert np.isfinite(plan.next_qpos).all()
    node.state.x.fill_(123.); node.state.best.fill_(456.)
    dag.launch(); wp.synchronize_stream(dag.stream)
    np.testing.assert_allclose(node.state.qpos.numpy(), first, atol=2.e-6)
    dag.close()


def test_real_contact_dynamics_adjoint_to_finger_command(tmp_path):
    """Object motion has no direct finger-command term: derivative is dynamics."""
    from forceaware.optimizer import step_forward, step_backward, BackwardCall, _zero_cotangent
    runtime = make_runtime(tmp_path, .025)
    node = GraspDeviceNode(runtime, capture_standalone=False)
    b = node.backend
    node._evaluate(False)
    wp.synchronize_device(b.device)
    step = b.steps[0]
    assert b.loss.immediate[0].job.state.contact_active.numpy()[0].any()
    _zero_cotangent(step.cotangent)
    # Seed d(sum object translation)/d(next qpos), not an attraction cost.
    cotangent = np.zeros(step.cotangent.qpos.shape, np.float32)
    obj = runtime.index.object_qpos
    cotangent[0, obj:obj+3] = [1., .3, -.2]
    step.cotangent.qpos.assign(cotangent)
    derivative = step_backward(BackwardCall(b.compiled, step.inputs, step.workspace,
                                            step.recorded, step.cotangent))
    analytic = derivative.ctrl.numpy()[0]
    controls = np.asarray(runtime.index.hand_ctrl)[6:]
    column = int(controls[np.argmax(np.abs(analytic[controls]))])
    assert abs(analytic[column]) > 1.e-6
    nominal = step.inputs.dynamics.ctrl.numpy().copy()
    values = []
    epsilon = 1.e-3
    for sign in (1., -1.):
        trial = nominal.copy()
        trial[0, column] += sign*epsilon
        step.inputs.dynamics.ctrl.assign(trial)
        step_forward(b.compiled, step.inputs, step.workspace)
        q = step.recorded.result.qpos.numpy()[0, obj:obj+3]
        values.append(float(q@np.array([1., .3, -.2])))
    finite_difference = (values[0]-values[1])/(2*epsilon)
    np.testing.assert_allclose(analytic[column], finite_difference, rtol=.05, atol=3.e-5)


@pytest.mark.parametrize('primitive', ['sphere', 'cylinder', 'cuboid'])
def test_card_static_export_matches_scored_pose_and_independent_fk(tmp_path, primitive):
    import mujoco
    runtime = make_runtime(tmp_path, .025, search='cem', primitive=primitive)
    node = CardCemNode(runtime, capture_standalone=False)
    dag = CudaNodeGraph([node]); dag.launch(); wp.synchronize_stream(dag.stream)
    plan, snapshot, qpos = node.export()
    measured = mujoco.MjData(runtime.model)
    measured.qpos[:] = qpos; mujoco.mj_kinematics(runtime.model, measured)
    ids = node.annotation_bodies
    local = snapshot['hand_local_points']
    world = measured.xpos[ids]+np.einsum('nij,nj->ni', measured.xmat[ids].reshape(-1, 3, 3), local)
    obj = runtime.index.object_body
    expected = (world-measured.xpos[obj])@measured.xmat[obj].reshape(3, 3)
    report, arrays = runtime.device_audit_result
    np.testing.assert_allclose(expected, arrays['hand_points'], atol=2.e-6)
    np.testing.assert_array_equal(qpos, node.state.qpos.numpy()[0])
    np.testing.assert_array_equal(qpos, runtime.predicted_terminal_qpos)
    active = arrays['realized_contact_mask']
    independent = force_closure(arrays['realized_object_points'][active],
        arrays['realized_object_normals'][active], friction=.6, radius=node.audit.auditor.radius)
    assert report['force_closure']['force_closure'] == independent['force_closure']
    assert plan.loss == pytest.approx(sum(runtime.optimizer.last_solver_diag['cost_groups'].values()))
    if plan.diag['optimizer_success']:
        assert runtime.optimizer.last_solver_diag['minimum_interfinger_gap_mm'] >= .5
        assert report['max_self_penetration_mm'] <= 1.e-3
    else:
        assert not report['feasible']
        np.testing.assert_allclose(qpos[:3], runtime.qpos[:3], atol=1.e-6)
    dag.close()


@pytest.mark.parametrize('hand_model', ['wuji1', 'wuji2'])
def test_card_cli_completes_dispatch_and_writes_result_files(tmp_path, monkeypatch, hand_model):
    """CLI exports both solved cases and IK-only failures without crashing."""
    import json
    from run_grasp import main
    import retarget.grasp_results as registry
    monkeypatch.setattr(registry, 'RESULTS_ROOT', tmp_path/'registry')
    checkout = ROOT/'third_party/Dexonomy'
    if not (checkout/'assets/hand/shadow/raw_anno/2_Small_Diameter.yaml').is_file():
        pytest.skip('Original Dexonomy annotation required')
    output = main(['--archive-results', '--no-explicit-pose', '--hand-model', hand_model, '--primitives', '--templates', '2', '--radius-scales', '1',
        '--gpu-nodes', '1', '--no-viewer', '--no-render', '--output', str(tmp_path),
        '--name', 'card_cli', '--cem-population', '8', '--cem-iterations', '2',
        '--init-points', '2', '--init-inplane', '2', '--init-iterations', '2',
        '--no-template-fit', '--template-refinements', '0', '--no-load-conditioning'])
    dispatch = json.loads((output/'gpu_dispatch.json').read_text())
    rows = json.loads((output/'summary.json').read_text())
    assert len(rows) == 1 and (output/'summary.csv').is_file()
    assert rows[0]['hand_model'] == hand_model
    case = Path(rows[0]['case'])
    report = json.loads((case/'result.json').read_text())
    failed_initialization = report['selected']['termination'] == 'initialization_failed'
    if failed_initialization:
        assert dispatch == []
        assert not rows[0]['optimizer_success'] and not rows[0]['feasible']
    else:
        assert len(dispatch) == 1 and dispatch[0]['worlds'] == 8
        assert report['selected']['solver_diagnostics']['evaluation_worlds'] == 8
    assert report['selected']['output_time_seconds'] == 0.
    for name in ('grasp.npz', 'contacts.npz', 'scene.xml'):
        assert (case/name).is_file()
    with np.load(case/'grasp.npz') as saved:
        np.testing.assert_array_equal(saved['stage_valid'], [1, int(not failed_initialization)])
        np.testing.assert_array_equal(saved['qpos'], saved['predicted_terminal_qpos'])
        assert np.isfinite(saved['qpos']).all()
