"""Independent numerical checks of CARD kernels, runnable on Warp CPU."""
import math
from types import SimpleNamespace
import numpy as np
import pytest
import warp as wp
from scipy.spatial.transform import Rotation
from retarget import grasp_cem as card


def array(value, dtype=float):
    return wp.array(np.asarray(value), dtype=dtype, device='cpu')


def state(n=4, elites=2, seed=None):
    s = card.CemState()
    seed = np.zeros(26, np.float32) if seed is None else np.asarray(seed, np.float32)
    s.population, s.iterations, s.elites = n, 16, elites
    s.random_seed, s.temperature, s.final_noise = 127, .1, .2
    s.root_rotation = wp.mat33(np.eye(3))
    s.seed, s.seed_qpos = array(seed), array(np.r_[seed, [0, 0, 0, 1, 0, 0, 0]])
    s.hand_ids = array(np.arange(26), int)
    s.low, s.high = array(seed-.25), array(seed+.25)
    s.initial_sigma = array(np.r_[np.full(3, .002), np.full(3, .016), np.full(20, .05)])
    s.mean, s.best = array(seed), array(seed)
    s.x, s.qpos = array(np.zeros((n, 26))), array(np.zeros((n, 33)))
    s.scores, s.rank = array(np.zeros(n)), array(np.zeros(n), int)
    s.selection_scores = s.scores
    s.restoration_history = array(np.zeros(16), int)
    s.best_score, s.history = array([np.inf]), array(np.zeros(16))
    return s


def launch(kernel, dim, *args):
    wp.launch(kernel, dim, inputs=list(args), device='cpu')


def test_servo_variables_preserve_the_geometric_sampling_and_restore_commands():
    plain, explicit = state(), state()
    explicit.seed = array(np.zeros(46))
    explicit.low, explicit.high = array(np.full(46, -.5)), array(np.full(46, .5))
    explicit.initial_sigma = array(np.r_[plain.initial_sigma.numpy(), np.full(20, 1/6)])
    explicit.mean, explicit.best = array(np.zeros(46)), array(np.zeros(46))
    explicit.x = array(np.zeros((explicit.population, 46)))
    for s in (plain, explicit):
        launch(card.sample, s.population, s, 0)
    np.testing.assert_array_equal(explicit.x.numpy()[:, :26], plain.x.numpy())
    np.testing.assert_array_equal(explicit.x.numpy()[:, 26:], 0.)
    values = explicit.best.numpy(); values[26:] = .1; explicit.best.assign(values)
    launch(card.scatter, explicit.population, explicit, 1)
    np.testing.assert_allclose(explicit.x.numpy()[:, 26:], .1)
    np.testing.assert_array_equal(explicit.qpos.numpy()[:, :26], 0.)


def test_force_priorities_are_optional_and_scale_invariant():
    np.testing.assert_array_equal(card.force_priorities(None, 3), [1, 1, 1])
    np.testing.assert_array_equal(card.force_priorities([0, 0, 0], 3), [1, 1, 1])
    np.testing.assert_allclose(card.force_priorities([0, 2, 4], 3), [1, 1.25, 1.5])
    np.testing.assert_allclose(card.force_priorities([0, 20, 40], 3), [1, 1.25, 1.5])
    for invalid in ([1, -1], [1, np.nan], [1], [[1, 2]]):
        with pytest.raises(ValueError):
            card.force_priorities(invalid, 2)


@pytest.mark.parametrize('rescale', [1., 1000., .001])
def test_elite_update_matches_standardized_softmax_and_incumbent(rescale):
    s = state()
    x = np.arange(104, dtype=np.float32).reshape(4, 26)/100
    score = np.array([4., 2., 1., 3.], np.float32)*rescale
    s.x.assign(x); s.scores.assign(score)
    launch(card.rank_candidates, 4, s)
    launch(card.update_distribution, 26, s)
    elite = np.array([2, 1])
    z = score[elite]
    weights = np.exp(-(z-z.min())/(z.std()*s.temperature)); weights /= weights.sum()
    np.testing.assert_allclose(s.mean.numpy(), weights@x[elite], rtol=2.e-6)
    np.testing.assert_allclose(s.best.numpy(), x[2])
    launch(card.commit_score, 1, s, 0)
    assert s.history.numpy()[0] == pytest.approx(rescale)
    # A worse generation must not overwrite the historical best.
    s.scores.assign(score+10*rescale)
    launch(card.update_distribution, 26, s); launch(card.commit_score, 1, s, 1)
    np.testing.assert_array_equal(s.best.numpy(), x[2])
    assert s.history.numpy()[1] == pytest.approx(rescale)


def test_equal_elites_invalid_elites_and_reset():
    s = state()
    x = np.arange(104, dtype=np.float32).reshape(4, 26)
    s.x.assign(x); s.scores.fill_(5.)
    launch(card.rank_candidates, 4, s); launch(card.update_distribution, 26, s)
    np.testing.assert_allclose(s.mean.numpy(), x[:2].mean(0))
    launch(card.initialize, 26, s)
    s.scores.fill_(float('inf'))
    launch(card.rank_candidates, 4, s); launch(card.update_distribution, 26, s)
    launch(card.commit_score, 1, s, 0)
    assert np.isinf(s.best_score.numpy()[0])
    np.testing.assert_array_equal(s.best.numpy(), s.seed.numpy())
    np.testing.assert_array_equal(s.mean.numpy(), s.seed.numpy())
    launch(card.scatter, 4, s, 1)
    np.testing.assert_array_equal(s.qpos.numpy(), np.tile(s.seed_qpos.numpy(), (4, 1)))


def test_sampling_reproducibility_caps_and_annealing():
    s = state(n=4096)
    launch(card.initialize, 26, s); launch(card.sample, s.population, s, 0)
    first = s.x.numpy().copy()
    np.testing.assert_array_equal(first[:2], np.zeros((2, 26)))
    launch(card.sample, s.population, s, 0)
    np.testing.assert_array_equal(s.x.numpy(), first)
    launch(card.sample, s.population, s, 15)
    ratio = s.x.numpy()[2:].std(0)/first[2:].std(0)
    np.testing.assert_allclose(ratio, .2**(15/16), rtol=.08)
    s.initial_sigma.fill_(10.)
    launch(card.sample, s.population, s, 1)
    x = s.x.numpy()
    assert np.max(np.linalg.norm(x[:, :3], axis=1)) <= .020001
    assert np.max(np.abs(x[:, 6:])) <= .250001


@pytest.mark.parametrize('angles', [[.5, -.2, 2.9], [0., np.pi/2, .4], [.2, -np.pi/2, 0.]])
def test_so3_scatter_preserves_object_and_mujoco_xyz_convention(angles):
    s = state(n=2)
    root = Rotation.from_euler('XYZ', angles)
    s.root_rotation = wp.mat33(root.as_matrix())
    x = np.zeros((2, 26), np.float32); x[:, :3] = [.03, -.01, .02]
    x[1, 3:6] = [.01, -.02, .005]
    s.x.assign(x)
    launch(card.scatter, 2, s, 0)
    q = s.qpos.numpy()
    for i in range(2):
        expected = root*Rotation.from_rotvec(x[i, 3:6])
        np.testing.assert_allclose(Rotation.from_euler('XYZ', q[i, 3:6]).as_matrix(), expected.as_matrix(), atol=3.e-6)
    np.testing.assert_array_equal(q[:, 26:], np.tile(s.seed_qpos.numpy()[26:], (2, 1)))


def cost_data():
    d = card.CostData()
    # One object and two links, with a nontrivial rigid transform on link 1.
    rotation = np.tile(np.eye(3), (1, 3, 1, 1))
    rotation[0, 1] = Rotation.from_rotvec([.1, -.05, .2]).as_matrix()
    position = np.array([[[0, 0, 0], [.003, -.001, .002], [.03, 0, 0]]])
    d.position, d.rotation = array(position, wp.vec3), array(rotation, wp.mat33)
    d.seed_position = array(np.zeros((3, 3)), wp.vec3)
    d.seed_rotation = array(np.tile(np.eye(3), (3, 1, 1)), wp.mat33)
    d.sample_body = array([1, 1], int)
    d.sample_pad = array([1, 1], int)
    d.sample_local = array([[0., 0., 0.], [.01, 0., 0.]], wp.vec3)
    d.sample_normal = array([[0, 1, 0], [0, 1, 0]], wp.vec3)
    d.group_body, d.group_start, d.group_end = array([1], int), array([0], int), array([2], int)
    d.preference = array([[.01, 0, 0]], wp.vec3)
    d.previous_local = array([[0, 0, 0]], wp.vec3)
    d.anchor = array([[.009, 0., .002]], wp.vec3)
    d.push = array([[0, 1, 0]], wp.vec3)
    d.rho, d.alpha = array([1.25]), array([.15])
    d.selected = array([[0]], int)
    d.parts = array(np.zeros((1, 4)))
    d.penetration, d.palm_gap = array([0.]), array([100.])
    d.self_gap, d.self_clearance = array([100.]), .5
    d.object_body = 0
    d.style_body = array([1]*15, int)
    d.style_apex = array([[.01, 0, 0]]*15, wp.vec3)
    d.style_normal = array([[0, 1, 0]]*15, wp.vec3)
    d.style_axis = array([[1, 0, 0]]*15, wp.vec3)
    d.capsule_body = array([1, 2], int)
    d.capsule_start = array([[0, 0, 0]]*2, wp.vec3)
    d.capsule_end = array([[0, .01, 0]]*2, wp.vec3)
    d.capsule_radius = array([.01, .01])
    d.pair_left, d.pair_right = array([0], int), array([1], int)
    return d


def test_contact_style_smooth_match_independent_numpy_formulas():
    d, s = cost_data(), state(n=1)
    launch(card.contact_cost, (1, 1), d)
    launch(card.style_cost, (1, 15), d)
    x = np.zeros((1, 26)); x[0, :3] = [.001, -.002, .003]; x[0, 3:6] = [.02, .01, -.03]; x[0, 6:] = .1
    s.x.assign(x); launch(card.smooth_cost, 1, d, s)
    rotation = d.rotation.numpy()[0, 1]; position = d.position.numpy()[0, 1]
    local = d.sample_local.numpy(); points = local@rotation.T+position
    anchor = d.anchor.numpy()[0]; pref = rotation@d.preference.numpy()[0]+position
    index = np.argmin(((points-anchor)**2).sum(-1)+.15*((points-pref)**2).sum(-1))
    assert d.selected.numpy()[0, 0] == index
    reach = ((points[index]-anchor)*1000)@((points[index]-anchor)*1000)
    cone = 200*(1-(rotation@np.array([0, 1, 0]))@np.array([0, 1, 0]))
    fix = np.linalg.norm((points[index]-position)*1000)**2
    expected_contact = 1.25**2*(40**2*(reach+cone)+2.5**2*fix)
    tip = (rotation@local[1]+position-local[1])*1000
    expected_style = 15*20**2*(1-rotation[0, 0])+5*(3**2*(tip@tip)+60**2*(1-rotation[1, 1]))
    expected_smooth = 125*np.sum((x[0, :3]*1000)**2)+100*np.sum(x[0, 3:6]**2)+.01*np.sum(x[0, 6:]**2)
    np.testing.assert_allclose(d.parts.numpy()[0, :3], [expected_contact, expected_style, expected_smooth], rtol=3.e-5)
    # A push normal pointing away must have higher contact energy.
    d.parts.zero_(); d.push.assign([[0, -1, 0]])
    launch(card.contact_cost, (1, 1), d)
    assert d.parts.numpy()[0, 0] > expected_contact


def mesh_box(extents):
    import trimesh
    mesh = trimesh.creation.box(extents=extents)
    vertices, faces = array(mesh.vertices, wp.vec3), array(mesh.faces.reshape(-1), int)
    return wp.Mesh(points=vertices, indices=faces), vertices, faces


def test_contact_selection_excludes_the_back_of_the_link():
    d = cost_data()
    # The geometrically closest sample is on the excluded dorsal face.
    d.sample_pad.assign([1, 0])
    d.anchor.assign([d.position.numpy()[0, 1]+d.rotation.numpy()[0, 1]@d.sample_local.numpy()[1]])
    launch(card.contact_cost, (1, 1), d)
    assert d.selected.numpy()[0, 0] == 0


def test_sdf_penetration_allowance_and_hard_floor_are_distinct():
    d, s = cost_data(), state(n=1)
    mesh, vertices, faces = mesh_box([.02, .02, .02])
    d.object_mesh = mesh.id
    d.position.assign(np.zeros((1, 3, 3)))
    d.rotation.assign(np.tile(np.eye(3), (1, 3, 1, 1)))
    # A point 2.5mm inside costs (2.5-2)^2 * 14^2, but passes the 3mm hard floor.
    d.sample_local = array([[.0075, 0, 0]], wp.vec3); d.sample_body = array([1], int)
    launch(card.clear_cost, 1, d); d.palm_gap.fill_(7.)
    launch(card.object_cost, (1, 1), d); launch(card.finish_cost, 1, d, s)
    assert d.penetration.numpy()[0] == pytest.approx(2.5, abs=1.e-4)
    assert s.scores.numpy()[0] == pytest.approx(14**2*.5**2, rel=1.e-4)
    d.sample_local.assign([[.006, 0, 0]])
    launch(card.clear_cost, 1, d); d.palm_gap.fill_(7.)
    launch(card.object_cost, (1, 1), d); launch(card.finish_cost, 1, d, s)
    assert np.isinf(s.scores.numpy()[0])
    # The independent final grasp audit can still require <=2mm.
    assert card.COST_PROFILE['object_allowance_mm'] == 2.
    assert card.COST_PROFILE['hard_penetration_mm'] == 3.


def test_dorsal_clearance_rejects_a_nonpenetrating_contact():
    d, s = cost_data(), state(n=1)
    mesh, vertices, faces = mesh_box([.02, .02, .02])
    d.object_mesh = mesh.id
    d.region_enabled = 1
    d.sample_forbidden = array([1, 1], int)
    d.forbidden_gap = array([float('inf')])
    d.rotation.assign(np.tile(np.eye(3), (1, 3, 1, 1)))
    d.position.assign([[[0., 0., 0.], [.0105, 0., 0.], [.04, 0., 0.]]])
    d.sample_local.assign([[0., 0., 0.], [0., 0., .001]])
    launch(card.clear_cost, 1, d)
    launch(card.object_cost, (1, 2), d)
    launch(card.finish_cost, 1, d, s)
    assert d.penetration.numpy()[0] == 0.
    assert d.forbidden_gap.numpy()[0] == pytest.approx(.5, abs=1.e-4)
    assert not np.isfinite(s.scores.numpy()[0])
    d.sample_forbidden.assign([0, 0])
    launch(card.clear_cost, 1, d)
    launch(card.object_cost, (1, 2), d)
    launch(card.finish_cost, 1, d, s)
    assert np.isfinite(s.scores.numpy()[0])


def test_interfinger_overlap_is_hard_rejected_even_when_contact_reward_wins():
    d, s = cost_data(), state(n=1)
    d.rotation.assign(np.tile(np.eye(3), (1, 3, 1, 1)))
    d.position.assign([[[0, 0, 0], [0, 0, 0], [.019, 0, 0]]])
    launch(card.capsule_cost, (1, 1), d); launch(card.finish_cost, 1, d, s)
    assert d.parts.numpy()[0, 3] == pytest.approx(1.5**2*7**2, rel=1.e-4)
    assert d.self_gap.numpy()[0] == pytest.approx(-1., abs=1.e-4)
    assert np.isinf(s.scores.numpy()[0])
    d.parts.zero_(); d.self_gap.fill_(100.)
    d.position.assign([[[0, 0, 0], [0, 0, 0], [.021, 0, 0]]])
    launch(card.capsule_cost, (1, 1), d); launch(card.finish_cost, 1, d, s)
    assert np.isfinite(s.scores.numpy()[0])
    d.palm_gap.fill_(5.99); launch(card.finish_cost, 1, d, s)
    assert np.isinf(s.scores.numpy()[0])


def test_restoration_moves_mean_without_returning_invalid_pose():
    d, s = cost_data(), state(n=4)
    d.parts = array(np.zeros((4, 4)))
    d.penetration, d.palm_gap = array(np.zeros(4)), array(np.full(4, 10.))
    d.self_gap = array([-4., -3., -2., -1.])
    s.selection_scores = array(np.zeros(4))
    x = np.arange(104, dtype=np.float32).reshape(4, 26)/100
    s.x.assign(x); s.scores.fill_(float('inf'))
    launch(card.prepare_ranking, 4, d, s, 0)
    np.testing.assert_allclose(s.selection_scores.numpy(), np.array([4.5, 3.5, 2.5, 1.5])**2)
    launch(card.rank_candidates, 4, s); launch(card.update_distribution, 26, s)
    launch(card.commit_score, 1, s, 0)
    assert not np.allclose(s.mean.numpy(), s.seed.numpy())
    assert np.isinf(s.best_score.numpy()[0])
    np.testing.assert_array_equal(s.best.numpy(), s.seed.numpy())
    assert s.restoration_history.numpy()[0] == 1
    # Once a feasible candidate appears, infeasible candidates stop competing.
    s.scores.assign([np.inf, 10., np.inf, np.inf])
    launch(card.prepare_ranking, 4, d, s, 1)
    launch(card.rank_candidates, 4, s); launch(card.update_distribution, 26, s)
    launch(card.commit_score, 1, s, 1)
    assert s.best_score.numpy()[0] == 10.
    np.testing.assert_array_equal(s.best.numpy(), x[1])
    assert s.restoration_history.numpy()[1] == 0


def test_palm_mesh_clearance_uses_nonattached_capsules_only():
    d = cost_data()
    mesh, vertices, faces = mesh_box([.02, .02, .02])
    d.palm_mesh, d.palm_body = mesh.id, 0
    d.rotation.assign(np.tile(np.eye(3), (1, 3, 1, 1)))
    d.position.assign([[[0, 0, 0], [0, 0, 0], [.03, 0, 0]]])
    d.palm_gap.fill_(float('inf'))
    launch(card.palm_clearance_masked, (1, 1), d, array([1], int))
    assert d.palm_gap.numpy()[0] == pytest.approx(10., abs=1.e-4)


def test_exported_contact_is_the_scored_surface_point():
    d = cost_data(); launch(card.contact_cost, (1, 1), d)
    local, normal = array(np.zeros((2, 3)), wp.vec3), array(np.zeros((2, 3)), wp.vec3)
    launch(card.export_contacts, 2, d, array([0, 0], int), local, normal)
    chosen = d.selected.numpy()[0, 0]
    np.testing.assert_array_equal(local.numpy(), np.tile(d.sample_local.numpy()[chosen], (2, 1)))
    np.testing.assert_array_equal(normal.numpy(), np.tile(d.sample_normal.numpy()[chosen], (2, 1)))


@pytest.mark.parametrize('option', ['--cem-gradient-steps', '--cem-sigma', '--no-cem-optimize-pose', '--grasp-endpoint-weight'])
def test_removed_cli_options_have_migration_message(option, capsys):
    from run_grasp import main
    with pytest.raises(SystemExit) as error:
        main(['--primitives', option+'=0'])
    assert error.value.code == 2
    assert 'Removed dynamics CEM options' in capsys.readouterr().err


@pytest.mark.parametrize('primitive', ['sphere', 'cylinder', 'cuboid'])
def test_real_wuji_geometry_static_search_and_export_contacts_on_cpu(tmp_path, primitive, monkeypatch):
    """Run the actual numerical search on CPU; this is not a CUDA graph test."""
    import comfree_warp
    import mujoco
    from retarget.dexonomy import ROOT
    from retarget.grasp import build_scene, StaticHand
    from retarget.grasp_initialization import GraspInitialization
    from retarget.grasp_primitives import primitive_mesh
    from retarget.grasp_runtime import prepare_experiment
    from retarget.grasp_audit import GraspAudit
    from forceaware.targets import build_index
    path = primitive_mesh(tmp_path/'object.obj', primitive, .025, .05)
    model, surface, _, _ = build_scene(path)
    bodies, local, normals = [], [], []
    for finger in (1, 2, 3):
        body = model.body(f'right_finger{finger}_link4').id
        with np.load(ROOT/'assets/wuji/qp_right'/f'right_finger{finger}_link4.npz') as asset:
            bodies.append(body); local.append(asset['local_pos'][10]); normals.append(asset['local_normal'][10])
    hand = StaticHand(model, surface, np.array(bodies), np.array(local), np.array(normals))
    q = np.r_[[0., 0., .2, .3, -.1, .2], np.full(20, .1)]
    hand.set(q)
    object_points = np.array([[.025, 0, 0], [-.025, 0, 0], [0, .025, 0]])
    forces = np.array([1., 2., 4.]) if primitive == 'cylinder' else None
    guess = GraspInitialization(q, object_points, object_points/.025, {}, {}, contact_forces=forces)
    args = SimpleNamespace(grasp_optimizer='cem', solver_config=ROOT/'data/example/forceaware.yaml',
        object_mass=.2, friction=.6, load_conditioning=True, closing_depth_mm=4.,
        cem_population=8, cem_iterations=2, seed=0)
    experiment = prepare_experiment(tmp_path/'input', hand, guess, args)
    assert not (experiment/'load_input.npz').exists()
    assert not (experiment/'closing_reference.json').exists()
    np.testing.assert_array_equal(np.load(experiment/'contactaware/wuji_right_qpos.npy')[0], q.astype(np.float32))
    with np.load(experiment/'card_input.npz') as supplied:
        assert ('contact_forces' in supplied) == (forces is not None)
        np.testing.assert_array_equal(supplied['object_normals'], object_points/.025)
    dm, full = comfree_warp.load_model(str(experiment/'sequence/scene/wuji_right/primitive_sdf.xml'), device='cpu')
    index = build_index(full)
    initial = full.qpos0.copy(); initial[index.hand_qpos] = q
    runtime = SimpleNamespace(model=full, device_model=dm, index=index, qpos=initial,
        qvel=np.zeros(full.nv), experiment=experiment, args=args,
        cfg=SimpleNamespace(sequence=SimpleNamespace(hand='wuji_right')))
    card.initialize_runtime(runtime)
    runtime.audit_context = GraspAudit(surface, 'cpu'), hand, args
    node = card.CardCemNode.__new__(card.CardCemNode)
    node.runtime, node.backend, node.device = runtime, runtime.optimizer.backend, 'cpu'
    node.selected = wp.zeros(1, dtype=int, device='cpu')
    node._initialize_search(); node._compute()
    np.testing.assert_allclose(node.cost.rho.numpy(), card.force_priorities(forces, 3))
    assert all(full.body(int(node.cost.capsule_body.numpy()[i])).name.endswith(('link3', 'link4'))
               for i in node.palm_capsules.numpy())
    output = node.state.qpos.numpy()[0]
    fk = mujoco.MjData(full); fk.qpos[:] = output; mujoco.mj_kinematics(full, fk)
    np.testing.assert_allclose(runtime.static_pose.body_position.numpy()[0], fk.xpos, atol=2.e-6)
    np.testing.assert_allclose(runtime.static_pose.body_matrix.numpy()[0], fk.xmat.reshape(-1, 3, 3), atol=2.e-6)
    np.testing.assert_array_equal(output[26:], initial[26:])
    assert np.isfinite(node.cost.parts.numpy()[0]).all()
    chosen = node.cost.selected.numpy()[0][node.annotation_group.numpy()]
    assert node.cost.sample_pad.numpy()[chosen].all()
    np.testing.assert_array_equal(runtime.card_contact_local.numpy(), node.cost.sample_local.numpy()[chosen])
    if not np.isfinite(node.state.best_score.numpy()[0]):
        # No invalid proposal is exported as an optimized answer.
        np.testing.assert_allclose(output[:3], initial[:3], atol=1.e-6)
        np.testing.assert_allclose(output[6:26], initial[6:26], atol=1.e-6)
    else:
        assert node.cost.penetration.numpy()[0] <= 3.
        assert node.cost.palm_gap.numpy()[0] >= 6.
        assert node.cost.self_gap.numpy()[0] >= node.cost.self_clearance

    # This far-from-object pose has valid geometry even though its contacts
    # cannot certify a grasp. The optimizer and independent audit are distinct.
    assert np.isfinite(node.state.best_score.numpy()[0])
    node.prepare_seconds = 0.; node.start_event = node.end_event = None
    node.audit = SimpleNamespace(export=lambda: ({'feasible': False}, {}))
    monkeypatch.setattr(wp, 'get_event_elapsed_time', lambda *a, **kw: 0.)
    plan, snap, exported = node.export()
    np.testing.assert_array_equal(exported, output)
    assert plan.diag['optimizer_success'] and not runtime.device_audit_result[0]['feasible']
    assert runtime.optimizer.last_solver_diag['dynamics_rollouts'] == 0
    assert runtime.optimizer.last_solver_diag['evaluation_worlds'] == args.cem_population
    assert not snap['admm_active']
    # Put the entire hand inside an object. Every candidate must be rejected,
    # preserving the initial pose and overriding even a positive audit flag.
    enclosing, enclosing_vertices, enclosing_faces = mesh_box([2., 2., 2.])
    node.cost.object_mesh = enclosing.id
    node._compute()
    node.audit = SimpleNamespace(export=lambda: ({'feasible': True}, {}))
    plan, snap, exported = node.export()
    assert not plan.diag['optimizer_success']
    assert not runtime.device_audit_result[0]['feasible']
    assert runtime.optimizer.last_solver_diag['termination'] == 'no_valid_candidate_initialization_saved'
    np.testing.assert_allclose(exported, initial, atol=1.e-6)


def test_cli_keeps_cem_default_and_adamw_legacy_and_uses_paper_budget(monkeypatch):
    import retarget.grasp
    import retarget.grasp_compact
    import retarget.grasp_results
    from run_grasp import main
    monkeypatch.setattr(retarget.grasp, 'run', lambda args: args)
    monkeypatch.setattr(retarget.grasp_compact, 'run_compact', lambda args: args)
    monkeypatch.setattr(retarget.grasp_results, 'register_results', lambda *args, **kwargs: None)
    common = ['--primitives', '--templates', 'test', '--no-viewer', '--no-render']
    args = main(common)
    assert args.grasp_optimizer == 'cem'
    assert args.hand_model == 'wuji2'
    assert args.explicit_pose
    assert (args.explicit_horizon, args.explicit_dt, args.explicit_sigma) == (4, .1, .5)
    assert args.explicit_pose_weight == 1.e6
    assert not main(common+['--no-explicit-pose']).explicit_pose
    assert main(common+['--hand-model', 'wuji1']).hand_model == 'wuji1'
    assert (args.cem_population, args.cem_iterations) == (1024, 16)
    assert (args.cem_elite_fraction, args.cem_temperature, args.cem_final_noise) == (.05, .1, .2)
    assert (args.cem_translation_sigma_mm, args.cem_rotation_sigma, args.cem_joint_sigma) == (2., .016, .05)
    assert not hasattr(args, 'cem_gradient_steps')
    assert main(common+['--grasp-optimizer', 'adamw']).grasp_optimizer == 'adamw'
    assert main(common+['--legacy-serial', '--no-drop-test']).grasp_optimizer == 'adamw'


@pytest.mark.parametrize('option,value', [('--cem-temperature', 'nan'), ('--cem-final-noise', '0'),
                                        ('--cem-elite-fraction', 'inf'), ('--cem-population', '1')])
def test_invalid_sampling_configuration_rejected_by_cli(option, value, capsys):
    from run_grasp import main
    with pytest.raises(SystemExit) as error:
        main(['--primitives', '--no-render', option, value])
    assert error.value.code == 2
    assert 'Invalid CARD CEM' in capsys.readouterr().err
