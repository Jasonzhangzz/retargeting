"""Static grasp invariants, analytic kinematics and independent closure oracle."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from retarget.grasp import build_scene, StaticHand
from retarget.grasp_closure import force_closure
from retarget.grasp_initialization import (global_alignment, pose_in_hand,
                                           from_dexonomy_result, upstream_modules,
                                           matching_diagnostics, template_contact_layout)
from retarget.grasp_primitives import primitive_mesh, catalog, jobs
from retarget.dexonomy import ROOT, nearest_sdf
from test_dexonomy import checkout


@pytest.fixture
def hand(tmp_path):
    path = primitive_mesh(tmp_path / 'box.obj', 'cuboid', .035, .09)
    model, surface, xml, center = build_scene(path)
    assert model.nq == model.nv == 26
    assert 'contactaware' not in xml and 'sugar_box' not in xml
    bodies, local, normal = [], [], []
    for name in ['right_finger1_link4', 'right_finger2_link4', 'right_finger3_link4']:
        with np.load(ROOT / 'assets/wuji/qp_right' / f'{name}.npz') as query:
            bodies.append(model.body(name).id)
            local.append(query['local_pos'][10]); normal.append(query['local_normal'][10])
    normal = np.asarray(normal); normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    return StaticHand(model, surface, np.array(bodies), np.asarray(local), normal)


def test_analytic_mujoco_contact_jacobian_matches_independent_perturbation(hand):
    q = np.r_[[.03, .01, -.1, .2, -.1, .4], np.full(20, .2)]
    hand.set(q)
    _, analytic, _ = hand.points(hand.bodies, hand.local, True)
    finite = np.empty_like(analytic)
    for column in range(26):
        delta = np.eye(26)[column]*1e-7
        hand.set(q+delta); plus = hand.points(hand.bodies, hand.local)
        hand.set(q-delta); minus = hand.points(hand.bodies, hand.local)
        finite[:, :, column] = (plus-minus)/2e-7
    np.testing.assert_allclose(analytic, finite, atol=1e-8, rtol=1e-6)


def test_upstream_normal_filter_rejects_one_opposed_contact_despite_small_mean():
    # One reversed normal must fail even when 19 perfect contacts hide it in
    # the mean objective. Upstream normal_thre is a squared residual, not rad.
    n=np.zeros((2,20));n[0,-1]=4.;n[1,-1]=.2
    valid,diag=matching_diagnostics(np.zeros_like(n),n,
        normal_threshold=.3,position_threshold=.001)
    np.testing.assert_array_equal(valid,[False,True])
    assert diag['matching_normal_cost'][0] == .2
    assert diag['normal_angle_deg'][0][-1] == 180.
    np.testing.assert_allclose(diag['normal_angle_deg'][1][-1],np.degrees(np.arccos(.9)))


def test_contact_layout_uses_oriented_separation_not_just_pair_distance(hand):
    q=np.r_[np.zeros(6),np.full(20,.3)]
    diagnostic=template_contact_layout(hand,q)
    p=hand.points(hand.bodies,hand.local)
    n=np.einsum('nij,nj->ni',hand.data.xmat[hand.bodies].reshape(-1,3,3),hand.normals)
    expected=np.array([[(p[j]-p[i])@n[i]*1000 for j in range(len(p))] for i in range(len(p))])
    np.testing.assert_allclose(diagnostic['normal_toward_other_contact_mm'],expected,atol=1e-12)


def test_cylinder_matching_normals_preserve_sharp_cap_edges_and_exact_triangles(tmp_path):
    from retarget.grasp_initialization import matching_mesh
    _,surface,_,_=build_scene(primitive_mesh(tmp_path/'cylinder.obj','cylinder',.02,.08))
    mesh=matching_mesh(surface)
    # Same oriented triangle soup, including winding, regardless of face order.
    expected=sorted(map(tuple,surface.triangles.reshape(-1,9)))
    actual=sorted(map(tuple,mesh.triangles.reshape(-1,9)))
    np.testing.assert_array_equal(actual,expected)
    cap=np.abs(mesh.face_normals[:,2])>.99
    vn=mesh.vertex_normals[mesh.faces]
    np.testing.assert_allclose(vn[~cap,:,2],0.,atol=1e-7)
    np.testing.assert_allclose(np.abs(vn[cap,:,2]),1.,atol=1e-7)


def test_input_bridge_preserves_geometry_under_nonidentity_world_pose(hand):
    q = np.r_[[.02, -.04, .01, .4, -.2, .6], np.full(20, .3)]
    hand.set(q)
    points = hand.points(hand.bodies, hand.local)
    outward = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    obj_rot = Rotation.from_euler('XYZ', [.7, -.8, .9])
    obj_translation = np.array([.1, .2, .3])
    hand_rotation_world = obj_rot*Rotation.from_euler('XYZ', q[3:6])
    raw = dict(obj_pose=np.r_[obj_translation, obj_rot.as_quat(scalar_first=True)],
               grasp_pose=np.r_[obj_rot.apply(q[:3])+obj_translation,
                                hand_rotation_world.as_quat(scalar_first=True)],
               grasp_qpos=q[6:], obj_cp_w=obj_rot.apply(points)+obj_translation,
               obj_cn_w=-obj_rot.apply(outward))
    guess = from_dexonomy_result(raw)
    hand.set(guess.hand_qpos)
    np.testing.assert_allclose(hand.points(hand.bodies, hand.local), points, atol=1e-14)
    np.testing.assert_allclose(guess.object_points, points, atol=1e-14)
    np.testing.assert_allclose(guess.object_normals, outward, atol=1e-14)
    np.testing.assert_allclose(pose_in_hand(guess.hand_qpos), pose_in_hand(q), atol=1e-14)
    for key in raw:
        np.testing.assert_array_equal(guess.raw[key], raw[key])


def test_global_alignment_calls_original_sampler_loader_and_gpu_matcher(hand, monkeypatch):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available() or not (ROOT/'third_party/Dexonomy/dexonomy/op/gen_init.py').is_file():
        pytest.skip('Original Dexonomy integration needs checkout and CUDA')
    args = SimpleNamespace(init_points=3, init_inplane=3, starts=1,
                           init_iterations=4, dexonomy_root=ROOT/'third_party/Dexonomy')
    modules = upstream_modules(args.dexonomy_root)
    called, original_output = [], {}
    for owner, name in ((modules[1], 'sample_init_pose'),
                        (modules[2].HandTemplateLoader, '_load_data'),
                        (modules[0].HandObjMatcher, 'forward')):
        original = getattr(owner, name)
        def wrapper(*a, _name=name, _original=original, **kw):
            called.append(_name)
            value = _original(*a, **kw)
            if _name == 'forward':
                original_output.update({key: v.detach().cpu().numpy().copy() for key, v in value.items()})
            return value
        monkeypatch.setattr(owner, name, wrapper)
    q = np.r_[np.zeros(6), np.full(20, .3)]
    guesses = global_alignment(hand, q, args, np.random.default_rng(0))
    assert len(guesses) == 1
    guess = guesses[0]
    assert set(called) == {'sample_init_pose', '_load_data', 'forward'}
    np.testing.assert_allclose(guess.hand_qpos[6:], q[6:], atol=2e-8)
    assert guess.metadata['fixed_hand_joint_angles']
    if guess.metadata['geometry_valid_candidates']:
        assert guess.metadata['initialization_geometry_valid']
        assert guess.metadata['initial_penetration_mm'] <= guess.metadata['penetration_tolerance_mm']
    i = guess.metadata['candidate_index']
    for key, value in original_output.items():
        np.testing.assert_array_equal(guess.raw[key], value[i])
    np.testing.assert_array_equal(guess.object_points, original_output['obj_cp_w'][i])
    np.testing.assert_array_equal(guess.object_normals, -original_output['obj_cn_w'][i])
    hand.set(guess.hand_qpos)
    np.testing.assert_allclose(hand.points(hand.bodies, hand.local), guess.raw['hand_cpn_w'][:, :3], atol=1e-7)


def test_original_surface_sampler_seed_is_reproducible_and_scoped():
    import trimesh
    from retarget.grasp_initialization import sample_original_poses
    checkout=ROOT/'third_party/Dexonomy'
    if not (checkout/'dexonomy/data/obj_loader.py').is_file():
        pytest.skip('Original sampler requires Dexonomy checkout')
    modules=upstream_modules(checkout)
    sampler=modules[1]
    mesh=trimesh.creation.cylinder(radius=.02,height=.08)
    original=(trimesh.sample.sample_surface_even,trimesh.sample.sample_surface)
    a=sample_original_poses(sampler,mesh,12,4,7)
    b=sample_original_poses(sampler,mesh,12,4,7)
    c=sample_original_poses(sampler,mesh,12,4,8)
    for left,right in zip(a,b):
        np.testing.assert_array_equal(left,right)
    assert not np.array_equal(a[1],c[1])
    assert original==(trimesh.sample.sample_surface_even,trimesh.sample.sample_surface)


def test_force_closure_distinguishes_rank_loss_and_six_axis_positive_span():
    points = np.array([[-.05, 0., 0.], [.05, 0., 0.], [-.05, 0., 0.]])
    normals = np.array([[-1., 0., 0.], [1., 0., 0.], [-1., 0., 0.]])
    before = force_closure(points, normals, radius=.1)
    assert before['positive_margin'] > 0 and before['rank'] == 5
    assert not before['force_closure']
    points[2, 1] = .001
    after = force_closure(points, normals, radius=.1)
    assert after['force_closure'] and after['rank'] == 6
    assert after['equilibrium_residual'] < 1e-8
    two = force_closure(points[:2], normals[:2], radius=.1)
    assert not two['force_closure'] and two['rank'] == 5


def test_gravity_support_does_not_imply_force_closure():
    points = np.array([[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.]])
    normals = np.tile([0., 0., -1.], (3, 1))
    assert not force_closure(points, normals, radius=.1)['force_closure']


def test_primitive_radius_and_height_are_independent(tmp_path):
    paths = [primitive_mesh(tmp_path / f'c{i}.obj', 'cylinder', r, h)
             for i, (r, h) in enumerate([(.02, .1), (.04, .1), (.02, .2)])]
    extents = []
    for path in paths:
        _, surface, _, _ = build_scene(path)
        vertices = surface.triangles.reshape(-1, 3)
        extents.append(np.ptp(vertices, axis=0))
    np.testing.assert_allclose(extents[1], extents[0]*[2, 2, 1], atol=1e-8)
    np.testing.assert_allclose(extents[2], extents[0]*[1, 1, 2], atol=1e-8)


def test_all_33_types_have_provenance_and_special_cases_are_not_cylinders(tmp_path):
    annotation = tmp_path / 'upstream/assets/hand/shadow/raw_anno'
    annotation.mkdir(parents=True)
    for i in range(1, 34):
        (annotation / f'{i}_Test.yaml').write_text('{}')
    rows = catalog(tmp_path / 'upstream')
    assert len(rows) == 33
    assert all('not_Dexonomy_metadata' in row['mapping_status'] for row in rows)
    assert not rows[18]['supported'] and not rows[20]['supported']
    args = SimpleNamespace(primitive=None, radius_scales=[.75, 1., 1.25], height_scales=[1.], radius=None, height=None)
    generated, skipped = jobs(tmp_path / 'output', ['1_Test', '19_Test', '21_Test'], args)
    assert len(generated) == 6 and len(skipped) == 2


def test_independent_cli_passes_original_contacts_directly_to_admm(checkout, tmp_path, monkeypatch):
    import json
    import mujoco
    from run_grasp import main
    import retarget.grasp_results as registry
    monkeypatch.setattr(registry, 'RESULTS_ROOT', tmp_path/'registry')
    import retarget.grasp as grasp
    import retarget.grasp_initialization as initialization
    import retarget.grasp_runtime as runtime_module
    import retarget.fa.solver as solver_module
    import retarget.grasp_results as store
    monkeypatch.setattr(store, 'RESULTS_ROOT', tmp_path/'store')
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('Shared optimizer integration requires CUDA')
    observed = {}
    def supplied_result(hand, q, args, rng, **kwargs):
        # Deliberately off-surface contacts detect any extra projection in the
        # input path. The real ADMM is allowed to optimize them after entry.
        q = q.copy(); q[:3] += [.4, .4, .4]
        hand.set(q)
        points = hand.points(hand.bodies, hand.local)+[.013, -.017, .019]
        raw = dict(obj_pose=np.array([0., 0., 0., 1., 0., 0., 0.]),
                   grasp_pose=np.r_[q[:3], Rotation.from_euler('XYZ', q[3:6]).as_quat(scalar_first=True)],
                   grasp_qpos=q[6:], obj_cp_w=points, obj_cn_w=np.tile([0., 0., -1.], (len(points), 1)))
        guess = from_dexonomy_result(raw, dict(fixed_hand_joint_angles=True))
        observed['guess'] = guess
        return [guess]
    original_solve = runtime_module.GraspRuntime.solve
    def solve(runtime):
        expected = observed['guess'].object_points.astype(np.float32)
        np.testing.assert_array_equal(runtime.targets.contact_position_object[0],expected)
        np.testing.assert_allclose(runtime.qpos[runtime.index.hand_qpos],observed['guess'].hand_qpos,atol=1e-7)
        assert runtime.optimizer.backend.device.is_cuda
        assert runtime.settings.hand_solver == 'adamw'
        result = original_solve(runtime)
        observed['entered_admm'] = True
        observed['settings'] = runtime.settings
        return result
    monkeypatch.setattr(initialization, 'global_alignment', supplied_result)
    monkeypatch.setattr(runtime_module.GraspRuntime, 'solve', solve)
    root, _ = checkout
    mesh = primitive_mesh(tmp_path / 'input.obj', 'sphere', .025, .05)
    output = main(['--archive-results', '--object', str(mesh), '--dexonomy-root', str(root), '--templates', 'test',
                   '--legacy-serial',
                   '--output', str(tmp_path / 'runs'), '--name', 'cli', '--starts', '1',
                   '--rounds', '1', '--hand-iterations', '1', '--multistarts','1',
                   '--init-points', '2', '--init-inplane', '2',
                   '--init-iterations', '2', '--contact-samples', '1', '--qp-iterations', '8',
                   '--no-viewer','--no-render','--no-drop-test'])
    assert (output / 'summary.csv').is_file()
    assert observed['entered_admm']
    summary = json.loads((output / 'summary.json').read_text())
    assert len(summary) == 1
    case = next(output.rglob('grasp.npz')).parent
    report = json.loads((case / 'result.json').read_text())
    assert 'fixed_hand_joint_angles' in report['selected']['initialization']
    with np.load(case / 'grasp.npz') as saved:
        assert saved['initial_qpos'].shape == saved['predicted_terminal_qpos'].shape == saved['qpos'].shape
        assert report['selected']['output_time_seconds'] == .01
        assert report['selected']['prediction_horizon_seconds'] == .3
        with np.load(case/'contacts.npz') as snapshots:
            np.testing.assert_array_equal(saved['object_points'],snapshots['z'][0])
            np.testing.assert_array_equal(saved['dual'],snapshots['dual'][0])
        np.testing.assert_array_equal(saved['initial_object_points'], observed['guess'].object_points)
        np.testing.assert_array_equal(saved['initial_object_normals'], observed['guess'].object_normals)
        np.testing.assert_array_equal(saved['template_object_points'], observed['guess'].object_points)
        for key, value in observed['guess'].raw.items():
            np.testing.assert_array_equal(saved['dexonomy_'+key], value)
        assert saved['initial_object_pose_in_hand'].shape == (7,)
        assert saved['final_object_pose_in_hand'].shape == (7,)
        assert saved['initial_hand_points'].shape == saved['hand_points'].shape
        model = mujoco.MjModel.from_xml_path(str(case / 'scene.xml'))
        data = mujoco.MjData(model)
        data.qpos[:] = saved['qpos']
        mujoco.mj_kinematics(model, data)
        body = saved['hand_body_ids']
        actual = data.xpos[body]+np.einsum('nij,nj->ni', data.xmat[body].reshape(-1, 3, 3), saved['hand_local_points'])
        obj = model.body('grasp_object').id
        canonical = (actual-data.xpos[obj])@data.xmat[obj].reshape(3,3)
        np.testing.assert_allclose(canonical, saved['hand_points'], atol=1e-6)
    # Previously generated full-horizon output must not silently reappear as
    # a corrected grasp when a user reuses their old replay command.
    report['output_pose']='terminal predicted state of selected common-solver rollout'
    (case/'result.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='full prediction'):
        grasp.replay(case,viewer=False)
    seen=[]
    monkeypatch.setattr(grasp,'display',lambda model,q,case,**kw:seen.append((q.copy(),kw)))
    grasp.replay(case,viewer=False,stage='prediction')
    assert seen[0][1]['image_name']=='prediction.png'


def test_fitted_root_is_evaluated_by_original_matcher_without_rerandomizing(hand):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available() or not (ROOT/'third_party/Dexonomy/dexonomy/op/gen_init.py').is_file():
        pytest.skip('Original Dexonomy integration needs checkout and CUDA')
    args = SimpleNamespace(init_points=3, init_inplane=3, starts=10,
        init_iterations=4, dexonomy_root=ROOT/'third_party/Dexonomy')
    q = np.r_[[.03, .01, -.02, .2, .15, -.1], np.full(20, .3)]
    guesses = global_alignment(hand, q, args, np.random.default_rng(2), pose_seed=q)
    assert len(guesses) == 10
    seed = next(g for g in guesses if g.metadata['candidate_index'] == 9)
    assert seed.metadata['random_sampled_poses'] == 9
    assert seed.metadata['fitted_pose_seed_candidates'] == 1
    np.testing.assert_allclose(seed.hand_qpos, q, atol=2e-7)
    hand.set(seed.hand_qpos)
    np.testing.assert_allclose(hand.points(hand.bodies,hand.local), seed.raw['hand_cpn_w'][:,:3], atol=1e-7)
    np.testing.assert_array_equal(seed.object_points,seed.raw['obj_cp_w'])
    np.testing.assert_array_equal(seed.object_normals,-seed.raw['obj_cn_w'])


def test_invalid_initialization_skips_physics_but_allows_a_repaired_plan():
    from retarget.grasp_preview import invalid_preload_report
    args = SimpleNamespace(load_conditioning=True)
    initial = dict(initialization_valid=False, load_conditioning_required=True,
        initial_self_penetration_mm=5.,load_conditioning=dict(feasible=False))
    result = dict(selected=dict(initialization=initial,feasible=False))
    # Legacy closing/MPC formation may repair a bad first frame. Do not skip
    # those existing paths just because the initial audit failed.
    assert invalid_preload_report(result,args) is None
    initial['template_fit']=dict(object_contact_refinements=[dict(
        method='CUDA_dense_power_cylinder_input_IK',accepted=True)])
    initial['load_conditioning']['force_cone_diagnostic']=dict(static_load_excluded=True)
    report = invalid_preload_report(result,args)
    assert report['status']=='initialization_failed'
    assert not report['gravity_test_ran'] and not report['dropped'] and not report['verified_grasp']
    assert report['failure_reasons']==['finger_self_collision','contact_force_cones_do_not_balance_load']
    result['selected']['feasible']=True
    assert invalid_preload_report(result,args) is None
    result['selected']['feasible']=False
    assert invalid_preload_report(result,SimpleNamespace(load_conditioning=False)) is None
    initial['initialization_valid']=True
    assert invalid_preload_report(result,args) is None
