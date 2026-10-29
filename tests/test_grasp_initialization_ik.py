"""Regression coverage for folded IK seeds and failure propagation."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from retarget.grasp import build_scene, StaticHand
from retarget.dexonomy import ROOT, load_template, map_contacts, anatomical_joints
from retarget.grasp_primitives import primitive_mesh
from retarget.grasp_initialization import validate_initialization

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def mapped_hand(tmp_path, name, radius, height, shape, hand_model='wuji2'):
    model, surface, _, _ = build_scene(
        primitive_mesh(tmp_path/'object.obj', shape, radius, height), hand_model=hand_model)
    shadow, posed, contacts, _ = load_template(ROOT/'third_party/Dexonomy', name)
    bodies, local, normals, _ = map_contacts(shadow, contacts, model)
    return StaticHand(model, surface, bodies, local, normals), shadow, posed, contacts


@cuda
def test_small_diameter_multistart_escapes_reverse_flexion(tmp_path):
    from retarget.grasp_seed import bone_seed_cuda
    hand, shadow, posed, _ = mapped_hand(tmp_path, '2_Small_Diameter', .012, .1, 'cylinder')
    q, info = bone_seed_cuda(hand.model, anatomical_joints(shadow, posed)[None], device='cuda:0')
    assert info['multistarts'] == 8 and info['iterations'] == 96
    assert max(info['finger_direction_rms'][1:4]) < .1
    assert np.min(q[0, 6:].reshape(5, 4)[1:4, 2:]) > 0.
    errors = np.array(info['candidate_direction_rms'])
    assert np.all(np.array(info['finger_direction_rms']) <= errors[0]+1.e-12)
    again, meta = bone_seed_cuda(hand.model, anatomical_joints(shadow, posed)[None], device='cuda:0')
    np.testing.assert_allclose(q, again, atol=1.e-12)
    assert info['selected_candidates'] == meta['selected_candidates']


@cuda
@pytest.mark.parametrize('scale', [.75, 1., 1.25])
@pytest.mark.parametrize('name,shape,radius,height', [
    ('2_Small_Diameter', 'cylinder', .012, .1),
    ('6_Prismatic_4_Finger', 'cylinder', .02, .08),
    ('13_Precision_Sphere', 'sphere', .02, .04),
    ('14_Tripod', 'sphere', .015, .03),
    ('17_Index_Finger_Extension', 'cylinder', .025, .1)])
def test_fits_preserve_original_skin_and_unannotated_fingers(tmp_path, name, shape, radius, height, scale):
    from retarget.grasp_seed import bone_seed_cuda
    from retarget.grasp_template import fit_contact_template, BoneReference
    from retarget.grasp_input_geometry import dense_power_template, fit_power_cylinder_input
    height = height*scale if shape == 'sphere' else height
    hand, shadow, posed, contacts = mapped_hand(tmp_path, name, radius*scale, height, shape)
    seed, _ = bone_seed_cuda(hand.model, anatomical_joints(shadow, posed)[None], device='cuda:0')
    reference = seed[0].copy()
    regions = hand.normals.copy()
    kwargs = dict(device='cuda:0', joint_prior=1., reference_qpos=reference, reference_normals=regions)
    q, info = fit_contact_template(hand, shadow, posed, contacts, reference,
                                  opening=2*radius*scale, **kwargs)
    if dense_power_template(hand, dict(kind=shape)):
        q, info = fit_power_cylinder_input(hand, q, **kwargs)
    assert len(hand.local) == len(contacts)
    assert min(info['contact_region_normal_dot']) > .5
    assert info['bone_direction_max_deviation_deg'] <= 30.+1.e-5
    np.testing.assert_array_equal(info['reference_qpos'], reference)
    np.testing.assert_array_equal(regions, kwargs['reference_normals'])
    bones = BoneReference(hand.model, reference, 'cuda:0')
    actual = bones.directions(bones.fk.tensor(q)).cpu().numpy()
    expected = bones.target.cpu().numpy()
    names = [hand.model.body(int(body)).name for body in hand.bodies]
    for finger in range(1, 6):
        if not any(f'right_finger{finger}_' in n for n in names):
            assert (actual[finger-1]*expected[finger-1]).sum(-1).min() > np.cos(np.deg2rad(10))
    if name == '17_Index_Finger_Extension':
        assert (actual[1]*expected[1]).sum(-1).min() > 0.  # no reversal of the extended index


@cuda
def test_optional_contacts_accept_shape_but_reject_unannotated_dorsal_contact(tmp_path):
    from retarget.grasp_seed import bone_seed_cuda
    from retarget.grasp_regions import hand_forbidden_mask
    hand, shadow, posed, _ = mapped_hand(tmp_path, '2_Small_Diameter', .005, .01, 'sphere')
    reference, _ = bone_seed_cuda(hand.model, anatomical_joints(shadow, posed)[None], device='cuda:0')
    hand.ik_reference_qpos = reference[0]
    hand.contact_reference_normals = hand.normals.copy()
    args = SimpleNamespace(init_device='cuda:0', contact_tolerance_mm=2., penetration_tolerance_mm=10.)
    def guess(q):
        return SimpleNamespace(hand_qpos=q, metadata=dict(initialization_valid=True,
            contacts_optional=True, penetration_tolerance_mm=10.,
            matching=dict(gap_mm=[100.]*len(hand.bodies), normal_squared_residual_max=4.),
            load_conditioning_required=True, load_conditioning=dict(feasible=False)))
    separated = reference[0].copy();separated[:3] = [1., 1., 1.]
    g = guess(separated)
    assert validate_initialization(hand, g, args, hand.contact_reference_normals)
    assert g.metadata['contacts_optional'] and g.metadata['forbidden_region_valid']
    assert g.metadata['bone_direction_max_deviation_deg'] < 1.e-4
    # Put an unannotated back surface at the sphere's centre. Annotated local
    # normals remain unchanged, so checking only contact anchors misses this.
    hand.set(reference[0])
    mask = hand_forbidden_mask(hand, hand.contact_reference_normals)
    point = hand.points(hand.all_bodies, hand.all_local)[np.flatnonzero(mask)[0]]
    intersecting = reference[0].copy();intersecting[:3] -= point
    g = guess(intersecting)
    assert not validate_initialization(hand, g, args, hand.contact_reference_normals)
    assert 'forbidden_region_contact' in g.metadata['failure_reasons']
    assert g.metadata['forbidden_region_min_gap_mm'] < 0.
    from retarget.grasp_regions import SurfaceClearance
    clearance = SurfaceClearance(hand, hand.contact_reference_normals, 'cuda:0')
    value = clearance.fk.tensor(intersecting)
    clearance.refresh(value)
    direction = torch.zeros_like(value);direction[0] = 1.;direction[12] = .3
    _, derivative = torch.func.jvp(clearance.residual, (value,), (direction,))
    epsilon = 1.e-6
    finite = (clearance.residual(value+epsilon*direction)-clearance.residual(value-epsilon*direction))/(2*epsilon)
    assert float(derivative.abs().max()) > .1
    torch.testing.assert_close(derivative, finite, atol=1.e-6, rtol=1.e-4)


def test_final_ik_filter_uses_mm_tolerance_limits_and_fixed_skin():
    model = SimpleNamespace(jnt_qposadr=np.arange(26), jnt_range=np.tile([-2., 2.], (26, 1)))
    hand = SimpleNamespace(model=model, normals=np.array([[0., 1., 0.]]))
    reference = hand.normals.copy()
    def guess(gap):
        return SimpleNamespace(hand_qpos=np.zeros(26), metadata=dict(initialization_valid=True,
            matching=dict(gap_mm=[gap], normal_squared_residual_max=.1)))
    args = SimpleNamespace(contact_tolerance_mm=2.)
    valid = guess(1.)
    assert validate_initialization(hand, valid, args, reference)
    invalid = guess(3.)
    assert not validate_initialization(hand, invalid, args, reference)
    assert invalid.metadata['failure_reasons'] == ['contact_position_mismatch']
    hand.normals *= -1
    invalid = guess(1.)
    assert not validate_initialization(hand, invalid, args, reference)
    assert 'contact_skin_region_mismatch' in invalid.metadata['failure_reasons']


@cuda
def test_failed_ik_skips_all_optimizers_and_continues_batch(tmp_path, monkeypatch):
    from run_grasp import main
    import retarget.grasp_runtime as runtime
    import retarget.grasp_initialization as initialization
    from scripts.validate_grasp_matrix import audit_batch
    from scripts.view_grasp_results import available_stages, load_geometry
    from retarget.grasp_compact import write_batch, read_rows, audit_records
    from retarget.grasp import replay
    from retarget.grasp_playback import simulate_playback
    from retarget.grasp_preview import invalid_preload_report
    import retarget.grasp_results as store
    monkeypatch.setattr(store, 'RESULTS_ROOT', tmp_path/'store')

    original = initialization.global_alignment
    def failed(*args, **kwargs):
        guesses = original(*args, **kwargs)
        for guess in guesses:
            guess.metadata['initialization_valid'] = False
            guess.hand_qpos[6] = 10.
        return guesses
    monkeypatch.setattr(initialization, 'global_alignment', failed)
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid initialization constructed an optimizer or ran physics')
    monkeypatch.setattr(runtime, 'GraspRuntime', forbidden)
    import retarget.grasp_preview as preview
    monkeypatch.setattr(preview, 'simulate_case', forbidden)
    output = main(['--archive-results', '--primitives', '--templates', '13', '--primary-shape-only',
        '--radius-scales', '.75', '1', '--template-refinements', '0', '--no-load-conditioning',
        '--init-points', '2', '--init-inplane', '2', '--init-iterations', '2',
        '--output', str(tmp_path), '--name', 'failures', '--no-viewer', '--no-render'])
    rows = json.loads((output/'summary.json').read_text())
    assert len(rows) == 2
    proofs = audit_batch(output)
    assert all(row['saved_data_valid'] and row['closure_agrees'] for row in proofs)
    for row in rows:
        assert row['termination'] == 'initialization_failed'
        assert not row['optimizer_success'] and not row['feasible']
        assert row['timings']['solve_seconds'] == 0.
        assert row['prediction_horizon_seconds'] == 0.
        assert available_stages(row['case']) == ['ik']
        load_geometry(row['case'], 'ik')
        with pytest.raises(ValueError):
            load_geometry(row['case'], 'optimized')
        with pytest.raises(ValueError, match='Initialization failed'):
            replay(Path(row['case']), viewer=False)
        with pytest.raises(ValueError, match='Initialization failed'):
            simulate_playback(row['case'])
        skipped = invalid_preload_report(dict(selected=row), SimpleNamespace(load_conditioning=False))
        assert not skipped['gravity_test_ran']
    compact = tmp_path/'compact'
    write_batch(rows, compact, method='cem_explicit', audited=proofs)
    for row in read_rows(compact):
        assert row['termination'] == 'initialization_failed'
        assert 'hand_joint_limits' in row['initialization']['failure_reasons']
        assert available_stages(row['case']) == ['ik']
        load_geometry(row['case'], 'ik')
    assert all(r['saved_data_valid'] for r in audit_records(compact))
