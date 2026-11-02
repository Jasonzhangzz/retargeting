"""Patch reporting must neither relocate anchors nor change the MPPI cost."""
import copy

import mujoco
import numpy as np
import pytest
import torch
import trimesh

from retarget.config.run import ROOT
from retarget.env.grasp.audit import GraspAudit
from retarget.env.geometry import ObjectMeshSurface
from retarget.env.grasp.template_mppi import TemplateMPPISettings
from retarget.env.grasp.template_mppi_inputs import TemplateContacts
from retarget.env.grasp.template_mppi_patch_coverage import (
    ContactPatch, attach_patch_coverage, build_patches, diagnose_patch_coverage,
    evaluate_patches, fitted_queries, interior_mask, sharp_edges,
)


def test_interior_rejects_sharp_edges_semantic_borders_and_wrong_side():
    frame = dict(origin=np.zeros(3), lateral=np.eye(3)[0],
                 palmar=np.eye(3)[1], longitudinal=np.eye(3)[2])
    points = np.array([[.005, 0, .01], [.0005, 0, .01], [0, .0005, .01], [.01, 0, .01]])
    normals = np.array([[0, 1, 0], [0, 1, 0], [1, 0, 0], [0, -1, 0]])
    pool = dict(frame=frame, normals=normals, points=points, chart=points,
                length=.02, base=np.ones(4, bool))
    edges = np.array([[[0, 0, 0], [0, 0, .02]]])
    mask, clearance = interior_mask(pool, 'palmar', 'middle', segments=edges)
    assert mask.tolist() == [True, False, False, False]
    assert clearance[1] < .001


def test_smooth_triangulation_is_not_an_edge_but_cube_creases_are():
    mesh = trimesh.creation.box()
    edges = sharp_edges(mesh.vertices, mesh.faces)
    assert edges.shape == (12, 2, 3)


def test_contact_distance_and_normal_must_hold_at_the_same_query():
    gap = torch.tensor([0., .003, -.003, .001, .001, float('nan')])
    h = torch.tensor([[0., 1, 0]]).expand(6, -1)
    o = -h.clone()
    o[0] *= -1
    fits = fitted_queries(gap, h, o, distance_m=.002, allowance_m=.002, angle_deg=30)
    assert fits.tolist() == [False, False, False, True, True, False]


def test_interior_contact_can_cover_patch_away_from_original_anchor():
    # Only the second interior query contacts the object. The first query (or
    # an arbitrary original anchor elsewhere) must not dictate coverage.
    patch = ContactPatch('right_finger2_link4', 'index.distal.palmar', (0, 1),
        np.array([[.02, 0, 0], [0, 0, 0]]), np.array([[0, 1., 0]]*2),
        np.array([3, 8]), np.array([.002, .003]), 'test')
    class Query:
        def query(self, points):
            normals = torch.tensor([[0, -1., 0]], device=points.device).expand_as(points)
            return points, normals, points[:, 0]
    report = evaluate_patches([patch], torch.zeros(1, 3), torch.eye(3)[None],
                              Query(), TemplateMPPISettings())
    assert report['coverage'] == 1 and report['required_patch_count'] == 1
    assert report['slot_to_patch'] == [0, 0]
    assert report['patches'][0]['fitted_query_indices'] == [8]
    assert report['patches'][0]['witness']['query_index'] == 8
    side_report = evaluate_patches([patch], torch.zeros(1, 3), torch.eye(3)[None],
        Query(), TemplateMPPISettings(), side_bodies=(patch.body,), side_audit=Query())
    assert side_report['covered_side_finger_count'] == side_report['required_side_finger_count'] == 1


def test_empty_interior_remains_uncovered_without_body_fallback():
    patch = ContactPatch('right_finger2_link4', 'index.distal.palmar', (0,),
        np.empty((0, 3)), np.empty((0, 3)), np.array([], int), np.array([]), 'test')
    class Query:
        def query(self, _):
            pytest.fail('No eligible points should be queried')
    report = evaluate_patches([patch], torch.zeros(1, 3), torch.eye(3)[None], Query(), TemplateMPPISettings())
    assert report['coverage'] == 0
    assert report['empty_patches'] == ['index.distal.palmar']


def test_required_object_side_cannot_be_satisfied_by_a_cap_patch_contact():
    patch = ContactPatch('right_finger2_link4', 'index.distal.palmar', (0,),
        np.array([[0.,0.,0.]]), np.array([[0.,0.,1.]]), np.array([1]), np.array([.002]), 'test')
    class Cap:
        def query(self, points):
            return points, torch.tensor([[0.,0.,-1.]]).expand_as(points), torch.zeros(len(points))
    class Side:
        def query(self, points):
            return points+torch.tensor([.03,0.,0.]), torch.tensor([[-1.,0.,0.]]).expand_as(points), torch.zeros(len(points))
    args = ([patch],torch.zeros(1,3),torch.eye(3)[None],Cap(),TemplateMPPISettings())
    assert evaluate_patches(*args)['coverage'] == 1.
    restricted = evaluate_patches(*args,side_bodies=[patch.body],side_audit=Side())
    assert restricted['coverage'] == 0.
    assert restricted['object_surface_regions'] == {patch.body:'cylinder_side'}
    assert restricted['patches'][0]['witness']['hand_point'] == [0.,0.,0.]


def test_attaching_metric_preserves_cost_force_slots_and_soft_acceptance():
    original = dict(coverage=.1, cost=123., cost_parts={'distance': 3},
        fitted=[False, False], selected_slots=[0], nominal_preloads_n=[2, 2],
        response_contact_forces_n=[[[1, 2, 3]]], scenario_residuals=[[1, 2, 3]],
        projected_closure=True, soft_contact_success=True, geometry_success=False)
    before = copy.deepcopy(original)
    report = dict(coverage=1., required_patch_count=1, covered_patch_count=1)
    updated = attach_patch_coverage(original, report)
    assert original == before
    for key in set(original)-{'coverage', 'geometry_success'}:
        assert updated[key] == original[key]
    assert updated['template_point_coverage'] == .1
    assert updated['geometry_success'] is True
    assert attach_patch_coverage(updated, report)['template_point_coverage'] == .1


@pytest.fixture(scope='module')
def real_template():
    from retarget.task.grasp.dexonomy import load_template, map_contacts
    model = mujoco.MjModel.from_xml_path(str(ROOT/'data/grasp/assets/wuji2/right.xml'))
    shadow, _, contacts, annotation = load_template(ROOT/'data/grasp/templates', '24_Tip_Pinch')
    bodies, points, normals, mapping = map_contacts(shadow, contacts, model)
    names = tuple(model.body(int(i)).name for i in bodies)
    template = TemplateContacts(names, points, normals, np.zeros(20), (names,),
                               dict(mapping=mapping, annotation=str(annotation)))
    return model, template


def test_patch_identity_follows_fixed_wuji_surface_and_immutable_local_coordinates(real_template):
    model, template = real_template
    before = template.to_dict()
    patches = build_patches(model, template)
    # The raw Shadow thumb tip mapped to a Wuji lateral surface. Reporting
    # must describe the patch containing that fixed point, without remapping.
    assert [p.label for p in patches] == ['index.distal.tip', 'thumb.distal.lateral_positive']
    assert all(len(p.local_points) and np.min(p.clearance_m) >= .001 for p in patches)
    assert template.to_dict() == before
    repeated = TemplateContacts((template.body_names[0],)*2,
        np.repeat(template.local_points[:1], 2, axis=0), np.repeat(template.local_normals[:1], 2, axis=0),
        template.finger_qpos, ((template.body_names[0],),),
        dict(mapping=[template.to_dict()['provenance']['mapping'][0]]*2,
             annotation=template.provenance['annotation']))
    assert len(build_patches(model, repeated)) == 1
    assert build_patches(model, repeated)[0].slots == (0, 1)


def test_diagnostic_does_not_invoke_force_solver_or_mutate_candidate_parts(real_template, tmp_path, monkeypatch):
    from retarget.env.grasp.template_mppi import TemplatePoseObjective
    _, template = real_template
    path = tmp_path/'object.obj'
    trimesh.creation.icosphere(radius=.035).export(path)
    objective = TemplatePoseObjective(path, device='cpu')
    q = objective.cpu.qpos0.copy()
    objective.bind(template, q)
    sentinel = {'coverage': torch.tensor([.125]), 'closure': torch.tensor([123.])}
    objective.parts = sentinel
    def forbidden(*args, **kwargs):
        pytest.fail('Patch reporting must not evaluate MPPI candidates or solve forces')
    monkeypatch.setattr(objective, 'evaluate', forbidden)
    monkeypatch.setattr(objective.closure_solver, 'solve', forbidden)
    report = objective.patch_diagnostics(q)
    assert report['required_patch_count'] == 2
    assert objective.parts is sentinel
    assert sentinel['coverage'].item() == .125


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_patch_fk_and_projection_agree_with_cpu_final_diagnostic(real_template):
    model, template = real_template
    mesh = trimesh.creation.icosphere(radius=.035)
    surface = ObjectMeshSurface(mesh.vertices, mesh.faces)
    q = model.qpos0.copy()
    q[:6] = [.1, -.03, .02, .2, -.1, .1]
    q[6:] = np.clip(np.linspace(.1, .5, 20), model.jnt_range[6:, 0], model.jnt_range[6:, 1])
    settings = TemplateMPPISettings()
    a = diagnose_patch_coverage(model, template, q, GraspAudit(surface, 'cpu'), settings, device='cpu')
    b = diagnose_patch_coverage(model, template, q, GraspAudit(surface, 'cuda:0'), settings)
    assert a['coverage'] == b['coverage']
    for left, right in zip(a['patches'], b['patches']):
        np.testing.assert_allclose(left['witness']['hand_point'], right['witness']['hand_point'], atol=1.e-6)
        assert left['witness']['signed_distance_m'] == pytest.approx(right['witness']['signed_distance_m'], abs=1.e-6)
