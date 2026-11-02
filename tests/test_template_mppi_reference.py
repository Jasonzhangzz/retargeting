"""Canonical object fitting: frames, original loss, exact 1x and reuse."""
from dataclasses import replace
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from retarget.config.run import ROOT, load
from retarget.env.grasp import template_mppi_inputs as inputs
from retarget.env.grasp import template_mppi_reference as reference
from retarget.env.grasp.inputs import build_scene, StaticHand
from retarget.env.grasp.primitives import canonical_specification, jobs, primitive_mesh
from retarget.env.grasp.template_mppi_runner import validate


def _args():
    return validate(load(ROOT / "configs/grasp_template_mppi_all.yaml"))


def test_exact_1x_does_not_depend_on_grid_and_respects_physical_policy():
    args = _args()
    sphere = canonical_specification(26, "sphere", args)
    assert sphere["size_scale"] == sphere["radial_scale"] == 1.
    assert sphere["radius_m"] == pytest.approx(.030)
    assert sphere["height_m"] == pytest.approx(.060)
    # 1x is intentionally absent from the actual Type 26 grid.
    from retarget.env.grasp.primitives import size_specifications
    assert 1. not in [r["size_scale"] for r in size_specifications(26, "sphere")]
    cylinder = canonical_specification(1, "cylinder", args)
    assert cylinder["height_m"] == pytest.approx(.18)
    plate = canonical_specification(16, "plate", args)
    assert plate["radius_m"] == pytest.approx(.025 * 1.5)
    assert plate["height_m"] == pytest.approx(.003)
    assert plate["size_scale"] == plate["axial_scale"] == 1.
    assert plate["radial_scale"] == 1.5
    args.primitive_grid_explicit = True
    args.radius, args.height = .021, .091
    args.radius_scales = [.5, 1.5]
    explicit = canonical_specification(1, "cylinder", args)
    assert explicit["radius_m"] == .021 and explicit["height_m"] == .091


@pytest.mark.parametrize("number", [18, 30])
def test_corrected_disks_use_broad_1x_reference(number):
    p = canonical_specification(number, "disk", _args())
    assert p["radius_m"] == (.08 if number == 18 else .05) and p["height_m"] == .008
    assert p["size_scale"] == 1. and p["size_policy_version"] == 5


def test_selection_matches_original_all_slot_objective_and_rejects_invalid():
    q = np.zeros((3, 26))
    q[:, 0] = [.1, .2, .3]
    report = dict(loss_dist=[[.001, 0.], [0., 0.], [0., 0.]],
                  loss_normal=[[0., 0.], [.1, .1], [np.nan, 0.]])
    selected, selection = reference.select_match(q, report)
    np.testing.assert_array_equal(selected, q[1])
    assert selection["candidate_costs"][:2] == [.5, .1]
    with pytest.raises(FloatingPointError):
        reference.select_match(q, dict(loss_dist=np.full((3, 2), np.inf), loss_normal=np.zeros((3, 2))))
    with pytest.raises(ValueError, match="shape"):
        reference.select_match(q, dict(loss_dist=[0., 0., 0.], loss_normal=[0., 0., 0.]))


def test_saved_inverse_transform_matches_mujoco_wrist_and_contact_frames(tmp_path, monkeypatch):
    mesh = primitive_mesh(tmp_path / "sphere.obj", "sphere", .035, .07)
    model, surface, _, _ = build_scene(mesh, 1., "wuji2")
    template = inputs.load_template_contacts("11_Power_Sphere", _args().dexonomy_root, model, device="cpu")
    bodies = np.array([model.body(name).id for name in template.body_names])
    hand = StaticHand(model, surface, bodies, template.local_points, template.local_normals)
    q = np.r_[[.031, -.027, .013, .2, -.5, .7], template.finger_qpos]
    points = np.arange(30).reshape(10, 3) * .0001
    inward = np.tile([0., -1., 0.], (10, 1))

    def matcher(h, t, **kwargs):
        assert h is hand and t is template
        return q[None], dict(loss_dist=np.zeros((1, 10)), loss_normal=np.zeros((1, 10)),
                            obj_cp_w=points[None], obj_cn_w=inward[None])

    monkeypatch.setattr(inputs, "initialize_root_candidates", matcher)
    path = reference.fit_reference(hand, template, name="11_Power_Sphere", directory=tmp_path / "reference",
        primitive=None, seed=0, device="cpu", initialization={})
    actual, metadata = reference.load_reference(path, template, "11_Power_Sphere")
    np.testing.assert_array_equal(actual, q)
    data = json.loads(path.read_text())
    hand.set(q)
    wrist = model.body("grasp_wrist_rz").id
    T = np.eye(4)
    T[:3, :3] = hand.data.xmat[wrist].reshape(3, 3)
    T[:3, 3] = hand.data.xpos[wrist]
    np.testing.assert_allclose(data["wrist_in_object"], T, atol=1e-15)
    np.testing.assert_allclose(T @ np.asarray(data["object_in_wrist"]), np.eye(4), atol=1e-15)
    # Fixed canonical template world points equal actual FK expressed in wrist.
    world = hand.points(bodies, template.local_points)
    in_wrist = (world - T[:3, 3]) @ T[:3, :3]
    hand.set(np.r_[np.zeros(6), template.finger_qpos])
    np.testing.assert_allclose(in_wrist, hand.points(bodies, template.local_points), atol=1e-15)

    object_to_wrist = np.asarray(data["object_in_wrist"])
    np.testing.assert_allclose(data["object_points_in_wrist"], points @ object_to_wrist[:3, :3].T + object_to_wrist[:3, 3])
    np.testing.assert_allclose(data["object_outward_normals"], -inward)
    assert len(data["object_points"]) == 10
    assert metadata["translation_scaled"] is False
    # Identity and local coordinates are required to reuse a reference.
    changed = replace(template, local_points=template.local_points + .001,
                      provenance=template.to_dict()["provenance"])
    with pytest.raises(ValueError, match="does not match"):
        reference.load_reference(path, changed, "11_Power_Sphere")
    data["object_in_wrist"][0][3] += .001
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Inconsistent"):
        reference.load_reference(path, template, "11_Power_Sphere")


def test_collision_aware_canonical_selection_preserves_template_and_reuses_matcher(tmp_path, monkeypatch):
    import torch
    from retarget.env.grasp import template_mppi
    mesh = primitive_mesh(tmp_path/'sphere.obj', 'sphere', .035, .07)
    model, surface, _, _ = build_scene(mesh, 1., 'wuji2')
    template = inputs.load_template_contacts('11_Power_Sphere', _args().dexonomy_root, model, device='cpu')
    bodies = np.array([model.body(name).id for name in template.body_names])
    hand = StaticHand(model, surface, bodies, template.local_points, template.local_normals)
    qs = np.tile(np.r_[np.zeros(6), template.finger_qpos], (2, 1))
    qs[1, 0] = .03
    before = template.to_dict()
    def matcher(*args, **kwargs):
        assert 'selection' not in kwargs
        return qs, dict(loss_dist=np.zeros((2, 10)), loss_normal=np.array([[0.]*10, [.1]*10]),
            obj_cp_w=np.zeros((2, 10, 3)), obj_cn_w=np.tile([0., -1., 0.], (2, 10, 1)))
    class GeometryOnly:
        def __init__(self, *args, **kwargs):
            pass
        def bind(self, actual, initial):
            assert actual is template
        def set_batch_size(self, count):
            pass
        def evaluate_initialization(self, controls):
            np.testing.assert_array_equal(controls[:, 0, 6:], qs[:, 6:].astype(np.float32))
            return torch.tensor([100., 1.])
    monkeypatch.setattr(inputs, 'initialize_root_candidates', matcher)
    monkeypatch.setattr(template_mppi, 'TemplatePoseObjective', GeometryOnly)
    path = reference.fit_reference(hand, template, name='11_Power_Sphere', directory=tmp_path/'reference',
        primitive=None, seed=0, device='cpu', initialization=dict(selection='collision_aware'))
    record = json.loads(path.read_text())
    assert record['selected'] == 1 and record['matcher_selected'] == 0
    assert record['closure_solver_used_for_selection'] is False
    np.testing.assert_array_equal(record['initial_qpos'], qs[1])
    assert template.to_dict() == before


def test_solved_canonical_grasp_replaces_root_match_contacts_and_seed_only(tmp_path, monkeypatch):
    from retarget.env.grasp.template_mppi_size_ik import SizeIKSettings
    from retarget.common.results import write_json
    mesh = primitive_mesh(tmp_path/'sphere.obj', 'sphere', .035, .07)
    model, surface, _, _ = build_scene(mesh, 1., 'wuji2')
    args = _args()
    template = inputs.load_template_contacts('11_Power_Sphere', args.dexonomy_root, model, device='cpu')
    hand = StaticHand(model, surface, [model.body(n).id for n in template.body_names], template.local_points, template.local_normals)
    raw = np.r_[np.zeros(6), template.finger_qpos]
    solved = raw.copy()
    solved[0], solved[6] = .02, raw[6]+.1
    original = template.to_dict()
    canonical = dict(kind='sphere', radius_m=.035, height_m=.07)
    def matcher(*a, **kw):
        return raw[None],dict(loss_dist=np.zeros((1,10)),loss_normal=np.zeros((1,10)),
            obj_cp_w=np.zeros((1,10,3)),obj_cn_w=np.tile([0.,-1.,0.],(1,10,1)))
    monkeypatch.setattr(inputs,'initialize_root_candidates',matcher)
    path = reference.fit_reference(hand,template,name='11_Power_Sphere',directory=tmp_path/'reference',primitive=canonical,seed=0,device='cpu',initialization={})
    directory = tmp_path/'cache'
    points = np.tile([.035,0.,0.],(10,1))
    write_json(directory/'11_Power_Sphere.json',dict(template_signature=reference.template_signature(template),
        primitive=canonical,qpos=solved,object_points=points,object_outward_normals=np.tile([1.,0.,0.],(10,1)),source_sha256={}))
    args.settings.size_ik = SizeIKSettings(canonical_grasp_directory=str(directory))
    reference.attach_canonical_grasp(path,template,args,'11_Power_Sphere','cpu',0)
    q,metadata = reference.load_reference(path,template,'11_Power_Sphere')
    record = json.loads(path.read_text())
    np.testing.assert_array_equal(q,solved)
    np.testing.assert_array_equal(record['root_match_initial_qpos'],raw)
    np.testing.assert_array_equal(record['object_points'],points)
    np.testing.assert_array_equal(record['matcher_object_points'],np.zeros((10,3)))
    assert metadata['object_contact_reference']=='solved_1x_grasp'
    assert metadata['template_finger_angles_preserved'] is False
    assert template.to_dict()==original


def test_parallel_preparation_fits_once_per_template_family_not_per_size(tmp_path, monkeypatch):
    import torch
    import warp as wp
    config = load(ROOT / "configs/grasp_template_mppi_all.yaml")
    config.task["templates"] = [1, 11, 26]
    tasks, _ = jobs(tmp_path, validate(config).templates, validate(config))
    seen = []

    def prepare(mesh, name, primitive, args, output, **kwargs):
        seen.append((name, primitive["kind"]))
        return tmp_path / f"{name}.json"

    monkeypatch.setattr(reference, "prepare_reference", prepare)
    monkeypatch.setattr(reference, "prepare_canonical_templates", lambda *args, **kwargs: ({}, {}))
    monkeypatch.setattr(torch, "set_num_threads", lambda n: None)
    monkeypatch.setattr(wp, "init", lambda: None)
    monkeypatch.setattr(wp, "set_device", lambda d: None)
    paths = reference.prepare_all_references(config, tmp_path, tasks)
    assert len(tasks) == 15 and len(seen) == 3
    for name in validate(config).templates:
        assert len({path for path, task in zip(paths, tasks) if task[1] == name}) == 1
    assert json.loads((tmp_path / "template_initialization/manifest.json").read_text())["fits"] == 3


def test_serial_sizes_and_worker_reuse_exact_same_pose_without_rematching(tmp_path, monkeypatch):
    import torch
    import warp as wp
    from retarget.env.grasp import template_mppi as objective_module
    from retarget.env.grasp import template_mppi_runner as runner

    config = load(ROOT / "configs/grasp_template_mppi_all.yaml")
    config.environment["template_mppi"].pop("parallel")
    config.task["templates"] = [26]
    # No 1x case in this grid; the reference must still be at exactly 30mm.
    config.task["radius_scales"] = [.75, 1.25]
    config.task["height_scales"] = [1.]
    calls, starts = [], []
    q = None

    def matcher(hand, template, **kwargs):
        nonlocal q
        calls.append(hand.surface.triangles.copy())
        np.testing.assert_allclose(np.ptp(hand.surface.triangles.reshape(-1, 3), axis=0),
                                   np.ptp(primitive_geometry(26), axis=0), atol=1e-8)
        q = np.r_[[.03, -.02, .01, .4, -.3, .6], template.finger_qpos]
        n = len(template.body_names)
        return q[None], dict(loss_dist=np.zeros((1, n)), loss_normal=np.zeros((1, n)),
                            obj_cp_w=np.zeros((1, n, 3)), obj_cn_w=np.tile([0., -1., 0.], (1, n, 1)))

    def primitive_geometry(type_id):
        from retarget.env.grasp.primitives import primitive_geometry as geometry
        return geometry("sphere", .03, .06)[0]

    class Objective:
        xml = "<mujoco/>"

        def __init__(self, *args, **kwargs):
            pass

        def bind(self, template, initial):
            np.testing.assert_array_equal(initial, q)

        def optimize(self, initial, *args):
            starts.append(initial.copy())
            return initial, dict(trajectory=initial[None], initial_cost=0., final_cost=0.)

        def diagnostics(self, q):
            return dict(geometry_success=False, soft_contact_success=False)

    monkeypatch.setattr(inputs, "initialize_root_candidates", matcher)
    monkeypatch.setattr(objective_module, "TemplatePoseObjective", Objective)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(wp, "init", lambda: None)
    monkeypatch.setattr(wp, "set_device", lambda d: None)
    output = tmp_path / "serial"
    entries = runner.run(config, output)
    assert len(entries) == len(starts) == 2 and len(calls) == 1
    np.testing.assert_array_equal(starts[0], starts[1])
    initializations = [json.loads((output / entry["summary"]).with_name("initialization.json").read_text())
                       for entry in entries]
    assert initializations[0]["reference_sha256"] == initializations[1]["reference_sha256"]
    canonical = json.loads(open(initializations[0]["reference"]).read())
    assert canonical["canonical_primitive"]["radius_m"] == .03
    # Parallel workers receive a path and must never run a second matcher on
    # their size-specific mesh, which is represented as a custom object.
    child = config
    child.task["object"] = str(next((output / "meshes").glob("*.obj")))
    def forbidden(*args, **kwargs):
        pytest.fail("Worker must reuse its supplied 1x reference")
    monkeypatch.setattr(reference, "prepare_reference", forbidden)
    worker = runner.run(child, tmp_path / "worker", initialization_reference=initializations[0]["reference"])
    assert len(worker) == 1 and len(calls) == 1
    np.testing.assert_array_equal(starts[-1], starts[0])


def test_changed_contact_template_recomputes_instead_of_reusing_tip_cache(tmp_path, monkeypatch):
    from retarget.env.grasp import template_mppi
    from retarget.env.grasp.template_mppi_size_ik import SizeIKSettings
    mesh = primitive_mesh(tmp_path/'sphere.obj','sphere',.035,.07)
    model,surface,_,_ = build_scene(mesh,1.,'wuji2')
    args = _args()
    original = inputs.load_template_contacts('11_Power_Sphere',args.dexonomy_root,model,device='cpu')
    # Isolate cache invalidation from the separately tested geometric adapter.
    template = replace(original,provenance=dict(original.to_dict()['provenance'],policy='fixed_prismatic_pad_center_v1'))
    q = np.r_[np.zeros(6),template.finger_qpos]
    hand = StaticHand(model,surface,[model.body(n).id for n in template.body_names],template.local_points,template.local_normals)
    def matcher(*a,**kw):
        return q[None],dict(loss_dist=np.zeros((1,10)),loss_normal=np.zeros((1,10)),
            obj_cp_w=np.zeros((1,10,3)),obj_cn_w=np.tile([0.,-1.,0.],(1,10,1)))
    monkeypatch.setattr(inputs,'initialize_root_candidates',matcher)
    path = reference.fit_reference(hand,template,name='11_Power_Sphere',directory=tmp_path/'reference',
        primitive=dict(kind='sphere',radius_m=.035,height_m=.07),seed=0,device='cpu',initialization={})
    directory=tmp_path/'old_cache';directory.mkdir()
    (directory/'11_Power_Sphere.json').write_text(json.dumps(dict(template_signature='0'*64)))
    args.settings.size_ik=SizeIKSettings(canonical_grasp_directory=str(directory))
    called=[]
    class FreshSolve:
        def __init__(self,*a,**kw):pass
        def bind(self,t,initial):assert t is template
        def configure_object_regions(self,*a):pass
        def initialize_size(self,ref,primitive,initial,*a):
            called.append(initial.copy());return initial,{}
        def optimize(self,initial,*a):return initial,{}
        def final_diagnostics(self,q):
            return dict(projected_points=np.tile([.035,0.,0.],(10,1)),object_normals=np.tile([1.,0.,0.],(10,1)))
    monkeypatch.setattr(template_mppi,'TemplatePoseObjective',FreshSolve)
    reference.attach_canonical_grasp(path,template,args,'11_Power_Sphere','cpu',0)
    record=json.loads(path.read_text());assert len(called)==1
    assert record['canonical_grasp_reference']['from_cache'] is False
    assert 'superseded_cache' in record['canonical_grasp_reference']
    np.testing.assert_array_equal(record['canonical_grasp_qpos'],q)
