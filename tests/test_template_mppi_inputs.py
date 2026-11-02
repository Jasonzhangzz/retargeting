"""Direct original-template identity and root-only initializer regressions."""

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
import torch
import trimesh
import warp as wp

wp.config.kernel_cache_dir = "/tmp/retarget-template-mppi-warp"

from retarget.config.run import ROOT
from retarget.env.geometry import ObjectMeshSurface
from retarget.env.grasp.hand import hand_tree, query_path
from retarget.env.grasp.inputs import StaticHand
from retarget.env.grasp import template_mppi_inputs as inputs
from retarget.task.grasp.dexonomy import load_template, map_contacts


CHECKOUT = ROOT / "data/grasp/templates"


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_string(ET.tostring(hand_tree("wuji2"), encoding="unicode"))


def test_immutable_slots_preserve_multiple_contacts_on_the_same_body():
    points = np.array([[0., 0., 0.], [.01, 0., 0.]])
    template = inputs.TemplateContacts(("palm", "palm"), points, [[0., 1., 0.]] * 2,
                                       np.zeros(20), (("palm",),), {"nested": {"value": [1]}})
    points[0, 0] = 1.
    assert template.local_points[0, 0] == 0.
    assert len(template.body_names) == 2
    with pytest.raises(ValueError):
        template.local_points[0, 0] = 1.
    with pytest.raises(ValueError):
        template.local_points.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        template.body_names = ("other",)
    with pytest.raises(TypeError):
        template.provenance["nested"]["value"] = [2]
    exported = template.to_dict()
    assert json.loads(json.dumps(exported))["body_names"] == ["palm", "palm"]
    exported["local_points"][0][0] = 10.
    exported["provenance"]["nested"]["value"][0] = 10
    assert template.local_points[0, 0] == 0.
    assert template.provenance["nested"]["value"] == (1,)


@pytest.mark.parametrize('name,count', [('6_Prismatic_4_Finger',5),('7_Prismatic_3_Finger',4)])
def test_solver_pad_centers_are_fixed_surface_queries_with_original_provenance(model,name,count):
    from retarget.env.grasp.contact_regions import region_coordinates
    raw = inputs.load_template_contacts(name,CHECKOUT,model,device='cpu')
    template = inputs.load_solver_template_contacts(name,CHECKOUT,model,device='cpu')
    assert len(template.body_names) == count and template.body_names == raw.body_names
    np.testing.assert_array_equal(template.finger_qpos,raw.finger_qpos)
    assert template.required_cbody == raw.required_cbody
    assert template.provenance['policy'] == 'fixed_prismatic_pad_center_v1'
    np.testing.assert_array_equal(template.provenance['original_mapped_contacts']['local_points'],raw.local_points)
    for body,p,n,record in zip(template.body_names,template.local_points,template.local_normals,template.provenance['mapping']):
        u,facing,_ = region_coordinates(body,p,n)
        assert .3 <= u <= .7 and abs(u-.5) < .08 and facing > .9
        with np.load(query_path(model,body)) as pool:
            np.testing.assert_array_equal(p,pool['local_pos'][record['query_index']])
    with pytest.raises(ValueError):template.local_points[0,0] = 0.


def test_solver_policy_leaves_type8_and_direct_annotation_loader_unchanged(model):
    name='8_Prismatic_2_Finger'
    raw=inputs.load_template_contacts(name,CHECKOUT,model,device='cpu')
    solver=inputs.load_solver_template_contacts(name,CHECKOUT,model,device='cpu')
    assert raw.to_dict() == solver.to_dict()


@pytest.mark.parametrize("name,count", [("1_Large_Diameter", 16), ("6_Prismatic_4_Finger", 5),
                                        ("11_Power_Sphere", 10)])
def test_original_annotations_are_the_exact_contact_source(model, name, count):
    template = inputs.load_template_contacts(name, CHECKOUT, model, device="cpu")
    shadow, _, contacts, _ = load_template(CHECKOUT, name)
    body, points, normals, mapping = map_contacts(shadow, contacts, model)
    assert len(template.body_names) == count == len(contacts)
    assert template.body_names == tuple(model.body(int(b)).name for b in body)
    np.testing.assert_array_equal(template.local_points, points)
    np.testing.assert_array_equal(template.local_normals, normals)
    for point, normal, record in zip(template.local_points, template.local_normals, mapping):
        with np.load(query_path(model, record["wuji_body"])) as query:
            np.testing.assert_array_equal(point, query["local_pos"][record["query_index"]])
            expected = query["local_normal"][record["query_index"]]
            np.testing.assert_allclose(normal, expected / np.linalg.norm(expected), atol=1.e-15)
    assert template.provenance["primitive_contact_adaptations"] is False
    assert template.provenance["region_adaptations"] is False
    if name == "11_Power_Sphere":
        assert "right_palm_link" not in template.body_names
    if name == "1_Large_Diameter":
        assert template.body_names.count("right_palm_link") == 2


def test_checked_pose_is_reused_independently_of_checked_contact_adaptations(model):
    path = ROOT / "data/grasp/assets/wuji2/template_bone_poses.json"
    checked = json.loads(path.read_text())
    for name in ("6_Prismatic_4_Finger", "11_Power_Sphere"):
        row = next(row for row in checked["cases"] if row["name"] == name)
        template = inputs.load_template_contacts(name, CHECKOUT, model, device="cpu")
        np.testing.assert_array_equal(template.finger_qpos, row["finger_qpos"])
        conversion = template.provenance["pose_conversion"]
        assert conversion["method"] == "checked_object_independent_bone_pose"
        assert all(conversion["source_hash_checks"].values())
        assert "mapping" not in row  # Input asset contains bone angles only.
        assert row['conversion']['object_or_contact_objective_used'] is False
        if name == "11_Power_Sphere":
            assert len(template.body_names) == 10  # Original annotation only.


def test_checked_pose_rejects_an_annotation_hash_mismatch(tmp_path, monkeypatch):
    artifact = json.loads((ROOT / "data/grasp/assets/wuji2/template_bone_poses.json").read_text())
    row = next(row for row in artifact["cases"] if row["name"] == "11_Power_Sphere")
    row["annotation_sha256"] = "0" * 64
    folder = tmp_path / "output/template_mapping"
    folder.mkdir(parents=True)
    (folder / "data.json").write_text(json.dumps(artifact))
    monkeypatch.setattr(inputs, "ROOT", tmp_path)
    annotation = CHECKOUT / "assets/hand/shadow/raw_anno/11_Power_Sphere.yaml"
    assert inputs._checked_finger_qpos("11_Power_Sphere", CHECKOUT, annotation) is None


def _hand(model, template):
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=.03)
    surface = ObjectMeshSurface(mesh.vertices, mesh.faces)
    body = np.array([model.body(name).id for name in template.body_names])
    return StaticHand(model, surface, body, template.local_points, template.local_normals)


def test_root_matcher_preserves_fingers_slots_and_rng_without_validity_filter(model, monkeypatch):
    # Exercise the retained sampler, contact frame, Adam matcher and result
    # conversion on CPU; only its native mesh query is replaced by an analytic
    # sphere to make this invariant test independent of CUDA availability.
    native = inputs._root_matcher_modules()

    class SphereEnvironment:
        def __init__(self, device):
            self.device = device

        def load(self, plane, meshes):
            assert plane is None
            assert len(meshes) == 1

        def query_point(self, points):
            normal = torch.nn.functional.normalize(points, dim=-1)
            return .03 * normal, -normal

    monkeypatch.setattr(inputs, "_root_matcher_modules", lambda: (*native[:3], SimpleNamespace(WarpCollisionEnv=SphereEnvironment)))
    template = inputs.load_template_contacts("9_Palmar_Pinch", CHECKOUT, model, device="cpu")
    hand = _hand(model, template)
    hand.set(np.r_[[.03, -.02, .01, .1, .2, .3], template.finger_qpos])
    initial_q = hand.data.qpos.copy()
    initial_local = hand.local.copy()
    previous = np.random.get_state()
    expected_rng = np.random.RandomState()
    expected_rng.set_state(previous)
    with torch.no_grad():
        candidates, report = inputs.initialize_root_candidates(hand, template, seed=17, device="cpu",
                                                               points=3, inplane=2, iterations=3)
    np.testing.assert_array_equal(hand.data.qpos, initial_q)
    np.testing.assert_array_equal(hand.local, initial_local)
    np.testing.assert_array_equal(np.random.random(4), expected_rng.random(4))
    assert candidates.shape == (6, 26)
    assert np.isfinite(candidates).all()
    np.testing.assert_array_equal(candidates[:, 6:], np.broadcast_to(template.finger_qpos, (6, 20)))
    assert report["sampled"] == report["finite_candidates"] == 6
    assert not report["initialization_filter"] and not report["closure_filter"] and not report["qp_filter"]
    repeated, _ = inputs.initialize_root_candidates(hand, template, seed=17, device="cpu",
                                                   points=3, inplane=2, iterations=3)
    np.testing.assert_array_equal(candidates, repeated)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native root matching requires CUDA")
def test_native_root_matcher_fits_original_slots_without_finger_updates(model):
    template = inputs.load_template_contacts("6_Prismatic_4_Finger", CHECKOUT, model)
    hand = _hand(model, template)
    candidates, report = inputs.initialize_root_candidates(hand, template, seed=3, device="cuda:0",
                                                           points=4, inplane=2, iterations=3)
    assert candidates.shape == (8, 26) and np.isfinite(candidates).all()
    np.testing.assert_array_equal(candidates[:, 6:], np.broadcast_to(template.finger_qpos, (8, 20)))
    assert np.asarray(report["loss_dist"]).shape == (8, len(template.body_names))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Canonical batch FK requires CUDA")
def test_canonical_gpu_fk_batches_templates_and_matches_cpu_without_changing_slots(model):
    names = ['1_Large_Diameter', '6_Prismatic_4_Finger', '11_Power_Sphere']
    templates = [inputs.load_template_contacts(name, CHECKOUT, model, device='cpu') for name in names]
    values = inputs.canonical_template_contacts(model, templates, device='cuda:0')
    assert len(values) == 3
    for template, value in zip(templates, values):
        hand = _hand(model, template)
        hand.set(np.r_[np.zeros(6), template.finger_qpos])
        body = hand.bodies
        world = hand.points(body, template.local_points)
        normal = np.einsum('nij,nj->ni', hand.data.xmat[body].reshape(-1, 3, 3), template.local_normals)
        np.testing.assert_allclose(value['contacts'], np.c_[world, normal], atol=6e-8, rtol=1e-7)
        assert value['backend'] == 'torch_cuda_batched_WujiContactKinematics'
        assert value['batch_size'] == 3
        assert value['contacts'].shape == (len(template.body_names), 6)
    assert values[-1]['contacts'].shape == (10, 6)  # Original Type 11 slots.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Canonical batch FK requires CUDA")
def test_gpu_initializer_never_calls_cpu_fk_and_rejects_wrong_precomputed_template(model, monkeypatch):
    template = inputs.load_template_contacts('6_Prismatic_4_Finger', CHECKOUT, model, device='cpu')
    hand = _hand(model, template)
    hand.set(np.r_[[.01, -.02, .03, .1, -.2, .3], template.finger_qpos])
    before = hand.data.qpos.copy(), hand.data.xpos.copy(), hand.data.xmat.copy()

    def forbidden(*args, **kwargs):
        raise AssertionError('Production CUDA initializer must not run CPU FK')

    monkeypatch.setattr(mujoco, 'mj_kinematics', forbidden)
    monkeypatch.setattr(mujoco, 'mj_comPos', forbidden)
    canonical = inputs.canonical_template_contacts(model, [template], device='cuda:0')[0]
    candidates, report = inputs.initialize_root_candidates(hand, template, seed=3, device='cuda:0',
        points=4, inplane=2, iterations=3, canonical_contacts=canonical)
    assert candidates.shape == (8, 26)
    assert report['canonical_fk']['backend'] == 'torch_cuda_batched_WujiContactKinematics'
    for actual, expected in zip((hand.data.qpos, hand.data.xpos, hand.data.xmat), before):
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match='do not match'):
        inputs.initialize_root_candidates(hand, template, seed=3, device='cuda:0', points=4, inplane=2,
            iterations=3, canonical_contacts=dict(canonical, template_signature='invalid'))


def test_packaged_checked_bone_poses_are_inputs_without_output_cache(monkeypatch):
    import json
    from retarget.env.grasp import template_mppi_inputs as module
    data=json.loads((ROOT/'data/grasp/assets/wuji2/template_bone_poses.json').read_text())
    assert len(data['cases'])==31
    for row in data['cases']:
        annotation=CHECKOUT/'assets/hand/shadow/raw_anno'/f"{row['name']}.yaml"
        result=module._checked_finger_qpos(row['name'],CHECKOUT,annotation)
        assert result is not None
        np.testing.assert_array_equal(result[0],row['finger_qpos'])
        assert result[1]['source_hash_checks']==dict(annotation=True,shadow_xml=True,wuji_xml=True)


def test_fixed_reference_template_preserves_exact_slots_and_rejects_tampering(tmp_path,model):
    import json
    from retarget.env.grasp.template_mppi_inputs import load_fixed_reference_template
    from retarget.env.grasp.template_mppi_reference import template_signature
    t=inputs.load_template_contacts('11_Power_Sphere',CHECKOUT,model,device='cpu')
    ref=tmp_path/'reference.json';ref.write_text(json.dumps(dict(template='11_Power_Sphere',template_signature=template_signature(t))))
    source=ref.with_name('template_contacts.json');source.write_text(json.dumps(t.to_dict()))
    actual=load_fixed_reference_template(ref,CHECKOUT,model,'11_Power_Sphere')
    assert actual.to_dict()==t.to_dict() and len(actual.body_names)==10
    record=t.to_dict();record['local_points'][0][0]+=.001;source.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='signature'):
        load_fixed_reference_template(ref,CHECKOUT,model,'11_Power_Sphere')
