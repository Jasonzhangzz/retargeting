"""Generated Hand 2.2 assets retain source mechanics and query coordinates."""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
import torch

from retarget.config.run import ROOT
from retarget.env.grasp.hand import HAND_BODIES, model_hand, query_path
from scripts.setup_grasp_assets import sample_surface


@pytest.fixture
def hand():
    xml = ROOT / "data/grasp/assets/wuji2/right.xml"
    if not xml.is_file():
        pytest.skip("Run scripts/setup_grasp_assets.py to prepare official Hand 2.2")
    return mujoco.MjModel.from_xml_path(str(xml))


def test_wuji2_adapter_preserves_original_finger_mechanics(hand):
    base = ROOT / "data/grasp/assets/wuji2"
    source = ET.parse(base / "upstream/mjcf/right.xml").getroot()
    adapted = ET.parse(base / "right.xml").getroot()
    assert source.get("model") == "wujihand2_2-right"
    assert (hand.nq, hand.nv, hand.nu) == (26, 26, 26)
    assert model_hand(hand) == "wuji2"
    original_joints = {j.get("name"): j.attrib for j in source.findall(".//joint") if j.get("name")}
    actual_joints = {j.get("name"): j.attrib for j in adapted.findall(".//joint") if j.get("name")}
    for name, attributes in original_joints.items():
        assert actual_joints[name] == attributes
    assert [a.attrib for a in adapted.findall("actuator/position")][6:] == [a.attrib for a in source.findall("actuator/position")]
    np.testing.assert_allclose(hand.body_ipos[hand.jnt_bodyid[:6]], 0)
    assert np.all(hand.actuator_forcelimited)
    assert np.all(np.isfinite(hand.actuator_forcerange))
    for finger, name in enumerate(("thumb", "index", "middle", "ring", "pinky"), 1):
        assert hand.site(f"right_{name}_tip").bodyid == hand.body(f"right_finger{finger}_link4").id


def test_surface_points_have_finite_unit_normals_and_zero_radii(hand):
    for name in HAND_BODIES:
        with np.load(query_path(hand, name)) as samples:
            points, normals = samples["local_pos"], samples["local_normal"]
            assert points.shape == normals.shape and points.shape[1] == 3
            assert len(points) >= 768
            assert np.isfinite(points).all()
            np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=2e-7)
            np.testing.assert_array_equal(samples["radius"], np.zeros(len(points)))


def test_template_matching_configuration_is_bundled(hand):
    import yaml

    path = ROOT / "data/grasp/templates/dexonomy/config/op/init.yaml"
    loss = yaml.safe_load(path.read_text())["filter"]["loss"]
    assert loss["pos_thre"] == pytest.approx(0.001)
    assert loss["normal_thre"] == pytest.approx(0.3)


def test_query_component_metadata_identifies_the_actual_source_triangles(hand):
    data = mujoco.MjData(hand)
    mujoco.mj_kinematics(hand, data)
    for name in HAND_BODIES:
        body = hand.body(name).id
        with np.load(query_path(hand, name)) as raw:
            position = raw["local_pos"].astype(float)
            geom = raw["source_geom"]
            face = raw["source_face"]
            mesh = hand.geom_dataid[geom]
            owner = hand.geom_bodyid[geom]
            assert raw["source_body"].tolist() == [hand.body(int(i)).name for i in owner]
            assert raw["source_mesh"].tolist() == [hand.mesh(int(i)).name for i in mesh]
            assert ((face >= 0) & (face < hand.mesh_facenum[mesh])).all()
            ids = hand.mesh_face[hand.mesh_faceadr[mesh]+face] + hand.mesh_vertadr[mesh,None]
            triangles = hand.mesh_vert[ids]
            triangles = np.einsum("nij,nkj->nki",data.geom_xmat[geom].reshape(-1,3,3),triangles) + data.geom_xpos[geom,None]
            points = position @ data.xmat[body].reshape(3,3).T + data.xpos[body]
            a,b,c = triangles.transpose(1,0,2)
            u,v,delta = b-a,c-a,points-a
            normal = np.cross(u,v)
            normal /= np.linalg.norm(normal,axis=1,keepdims=True)
            np.testing.assert_allclose((delta*normal).sum(1),0,atol=2e-8)
            # The source face contains the sample, including triangle edges.
            uu,vv,uv = (u*u).sum(1),(v*v).sum(1),(u*v).sum(1)
            du,dv = (delta*u).sum(1),(delta*v).sum(1)
            denominator=uu*vv-uv*uv
            x=(du*vv-dv*uv)/denominator
            y=(dv*uu-du*uv)/denominator
            assert ((x>=-2e-4)&(y>=-2e-4)&(x+y<=1+2e-4)).all()


def test_sampling_is_deterministic_and_on_triangles():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    faces = np.array([[0, 1, 2]])
    p, n = sample_surface(vertices, faces, 128, 3)
    p2, n2 = sample_surface(vertices, faces, 128, 3)
    np.testing.assert_array_equal(p, p2)
    np.testing.assert_array_equal(n, n2)
    assert (p >= 0).all() and (p.sum(1) <= 1).all()
    np.testing.assert_array_equal(n, np.tile([0, 0, 1], (128, 1)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required by template FK")
def test_template_fk_matches_cpu_for_adapted_official_model(hand):
    from retarget.env.grasp.template import WujiContactKinematics

    bodies = np.array([hand.body(name).id for name in HAND_BODIES])
    local = np.zeros((len(bodies), 3))
    fk = WujiContactKinematics(hand, bodies, "cuda:0")
    q = np.r_[0.03, -0.02, 0.01, 0.1, -0.2, 0.15, np.full(20, 0.2)]
    positions, rotations = fk.frames(fk.tensor(q))
    data = mujoco.MjData(hand)
    data.qpos[:] = q
    mujoco.mj_kinematics(hand, data)
    np.testing.assert_allclose(positions.cpu().numpy(), data.xpos[bodies], atol=1e-10)
    np.testing.assert_allclose(rotations.cpu().numpy(), data.xmat[bodies].reshape(-1, 3, 3), atol=1e-10)


# Additional checks for the same subsystem.
from types import SimpleNamespace

import numpy as np
import pytest

from retarget.env.grasp.physics import grasp_physics


def test_standalone_physics_does_not_require_a_dexycb_scene():
    values = grasp_physics(SimpleNamespace(physics_reference=None, friction=0.7))
    assert values["friction"] == [0.7, 0.005, 0.0001]
    assert values["condim"] == 3
    np.testing.assert_equal(values["solref"], [0.02, 1.0])


@pytest.mark.parametrize("friction", [-1.0, float("nan")])
def test_standalone_physics_rejects_invalid_friction(friction):
    with pytest.raises(ValueError, match="friction"):
        grasp_physics(SimpleNamespace(physics_reference=None, friction=friction))


def test_disabled_wrist_prior_and_physics_reference_need_no_files(tmp_path, monkeypatch):
    from retarget.env.grasp import runner, hand

    template = tmp_path / "templates/assets/hand/shadow/raw_anno/test.yaml"
    template.parent.mkdir(parents=True)
    template.touch()
    (template.parent.parent / "right.xml").touch()
    (template.parent.parent / "keypoint.yaml").touch()
    initializer_config = tmp_path / "templates/dexonomy/config/op/init.yaml"
    initializer_config.parent.mkdir(parents=True)
    initializer_config.touch()
    model = tmp_path / "model.yaml"
    model.touch()
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "right.xml").touch()
    (assets / "qp_right").mkdir()
    for name in hand.HAND_BODIES:
        (assets / "qp_right" / f"{name}.npz").touch()
    args = SimpleNamespace(
        hand_model="wuji2", objective="dex_tracking", starts=1,
        init_points=1, init_inplane=1, init_iterations=1,
        radius_scales=[1.0], height_scales=[1.0], templates=["test"],
        dexonomy_root=tmp_path / "templates", solver_config=model,
        physics_reference=None, wrist_pose_prior=False,
        wrist_pose_reference=tmp_path / "absent.json",
    )
    monkeypatch.setattr(runner, "arguments", lambda _: args)
    monkeypatch.setattr(hand, "asset_root", lambda _: assets)
    config = SimpleNamespace(solver=SimpleNamespace(gradient_steps=0), task={})
    assert runner.validate(config) is args
    args.wrist_pose_prior = True
    with pytest.raises(ValueError, match="absent.json"):
        runner.validate(config)
