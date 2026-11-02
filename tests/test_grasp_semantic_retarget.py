"""Semantic contact regions remain fixed while fitting the target hand."""

from types import SimpleNamespace
import numpy as np
import pytest
import torch

from retarget.env.grasp.template import (
    matched_patch_targets, resize_contact_opening,
    semantic_query_diagnostics, semantic_surface_mask,
)


class LinearKinematics:
    """Small independent FK stand-in to inspect the immutable target inputs."""
    @staticmethod
    def tensor(value):
        return torch.as_tensor(np.asarray(value).copy(), dtype=torch.float64)

    @staticmethod
    def contacts(q, local, normals):
        return local + q[:3] + torch.tensor([q[6], 0.0, 0.0]), normals.clone()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_wuji_wxyz_body_rotations_match_mujoco_without_scipy_scalar_first():
    import mujoco
    import xml.etree.ElementTree as ET
    from retarget.config.run import ROOT
    from retarget.env.grasp.hand import hand_tree
    from retarget.env.grasp.template import WujiContactKinematics

    if not (ROOT / "data/grasp/assets/wuji2/right.xml").exists():
        pytest.skip("WUJI2 assets required")
    model = mujoco.MjModel.from_xml_string(ET.tostring(hand_tree("wuji2"), encoding="unicode"))
    body = model.body("right_finger1_link1").id
    fk = WujiContactKinematics(model, [body], "cuda:0")
    expected = np.empty((5, 4, 3, 3))
    for finger in range(1, 6):
        for link in range(1, 5):
            body_id = model.body(f"right_finger{finger}_link{link}").id
            mujoco.mju_quat2Mat(expected[finger-1, link-1].reshape(-1), model.body_quat[body_id])
    np.testing.assert_allclose(fk.rotation.cpu().numpy(), expected, rtol=0, atol=1e-14)


def test_matched_target_uses_target_hand_geometry_and_fixed_seed():
    fixed_local = np.array([[0.02, 0.004, 0.01], [-0.03, 0.008, 0.02]])
    fixed_normals = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
    hand = SimpleNamespace(
        bodies=np.array([1, 2]), semantic_contact_local=fixed_local.copy(),
        semantic_contact_normals=fixed_normals.copy(), local=fixed_local.copy(),
    )
    q = np.r_[np.ones(6), 0.01, np.zeros(19)]
    points, normals, _ = matched_patch_targets(hand, LinearKinematics(), q)
    np.testing.assert_allclose(points, fixed_local + [0.01, 0, 0])
    np.testing.assert_allclose(normals, fixed_normals)
    # Surface updates and arbitrary world root transformations do not move
    # the fixed canonical target or change the reference seed's bone shape.
    hand.local[:] = 100
    q[:6] = np.arange(6) * 3
    after, after_n, _ = matched_patch_targets(hand, LinearKinematics(), q)
    np.testing.assert_array_equal(after, points)
    np.testing.assert_array_equal(after_n, normals)
    np.testing.assert_array_equal(hand.semantic_contact_local, fixed_local)


def test_opening_resize_preserves_target_normals_and_input_arrays():
    points = np.array([[-0.01, 0.02, 0.03], [0.01, 0.02, 0.03]])
    normals = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
    output, output_normals, info = resize_contact_opening(points, normals, 0.06)
    assert info["opening_resized"]
    assert output[1, 0] - output[0, 0] == pytest.approx(0.06)
    np.testing.assert_array_equal(output[:, 1:], points[:, 1:])
    np.testing.assert_array_equal(output_normals, normals)
    np.testing.assert_allclose(points[:, 0], [-0.01, 0.01])


def test_each_annotation_keeps_its_own_patch_even_on_the_same_body():
    hand = SimpleNamespace(
        bodies=np.array([7, 7]), semantic_patch_labels=["palmar/base", "palmar/tip"],
        semantic_contact_masks=[np.array([True, True, False, False]),
                                np.array([False, False, True, True])],
    )
    assert semantic_surface_mask(hand, 0, 4).tolist() == [True, True, False, False]
    assert semantic_surface_mask(hand, 1, 4).tolist() == [False, False, True, True]
    info = semantic_query_diagnostics(hand, [0, 3])
    assert info["semantic_patch_violation_count"] == 0
    assert info["semantic_patch_labels"] == hand.semantic_patch_labels
    with pytest.raises(ValueError, match="outside fixed semantic patches"):
        semantic_query_diagnostics(hand, [3, 0])
    with pytest.raises(ValueError, match="disagree"):
        semantic_query_diagnostics(hand, [0])
    hand.semantic_contact_masks[0][:] = False
    with pytest.raises(ValueError, match="empty semantic patch"):
        semantic_surface_mask(hand, 0, 4)


def test_legacy_inputs_do_not_gain_patch_restrictions():
    hand = SimpleNamespace(bodies=np.array([1]))
    np.testing.assert_array_equal(semantic_surface_mask(hand, 0, 3), np.ones(3, bool))
    assert semantic_query_diagnostics(hand, [0]) == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_wuji_fit_keeps_patch_and_uses_matched_layout(monkeypatch):
    import mujoco
    from retarget.env.grasp import template
    from retarget.env.grasp.hand import hand_tree, query_path
    from retarget.env.grasp.inputs import StaticHand
    from retarget.config.run import ROOT

    asset = ROOT / "data/grasp/assets/wuji2/right.xml"
    if not asset.exists():
        pytest.skip("WUJI2 assets required")
    import xml.etree.ElementTree as ET
    model = mujoco.MjModel.from_xml_string(ET.tostring(hand_tree("wuji2"), encoding="unicode"))
    body = model.body("right_finger2_link4").id
    with np.load(query_path(model, body)) as raw:
        points, normals = raw["local_pos"].copy(), raw["local_normal"].copy()
    from retarget.env.grasp.regions import belly_axis
    belly = belly_axis(model, body)
    candidates = np.flatnonzero(normals @ belly > 0.8)
    assert candidates.size
    index = candidates[len(candidates) // 2]
    mask = (normals @ normals[index] > 0.9) & (np.linalg.norm(points - points[index], axis=-1) < 0.004)
    hand = StaticHand(model, None, np.array([body]), points[[index]], normals[[index]])
    hand.semantic_contact_masks = [mask]
    hand.semantic_patch_labels = ["index/distal/palmar/test"]
    hand.semantic_contact_local = hand.local.copy()
    hand.semantic_contact_normals = hand.normals.copy()
    q = np.clip(np.zeros(26), model.jnt_range[:26, 0] + 1e-6, model.jnt_range[:26, 1] - 1e-6)
    fixed_points = hand.semantic_contact_local.copy()
    # A clearly incompatible Shadow metric layout must be diagnostic only;
    # it must not pull the target hand to a different morphology.
    monkeypatch.setattr(template, "contact_targets", lambda *a, **k: (
        np.array([[1.0, 1.0, 1.0]]), np.array([[0.0, 1.0, 0.0]]), {}
    ))
    fitted, info = template.fit_contact_template(
        hand, None, None, [], q, device="cuda:0", iterations=1, surface_rounds=1,
        reference_qpos=q, reference_normals=hand.semantic_contact_normals,
    )
    assert np.isfinite(fitted).all()
    assert "matched Wuji semantic patches" in info["target_source"]
    assert info["semantic_patch_violation_count"] == 0
    assert mask[info["surface_query_indices"][0]]
    assert info["semantic_source_layout_position_error_mm"][0] > 100
    assert info["semantic_target_layout_position_error_mm"][0] < 30
    np.testing.assert_array_equal(hand.semantic_contact_local, fixed_points)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_power_cylinder_refinement_cannot_switch_semantic_patches():
    import mujoco
    import trimesh
    import xml.etree.ElementTree as ET
    from retarget.config.run import ROOT
    from retarget.env.grasp.hand import hand_tree, query_path
    from retarget.env.grasp.inputs import StaticHand
    from retarget.env.grasp.input_geometry import fit_power_cylinder_input
    from retarget.env.grasp.regions import belly_axis

    if not (ROOT / "data/grasp/assets/wuji2/right.xml").exists():
        pytest.skip("WUJI2 assets required")
    model = mujoco.MjModel.from_xml_string(ET.tostring(hand_tree("wuji2"), encoding="unicode"))
    bodies = np.array([model.body("right_palm_link").id, model.body("right_finger2_link4").id])
    local, normals, masks, indices = [], [], [], []
    for body in bodies:
        with np.load(query_path(model, int(body))) as raw:
            p, n = raw["local_pos"], raw["local_normal"]
        candidates = np.flatnonzero(n @ belly_axis(model, int(body)) > 0.8)
        index = int(candidates[len(candidates) // 2])
        mask = np.zeros(len(p), bool)
        mask[index] = True
        local.append(p[index])
        normals.append(n[index])
        masks.append(mask)
        indices.append(index)
    hand = StaticHand(model, SimpleNamespace(
        triangles=trimesh.creation.cylinder(radius=0.035, height=0.12).triangles,
    ), bodies, np.array(local), np.array(normals))
    hand.semantic_contact_masks = masks
    hand.semantic_patch_labels = ["palm/central", "index/distal/palmar"]
    hand.semantic_contact_local = hand.local.copy()
    hand.semantic_contact_normals = hand.normals.copy()
    q = np.clip(np.zeros(26), model.jnt_range[:26, 0] + 1e-6, model.jnt_range[:26, 1] - 1e-6)
    fitted, info = fit_power_cylinder_input(
        hand, q, device="cuda:0", reference_qpos=q,
        reference_normals=hand.semantic_contact_normals,
    )
    assert np.isfinite(fitted).all()
    assert info["semantic_patch_violation_count"] == 0
    assert info["surface_query_indices"] == indices
    assert len(info["semantic_reference_layout_position_error_mm"]) == len(bodies)
    assert len(info["semantic_target_layout_position_error_mm"]) == len(bodies)
    np.testing.assert_array_equal(hand.local, hand.semantic_contact_local)
