"""Saved-reference contracts; mock solver proposals do not certify geometry."""

import copy
import hashlib
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from retarget.config.run import ROOT
from retarget.env.grasp import forceaware_ca_refinement as refinement
from retarget.env.grasp.forceaware_reference import _read_ca
from retarget.env.grasp.forceaware_runner import _experiments
from retarget.env.grasp.hand import palmar_axis, query_path


@pytest.fixture
def source():
    paths = sorted((ROOT/"output/grasp_contactaware_mppi").glob("000*/*/input_00"))
    if not paths:
        pytest.skip("Accepted small primitive CA input required")
    return paths[0]


def _hashes(source):
    return {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
        for name in ("card_input.npz", "contactaware_reference.json", "runtime_scene.xml",
                     "contactaware/contact_guidance.npz")}


@pytest.mark.parametrize("accepted", [False, True])
def test_saved_refinement_has_independent_reference_and_preserves_semantics(source, tmp_path, monkeypatch, accepted):
    before = _hashes(source)
    inputs = refinement.saved_reference_inputs(source)
    original = inputs["seed"].hand_qpos.copy()
    destination = tmp_path/"refined"

    def propose(hand, seed, args):
        assert args.contactaware["contact_tolerance_mm"] == .2
        assert args.contactaware["penetration_tolerance_mm"] == .5
        assert args.contactaware["normal_cosine"] == .5
        assert seed.metadata["initialization_valid"] is False
        np.testing.assert_array_equal(seed.hand_qpos, original)
        for i, (body, index) in enumerate(zip(hand.bodies, inputs["query_indices"])):
            with np.load(query_path(hand.model, int(body))) as asset:
                np.testing.assert_array_equal(hand.normals[i], asset["local_normal"][index])
            np.testing.assert_array_equal(hand.semantic_contact_masks[i], inputs["hand"].semantic_contact_masks[i])
        solved = copy.deepcopy(seed)
        solved.hand_qpos[0] += .00001
        solved.object_points[0, 0] += .00001
        solved.contact_mask = np.asarray(inputs["card"]["reference_contact_mask"], bool) if accepted else np.zeros(len(hand.bodies), bool)
        report = copy.deepcopy(inputs["report"])
        report.update(success=accepted, settings=args.contactaware,
            whole_hand_penetration_mm=.4 if accepted else .7,
            active_contact_mask=solved.contact_mask.tolist())
        for row in report["required_finger_contact"].values():
            row.update(contact=accepted, minimum_error_mm=.1 if accepted else .4)
        for i, row in enumerate(report["per_patch"]):
            row.update(active=bool(solved.contact_mask[i]), contact_error_mm=.1 if solved.contact_mask[i] else .4)
        solved.metadata["contactaware"] = report
        if accepted:
            solved.contactaware_reference_qpos = solved.hand_qpos.copy()
        return solved

    monkeypatch.setattr(refinement, "refine_grasp_reference", propose)
    assert refinement.refine_saved_reference(source, destination, active_only=False) == destination
    assert _hashes(source) == before
    assert _experiments(tmp_path) == [destination]
    assert (destination/"source_baseline/source_card_input.npz").is_file()
    assert not (destination/"source_baseline/card_input.npz").exists()
    with np.load(destination/"card_input.npz") as card:
        assert str(card["hand_reference_source"].item()) == "dexforge_contactaware"
        assert card["reference_valid"].item() is accepted
        for key in ("semantic_patch_offsets", "semantic_patch_masks", "semantic_patch_labels",
                    "semantic_reference_local", "semantic_reference_normals", "region_reference_normals"):
            np.testing.assert_array_equal(card[key], inputs["card"][key])
        if accepted:
            assert card["hand_reference_qpos"][0] != original[0]
            np.testing.assert_array_equal(card["hand_initial_qpos"], card["hand_reference_qpos"])
        else:
            assert "hand_reference_qpos" not in card
            assert "hand_candidate_qpos" in card
    np.testing.assert_array_equal(np.load(destination/"contactaware/object_pose_7.npy"), inputs["poses"])
    with np.load(destination/"contactaware/contact_guidance.npz") as guide:
        np.testing.assert_array_equal(guide["hand_query_body_ids"], inputs["guidance"]["hand_query_body_ids"])
        np.testing.assert_array_equal(guide["hand_query_local_pos"], inputs["guidance"]["hand_query_local_pos"])
    if accepted:
        _read_ca(destination)
    else:
        with pytest.raises(ValueError, match="hand reference"):
            _read_ca(destination)


def test_solver_exception_is_saved_without_falling_back_to_accepted_seed(source, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate solver failure")
    monkeypatch.setattr(refinement, "refine_grasp_reference", fail)
    destination = refinement.refine_saved_reference(source, tmp_path/"failed")
    report = json.loads((destination/"contactaware_reference.json").read_text())
    assert report["success"] is False
    assert report["error_type"] == "RuntimeError"
    assert "deliberate solver failure" in report["traceback"]
    manifest = json.loads((destination/"ca_refinement_manifest.json").read_text())
    assert manifest["template_angle_fallback"] is False
    assert "diagnostics only" in manifest["candidate_origin"]
    with np.load(destination/"card_input.npz") as card:
        assert "hand_reference_qpos" not in card


def test_palmar_axis_wxyz_rotation_matches_mujoco_on_all_five_real_mcp_frames(source):
    model = refinement.saved_reference_inputs(source)["hand"].model
    for finger in range(1, 6):
        body = model.body(f"right_finger{finger}_link2").id
        child = model.body(f"right_finger{finger}_link3").id
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.body_quat[child])
        hinge = rotation.reshape(3, 3)@model.jnt_axis[model.body_jntadr[child]]
        expected = np.cross(hinge, model.body_pos[child]); expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(palmar_axis(model, body, model.body_pos[child]), expected, rtol=0, atol=1e-14)


def test_active_only_solves_five_contacts_but_serializes_all_original_slots(source, tmp_path, monkeypatch):
    values = refinement.saved_reference_inputs(source)
    slots = np.flatnonzero(values["card"]["reference_contact_mask"])
    inactive = np.flatnonzero(~values["card"]["reference_contact_mask"].astype(bool))
    original_anchors = values["seed"].object_points.copy()

    def propose(hand, seed, args):
        assert len(hand.bodies) == len(slots) == 5
        assert args.contactaware["normal_cosine"] == .8
        assert len(hand.all_bodies) == len(values["hand"].all_bodies) == 16384
        np.testing.assert_array_equal(seed.object_points, original_anchors[slots])
        assert hand.semantic_patch_labels == [values["hand"].semantic_patch_labels[i] for i in slots]
        solved = copy.deepcopy(seed)
        solved.hand_qpos[0] += .0003
        hand.set(solved.hand_qpos)
        solved.object_points = hand.points(hand.bodies, hand.local)
        solved.contact_mask = np.ones(5, bool)
        solved.contactaware_reference_qpos = solved.hand_qpos.copy()
        report = copy.deepcopy(values["report"])
        report.update(success=True, settings=args.contactaware,
            per_patch=[report["per_patch"][i] for i in slots], active_contact_mask=[True]*5,
            whole_hand_penetration_mm=.4)
        solved.metadata["contactaware"] = report
        return solved

    def geometry(hand, qpos, anchors, device):
        assert len(hand.bodies) == len(anchors) == 16
        hand.set(qpos)
        gaps = np.linalg.norm(hand.points(hand.bodies, hand.local)-anchors, axis=1)*1000.
        return gaps, np.ones(16), np.zeros(16)

    monkeypatch.setattr(refinement, "refine_grasp_reference", propose)
    monkeypatch.setattr(refinement, "_slot_geometry", geometry)
    destination = refinement.refine_saved_reference(source, tmp_path/"active_default", normal_cosine=.8)
    _, card, report, guidance, _ = _read_ca(destination)
    assert report["active_only"] is True
    assert report["settings"]["normal_cosine"] == .8
    assert report["optimization_slot_count"] == 5 and report["saved_slot_count"] == 16
    assert len(report["per_patch"]) == 16 and len(report["optimization_per_patch"]) == 5
    np.testing.assert_array_equal(card["object_points"][inactive], original_anchors[inactive])
    np.testing.assert_array_equal(card["reference_contact_mask"], values["card"]["reference_contact_mask"])
    np.testing.assert_array_equal(guidance["hand_query_body_ids"], values["guidance"]["hand_query_body_ids"])
    assert all(report["per_patch"][i]["release_reason"] == "not_in_accepted_contact_mask" for i in inactive)
    assert all(report["per_patch"][i]["optimized_contact_slot"] is False for i in inactive)
    assert max(report["per_patch"][i]["contact_error_mm"] for i in slots) < 1e-8
    assert report["full_hand_geometry_preserved"] is True


def test_problem_active_topology_preserves_full_forbidden_mask_and_validates_override(source, monkeypatch):
    import torch
    from retarget.env.grasp import contactaware
    from retarget.env.grasp.regions import hand_forbidden_mask
    contactaware.require_upstream()
    values = refinement.saved_reference_inputs(source)
    hand, seed, slots, full_forbidden = refinement.active_reference_inputs(values)
    expected = hand_forbidden_mask(values["hand"], values["hand"].contact_reference_normals)
    np.testing.assert_array_equal(full_forbidden, expected)

    class GeometryStub:
        def __init__(self, surface, device):
            self.radius = np.linalg.norm(surface.triangles.reshape(-1, 3), axis=1).max()
        def query(self, points):
            return points, torch.nn.functional.normalize(torch.ones_like(points), dim=-1), torch.zeros(len(points))

    class CapsulesStub:
        def __init__(self, model, device): pass
        def depths(self, q): return torch.zeros(1)

    monkeypatch.setattr(contactaware, "GraspAudit", GeometryStub)
    monkeypatch.setattr(contactaware, "WujiSelfCollision", CapsulesStub)
    args = SimpleNamespace(init_device="cpu", friction=1., contactaware={
        "contact_tolerance_mm": .2, "penetration_tolerance_mm": .5})
    settings = contactaware.validate_settings(args)
    problem = contactaware._Problem(hand, seed, args, settings)
    assert problem.count == 5 and len(problem.topology.pairs) == 10
    assert len(problem.groups_by_body) == 21
    np.testing.assert_array_equal(problem.forbidden, expected)
    assert not np.shares_memory(problem.forbidden, hand.contactaware_forbidden_mask)
    legacy = contactaware._Problem(values["hand"], values["seed"], args, settings)
    assert legacy.count == 16 and len(legacy.topology.pairs) == 120
    np.testing.assert_array_equal(legacy.forbidden, expected)
    hand.contactaware_forbidden_mask = expected.astype(np.float32)
    with pytest.raises(ValueError, match="boolean in full hand-query order"):
        contactaware._Problem(hand, seed, args, settings)


def test_rim_normal_is_independent_of_equal_nearest_triangle_face():
    from retarget.env.grasp.contactaware import signed_distance_normal, Settings
    point = np.asarray([[-.001, 0., -.01]])
    projected = np.zeros((1, 3)); distance = np.linalg.norm(point, axis=1)
    side = signed_distance_normal(point, projected, [[-1., 0., 0.]], distance)
    cap = signed_distance_normal(point, projected, [[0., 0., -1.]], distance)
    np.testing.assert_allclose(side, point/distance[:, None], atol=1e-14)
    np.testing.assert_array_equal(side, cap)
    hand_normal = np.asarray([.97, 0., .25]); hand_normal /= np.linalg.norm(hand_normal)
    assert -hand_normal@np.asarray([-1., 0., 0.]) > .8
    assert -hand_normal@side[0] < Settings().normal_cosine == .5
    inside = signed_distance_normal([[0., 0., -.01]], projected, [[0., 0., 1.]], [-.01])
    np.testing.assert_array_equal(inside, [[0., 0., 1.]])
    near = signed_distance_normal(projected, projected, [[-1., 0., 0.]], [0.])
    np.testing.assert_array_equal(near, [[-1., 0., 0.]])


def test_guided_edge_normal_jacobian_includes_geometric_normal_rotation():
    from retarget.env.grasp.contactaware import _Problem
    # An edge along Y has projection (0,y,0), so its normal rotates in XZ.
    def query(points):
        projected = np.zeros_like(points); projected[:, 1] = points[:, 1]
        delta = points-projected; length = np.linalg.norm(delta, axis=1)
        return projected, delta/length[:, None], (length, delta/length[:, None])
    points = np.asarray([[.001, .002, -.003]])
    jacobian = _Problem.normal_jacobian(SimpleNamespace(query=query), points)
    normal = query(points)[1][0]; edge = np.asarray([0., 1., 0.])
    expected = (np.eye(3)-np.outer(normal, normal)-np.outer(edge, edge))/np.linalg.norm([.001, 0., -.003])
    np.testing.assert_allclose(jacobian[0], expected, rtol=1e-6, atol=1e-6)
