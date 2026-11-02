"""Pinned DexForge CA solves actual Wuji/primitive geometry without MANO."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from retarget.config.run import ROOT
from retarget.env.grasp.contactaware import (
    COMMIT, _Problem, refine_grasp_reference, require_upstream, validate_settings,
)


@pytest.mark.parametrize("values", [
    {"invented": 2}, {"max_iterations": 0}, {"max_iterations": 1.5},
    {"contact_tolerance_mm": 0}, {"normal_cosine": 1},
    {"required_fingers": ["unknown"]}, {"anchor_weight": float("nan")},
])
def test_contactaware_settings_are_explicit_and_validated(values):
    with pytest.raises(ValueError):
        validate_settings(SimpleNamespace(contactaware=values))


def test_contactaware_core_loads_without_mano_weights():
    import sys
    source = require_upstream()
    assert source["commit"] == COMMIT
    assert "contactaware.models.mano" not in sys.modules


@pytest.fixture
def actual_seed(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required by the grasp geometry adapter")
    if not (ROOT / "data/grasp/assets/wuji2/right.xml").is_file():
        pytest.skip("Prepare the pinned Wuji2 assets")
    from retarget.env.grasp.inputs import build_scene, StaticHand
    from retarget.env.grasp.primitives import primitive_mesh
    from retarget.env.grasp.semantic_patches import match_semantic_patches
    from retarget.env.grasp.hand import query_path
    from retarget.env.grasp.initialization import GraspInitialization
    from retarget.task.grasp.dexonomy import load_template

    raw = json.loads((ROOT / "tests/fixtures/grasp_contactaware_seed.json").read_text())
    path = primitive_mesh(tmp_path / "cylinder.obj", "cylinder", .035, .12)
    model, surface, _, _ = build_scene(path, hand_model="wuji2")
    shadow, _, contacts, _ = load_template(ROOT / "data/grasp/templates", "1_Large_Diameter")
    _, _, _, _, patches = match_semantic_patches(shadow, contacts, model)
    bodies = np.asarray([model.body(name).id for name in raw["body_names"]])
    local, normal = [], []
    for body, index in zip(bodies, raw["query_indices"]):
        with np.load(query_path(model, int(body))) as query:
            local.append(query["local_pos"][index]); normal.append(query["local_normal"][index])
    hand = StaticHand(model, surface, bodies, np.asarray(local), np.asarray(normal))
    hand.semantic_contact_masks = [p["mask"] for p in patches]
    hand.semantic_patch_labels = [p["id"] for p in patches]
    hand.contact_reference_normals = np.asarray(raw["region_reference_normals"])
    seed = GraspInitialization(np.asarray(raw["qpos"]), np.asarray(raw["object_points"]),
        np.asarray(raw["object_normals"]), {}, {"initialization_valid": True, "contacts_optional": True})
    args = SimpleNamespace(init_device="cuda:0", friction=1., contactaware={
        "contact_tolerance_mm": .5, "max_iterations": 40})
    return hand, seed, args


def test_actual_ca_solves_contacts_and_ik_instead_of_reusing_template(actual_seed, monkeypatch):
    hand, seed, args = actual_seed
    require_upstream()
    import contactaware.solver.grasp_sqp as upstream
    original = upstream.solve_joint_grasp
    calls = []

    def observed(*a, **kw):
        calls.append(True)
        return original(*a, **kw)

    monkeypatch.setattr(upstream, "solve_joint_grasp", observed)
    local = hand.local.copy()
    solved = refine_grasp_reference(hand, seed, args)
    report = solved.metadata["contactaware"]
    assert calls and report["success"]
    assert all(item["contact"] for item in report["required_finger_contact"].values())
    assert max(item["minimum_error_mm"] for item in report["required_finger_contact"].values()) <= .5
    assert report["whole_hand_penetration_mm"] <= 2
    assert report["semantic_patch_violation_count"] == 0
    assert report["object_pose_fixed"]
    assert np.linalg.norm(solved.hand_qpos - seed.hand_qpos) > .01
    assert np.linalg.norm(solved.object_points - seed.object_points) > .001
    np.testing.assert_array_equal(hand.local, local)
    np.testing.assert_array_equal(solved.hand_qpos, solved.hand_reference_qpos)
    assert not np.shares_memory(solved.hand_qpos, solved.hand_reference_qpos)
    np.testing.assert_allclose(solved.hand_qpos[3:6], Rotation.from_euler(
        "XYZ", solved.hand_qpos[3:6]).as_euler("XYZ"), atol=1e-12)
    assert solved.contact_mask.dtype == bool and solved.contact_mask.any()
    assert any(not p["active"] and p["release_reason"] for p in report["per_patch"])


def test_ca_failure_never_inherits_shape_first_success_or_seed_reference(actual_seed):
    hand, seed, args = actual_seed
    seed.hand_reference_qpos=seed.hand_qpos.copy()
    seed.contactaware_reference_qpos=seed.hand_qpos.copy()
    args.contactaware.update(max_iterations=1, trust_translation_mm=1e-5,
                             trust_rotation_rad=1e-7, trust_joint_rad=1e-7)
    solved = refine_grasp_reference(hand, seed, args)
    assert not solved.metadata["contactaware"]["success"]
    assert not solved.metadata["initialization_valid"]
    assert not solved.metadata["contacts_optional"]
    assert not hasattr(solved, "hand_reference_qpos")
    assert not hasattr(solved, "contactaware_reference_qpos")


def test_query_outside_original_patch_is_rejected(actual_seed):
    hand, seed, args = actual_seed
    hand.semantic_contact_masks[0] = np.zeros_like(hand.semantic_contact_masks[0])
    with pytest.raises(ValueError, match="semantic query masks"):
        refine_grasp_reference(hand, seed, args)


def test_configured_non_cylinder_intent_is_not_reloaded_as_a_custom_template(actual_seed):
    hand, seed, args = actual_seed
    hand.pose_intent_qpos = seed.hand_qpos.copy()
    # The initialization already loaded the bank; CA must use that configured
    # hand for every primitive family without trying a second "custom" lookup.
    args.grasp_intent_reference = "/not/a/second/input/bank.json"
    args.contactaware["max_iterations"] = 1
    solved = refine_grasp_reference(hand, seed, args, primitive=dict(kind="disk"))
    assert "nominal_rotation_error_deg" in solved.metadata["contactaware"]["grasp_intent"]


def test_wrench_geometry_uses_actual_material_points_not_free_anchors(actual_seed):
    hand, seed, args = actual_seed
    require_upstream()
    problem=_Problem(hand,seed,args,validate_settings(args))
    original=problem.wrench
    observed=[]

    def inspect_points(points,normals,point_rows,normal_rows):
        observed.append(np.array(points,copy=True))
        return original(points,normals,point_rows,normal_rows)

    problem.wrench=inspect_points
    initial=np.zeros(problem.dimension)
    first=problem.evaluate(initial)
    moved=initial.copy();moved[26]+=.1
    second=problem.evaluate(moved)
    assert np.linalg.norm(first["anchors"]-second["anchors"])>.001
    np.testing.assert_array_equal(observed[0],observed[1])
    np.testing.assert_allclose(observed[0],first["points"])
    assert not np.allclose(observed[0],first["anchors"])


def test_intent_constraints_differentiate_real_middle_points_anchors_and_rotation(actual_seed, tmp_path):
    from retarget.env.grasp.intent import configure_intent, intent_pose_seeds, intent_diagnostics
    hand, seed, args = actual_seed
    hand.ik_reference_qpos = seed.hand_qpos.copy()
    bank=tmp_path/"intent.json"
    bank.write_text(json.dumps(dict(references={"1_Large_Diameter:cylinder":
        dict(qpos=seed.hand_qpos.tolist(),radius_m=.035)})))
    args.grasp_intent_reference=str(bank)
    configure_intent(hand, "1_Large_Diameter", dict(kind="cylinder", radius_m=.035, height_m=.18), args)
    poses = intent_pose_seeds(hand, seed.hand_qpos, dict(radius_m=.035))
    assert max(abs(z) for z in intent_diagnostics(hand, poses[0])["middle_contact_axial_mm"]) < 1e-8
    args.contactaware.update(bone_max_deviation_deg=45., bone_style_weight=.02)
    require_upstream()
    problem = _Problem(hand, seed, args, validate_settings(args))
    x = np.zeros(problem.dimension); x[:4]=[.01,-.02,.01,.03]
    value = problem.evaluate(x)
    # Final rows: 15 bone cones, one pose cone, real middle +/-Z and anchor +/-Z.
    count = 15+1+4*len(hand.cylinder_middle_slots)+1
    for axis in [2, 3, 10, 26+3*int(hand.cylinder_middle_slots[0])+2]:
        offset = np.zeros_like(x); offset[axis] = 1e-6
        actual = (problem.evaluate(x+offset)["constraints"][-count:]-
                  problem.evaluate(x-offset)["constraints"][-count:])/(2e-6)
        np.testing.assert_allclose(actual, value["rows"][-count:, axis], atol=2e-5, rtol=2e-4)
    assert "grasp_intent" in problem.diagnostics(x)
    shifted = poses[0].copy(); shifted[2] += .07
    assert not intent_diagnostics(hand, shifted)["middle_in_center_band"]
