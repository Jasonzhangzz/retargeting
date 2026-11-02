"""Type 9 side-input geometry has no new optimizer or success gate."""

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from retarget.env.grasp.graspadmm_seed_geometry import adapt_primitive_seed
from retarget.env.grasp.graspadmm_scene import write_scene
from retarget.env.grasp.graspadmm_source import require_upstream
from retarget.env.grasp.primitives import primitive_mesh


@pytest.fixture(params=[.0225, .027])
def side_case(tmp_path, request):
    record = json.loads((Path(__file__).parent / "fixtures/graspadmm_type9_seed.json").read_text())
    data = record["data"]
    for name in ("grasp_qpos", "hand_cpn_w", "obj_cpn_w", "ext_wrench", "ext_center", "n_evo"):
        data[name] = np.asarray(data[name], np.float32)
    mesh = primitive_mesh(tmp_path / "cylinder.obj", "cylinder", request.param, .12)
    scene = write_scene(tmp_path / "scene", mesh)
    data["scene_path"] = str(scene.scene_path)
    return data, scene, dict(kind="cylinder", radius_m=request.param, height_m=.12)


def assert_data_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], np.ndarray):
            assert left[key].dtype == right[key].dtype
            assert left[key].shape == right[key].shape
            assert left[key].tobytes() == right[key].tobytes()
        else:
            assert left[key] == right[key]


def test_cap_input_becomes_actual_lateral_targets_without_changing_fingers(side_case):
    raw, scene, primitive = side_case
    snapshot = copy.deepcopy(raw)
    adapted, info = adapt_primitive_seed(raw, scene, primitive, "9_Palmar_Pinch")
    assert info["applied"] and not info["condition"]["all_required_targets_already_side"]
    assert info["method"] == "opposed_rigid_contact_frame_with_actual_lateral_projection"
    assert_data_equal(raw, snapshot)
    assert adapted["grasp_qpos"][:, 7:].tobytes() == raw["grasp_qpos"][:, 7:].tobytes()
    mesh = trimesh.load_mesh(scene.mesh_path, process=False)
    lateral = mesh.submesh([np.flatnonzero(np.abs(mesh.face_normals[:, 2]) < 1e-6)], append=True, repair=False)
    _, distance, _ = trimesh.proximity.closest_point(lateral, adapted["obj_cpn_w"][:, :3])
    assert np.all(distance < 1e-7)
    assert np.all(np.abs(adapted["obj_cpn_w"][:, 5]) < 1e-6)
    assert min(info["object_target_cap_clearance_m"]) > .029
    assert info["world_anchor_fk_consistency_error_m"] < 1e-7
    assert info["adapted_geometry"]["hand_object_max_penetration_mm"] > 10
    assert info["penetration_is_diagnostic_only"]


def test_already_side_input_is_strict_identity_even_with_penetration(side_case):
    raw, scene, primitive = side_case
    side, _ = adapt_primitive_seed(raw, scene, primitive, "9_Palmar_Pinch")
    result, info = adapt_primitive_seed(side, scene, primitive, "9_Palmar_Pinch")
    assert not info["applied"] and info["condition"]["all_required_targets_already_side"]
    assert_data_equal(side, result)


@pytest.mark.parametrize("policy,template", [("upstream", "9_Palmar_Pinch"), ("taxonomy", "11_Power_Sphere")])
def test_upstream_and_other_types_are_exact_noops(side_case, policy, template):
    raw, scene, primitive = side_case
    adapted, info = adapt_primitive_seed(raw, scene, primitive, template, policy=policy)
    assert not info["applied"] and info["data_unchanged"]
    assert_data_equal(raw, adapted)


def test_adapter_does_not_replace_the_original_core(side_case, tmp_path):
    from retarget.env.grasp.graspadmm_optimizer import optimize
    source = require_upstream()
    original = source.core._single_grasp.__wrapped__
    assert Path(inspect.getfile(original)).resolve() == (
        Path(__file__).resolve().parents[1] / "third_party/Dexonomy/dexonomy/op/gen_grasp_admm.py")
    raw, scene, primitive = side_case
    adapted, _ = adapt_primitive_seed(raw, scene, primitive, "9_Palmar_Pinch")
    _, status, trace = optimize(adapted, scene, tmp_path / "native", dict(step=2))
    assert len(trace) == status["outer_iterations"] == 2
    assert not status["fallback_used"]
    assert source.core._single_grasp.__wrapped__ is original
