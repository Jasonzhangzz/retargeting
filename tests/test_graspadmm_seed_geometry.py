"""Shared 1.5x seed poses and the optimization-only orientation equality."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from retarget.env.grasp.contact_regions import (
    lock_root_orientation,
    orientation_error_deg,
    region_model,
)
from retarget.env.grasp.graspadmm_scene import from_upstream_qpos, write_scene
from retarget.env.grasp.graspadmm_seed_geometry import adapt_primitive_seed
from retarget.env.grasp.primitives import primitive_mesh


@pytest.fixture
def reference():
    record = json.loads((Path(__file__).parent / "fixtures/graspadmm_type9_seed.json").read_text())["data"]
    for name in ("grasp_qpos", "hand_cpn_w", "obj_cpn_w", "ext_wrench", "ext_center", "n_evo"):
        record[name] = np.asarray(record[name], np.float32)
    return record


def assert_data_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], np.ndarray):
            assert left[key].dtype == right[key].dtype
            assert left[key].shape == right[key].shape
            assert left[key].tobytes() == right[key].tobytes()
        else:
            assert left[key] == right[key]


def test_all_sizes_reuse_reference_fingers_root_and_rotate_object_in_palm(reference, tmp_path):
    pytest.importorskip("rtree")
    snapshot = copy.deepcopy(reference)
    poses = []
    for i, radius in enumerate([.009, .0135, .018, .0225, .027]):
        mesh_path = primitive_mesh(tmp_path / f"cylinder{i}.obj", "cylinder", radius, .12)
        raw = copy.deepcopy(reference)
        raw["grasp_qpos"][0, 0] += i * .01
        raw["grasp_qpos"][0, 7:] += i * .01
        adapted, info = adapt_primitive_seed(raw, SimpleNamespace(mesh_path=mesh_path),
            dict(kind="cylinder", radius_m=radius), "9_Palmar_Pinch", reference_seed=reference,
            reference_info={"seed": 4, "size_scale": 1.5})
        assert info["applied"] and info["reference"]["seed"] == 4
        np.testing.assert_array_equal(adapted["grasp_qpos"][:, 7:], reference["grasp_qpos"][:, 7:])
        poses.append(adapted["grasp_qpos"])
        mesh = trimesh.load_mesh(mesh_path, process=False)
        lateral = mesh.submesh([np.flatnonzero(np.abs(mesh.face_normals[:, 2]) < 1e-6)], append=True, repair=False)
        _, distance, _ = trimesh.proximity.closest_point(lateral, adapted["obj_cpn_w"][:, :3])
        assert np.all(distance < 1e-7)
        assert np.all(np.abs(adapted["obj_cpn_w"][:, 5]) < 1e-6)
        assert min(info["object_target_cap_clearance_m"]) > .029
        before = Rotation.from_quat(reference["grasp_qpos"][0, 3:7], scalar_first=True)
        after = Rotation.from_quat(adapted["grasp_qpos"][0, 3:7], scalar_first=True)
        expected = Rotation.from_rotvec([0., np.pi/2, 0.]) * before.inv()
        assert np.degrees((expected.inv() * after.inv()).magnitude()) < 1e-5
        np.testing.assert_allclose(-after.inv().apply(adapted["grasp_qpos"][0, :3]),
                                   -before.inv().apply(reference["grasp_qpos"][0, :3]), atol=1e-8)
        # The exact body-local anchor is retained through the rigid root change.
        model = region_model()
        data = mujoco.MjData(model)
        locals_ = []
        for q, anchors in [(reference["grasp_qpos"][0], reference["hand_cpn_w"]),
                           (adapted["grasp_qpos"][0], adapted["hand_cpn_w"])]:
            data.qpos[:] = from_upstream_qpos(q)
            mujoco.mj_kinematics(model, data)
            locals_.append([(p[:3]-data.xpos[model.body(name).id]) @ data.xmat[model.body(name).id].reshape(3,3)
                            for name,p in zip(reference["hand_cbody"], anchors)])
        np.testing.assert_allclose(locals_[0], locals_[1], atol=3e-8)
        repeated, _ = adapt_primitive_seed(adapted, SimpleNamespace(mesh_path=mesh_path),
            dict(kind="cylinder", radius_m=radius), "9_Palmar_Pinch", reference_seed=reference)
        np.testing.assert_array_equal(repeated["grasp_qpos"], adapted["grasp_qpos"])
    for q in poses[1:]:
        np.testing.assert_array_equal(q, poses[0])
    assert_data_equal(reference, snapshot)


@pytest.mark.parametrize("policy,template", [("upstream", "9_Palmar_Pinch"), ("taxonomy", "11_Power_Sphere")])
def test_upstream_and_other_types_are_exact_noops(reference, policy, template):
    adapted, info = adapt_primitive_seed(reference, None, {"kind":"cylinder"}, template, policy=policy)
    assert not info["applied"] and info["data_unchanged"]
    assert_data_equal(reference, adapted)


def test_type9_never_silently_uses_a_size_specific_seed(reference):
    with pytest.raises(ValueError, match="1.5x reference"):
        adapt_primitive_seed(reference, None, {"kind":"cylinder"}, "9_Palmar_Pinch")


def test_orientation_lock_applies_every_simulation_substep_and_keeps_other_dofs_free(tmp_path):
    pytest.importorskip("rtree")
    from retarget.env.grasp.graspadmm_scene import WujiOptEnv
    from retarget.env.grasp.graspadmm_source import require_upstream
    source = require_upstream()
    mesh = primitive_mesh(tmp_path/"object.obj", "cylinder", .027, .12)
    scene = write_scene(tmp_path/"scene", mesh)
    env = WujiOptEnv(source.sim.HandCfg(str(scene.hand_xml), freejoint=True), scene.scene_cfg)
    q = np.r_[[.3, .1, .2], Rotation.from_euler("XYZ", [.2, .3, .4]).as_quat(scalar_first=True), np.full(20, .1)]
    env.reset_qpos(q)
    observed = []
    step = env.step_sim
    def observe(n):
        observed.append(env._data.qpos[3:7].copy())
        step(n)
    env.step_sim = observe
    lock_root_orientation(env, q[3:7])
    env._data.qvel[:3] = [.1, 0, 0]
    env._data.qvel[3:6] = [3, -2, 1]
    env._data.qfrc_applied[3:6] = [2, 3, -1]
    env.step_sim(5)
    assert len(observed) == 5
    for quat in observed:
        np.testing.assert_allclose(quat, q[3:7], atol=1e-12)
    assert orientation_error_deg(env.get_hand_qpos(), q[3:7]) < 1e-9
    np.testing.assert_array_equal(env._data.qvel[3:6], 0.)
    assert env.get_hand_qpos()[0] != q[0]
    assert env._data.time > 0


def test_reference_seed_is_initialized_once_and_reused_across_size_workers(tmp_path, monkeypatch):
    from retarget.env.grasp import graspadmm_inputs as inputs
    root = Path(__file__).resolve().parents[1]
    args = SimpleNamespace(dexonomy_root=root/"data/grasp/templates", scale=1., seed=3, init_device="cpu")
    monkeypatch.setattr(inputs, "_template_fingerprint", lambda *args: ("test-template", {}))
    def finger_angles(*args, **kwargs):
        return np.full(20, .1), dict(template_conversion_seconds=0., cache_hit=False)
    monkeypatch.setattr(inputs, "fixed_finger_angles", finger_angles)
    calls = []
    def initialize(hand, q, template, template_path, args, scene, *, seed, settings):
        calls.append(seed)
        mesh = trimesh.load_mesh(scene.mesh_path, process=False)
        np.testing.assert_allclose(mesh.extents, [.054, .054, .12], atol=1e-7)
        data = copy.deepcopy(template)
        data.update(scene_path=str(scene.scene_path), obj_cpn_w=np.zeros((2,6),np.float32))
        return data, {"seed":seed}
    monkeypatch.setattr(inputs, "initialize_one", initialize)
    first, info = inputs.type9_reference_seed(args, "9_Palmar_Pinch", {}, tmp_path)
    first["grasp_qpos"][0, 7] = 20
    second, cached = inputs.type9_reference_seed(args, "9_Palmar_Pinch", {}, tmp_path)
    assert calls == [3]
    assert cached == info
    assert second["grasp_qpos"][0,7] != 20
    assert info["identity"]["primitive"]["size_scale"] == 1.5
    # A different RNG seed gets its own reference; size is deliberately absent
    # from the key, so separately launched size jobs share the same file.
    args.radius = .009
    third, _ = inputs.type9_reference_seed(args, "9_Palmar_Pinch", {}, tmp_path)
    assert calls == [3]
    np.testing.assert_array_equal(second["grasp_qpos"], third["grasp_qpos"])
    args.seed = 4
    inputs.type9_reference_seed(args, "9_Palmar_Pinch", {}, tmp_path)
    assert calls == [3,4]


def test_native_optimizer_honors_new_metadata_without_modifying_vendor_source(reference, tmp_path):
    pytest.importorskip("rtree")
    import inspect

    from retarget.env.grasp.graspadmm_optimizer import optimize
    from retarget.env.grasp.graspadmm_source import require_upstream
    mesh = primitive_mesh(tmp_path/"object.obj", "cylinder", .027, .12)
    scene = write_scene(tmp_path/"scene", mesh)
    raw = copy.deepcopy(reference)
    raw["scene_path"] = str(scene.scene_path)
    adapted, _ = adapt_primitive_seed(raw, scene, dict(kind="cylinder", radius_m=.027),
                                     "9_Palmar_Pinch", reference_seed=reference)
    source = require_upstream()
    function = source.core._single_grasp.__wrapped__
    assert Path(inspect.getfile(function)).resolve() == Path(__file__).resolve().parents[1]/"third_party/Dexonomy/dexonomy/op/gen_grasp_admm.py"
    _, status, trace = optimize(adapted, scene, tmp_path/"native", dict(step=2))
    assert len(trace) == status["outer_iterations"] == 2
    assert status["reference_orientation_error_deg"] < 1e-4
    assert status["max_trace_orientation_error_deg"] < 1e-4
    assert source.core._single_grasp.__wrapped__ is function


@pytest.mark.parametrize("template", ["6_Prismatic_4_Finger", "7_Prismatic_3_Finger", "8_Prismatic_2_Finger"])
def test_native_admm_records_center_anchors_through_projection_and_finalization(tmp_path, monkeypatch, template):
    pytest.importorskip("rtree")
    from retarget.env.grasp import graspadmm_inputs as inputs
    from retarget.env.grasp.graspadmm_optimizer import optimize
    from retarget.env.grasp.graspadmm_scene import normalize_grasp_data
    def fingers(*args, **kwargs):
        return np.full(20,.1), dict(template_conversion_seconds=0.,cache_hit=False)
    monkeypatch.setattr(inputs,"fixed_finger_angles",fingers)
    args=SimpleNamespace(dexonomy_root=Path(__file__).resolve().parents[1]/"data/grasp/templates",scale=1.)
    mesh_path=primitive_mesh(tmp_path/"object.obj","cylinder",.015,.12)
    _,_,data,_,_=inputs.fixed_template(args,template,mesh_path,tmp_path/"template")
    scene=write_scene(tmp_path/"scene",mesh_path)
    mesh=trimesh.load_mesh(scene.mesh_path,process=False)
    positions,_,triangles=trimesh.proximity.closest_point(mesh,data["hand_cpn_w"][:,:3])
    raw=normalize_grasp_data(data["grasp_qpos"][0],scene,hand_cbody=data["hand_cbody"],
        hand_cpn_w=data["hand_cpn_w"],obj_cpn_w=np.c_[positions,mesh.face_normals[triangles]],
        required_cbody=data["required_cbody"],template=template)
    raw["contact_regions"]=data["contact_regions"]
    _,status,trace=optimize(raw,scene,tmp_path/"native",dict(step=2))
    assert len(trace)==2
    diagnostic=status["contact_region_diagnostics"]
    assert diagnostic["initial_anchors"]["semantic_success"]
    assert diagnostic["optimized_anchors"]["mode"]=="soft"
    assert all(np.isfinite(row["center_preference_cost"]) for row in diagnostic["optimized_anchors"]["contacts"])
    assert "actual_contacts" in diagnostic
    saved=json.loads((tmp_path/"native/contact_region_diagnostics.json").read_text())
    assert saved==diagnostic


def test_native_evaluation_records_actual_regions_even_when_filter_rejects(tmp_path):
    pytest.importorskip("rtree")
    from retarget.env.grasp.contact_regions import template_regions
    from retarget.env.grasp.grasp_evaluation import (
        evaluate_native_grasp,
        native_evaluate,
    )
    from retarget.env.grasp.graspadmm_scene import to_upstream_qpos
    name="right_finger2_link4"
    region=template_regions("6_Test",[name])
    q=region_model().qpos0.copy();q[0]=.5
    mesh=primitive_mesh(tmp_path/"object.obj","cylinder",.02,.12)
    worker=SimpleNamespace(evaluate=lambda directory,timeout:native_evaluate(directory))
    result=evaluate_native_grasp(to_upstream_qpos(q),mesh,tmp_path/"evaluation",required_cbody=[[name]],
                                 contact_regions=region,worker=worker)
    assert result["state"]=="completed" and not result["success"]
    assert not result["contact_region_diagnostics"]["semantic_success"]
    assert not result["contact_region_diagnostics"]["required_bodies_in_region"][name]
