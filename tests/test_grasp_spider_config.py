"""CPU input/configuration/provenance checks; no synthetic grasp-success claims."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
import torch
import yaml

from retarget.config.run import Config, ROOT, load
from retarget.dynamics.dexforge_upstream import COMMIT
from retarget.env.grasp import spider_reference, spider_runner, spider_source
from retarget.solver.base import Settings


@pytest.fixture
def forbid_forceaware(monkeypatch):
    """Pure CA readers/XML serialization must not activate a FA runtime."""
    from retarget.dynamics import dexforge_upstream
    from retarget.env.grasp import forceaware_reference

    def forbidden(*args, **kwargs):
        raise AssertionError("SPIDER must not invoke the ForceAware optimizer/runtime")

    monkeypatch.setattr(dexforge_upstream, "activate_upstream", forbidden)
    monkeypatch.setattr(dexforge_upstream, "require_environment", forbidden)
    monkeypatch.setattr(forceaware_reference, "activate_upstream", forbidden)
    monkeypatch.setattr(forceaware_reference, "create_optimizer", forbidden)
    monkeypatch.setattr(forceaware_reference, "register_wuji2_hand", forbidden)


@pytest.fixture
def accepted(tmp_path):
    """An explicit 26-DOF format fixture, not an accepted geometric grasp."""
    source = tmp_path/"accepted"
    scene, contacts = source/"sequence/scene/wuji_right", source/"contactaware"
    scene.mkdir(parents=True); contacts.mkdir()
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep=".004", iterations="17", ls_iterations="18")
    world = ET.SubElement(root, "worldbody")
    palm = ET.SubElement(world, "body", name="right_palm_link")
    inertia = dict(pos="0 0 0", mass=".1", diaginertia=".001 .001 .001")
    ET.SubElement(palm, "inertial", **inertia)
    joints = []
    for i in range(6):
        name = f"wrist_{i}"; joints.append(name)
        ET.SubElement(palm, "joint", name=name, type="slide" if i < 3 else "hinge",
            axis=" ".join(map(str, np.eye(3)[i % 3])), range="-2 2")
    for finger in range(1, 6):
        parent = palm
        for link in range(1, 5):
            parent = ET.SubElement(parent, "body", name=f"right_finger{finger}_link{link}", pos="0 0 .02")
            ET.SubElement(parent, "inertial", **inertia)
            name = f"finger_{finger}_{link}"; joints.append(name)
            ET.SubElement(parent, "joint", name=name, axis="1 0 0", range="-1 1")
    obj = ET.SubElement(world, "body", name="grasp_object")
    ET.SubElement(obj, "freejoint", name="obj_joint")
    ET.SubElement(obj, "geom", name="object_collision", type="box", size=".01 .02 .03",
                  mass=".07", friction=".8 .02 .003")
    actuators = ET.SubElement(root, "actuator")
    for name in joints:
        ET.SubElement(actuators, "position", name=f"a_{name}", joint=name,
                      kp="10", kv="2", ctrlrange="-2 2", forcerange="-1 1")
    xml = scene/"primitive_sdf.xml"
    ET.ElementTree(root).write(xml)
    model = mujoco.MjModel.from_xml_path(str(xml))
    qref = np.linspace(-.02, .02, 26).astype(np.float32)
    mask = np.asarray([1, 1, 1, 1, 1, 0], np.float32)
    bodies = np.asarray([model.body(f"right_finger{i}_link4").id for i in (1, 2, 3, 4, 5, 1)])
    anchors = np.arange(18, dtype=np.float32).reshape(6, 3)*.001
    np.savez(source/"card_input.npz", hand_reference_qpos=qref,
        hand_reference_source=np.asarray("dexforge_contactaware"), reference_contact_mask=mask)
    (source/"contactaware_reference.json").write_text(json.dumps(dict(success=True,
        source_commit=COMMIT, required_finger_contact={name: dict(contact=True)
            for name in ("thumb", "index", "middle", "ring", "pinky")})))
    np.savez(contacts/"contact_guidance.npz", contact_mask=np.tile(mask, (2, 1)),
        contact_weight=np.ones((2, 6), np.float32), contact_pos_obj=np.tile(anchors, (2, 1, 1)),
        hand_query_body_ids=np.tile(bodies, (2, 1)),
        hand_query_local_pos=np.tile(np.asarray([0, 0, .01], np.float32), (2, 6, 1)))
    np.save(contacts/"wuji_right_qpos.npy", np.tile(qref, (2, 1)))
    pose = np.asarray([0., 0., np.sqrt(.5), np.sqrt(.5), .01, -.02, .03])
    np.save(contacts/"object_pose_7.npy", np.tile(pose, (2, 1)))
    return source


def config(source, spider=None):
    return Config(source=source.parent/"task.yaml", mode="solve",
        task=dict(type="grasp", name="primitive", data=str(source), options=dict(hand_model="wuji2")),
        solver=Settings(type="mppi", population=100, worlds=2, iterations=1),
        environment=dict(device="cuda:0", dynamics="mjwarp", spider={} if spider is None else spider),
        output=dict(name="spider_config_fixture"), visualization=dict(save_video=False, live=False), seed=0)


def save_config(cfg):
    cfg.source.write_text(yaml.safe_dump(cfg.resolved()))
    return cfg.source


def test_valid_options_keep_explicit_native_physics_and_contact_cost():
    default = spider_runner.options(SimpleNamespace(environment={"spider": {}}))
    assert default["sim_dt"] == .01 and default["exec_dt"] == .002
    assert default["solver_iterations"] == 20 and default["ls_iterations"] == 50
    assert default["cost"] == {"contact_rew_scale": 1.}
    values = dict(duration_seconds=.02, sim_dt=.004, exec_dt=.002, horizon=.024,
        knot_dt=.008, ctrl_dt=.008, nconmax=32, njmax=256, solver_iterations=7,
        ls_iterations=9, preload=False, cost={"terminal_rew_scale": 0., "contact_rew_scale": 0.})
    resolved = spider_runner.options(SimpleNamespace(environment={"spider": values}))
    assert all(resolved[key] == value for key, value in values.items())


@pytest.mark.parametrize("values", [
    {"unknown": 1}, {"duration_seconds": 0.}, {"sim_dt": float("nan")},
    {"exec_dt": float("inf")}, {"joint_noise": -1.}, {"first_noise": True},
    {"nconmax": 32.5}, {"solver_iterations": True}, {"preload": 1}, {"cost": []},
    {"cost": {"contact_rew_scale": -1.}}, {"cost": {"joint_rew_scale": True}},
    {"cost": {"unknown_weight": 1.}},
    {"horizon": .31}, {"knot_dt": .07}, {"ctrl_dt": .035},
    {"sim_dt": .011}, {"duration_seconds": .401}, {"ctrl_dt": .6},
])
def test_invalid_options_reject_before_physics(values):
    with pytest.raises(ValueError):
        spider_runner.options(SimpleNamespace(environment={"spider": values}))


@pytest.mark.parametrize("values", [True, 1, [], ["preload"], "options", None])
def test_spider_settings_must_be_an_explicit_mapping(values):
    with pytest.raises(ValueError, match="environment.spider must be a mapping"):
        spider_runner.options(SimpleNamespace(environment={"spider": values}))


def test_config_file_resolves_data_and_preserves_spider_mapping(accepted):
    cfg = config(accepted, {"preload": False})
    path = save_config(cfg)
    loaded = load(path)
    assert loaded.task["data"] == str(accepted.resolve())
    assert loaded.environment["spider"] == {"preload": False}
    assert loaded.solver.type == "mppi" and loaded.device == "cuda:0"


@pytest.mark.parametrize("mutation", ["humanoid", "cem", "other_dynamics"])
def test_unified_config_rejects_spider_on_other_runtime(accepted, mutation):
    cfg = config(accepted)
    if mutation == "humanoid":
        cfg.task["type"] = "humanoid"
    elif mutation == "cem":
        cfg.solver = replace(cfg.solver, type="cem")
    else:
        cfg.environment["dynamics"] = "explicit"
    with pytest.raises(ValueError, match="environment.spider requires"):
        load(save_config(cfg))


def test_accepted_input_reads_exact_ca_state_active_points_and_hashes(accepted, forbid_forceaware):
    before = {path: spider_reference.sha256(path) for path in accepted.rglob("*") if path.is_file()}
    reference = spider_reference.read_input(accepted)
    card = np.load(accepted/"card_input.npz", allow_pickle=False)
    np.testing.assert_array_equal(reference.qpos[:26], card["hand_reference_qpos"])
    np.testing.assert_allclose(reference.qpos[26:], [.01, -.02, .03, np.sqrt(.5), 0, 0, np.sqrt(.5)])
    np.testing.assert_array_equal(reference.qvel, np.zeros(32))
    np.testing.assert_array_equal(reference.active_slots, np.arange(5))
    assert reference.contact_bodies.shape == (5,) and reference.object_local.shape == (5, 3)
    assert reference.scene == accepted/"sequence/scene/wuji_right/primitive_sdf.xml"
    assert reference.source_hashes[str(reference.scene)] == before[reference.scene]
    for name in ("card_input.npz", "contactaware_reference.json", "contactaware/contact_guidance.npz",
                 "contactaware/wuji_right_qpos.npy", "contactaware/object_pose_7.npy"):
        path = accepted/name
        assert reference.source_hashes[str(path)] == before[path]
    assert {path: spider_reference.sha256(path) for path in before} == before


def test_scene_and_local_primitive_asset_are_covered_by_source_hashes(accepted):
    primitive = accepted/"sequence/scene/wuji_right/object.obj"
    primitive.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    scene = accepted/"sequence/scene/wuji_right/primitive_sdf.xml"
    tree = ET.parse(scene)
    ET.SubElement(ET.SubElement(tree.getroot(), "asset"), "mesh", name="fixture_mesh", file="object.obj")
    tree.write(scene)
    reference = spider_reference.read_input(accepted)
    assert reference.source_hashes[str(primitive)] == spider_reference.sha256(primitive)
    assert reference.source_hashes[str(reference.scene)] == spider_reference.sha256(reference.scene)
    primitive.write_text(primitive.read_text()+"# changed\n")
    assert reference.source_hashes[str(primitive)] != spider_reference.sha256(primitive)


@pytest.mark.parametrize("attributes,tag,filename,relative", [
    ({}, "mesh", "models/part.obj", "models/part.obj"),
    ({"meshdir": "meshes", "assetdir": "unused"}, "mesh", "part.obj", "meshes/part.obj"),
    ({"texturedir": "textures", "assetdir": "unused"}, "texture", "part.png", "textures/part.png"),
    ({"assetdir": "assets"}, "mesh", "part.obj", "assets/part.obj"),
    ({"assetdir": "assets"}, "texture", "part.png", "assets/part.png"),
    ({"assetdir": "assets"}, "hfield", "part.bin", "assets/part.bin"),
    ({"strippath": "true", "meshdir": "new"}, "mesh", "old/location/part.obj", "new/part.obj"),
])
def test_scene_asset_resolution_respects_compiler_directories(tmp_path, attributes, tag, filename, relative):
    scene, asset = tmp_path/"scene.xml", tmp_path/relative
    asset.parent.mkdir(parents=True, exist_ok=True); asset.write_bytes(b"format-only asset")
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", **attributes)
    assets = ET.SubElement(root, "asset")
    ET.SubElement(assets, tag, name="file_backed", file=filename)
    ET.SubElement(assets, "mesh", name="inline", vertex="0 0 0 1 0 0 0 1 0")
    ET.ElementTree(root).write(scene)
    assert spider_reference.scene_files(scene) == [scene, asset.resolve()]


@pytest.mark.parametrize("absolute_directory", [False, True])
def test_scene_asset_resolution_retains_absolute_asset_paths(tmp_path, absolute_directory):
    directory = tmp_path/"external"; directory.mkdir()
    asset = directory/"object.obj"; asset.write_text("format-only mesh")
    scene = tmp_path/"scene.xml"
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", meshdir=str(directory) if absolute_directory else "unused")
    ET.SubElement(ET.SubElement(root, "asset"), "mesh", name="object",
                  file=asset.name if absolute_directory else str(asset))
    ET.ElementTree(root).write(scene)
    assert spider_reference.scene_files(scene) == [scene, asset]


def test_missing_referenced_scene_asset_rejects_before_physics(accepted):
    scene = accepted/"sequence/scene/wuji_right/primitive_sdf.xml"
    tree = ET.parse(scene)
    ET.SubElement(ET.SubElement(tree.getroot(), "asset"), "mesh", name="missing", file="absent.obj")
    tree.write(scene)
    with pytest.raises(ValueError, match="scene asset is missing"):
        spider_runner.validate(config(accepted))


def test_saved_accepted_input_hashes_actual_external_primitive_asset(forbid_forceaware):
    source = ROOT/"output/grasp_contactaware_feature_normals_final/001_radius_0.03500/1_Large_Diameter/input_00"
    if not (source/"runtime_scene.xml").is_file():
        pytest.skip("Saved accepted CA fixture is not present; generic asset-path tests remain independent")
    reference = spider_reference.read_input(source)
    asset_node = ET.parse(reference.scene).find("./asset/mesh[@name='grasp_object_mesh']")
    primitive = Path(asset_node.get("file")).resolve()
    assert primitive.is_absolute() and primitive.is_file()
    assert reference.source_hashes[str(primitive)] == spider_reference.sha256(primitive)
    assert reference.source_hashes[str(reference.scene)] == spider_reference.sha256(reference.scene)


def test_missing_physical_scene_is_rejected_during_validation(accepted, forbid_forceaware):
    (accepted/"sequence/scene/wuji_right/primitive_sdf.xml").unlink()
    with pytest.raises(ValueError, match="missing its ordinary physical scene"):
        spider_runner.validate(config(accepted))


@pytest.mark.parametrize("preferred", ["runtime_scene.xml", "sequence/scene/wuji_right/primitive_mesh.xml"])
def test_scene_priority_is_explicit_and_selected_xml_is_hashed(accepted, preferred):
    target = accepted/preferred
    shutil.copy2(accepted/"sequence/scene/wuji_right/primitive_sdf.xml", target)
    reference = spider_reference.read_input(accepted)
    assert reference.scene == target
    assert reference.source_hashes[str(target)] == spider_reference.sha256(target)


@pytest.mark.parametrize("mutation,match", [
    ("rejected", "failed or unverified"), ("wrong_commit", "failed or unverified"),
    ("missing_required_contact", "required-finger acceptance"),
    ("moving_object", "stationary"), ("wrong_hand_trajectory", "accepted stationary"),
    ("different_guidance_mask", "active mask disagree"), ("invalid_point", "Malformed ContactAware guidance"),
])
def test_rejected_or_inconsistent_ca_cannot_seed_spider(accepted, forbid_forceaware, mutation, match):
    report_path = accepted/"contactaware_reference.json"
    report = json.loads(report_path.read_text())
    if mutation == "rejected":
        report["success"] = False
    elif mutation == "wrong_commit":
        report["source_commit"] = "unknown"
    elif mutation == "missing_required_contact":
        report["required_finger_contact"]["pinky"]["contact"] = False
    elif mutation in ("moving_object", "wrong_hand_trajectory"):
        path = accepted/"contactaware"/("object_pose_7.npy" if mutation == "moving_object" else "wuji_right_qpos.npy")
        value = np.load(path); value[1, -1] += .001; np.save(path, value)
    else:
        path = accepted/"contactaware/contact_guidance.npz"
        with np.load(path) as saved:
            value = {key: saved[key].copy() for key in saved.files}
        if mutation == "different_guidance_mask":
            value["contact_mask"][1, 0] = 0.
        else:
            value["hand_query_local_pos"][0, 0, 0] = np.nan
        np.savez(path, **value)
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=match):
        spider_runner.validate(config(accepted))


def test_input_selection_is_explicit_and_does_not_enter_forceaware(accepted, tmp_path, forbid_forceaware):
    pool = tmp_path/"pool"
    a, b = pool/"a", pool/"b"
    shutil.copytree(accepted, a); shutil.copytree(accepted, b)
    assert spider_runner.validate(config(pool, {"case_indices": [1, 0]})) == [b.resolve(), a.resolve()]
    assert spider_runner.validate(config(accepted)) == [accepted]


@pytest.mark.parametrize("indices", [[], [0, 0], [-1], [1], [True], [0.], "0", [[0]], [{"index": 0}]])
def test_invalid_case_selection_rejects(accepted, indices):
    with pytest.raises(ValueError, match="case_indices"):
        spider_runner.validate(config(accepted, {"case_indices": indices}))


def test_validation_checks_source_bytes_before_using_ca_or_physics(accepted, monkeypatch):
    def source_failure():
        raise ValueError("test source hash mismatch")

    def must_not_read(*args):
        raise AssertionError("Inputs must not be used after failed source verification")

    monkeypatch.setattr(spider_source, "verify_sources", source_failure)
    monkeypatch.setattr(spider_runner, "read_input", must_not_read)
    with pytest.raises(ValueError, match="test source hash mismatch"):
        spider_runner.validate(config(accepted))


def test_preload_bridge_requests_only_static_control_proposal(accepted, tmp_path, monkeypatch, forbid_forceaware):
    from retarget.env.grasp import forceaware_preload
    reference = spider_reference.read_input(accepted)
    initial = reference.qpos.copy()
    model_path = tmp_path/"ordinary_model.xml"
    calls, proposed = [], np.arange(26, dtype=np.float64)*.01

    def static_proposal(artifacts, cfg, **kwargs):
        calls.append((artifacts, cfg, kwargs))
        return proposed.copy(), {"static_qp_only": True, "physical_hold_certified": False}

    monkeypatch.setattr(forceaware_preload, "prepare_preload", static_proposal)
    command, report = spider_reference.preload(reference, model_path, 5)
    np.testing.assert_array_equal(reference.qpos, initial)
    np.testing.assert_array_equal(command, proposed.astype(np.float32))
    assert report["static_qp_only"] is True and report["physical_hold_certified"] is False
    artifacts, cfg, kwargs = calls[0]
    assert artifacts.root == accepted and artifacts.viewer_xml == model_path
    assert cfg.sequence.hand == "wuji_right" and cfg.optimizer.horizon == 5
    assert kwargs == dict(minimum_per_contact=dict(thumb=.1, index=.1, middle=.1, ring=.2, pinky=.3),
                          minimum_total=4.)
    monkeypatch.setattr(forceaware_preload, "prepare_preload", lambda *args, **kwargs: (None, {"feasible": False}))
    with pytest.raises(ValueError, match="no feasible bounded static control"):
        spider_reference.preload(reference, model_path, 5)


@pytest.mark.parametrize("mutation", ["wrong_hand", "extra_hand_option", "cpu", "gradient", "wrong_name"])
def test_runner_validates_wuji2_mppi_cuda_contract(accepted, mutation):
    cfg = config(accepted)
    if mutation == "wrong_hand":
        cfg.task["options"]["hand_model"] = "wuji1"
    elif mutation == "extra_hand_option":
        cfg.task["options"]["reference_method"] = "template"
    elif mutation == "cpu":
        cfg.environment["device"] = "cpu"
    elif mutation == "gradient":
        cfg.solver = replace(cfg.solver, gradient_steps=1)
    else:
        cfg.task["name"] = "other"
    with pytest.raises(ValueError):
        spider_runner.validate(cfg)


def test_ordinary_model_serialization_preserves_physics_without_forceaware(accepted, tmp_path, forbid_forceaware):
    reference = spider_reference.read_input(accepted)
    before = spider_reference.sha256(reference.scene)
    original = mujoco.MjModel.from_xml_path(str(reference.scene))
    prepared, path = spider_reference.prepare_model(reference, tmp_path/"prepared", .002,
                                                    solver_iterations=20, ls_iterations=50)
    assert (prepared.nq, prepared.nv, prepared.nu) == (33, 32, 26)
    assert prepared.opt.timestep == .002
    assert prepared.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert prepared.opt.iterations == 20 and prepared.opt.ls_iterations == 50
    for name in ("body_mass", "body_inertia", "body_pos", "body_quat", "actuator_gainprm",
                 "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange", "geom_friction"):
        np.testing.assert_allclose(getattr(prepared, name), getattr(original, name), rtol=0, atol=1e-12)
    assert path.is_file() and spider_reference.sha256(reference.scene) == before


def test_unified_entry_point_routes_to_spider_without_forceaware(accepted, tmp_path, monkeypatch, forbid_forceaware):
    import run as entry
    from retarget.common import provenance
    cfg = config(accepted)
    output = tmp_path/"run_output"; output.mkdir()
    calls = []
    module = SimpleNamespace(validate=lambda value: calls.append(("validate", value)),
        run=lambda value, directory: (calls.append(("run", value, directory)), [])[1])

    def importer(name):
        assert name == "retarget.env.grasp.spider_runner"
        calls.append(("import", name))
        return module

    monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(save_config(cfg))])
    monkeypatch.setattr(entry.importlib, "import_module", importer)
    monkeypatch.setattr(entry, "prepare_output", lambda value: output)
    monkeypatch.setattr(provenance, "record", lambda *args: {"scope": "routing fixture; no GPU solve"})
    assert entry.main() == output
    assert [item[0] for item in calls] == ["import", "validate", "run"]
    assert json.loads((output/"status.json").read_text())["state"] == "completed"
    assert json.loads((output/"manifest.json").read_text())["method"] == "mppi"


def test_provenance_records_spider_sources_and_ca_input_manifests(accepted, tmp_path, monkeypatch, forbid_forceaware):
    from retarget.common import provenance
    cfg = config(accepted); save_config(cfg)
    output, case = tmp_path/"output", tmp_path/"output/case"
    case.mkdir(parents=True)
    shutil.copy2(accepted/"sequence/scene/wuji_right/primitive_sdf.xml", case/"model.xml")
    for name in ("ca_input_hashes.json", "reference_manifest.json", "control_initialization.json"):
        (case/name).write_text('{"scope":"format fixture"}')
    cases = [dict(model="case/model.xml", trajectory="case/trajectory.npz")]
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "CPU configuration test fixture")
    result = provenance.record(cfg, cases, output)
    assert result["spider"]["commit"] == "44717007de41cbef7565dff7ff9f4453557a2d3d"
    assert result["spider"]["verified_files"] == 17
    assert result["contactaware_inputs_changed"] is False and result["forceaware_optimizer_used"] is False
    assert "dexforge" not in result
    for name in ("ca_input_hashes.json", "reference_manifest.json", "control_initialization.json"):
        assert result["input_sha256"][str(case/name)] == spider_reference.sha256(case/name)
    assert "retarget/env/grasp/spider_runner.py" in result["source_sha256"]


def test_pinned_source_verification_rejects_missing_changed_and_wrong_revision(tmp_path, monkeypatch):
    original = ROOT/"third_party/SPIDER/dex"
    target = tmp_path/"third_party/SPIDER/dex"
    shutil.copytree(original, target)
    monkeypatch.setattr(spider_source, "ROOT", tmp_path)
    assert spider_source.verify_sources()["verified_files"] == 17
    path = target/"spider/optimizers/sampling.py"
    saved = path.read_bytes(); path.write_bytes(saved+b"\n# changed\n")
    with pytest.raises(ValueError, match="verification failed: spider/optimizers/sampling.py"):
        spider_source.verify_sources()
    path.write_bytes(saved); path.unlink()
    with pytest.raises(ValueError, match="verification failed"):
        spider_source.verify_sources()
    path.write_bytes(saved)
    manifest_path = target/"UPSTREAM.json"
    manifest = json.loads(manifest_path.read_text()); manifest["commit"] = "other"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unexpected SPIDER source revision"):
        spider_source.verify_sources()
