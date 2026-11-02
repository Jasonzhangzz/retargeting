"""Input/topology bridge tests; these do not claim synthetic CA success."""

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from retarget.dynamics.dexforge_upstream import COMMIT, activate_upstream, verify_upstream
from retarget.env.grasp.forceaware_reference import (
    HAND, TIP_SITES, create_optimizer, load_reference, make_config,
    prepare_reference, register_wuji2_hand,
)


@pytest.fixture
def prepared(tmp_path):
    """A tiny scalar-joint scene tests format and remapping, not CA geometry."""
    source = tmp_path / "ca_input"
    scene = source / "sequence/scene/wuji_right"
    contacts = source / "contactaware"
    scene.mkdir(parents=True); contacts.mkdir()
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    world = ET.SubElement(root, "worldbody")
    palm = ET.SubElement(world, "body", name="right_palm_link")
    inertia = dict(pos="0 0 0", mass="1", diaginertia=".01 .01 .01")
    ET.SubElement(palm, "inertial", **inertia)
    joint_names = []
    for i, label in enumerate(("x", "y", "z", "rx", "ry", "rz")):
        axis = np.eye(3)[i % 3]
        name = f"wrist_{label}"; joint_names.append(name)
        ET.SubElement(palm, "joint", name=name, type="slide" if i < 3 else "hinge",
            axis=" ".join(map(str, axis)), range="-2 2")
    names = ("thumb", "index", "middle", "ring", "pinky")
    for finger, name in enumerate(names, 1):
        parent = palm
        for link in range(1, 5):
            body = ET.SubElement(parent, "body", name=f"right_finger{finger}_link{link}", pos="0 0 .02")
            ET.SubElement(body, "inertial", **inertia)
            joint = f"finger_{finger}_{link}"; joint_names.append(joint)
            ET.SubElement(body, "joint", name=joint, axis="1 0 0", range="-1 1")
            parent = body
        ET.SubElement(parent, "site", name=f"right_{name}_tip", pos="0 0 .02")
    obj = ET.SubElement(world, "body", name="grasp_object")
    ET.SubElement(obj, "freejoint", name="obj_joint")
    ET.SubElement(obj, "inertial", **inertia)
    actuators = ET.SubElement(root, "actuator")
    for name in joint_names:
        ET.SubElement(actuators, "position", name=f"a_{name}", joint=name, kp="1", ctrlrange="-2 2")
    xml = scene / "primitive_sdf.xml"
    ET.ElementTree(root).write(xml)
    model = mujoco.MjModel.from_xml_path(str(xml))
    qref = np.linspace(-.05, .05, 26).astype(np.float32)
    bodies = np.asarray([model.body(f"right_finger{i}_link4").id for i in range(1, 6)])
    anchors = np.arange(15, dtype=np.float32).reshape(5, 3) * .001
    np.savez(source / "card_input.npz", hand_reference_qpos=qref,
        hand_reference_source=np.asarray("dexforge_contactaware"),
        reference_contact_mask=np.ones(5, np.float32), object_points=anchors)
    (source / "contactaware_reference.json").write_text(json.dumps(dict(
        success=True, source_commit=COMMIT,
        required_finger_contact={name:dict(contact=True) for name in names})))
    guidance = dict(contact_mask=np.ones((2, 5), np.float32),
        contact_weight=np.ones((2, 5), np.float32),
        contact_pos_obj=np.tile(anchors, (2, 1, 1)),
        hand_query_body_ids=np.tile(bodies, (2, 1)),
        hand_query_local_pos=np.tile([[0, 0, .01]], (2, 5, 1)).astype(np.float32))
    np.savez(contacts / "contact_guidance.npz", **guidance)
    np.save(contacts / f"{HAND}_qpos.npy", np.tile(qref, (2, 1)))
    np.save(contacts / "object_pose_7.npy", np.tile([0, 0, 0, 1, 0, 0, 0], (2, 1)).astype(np.float32))
    (source / "sequence/mano_raw").mkdir()
    (source / "sequence/mano_raw/meta.json").write_text('{"object":"primitive"}')
    native = tmp_path / "native.xml"
    # Equivalent explicit defaults plus a changed sibling ordering exercise
    # material frames, contact body IDs, and scalar hand-reference remapping.
    palm.set("pos", "0 0 0"); palm.set("quat", "1 0 0 0")
    last = palm.find("body[@name='right_finger5_link1']")
    palm.remove(last); palm.insert(7, last)
    ET.ElementTree(root).write(native)
    return source, native, xml, qref


def test_config_preserves_official_forceaware_defaults(prepared, tmp_path):
    source, native, visual, qref = prepared
    artifacts = prepare_reference(source, tmp_path / "bundle", native, viewer_xml=visual)
    cfg = make_config(artifacts, duration_seconds=.4)
    activate_upstream()
    from forceaware import config as fa
    defaults = fa.load_config()
    assert asdict(cfg.loss) == defaults["loss"]
    assert asdict(cfg.smoothing) == defaults["smoothing"]
    assert asdict(cfg.optimizer) == defaults["optimizer"]
    assert asdict(cfg.bounds) == defaults["bounds"]
    assert cfg.simulator.time_grid.mpc_dt == cfg.simulator.time_grid.exec_dt == .01
    assert cfg.sequence.hand == HAND and cfg.sequence.hold_frame == 0
    assert cfg.sequence.hold_seconds == .4
    assert artifacts.config_path.is_file()
    assert cfg.sequence.output_dir_name == "retarget/wuji_right/forceaware"


class Array:
    def __init__(self, value): self.value = np.asarray(value)
    def numpy(self): return self.value


def test_body_mapping_static_targets_and_original_window_loader(prepared, tmp_path):
    source, native, visual, qref = prepared
    artifacts = prepare_reference(source, tmp_path / "bundle", native, viewer_xml=visual)
    cfg = make_config(artifacts, duration_seconds=.1,
        optimizer_overrides=dict(horizon=3, n_iter=10, multistart_count=8))
    model = mujoco.MjModel.from_xml_path(str(native))
    bodies = [model.body(f"right_finger{i}_link4").id for i in range(1, 6)]
    device = SimpleNamespace(gaussian_collision=SimpleNamespace(
        source_body_ids=Array(bodies), output_starts=Array(np.arange(5)),
        target_output_ids=Array(np.arange(5))))
    reference = load_reference(artifacts, model, device, cfg)
    assert reference.qpos.shape == (33,) and reference.qvel.shape == (32,)
    np.testing.assert_array_equal(reference.qvel, 0)
    np.testing.assert_array_equal(reference.sequence.contact.query_body[0], bodies)
    assert reference.targets.actions.shape == (3, 26)
    assert reference.targets.dense_actions.shape == (18, 26)
    assert reference.targets.contact_mask.shape == (18, 5)
    np.testing.assert_array_equal(reference.targets.contact_mask, 1)
    np.testing.assert_allclose(reference.targets.actions,
        np.broadcast_to(reference.sequence.hand[0], reference.targets.actions.shape))
    np.testing.assert_allclose(reference.targets.object_qpos, np.tile([0, 0, 0, 1, 0, 0, 0], (18, 1)), atol=1e-7)
    assert np.all(reference.window(executed_steps=9).frames == 0)
    assert reference.metadata["original_first_window_initial_correction"]
    # Wrist and each finger joint refer to the same physical scalar joint,
    # even after the XML sibling ordering changes.
    old = mujoco.MjModel.from_xml_path(str(visual))
    for address in reference.index.hand_qpos:
        joint = np.flatnonzero(model.jnt_qposadr == address)[0]
        old_joint = old.joint(model.joint(joint).name).id
        assert reference.qpos[address] == pytest.approx(qref[old.jnt_qposadr[old_joint]])


def test_topology_registration_keeps_original_optimizer_and_source_bytes(monkeypatch):
    before = verify_upstream()
    first = register_wuji2_hand(); second = register_wuji2_hand()
    from forceaware import hand_specs, mpc_loss, servo_prior
    assert first == second
    assert sum(item.name == HAND for item in hand_specs.ASSET_HANDS) == 1
    assert mpc_loss._asset_hand_spec(HAND).tip_sites == TIP_SITES
    assert verify_upstream() == before
    args = []
    marker = object()
    monkeypatch.setattr(servo_prior, "ServoPriorOptimizer", lambda *a, **kw: (args.append((a, kw)), marker)[1])
    assert create_optimizer(1, 2, 3, cfg=4, guidance_count=5) is marker
    assert args == [((1, 2, 3), dict(cfg=4, guidance_count=5))]


def test_failed_ca_or_true_frame_change_is_rejected(prepared, tmp_path):
    source, native, visual, _ = prepared
    tree = ET.parse(native)
    tree.find(".//body[@name='right_finger1_link4']").set("pos", "0 0 .025")
    tree.write(native)
    with pytest.raises(ValueError, match="material-point frame"):
        prepare_reference(source, tmp_path / "wrong_frame", native, viewer_xml=visual)
    report = json.loads((source / "contactaware_reference.json").read_text())
    report["success"] = False
    (source / "contactaware_reference.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="failed or unverified"):
        prepare_reference(source, tmp_path / "failed_ca", visual, viewer_xml=visual)


def test_only_explicit_optimizer_and_grid_overrides_are_allowed(prepared, tmp_path):
    source, native, visual, _ = prepared
    artifacts = prepare_reference(source, tmp_path / "bundle", native, viewer_xml=visual)
    with pytest.raises(ValueError, match="optimizer overrides"):
        make_config(artifacts, optimizer_overrides=dict(contact_robustness=0))
    cfg = make_config(artifacts, duration_seconds=.1, simulator_overrides=dict(exec_dt=.002))
    assert cfg.simulator.time_grid.executor_substeps_per_mpc_step == 5


def test_joint_axis_change_cannot_reuse_ca_material_points(prepared, tmp_path):
    source, native, visual, _ = prepared
    tree = ET.parse(native)
    tree.find(".//joint[@name='finger_1_4']").set("axis", "0 1 0")
    tree.write(native)
    with pytest.raises(ValueError, match="joint coordinate convention: finger_1_4"):
        prepare_reference(source, tmp_path / "wrong_axis", native, viewer_xml=visual)
