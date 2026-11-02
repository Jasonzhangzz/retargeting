"""Controller variants change only the three requested virtual wrist servos."""

import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from retarget.env.grasp.forceaware_controller import apply_virtual_wrist_rotation_pd, scale_finger_servos


def test_virtual_wrist_variant_preserves_mass_fingers_limits_and_source_files(tmp_path):
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    world = ET.SubElement(root, "worldbody")
    hand = ET.SubElement(world, "body", name="palm")
    ET.SubElement(hand, "inertial", pos="0 0 0", mass=".7", diaginertia=".01 .02 .025")
    names = ["tx", "ty", "tz", "rx", "ry", "rz"]
    for index, name in enumerate(names):
        ET.SubElement(hand, "joint", name=name, type="slide" if index < 3 else "hinge",
            axis=" ".join(map(str, np.eye(3)[index % 3])), range="-2 2", armature=".0002")
    finger = ET.SubElement(hand, "body", name="finger")
    ET.SubElement(finger, "inertial", pos="0 0 .02", mass=".02", diaginertia=".0001 .0001 .0001")
    ET.SubElement(finger, "joint", name="finger_joint", axis="1 0 0", range="-1 1")
    obj = ET.SubElement(world, "body", name="object")
    ET.SubElement(obj, "freejoint", name="obj_joint")
    ET.SubElement(obj, "inertial", pos="0 0 0", mass=".2", diaginertia=".001 .001 .001")
    actuators = ET.SubElement(root, "actuator")
    # Deliberately scramble actuator order: DOFs 3:6 determine the wrist,
    # rather than assuming actuator indices 3:6 or matching a name prefix.
    for name in ("finger_joint", "rz", "tx", "ry", "ty", "rx", "tz"):
        ET.SubElement(actuators, "general", name=f"servo_{name}", joint=name,
            biastype="affine", gainprm="100 0 0 0 0 0 0 0 0 0",
            biasprm="0 -100 -10 0 0 0 0 0 0 0", forcerange="-3 3", ctrlrange="-2 2")
    native, viewer = tmp_path / "native.xml", tmp_path / "viewer.xml"
    ET.ElementTree(root).write(viewer)
    custom = ET.SubElement(root, "custom")
    ET.SubElement(custom, "numeric", name="native_marker", data="17")
    ET.ElementTree(root).write(native)
    source_bytes = [path.read_bytes() for path in (native, viewer)]
    original = mujoco.MjModel.from_xml_path(str(viewer))
    out_native, out_viewer, manifest = apply_virtual_wrist_rotation_pd(native, viewer, tmp_path / "variant")
    assert [path.read_bytes() for path in (native, viewer)] == source_bytes
    modified = mujoco.MjModel.from_xml_path(str(out_viewer))
    selected = [original.actuator(f"servo_{name}").id for name in ("rx", "ry", "rz")]
    expected_gain, expected_bias = original.actuator_gainprm.copy(), original.actuator_biasprm.copy()
    expected_gain[selected, 0] = 10.
    expected_bias[selected, 1], expected_bias[selected, 2] = -10., -.1
    np.testing.assert_array_equal(modified.actuator_gainprm, expected_gain)
    np.testing.assert_array_equal(modified.actuator_biasprm, expected_bias)
    for key in ("body_mass", "body_inertia", "dof_armature", "actuator_forcerange", "actuator_ctrlrange"):
        np.testing.assert_array_equal(getattr(original, key), getattr(modified, key))
    selected_names = {f"servo_{name}" for name in ("rx", "ry", "rz")}
    for source, result in ((native, out_native), (viewer, out_viewer)):
        for before, after in zip(ET.parse(source).getroot().iter(), ET.parse(result).getroot().iter(), strict=True):
            expected = dict(before.attrib)
            if before.get("name") in selected_names:
                expected.update(gainprm="10 0 0 0 0 0 0 0 0 0", biasprm="0 -10 -0.10000000000000001 0 0 0 0 0 0 0")
            assert after.attrib == expected
    report = json.loads(manifest.read_text())
    assert report["controller_changed"] and not report["upstream_core_modified"]
    assert all(report["compiled_invariants"].values())
    assert {row["joint"] for row in report["changes"]} == {"rx", "ry", "rz"}
    assert report["source_sha256"]["native"] == hashlib.sha256(source_bytes[0]).hexdigest()
    with pytest.raises(ValueError, match="overwrite"):
        apply_virtual_wrist_rotation_pd(out_native, out_viewer, out_native.parent)


def test_finger_gain_scaling_changes_exactly_twenty_servos_and_keeps_wrist_and_force_limits(tmp_path):
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    world = ET.SubElement(root, "worldbody")
    hand = ET.SubElement(world, "body", name="palm")
    ET.SubElement(hand, "inertial", pos="0 0 0", mass=".7", diaginertia=".01 .02 .025")
    joints = ["tx", "ty", "tz", "rx", "ry", "rz"]
    for index, name in enumerate(joints):
        ET.SubElement(hand, "joint", name=name, type="slide" if index < 3 else "hinge",
            axis=" ".join(map(str, np.eye(3)[index % 3])), range="-2 2", armature=".0002")
    for finger in range(5):
        parent = hand
        for link in range(4):
            name = f"finger{finger}_{link}"; joints.append(name)
            parent = ET.SubElement(parent, "body", name=name)
            ET.SubElement(parent, "inertial", pos="0 0 .02", mass=".02", diaginertia=".0001 .0001 .0001")
            ET.SubElement(parent, "joint", name=name, axis="1 0 0", range="-1 1", armature=".0002")
    actuators = ET.SubElement(root, "actuator")
    for joint_index in np.random.default_rng(21).permutation(26):
        name = joints[joint_index]
        kp, kv = (100., 10.) if joint_index < 6 else (.2 + joint_index*.013, .01 + joint_index*.0017)
        ET.SubElement(actuators, "general", name=f"servo_{name}", joint=name, biastype="affine",
            gainprm=f"{kp:.17g} 0 0 0 0 0 0 0 0 0", biasprm=f"0 {-kp:.17g} {-kv:.17g} 0 0 0 0 0 0 0",
            forcerange="-.3 .3", ctrlrange="-1.2 1.3")
    source = tmp_path / "source.xml"; ET.ElementTree(root).write(source)
    # Scaling must also preserve a previously explicit wrist-controller variant.
    wrist_native, wrist_viewer, _ = apply_virtual_wrist_rotation_pd(source, source, tmp_path / "wrist")
    before_bytes = [p.read_bytes() for p in (wrist_native, wrist_viewer)]
    original = mujoco.MjModel.from_xml_path(str(wrist_viewer))
    native, viewer, manifest = scale_finger_servos(wrist_native, wrist_viewer, tmp_path / "finger_scaled")
    modified = mujoco.MjModel.from_xml_path(str(viewer))
    finger_ids = [original.actuator(f"servo_{name}").id for name in joints[6:]]
    wrist_ids = [original.actuator(f"servo_{name}").id for name in joints[:6]]
    expected_gain, expected_bias = original.actuator_gainprm.copy(), original.actuator_biasprm.copy()
    expected_gain[finger_ids, 0] *= 2.
    expected_bias[finger_ids, 1] *= 2.
    expected_bias[finger_ids, 2] *= np.sqrt(2.)
    np.testing.assert_array_equal(modified.actuator_gainprm, expected_gain)
    np.testing.assert_array_equal(modified.actuator_biasprm, expected_bias)
    np.testing.assert_array_equal(modified.actuator_gainprm[wrist_ids], original.actuator_gainprm[wrist_ids])
    np.testing.assert_array_equal(modified.actuator_biasprm[wrist_ids], original.actuator_biasprm[wrist_ids])
    for key in ("body_mass", "body_inertia", "dof_armature", "actuator_forcerange", "actuator_ctrlrange"):
        np.testing.assert_array_equal(getattr(original, key), getattr(modified, key))
    # The same unsaturated spring torque needs half the positional command
    # offset after doubling kp, while the available physical torque stays fixed.
    np.testing.assert_allclose(original.actuator_gainprm[finger_ids, 0]*.3,
                               modified.actuator_gainprm[finger_ids, 0]*.15)
    assert [p.read_bytes() for p in (wrist_native, wrist_viewer)] == before_bytes
    for old_path, new_path in ((wrist_native, native), (wrist_viewer, viewer)):
        for old, new in zip(ET.parse(old_path).getroot().iter(), ET.parse(new_path).getroot().iter(), strict=True):
            if old.tag == "general" and old.get("joint") in joints[6:]:
                assert {k:v for k,v in old.attrib.items() if k not in ("gainprm","biasprm")} == {k:v for k,v in new.attrib.items() if k not in ("gainprm","biasprm")}
            else:
                assert old.attrib == new.attrib
    report = json.loads(manifest.read_text())
    assert report["controller_changed"] and report["calibration_kind"] == "controller gain calibration"
    assert not report["original_controller_parameters_preserved"] and not report["upstream_core_modified"]
    assert report["finger_servo_count"] == 20 and report["wrist_controller_parameters_preserved"]
    assert all(report["compiled_invariants"].values())
    with pytest.raises(ValueError, match="finite"):
        scale_finger_servos(wrist_native, wrist_viewer, tmp_path / "invalid", kp_scale=float("nan"))
