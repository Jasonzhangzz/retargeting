"""Explicit, output-local controller gain calibration for FA scenes.

This changes controller parameters, never the pinned FA algorithms. The
default scene and source files remain unchanged. Wrist and finger variants
are separate opt-in operations, each limited to its selected servo coefficients.
"""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parameters(text, size=10):
    values = np.fromstring(text or "", sep=" ")
    if not len(values) or len(values) > size or not np.isfinite(values).all():
        raise ValueError("Controller servos require explicit finite gainprm/biasprm")
    return np.pad(values, (0, size-len(values)))


def _rotation_servos(model):
    if model.nv < 6:
        raise ValueError("Virtual wrist requires six leading scalar degrees of freedom")
    joints = []
    for dof in range(6):
        candidates = np.flatnonzero(model.jnt_dofadr == dof)
        if len(candidates) != 1:
            raise ValueError("Virtual wrist must use six leading scalar joints")
        joint = int(candidates[0])
        expected = mujoco.mjtJoint.mjJNT_SLIDE if dof < 3 else mujoco.mjtJoint.mjJNT_HINGE
        if model.jnt_type[joint] != expected or model.jnt_qposadr[joint] != dof:
            raise ValueError("Virtual wrist must contain three translations then three rotations")
        joints.append(joint)
    return _position_servos(model, joints[3:6])


def _position_servos(model, joints):
    result = []
    for joint in joints:
        actuators = np.flatnonzero((model.actuator_trnid[:, 0] == joint)
            & (model.actuator_trntype == int(mujoco.mjtTrn.mjTRN_JOINT)))
        if len(actuators) != 1:
            raise ValueError("Each selected scalar joint must have one direct position servo")
        actuator = int(actuators[0])
        if (model.actuator_gaintype[actuator] != int(mujoco.mjtGain.mjGAIN_FIXED)
                or model.actuator_biastype[actuator] != int(mujoco.mjtBias.mjBIAS_AFFINE)
                or model.actuator_dyntype[actuator] != int(mujoco.mjtDyn.mjDYN_NONE)
                or not np.array_equal(model.actuator_gear[actuator], [1., 0., 0., 0., 0., 0.])
                or model.actuator_gainprm[actuator, 0] <= 0
                or not np.isclose(model.actuator_biasprm[actuator, 1], -model.actuator_gainprm[actuator, 0])
                or model.actuator_biasprm[actuator, 2] > 0):
            raise ValueError("Controller calibration requires fixed-gain affine position PD servos with unit gear")
        name, joint_name = model.actuator(actuator).name, model.joint(joint).name
        if not name or not joint_name:
            raise ValueError("Selected position servos and joints must be named")
        result.append((actuator, name, joint_name))
    return result


def _edited_tree(path, selected, model, kp, kv):
    tree = ET.parse(path)
    section = tree.getroot().find("actuator")
    if section is None:
        raise ValueError("Scene is missing actuator definitions")
    changes = []
    for actuator, name, joint in selected:
        new_kp = float(kp if np.ndim(kp) == 0 else kp[actuator])
        new_kv = float(kv if np.ndim(kv) == 0 else kv[actuator])
        matches = [node for node in section if node.get("name") == name]
        if len(matches) != 1 or matches[0].tag != "general" or matches[0].get("joint") != joint:
            raise ValueError(f"Scene must preserve canonical general servo {name} for joint {joint}")
        node = matches[0]
        gain, bias = _parameters(node.get("gainprm")), _parameters(node.get("biasprm"))
        if not (np.allclose(gain, model.actuator_gainprm[actuator], rtol=0., atol=1e-12)
                and np.allclose(bias, model.actuator_biasprm[actuator], rtol=0., atol=1e-12)):
            raise ValueError(f"Native/viewer controller parameters disagree for {name}")
        before = dict(gainprm=node.get("gainprm"), biasprm=node.get("biasprm"))
        gain_tokens, bias_tokens = before["gainprm"].split(), before["biasprm"].split()
        if len(bias_tokens) < 3:
            bias_tokens += ["0"] * (3-len(bias_tokens))
        gain_tokens[0], bias_tokens[1], bias_tokens[2] = format(new_kp, ".17g"), format(-new_kp, ".17g"), format(-new_kv, ".17g")
        node.set("gainprm", " ".join(gain_tokens)); node.set("biasprm", " ".join(bias_tokens))
        changes.append(dict(actuator=name, joint=joint, actuator_id=actuator,
            before=before, after=dict(gainprm=node.get("gainprm"), biasprm=node.get("biasprm")),
            kp_before=float(gain[0]), kv_before=float(-bias[2]), kp_after=new_kp, kv_after=new_kv))
    # Audit every XML attribute rather than relying on the edit loop's intent.
    previous = list(ET.parse(path).getroot().iter())
    allowed = {name for _, name, _ in selected}
    for old, new in zip(previous, tree.getroot().iter(), strict=True):
        if old.tag != new.tag:
            raise RuntimeError("Controller variant changed XML topology")
        for key in set(old.attrib) | set(new.attrib):
            if old.tag == "general" and old.get("name") in allowed and key in ("gainprm", "biasprm"):
                continue
            if old.get(key) != new.get(key):
                raise RuntimeError(f"Controller variant changed unrelated XML attribute {key}")
    return tree, changes


def apply_virtual_wrist_rotation_pd(native_xml, viewer_xml, output_dir, *, kp=10., kv=.1):
    """Return native, viewer, manifest paths for an explicit controller variant.

    Inputs are the prepared canonical scenes with absolute asset paths. Only
    gainprm[0], biasprm[1] and biasprm[2] of the three wrist rotation servos
    change. Force limits and every other XML attribute remain identical.
    """
    if isinstance(kp, bool) or isinstance(kv, bool):
        raise ValueError("Virtual wrist kp/kv must be finite numeric coefficients")
    kp, kv = float(kp), float(kv)
    if not np.isfinite(kp) or kp <= 0 or not np.isfinite(kv) or kv < 0:
        raise ValueError("Virtual wrist kp must be positive and kv nonnegative, both finite")
    source_native, source_viewer = Path(native_xml).resolve(), Path(viewer_xml).resolve()
    original = mujoco.MjModel.from_xml_path(str(source_viewer))
    selected = _rotation_servos(original)
    return _write_variant(source_native, source_viewer, output_dir, original, selected, kp, kv,
        scope="only three virtual wrist rotation servo P/D coefficients", settings=dict(kp=kp, kv=kv))


def _write_variant(source_native, source_viewer, output_dir, original, selected, kp, kv, *, scope, settings):
    destination = Path(output_dir).resolve()
    native, viewer, manifest = destination / "native_scene.xml", destination / "source_scene.xml", destination / "controller_manifest.json"
    if {source_native, source_viewer} & {native, viewer, manifest}:
        raise ValueError("Controller variant output must not overwrite either source scene")
    native_tree, changes = _edited_tree(source_native, selected, original, kp, kv)
    viewer_tree, _ = _edited_tree(source_viewer, selected, original, kp, kv)
    destination.mkdir(parents=True, exist_ok=True)
    native_tree.write(native, encoding="utf-8", xml_declaration=False)
    viewer_tree.write(viewer, encoding="utf-8", xml_declaration=False)
    modified = mujoco.MjModel.from_xml_path(str(viewer))
    selected_ids = [actuator for actuator, _, _ in selected]
    invariants = {}
    for name in ("body_mass", "body_inertia", "body_pos", "body_quat", "body_ipos", "body_iquat",
                 "dof_armature", "dof_damping", "actuator_forcerange", "actuator_ctrlrange",
                 "jnt_actfrclimited", "jnt_actfrcrange",
                 "actuator_forcelimited", "actuator_ctrllimited", "actuator_gear", "actuator_trnid", "actuator_dynprm"):
        invariants[name] = bool(np.array_equal(getattr(original, name), getattr(modified, name)))
        if not invariants[name]:
            raise RuntimeError(f"Controller variant changed compiled {name}")
    expected_gain, expected_bias = original.actuator_gainprm.copy(), original.actuator_biasprm.copy()
    new_kp = np.broadcast_to(kp, (original.nu,))[selected_ids]
    new_kv = np.broadcast_to(kv, (original.nu,))[selected_ids]
    expected_gain[selected_ids, 0] = new_kp
    expected_bias[selected_ids, 1], expected_bias[selected_ids, 2] = -new_kp, -new_kv
    for name, expected in (("actuator_gainprm", expected_gain), ("actuator_biasprm", expected_bias)):
        invariants[name + "_only_requested_coefficients"] = bool(np.array_equal(expected, getattr(modified, name)))
        if not invariants[name + "_only_requested_coefficients"]:
            raise RuntimeError(f"Controller variant changed unrelated compiled {name} coefficients")
    report = dict(controller_changed=True, original_controller_parameters_preserved=False,
        upstream_core_modified=False, calibration_kind="controller gain calibration", scope=scope,
        changes=changes, compiled_invariants=invariants,
        source_native=str(source_native), source_viewer=str(source_viewer),
        native_xml=str(native), viewer_xml=str(viewer),
        source_sha256=dict(native=_sha256(source_native), viewer=_sha256(source_viewer)),
        output_sha256=dict(native=_sha256(native), viewer=_sha256(viewer)))
    report.update(settings)
    manifest.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return native, viewer, manifest


def scale_finger_servos(native_xml, viewer_xml, output_dir, *, kp_scale=2., kv_scale=np.sqrt(2.)):
    """Scale the 20 scalar finger PD servos after the six virtual wrist DOFs.

    This explicit controller variant scales gainP and biasP together, and
    biasV by the damping scale. All wrist servo coefficients, physical torque
    limits, control ranges, other coefficients, masses and inertia stay fixed.
    """
    if isinstance(kp_scale, bool) or isinstance(kv_scale, bool):
        raise ValueError("Finger servo scale factors must be finite numeric values")
    kp_scale, kv_scale = float(kp_scale), float(kv_scale)
    if not np.isfinite(kp_scale) or kp_scale <= 0 or not np.isfinite(kv_scale) or kv_scale < 0:
        raise ValueError("Finger kp scale must be positive and kv scale nonnegative, both finite")
    source_native, source_viewer = Path(native_xml).resolve(), Path(viewer_xml).resolve()
    original = mujoco.MjModel.from_xml_path(str(source_viewer))
    _rotation_servos(original)
    if original.nu != 26:
        raise ValueError("Finger calibration requires six virtual wrist and exactly 20 finger servos")
    joints = []
    for dof in range(6, 26):
        candidates = np.flatnonzero(original.jnt_dofadr == dof)
        if len(candidates) != 1:
            raise ValueError("Finger calibration requires 20 scalar joints after the virtual wrist")
        joint = int(candidates[0])
        if (original.jnt_type[joint] not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
                or original.jnt_qposadr[joint] != dof):
            raise ValueError("Finger calibration requires 20 scalar joints after the virtual wrist")
        joints.append(joint)
    selected = _position_servos(original, joints)
    kp, kv = original.actuator_gainprm[:, 0].copy(), -original.actuator_biasprm[:, 2].copy()
    selected_ids = [actuator for actuator, _, _ in selected]
    kp[selected_ids] *= kp_scale
    kv[selected_ids] *= kv_scale
    if not np.isfinite(kp).all() or not np.isfinite(kv).all():
        raise ValueError("Scaled finger PD coefficients must remain finite")
    return _write_variant(source_native, source_viewer, output_dir, original, selected, kp, kv,
        scope="only twenty finger position-servo P/D coefficients; all six virtual wrist servos unchanged",
        settings=dict(kp_scale=kp_scale, kv_scale=kv_scale, finger_servo_count=len(selected),
                      wrist_controller_parameters_preserved=True))
