"""Bridge accepted WUJI2 ContactAware inputs to the unmodified DexForge FA.

The bridge changes input paths, stationary references and the hand topology
descriptor.  Optimizer, objective, time interpolation, native dynamics,
execution and first-window initial-state policy remain the pinned FA code.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from retarget.dynamics.dexforge_upstream import COMMIT, activate_upstream, verify_upstream


HAND = "wuji_right"
TIP_SITES = {
    "th": "right_thumb_tip", "if": "right_index_tip", "mf": "right_middle_tip",
    "rf": "right_ring_tip", "pk": "right_pinky_tip",
}


@dataclass(frozen=True)
class ReferenceArtifacts:
    source_experiment: Path
    root: Path
    sequence_dir: Path
    contactaware_dir: Path
    xml: Path
    viewer_xml: Path
    config_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class ForceAwareReference:
    cfg: object
    sequence: object
    index: object
    qpos: np.ndarray
    qvel: np.ndarray
    metadata: dict

    def window(self, executed_steps=0):
        from forceaware.targets import make_window
        return make_window(self.sequence, executed_steps=executed_steps,
            horizon=self.cfg.optimizer.horizon, time_grid=self.cfg.simulator.time_grid,
            start_frame=float(self.cfg.sequence.start), end_frame=self.cfg.sequence.end,
            hold_frame=self.cfg.sequence.hold_frame)

    @property
    def targets(self):
        return self.window()


def register_wuji2_hand():
    """Supply a topology descriptor without editing or subclassing FA code."""
    activate_upstream()
    from forceaware import hand_specs, mpc_loss

    descriptor = hand_specs.HandSpec(HAND, "right_palm_link", TIP_SITES.copy())
    found = [item for item in hand_specs.ASSET_HANDS if item.name == HAND]
    if found and (len(found) != 1 or found[0] != descriptor):
        raise ValueError("An incompatible wuji_right ForceAware topology is already registered")
    if not found:
        hand_specs.ASSET_HANDS = (*hand_specs.ASSET_HANDS, descriptor)
    # mpc_loss imports the tuple by value.  Register the same external model
    # descriptor at its consumer; source bytes and all loss functions stay fixed.
    mpc_loss.ASSET_HANDS = hand_specs.ASSET_HANDS
    return dict(name=HAND, palm_body=descriptor.palm_body, tip_sites=TIP_SITES.copy(),
                registration="external topology descriptor; no upstream source changes")


def _read_ca(experiment):
    experiment = Path(experiment).expanduser().resolve()
    with np.load(experiment / "card_input.npz", allow_pickle=False) as raw:
        card = {key: raw[key].copy() for key in raw.files}
    source = card.get("hand_reference_source")
    if source is None or source.shape != () or str(source.item()) != "dexforge_contactaware":
        raise ValueError("ForceAware requires an explicit accepted DexForge ContactAware reference")
    qref = np.asarray(card.get("hand_reference_qpos"), dtype=np.float64)
    if qref.shape != (26,) or not np.isfinite(qref).all():
        raise ValueError("ContactAware hand reference must contain 26 finite Wuji coordinates")
    with (experiment / "contactaware_reference.json").open() as stream:
        report = json.load(stream)
    if report.get("success") is not True or report.get("source_commit") != COMMIT:
        raise ValueError("A failed or unverified ContactAware proposal cannot seed ForceAware")
    required = report.get("required_finger_contact", {})
    if not required or not all(row.get("contact") is True for row in required.values()):
        raise ValueError("ContactAware required-finger acceptance is missing")
    contacts = experiment / "contactaware"
    if not contacts.is_dir():
        contacts = experiment / "sequence/retarget" / HAND / "contactaware"
    with np.load(contacts / "contact_guidance.npz", allow_pickle=False) as raw:
        guidance = {key: raw[key].copy() for key in raw.files}
    mask = np.asarray(card.get("reference_contact_mask"), dtype=np.float32)
    if (mask.ndim != 1 or not mask.size or not np.isfinite(mask).all()
            or not np.isin(mask, (0, 1)).all() or not mask.any()):
        raise ValueError("ForceAware requires the explicit active ContactAware contact mask")
    frames, count = guidance["contact_mask"].shape
    if frames < 2 or count != len(mask) or not np.array_equal(
            guidance["contact_mask"], np.broadcast_to(mask, (frames, count))):
        raise ValueError("ContactAware guidance and accepted active mask disagree")
    expected = {"contact_weight": (frames, count), "contact_pos_obj": (frames, count, 3),
                "hand_query_body_ids": (frames, count), "hand_query_local_pos": (frames, count, 3)}
    for name, shape in expected.items():
        if guidance[name].shape != shape or not np.isfinite(guidance[name]).all():
            raise ValueError(f"Malformed ContactAware guidance: {name}")
    hand = np.load(contacts / f"{HAND}_qpos.npy", allow_pickle=False)
    poses = np.load(contacts / "object_pose_7.npy", allow_pickle=False)
    if hand.shape != (frames, 26) or not np.allclose(hand, qref[None], rtol=0, atol=2e-6):
        raise ValueError("Prepared hand trajectory is not the accepted stationary CA reference")
    if poses.shape != (frames, 7) or not np.isfinite(poses).all():
        raise ValueError("Malformed stationary object reference")
    if not np.allclose(poses, poses[:1], rtol=0, atol=1e-7):
        raise ValueError("Primitive ForceAware targets must keep the object stationary")
    return experiment, card, report, guidance, poses


def _scene_topology(path):
    root = ET.parse(path).getroot()
    world = root.find("worldbody")
    if world is None:
        raise ValueError("ForceAware input scene is missing worldbody")
    bodies = [None, *world.iter("body")]
    names = ["world", *[body.get("name") for body in bodies[1:]]]
    if any(name is None for name in names) or len(set(names)) != len(names):
        raise ValueError("ForceAware input bodies must have distinct explicit names")
    joints = [node for node in world.iter() if node.tag in ("joint", "freejoint")]
    object_index = next((i for i, joint in enumerate(joints) if joint.get("name") == "obj_joint"), None)
    if object_index is None:
        raise ValueError("ForceAware input scene is missing obj_joint")
    hand_joints = joints[:object_index]
    if len(hand_joints) != 26 or any(node.get("type", "hinge") in ("free", "ball") for node in hand_joints):
        raise ValueError("ForceAware Wuji scene must have 26 scalar hand joints before obj_joint")
    return root, names, {name: body for name, body in zip(names[1:], bodies[1:])}, [node.get("name") for node in hand_joints]


def _copy(path, destination):
    if Path(path).resolve() != Path(destination).resolve():
        shutil.copy2(path, destination)


def _frame_model(path):
    """Compile only kinematic frames, accepting native custom collision XML.

    Collision geoms, assets and actuation are irrelevant to material frames.
    Unit inertials here merely let MuJoCo compile this comparison-only model;
    neither physical scene nor its masses are written or changed.
    """
    import mujoco
    root = ET.parse(path).getroot()
    excluded = {"geom", "asset", "contact", "sensor", "tendon", "actuator", "keyframe", "equality", "inertial"}
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag in excluded:
                parent.remove(child)
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("fusestatic", "false")
    for body in root.findall(".//body"):
        ET.SubElement(body, "inertial", pos="0 0 0", mass="1", diaginertia="1 1 1")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def _diagnostic_camera():
    position = np.asarray([.35, -.4, .25])
    forward = -position / np.linalg.norm(position)
    right = np.cross(forward, [0., 0., 1.]); right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    quaternion = Rotation.from_matrix(np.column_stack((right, down, forward))).as_quat()
    return np.r_[quaternion, position].astype(np.float32)


def prepare_reference(source_experiment, destination, native_scene_xml, *, viewer_xml=None):
    """Write an output-local sequence accepted by original rolling_mpc_retarget."""
    source, card, report, guidance, poses = _read_ca(source_experiment)
    destination = Path(destination).expanduser().resolve()
    if destination == source or destination.is_relative_to(source):
        raise ValueError("ForceAware reference output must not modify its source CA experiment")
    source_xml = source / "sequence/scene" / HAND / "primitive_sdf.xml"
    native_scene_xml = Path(native_scene_xml).expanduser().resolve()
    viewer_xml = source_xml if viewer_xml is None else Path(viewer_xml).expanduser().resolve()
    _, old_names, old_bodies, old_joints = _scene_topology(source_xml)
    _, new_names, new_bodies, new_joints = _scene_topology(native_scene_xml)
    if set(old_names) != set(new_names) or set(old_joints) != set(new_joints):
        raise ValueError("Native scene must preserve CA bodies and scalar joints")
    old_frames, new_frames = _frame_model(source_xml), _frame_model(native_scene_xml)
    position_errors, rotation_errors = [], []
    for name in old_bodies:
        old, new = old_frames.body(name).id, new_frames.body(name).id
        position_error = float(np.max(np.abs(old_frames.body_pos[old]-new_frames.body_pos[new])))
        same_position = position_error <= 1e-7
        a, b = old_frames.body_quat[old], new_frames.body_quat[new]
        rotation_error = float(min(np.linalg.norm(a-b), np.linalg.norm(a+b)))
        same_rotation = rotation_error <= 1e-6
        position_errors.append(position_error); rotation_errors.append(rotation_error)
        old_parent = old_frames.body(int(old_frames.body_parentid[old])).name
        new_parent = new_frames.body(int(new_frames.body_parentid[new])).name
        if not same_position or not same_rotation or old_parent != new_parent:
            raise ValueError(f"Native scene changed the CA material-point frame: {name}")
    if old_joints[:6] != new_joints[:6]:
        raise ValueError("Native scene changed the six CA wrist coordinate conventions")
    for name in old_joints:
        old, new = old_frames.joint(name).id, new_frames.joint(name).id
        old_address, new_address = old_frames.jnt_qposadr[old], new_frames.jnt_qposadr[new]
        if (old_frames.jnt_type[old] != new_frames.jnt_type[new]
                or not np.allclose(old_frames.jnt_pos[old], new_frames.jnt_pos[new], rtol=0, atol=1e-7)
                or not np.allclose(old_frames.jnt_axis[old], new_frames.jnt_axis[new], rtol=0, atol=1e-6)
                or not np.isclose(old_frames.qpos0[old_address], new_frames.qpos0[new_address], rtol=0, atol=1e-6)):
            raise ValueError(f"Native scene changed the CA joint coordinate convention: {name}")
    joint_map = np.asarray([old_joints.index(name) for name in new_joints])
    qref = card["hand_reference_qpos"][joint_map]
    mapping = np.asarray([new_names.index(name) for name in old_names], dtype=np.int32)
    body_ids = guidance["hand_query_body_ids"].astype(int)
    if body_ids.min() < 0 or body_ids.max() >= len(mapping):
        raise ValueError("ContactAware body ID is outside the source scene")
    guidance["hand_query_body_ids"] = mapping[body_ids]
    frames = len(poses)
    hand = np.repeat(qref[None], frames, axis=0).astype(np.float32)
    sequence = destination / "sequence"
    contacts = sequence / "retarget" / HAND / "contactaware"
    scene = sequence / "scene" / HAND
    metadata = sequence / "mano_raw"
    for folder in (contacts, scene, metadata):
        folder.mkdir(parents=True, exist_ok=True)
    xml, visual = scene / "primitive_sdf.xml", scene / "primitive_mesh.xml"
    _copy(native_scene_xml, xml); _copy(viewer_xml, visual)
    np.save(contacts / f"{HAND}_qpos.npy", hand)
    np.save(contacts / "object_pose_7.npy", poses.astype(np.float32))
    np.savez_compressed(contacts / "contact_guidance.npz", **guidance)
    camera = np.repeat(_diagnostic_camera()[None], frames, axis=0)
    np.save(contacts / "camera_pose_7.npy", camera)
    np.save(metadata / "camera_pose_7.npy", camera)
    (metadata / "meta.json").write_text(json.dumps(dict(object="primitive", camera=dict(fovy=45.0)), indent=2))
    card["hand_reference_qpos"] = qref
    if "hand_initial_qpos" in card:
        card["hand_initial_qpos"] = card["hand_initial_qpos"][joint_map]
    np.savez_compressed(destination / "card_input.npz", **card)
    _copy(source / "contactaware_reference.json", destination / "contactaware_reference.json")
    artifacts = ReferenceArtifacts(source, destination, sequence, contacts, xml, visual,
        destination / "forceaware_model.yaml", destination / "reference_manifest.json")
    manifest = dict(source_experiment=str(source), source_ca_commit=report["source_commit"],
        hand=HAND, qpos_dimension=26, stationary_object=True,
        hand_topology=dict(name=HAND, palm_body="right_palm_link", tip_sites=TIP_SITES.copy(),
            registration="external HandSpec registration; upstream source files unchanged"),
        contact_count=int(len(card["reference_contact_mask"])), active_contact_count=int(card["reference_contact_mask"].sum()),
        body_mapping=[dict(source_id=i, body=name, native_id=int(mapping[i])) for i, name in enumerate(old_names)],
        native_joint_order=new_joints, source_hand_joint_indices=joint_map.tolist(),
        material_frame_maximum_position_error_m=max(position_errors,default=0.),
        material_frame_maximum_quaternion_error=max(rotation_errors,default=0.),
        material_frame_tolerance="1e-7 m position / 1e-6 sign-invariant quaternion norm; permits MuJoCo XML serialization rounding",
        camera="fixed diagnostic camera; visualization only", original_fa_first_window_initial_correction=True,
        source_hashes={name: __import__("hashlib").sha256((source / name).read_bytes()).hexdigest()
            for name in ("card_input.npz", "contactaware_reference.json", "contactaware/contact_guidance.npz")})
    artifacts.manifest_path.write_text(json.dumps(manifest, indent=2))
    return artifacts


def make_config(experiment, native_scene_xml=None, *, duration_seconds=.4,
                optimizer_overrides=None, simulator_overrides=None, contact_topk=1):
    """Preserve the official FA profile; override only paths, hold and explicit budgets."""
    root = activate_upstream()
    from forceaware import config as fa

    values = copy.deepcopy(fa.load_config())
    values["sequence"].update(hand=HAND, scene="sdf", start=0, hold_frame=0,
                              hold_seconds=float(duration_seconds))
    for section, changes in (("optimizer", optimizer_overrides), ("simulator", simulator_overrides)):
        changes = {} if changes is None else changes
        if not isinstance(changes, dict) or set(changes) - set(values[section]):
            raise ValueError(f"Invalid ForceAware {section} overrides")
        values[section].update(changes)
    values["contact_physics"]["contact_topk"] = contact_topk
    if isinstance(experiment, ReferenceArtifacts):
        sequence, contacts, xml, config_path = experiment.sequence_dir, experiment.contactaware_dir, experiment.xml, experiment.config_path
    else:
        experiment = Path(experiment).expanduser().resolve()
        sequence, contacts = experiment / "sequence", experiment / "contactaware"
        xml = sequence / "scene" / HAND / "primitive_sdf.xml" if native_scene_xml is None else Path(native_scene_xml).resolve()
        config_path = None
    cfg = fa.resolve_retarget_config(values, sequence, contactaware_dir=contacts)
    # The direct diagnostic API can use an external native scene without
    # copying the accepted CA inputs. Original workflow uses prepared artifacts.
    if cfg.sequence.xml != xml:
        from dataclasses import replace
        cfg = replace(cfg, sequence=replace(cfg.sequence, xml=xml))
    if not xml.is_file():
        raise ValueError(f"Native ForceAware scene is missing: {xml}")
    register_wuji2_hand()
    if config_path is not None:
        config_path.write_text(yaml.safe_dump(values, sort_keys=False))
    return fa.validate_retarget_config(cfg)


def load_reference(experiment, cpu_model, device_model, cfg):
    """Use the original FA target loader, layout mapping and initial-state code."""
    activate_upstream()
    from forceaware.targets import build_index, load_sequence, initial_state
    from forceaware.workflow import _collision_output_layout
    from forceaware.wrist_reference import continuous_wrist_reference
    from dataclasses import replace

    _read_ca(experiment.root if isinstance(experiment, ReferenceArtifacts) else experiment)
    output_bodies, enabled_outputs = _collision_output_layout(device_model)
    sequence = load_sequence(cfg, output_body_ids=output_bodies, enabled_output_indices=enabled_outputs)
    index = build_index(cpu_model)
    if index.action_dim != 26 or sequence.hand.shape[1] != 26:
        raise ValueError("ForceAware Wuji reference/control dimension must remain 26")
    sequence = replace(sequence, hand=continuous_wrist_reference(cpu_model, index.hand_qpos, sequence.hand))
    qpos, qvel = initial_state(cpu_model, index, sequence, start_frame=float(cfg.sequence.start),
        ref_dt=cfg.simulator.time_grid.ref_dt, hold=True)
    metadata = dict(source="DexForge ContactAware", source_commit=COMMIT,
        static_hand_reference=True, static_object_reference=True,
        action_dimension=26, active_contact_count=int(np.count_nonzero(sequence.contact.mask[0])),
        original_first_window_initial_correction=True)
    return ForceAwareReference(cfg, sequence, index, qpos, qvel, metadata)


def create_optimizer(cpu_model, device_model, index, *, cfg, guidance_count):
    """Return exactly the upstream ServoPriorOptimizer and MpcObjective."""
    register_wuji2_hand()
    from forceaware.servo_prior import ServoPriorOptimizer
    return ServoPriorOptimizer(cpu_model, device_model, index, guidance_count=guidance_count, cfg=cfg)
