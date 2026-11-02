"""Shared orientation convention and middle-cylinder grasp constraints.

Nominal finger angles are initialization only; contact/IK solves the reference.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .hand import palmar_axis

ANGLE_NUMERIC_TOL_DEG = .001
POSITION_NUMERIC_TOL_M = .00001


def configure_intent(hand, template, primitive, args):
    source = getattr(args, "grasp_intent_reference", None)
    if source:
        bank = json.loads(Path(source).read_text())
        entry = bank["references"].get(f"{template}:{primitive['kind']}")
        legacy_side_seed = False
        if entry is None and template.partition("_")[0] == "9" and primitive["kind"] == "cylinder":
            # The old disk pose supplies finger articulation only. Reorient its
            # root as a side-pinch seed for the corrected long cylinder.
            entry = bank["references"].get(f"{template}:disk")
            legacy_side_seed = entry is not None
        if entry is None:
            raise ValueError(f"Missing nominal grasp intent for {template}")
        q = np.asarray(entry["qpos"], float)
        if q.shape != (26,) or not np.isfinite(q).all():
            raise ValueError("Nominal grasp intent must contain 26 finite coordinates")
        if legacy_side_seed:
            from .orientation import primitive_pose_seeds
            seeds = primitive_pose_seeds(hand, q)
            if len(seeds):
                q = seeds[0]
            hand.pose_intent_source_adaptation = "type9 legacy disk finger articulation with cylinder-side root seed"
        hand.pose_intent_qpos = q.copy()
        hand.pose_intent_radius = float(entry["radius_m"])
        hand.pose_intent_radius_delta = (float(primitive["radius_m"])-hand.pose_intent_radius
                                        if primitive["kind"] in ("cylinder", "sphere") else 0.)
        hand.set(np.zeros(26))
        palm = hand.model.body("right_palm_link").id
        hand.intent_palm_position = hand.data.xpos[palm].copy()
        hand.intent_palm_rotation = hand.data.xmat[palm].reshape(3, 3).copy()
        hand.pose_intent_center = palm_object_center(hand, q)
        hand.pose_intent_center += palmar_axis(hand.model, palm, None)*hand.pose_intent_radius_delta
    number = template.partition("_")[0]
    if (primitive and primitive["kind"] == "cylinder" and number.isdigit()
            and 1 <= int(number) <= 6):
        slots = [i for i, label in enumerate(getattr(hand, "semantic_patch_labels", []))
                 if label.startswith("middle.distal.")]
        if not slots:
            slots = [i for i, body in enumerate(hand.bodies)
                     if hand.model.body(int(body)).name.startswith("right_finger3_")][:1]
        if not slots:
            raise ValueError("Types 1–6 require a middle-finger semantic contact")
        hand.cylinder_middle_slots = np.asarray(slots, int)
        hand.cylinder_middle_half_band = .1*float(primitive["height_m"])


def intent_pose_seeds(hand, template_q, primitive):
    if not hasattr(hand, "pose_intent_qpos"):
        return []
    q = np.asarray(template_q).copy()
    q[:6] = hand.pose_intent_qpos[:6]
    hand.set(q)
    palm = hand.model.body("right_palm_link").id
    normal = hand.data.xmat[palm].reshape(3, 3) @ palmar_axis(hand.model, palm, None)
    q[:3] -= normal*getattr(hand, "pose_intent_radius_delta", primitive["radius_m"]-hand.pose_intent_radius)
    if hasattr(hand, "cylinder_middle_slots"):
        hand.set(q)
        q[2] -= hand.points(hand.bodies, hand.local)[hand.cylinder_middle_slots, 2].mean()
    poses = [q.copy()]
    for offset in (-.01, .01):
        trial = q.copy(); trial[:3] += normal*offset; poses.append(trial)
    return poses


def rotation_residual(q, reference):
    actual = Rotation.from_euler("XYZ", np.asarray(q)[3:6])
    target = Rotation.from_euler("XYZ", np.asarray(reference)[3:6])
    return (actual.as_matrix()-target.as_matrix()).ravel()


def rotation_angle(q, reference):
    actual = Rotation.from_euler("XYZ", np.asarray(q)[..., 3:6])
    target = Rotation.from_euler("XYZ", np.asarray(reference)[3:6])
    return np.degrees((actual*target.inv()).magnitude())


def palm_object_center(hand, q):
    q = np.asarray(q)
    root = Rotation.from_euler("XYZ", q[3:6]).as_matrix()
    position = q[:3]+root@hand.intent_palm_position
    return -(root@hand.intent_palm_rotation).T@position


def intent_diagnostics(hand, q, points=None):
    result = {}
    if hasattr(hand, "pose_intent_qpos"):
        result["nominal_rotation_error_deg"] = float(rotation_angle(q, hand.pose_intent_qpos))
    if hasattr(hand, "pose_intent_center"):
        result["nominal_center_error_mm"] = float(np.linalg.norm(
            palm_object_center(hand, q)-hand.pose_intent_center)*1000)
    if hasattr(hand, "cylinder_middle_slots"):
        if points is None:
            hand.set(q); points = hand.points(hand.bodies, hand.local)
        z = np.asarray(points)[hand.cylinder_middle_slots, 2]
        result.update(middle_contact_axial_mm=(z*1000).tolist(),
                      middle_center_band_mm=hand.cylinder_middle_half_band*1000,
                      middle_in_center_band=bool(np.max(np.abs(z)) <= hand.cylinder_middle_half_band+POSITION_NUMERIC_TOL_M))
    return result
