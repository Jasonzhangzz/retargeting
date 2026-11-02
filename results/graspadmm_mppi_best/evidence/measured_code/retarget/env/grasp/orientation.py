"""Primitive orientation references relative to the annotated palm surface."""

import numpy as np
from scipy.spatial.transform import Rotation


def configure_orientation(hand, template, primitive):
    """Retain Power Disk guidance; new axis conventions supply seeds only."""
    number = template.partition("_")[0]
    if primitive and number in ("9", "16", "22", "30"):
        hand.primitive_axis_seed = dict(type_id=int(number), kind=primitive["kind"],
            thickness_m=float(primitive["height_m"]),
            convention="cylinder_side" if number == "9" else
                "palm_normal" if number == "30" else "thumb_contact_normal",
            role="initial_pose_only_no_additional_cost_or_success_threshold")
    if (
        template.partition("_")[0] != "10"
        or not primitive
        or primitive["kind"] != "disk"
    ):
        return
    palm = hand.model.body("right_palm_link").id
    selected = hand.bodies == palm
    if not selected.any():
        raise ValueError("Power disk orientation requires an annotated palm normal")
    normal = np.asarray(hand.contact_reference_normals)[selected].mean(axis=0)
    normal /= np.linalg.norm(normal)
    hand.set(np.r_[np.zeros(6), hand.ik_reference_qpos[6:]])
    hand.object_orientation = dict(
        axis_palm=normal.tolist(),
        axis_root=(hand.data.xmat[palm].reshape(3, 3) @ normal).tolist(),
        maximum_angle_deg=25.0,
        thickness_m=primitive["height_m"],
    )


def orientation_angles(hand, configurations):
    """Unsigned disk-axis angle; the two equivalent disk faces share a reference."""
    q = np.asarray(configurations)
    rotation = Rotation.from_euler("XYZ", q[..., 3:6])
    axis = rotation.apply(hand.object_orientation["axis_root"])
    cosine = np.clip(np.abs(axis[..., 2]), 0.0, 1.0)
    return np.degrees(np.arccos(cosine))


def aligned_disk_seeds(hand, template_q):
    """Add face-aligned poses, evaluated unmodified by the upstream matcher.

    These are additional candidates, never edits to a matcher output. Clearance
    and shape checks still determine whether any of them is a valid input.
    """
    reference = hand.object_orientation
    axis = np.asarray(reference["axis_root"])
    hand.set(np.r_[np.zeros(6), template_q[6:]])
    points = hand.points(hand.bodies, hand.local)
    palm = points[hand.bodies == hand.model.body("right_palm_link").id].mean(axis=0)
    centers = [points.mean(axis=0)]
    centers += [
        palm + axis * (reference["thickness_m"] / 2 + gap)
        for gap in (0.002, 0.01, 0.025, 0.04, 0.06)
    ]
    seeds = []
    for sign in (-1.0, 1.0):
        rotation, _ = Rotation.align_vectors([[0.0, 0.0, sign]], [axis])
        for yaw in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
            root = Rotation.from_rotvec([0.0, 0.0, yaw]) * rotation
            for center in centers:
                seeds.append(
                    np.r_[-root.apply(center), root.as_euler("XYZ"), template_q[6:]]
                )
    return np.asarray(seeds)


def primitive_pose_seeds(hand, template_q):
    """Place thickness along the thumb/palm normal, or a cylinder along the hand.

    The object stays in its canonical Z-axis frame. Rotating the hand relative
    to that frame provides the requested interpretation without adding a cost
    or an acceptance constraint to any solver.
    """
    if not hasattr(hand, "primitive_axis_seed"):
        return np.empty((0, 26))
    reference = hand.primitive_axis_seed
    hand.set(np.r_[np.zeros(6), np.asarray(template_q)[6:]])
    points = hand.points(hand.bodies, hand.local)
    rotations = hand.data.xmat[hand.bodies].reshape(-1, 3, 3)
    normals = np.einsum("nij,nj->ni", rotations, hand.normals)
    names = [hand.model.body(int(body)).name for body in hand.bodies]
    thumb = np.asarray([name.startswith("right_finger1_") for name in names])
    palm_body = hand.model.body("right_palm_link").id
    palm = hand.bodies == palm_body
    if reference["convention"] == "palm_normal":
        from .hand import palmar_axis
        axis = hand.data.xmat[palm_body].reshape(3, 3) @ palmar_axis(hand.model, palm_body, None)
    else:
        if not thumb.any():
            return np.empty((0, 26))
        axis = normals[thumb].mean(axis=0)
        axis /= max(np.linalg.norm(axis), 1.e-12)
        if reference["convention"] == "cylinder_side":
            distal = hand.model.body("right_finger3_link2").id
            longitudinal = hand.data.xpos[distal] - hand.data.xpos[palm_body]
            axis = longitudinal - axis * np.dot(longitudinal, axis)
    norm = np.linalg.norm(axis)
    if norm < 1.e-10:
        raise ValueError("Primitive orientation seed has a degenerate axis")
    axis = axis / norm
    center = points.mean(axis=0)
    if reference["convention"] == "palm_normal" and palm.any():
        center = points[palm].mean(axis=0) + axis * reference["thickness_m"] / 2
    seeds = []
    for sign in (-1., 1.):
        rotation, _ = Rotation.align_vectors([[0., 0., sign]], [axis])
        for yaw in (0., np.pi / 2, np.pi, 3 * np.pi / 2):
            root = Rotation.from_rotvec([0., 0., yaw]) * rotation
            seeds.append(np.r_[-root.apply(center), root.as_euler("XYZ"), np.asarray(template_q)[6:]])
    return np.asarray(seeds)


def primitive_contact_diagnostics(type_id, primitive, ho_c=None, *, initial_hand_contacts=None):
    """Observe primitive semantics without changing a cost or success verdict.

    ``ho_c`` is the native real hand/object contact dictionary with ``bn1``,
    ``pos`` and ``normal`` in the canonical object frame. The object center is
    the origin and its cylinder/thickness axis is Z. ``initial_hand_contacts``
    may be the native initialization record (``hand_cbody``, ``hand_cpn_w``),
    or a dictionary with the same three contact-array keys. Those initial
    anchors are reported separately and never presented as physical contacts.
    """
    type_id = int(type_id)
    report = dict(type_id=type_id, applicable=type_id in (9, 11, 16, 22),
        diagnostic_only=True, affects_task_success=False,
        frame="canonical object frame; object center at origin; thickness/cylinder axis is Z",
        initialization_policy="original native sampler and filters; no added semantic constraint")
    if not report["applicable"]:
        return report
    if primitive is None:
        return dict(report, available=False, reason="primitive_metadata_unavailable")

    def arrays(record, *, anchors=False):
        if record is None:
            return None, "not_recorded"
        try:
            if anchors and "hand_cpn_w" in record:
                names = list(map(str, record["hand_cbody"]))
                points_normals = np.asarray(record["hand_cpn_w"], float).reshape(-1, 6)
                points, normals = points_normals[:, :3], points_normals[:, 3:]
            else:
                names = list(map(str, record["bn1"]))
                points = np.asarray(record["pos"], float).reshape(-1, 3)
                normals = np.asarray(record["normal"], float).reshape(-1, 3)
            if len(names) != len(points) or len(names) != len(normals):
                return None, "contact_array_lengths_disagree"
            if not np.isfinite(points).all() or not np.isfinite(normals).all():
                return None, "nonfinite_contact_arrays"
            return (names, points, normals), None
        except (KeyError, TypeError, ValueError):
            return None, "malformed_contact_arrays"

    def observe(record, *, anchors=False):
        values, reason = arrays(record, anchors=anchors)
        result = dict(available=values is not None,
            source="initial_hand_anchors_not_physical_contacts" if anchors else "native_hand_object_contacts")
        if values is None:
            return dict(result, reason=reason)
        names, positions, normals = values
        result.update(contact_count=len(names), contact_bodies=names)
        if type_id == 11:
            palm = [i for i, name in enumerate(names) if name == "right_palm_link"]
            result.update(palm_contact_present=bool(palm), palm_contact_count=len(palm),
                palm_contact_indices=palm)
        elif type_id in (16, 22):
            thumb = [i for i, name in enumerate(names)
                     if name.startswith("right_finger1_") or name == "r_thumb_tip_sensor_frame"]
            valid = [i for i in thumb if np.linalg.norm(normals[i]) > 1.e-12]
            if not valid:
                result.update(thumb_axis_angle_available=False,
                    reason="no_thumb_contact" if not thumb else "zero_thumb_contact_normal",
                    thumb_contact_count=len(thumb), thumb_normal_axis_unsigned_angle_deg=None)
            else:
                angles = np.degrees(np.arccos(np.clip(
                    np.abs(normals[valid, 2]) / np.linalg.norm(normals[valid], axis=1), 0., 1.)))
                result.update(thumb_axis_angle_available=True, thumb_contact_count=len(thumb),
                    thumb_contact_indices=valid,
                    thumb_normal_axis_unsigned_angle_deg=angles.tolist(),
                    thumb_normal_axis_unsigned_angle_min_deg=float(angles.min()),
                    thumb_normal_axis_unsigned_angle_max_deg=float(angles.max()))
        else:
            radius, half_height = float(primitive["radius_m"]), float(primitive["height_m"]) / 2.
            contacts = []
            for index, (name, pos, normal) in enumerate(zip(names, positions, normals)):
                radial_error = abs(float(np.linalg.norm(pos[:2])) - radius)
                clearance = half_height - abs(float(pos[2]))
                cap_distance = abs(clearance)
                if abs(radial_error - cap_distance) <= 1.e-9:
                    region = "rim"
                elif cap_distance < radial_error:
                    region = "cap"
                else:
                    region = "side" if clearance >= 0 else "outside_axial_span"
                length = np.linalg.norm(normal)
                angle = (float(np.degrees(np.arccos(np.clip(abs(normal[2])/length, 0., 1.))))
                         if length > 1.e-12 else None)
                contacts.append(dict(index=index, body=name, surface_region=region,
                    is_side_contact=region == "side", cap_distance_m=cap_distance,
                    cap_clearance_m=clearance, radial_surface_error_m=radial_error,
                    normal_axis_unsigned_angle_deg=angle))
            result.update(expected_family="cylinder", family_matches_request=primitive.get("kind") == "cylinder",
                contacts=contacts, side_contact_count=sum(c["is_side_contact"] for c in contacts),
                all_observed_contacts_on_side=bool(contacts) and all(c["is_side_contact"] for c in contacts),
                minimum_cap_distance_m=min((c["cap_distance_m"] for c in contacts), default=None))
        return result

    report["native_contacts"] = observe(ho_c)
    report["initial_hand_anchors"] = observe(initial_hand_contacts, anchors=True)
    return report
