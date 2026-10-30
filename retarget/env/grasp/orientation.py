"""Primitive orientation references relative to the annotated palm surface."""

import numpy as np
from scipy.spatial.transform import Rotation


def configure_orientation(hand, template, primitive):
    """Power disks face the palm; other templates keep their original freedom."""
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
