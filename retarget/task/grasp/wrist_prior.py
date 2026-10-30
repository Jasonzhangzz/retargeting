"""Frozen size-one object poses in the actual FK wrist frame."""

from pathlib import Path
import hashlib
import json
import math

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp

DEFAULT_WEIGHT = 1.0e5
DEFAULT_TRANSLATION_MM = 5.0
DEFAULT_ROTATION_DEG = 15.0


@wp.struct
class WristPosePrior:
    enabled: int
    position: wp.vec3
    rotation: wp.mat33
    translation_weight: float
    rotation_weight: float


@wp.func
def wrist_pose_value(
    prior: WristPosePrior,
    wrist_position: wp.vec3,
    wrist_rotation: wp.mat33,
    object_position: wp.vec3,
    object_rotation: wp.mat33,
):
    value = float(0.0)
    if prior.enabled != 0:
        relative_position = wp.transpose(wrist_rotation) @ (
            object_position - wrist_position
        )
        relative_rotation = wp.transpose(wrist_rotation) @ object_rotation
        value = prior.translation_weight * wp.length_sq(
            relative_position - prior.position
        )
        for i in range(3):
            for j in range(3):
                value += prior.rotation_weight * wp.pow(
                    relative_rotation[i, j] - prior.rotation[i, j], 2.0
                )
    return value


def reference_key(hand_model, template, primitive):
    if not primitive:
        return None
    # Distinguish primitive families and changes to their nominal dimensions.
    r = primitive["radius_m"] / primitive["radial_scale"]
    h = (
        2 * r
        if primitive["kind"] == "sphere"
        else primitive["height_m"] / primitive["axial_scale"]
    )
    return f"{hand_model}:{template}:{primitive['kind']}:{r:.9g}:{h:.9g}"


def is_size_one(primitive):
    return bool(
        primitive
        and math.isclose(primitive["radial_scale"], 1.0, abs_tol=1.0e-9)
        and math.isclose(primitive["axial_scale"], 1.0, abs_tol=1.0e-9)
    )


def initialize_references(args, tasks, output):
    if not args.wrist_pose_prior:
        return
    source = Path(args.wrist_pose_reference)
    data = json.loads(source.read_text())
    args._wrist_pose_references = data["references"]
    save_references(args, output)


def save_references(args, output):
    if hasattr(args, "_wrist_pose_references"):
        path = Path(output) / "wrist_pose_references.json"
        temporary = path.with_suffix(".pending.json")
        temporary.write_text(
            json.dumps(
                dict(
                    frame="right_palm_link (actual FK wrist)",
                    reference_scale=1.0,
                    references=args._wrist_pose_references,
                ),
                indent=2,
            )
        )
        temporary.replace(path)


def configure_reference(hand, template, primitive, args):
    bank = getattr(args, "_wrist_pose_references", {})
    key = reference_key(hand.hand_model, template, primitive)
    if key not in bank:
        if primitive and getattr(args, "wrist_pose_prior", False):
            hand.wrist_reference_missing = True
        return
    # The true wrist has a fixed offset from the six root coordinates. Do not
    # replace it with the inverse root pose or a viewer's display transform.
    q = hand.ik_reference_qpos.copy()
    q[:6] = 0.0
    hand.set(q)
    palm = hand.model.body("right_palm_link").id
    hand.wrist_root_position = hand.data.xpos[palm].copy()
    hand.wrist_root_rotation = hand.data.xmat[palm].reshape(3, 3).copy()
    hand.wrist_pose_reference = dict(
        bank[key],
        weight=getattr(args, "wrist_pose_weight", DEFAULT_WEIGHT),
        translation_scale_mm=getattr(
            args, "wrist_pose_translation_mm", DEFAULT_TRANSLATION_MM
        ),
        rotation_scale_deg=getattr(
            args, "wrist_pose_rotation_deg", DEFAULT_ROTATION_DEG
        ),
    )


def relative_poses(hand, configurations):
    q = np.asarray(configurations).reshape(-1, 26)
    root = Rotation.from_euler("XYZ", q[:, 3:6]).as_matrix()
    wrist = root @ hand.wrist_root_rotation
    position = q[:, :3] + np.einsum("nij,j->ni", root, hand.wrist_root_position)
    inverse = wrist.transpose(0, 2, 1)
    return -np.einsum("nij,nj->ni", inverse, position), inverse


def prior_diagnostics(hand, configurations):
    prior = hand.wrist_pose_reference
    target = np.asarray(prior["object_pose_wrist"])
    translation, rotation = relative_poses(hand, configurations)
    delta = translation - target[:3]
    reference_rotation = Rotation.from_quat(target[3:], scalar_first=True).as_matrix()
    dr = np.sum((rotation - reference_rotation) ** 2, axis=(1, 2))
    cost = prior["weight"] * (
        np.sum(delta * delta, axis=1) / (prior["translation_scale_mm"] / 1000) ** 2
        + dr / (4 * (1 - np.cos(np.deg2rad(prior["rotation_scale_deg"]))))
    )
    return dict(
        translation_error_mm=np.linalg.norm(delta, axis=1) * 1000,
        rotation_error_deg=np.degrees(np.arccos(np.clip(1 - dr / 4, -1.0, 1.0))),
        cost=cost,
    )


def reference_pose_seeds(hand, template_q):
    """Exact wrist reference plus small local offsets for size-dependent clearance."""
    pose = np.asarray(hand.wrist_pose_reference["object_pose_wrist"])
    wrist_rotation = Rotation.from_quat(pose[3:], scalar_first=True).as_matrix().T
    root_rotation = wrist_rotation @ hand.wrist_root_rotation.T
    wrist_position = -wrist_rotation @ pose[:3]
    q = template_q.copy()
    q[:3] = wrist_position - root_rotation @ hand.wrist_root_position
    q[3:6] = Rotation.from_matrix(root_rotation).as_euler("XYZ")
    poses = [q.copy()]
    for j in range(3):
        for sign in (-1.0, 1.0):
            offset = np.eye(3)[j] * sign * 0.005
            trial = q.copy()
            trial[:3] += wrist_rotation @ offset
            poses.append(trial)
    return poses


def warp_prior(reference=None):
    prior = WristPosePrior()
    if reference is not None:
        pose = np.asarray(reference["object_pose_wrist"])
        prior.enabled = int(reference["weight"] > 0)
        prior.position = wp.vec3(pose[:3])
        prior.rotation = wp.mat33(
            Rotation.from_quat(pose[3:], scalar_first=True).as_matrix()
        )
        prior.translation_weight = (
            reference["weight"] / (reference["translation_scale_mm"] / 1000) ** 2
        )
        prior.rotation_weight = reference["weight"] / (
            4 * (1 - np.cos(np.deg2rad(reference["rotation_scale_deg"])))
        )
    return prior
