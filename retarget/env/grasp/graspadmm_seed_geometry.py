"""Explicit taxonomy input geometry; the native optimizer and tests are unchanged.

Type 9 uses a rigid hand-root transform and opposed targets on the cylinder's
actual lateral mesh. This does not open fingers or add an IK/cost/filter. A
narrow original template can therefore start with substantial penetration;
that fact is measured and retained, never used as a new acceptance condition.
"""

import copy
import hashlib

import numpy as np
from scipy.spatial.transform import Rotation


def _unit(vector, label):
    vector = np.asarray(vector, float)
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length < 1e-10:
        raise ValueError(f"Degenerate type 9 {label}")
    return vector / length


def _input_digest(data):
    digest = hashlib.sha256()
    for name in ("grasp_qpos", "hand_cpn_w", "obj_cpn_w"):
        value = np.asarray(data[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _geometry(env, data):
    env.reset_qpos(np.asarray(data["grasp_qpos"])[0])
    ho, hh = env.get_contacts(.002)

    def minimum(values):
        return float(np.min(values)) if len(values) else None

    hand_object = minimum(ho["dist"])
    hand_hand = minimum(hh["dist"])
    return dict(
        hand_object_min_distance_m=hand_object,
        hand_object_max_penetration_mm=max(0., -hand_object) * 1000 if hand_object is not None else 0.,
        hand_hand_min_distance_m=hand_hand,
        hand_hand_max_penetration_mm=max(0., -hand_hand) * 1000 if hand_hand is not None else 0.,
        actual_hand_object_contact_count=len(ho["dist"]),
        contact_bodies=list(ho["bn1"]),
        joint_limits_within_native_bounds=bool(env.check_qpos_limit(0.)),
    )


def adapt_primitive_seed(raw, scene, primitive, template, policy="taxonomy"):
    """Return a new input record and an audit; never mutate the raw input.

    ``upstream`` and every non-type-9 input are exact data no-ops. Type 9 keeps
    all 20 finger angles bitwise, preserves the raw azimuth, places the two
    opposed contact targets around the cylinder midsection, and transforms
    the annotated hand anchors by the same root rigid transform.
    """
    if policy not in ("taxonomy", "upstream"):
        raise ValueError("primitive_seed_policy must be taxonomy or upstream")
    adapted = copy.deepcopy(raw)
    info = dict(version=1, policy=policy, applied=False,
                adapter="type9_opposed_cylinder_side_frame_v1",
                method="identity",
                optimizer_cost_filter_unchanged=True, extra_ik_or_joint_optimizer=False,
                penetration_is_diagnostic_only=True, raw_input_sha256=_input_digest(raw))
    if policy == "upstream":
        return adapted, dict(info, condition="explicit_upstream_bypass", reason="explicit_original_upstream_input", data_unchanged=True)
    if str(template).partition("_")[0] != "9":
        return adapted, dict(info, condition="other_grasp_type", reason="other_grasp_type", data_unchanged=True)
    if primitive is None or primitive.get("kind") != "cylinder":
        return adapted, dict(info, condition="no_explicit_cylinder_primitive", reason="no_explicit_cylinder_primitive", data_unchanged=True)
    import trimesh
    from .graspadmm_scene import WujiOptEnv
    from .graspadmm_source import require_upstream

    names = list(raw["hand_cbody"])
    if (len(names) != 2 or names.count("right_finger2_link4") != 1
            or names.count("right_finger1_link4") != 1):
        raise ValueError("Type 9 side input requires the annotated index and thumb distal slots")
    index, thumb = names.index("right_finger2_link4"), names.index("right_finger1_link4")
    hand = np.asarray(raw["hand_cpn_w"])
    qpos = np.asarray(raw["grasp_qpos"])
    if hand.shape != (2, 6) or qpos.shape != (1, 27) or not np.isfinite(hand).all() or not np.isfinite(qpos).all():
        raise ValueError("Malformed type 9 native input record")
    mesh = trimesh.load_mesh(scene.mesh_path, process=False)
    mesh_center = mesh.bounds.mean(axis=0)
    height = float(np.ptp(mesh.vertices[:, 2]))
    radius = float(primitive["radius_m"])
    if not np.isfinite(radius) or radius <= 0 or height <= 0:
        raise ValueError("Type 9 side input requires positive cylinder dimensions")
    lateral_faces = np.flatnonzero(np.abs(mesh.face_normals[:, 2]) < 1e-6)
    if len(lateral_faces) == 0:
        raise ValueError("The canonical cylinder has no actual lateral triangles")
    lateral = mesh.submesh([lateral_faces], append=True, repair=False)
    original_targets = np.asarray(raw["obj_cpn_w"])
    if original_targets.shape != (2, 6) or not np.isfinite(original_targets).all():
        raise ValueError("Malformed type 9 object contact targets")
    _, original_side_distance, _ = trimesh.proximity.closest_point(lateral, original_targets[:, :3])
    original_cap_clearance = height / 2 - np.abs(original_targets[:, 2] - mesh_center[2])
    # Numerical surface membership only; this chooses an input transform and
    # never becomes an optimization or success constraint.
    already_side = (original_side_distance <= 1e-7) & (original_cap_clearance > 1e-7)
    info["condition"] = dict(
        trigger="any_required_distal_object_target_not_on_cylinder_side",
        all_required_targets_already_side=bool(already_side.all()),
        geometric_tolerance_m=1e-7,
        slots=[dict(body=name, already_side=bool(side), lateral_distance_m=float(distance),
                    cap_clearance_m=float(clearance))
               for name, side, distance, clearance in zip(names, already_side, original_side_distance, original_cap_clearance)],
    )
    if already_side.all():
        return adapted, dict(info, reason="required_distal_object_targets_already_on_lateral_surface",
                             data_unchanged=True, finger_angles_bitwise_unchanged=True)
    points = hand[:, :3].astype(float)
    normals = hand[:, 3:].astype(float)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    closing_axis = _unit(normals[thumb] - normals[index], "closing axis")
    cylinder_axis = _unit(np.cross(normals[index], normals[thumb]), "contact plane axis")
    tangent = _unit(np.cross(cylinder_axis, closing_axis), "contact tangent")
    original_frame = np.column_stack((closing_axis, tangent, cylinder_axis))
    center = points.mean(axis=0)
    azimuth = center.copy()
    azimuth[2] = 0.
    if np.linalg.norm(azimuth) < 1e-10:
        azimuth = np.array([1., 0., 0.])
    azimuth = _unit(azimuth, "azimuth")
    world_z = np.array([0., 0., 1.])
    target_frame = np.column_stack((azimuth, np.cross(world_z, azimuth), world_z))
    rotation = target_frame @ original_frame.T

    translation = mesh_center - rotation @ center
    moved_points = points @ rotation.T + translation
    moved_normals = normals @ rotation.T
    targets = moved_points.copy()
    targets[index, :2] = mesh_center[:2] + radius * azimuth[:2]
    targets[thumb, :2] = mesh_center[:2] - radius * azimuth[:2]
    targets[:, 2] = np.clip(targets[:, 2], mesh_center[2] - height / 4, mesh_center[2] + height / 4)
    projected, projection_distance, triangles = trimesh.proximity.closest_point(lateral, targets)
    object_normals = lateral.face_normals[triangles]

    original_rotation = Rotation.from_quat(qpos[0, 3:7], scalar_first=True).as_matrix()
    adapted["grasp_qpos"][0, :3] = rotation @ qpos[0, :3] + translation
    adapted["grasp_qpos"][0, 3:7] = Rotation.from_matrix(rotation @ original_rotation).as_quat(scalar_first=True)
    adapted["hand_cpn_w"] = np.c_[moved_points, moved_normals].astype(hand.dtype)
    adapted["obj_cpn_w"] = np.c_[projected, object_normals].astype(np.asarray(raw["obj_cpn_w"]).dtype)
    if not np.array_equal(adapted["grasp_qpos"][:, 7:], qpos[:, 7:]):
        raise AssertionError("Input geometry must never modify finger angles")

    source = require_upstream()
    env = WujiOptEnv(source.sim.HandCfg(str(scene.hand_xml), freejoint=True), scene.scene_cfg)
    raw_geometry = _geometry(env, raw)
    local_contacts = env.transform_cpn_w2b(names, hand)
    new_geometry = _geometry(env, adapted)
    reconstructed = env.transform_cpn_b2w(names, local_contacts)
    info.update(
        applied=True, data_unchanged=False,
        method="opposed_rigid_contact_frame_with_actual_lateral_projection",
        reason="requested_type9_cylinder_side_initialization", adapted_input_sha256=_input_digest(adapted),
        preserved_seed_variation="azimuth from original native contact midpoint; no outcome-based selection",
        finger_angles_bitwise_unchanged=True,
        root_rotation_delta_deg=float(Rotation.from_matrix(rotation).magnitude() * 180 / np.pi),
        root_translation_delta_m=(adapted["grasp_qpos"][0, :3] - qpos[0, :3]).tolist(),
        world_rigid_rotation=rotation.tolist(), world_rigid_translation=translation.tolist(),
        initial_anchor_separation_m=float(np.linalg.norm(points[index] - points[thumb])),
        object_target_separation_m=float(np.linalg.norm(projected[index] - projected[thumb])),
        actual_lateral_triangle_count=len(lateral_faces), target_mesh_projection_distance_m=projection_distance.tolist(),
        object_target_cap_clearance_m=(height / 2 - np.abs(projected[:, 2] - mesh_center[2])).tolist(),
        object_target_normal_axis_angle_deg=np.degrees(np.arccos(np.clip(np.abs(object_normals[:, 2]), 0., 1.))).tolist(),
        hand_to_object_target_gap_m=np.linalg.norm(moved_points - projected, axis=1).tolist(),
        world_anchor_fk_consistency_error_m=float(np.max(np.linalg.norm(reconstructed[:, :3] - moved_points, axis=1))),
        raw_geometry=raw_geometry, adapted_geometry=new_geometry,
        warning="A rigid root cannot widen the fixed finger opening; measured initial penetration is retained without a new gate.",
    )
    return adapted, info
