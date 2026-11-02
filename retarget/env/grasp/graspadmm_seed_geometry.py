"""Taxonomy seed geometry in the canonical, fixed-object frame."""

import copy
import hashlib

import numpy as np
from scipy.spatial.transform import Rotation


def _input_digest(data):
    digest = hashlib.sha256()
    for name in ("grasp_qpos", "hand_cpn_w", "obj_cpn_w"):
        value = np.asarray(data[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def adapt_primitive_seed(raw, scene, primitive, template, policy="taxonomy", *,
                         reference_seed=None, reference_info=None):
    """Reuse the native 1.5x initial pose and rotate the object +90deg in palm.

    The rotation is about the object center and palm +Y. The fixed-object
    representation uses the inverse hand-root rotation, leaving joint angles
    and the object center in the palm unchanged. This applies to every size.
    """
    if policy not in ("taxonomy", "upstream"):
        raise ValueError("primitive_seed_policy must be taxonomy or upstream")
    adapted = copy.deepcopy(raw)
    info = dict(version=2, policy=policy, applied=False, raw_input_sha256=_input_digest(raw))
    if policy == "upstream" or str(template).partition("_")[0] != "9":
        return adapted, dict(info, data_unchanged=True, reason="explicit_upstream_bypass" if policy == "upstream" else "other_grasp_type")
    if primitive is None or primitive.get("kind") != "cylinder":
        return adapted, dict(info, data_unchanged=True, reason="no_explicit_cylinder_primitive")
    if reference_seed is None:
        raise ValueError("Type 9 taxonomy initialization requires its native 1.5x reference seed pose")
    import trimesh

    names = list(reference_seed["hand_cbody"])
    if sorted(names) != ["right_finger1_link4", "right_finger2_link4"] or names != list(raw["hand_cbody"]):
        raise ValueError("Type 9 requires matching annotated index and thumb distal slots")
    q = np.asarray(reference_seed["grasp_qpos"])
    hand = np.asarray(reference_seed["hand_cpn_w"])
    if q.shape != (1, 27) or hand.shape != (2, 6) or not np.isfinite(q).all() or not np.isfinite(hand).all():
        raise ValueError("Malformed type 9 reference seed pose")
    root = Rotation.from_quat(q[0, 3:7], scalar_first=True)
    # The canonical WUJI2 palm is the free hand root, with palmar normal +Y.
    palm_rotation = Rotation.from_rotvec([0., np.pi / 2, 0.])
    moved_root = (palm_rotation * root.inv()).inv()
    delta = moved_root * root.inv()
    new_q = q.copy()
    new_q[0, :3] = delta.apply(q[0, :3])
    new_q[0, 3:7] = moved_root.as_quat(scalar_first=True)
    points, normals = delta.apply(hand[:, :3]), delta.apply(hand[:, 3:])
    mesh = trimesh.load_mesh(scene.mesh_path, process=False)
    lateral_faces = np.flatnonzero(np.abs(mesh.face_normals[:, 2]) < 1e-6)
    if not len(lateral_faces):
        raise ValueError("Type 9 cylinder has no lateral triangles")
    lateral = mesh.submesh([lateral_faces], append=True, repair=False)
    center = mesh.bounds.mean(axis=0)
    height = np.ptp(mesh.vertices[:, 2])
    radius = float(primitive["radius_m"])
    if not np.isfinite([height, radius]).all() or min(height, radius) <= 0:
        raise ValueError("Type 9 cylinder dimensions must be finite and positive")
    targets = points.copy()
    targets[:, 2] = np.clip(targets[:, 2], center[2] - height / 4, center[2] + height / 4)
    projected, distance, triangles = trimesh.proximity.closest_point(lateral, targets)
    adapted.update(grasp_qpos=new_q, hand_cpn_w=np.c_[points, normals].astype(hand.dtype),
                   obj_cpn_w=np.c_[projected, lateral.face_normals[triangles]].astype(np.asarray(raw["obj_cpn_w"]).dtype),
                   reference_orientation_wxyz=moved_root.as_quat(scalar_first=True).tolist())
    transform = np.eye(4)
    transform[:3, :3] = palm_rotation.as_matrix()
    object_center_palm = -root.inv().apply(q[0, :3])
    transform[:3, 3] = object_center_palm - palm_rotation.apply(object_center_palm)
    info.update(applied=True, data_unchanged=False, adapter="type9_shared_1p5x_seed_palm_rotation_v2",
                reference_input_sha256=_input_digest(reference_seed), reference=reference_info,
                reference_size_scale=1.5, reference_grasp_qpos=q[0].tolist(),
                rotation_axis_palm=[0., 1., 0.], rotation_deg=90.,
                object_rotation_about_center_palm=transform.tolist(),
                inverse_hand_world_rotation=delta.as_matrix().tolist(),
                reference_orientation_wxyz=adapted["reference_orientation_wxyz"],
                finger_angles_bitwise_unchanged=bool(np.array_equal(new_q[:, 7:], q[:, 7:])),
                object_center_palm=object_center_palm.tolist(),
                target_mesh_projection_distance_m=distance.tolist(),
                object_target_cap_clearance_m=(height / 2 - np.abs(projected[:, 2] - center[2])).tolist(),
                adapted_input_sha256=_input_digest(adapted))
    return adapted, info
