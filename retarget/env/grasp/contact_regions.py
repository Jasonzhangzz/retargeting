"""Shared, body-local contact regions for the local GraspADMM/MPPI bridge.

Query samples lie on the official meshes. Center preference is finite over the
whole exposed distal surface; the central band is only a diagnostic/reference.
"""

import xml.etree.ElementTree as ET
from functools import lru_cache

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .hand import hand_tree, palmar_axis, query_path, rigid_mesh_arrays


@lru_cache(maxsize=1)
def region_model():
    return mujoco.MjModel.from_xml_string(ET.tostring(hand_tree("wuji2"), encoding="unicode"))


def template_regions(template, body_names):
    if str(template).partition("_")[0] not in ("6", "7", "8"):
        return None
    return dict(version=2, kind="distal_palmar_center", mode="soft", u_min=.3, u_max=.7,
                normal_min=.5, center_weight=.25, palmar_weight=.1,
                longitudinal_scale=.2, projection_scale_m=.002,
                bodies=list(dict.fromkeys(body_names)))


def region_coordinates(name, points, normals):
    model = region_model()
    if not name.startswith("right_finger") or not name.endswith("_link4"):
        raise ValueError(f"Distal contact region requires a distal body: {name}")
    finger = int(name[len("right_finger")])
    label = ("thumb", "index", "middle", "ring", "pinky")[finger - 1]
    endpoint = model.site(f"right_{label}_tip").pos
    length = np.linalg.norm(endpoint)
    longitudinal = endpoint / length
    palmar = palmar_axis(model, model.body(name).id, endpoint)
    palmar -= longitudinal * np.dot(palmar, longitudinal)
    palmar /= np.linalg.norm(palmar)
    side = np.cross(palmar, longitudinal)
    points, normals = np.asarray(points), np.asarray(normals)
    normals = normals / np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
    return points @ longitudinal / length, normals @ palmar, points @ side


def region_mask(name, points, normals, region):
    if region.get("version") not in (1, 2) or region.get("kind") != "distal_palmar_center":
        raise ValueError("Unsupported contact region")
    u, facing, _ = region_coordinates(name, points, normals)
    return (u >= region["u_min"]) & (u <= region["u_max"]) & (facing > region["normal_min"])


def region_preference(name, points, normals, region):
    """Dimensionless soft cost: center=0, every other surface point is allowed."""
    u, facing, _ = region_coordinates(name, points, normals)
    return (region["center_weight"] * ((u - .5) / region["longitudinal_scale"]) ** 2
            + region["palmar_weight"] * np.maximum(region["normal_min"] - facing, 0.) ** 2)


def region_pool(name, region):
    return _region_pool(name, region["version"], region["kind"], region["u_min"], region["u_max"], region["normal_min"])


@lru_cache(maxsize=32)
def _region_pool(name, version, kind, u_min, u_max, normal_min):
    region = dict(version=version, kind=kind, u_min=u_min, u_max=u_max, normal_min=normal_min)
    points, normals, indices = surface_pool(name)
    mask = region_mask(name, points, normals, region)
    if not mask.any():
        raise ValueError(f"Empty exposed distal palmar center region: {name}")
    return points[mask], normals[mask], indices[mask]


@lru_cache(maxsize=16)
def surface_pool(name):
    """All exposed samples, including tips, edges and the dorsal surface."""
    import trimesh
    with np.load(query_path(region_model(), name), allow_pickle=False) as raw:
        points, normals = raw["local_pos"].copy(), raw["local_normal"].copy()
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    indices = np.arange(len(points))
    vertices, faces = rigid_mesh_arrays(region_model(), region_model().body(name).id)
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    # Outward-facing internal pad/mount faces can have a palmar normal too.
    # Reject them against all meshes of the rigid owner, including fixed pads.
    origins = points[indices] + 1.e-5 * normals[indices]
    hits, rays, _ = mesh.ray.intersects_location(origins, normals[indices], multiple_hits=False)
    visible = np.ones(len(indices), bool)
    if len(rays):
        visible[rays[np.linalg.norm(hits - origins[rays], axis=1) < .25]] = False
    indices = indices[visible]
    if not len(indices):
        raise ValueError(f"Empty exposed distal surface: {name}")
    result = points[indices], normals[indices], indices
    for array in result:
        array.setflags(write=False)
    return result


def adapt_contact_regions(model, template, bodies, local, normals, mapping):
    names = [model.body(int(body)).name for body in bodies]
    region = template_regions(template, names)
    if region is None:
        return local, normals, mapping, None
    local, normals = local.copy(), normals.copy()
    mapping = [dict(record) for record in mapping]
    for i, name in enumerate(names):
        points, facing, indices = region_pool(name, region)
        u, _, lateral = region_coordinates(name, points, facing)
        finger = int(name[len("right_finger")])
        label = ("thumb", "index", "middle", "ring", "pinky")[finger - 1]
        length = np.linalg.norm(model.site(f"right_{label}_tip").pos)
        selected = int(np.argmin(((u - .5) * length) ** 2 + lateral ** 2))
        local[i], normals[i] = points[selected], facing[selected]
        mapping[i].update(original_wuji_local=mapping[i]["wuji_local"],
                          original_query_index=mapping[i]["query_index"],
                          query_index=int(indices[selected]), wuji_local=local[i].tolist(),
                          wuji_local_normal=normals[i].tolist(), contact_region=region["kind"],
                          original_mapping_distance_mm=mapping[i]["mapping_distance_mm"],
                          mapping_distance_mm=float(np.sqrt(((u[selected] - .5) * length) ** 2 + lateral[selected] ** 2) * 1000),
                          mapping_distance_reference="longitudinal_and_lateral_distal_center",
                          normalized_longitudinal=float(u[selected]))
    return local, normals, mapping, region


def contact_region_diagnostics(qpos, names, points_normals, region):
    """Measure recorded world anchors/real contacts in each rigid-owner frame."""
    if region is None:
        return None
    model = region_model()
    data = mujoco.MjData(model)
    qpos = np.asarray(qpos)
    if qpos.shape == (27,):
        qpos = np.r_[qpos[:3], Rotation.from_quat(qpos[3:7], scalar_first=True).as_euler("XYZ"), qpos[7:]]
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    rows = []
    for name, value in zip(names, points_normals):
        if name not in region["bodies"]:
            continue
        body = model.body(name).id
        rotation = data.xmat[body].reshape(3, 3)
        point = (np.asarray(value[:3]) - data.xpos[body]) @ rotation
        normal = np.asarray(value[3:]) @ rotation
        u, facing, _ = region_coordinates(name, point, normal)
        rows.append(dict(body=name, normalized_longitudinal=float(u), palmar_normal_dot=float(facing),
                         in_region=bool(region_mask(name, point, normal, region)),
                         **({"center_preference_cost":float(region_preference(name, point, normal, region))}
                            if region.get("mode") == "soft" else {})))
    covered = {r["body"] for r in rows if r["in_region"]}
    return dict(region=region, contacts=rows,
                mode=region.get("mode", "hard"), diagnostic_only=True,
                required_bodies_in_region={name: name in covered for name in region["bodies"]},
                semantic_success=all(name in covered for name in region["bodies"]))


def orientation_error_deg(qpos, reference_quat):
    return float(np.degrees((Rotation.from_quat(reference_quat, scalar_first=True).inv()
                            * Rotation.from_quat(np.asarray(qpos)[3:7], scalar_first=True)).magnitude()))


def lock_root_orientation(env, reference_quat):
    """Install an optimization-only equality constraint, including every substep."""
    quat = np.asarray(reference_quat, float).copy()
    if quat.shape != (4,) or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-12:
        raise ValueError("Need a finite reference quaternion")
    quat /= np.linalg.norm(quat)
    free = np.flatnonzero(env._model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise ValueError("Orientation lock requires exactly one free hand root")
    joint = int(free[0])
    qa, va = int(env._model.jnt_qposadr[joint]), int(env._model.jnt_dofadr[joint])
    original_step = env.step_sim

    def enforce():
        env._data.qpos[qa + 3:qa + 7] = quat
        env._data.qvel[va + 3:va + 6] = 0.
        mujoco.mj_forward(env._model, env._data)

    def step(substep):
        for _ in range(substep):
            enforce()
            original_step(1)
            enforce()

    env.step_sim = step
    return quat
