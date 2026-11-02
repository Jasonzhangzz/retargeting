"""Intrinsic Shadow/Wuji surface patches for template contact correspondence.

Patch identity is selected before pose fitting. Every returned point is an
existing surface query, and each mask remains in that query pool's order.
"""

import mujoco
import numpy as np
import warp as wp

from retarget.env.grasp.hand import (
    model_hand, palmar_axis, query_path, rigid_mesh_arrays,
)


_FINGERS = {"th": (1, "thumb"), "ff": (2, "index"), "mf": (3, "middle"),
            "rf": (4, "ring"), "lf": (5, "pinky")}
_CACHE = {}


@wp.kernel(enable_backward=False)
def _visible_queries(mesh: wp.uint64, points: wp.array(dtype=wp.vec3),
                     normals: wp.array(dtype=wp.vec3), visible: wp.array(dtype=int)):
    i = wp.tid()
    ray = wp.mesh_query_ray(mesh, points[i] + 1.0e-5 * normals[i], normals[i], 0.25)
    visible[i] = int(not ray.result)


def _unit(value):
    value = np.asarray(value, float)
    length = np.linalg.norm(value)
    if length < 1.0e-10:
        raise ValueError("Degenerate semantic patch frame")
    return value / length


def _frame(longitudinal, palmar, origin=None):
    longitudinal = _unit(longitudinal)
    palmar = _unit(np.asarray(palmar) - longitudinal * np.dot(palmar, longitudinal))
    lateral = _unit(np.cross(palmar, longitudinal))
    return dict(origin=np.zeros(3) if origin is None else np.asarray(origin),
                longitudinal=longitudinal, palmar=palmar, lateral=lateral)


def _matrix(frame):
    return np.column_stack([frame["lateral"], frame["palmar"], frame["longitudinal"]])


def _json_frame(frame):
    return {key: value.tolist() for key, value in frame.items()}


def _source_semantic(source):
    if source == "rh_palm":
        return "palm", None, None, "right_palm_link"
    if source == "rh_lfmetacarpal":
        return "palm_ulnar", None, None, "right_palm_link"
    for prefix, (finger, name) in _FINGERS.items():
        for segment, link in (("proximal", 2), ("middle", 3), ("distal", 4)):
            if source == f"rh_{prefix}{segment}":
                return segment, prefix, name, f"right_finger{finger}_link{link}"
    raise ValueError(f"Unsupported Shadow semantic patch body: {source}")


def _palm_frame(model, data, body, *, shadow):
    inverse = data.xmat[body].reshape(3, 3).T
    if shadow:
        roots = np.array([data.xpos[model.body(f"rh_{prefix}proximal").id]
                          for prefix in _FINGERS])
        palmar = np.array([0., -1., 0.])
    else:
        # Link 1 is the physical MCP/CMC base; link 2 is the long phalanx
        # after the offset abduction joint on Hand 2.
        roots = np.array([data.xpos[model.body(f"right_finger{i}_link1").id]
                          for i in range(1, 6)])
        palmar = palmar_axis(model, body, None)
    local = (roots - data.xpos[body]) @ inverse.T
    forward = local[1:].mean(axis=0)
    forward -= palmar * np.dot(forward, palmar)
    frame = _frame(forward, palmar)
    length = np.linalg.norm(forward)
    width = max(np.ptp(local[1:] @ frame["lateral"]) / 2., .005)
    return frame, length, width


def _source_chart(model, data, body, segment, prefix):
    if segment.startswith("palm"):
        palm = model.body("rh_palm").id
        frame, length, width = _palm_frame(model, data, palm, shadow=True)
        if body != palm:
            inverse = data.xmat[body].reshape(3, 3).T
            transform = inverse @ data.xmat[palm].reshape(3, 3)
            frame = {name: transform @ axis for name, axis in frame.items()}
            frame["origin"] = inverse @ (data.xpos[palm] - data.xpos[body])
        return frame, length, np.array([width, .011])
    rotation = data.xmat[body].reshape(3, 3)
    if segment == "distal":
        endpoint = np.array([0., 0., .027])
    else:
        child = model.body(f"rh_{prefix}{'middle' if segment == 'proximal' else 'distal'}").id
        endpoint = rotation.T @ (data.xpos[child] - data.xpos[body])
    palmar = [-1., 0., 0.] if prefix == "th" and segment == "middle" else [0., -1., 0.]
    frame = _frame(endpoint, palmar)
    radius = .0105 if prefix == "th" and segment == "proximal" else .009
    if segment == "distal":
        radius = .008
    return frame, np.linalg.norm(endpoint), np.array([radius, radius])


def _target_pool(model, body, segment, finger_name):
    path = query_path(model, body)
    key = (str(path.resolve()), path.stat().st_mtime_ns)
    if key in _CACHE:
        return _CACHE[key]
    with np.load(path, allow_pickle=False) as raw:
        points = raw["local_pos"].astype(float)
        normals = raw["local_normal"].astype(float)
        source_body = raw["source_body"].astype(str) if "source_body" in raw else None
        source_mesh = raw["source_mesh"].astype(str) if "source_mesh" in raw else None
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    if segment.startswith("palm"):
        frame, length, width = _palm_frame(model, data, body, shadow=False)
    else:
        name = model.body(body).name
        endpoint = (model.site(f"right_{finger_name}_tip").pos if segment == "distal"
                    else model.body(name[:-1] + str(int(name[-1]) + 1)).pos)
        frame = _frame(endpoint, palmar_axis(model, body, endpoint))
        length = np.linalg.norm(endpoint)
        width = None
    chart = (points - frame["origin"]) @ _matrix(frame)
    base = np.ones(len(points), bool)
    if segment.startswith("palm") and source_body is not None:
        base &= source_body == model.body(body).name
    vertices, faces = rigid_mesh_arrays(model, body)
    mesh = wp.Mesh(points=wp.array(vertices, dtype=wp.vec3, device="cpu"),
                   indices=wp.array(faces.reshape(-1), dtype=int, device="cpu"))
    visible = wp.empty(len(points), dtype=int, device="cpu")
    wp.launch(_visible_queries, len(points), inputs=[mesh.id,
              wp.array(points, dtype=wp.vec3, device="cpu"),
              wp.array(normals, dtype=wp.vec3, device="cpu"), visible], device="cpu")
    base &= visible.numpy().astype(bool)
    # Signed radial scales describe the exposed skin, not mount geometry or
    # the thickness of the opposite dorsal shell.
    radial = chart[base, :2]
    positive = np.maximum(np.quantile(np.maximum(radial, 0.), .95, axis=0), .001)
    negative = np.maximum(np.quantile(np.maximum(-radial, 0.), .95, axis=0), .001)
    if width is not None:
        positive[0] = negative[0] = width
    result = dict(points=points, normals=normals, frame=frame, length=length,
                  positive_radius=positive, negative_radius=negative,
                  chart=chart, base=base, source_mesh=source_mesh)
    _CACHE[key] = result
    return result


def _region(normal, u, segment):
    side, palmar, longitudinal = normal
    if segment == "distal" and longitudinal > .55 and u > .8:
        return "tip", np.array([0., 0., 1.])
    if abs(palmar) >= abs(side):
        return ("palmar", np.array([0., 1., 0.])) if palmar >= 0 else ("dorsal", np.array([0., -1., 0.]))
    return ("lateral_positive", np.array([1., 0., 0.])) if side >= 0 else ("lateral_negative", np.array([-1., 0., 0.]))


def match_semantic_patches(shadow, contacts, wuji):
    """Return surface correspondences and fixed intrinsic masks per contact.

    Normal regions and longitudinal bands are hard membership conditions.
    Empty regions raise a diagnostic error instead of opening the full link.
    Metadata is JSON safe; only ``patch['mask']`` is a NumPy boolean array.
    """
    source_data = mujoco.MjData(shadow)
    mujoco.mj_kinematics(shadow, source_data)
    bodies, points, normals, records, patches = [], [], [], [], []
    generation = model_hand(wuji)
    for source, local in contacts:
        segment, prefix, finger_name, target = _source_semantic(source)
        source_body, body = shadow.body(source).id, wuji.body(target).id
        sf, slength, sradius = _source_chart(shadow, source_data, source_body, segment, prefix)
        pool = _target_pool(wuji, body, segment, finger_name)
        source_chart = (np.asarray(local[:3]) - sf["origin"]) @ _matrix(sf)
        normalized = source_chart / np.r_[sradius, slength]
        source_normal = _unit(local[3:]) @ _matrix(sf)
        region, axis = _region(source_normal, normalized[2], segment)
        target_normal = _matrix(pool["frame"]) @ source_normal
        radius = np.where(normalized[:2] >= 0., pool["positive_radius"], pool["negative_radius"])
        row_radius = np.where(pool["chart"][:, :2] >= 0.,
                              pool["positive_radius"], pool["negative_radius"])
        target_chart = pool["chart"] / np.column_stack((row_radius, np.full(len(row_radius), pool["length"])))
        normal_chart = pool["normals"] @ _matrix(pool["frame"])
        mask = pool["base"] & (normal_chart @ axis > .5)
        if segment.startswith("palm"):
            longitudinal_window = [max(0., normalized[2] - .25), min(1.2, normalized[2] + .25)]
            mask &= np.abs(target_chart[:, 0] - normalized[0]) <= .45
        elif region == "tip":
            longitudinal_window = [.75, 1.35]
        else:
            center = np.clip(normalized[2], .05, .95)
            longitudinal_window = [max(-.05, center - .22), min(1.05, center + .22)]
        mask &= (target_chart[:, 2] >= longitudinal_window[0]) & (target_chart[:, 2] <= longitudinal_window[1])
        # A normal cone can include internal parallel faces. Restrict to the
        # exposed half of the local cross section as well as ray visibility.
        projection = pool["chart"] @ axis
        if not segment.startswith("palm"):
            mask &= projection >= 0.0
        if not mask.any():
            raise ValueError(f"Empty semantic patch {source}->{target}/{region}; normalized={normalized.tolist()}")
        desired = np.r_[normalized[:2] * radius, normalized[2] * pool["length"]]
        delta = (pool["chart"] - desired) / np.r_[radius, pool["length"]]
        if segment.startswith("palm"):
            # The wrist body datum is not the palm's medial sheet. Distal
            # exterior palmar skin can have negative local Y while facing
            # +Y; transfer its lateral/longitudinal chart and project onto
            # the exposed skin rather than matching arbitrary shell depth.
            delta[:, 1] = 0.0
        score = np.sum(delta * delta, axis=1) + 2.0 * (1.0 - pool["normals"] @ target_normal)
        index = int(np.argmin(np.where(mask, score, np.inf)))
        label = f"{finger_name or segment}.{segment if finger_name else 'surface'}.{region}"
        error = np.degrees(np.arccos(np.clip(pool["normals"][index] @ target_normal, -1., 1.)))
        record = dict(
            shadow_body=source, wuji_body=target, shadow_local=np.asarray(local).tolist(),
            source_semantic_id=f"shadow.{label}", target_semantic_id=f"{generation}.{label}",
            semantic_patch_id=label, semantic_region=region,
            source_normalized_coordinates=normalized.tolist(),
            target_normalized_coordinates=target_chart[index].tolist(),
            source_chart_scales_m=np.r_[sradius, slength].tolist(),
            target_chart_scales_m=np.r_[radius, pool["length"]].tolist(),
            source_chart_scale_policy=("palm width/length from MCP landmarks; palmar depth 0.011m anatomical constant"
                if segment.startswith("palm") else "bone length from joint landmarks (distal 0.027m reference); radial anatomical constants 0.009m, distal 0.008m, thumb proximal 0.0105m; not measured Shadow mesh"),
            target_chart_scale_policy="joint/tip landmarks and 95th percentile exposed-query signed radial extents",
            source_chart_normal=source_normal.tolist(),
            target_chart_normal=normal_chart[index].tolist(),
            source_frame=_json_frame(sf), target_frame=_json_frame(pool["frame"]),
            longitudinal_window=longitudinal_window,
            query_index=index, wuji_local=pool["points"][index].tolist(),
            wuji_local_normal=pool["normals"][index].tolist(),
            mapping_distance_mm=float(np.linalg.norm(pool["chart"][index] - desired) * 1000),
            normal_error_deg=float(error), semantic_patch_samples=int(mask.sum()),
            chart_normal_error_deg=float(error),
            normalized_longitudinal_error=float(target_chart[index, 2] - normalized[2]),
            visibility_filtered=True,
            visibility_policy="outward normal ray from sample plus 10um; reject another rigid-owner mesh hit within 0.25m",
            mesh_owner_filtered=bool(segment.startswith("palm")),
            correspondence_method="intrinsic_bone_surface_patch",
            palm_depth_policy="exposed-surface projection; shell depth is not a transferred palm coordinate" if segment.startswith("palm") else None,
        )
        bodies.append(body); points.append(pool["points"][index]); normals.append(pool["normals"][index])
        records.append(record)
        patches.append(dict(id=label, mask=mask.copy(), frame=record["target_frame"],
                            region=region, longitudinal_window=longitudinal_window))
    return np.asarray(bodies, np.int32), np.asarray(points), np.asarray(normals), records, patches
