"""Central fingertip-pad guidance for the three prismatic grasp templates."""

from pathlib import Path
import hashlib
import numpy as np
import mujoco
import yaml

from retarget.env.grasp.hand import palmar_axis, query_path, rigid_mesh_arrays

TYPES = (6, 7, 8)
CENTER_RADIUS_M = 0.003
CENTER_WEIGHT = 4.0


def centered_template_contacts(checkout, template, contacts):
    """Use the upstream palmar pad keypoint, preserving annotation count/order.

    The original YAML is unchanged. Its custom extreme-tip points are retained
    in the adaptation diagnostics rather than silently called original inputs.
    """
    if template.partition("_")[0] not in {str(i) for i in TYPES}:
        return contacts, []
    keypoint_path = Path(checkout) / "assets/hand/shadow/keypoint.yaml"
    keys = yaml.safe_load(keypoint_path.read_text())
    updated, changes = [], []
    for body, local in contacts:
        replacement = np.asarray(local).copy()
        if body.endswith("distal"):
            alias = body
            seen = set()
            while isinstance(keys[alias], str):
                if alias in seen:
                    raise ValueError("Cyclic fingertip keypoint alias")
                seen.add(alias)
                alias = keys[alias]
            replacement = np.asarray(keys[alias][0], dtype=float)
            changes.append(
                dict(
                    body=body,
                    original_local=np.asarray(local).tolist(),
                    adapted_local=replacement.tolist(),
                    source="Dexonomy keypoint 0: palmar tip-pad center",
                    source_keypoint_sha256=hashlib.sha256(
                        keypoint_path.read_bytes()
                    ).hexdigest(),
                )
            )
        updated.append((body, replacement))
    return updated, changes


def tip_patch(model, body):
    """Area-weighted center of the exposed sensor pad, snapped to a real query.

    Hand 2 has a distinct rigid sensor-pad mesh. Hand 1 uses the palmar face
    of its attached tip mesh. No object contacts or force-query samples change.
    """
    name = model.body(int(body)).name
    finger = int(name[len("right_finger")])
    endpoint = model.site(
        f"right_{('thumb', 'index', 'middle', 'ring', 'pinky')[finger - 1]}_tip"
    ).pos.copy()
    belly = palmar_axis(model, body, endpoint)
    sensor = next(
        (
            i
            for i in range(model.nbody)
            if model.body_parentid[i] == body
            and "tip_sensor_frame" in model.body(i).name
        ),
        None,
    )
    owner = body if sensor is None else sensor
    vertices, faces = rigid_mesh_arrays(model, owner)
    if owner != body:
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        world = vertices @ data.xmat[owner].reshape(3, 3).T + data.xpos[owner]
        vertices = (world - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    area = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(area[:, None], 1.0e-15)
    centers = triangles.mean(axis=1)
    projection = vertices @ belly
    front = (
        (normals @ belly > 0.85)
        & (area > 1.0e-15)
        & (centers @ belly > (projection.min() + projection.max()) / 2)
    )
    if not front.any():
        raise ValueError(f"No exposed central fingertip pad: {name}")
    centroid = np.average(centers[front], axis=0, weights=area[front])
    with np.load(query_path(model, body)) as asset:
        points, normals = asset["local_pos"].copy(), asset["local_normal"].copy()
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
    exposed = normals @ belly > 0.75
    score = np.sum((points - centroid) ** 2, axis=1)
    index = int(np.argmin(np.where(exposed, score, np.inf)))
    center = points[index]
    allowed = exposed & (
        np.linalg.norm(points - center, axis=1) <= CENTER_RADIUS_M + 1.0e-9
    )
    return dict(
        center=center,
        normal=normals[index],
        mask=allowed,
        index=index,
        surface_centroid=centroid,
        radius_m=CENTER_RADIUS_M,
        weight=CENTER_WEIGHT,
        sensor_mesh=(model.body(owner).name if sensor is not None else None),
    )


def configure_tip_centers(hand, template, mapping):
    if template.partition("_")[0] not in {str(i) for i in TYPES}:
        return
    hand.tip_patches = {}
    for i, body in enumerate(hand.bodies):
        if not hand.model.body(int(body)).name.endswith("link4"):
            continue
        if int(body) not in hand.tip_patches:
            hand.tip_patches[int(body)] = tip_patch(hand.model, int(body))
        patch = hand.tip_patches[int(body)]
        mapping[i]["original_mapped_query_index"] = mapping[i]["query_index"]
        mapping[i]["contact_policy"] = "central_tip_sensor_pad"
        mapping[i]["tip_patch_reanchor_shift_mm"] = float(
            np.linalg.norm(hand.local[i] - patch["center"]) * 1000
        )
        mapping[i].update(
            query_index=patch["index"],
            wuji_local=patch["center"].tolist(),
            wuji_local_normal=patch["normal"].tolist(),
            tip_patch_center=patch["center"].tolist(),
            tip_patch_radius_mm=patch["radius_m"] * 1000,
        )
        hand.local[i] = patch["center"]
        hand.normals[i] = patch["normal"]
