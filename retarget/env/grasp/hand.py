"""Grasp hand assets, independent of the legacy sequence-retargeting defaults."""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from retarget.config.run import ROOT

DEFAULT_HAND = "wuji2"
HAND_BODIES = ["right_palm_link"] + [
    f"right_finger{f}_link{k}" for f in range(1, 6) for k in range(1, 5)
]


def model_hand(model):
    marker = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_NUMERIC, "grasp_hand_generation"
    )
    return "wuji2" if marker >= 0 and model.numeric(marker).data[0] == 2 else "wuji1"


def asset_root(hand):
    if hand not in ("wuji1", "wuji2"):
        raise ValueError(f"Unsupported grasp hand: {hand}")
    return ROOT / "data/grasp/assets" / ("wuji2" if hand == "wuji2" else "wuji")


def query_path(model, body):
    name = body if isinstance(body, str) else model.body(int(body)).name
    return asset_root(model_hand(model)) / "qp_right" / f"{name}.npz"


def hand_tree(hand):
    root = asset_root(hand)
    tree = ET.parse(root / "right.xml").getroot()
    for element in [*tree.findall("./asset/mesh"), *tree.iter("geom")]:
        if element.get("file"):
            element.set("file", str((root / element.get("file")).resolve()))
    return tree


def rigid_mesh_arrays(model, body):
    """All distinct meshes rigidly attached to a body, including fixed pads.

    Express compiled mesh coordinates in the owner body's frame. The same
    geometry is used for query generation and conservative collision envelopes.
    Visual/collision duplicates are removed; distinct component meshes remain.
    """
    body = int(body)
    attached = {body}
    for child in range(body + 1, model.nbody):
        if (
            int(model.body_parentid[child]) in attached
            and model.body_jntnum[child] == 0
        ):
            attached.add(child)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    rotation = data.xmat[body].reshape(3, 3)
    vertices, faces, seen = [], [], set()
    count = 0
    for geom in range(model.ngeom):
        if (
            model.geom_bodyid[geom] not in attached
            or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH
        ):
            continue
        mesh = int(model.geom_dataid[geom])
        key = (
            int(model.geom_bodyid[geom]),
            mesh,
            *model.geom_pos[geom],
            *model.geom_quat[geom],
        )
        if key in seen:
            continue
        seen.add(key)
        start, size = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        world = (
            model.mesh_vert[start : start + size] @ data.geom_xmat[geom].reshape(3, 3).T
            + data.geom_xpos[geom]
        )
        local = (world - data.xpos[body]) @ rotation
        start, size = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        faces.append(model.mesh_face[start : start + size] + count)
        vertices.append(local)
        count += len(local)
    if not vertices:
        raise ValueError(f"No attached mesh on {model.body(body).name}")
    return np.concatenate(vertices), np.concatenate(faces)


def palmar_axis(model, body, endpoint):
    name = model.body(int(body)).name
    if name == "right_palm_link":
        return (
            np.array([0.0, 1.0, 0.0])
            if model_hand(model) == "wuji2"
            else np.array([1.0, 0.0, 0.0])
        )
    hinge = model.jnt_axis[model.body_jntadr[body]].copy()
    # Link 2's hinge is MCP abduction on Hand 2. Its pad faces along the
    # flexion direction, supplied by the following PIP/MCP flexion joint.
    if model_hand(model) == "wuji2" and name.endswith("link2"):
        child = model.body(name[:-1] + "3").id
        hinge = Rotation.from_quat(model.body_quat[child], scalar_first=True).apply(
            model.jnt_axis[model.body_jntadr[child]]
        )
    direction = np.cross(hinge, endpoint)
    if np.linalg.norm(direction) < 1.0e-10:
        raise ValueError(f"Degenerate palmar axis: {name}")
    return direction / np.linalg.norm(direction)


def body_boxes(model, bodies):
    """Body-local boxes enclosing complete meshes, including fixed attachments."""
    centers, half = [], []
    for body in bodies:
        vertices, _ = rigid_mesh_arrays(model, int(body))
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)
        centers.append((lo + hi) / 2)
        half.append((hi - lo) / 2 + 1.0e-6)
    return np.array(centers), np.array(half)
