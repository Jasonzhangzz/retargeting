"""Static hand/object scenes and template geometry for GraspADMM."""

from pathlib import Path


import xml.etree.ElementTree as ET

import mujoco

import numpy as np

from retarget.env.geometry import ObjectMeshSurface

from retarget.env.load_support import skew


def build_scene(mesh, scale=1.0, hand_model="wuji1"):
    """Independent fixed object and 26-DOF Wuji, without custom query geoms."""
    from retarget.env.grasp.hand import hand_tree

    tree = hand_tree(hand_model)
    for parent in tree.iter():
        for child in list(parent):
            if child.tag == "geom":
                visual = child.get("group") == "1"
                child.set("contype", "0" if visual else "1")
                child.set("conaffinity", "0" if visual else "1")
    ET.SubElement(
        tree.find("asset"),
        "mesh",
        name="grasp_object_mesh",
        file=str(Path(mesh).resolve()),
        scale=" ".join([str(scale)] * 3),
    )
    body = ET.SubElement(tree.find("worldbody"), "body", name="grasp_object")
    geom = ET.SubElement(
        body,
        "geom",
        name="grasp_object_geom",
        mesh="grasp_object_mesh",
        type="mesh",
        contype="2",
        conaffinity="1",
        rgba=".25 .65 .85 1",
    )
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    surface = ObjectMeshSurface.from_model(model, model.body("grasp_object").id)
    vertices = surface.triangles.reshape(-1, 3)
    center = (vertices.min(0) + vertices.max(0)) / 2
    geom.set("pos", " ".join(map(str, -center)))
    xml = ET.tostring(tree, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    surface = ObjectMeshSurface.from_model(model, model.body("grasp_object").id)
    return model, surface, xml, center


class StaticHand:
    """Input-only Wuji kinematics and query assets; contains no grasp solver."""

    def __init__(self, model, surface, bodies, local, normals):
        from retarget.env.grasp.hand import HAND_BODIES, query_path, model_hand

        self.model, self.surface = model, surface
        self.hand_model = model_hand(model)
        self.data = mujoco.MjData(model)
        self.bodies, self.local, self.normals = bodies, local, normals
        self.all_bodies, self.all_local = [], []
        for name in HAND_BODIES:
            with np.load(query_path(model, name)) as asset:
                queries = asset["local_pos"]
            self.all_bodies.extend([model.body(name).id] * len(queries))
            self.all_local.extend(queries)
        self.all_bodies, self.all_local = (
            np.asarray(self.all_bodies),
            np.asarray(self.all_local),
        )

    def set(self, q):
        self.data.qpos[:26] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        self.jacobians = {}

    def points(self, bodies, local, jacobian=False):
        rotation = self.data.xmat[bodies].reshape(-1, 3, 3)
        offsets = np.einsum("nij,nj->ni", rotation, local)
        points = self.data.xpos[bodies] + offsets
        if not jacobian:
            return points
        for body in np.unique(bodies):
            if body not in self.jacobians:
                jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
                mujoco.mj_jacBody(self.model, self.data, jp, jr, int(body))
                self.jacobians[body] = jp[:, :26], jr[:, :26]
        jp = np.array([self.jacobians[b][0] for b in bodies])
        jr = np.array([self.jacobians[b][1] for b in bodies])
        return points, jp - skew(offsets) @ jr, jr


def adapt_primitive_contacts(checkout, template, contacts):
    """Add the requested Power Sphere palm slot without editing vendor YAML."""
    if template.partition("_")[0] != "11" or any(
        body == "rh_palm" for body, _ in contacts
    ):
        return contacts, []
    import yaml

    keypoint_path = Path(checkout) / "assets/hand/shadow/keypoint.yaml"
    keys = yaml.safe_load(keypoint_path.read_text())
    palm = np.asarray(keys["rh_palm"][1], dtype=float)
    return [*contacts, ("rh_palm", palm.copy())], [
        dict(
            body="rh_palm",
            adapted_local=palm.tolist(),
            source="local Power Sphere adaptation: Dexonomy central palm keypoint 1",
            reason="Power Sphere requires an explicit palm contact slot",
        )
    ]
