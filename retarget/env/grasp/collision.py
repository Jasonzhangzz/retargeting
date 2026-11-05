"""Shared collision extraction and rigid WUJI2 scene construction."""

from pathlib import Path
import xml.etree.ElementTree as ET
import warp as wp
from .hand import hand_tree


@wp.kernel(enable_backward=False)
def _depths(
    count: wp.array[int],
    worlds: wp.array[int],
    geoms: wp.array[wp.vec2i],
    distance: wp.array[float],
    object_geom: wp.array[int],
    clearance: float,
    depths: wp.array2d[float],
):
    i = wp.tid()
    if i >= count[0]:
        return
    pair = geoms[i]
    if pair[0] < 0 or pair[1] < 0:
        return
    a = object_geom[pair[0]]
    b = object_geom[pair[1]]
    if a != 0 and b != 0:
        return
    kind = int(1)
    if a != b:
        kind = 0
    wp.atomic_max(depths, worlds[i], kind, wp.max(clearance - distance[i], 0.0))


def refinement_model(mesh_path, settings=None):
    """Same official hand and actual object-local mesh; no recentering/scaling."""
    tree = hand_tree("wuji2")
    if settings is None:
        from .contact_geometry import PoseObjectiveSettings

        settings = PoseObjectiveSettings()
    tree.find("option").set("gravity", "0 0 0")
    # MJWarp does not support positive margins with MULTICCD. The single-CCD
    # distance is only optimization guidance; the pinned original evaluation
    # reconstructs its own model and retains its original collision settings.
    ET.SubElement(tree.find("option"), "flag", multiccd="disable")
    for geom in tree.iter("geom"):
        if geom.get("group") == "1":
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
        else:
            geom.set("contype", "1")
            geom.set("conaffinity", "1")
            geom.set("margin", str(settings.collision_margin_m))
            geom.set("condim", "1")
    ET.SubElement(
        tree.find("asset"),
        "mesh",
        name="refinement_object",
        file=str(Path(mesh_path).resolve()),
    )
    obj = ET.SubElement(tree.find("worldbody"), "body", name="grasp_object")
    ET.SubElement(
        obj,
        "geom",
        type="mesh",
        mesh="refinement_object",
        contype="2",
        conaffinity="1",
        margin=str(settings.collision_margin_m),
        condim="1",
        rgba=".25 .65 .85 1",
    )
    return ET.tostring(tree, encoding="unicode")
