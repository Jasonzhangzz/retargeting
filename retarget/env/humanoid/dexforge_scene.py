"""Prepare native DexForge collision assets without changing source physics.

Original robot collision spheres become one-sphere GS clouds at their original
geom transforms and radii. A physical box becomes its exact, closed triangle
surface. Native plane collision and all original contact material/pair settings
are retained. Four palm/sole sites are observations, not substitute geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


OBSERVATION_SITES = ("left_palm", "right_palm", "left_foot", "right_foot")


def _upstream_paths():
    from retarget.dynamics.dexforge_upstream import activate_upstream

    return activate_upstream()


@dataclass(frozen=True)
class SceneArtifacts:
    xml: Path
    source_model: mujoco.MjModel
    model: mujoco.MjModel
    observation_site_ids: tuple[int, ...]
    sphere_assets: dict[str, Path]
    manifest: dict
    viewer_xml: Path


def _numbers(values):
    return " ".join(format(float(value), ".17g") for value in np.asarray(values).reshape(-1))


def _box_obj(path, half_size):
    # Outward-oriented triangles, shared vertices, and exactly two faces per
    # edge satisfy the native SDF target's closed-mesh contract.
    x, y, z = map(float, half_size)
    vertices = np.array([[-x,-y,-z], [x,-y,-z], [x,y,-z], [-x,y,-z],
                         [-x,-y,z], [x,-y,z], [x,y,z], [-x,y,z]])
    faces = np.array([[0,2,1], [0,3,2], [4,5,6], [4,6,7],
                      [0,1,5], [0,5,4], [1,2,6], [1,6,5],
                      [2,3,7], [2,7,6], [3,0,4], [3,4,7]])
    lines = ["v " + _numbers(vertex) for vertex in vertices]
    lines.extend("f " + " ".join(str(int(v)+1) for v in face) for face in faces)
    path.write_text("\n".join(lines) + "\n")


def prepare_scene(source_xml, output_dir, *, timestep=None, observation_sites=OBSERVATION_SITES):
    """Write an output-local native XML, sphere assets, and source manifest.

    Generated assets and their content provenance stay in output_dir. The source XML,
    NPZ reference, body inertias, actuator transmissions and gravity are never
    edited. ``model`` is an ordinary MuJoCo CPU preview for FK/reference indices.
    The runner loads ``xml`` with the original upstream native loader and uses
    that loader's returned CPU/device models for dynamics. ``viewer_xml`` is
    ordinary MuJoCo XML for replay.
    """
    source = Path(source_xml).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.mkdir(parents=True, exist_ok=True)
    original = mujoco.MjModel.from_xml_path(str(source))
    if timestep is not None and (not np.isfinite(timestep) or timestep <= 0):
        raise ValueError("native scene timestep must be finite and positive")
    dt = float(original.opt.timestep if timestep is None else timestep)
    ids = tuple(mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_SITE, name) for name in observation_sites)
    if len(ids) != 4 or len(set(ids)) != 4 or any(i < 0 for i in ids):
        raise ValueError("native scene requires four distinct existing palm/sole observation sites")
    viewer = destination / "source_scene.xml"
    mujoco.mj_saveLastXML(str(viewer), original)
    tree = ET.parse(viewer)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        for key in ("meshdir", "texturedir"):
            directory = compiler.get(key)
            if directory:
                compiler.set(key, str((source.parent / directory).resolve()))
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", format(dt, ".17g"))
    option.set("integrator", "Euler")
    option.set("cone", "pyramidal")
    # Preserve compiled inertias explicitly, including bodies whose original
    # inertia was inferred from physical spheres that will become GS assets.
    bodies = list(root.find("worldbody").iter("body"))
    for body_id, body in enumerate(bodies, start=1):
        body.set("pos", _numbers(original.body_pos[body_id]))
        body.set("quat", _numbers(original.body_quat[body_id]))
        inertia = body.find("inertial")
        if inertia is None:
            inertia = ET.Element("inertial")
            body.insert(0, inertia)
        inertia.attrib.clear()
        inertia.set("mass", format(float(original.body_mass[body_id]), ".17g"))
        inertia.set("pos", _numbers(original.body_ipos[body_id]))
        inertia.set("quat", _numbers(original.body_iquat[body_id]))
        inertia.set("diaginertia", _numbers(original.body_inertia[body_id]))
    # Resolve every original asset path before moving the generated XML.
    for asset in root.findall("asset/*"):
        filename = asset.get("file")
        if filename:
            directory = compiler.get("meshdir", "") if asset.tag == "mesh" and compiler is not None else (compiler.get("texturedir", "") if compiler is not None else "")
            asset.set("file", str((Path(directory or source.parent) / filename).resolve()))
    geoms = list(root.find("worldbody").iter("geom"))
    if len(geoms) != original.ngeom:
        raise ValueError("canonical MuJoCo scene changed the original geom ordering")
    # mj_saveLastXML uses short decimal strings for some compiled values (in
    # particular servo damping). Restore full precision before either model is
    # compiled, so changing collision representation does not alter actuators.
    for geom_id, element in enumerate(geoms):
        properties = [("friction", original.geom_friction), ("solref", original.geom_solref),
                      ("solimp", original.geom_solimp)]
        # Compiled mesh geom poses include MuJoCo's asset centering/principal
        # axes. Reapplying them to the original mesh would transform it twice.
        if int(original.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            properties += [("pos", original.geom_pos), ("quat", original.geom_quat)]
        for key, array in properties:
            element.set(key, _numbers(array[geom_id]))
        for orientation in ("euler", "axisangle", "xyaxes", "zaxis", "fromto"):
            element.attrib.pop(orientation, None)
    for actuator_id, actuator in enumerate(root.findall("actuator/*")):
        for key, array in (("gainprm", original.actuator_gainprm), ("biasprm", original.actuator_biasprm),
                           ("dynprm", original.actuator_dynprm), ("gear", original.actuator_gear)):
            actuator.set(key, _numbers(array[actuator_id]))
    for pair_id, pair in enumerate(root.findall("contact/pair")):
        for key, array in (("solref", original.pair_solref), ("solreffriction", original.pair_solreffriction),
                           ("solimp", original.pair_solimp), ("friction", original.pair_friction)):
            pair.set(key, _numbers(array[pair_id]))
    # The ordinary viewer scene retains the exact original collision shapes.
    tree.write(viewer, encoding="utf-8", xml_declaration=False)
    paired = set(map(int, original.pair_geom1)) | set(map(int, original.pair_geom2))
    sphere_assets = {}
    converted = []
    for geom_id, element in enumerate(geoms):
        physical = (original.geom_contype[geom_id] | original.geom_conaffinity[geom_id]) != 0 or geom_id in paired
        if not physical:
            continue
        name = element.get("name") or f"native_geom_{geom_id}"
        element.set("name", name)
        kind = int(original.geom_type[geom_id])
        if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            asset = destination / f"sphere_{geom_id}.npz"
            np.savez(asset, local_pos=np.zeros((1, 3), np.float32),
                     radius=np.asarray([original.geom_size[geom_id, 0]], np.float32))
            element.set("type", "gs")
            element.set("file", str(asset))
            element.set("size", format(float(original.geom_size[geom_id, 0]), ".17g"))
            sphere_assets[name] = asset
            converted.append(dict(name=name, source_type="sphere", native_type="gs", radius=float(original.geom_size[geom_id, 0])))
        elif kind == int(mujoco.mjtGeom.mjGEOM_BOX):
            asset = destination / f"box_{geom_id}.obj"
            _box_obj(asset, original.geom_size[geom_id])
            element.set("type", "sdfmesh")
            element.set("file", str(asset))
            element.set("size", format(float(original.geom_size[geom_id].max()), ".17g"))
            converted.append(dict(name=name, source_type="box", native_type="closed_mesh", half_size=original.geom_size[geom_id].tolist()))
        elif kind == int(mujoco.mjtGeom.mjGEOM_MESH):
            raise ValueError(f"Physical mesh geom {name!r} requires an explicit native SDF mesh conversion")
        elif kind != int(mujoco.mjtGeom.mjGEOM_PLANE):
            raise ValueError(f"Physical geom {name!r} must be a sphere, box, or plane")
        # Compiled pose removes ambiguity from inherited/default orientations.
        element.set("pos", _numbers(original.geom_pos[geom_id]))
        element.set("quat", _numbers(original.geom_quat[geom_id]))
        for orientation in ("euler", "axisangle", "xyaxes", "zaxis", "fromto"):
            element.attrib.pop(orientation, None)
    native_xml = destination / "native_scene.xml"
    tree.write(native_xml, encoding="utf-8", xml_declaration=False)
    model = mujoco.MjModel.from_xml_path(str(viewer))
    manifest = dict(source_xml=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    native_xml=str(native_xml), viewer_xml=str(viewer), observation_sites=list(observation_sites),
                    original_timestep=float(original.opt.timestep), native_timestep=dt,
                    native_integrator="Euler", native_cone="pyramidal", converted_geoms=converted,
                    original_pair_count=int(original.npair), gravity=original.opt.gravity.tolist(),
                    original_dimensions=dict(nq=int(original.nq), nv=int(original.nv), nu=int(original.nu)))
    (destination / "scene_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return SceneArtifacts(native_xml, original, model, ids, sphere_assets, manifest, viewer)
