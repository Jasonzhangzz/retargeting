"""Output-local native DexForge scenes for the existing Wuji grasp inputs.

Every hand surface sample becomes a positive-radius source sphere tangent to
the original surface. Rigid sensor pads and the mount remain in their physical
bodies for inertia/FK and are included in their articulated owner's cloud.
This is a measured sphere-cloud approximation, not an exact mesh collision
equivalence. Object collision uses the actual closed primitive mesh.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from retarget.env.grasp.hand import HAND_BODIES, query_path


@dataclass(frozen=True)
class SceneArtifacts:
    xml: Path
    viewer_xml: Path
    source_model: mujoco.MjModel
    model: mujoco.MjModel
    sphere_assets: dict
    manifest: dict


def _numbers(values):
    return " ".join(format(float(value), ".17g") for value in np.asarray(values).reshape(-1))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _owner(model, body):
    while body > 0:
        if model.body(body).name in HAND_BODIES:
            return body
        if model.body_jntnum[body] != 0:
            raise ValueError(f"Physical hand body has no Wuji surface cloud: {model.body(body).name}")
        body = int(model.body_parentid[body])
    raise ValueError("Hand collision geom is not attached to a Wuji surface owner")


def _canonical_tree(source, model, destination, timestep):
    viewer = destination / "source_scene.xml"
    mujoco.mj_saveLastXML(str(viewer), model)
    tree = ET.parse(viewer)
    root = tree.getroot()
    compiler = root.find("compiler")
    for asset in root.findall("asset/*"):
        filename = asset.get("file")
        if filename:
            directory = compiler.get("meshdir", "") if asset.tag == "mesh" and compiler is not None else (compiler.get("texturedir", "") if compiler is not None else "")
            asset.set("file", str((source.parent / directory / filename).resolve()))
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
        compiler.attrib.pop("texturedir", None)
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", _numbers([timestep]))
    bodies = list(root.find("worldbody").iter("body"))
    if len(bodies) != model.nbody - 1:
        raise ValueError("Canonical XML changed physical body ordering")
    for body_id, body in enumerate(bodies, 1):
        body.set("pos", _numbers(model.body_pos[body_id]))
        body.set("quat", _numbers(model.body_quat[body_id]))
        body.set("gravcomp", _numbers([model.body_gravcomp[body_id]]))
        if model.body_mass[body_id] > 0:
            inertial = body.find("inertial")
            if inertial is None:
                inertial = ET.Element("inertial")
                body.insert(0, inertial)
            inertial.attrib.clear()
            for name, values in (("pos", model.body_ipos[body_id]),
                                 ("quat", model.body_iquat[body_id]),
                                 ("mass", [model.body_mass[body_id]]),
                                 ("diaginertia", model.body_inertia[body_id])):
                inertial.set(name, _numbers(values))
    joints = list(root.find("worldbody").iter("joint"))
    if len(joints) != model.njnt:
        raise ValueError("Canonical XML changed joint ordering")
    for joint_id, joint in enumerate(joints):
        joint.set("limited", "true" if model.jnt_limited[joint_id] else "false")
        for name, values in (("pos", model.jnt_pos[joint_id]), ("axis", model.jnt_axis[joint_id]),
                             ("range", model.jnt_range[joint_id]),
                             ("stiffness", [model.jnt_stiffness[joint_id]])):
            joint.set(name, _numbers(values))
        if int(model.jnt_type[joint_id]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            dof = model.jnt_dofadr[joint_id]
            for name, values in (("damping", [model.dof_damping[dof]]),
                                 ("armature", [model.dof_armature[dof]]),
                                 ("frictionloss", [model.dof_frictionloss[dof]])):
                joint.set(name, _numbers(values))
    for actuator_id, actuator in enumerate(root.findall("actuator/*")):
        actuator.set("ctrllimited", "true" if model.actuator_ctrllimited[actuator_id] else "false")
        actuator.set("forcelimited", "true" if model.actuator_forcelimited[actuator_id] else "false")
        for name, values in (("gainprm", model.actuator_gainprm[actuator_id]),
                             ("biasprm", model.actuator_biasprm[actuator_id]),
                             ("dynprm", model.actuator_dynprm[actuator_id]),
                             ("gear", model.actuator_gear[actuator_id]),
                             ("ctrlrange", model.actuator_ctrlrange[actuator_id]),
                             ("forcerange", model.actuator_forcerange[actuator_id])):
            actuator.set(name, _numbers(values))
    geoms = list(root.find("worldbody").iter("geom"))
    if len(geoms) != model.ngeom:
        raise ValueError("Canonical XML changed original geometry ordering")
    for geom_id, geom in enumerate(geoms):
        if not geom.get("name"):
            geom.set("name", f"source_geom_{geom_id}")
        for name, values in (("friction", model.geom_friction[geom_id]),
                             ("solref", model.geom_solref[geom_id]),
                             ("solimp", model.geom_solimp[geom_id]),
                             ("margin", [model.geom_margin[geom_id]]),
                             ("gap", [model.geom_gap[geom_id]])):
            geom.set(name, _numbers(values))
    tree.write(viewer, encoding="utf-8", xml_declaration=False)
    return tree, viewer, bodies, geoms


def _object_mesh(model, geom, destination):
    if int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_MESH):
        raise ValueError("Primitive object collision must use one closed triangle mesh")
    mesh = int(model.geom_dataid[geom])
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    vertices = model.mesh_vert[start:start + count].astype(np.float64)
    rotation = np.empty(9, np.float64)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
    vertices = vertices @ rotation.reshape(3, 3).T + model.geom_pos[geom]
    start, count = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
    faces = model.mesh_face[start:start + count].copy()
    unique, inverse = np.unique(vertices, axis=0, return_inverse=True)
    faces = inverse[faces]
    edge = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, counts = np.unique(np.sort(edge, axis=1), axis=0, return_counts=True)
    if not np.all(counts == 2):
        raise ValueError("Primitive SDF object mesh must be closed and manifold")
    signed_volume = np.einsum("ij,ij->i", unique[faces[:, 0]],
                             np.cross(unique[faces[:, 1]], unique[faces[:, 2]])).sum() / 6
    if signed_volume <= 0:
        raise ValueError("Primitive object must have consistently outward-facing triangles")
    path = destination / "object_collision.obj"
    lines = ["v " + _numbers(row) for row in unique]
    lines.extend("f " + " ".join(str(int(i) + 1) for i in row) for row in faces)
    path.write_text("\n".join(lines) + "\n")
    return path, dict(vertices=len(unique), triangles=len(faces), signed_volume_m3=float(signed_volume),
                      original_mesh_name=model.mesh(mesh).name, object_body_local_coordinates=True)


def prepare_scene(source_xml, output_dir, *, timestep=0.01, sphere_radius=0.001, sphere_fit="tangent"):
    """Convert only collision representation to the pinned FA input contract.

    Body/joint/actuator names and ordering, inertias, transforms, gravity,
    servo parameters, friction and collision response settings are retained.
    Native source masks are hand-object-only, as required by upstream FA's
    one-output-per-guided-body mapping; self collision is its capsule cost.
    The ordinary viewer retains the original mesh collision masks and source
    integrator; native input uses Euler/pyramidal as required by original FA.
    """
    source, destination = Path(source_xml).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    if sphere_fit not in ("tangent", "convex_inside"):
        raise ValueError("sphere_fit must be tangent or convex_inside")
    if not source.is_file():
        raise FileNotFoundError(source)
    if (not np.isfinite(timestep) or timestep <= 0 or not np.isfinite(sphere_radius)
            or sphere_radius <= 0):
        raise ValueError("Native timestep and source sphere radius must be finite and positive")
    destination.mkdir(parents=True, exist_ok=True)
    original = mujoco.MjModel.from_xml_path(str(source))
    if (original.nq, original.nv, original.nu) != (33, 32, 26):
        raise ValueError("Native grasp scene requires 26 scalar hand coordinates and one free object")
    object_body = original.body("grasp_object").id
    physical = (original.geom_contype | original.geom_conaffinity) != 0
    objects = np.flatnonzero(physical & (original.geom_bodyid == object_body))
    if len(objects) != 1:
        raise ValueError("Native primitive scene needs exactly one object collision geom")
    tree, viewer, body_elements, geom_elements = _canonical_tree(source, original, destination, float(timestep))
    owner_geoms = {}
    for geom in np.flatnonzero(physical & (original.geom_bodyid != object_body)):
        owner = _owner(original, int(original.geom_bodyid[geom]))
        owner_geoms.setdefault(owner, []).append(int(geom))
    sphere_assets, records = {}, []
    for owner, ids in owner_geoms.items():
        name = original.body(owner).name
        reference = ids[0]
        for geom in ids:
            if int(original.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                raise ValueError(f"Hand collision surface must be a mesh: geom {geom}")
            for array in (original.geom_friction, original.geom_solref, original.geom_solimp,
                          original.geom_margin, original.geom_gap, original.geom_condim,
                          original.geom_priority):
                if not np.array_equal(array[geom], array[reference]):
                    raise ValueError(f"Rigid surfaces on {name} have different contact materials")
        query = query_path(original, name)
        with np.load(query, allow_pickle=False) as raw:
            surface = np.asarray(raw["local_pos"], np.float64)
            normals = np.asarray(raw["local_normal"], np.float64)
            source_meshes = raw["source_mesh"].astype(str) if "source_mesh" in raw else None
            source_bodies = raw["source_body"].astype(str) if "source_body" in raw else None
        lengths = np.linalg.norm(normals, axis=1)
        if (surface.ndim != 2 or surface.shape[1] != 3 or normals.shape != surface.shape
                or len(surface) < 3 or not np.isfinite(surface).all()
                or not np.isfinite(normals).all() or np.any(lengths < 1e-12)):
            raise ValueError(f"Invalid complete hand surface cloud: {query}")
        normals = normals / lengths[:, None]
        if source_meshes is not None:
            expected = {original.mesh(int(original.geom_dataid[i])).name for i in ids}
            if expected - set(source_meshes):
                raise ValueError(f"Surface cloud omits a physical mesh on {name}: {expected - set(source_meshes)}")
        asset = destination / f"gs_{name}.npz"
        fit_report = None
        if sphere_fit == "convex_inside":
            if source_meshes is None or source_bodies is None:
                raise ValueError("Contained spheres require explicit source mesh/body provenance")
            from .forceaware_geometry import constrain_tangent_spheres
            centers, radii, fit_report = constrain_tangent_spheres(original, owner, ids,
                surface, normals, source_meshes, source_bodies, sphere_radius)
        else:
            radii = np.full(len(surface), sphere_radius, np.float32)
            centers = (surface - sphere_radius * normals).astype(np.float32)
        # Explicit group_id prevents provenance integer arrays becoming
        # unintended collision-output partitions in the native loader.
        np.savez_compressed(asset, local_pos=centers, radius=radii,
                            local_normal=normals.astype(np.float32),
                            group_id=np.zeros(len(surface), np.int32))
        for geom in ids:
            geom_elements[geom].set("contype", "0")
            geom_elements[geom].set("conaffinity", "0")
        ET.SubElement(body_elements[owner - 1], "geom", name=f"fa_gs_{name}", type="gs", file=str(asset),
                      size=_numbers([sphere_radius]), pos="0 0 0", quat="1 0 0 0", mass="0",
                      contype="1", conaffinity="4", group="4",
                      condim=str(int(original.geom_condim[reference])),
                      priority=str(int(original.geom_priority[reference])),
                      friction=_numbers(original.geom_friction[reference]),
                      solref=_numbers(original.geom_solref[reference]),
                      solimp=_numbers(original.geom_solimp[reference]),
                      margin=_numbers([original.geom_margin[reference]]),
                      gap=_numbers([original.geom_gap[reference]]))
        sphere_assets[name] = asset
        records.append(dict(owner_body=name, owner_body_id=int(owner), samples=len(surface),
                            source_query=str(query), source_query_sha256=_sha256(query),
                            source_physical_geom_ids=ids,
                            rigid_component_bodies=sorted(set(source_bodies)) if source_bodies is not None else None,
                            radius_m=float(sphere_radius), radius_min_m=float(radii.min()), radius_max_m=float(radii.max()),
                            center_shift="surface_point - radius * unit_outward_normal; optional bounded numerical inset",
                            sphere_fit=sphere_fit, containment=fit_report,
                            native_cloud_sha256=_sha256(asset), group_id=0,
                            semantic_filter_applied=False, native_collision_mask=[1, 4],
                            original_collision_masks=[[int(original.geom_contype[i]), int(original.geom_conaffinity[i])] for i in ids]))
    obj = int(objects[0])
    object_mesh, object_record = _object_mesh(original, obj, destination)
    element = geom_elements[obj]
    element.attrib.pop("mesh", None)
    element.set("type", "sdfmesh")
    element.set("file", str(object_mesh))
    element.set("pos", "0 0 0")
    element.set("quat", "1 0 0 0")
    element.set("mass", "0")
    element.set("size", "0.001")
    element.set("contype", "4")
    element.set("conaffinity", "1")
    option = tree.getroot().find("option")
    source_integrator = option.get("integrator", "Euler")
    option.set("integrator", "Euler")
    option.set("cone", "pyramidal")
    xml = destination / "native_scene.xml"
    tree.write(xml, encoding="utf-8", xml_declaration=False)
    model = mujoco.MjModel.from_xml_path(str(viewer))
    if model.opt.integrator != original.opt.integrator:
        raise ValueError("Ordinary replay scene changed the source integrator")
    manifest = dict(source_xml=str(source), source_sha256=_sha256(source), native_xml=str(xml),
                    viewer_xml=str(viewer), native_sha256=_sha256(xml),
                    original_dimensions=dict(nq=original.nq, nv=original.nv, nu=original.nu),
                    native_dimensions=dict(nq=model.nq, nv=model.nv, nu=model.nu),
                    source_sphere_radius_m=float(sphere_radius), sphere_fit=sphere_fit,
                    geometry_approximation=("Full original samples fit inside their physical per-geom convex collision bodies; not exact surface coverage"
                        if sphere_fit == "convex_inside" else "Full original hand surface samples replaced by inward-offset tangent spheres; not exact mesh collision equivalence"),
                    hand_object_skin="all query samples, including rigid pads and mount; no contact-annotation or normal-facing filter",
                    hand_self_collision="Native GS hand-hand pairs disabled to satisfy original FA unique-guided-body mapping; original FA capsule self-collision objective retained by the workflow",
                    original_timestep=float(original.opt.timestep), viewer_timestep=float(model.opt.timestep), native_timestep=float(timestep),
                    source_integrator=source_integrator, viewer_integrator=source_integrator,
                    native_integrator="Euler", native_cone="pyramidal", gravity=original.opt.gravity.tolist(),
                    body_order=[original.body(i).name for i in range(original.nbody)],
                    converted_hand_surfaces=records, object_mesh=dict(**object_record, path=str(object_mesh), sha256=_sha256(object_mesh)))
    (destination / "scene_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return SceneArtifacts(xml, viewer, original, model, sphere_assets, manifest)
