"""Wuji2 geometry/frame adapters for the unmodified GraspADMM algorithm."""

from dataclasses import dataclass
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .graspadmm_source import require_upstream


def to_upstream_qpos(qpos):
    q = np.asarray(qpos, dtype=float)
    if q.shape[-1] == 27:
        if not np.isfinite(q).all() or np.any(np.linalg.norm(q[..., 3:7], axis=-1) < 1e-12):
            raise ValueError("Expected finite xyz/wxyz/20-joint coordinates and a nonzero quaternion")
        return q.copy()
    if q.shape[-1] != 26 or not np.isfinite(q).all():
        raise ValueError("Expected finite Wuji xyz/Euler/20-joint coordinates")
    quat = Rotation.from_euler("XYZ", q[..., 3:6]).as_quat(scalar_first=True)
    return np.concatenate((q[..., :3], quat, q[..., 6:]), axis=-1)


def from_upstream_qpos(qpos):
    q = np.asarray(qpos, dtype=float)
    if q.shape[-1] != 27 or not np.isfinite(q).all():
        raise ValueError("Expected finite Wuji xyz/wxyz/20-joint coordinates")
    angles = Rotation.from_quat(q[..., 3:7], scalar_first=True).as_euler("XYZ")
    return np.concatenate((q[..., :3], angles, q[..., 7:]), axis=-1)


def canonical_hand_xml(hand="wuji2"):
    if hand != "wuji2":
        raise ValueError("The GraspADMM bridge currently supports wuji2")
    from .hand import hand_tree

    tree = hand_tree(hand)
    world = tree.find("worldbody")
    palm = tree.find(".//body[@name='right_palm_link']")
    for parent in tree.iter():
        if palm in list(parent):
            parent.remove(palm)
            break
    for child in list(world):
        world.remove(child)
    palm.set("pos", "0 0 0")
    palm.set("quat", "1 0 0 0")
    world.append(palm)
    actuators = tree.find("actuator")
    for element in list(actuators):
        if element.get("joint", "").startswith("grasp_wrist_"):
            actuators.remove(element)
    # The upstream scene owns its numerical integration/contact settings.
    for tag in ("option", "keyframe"):
        element = tree.find(tag)
        if element is not None:
            tree.remove(element)
    return ET.tostring(tree, encoding="unicode")


@dataclass(frozen=True)
class SceneFiles:
    hand_xml: Path
    scene_path: Path
    scene_cfg: dict
    mesh_path: Path


def write_scene(directory, mesh_path, *, scale=1.0, hand="wuji2", object_mass=.1):
    """Write one fixed-object optimization / free-object evaluation scene.

    The supplied mesh is already in the desired object-local frame. Scaling is
    baked into its vertices, with no additional recentering or approximation.
    """
    import trimesh

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load_mesh(Path(mesh_path).resolve(), process=False)
    mesh.apply_scale(scale)
    output_mesh = directory / "object.obj"
    mesh.export(output_mesh, digits=17)
    volume = abs(float(mesh.volume))
    if not np.isfinite(volume) or volume <= 1e-12:
        raise ValueError("GraspADMM expects a closed, nonzero-volume object mesh")
    hand_path = directory / "wuji2_hand.xml"
    hand_path.write_text(canonical_hand_xml(hand))
    object_xml = directory / "object.xml"
    object_tree = ET.Element("mujoco", model="graspadmm_object")
    ET.SubElement(object_tree, "compiler", angle="radian")
    asset = ET.SubElement(object_tree, "asset")
    ET.SubElement(asset, "mesh", name="mesh", file=str(output_mesh))
    world = ET.SubElement(object_tree, "worldbody")
    ET.SubElement(world, "geom", name="mesh", type="mesh", mesh="mesh", density=str(object_mass / volume))
    object_xml.write_text(ET.tostring(object_tree, encoding="unicode"))
    info = directory / "object_info.json"
    info.write_text(json.dumps(dict(mass=object_mass, density=object_mass / volume, scale=1.0)))
    scene_cfg = {
        "scene_id": directory.name,
        "scene": {"object": {
            "type": "rigid_object", "xml_path": str(object_xml),
            "file_path": str(output_mesh), "info_path": str(info),
            "pose": [0., 0., 0., 1., 0., 0., 0.], "scale": [1., 1., 1.],
            "density": object_mass / volume,
        }},
        "task": {"type": "force_closure", "obj_name": "object"},
    }
    scene_path = directory / "scene.npy"
    np.save(scene_path, scene_cfg)
    return SceneFiles(hand_path, scene_path, scene_cfg, output_mesh)


_PAD_OWNERS = {
    f"r_{name}_tip_sensor_frame": f"right_finger{finger}_link4"
    for finger, name in enumerate(("thumb", "index_finger", "middle_finger", "ring_finger", "pinky"), 1)
}


def canonical_body(name):
    return _PAD_OWNERS.get(name, name)


def _canonical_contacts(contacts):
    contacts = dict(contacts)
    contacts["bn1"] = [canonical_body(name) for name in contacts["bn1"]]
    return contacts


def _env_classes():
    source = require_upstream()

    class OptEnv(source.sim.MuJoCo_OptEnv):
        def get_contacts(self, obj_margin=None):
            ho, hh = super().get_contacts(obj_margin)
            return _canonical_contacts(ho), hh

    class EvalEnv(source.sim.MuJoCo_EvalEnv):
        def get_contacts(self, obj_margin=None):
            ho, hh = super().get_contacts(obj_margin)
            return _canonical_contacts(ho), hh

    return OptEnv, EvalEnv


def WujiOptEnv(*args, **kwargs):
    return _env_classes()[0](*args, **kwargs)


def WujiEvalEnv(*args, **kwargs):
    return _env_classes()[1](*args, **kwargs)


def project_to_hand(pts, hand_cbody, env, meshes, proxyes, geom_filter="all"):
    """Original projection math applied to all rigid-owner collision meshes.

    Wuji2 fingertip pads are fixed child bodies. This geometry-only extension
    prevents the upstream first-geom rule from projecting onto the metal link
    when the fixed sensor pad is nearer.
    """
    from trimesh.proximity import ProximityQuery
    import trimesh

    points = np.asarray(pts)
    world = np.zeros_like(points)
    local = np.zeros_like(points)
    cache = getattr(env, "_wuji_projection_meshes", None)
    if cache is None:
        cache = {}
        env._wuji_projection_meshes = cache
    for i, name in enumerate(hand_cbody):
        owner = env._model.body(env._cfg.hand_prefix + name).id
        rotation = env._data.xmat[owner].reshape(3, 3)
        origin = env._data.xpos[owner]
        if owner not in cache:
            attached = {owner}
            for child in range(owner + 1, env._model.nbody):
                if env._model.body_parentid[child] in attached and env._model.body_jntnum[child] == 0:
                    attached.add(child)
            pool = []
            for geom in range(env._model.ngeom):
                if env._model.geom_bodyid[geom] not in attached:
                    continue
                visual = env._model.geom_contype[geom] == 0 and env._model.geom_conaffinity[geom] == 0
                if (geom_filter == "collision" and visual) or (geom_filter == "visual" and not visual):
                    continue
                if env._model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                    raise ValueError("Wuji2 rigid-owner projection expects mesh collision geometry")
                mesh = int(env._model.geom_dataid[geom])
                start, count = env._model.mesh_vertadr[mesh], env._model.mesh_vertnum[mesh]
                vertices = env._model.mesh_vert[start:start + count]
                verts_world = vertices @ env._data.geom_xmat[geom].reshape(3, 3).T + env._data.geom_xpos[geom]
                start, count = env._model.mesh_faceadr[mesh], env._model.mesh_facenum[mesh]
                faces = env._model.mesh_face[start:start + count]
                pool.append(trimesh.Trimesh((verts_world - origin) @ rotation, faces, process=False))
            if not pool:
                raise ValueError(f"No collision geometry for Wuji2 contact body {name}")
            combined = trimesh.util.concatenate(pool)
            cache[owner] = ProximityQuery(combined)
        p_local = ((points[i] - origin) @ rotation).reshape(1, 3)
        projected, _, _ = cache[owner].on_surface(p_local)
        local[i] = projected[0]
        world[i] = local[i] @ rotation.T + origin
    return world, local


def required_body_groups(hand_cbody):
    annotated = set(hand_cbody)
    groups = [[f"right_finger{finger}_link{k}" for k in range(1, 5)] for finger in range(1, 6)]
    groups.append(["right_palm_link"])
    return [present for group in groups if (present := [name for name in group if name in annotated])]


def normalize_grasp_data(qpos, scene, *, hand_cbody=(), hand_cpn_w=None, obj_cpn_w=None,
                         required_cbody=None, template="candidate"):
    import trimesh

    count = len(hand_cbody)
    hand_contacts = np.empty((0, 6)) if hand_cpn_w is None else np.asarray(hand_cpn_w)
    object_contacts = np.empty((0, 6)) if obj_cpn_w is None else np.asarray(obj_cpn_w)
    if hand_contacts.shape != (count, 6) or object_contacts.shape != (count, 6):
        raise ValueError("Each contact body must have one hand/object contact six-vector")
    return dict(
        hand_name="wuji2", tmpl_name=template, hand_cbody=list(hand_cbody),
        required_cbody=required_body_groups(hand_cbody) if required_cbody is None else required_cbody,
        grasp_qpos=to_upstream_qpos(np.asarray(qpos).reshape(-1))[None].astype(np.float32),
        hand_cpn_w=hand_contacts.astype(np.float32), obj_cpn_w=object_contacts.astype(np.float32),
        ext_wrench=np.zeros(6, np.float32),
        ext_center=np.asarray(trimesh.load_mesh(scene.mesh_path, process=False).center_mass, np.float32),
        scene_path=str(scene.scene_path), n_evo=np.array([0.], np.float32),
    )


def filter_diagnostics(env, cfg, info):
    """Record original filter values without adding acceptance conditions."""
    ho, hh = env.get_contacts(cfg.contact.thre if "contact" in cfg else None)
    groups = info.get("required_cbody", []) if info else []
    bodies = set(ho["bn1"])
    return dict(
        joint_limits=bool(env.check_qpos_limit(cfg.limit.thre)) if "limit" in cfg else None,
        hand_hand_min_distance_m=float(np.min(hh["dist"])) if len(hh["dist"]) else None,
        hand_object_min_distance_m=float(np.min(ho["dist"])) if len(ho["dist"]) else None,
        contacts=len(ho["dist"]), required_cbody=groups,
        required_groups_contact=[bool(bodies.intersection(group)) for group in groups],
        contact_bodies=sorted(bodies),
    )


def native_finalize(grasp_data, scene, settings=None):
    """Run the original final filter/squeeze/pregrasp stages on a final pose.

    Common evaluation calls this with settings=None to fix all thresholds to
    upstream defaults. No optimizer objective or semantic gate is added.
    """
    source = require_upstream()
    cfg = copy.deepcopy(source.grasp_config) if settings is None else settings
    data = copy.deepcopy(grasp_data)
    env = WujiOptEnv(
        hand_cfg=source.sim.HandCfg(str(scene.hand_xml), freejoint=True),
        scene_cfg=scene.scene_cfg,
        sim_cfg=source.sim.MuJoCo_OptCfg(obj_margin=cfg.pregrasp.cdist),
    )
    env.reset_qpos(data["grasp_qpos"][0])
    env.set_obj_margin(cfg.grasp.cdist)
    grasp_filter = source.core.GraspFilter(cfg.grasp.filter, env, "common grasp")
    ok, ho = grasp_filter.forward(data)
    diagnostics = filter_diagnostics(env, cfg.grasp.filter, data)
    diagnostics["success"] = bool(ok)
    data["ho_c"] = ho
    status = dict(success=False, failure_stage="grasp_filter", grasp_filter=diagnostics, pregrasp_filter=None)
    trace = {"pregrasp_qpos": []}
    if not ok:
        return data, status, trace
    data["squeeze_qpos"] = env.get_squeeze_qpos(data["grasp_qpos"][0], ho["bn1"], ho["pos"], ho["wrench"])[None].astype(np.float32)
    pregrasp_filter = source.core.GraspFilter(cfg.pregrasp.filter, env, "common pregrasp")
    pregrasps = []
    for iteration in range(cfg.pregrasp.step):
        margin = cfg.pregrasp.cdist * min((iteration + 1) / cfg.pregrasp.step * 2, 1)
        env.set_obj_margin(margin)
        env.keep_hand_stable()
        env.step_sim(cfg.grasp.substep)
        pregrasps.append(env.get_hand_qpos().copy())
        if iteration % 2 == 0 and margin >= cfg.pregrasp.cdist and pregrasp_filter.early_stop():
            break
    ok, _ = pregrasp_filter.forward(None)
    status["pregrasp_filter"] = dict(filter_diagnostics(env, cfg.pregrasp.filter, None), success=bool(ok))
    status.update(success=bool(ok), failure_stage=None if ok else "pregrasp_filter")
    data["pregrasp_qpos"] = np.stack(pregrasps)[::-2].astype(np.float32)
    trace["pregrasp_qpos"] = pregrasps
    return data, status, trace
