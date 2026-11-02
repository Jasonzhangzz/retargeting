#!/usr/bin/env python3
"""Prepare the official WUJI Hand 2.2 and Dexonomy static-grasp inputs.

The downloaded revision is fixed. Original finger joints, servos, meshes and
physical parameters are retained; the adapter only changes body/site names,
chooses the physical wrist as its coordinate origin, and adds six bounded
virtual wrist servos. Query points are measurements on the actual meshes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retarget.env.grasp.hand import HAND_BODIES, rigid_mesh_arrays

WUJI_COMMIT = "5c4dfa16917355d073e2e1d7e3d03897b6cf9508"
WUJI_URL = "https://github.com/wuji-technology/wuji-description"
WUJI_MODEL = Path("hand2/hand2_2/body/mjcf/right.xml")
RAW_URL = f"https://raw.githubusercontent.com/wuji-technology/wuji-description/{WUJI_COMMIT}"
ROOT_KP = (2000.0, 2000.0, 2000.0, 100.0, 100.0, 100.0)
ROOT_KV = (90.0, 90.0, 90.0, 10.0, 10.0, 10.0)
ROOT_FORCE = (200.0, 200.0, 200.0, 30.0, 30.0, 30.0)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fetch(relative, cache):
    path = cache / relative
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(f"{RAW_URL}/{relative.as_posix()}", headers={"User-Agent": "retarget-grasp-assets"})
        with urllib.request.urlopen(request, timeout=120) as response:
            contents = response.read()
        temporary = path.with_suffix(path.suffix + ".pending")
        temporary.write_bytes(contents)
        temporary.replace(path)
    return path


def download_wuji(cache):
    xml = _fetch(WUJI_MODEL, cache)
    root = ET.parse(xml).getroot()
    if root.get("model") != "wujihand2_2-right":
        raise ValueError("The source must be the official WUJI Hand 2.2 right hand")
    mesh_directory = WUJI_MODEL.parent / root.find("compiler").get("meshdir")
    files = [Path("LICENSE"), Path("README.md")]
    files += [mesh_directory / mesh.get("file") for mesh in root.findall("asset/mesh")]
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda path: _fetch(path, cache), files))
    return xml


def _numbers(value):
    return " ".join(f"{float(v):.17g}" for v in value)


def adapt_wuji(source_xml, destination):
    tree = ET.parse(source_xml)
    root = tree.getroot()
    if root.get("model") != "wujihand2_2-right":
        raise ValueError("Only official Hand 2.2 is accepted; beta assets are incompatible")
    compiler = root.find("compiler")
    source_meshes = source_xml.parent / compiler.get("meshdir", ".")
    compiler.attrib.pop("meshdir", None)
    compiler.set("autolimits", "true")
    # ComFree only supports Euler/implicitfast. This is explicitly recorded as
    # a runtime integrator adaptation, not a change to finger dynamics assets.
    root.find("option").set("integrator", "implicitfast")
    root.find("option").set("cone", "pyramidal")
    meshes = destination / "meshes"
    meshes.mkdir(parents=True, exist_ok=True)
    for mesh in root.findall("asset/mesh"):
        source = source_meshes / mesh.get("file")
        target = meshes / source.name
        shutil.copy2(source, target)
        mesh.set("file", f"meshes/{target.name}")
    world = root.find("worldbody")
    mount = world.find("body[@name='r_mount']")
    wrist = mount.find("body[@name='r_wrist']")
    position = np.fromstring(wrist.get("pos", "0 0 0"), sep=" ")
    quaternion = np.fromstring(wrist.get("quat", "1 0 0 0"), sep=" ")
    rotation = Rotation.from_quat(quaternion, scalar_first=True)
    mount.remove(wrist)
    world.remove(mount)
    wrist.attrib.pop("pos", None)
    wrist.attrib.pop("quat", None)
    mount.set("pos", _numbers(-rotation.inv().apply(position)))
    mount.set("quat", _numbers(rotation.inv().as_quat(scalar_first=True)))
    wrist.append(mount)
    names = {"r_wrist": "right_palm_link"}
    for finger, source in enumerate(("thumb", "index_finger", "middle_finger", "ring_finger", "pinky"), 1):
        for link, suffix in enumerate(("proximal", "proximal_abd", "middle", "distal"), 1):
            names[f"r_{source}_{suffix}"] = f"right_finger{finger}_link{link}"
        names[f"r_{source}_tip"] = f"right_{('thumb', 'index', 'middle', 'ring', 'pinky')[finger-1]}_tip"
    # Mesh names/files are unchanged. Only references to bodies/sites use the
    # semantic aliases required by the existing grasp kinematics.
    for element in [*root.iter(), *wrist.iter()]:
        for attribute in ("body1", "body2", "site"):
            if element.get(attribute) in names:
                element.set(attribute, names[element.get(attribute)])
        if element.tag in ("body", "site") and element.get("name") in names:
            element.set("name", names[element.get("name")])
    virtual = []
    parent = world
    actuators = root.find("actuator")
    for i, axis_name in enumerate(("tx", "ty", "tz", "rx", "ry", "rz")):
        body = ET.SubElement(parent, "body", name=f"grasp_wrist_{axis_name}", gravcomp="1")
        ET.SubElement(body, "inertial", pos="0 0 0", mass="0.00001", diaginertia="0.00000001 0.00000001 0.00000001")
        bound = 2.0 if i < 3 else 6.283185307179586
        name = f"grasp_wrist_{axis_name}_joint"
        axis = np.eye(3)[i % 3]
        ET.SubElement(body, "joint", name=name, type="slide" if i < 3 else "hinge", axis=_numbers(axis), range=f"{-bound} {bound}", damping="0", armature="0.0002")
        servo = ET.Element("position", name=f"grasp_wrist_{axis_name}_servo", joint=name, kp=str(ROOT_KP[i]), kv=str(ROOT_KV[i]), ctrlrange=f"{-bound} {bound}", forcerange=f"{-ROOT_FORCE[i]} {ROOT_FORCE[i]}")
        virtual.append(servo)
        parent = body
    parent.append(wrist)
    for i, servo in enumerate(virtual):
        actuators.insert(i, servo)
    custom = root.find("custom")
    if custom is None:
        custom = ET.SubElement(root, "custom")
    ET.SubElement(custom, "numeric", name="grasp_hand_generation", data="2")
    ET.indent(tree, space="  ")
    xml = destination / "right.xml"
    tree.write(xml, encoding="unicode")
    return xml, dict(
        source_to_adapter_body_names=names,
        root_frame="official r_wrist, with all physical relative transforms retained",
        official_mount_pose_in_wrist=dict(position=(-rotation.inv().apply(position)).tolist(), quaternion_wxyz=rotation.inv().as_quat(scalar_first=True).tolist()),
        added_wrist_servo=dict(order="xyz + intrinsic XYZ", kp=ROOT_KP, kv=ROOT_KV, force_limits=ROOT_FORCE, translation_range_m=2.0, rotation_range_rad=2*np.pi, virtual_body_mass_kg=1e-5, virtual_body_inertia_kg_m2=1e-8, virtual_body_gravcomp=1),
        integrator="implicitfast (official source: RK4); required by ComFree runtime",
        original_finger_joints_geometries_mesh_scales_and_servos_preserved=True,
    )


def sample_surface(vertices, faces, count, seed, *, return_face_ids=False):
    """Stratified area-weighted samples with outward triangle normals."""
    triangles = np.asarray(vertices)[np.asarray(faces)]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1)
    valid = areas > 1e-15
    triangles, cross, areas = triangles[valid], cross[valid], areas[valid]
    if not len(areas):
        raise ValueError("No nondegenerate triangles in hand mesh")
    rng = np.random.default_rng(seed)
    cumulative = np.cumsum(areas)
    selected = np.searchsorted(cumulative, (np.arange(count) + rng.random(count)) * cumulative[-1] / count)
    uv = rng.random((count, 2))
    root = np.sqrt(uv[:, :1])
    barycentric = np.c_[1-root[:, 0], root[:, 0]*(1-uv[:, 1]), root[:, 0]*uv[:, 1]]
    points = (triangles[selected] * barycentric[:, :, None]).sum(1)
    normals = cross[selected] / areas[selected, None]
    result = points.astype(np.float32), normals.astype(np.float32)
    if return_face_ids:
        return (*result, np.flatnonzero(valid)[selected])
    return result


def rigid_mesh_provenance(model, body):
    """Owner metadata for the unchanged rigid_mesh_arrays triangle ordering.

    Match its rigid-child traversal and visual/collision duplicate removal so
    metadata can be added without altering any existing point or normal.
    """
    attached = {int(body)}
    for child in range(int(body)+1, model.nbody):
        if int(model.body_parentid[child]) in attached and model.body_jntnum[child] == 0:
            attached.add(child)
    seen = set()
    owners, meshes, geoms, faces = [], [], [], []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] not in attached or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = int(model.geom_dataid[geom])
        key = (int(model.geom_bodyid[geom]), mesh, *model.geom_pos[geom], *model.geom_quat[geom])
        if key in seen:
            continue
        seen.add(key)
        count = int(model.mesh_facenum[mesh])
        owners.extend([model.body(int(model.geom_bodyid[geom])).name] * count)
        meshes.extend([model.mesh(mesh).name] * count)
        geoms.extend([geom] * count)
        faces.extend(range(count))
    return dict(source_body=np.asarray(owners), source_mesh=np.asarray(meshes), source_geom=np.asarray(geoms,np.int32), source_face=np.asarray(faces,np.int32))


def generate_queries(xml, destination, samples):
    model = mujoco.MjModel.from_xml_path(str(xml))
    output = destination / "qp_right"
    output.mkdir(exist_ok=True)
    records = {}
    for i, name in enumerate(HAND_BODIES):
        vertices, faces = rigid_mesh_arrays(model, model.body(name).id)
        count = max(1024, samples) if name == "right_palm_link" else samples
        points, normals, selected = sample_surface(vertices, faces, count, 7300+i, return_face_ids=True)
        metadata = rigid_mesh_provenance(model, model.body(name).id)
        if any(len(value) != len(faces) for value in metadata.values()):
            raise AssertionError("Rigid mesh metadata must match the unmodified triangle ordering")
        metadata = {key: values[selected] for key, values in metadata.items()}
        path = output / f"{name}.npz"
        np.savez_compressed(path, local_pos=points, local_normal=normals, radius=np.zeros(count, np.float32), **metadata)
        records[name] = dict(samples=count, seed=7300+i, sha256=sha256(path))
    return model, records


def copy_templates(checkout, output):
    source = checkout / "assets/hand/shadow"
    target = output / "templates/assets/hand/shadow"
    target.mkdir(parents=True, exist_ok=True)
    files = [source / "right.xml", source / "keypoint.yaml", *sorted((source / "raw_anno").glob("*.yaml"))]
    for path in files:
        destination = target / path.relative_to(source)
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(path, destination)
    # The retained initializer reads the original matching thresholds here.
    # Its other modules are local and need no external Python/config tree.
    init_config = checkout / "dexonomy/config/op/init.yaml"
    config_target = output / "templates/dexonomy/config/op/init.yaml"
    config_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(init_config, config_target)
    files.append(init_config)
    notice = ROOT / "licenses/Dexonomy-NOTICE.txt"
    shutil.copy2(notice, output / "templates/Dexonomy-NOTICE.txt")
    try:
        commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return dict(source="https://github.com/JYChen18/Dexonomy", local_source=str(checkout), commit=commit, license="CC BY-NC 4.0; see templates/Dexonomy-NOTICE.txt", files={str(path.relative_to(checkout)):sha256(path) for path in files})


def model_config():
    """Explicit standalone baseline; tracking weights are separately configured."""
    return dict(
        sequence=dict(hand="wuji_right", scene="sdf", start=0, hold_frame=0, hold_seconds=0.01),
        simulator=dict(exec_backend="comfree", mpc_dt=0.01, action_dt=0.03, knot_dt=0.06, exec_dt=0.002, ref_dt=0.03333333333333333),
        contact_physics=dict(stiffness=0.2, damping=0.001),
        prediction=dict(horizon=4, warm_shift_scale=1.0),
        loss=dict(obj_pos=1.0, obj_rot=0.3, terminal=1.0, reference_pose_scale=1.0, init=0.0, action_ref_base=0.0, action_ref_base_rot=0.0, action_ref_finger=0.0, contact_robustness=1.0, marked_query_pos=1.0, grasp_closure=0.0, grasp_wrench=0.0, soft_penetration=1.0, penetration=1.0, finger_joint_collision=1.0, obj_vel=0.1, obj_ang_vel=0.03),
        smoothing=dict(enabled=False, control_velocity_weight=0.0, control_acceleration_weight=0.0, hand_state_weight=0.0, control_velocity_base_position_scale=0.01, control_velocity_base_rotation_scale=0.1, control_velocity_finger_scale=0.2, control_acceleration_base_position_scale=0.01, control_acceleration_base_rotation_scale=0.1, control_acceleration_finger_scale=0.2),
        bounds=dict(init_base_pos=0.02, init_base_rot=0.2, init_finger=0.3),
        contact=dict(target_phi=-0.0005, penetration_limit=0.002, marked_query_dist_mm=2.0, marked_query_temp_mm=1.0, age_ramp_frames=0, finger_joint_surface_margin_mm=0.5),
        video=dict(fps=30.0),
    )


def validate_model(model, source_xml):
    """Check 26-DOF ordering and preserve original compiled finger mechanics."""
    source = mujoco.MjModel.from_xml_path(str(source_xml))
    if (model.nq, model.nv, model.nu) != (26,26,26):
        raise AssertionError("Expected 26-DOF Wuji including the six wrist servos")
    np.testing.assert_allclose(model.jnt_range[6:], source.jnt_range)
    np.testing.assert_allclose(model.jnt_axis[6:], source.jnt_axis)
    np.testing.assert_allclose(model.actuator_gainprm[6:], source.actuator_gainprm)
    np.testing.assert_allclose(model.actuator_biasprm[6:], source.actuator_biasprm)
    np.testing.assert_allclose(model.actuator_forcerange[6:], source.actuator_forcerange)
    np.testing.assert_allclose(model.mesh_scale, source.mesh_scale)
    # Re-rooting is a rigid coordinate change: check every source physical
    # body under arbitrary finger poses against the source wrist frame.
    data, original = mujoco.MjData(model), mujoco.MjData(source)
    for scale in (0.0, 0.2):
        q = np.clip(np.full(20,scale), source.jnt_range[:,0], source.jnt_range[:,1])
        data.qpos[6:] = original.qpos[:] = q
        mujoco.mj_kinematics(model,data)
        mujoco.mj_kinematics(source,original)
        wrist = source.body("r_wrist").id
        rotation = original.xmat[wrist].reshape(3,3)
        for finger, prefix in enumerate(("thumb", "index_finger", "middle_finger", "ring_finger", "pinky"),1):
            for link, suffix in enumerate(("proximal", "proximal_abd", "middle", "distal"),1):
                old=source.body(f"r_{prefix}_{suffix}").id
                new=model.body(f"right_finger{finger}_link{link}").id
                np.testing.assert_allclose(data.xpos[new],rotation.T@(original.xpos[old]-original.xpos[wrist]),atol=2e-9)
                np.testing.assert_allclose(data.xmat[new].reshape(3,3),rotation.T@original.xmat[old].reshape(3,3),atol=2e-9)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dexonomy-root", type=Path, required=True, help="Existing Dexonomy checkout containing assets/hand/shadow")
    parser.add_argument("--output", type=Path, default=ROOT/"data/grasp")
    parser.add_argument("--cache", type=Path, default=Path.home()/".cache/retarget-grasp"/WUJI_COMMIT)
    parser.add_argument("--samples", type=int, default=768, help="Surface points per articulated link; palm uses at least 1024")
    args=parser.parse_args(argv)
    if args.samples < 64:
        parser.error("--samples must be at least 64")
    source_xml=download_wuji(args.cache)
    args.output.mkdir(parents=True,exist_ok=True)
    destination=args.output/"assets/wuji2"
    destination.mkdir(parents=True,exist_ok=True)
    original=destination/"upstream"
    (original/"mjcf").mkdir(parents=True,exist_ok=True)
    # Keep a separately loadable, byte-identical official source model.
    shutil.copy2(source_xml,original/"mjcf/right.xml")
    source_tree=ET.parse(source_xml).getroot()
    source_meshes=source_xml.parent/source_tree.find("compiler").get("meshdir")
    original_meshes=original/"meshes/right"
    original_meshes.mkdir(parents=True,exist_ok=True)
    for mesh in source_tree.findall("asset/mesh"):
        shutil.copy2(source_meshes/mesh.get("file"),original_meshes/mesh.get("file"))
    shutil.copy2(args.cache/"LICENSE",destination/"LICENSE")
    xml, adapter=adapt_wuji(source_xml,destination)
    model,queries=generate_queries(xml,destination,args.samples)
    validate_model(model,source_xml)
    templates=copy_templates(args.dexonomy_root.resolve(),args.output)
    from retarget.task.grasp.dexonomy import load_template
    count=0
    for path in sorted((args.output/"templates/assets/hand/shadow/raw_anno").glob("*.yaml")):
        load_template(args.output/"templates",path.stem)
        count+=1
    config=args.output/"model.yaml"
    config.write_text(yaml.safe_dump(model_config(),sort_keys=False))
    provenance=dict(hand="official WUJI Hand 2.2 right (bare)",source=WUJI_URL,commit=WUJI_COMMIT,source_xml=str(WUJI_MODEL),source_xml_sha256=sha256(source_xml),license="MIT; see assets/wuji2/LICENSE",adapter=adapter,source_files={str(p.relative_to(args.cache)):sha256(p) for p in sorted(args.cache.rglob("*")) if p.is_file()},queries=dict(method="stratified triangle-area sampling; actual compiled rigid-attached meshes; radius=0",bodies=queries),templates=templates,model_yaml=dict(origin="standalone explicit baseline generated by setup_grasp_assets.py",sha256=sha256(config)),validation=dict(cpu_model=True,original_finger_parameters=True,source_wrist_frame_FK=True,loaded_templates=count))
    (destination/"source.json").write_text(json.dumps(provenance,indent=2)+"\n")
    (args.output/"source.json").write_text(json.dumps(dict(hand_provenance="assets/wuji2/source.json",templates=count,hand_model="wuji2",wrist_pose_prior=False,physics_reference=None),indent=2)+"\n")
    print(json.dumps(dict(output=str(args.output),hand="WUJI Hand 2.2",commit=WUJI_COMMIT,nq=model.nq,nu=model.nu,templates=count,query_points=sum(row["samples"] for row in queries.values()),cpu_validation="passed"),indent=2))


if __name__ == "__main__":
    main()
