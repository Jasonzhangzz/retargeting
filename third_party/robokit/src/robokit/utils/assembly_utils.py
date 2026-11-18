"""Compose robot descriptions at runtime."""

import copy
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import trimesh
import yaml
import yourdfpy

from robokit.utils.ee_coord_utils import AXIS_TO_VEC, MountCoordinateSpec, ee_coord_conversion
from robokit.xform.numpy import matrix_to_quaternion


def compute_mount_transform(
    base_mount_coord_spec: MountCoordinateSpec,
    ee_mount_coord_spec: MountCoordinateSpec,
    *,
    auto_mount_point: bool = False,
    base_mount_mesh: Optional[trimesh.Trimesh] = None,
    ee_mount_mesh: Optional[trimesh.Trimesh] = None,
) -> np.ndarray:
    """Return the EE-mount-to-base-mount transform from link-local geometry.

    Mount-point priority:
    1. Use explicit points, which are required for lateral offsets.
    2. With `auto_mount_point`, project mesh vertices on outward axes; use the base maximum and EE minimum.
    3. Otherwise, or for missing/empty meshes, use link origins.

    >>> from robokit.utils.ee_coord_utils import MountCoordinateSpec
    >>> base = MountCoordinateSpec("z", "x", mount_point_xyz=(0, 0, 0.1))
    >>> ee = MountCoordinateSpec("z", "x", mount_point_xyz=(0, 0, -0.2))
    >>> np.allclose(compute_mount_transform(base, ee)[:3, 3], [0, 0, 0.3])
    True
    """
    T_mount_ee = np.eye(4)
    T_mount_ee[:3, :3] = ee_coord_conversion(ee_mount_coord_spec, base_mount_coord_spec)
    points = []
    for name, spec, mesh in (
        ("base", base_mount_coord_spec, base_mount_mesh),
        ("EE", ee_mount_coord_spec, ee_mount_mesh),
    ):
        point = np.zeros(3)
        if spec.mount_point_xyz is not None:
            point = np.asarray(spec.mount_point_xyz, dtype=float)
        elif auto_mount_point and mesh is not None and len(mesh.vertices):
            outward = np.asarray(AXIS_TO_VEC[spec.outward_axis], dtype=float)
            projected = mesh.vertices @ outward
            if not np.isfinite(projected).all():
                raise ValueError(f"Cannot infer {name} mount point: mesh contains non-finite coordinates")
            point = outward * (projected.max() if name == "base" else projected.min())
        points.append(point)
    T_mount_ee[:3, 3] = points[0] - T_mount_ee[:3, :3] @ points[1]
    return T_mount_ee


def assemble_urdf(
    output_path: Path,
    base_urdf_path: Path,
    ee_urdf_path: Path,
    base_mount_link_name: str,
    ee_mount_link_name: str,
    T_mount_ee: np.ndarray,
    ee_name_prefix: str,
) -> Path:
    """Attach one EE to a base URDF and return the generated path."""
    if output_path.exists():
        return output_path

    tree = ET.parse(base_urdf_path)
    robot = tree.getroot()
    robot.set("name", output_path.stem)
    base = yourdfpy.URDF.load(str(base_urdf_path), load_meshes=False)
    for mesh in robot.iter("mesh"):
        mesh.set("filename", str(Path(base._filename_handler(mesh.get("filename"))).resolve()))

    ee_tree = ET.parse(ee_urdf_path)
    ee_root = ee_tree.getroot()
    joints = ee_root.findall("joint")
    kept = {ee_mount_link_name}
    for _ in range(len(joints)):
        kept.update(
            joint.find("child").get("link") or "" for joint in joints if joint.find("parent").get("link") in kept
        )
    if ee_mount_link_name not in {link.get("name") for link in ee_root.findall("link")}:
        raise ValueError(f"Unknown EE mount link: {ee_mount_link_name}")

    ee = yourdfpy.URDF.load(str(ee_urdf_path), load_meshes=False)
    for element in list(ee_root):
        if element.tag == "link" and element.get("name") not in kept:
            continue
        if element.tag == "joint" and (
            element.find("parent").get("link") not in kept or element.find("child").get("link") not in kept
        ):
            continue
        if element.tag not in ("link", "joint"):
            continue
        element = copy.deepcopy(element)
        element.set("name", ee_name_prefix + (element.get("name") or ""))
        for reference in list(element.findall("parent")) + list(element.findall("child")):
            reference.set("link", ee_name_prefix + (reference.get("link") or ""))
        for mimic in element.findall("mimic"):
            mimic.set("joint", ee_name_prefix + (mimic.get("joint") or ""))
        for mesh in element.iter("mesh"):
            mesh.set("filename", str(Path(ee._filename_handler(mesh.get("filename"))).resolve()))
        robot.append(element)

    xyz = T_mount_ee[:3, 3]
    rpy = trimesh.transformations.euler_from_matrix(T_mount_ee, axes="sxyz")
    joint = ET.SubElement(robot, "joint", {"name": ee_name_prefix + "mount_joint", "type": "fixed"})
    ET.SubElement(joint, "origin", {"xyz": " ".join(map(str, xyz)), "rpy": " ".join(map(str, rpy))})
    ET.SubElement(joint, "parent", {"link": base_mount_link_name})
    ET.SubElement(joint, "child", {"link": ee_name_prefix + ee_mount_link_name})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(ET, "indent"):  # Python 3.9+
        ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def assemble_mjcf(
    output_path: Path,
    base_mjcf_path: Path,
    ee_mjcf_path: Path,
    base_mount_link_name: str,
    ee_mount_link_name: str,
    T_mount_ee: np.ndarray,
    ee_name_prefix: str,
) -> Path:
    """Attach one EE to a base MJCF and return the generated path; needs mujoco 3.3.4+ for `MjSpec` editing."""
    if output_path.exists():
        return output_path

    import mujoco  # type: ignore  # lazily imported; mujoco ships only in the `mj` extra

    output_path.parent.mkdir(parents=True, exist_ok=True)

    def resolve_assets(source: Path, spec: mujoco.MjSpec):
        for directory, assets in (
            (spec.meshdir, spec.meshes),
            (spec.texturedir, spec.textures),
        ):
            for asset in assets:
                if not asset.file:
                    continue
                source_path = Path(asset.file)
                if not asset.name:
                    asset.name = source_path.stem
                if not source_path.is_absolute():
                    source_path = source.parent / directory / source_path
                asset.file = str(source_path.resolve())
        spec.meshdir = ""
        spec.texturedir = ""

    base = mujoco.MjSpec.from_file(str(base_mjcf_path))
    resolve_assets(base_mjcf_path, base)
    ee = mujoco.MjSpec.from_file(str(ee_mjcf_path))
    resolve_assets(ee_mjcf_path, ee)
    child = ee.body(ee_mount_link_name)
    for joint in list(child.joints):
        ee.delete(joint)
    # Replace the source body's parent transform with T_mount_ee, as URDF subtree attachment does.
    child.pos = [0, 0, 0]
    child.quat = [1, 0, 0, 0]
    child.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
    frame = base.body(base_mount_link_name).add_frame(
        pos=T_mount_ee[:3, 3].tolist(), quat=matrix_to_quaternion(T_mount_ee[:3, :3]).tolist()
    )
    frame.attach_body(child, ee_name_prefix, "")

    base.compile()
    base.to_file(str(output_path))
    return output_path


def merge_collision_spheres(output_path: Path, sources: Sequence[Tuple[Path, str]]) -> Optional[Path]:
    """Merge collision-sphere YAML files while prefixing their link names."""
    if not sources:
        return None
    if output_path.exists():
        return output_path
    spheres: Dict[str, object] = {}
    for source, link_name_prefix in sources:
        with open(source) as f:
            data: Mapping[str, object] = yaml.safe_load(f)
        for link_name, values in data.get("collision_spheres", {}).items():
            spheres[link_name_prefix + link_name] = values
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump({"collision_spheres": spheres}, f, sort_keys=False)
    return output_path
