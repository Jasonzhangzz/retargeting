"""Browse RoboKit robot models, collision spheres, and semantic coordinates.

Usage: ``uv run -m robokit.gallery.robot_models``
"""

import argparse
import hashlib
import time
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import trimesh
import viser
import yourdfpy
from viser.extras import ViserUrdf

from robokit.robo import Robot
from robokit.utils.assembly_utils import assemble_mjcf, assemble_urdf, compute_mount_transform, merge_collision_spheres
from robokit.utils.ee_coord_utils import (
    AXIS_TO_VEC,
    EECoordinateSpec,
    HandCoordinateSpec,
    MountCoordinateSpec,
)
from robokit.utils.visualize_utils import ViserMjcf
from robokit.xform.numpy import matrix_to_quaternion, quaternion_to_matrix


RobotEntry = Tuple[str, str, Optional[str]]
MODEL_FORMATS = ("URDF", "MJCF") if find_spec("mujoco") is not None else ("URDF",)
ROBOTS: Dict[str, Dict[str, RobotEntry]] = {
    "ee": {
        "Ability Hand Right": (
            "robokit.assets.robots.end_effectors.ability_hand",
            "URDF_PATH_RIGHT",
            "COLLISION_SPHERE_PATH_RIGHT",
        ),
        "Ability Hand Left": (
            "robokit.assets.robots.end_effectors.ability_hand",
            "URDF_PATH_LEFT",
            None,
        ),
        "Allegro Hand Right": (
            "robokit.assets.robots.end_effectors.allegro_hand",
            "URDF_PATH_RIGHT",
            "COLLISION_SPHERE_PATH_RIGHT",
        ),
        "Allegro Hand Left": (
            "robokit.assets.robots.end_effectors.allegro_hand",
            "URDF_PATH_LEFT",
            None,
        ),
        "Barrett Hand": (
            "robokit.assets.robots.end_effectors.barrett_hand",
            "URDF_PATH",
            "COLLISION_SPHERE_PATH",
        ),
        "DClaw Gripper": (
            "robokit.assets.robots.end_effectors.dclaw_gripper",
            "URDF_PATH",
            "COLLISION_SPHERE_PATH",
        ),
        "Inspire Hand Left": (
            "robokit.assets.robots.end_effectors.inspire_hand",
            "URDF_PATH_LEFT",
            "COLLISION_SPHERE_PATH_LEFT",
        ),
        "Inspire Hand Right": (
            "robokit.assets.robots.end_effectors.inspire_hand",
            "URDF_PATH_RIGHT",
            "COLLISION_SPHERE_PATH_RIGHT",
        ),
        "LEAP Hand Right": (
            "robokit.assets.robots.end_effectors.leap_hand",
            "URDF_PATH_RIGHT",
            "COLLISION_SPHERE_PATH_RIGHT",
        ),
        "LEAP Hand Left": (
            "robokit.assets.robots.end_effectors.leap_hand",
            "URDF_PATH_LEFT",
            None,
        ),
        "MANO Hand Right": (
            "robokit.assets.robots.end_effectors.mano_hand",
            "URDF_PATH_RIGHT",
            "COLLISION_SPHERE_PATH_RIGHT",
        ),
        "Orca Hand Right": (
            "robokit.assets.robots.end_effectors.orcahand",
            "URDF_PATH_RIGHT",
            "COLLISION_SPHERE_PATH_RIGHT",
        ),
        "Paxini Hand Right": (
            "robokit.assets.robots.end_effectors.paxini_hand",
            "URDF_PATH_RIGHT",
            "COLLISION_SPHERE_PATH_RIGHT",
        ),
        "Paxini Hand Left": (
            "robokit.assets.robots.end_effectors.paxini_hand",
            "URDF_PATH_LEFT",
            None,
        ),
        "Robotiq 2F-85": (
            "robokit.assets.robots.end_effectors.robotiq_2f85",
            "URDF_PATH",
            "COLLISION_SPHERE_PATH",
        ),
        "Schunk SVH Right": (
            "robokit.assets.robots.end_effectors.schunk_hand",
            "URDF_PATH_RIGHT",
            "COLLISION_SPHERE_PATH_RIGHT",
        ),
        "Schunk SVH Left": (
            "robokit.assets.robots.end_effectors.schunk_hand",
            "URDF_PATH_LEFT",
            None,
        ),
        "Shadow Hand Right": (
            "robokit.assets.robots.end_effectors.shadow_hand",
            "URDF_PATH_RIGHT",
            "COLLISION_SPHERE_PATH_RIGHT",
        ),
        "Shadow Hand Left": (
            "robokit.assets.robots.end_effectors.shadow_hand",
            "URDF_PATH_LEFT",
            None,
        ),
        "UMI Gripper": (
            "robokit.assets.robots.end_effectors.umi_gripper",
            "MJCF_PATH",
            None,
        ),
        "Sharpa Hand Left": (
            "robokit.assets.robots.end_effectors.sharpa_hand",
            "URDF_PATH_LEFT",
            "COLLISION_SPHERE_PATH_LEFT",
        ),
        "Sharpa Hand Right": (
            "robokit.assets.robots.end_effectors.sharpa_hand",
            "URDF_PATH_RIGHT",
            "COLLISION_SPHERE_PATH_RIGHT",
        ),
        "XHand Right": (
            "robokit.assets.robots.end_effectors.xhand",
            "URDF_PATH_RIGHT",
            "COLLISION_SPHERE_PATH_RIGHT",
        ),
        "XHand Left": (
            "robokit.assets.robots.end_effectors.xhand",
            "URDF_PATH_LEFT",
            None,
        ),
        "XHand Right (MJCF)": (
            "robokit.assets.robots.end_effectors.xhand",
            "MJCF_PATH_RIGHT",
            "COLLISION_SPHERE_PATH_RIGHT",
        ),
        "Wuji Hand Right": (
            "robokit.assets.robots.end_effectors.wuji_hand",
            "URDF_PATH_RIGHT",
            "COLLISION_SPHERE_PATH_RIGHT",
        ),
        "Wuji Hand Left": (
            "robokit.assets.robots.end_effectors.wuji_hand",
            "URDF_PATH_LEFT",
            "COLLISION_SPHERE_PATH_LEFT",
        ),
        "Wuji Hand 2.2 Right": (
            "robokit.assets.robots.end_effectors.wuji_hand_2_2",
            "URDF_PATH_RIGHT",
            "COLLISION_SPHERE_PATH_RIGHT",
        ),
        "Wuji Hand 2.2 Left": (
            "robokit.assets.robots.end_effectors.wuji_hand_2_2",
            "URDF_PATH_LEFT",
            "COLLISION_SPHERE_PATH_LEFT",
        ),
        "Wuji Hand 2.2 Right (MJCF)": (
            "robokit.assets.robots.end_effectors.wuji_hand_2_2",
            "MJCF_PATH_RIGHT",
            "COLLISION_SPHERE_PATH_RIGHT",
        ),
        "YAM Gripper": (
            "robokit.assets.robots.end_effectors.yam_gripper",
            "URDF_PATH",
            "COLLISION_SPHERE_PATH",
        ),
        "YAM Gripper (MJCF)": (
            "robokit.assets.robots.end_effectors.yam_gripper",
            "MJCF_PATH",
            "COLLISION_SPHERE_PATH",
        ),
    },
    "arm": {
        "YAM": ("robokit.assets.robots.arms.yam", "URDF_PATH", "COLLISION_SPHERE_PATH"),
        "KUKA iiwa 14": ("robokit.assets.robots.arms.iiwa14", "URDF_PATH", "COLLISION_SPHERE_PATH"),
        "Kinova Gen3": ("robokit.assets.robots.arms.gen3", "URDF_PATH", "COLLISION_SPHERE_PATH"),
        "Unitree Z1": ("robokit.assets.robots.arms.z1", "URDF_PATH", "COLLISION_SPHERE_PATH"),
        "Universal Robots UR5e": ("robokit.assets.robots.arms.ur5e", "URDF_PATH", "COLLISION_SPHERE_PATH"),
        "Franka FR3": ("robokit.assets.robots.arms.fr3", "URDF_PATH", "COLLISION_SPHERE_PATH"),
        "AgileX PiPER": ("robokit.assets.robots.arms.piper", "URDF_PATH", "COLLISION_SPHERE_PATH"),
        "Flexiv Rizon4": ("robokit.assets.robots.arms.rizon4", "URDF_PATH", "COLLISION_SPHERE_PATH"),
        "Franka Panda": (
            "robokit.assets.robots.arms.franka_panda",
            "ARM_URDF_PATH",
            "ARM_COLLISION_SPHERE_PATH",
        ),
        "Franka Panda (MJCF)": (
            "robokit.assets.robots.arms.franka_panda",
            "MJCF_PATH",
            None,
        ),
        "Universal Robots UR10": ("robokit.assets.robots.arms.ur10", "URDF_PATH", "COLLISION_SPHERE_PATH"),
        "xArm6": ("robokit.assets.robots.arms.xarm6", "URDF_PATH", "COLLISION_SPHERE_PATH"),
        "xArm7": ("robokit.assets.robots.arms.xarm7", "URDF_PATH", "COLLISION_SPHERE_PATH"),
        "xArm7 (MJCF)": ("robokit.assets.robots.arms.xarm7", "MJCF_PATH", "COLLISION_SPHERE_PATH"),
        "ABB YuMi": ("robokit.assets.robots.arms.yumi", "URDF_PATH", "COLLISION_SPHERE_PATH"),
    },
    "humanoid": {
        "DexMate Vega-1": (
            "robokit.assets.robots.humanoids.vega_1",
            "URDF_PATH",
            "COLLISION_SPHERE_PATH",
        ),
        "Berkeley Humanoid Lite": (
            "robokit.assets.robots.humanoids.berkeley_humanoid_lite",
            "URDF_PATH",
            "COLLISION_SPHERE_FULL_BODY_PATH",
        ),
        "Fourier GR3": (
            "robokit.assets.robots.humanoids.fourier_gr3",
            "URDF_PATH",
            "COLLISION_SPHERE_FULL_BODY_PATH",
        ),
        "Unitree G1": ("robokit.assets.robots.humanoids.unitree_g1", "URDF_PATH", "COLLISION_SPHERE_PATH"),
        "Unitree H1-2": (
            "robokit.assets.robots.humanoids.unitree_h1_2",
            "URDF_PATH",
            "COLLISION_SPHERE_FULL_BODY_PATH",
        ),
        "Booster K1": ("robokit.assets.robots.humanoids.booster_k1", "URDF_PATH", "COLLISION_SPHERE_FULL_BODY_PATH"),
        "Booster T1": ("robokit.assets.robots.humanoids.booster_t1", "URDF_PATH", "COLLISION_SPHERE_FULL_BODY_PATH"),
        "EngineAI PM01": (
            "robokit.assets.robots.humanoids.engineai_pm01",
            "URDF_PATH",
            "COLLISION_SPHERE_FULL_BODY_PATH",
        ),
        "PAL Talos": ("robokit.assets.robots.humanoids.pal_talos", "URDF_PATH", "COLLISION_SPHERE_FULL_BODY_PATH"),
        "PND Adam Lite": (
            "robokit.assets.robots.humanoids.pnd_adam_lite",
            "URDF_PATH",
            "COLLISION_SPHERE_FULL_BODY_PATH",
        ),
        "Tienkung Lite": (
            "robokit.assets.robots.humanoids.tienkung_lite",
            "URDF_PATH",
            "COLLISION_SPHERE_FULL_BODY_PATH",
        ),
        "ToddlerBot 2xc": (
            "robokit.assets.robots.humanoids.toddlerbot_2xc",
            "URDF_PATH",
            "COLLISION_SPHERE_FULL_BODY_PATH",
        ),
        "Unitree H1": ("robokit.assets.robots.humanoids.unitree_h1", "URDF_PATH", "COLLISION_SPHERE_FULL_BODY_PATH"),
    },
    "assembly": {
        "DexMate Vega-1 + Sharpa": (
            "robokit.assets.robots.assembly.vega_sharpa",
            "URDF_PATH",
            "COLLISION_SPHERE_PATH",
        ),
        "YAM + Gripper": (
            "robokit.assets.robots.assembly.yam_gripper",
            "URDF_PATH",
            "COLLISION_SPHERE_PATH",
        ),
        "Franka Panda + Gripper": (
            "robokit.assets.robots.arms.franka_panda",
            "URDF_PATH",
            "COLLISION_SPHERE_PATH",
        ),
        "xArm6 + Sharpa": ("robokit.assets.robots.assembly.xarm6_sharpa", "URDF_PATH", "COLLISION_SPHERE_PATH"),
        "xArm7 + Ability": ("robokit.assets.robots.assembly.xarm7_ability", "URDF_PATH", "COLLISION_SPHERE_PATH"),
        "xArm7 + MANO": ("robokit.assets.robots.assembly.xarm7_mano", "URDF_PATH", "COLLISION_SPHERE_PATH"),
        "xArm7 + Sharpa": ("robokit.assets.robots.assembly.xarm7_sharpa", "URDF_PATH", "COLLISION_SPHERE_PATH"),
        "xArm7 + XHand": ("robokit.assets.robots.assembly.xarm7_xhand", "URDF_PATH", "COLLISION_SPHERE_PATH"),
        "xArm7 + XHand (MJCF)": (
            "robokit.assets.robots.assembly.xarm7_xhand",
            "MJCF_PATH",
            "COLLISION_SPHERE_PATH",
        ),
        "G1 + Inspire": ("robokit.assets.robots.assembly.g1_inspire", "URDF_PATH", "COLLISION_SPHERE_PATH"),
        "G1 + MANO": ("robokit.assets.robots.assembly.g1_mano", "URDF_PATH", "COLLISION_SPHERE_PATH"),
    },
}
# Bare models available for online assembly; ordinary browsing keeps models with stock EEs.
ONLINE_BASES: Dict[str, RobotEntry] = {
    **{
        name: ROBOTS["arm"][name]
        for name in (
            "YAM",
            "KUKA iiwa 14",
            "Kinova Gen3",
            "Unitree Z1",
            "Universal Robots UR5e",
            "Franka FR3",
            "AgileX PiPER",
            "Flexiv Rizon4",
            "Franka Panda",
            "Franka Panda (MJCF)",
            "Universal Robots UR10",
            "xArm6",
            "xArm7",
            "xArm7 (MJCF)",
        )
    },
    "DexMate Vega-1": ROBOTS["humanoid"]["DexMate Vega-1"],
    "Unitree G1": (
        "robokit.assets.robots.humanoids.unitree_g1",
        "URDF_WITHOUT_HAND_PATH",
        "COLLISION_SPHERE_PATH",
    ),
    "Unitree H1-2": ROBOTS["humanoid"]["Unitree H1-2"],
}
ROBOTS = {
    category: {
        name: entry for name, entry in entries.items() if "MJCF" in MODEL_FORMATS or not entry[1].startswith("MJCF_")
    }
    for category, entries in ROBOTS.items()
}


Coordinate = Tuple[str, MountCoordinateSpec, Optional[EECoordinateSpec]]
Description = Union[Path, yourdfpy.URDF]


class EntryData(NamedTuple):
    """Asset constants an entry names; `mounts` holds the single mount, or the right then the left one."""

    path: Path
    spheres: Optional[Path]
    ignore: Optional[Path]
    rest_q: Dict[str, float]
    mounts: Tuple[Coordinate, ...]


def _entry_format(entry: RobotEntry) -> str:
    return "MJCF" if entry[1].startswith("MJCF_") else "URDF"


def entry_data(entry: RobotEntry) -> EntryData:
    """Read an entry's asset module; the model attribute's prefix and side name its other constants."""
    module_name, model_attr, spheres_attr = entry
    module = import_module(module_name)
    prefix = "MJCF_" if model_attr.startswith("MJCF_") else "ARM_" if model_attr.startswith("ARM_") else ""
    suffix = "_RIGHT" if model_attr.endswith("_RIGHT") else "_LEFT" if model_attr.endswith("_LEFT") else ""
    sides = (suffix,) if suffix else ("", "_RIGHT", "_LEFT")
    names = ("MOUNT_LINK_NAME", "MOUNT_COORD_SPEC", "EE_COORD_SPEC")
    mounts: List[Any] = [tuple(getattr(module, f"{prefix}{name}{side}", None) for name in names) for side in sides]
    return EntryData(
        getattr(module, model_attr),
        getattr(module, spheres_attr) if spheres_attr is not None else None,
        getattr(module, f"{prefix}SPHERE_SELF_COLLISION_IGNORE_PATH{suffix}", None),
        getattr(module, f"{prefix}REST_Q{suffix}"),
        tuple(mount for mount in mounts if mount[0] is not None),
    )


def _online_bases(model_format: str) -> Dict[str, RobotEntry]:
    return {
        name: entry
        for name, entry in ONLINE_BASES.items()
        if model_format in MODEL_FORMATS and _entry_format(entry) == model_format
    }


def _online_ees(model_format: str, side: str) -> Dict[str, RobotEntry]:
    entries = {}
    for name, entry in ROBOTS["ee"].items():
        model_attr = entry[1]
        entry_side = "right" if model_attr.endswith("_RIGHT") else "left" if model_attr.endswith("_LEFT") else ""
        if _entry_format(entry) == model_format and name != "UMI Gripper" and (not entry_side or entry_side == side):
            entries[name] = entry
    return entries


def _load_robot(
    category: str, name: str
) -> Tuple[Description, Optional[Robot], Tuple[Coordinate, ...], Dict[str, float]]:
    entry = ROBOTS[category][name]
    data = entry_data(entry)
    is_mjcf = _entry_format(entry) == "MJCF"
    description = data.path if is_mjcf else yourdfpy.URDF.load(str(data.path), mesh_dir=data.path.parent)
    coordinates = () if category == "assembly" else data.mounts
    robot = (
        Robot.load(
            description,
            load_collision_spheres=data.spheres is not None,
            collision_spheres_path=data.spheres,
        )
        if not is_mjcf or data.spheres is not None or coordinates
        else None
    )
    return description, robot, coordinates, data.rest_q


def _load_online(
    assembly_dir: Path,
    model_format: str,
    base_name: str,
    right_ee_name: str,
    left_ee_name: str,
    auto_mount_point: bool = False,
) -> Tuple[Description, Optional[Robot], Tuple[Coordinate, ...], Dict[str, float]]:
    base = entry_data(_online_bases(model_format)[base_name])
    bilateral = len(base.mounts) > 1
    chosen = zip(("_RIGHT", "_LEFT") if bilateral else ("",), (right_ee_name, left_ee_name), base.mounts)

    sphere_sources = [] if base.spheres is None else [(base.spheres, "")]
    coordinates = []
    rest_q = dict(base.rest_q)
    geometry_models: Dict[Path, Robot] = {}
    digest = hashlib.sha1("|".join((model_format, base_name, str(base.path))).encode())
    assembled_path = base.path
    out_dir = None
    for base_suffix, ee_name, (base_mount_link, base_mount_spec, _) in chosen:
        if ee_name == "None":
            continue
        ee = entry_data(_online_ees(model_format, base_suffix.lstrip("_").lower() or "right")[ee_name])
        ee_mount_link, ee_mount_spec, ee_coord_spec = ee.mounts[0]
        ee_name_prefix = f"{base_suffix.lstrip('_').lower()}_ee_" if bilateral else "ee_"
        mount_meshes = []
        for spec, path, link in (
            (base_mount_spec, base.path, base_mount_link),
            (ee_mount_spec, ee.path, ee_mount_link),
        ):
            mesh = None
            if auto_mount_point and spec.mount_point_xyz is None:
                if path not in geometry_models:
                    geometry_models[path] = Robot.load(path, load_meshes=True, mesh_dir=path.parent)
                geometry_spec = geometry_models[path].spec
                scene = geometry_spec.link_visual_geometries[link]
                if not scene.geometry:
                    scene = geometry_spec.link_collision_geometries[link]
                mesh = scene.to_mesh()
            mount_meshes.append(mesh)
        T_mount_ee = compute_mount_transform(
            base_mount_spec,
            ee_mount_spec,
            auto_mount_point=auto_mount_point,
            base_mount_mesh=mount_meshes[0],
            ee_mount_mesh=mount_meshes[1],
        )
        digest.update("|".join((base_suffix, ee_name)).encode())
        digest.update(T_mount_ee.tobytes())
        out_dir = assembly_dir / digest.hexdigest()[:12]
        output_path = out_dir / ("robot.xml" if model_format == "MJCF" else "robot.urdf")
        assembled_path = (
            assemble_mjcf(
                output_path,
                assembled_path,
                ee.path,
                base_mount_link,
                ee_mount_link,
                T_mount_ee,
                ee_name_prefix,
            )
            if model_format == "MJCF"
            else assemble_urdf(
                output_path,
                assembled_path,
                ee.path,
                base_mount_link,
                ee_mount_link,
                T_mount_ee,
                ee_name_prefix,
            )
        )
        rest_q.update({ee_name_prefix + joint_name: value for joint_name, value in ee.rest_q.items()})
        if ee.spheres is not None:
            sphere_sources.append((ee.spheres, ee_name_prefix))
        coordinates.append((ee_name_prefix + ee_mount_link, ee_mount_spec, ee_coord_spec))

    if out_dir is None:
        raise ValueError("Select at least one end effector")
    spheres_path = merge_collision_spheres(out_dir / "collision_spheres.yaml", sphere_sources)
    description = (
        assembled_path
        if model_format == "MJCF"
        else yourdfpy.URDF.load(str(assembled_path), mesh_dir=assembled_path.parent)
    )
    robot = Robot.load(
        description,
        load_collision_spheres=spheres_path is not None,
        collision_spheres_path=spheres_path,
    )
    return description, robot, tuple(coordinates), rest_q


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse robots and their collision spheres.")
    parser.add_argument("--category", choices=ROBOTS, default="ee")
    parser.add_argument("--robot")
    args = parser.parse_args()
    if args.robot is not None and args.robot not in ROBOTS[args.category]:
        parser.error(f"{args.robot!r} is not in category {args.category!r}")

    categories = tuple(ROBOTS)
    flat = tuple((category, name) for category, configs in ROBOTS.items() for name in configs)
    initial_robot = args.robot or next(iter(ROBOTS[args.category]))
    assembly_temp_dir = TemporaryDirectory(prefix="robokit-robot-models-")
    server = viser.ViserServer()
    with server.gui.add_folder("Robot"):
        category_dropdown = server.gui.add_dropdown("Category", categories, initial_value=args.category)
        robot_dropdown = server.gui.add_dropdown("Robot", tuple(ROBOTS[args.category]), initial_value=initial_robot)
        status = server.gui.add_text("Status", "Loading", disabled=True)
    with server.gui.add_folder("Online Assembly", expand_by_default=False):
        online_assembly = server.gui.add_checkbox("Assemble Robot", False)
        online_format = server.gui.add_dropdown("Format", MODEL_FORMATS, initial_value="URDF")
        online_base = server.gui.add_dropdown("Arm / Humanoid", tuple(_online_bases("URDF")), initial_value="xArm7")
        online_right_ee = server.gui.add_dropdown(
            "Right / Single EE", tuple(_online_ees("URDF", "right")), initial_value="XHand Right"
        )
        online_left_ee = server.gui.add_dropdown("Left EE", ("None",), initial_value="None")
        online_auto_mount = server.gui.add_checkbox("Auto Mount Point", True)
    with server.gui.add_folder("Settings"):
        show_robot = server.gui.add_checkbox("Show Robot", True)
        show_spheres = server.gui.add_checkbox("Show Collision Spheres", False)
        show_mount = server.gui.add_checkbox("Show Coordinate", False)
    joint_folder = server.gui.add_folder("Joint Control", expand_by_default=False)

    unit_sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    sphere_handle: Optional[viser.BatchedMeshHandle] = None
    coordinate_handles: List[Tuple[viser.FrameHandle, str, np.ndarray, np.ndarray]] = []
    robot_vis: Optional[Union[ViserUrdf, ViserMjcf]] = None
    robot: Optional[Robot] = None
    sliders = []
    loaded: Tuple[str, ...] = ()
    selected_category = args.category
    previous_q: Optional[np.ndarray] = None

    while True:
        assembling = online_assembly.value
        category_dropdown.disabled = assembling
        robot_dropdown.disabled = assembling
        for handle in (online_format, online_base, online_right_ee, online_left_ee, online_auto_mount):
            handle.disabled = not assembling

        if not assembling and category_dropdown.value != selected_category:
            selected_category = category_dropdown.value
            robot_dropdown.options = tuple(ROBOTS[selected_category])
            robot_dropdown.value = robot_dropdown.options[0]

        bilateral = False
        if assembling:
            base_options = tuple(_online_bases(online_format.value))
            if tuple(online_base.options) != base_options:
                online_base.options = base_options
                online_base.value = base_options[0]
            bilateral = len(entry_data(_online_bases(online_format.value)[online_base.value]).mounts) > 1
            right_options = (("None",) if bilateral else ()) + tuple(_online_ees(online_format.value, "right"))
            left_options = ("None",) + tuple(_online_ees(online_format.value, "left"))
            if tuple(online_right_ee.options) != right_options:
                online_right_ee.options = right_options
                online_right_ee.value = right_options[1] if bilateral else right_options[0]
            if tuple(online_left_ee.options) != left_options:
                online_left_ee.options = left_options
                online_left_ee.value = "None"
            if bilateral and online_right_ee.value == online_left_ee.value == "None":
                online_right_ee.value = right_options[1]
            online_left_ee.visible = bilateral
            selection = (
                "online",
                online_format.value,
                online_base.value,
                online_right_ee.value,
                online_left_ee.value if bilateral else "None",
                "auto" if online_auto_mount.value else "manual",
            )
        else:
            selection = ("offline", category_dropdown.value, robot_dropdown.value, "", "None", "manual")
        if selection != loaded:
            loaded = selection
            status.value = (
                f"Assembling {selection[2]}"
                if assembling
                else f"Loading {flat.index((selection[1], selection[2])) + 1}/{len(flat)}: {selection[1]}/{selection[2]}"
            )
            if robot_vis is not None:
                robot_vis.remove()
                robot_vis = None
            if sphere_handle is not None:
                sphere_handle.remove()
                sphere_handle = None
            for handle, _, _, _ in coordinate_handles:
                handle.remove()
            coordinate_handles = []
            for slider in sliders:
                slider.remove()
            sliders = []
            try:
                description, robot, coordinates, rest_q = (
                    _load_online(
                        Path(assembly_temp_dir.name),
                        selection[1],
                        selection[2],
                        selection[3],
                        selection[4],
                        auto_mount_point=selection[5] == "auto",
                    )
                    if assembling
                    else _load_robot(selection[1], selection[2])
                )
            except ValueError as error:
                status.value = str(error)
                continue
            robot_vis = (
                ViserMjcf(server, str(description), root_node_name="/robot", keep_world_link=False)
                if isinstance(description, Path)
                else ViserUrdf(server, description, root_node_name="/robot")
            )
            for i, (mount_link_name, mount_spec, ee_coord_spec) in enumerate(coordinates):
                outward = np.array(AXIS_TO_VEC[mount_spec.outward_axis], dtype=np.float32)
                right = np.array(AXIS_TO_VEC[mount_spec.right_axis], dtype=np.float32)
                R_mount_frame = np.column_stack([right, np.cross(outward, right), outward])
                path = f"/coordinate_{i}"
                mount_handle = server.scene.add_frame(path, axes_length=0.08, axes_radius=0.004)
                is_hand = isinstance(ee_coord_spec, HandCoordinateSpec)
                right_label = "palm right" if is_hand else "opening" if ee_coord_spec is not None else "right"
                outward_label = "four fingers up" if is_hand else "outward"
                server.scene.add_label(f"{path}/right", right_label, position=(0.09, 0.0, 0.0), font_screen_scale=0.7)
                if is_hand:
                    server.scene.add_label(
                        f"{path}/forward",
                        "palm forward",
                        position=(0.0, 0.09, 0.0),
                        font_screen_scale=0.7,
                    )
                server.scene.add_label(
                    f"{path}/outward", outward_label, position=(0.0, 0.0, 0.09), font_screen_scale=0.7
                )
                mount_point = (
                    np.zeros(3) if mount_spec.mount_point_xyz is None else np.array(mount_spec.mount_point_xyz)
                )
                coordinate_handles.append((mount_handle, mount_link_name, R_mount_frame, mount_point))
            previous_q = None
            sliders = []
            joint_limits = robot_vis.get_actuated_joint_limits()
            if set(joint_limits) != set(rest_q):
                raise ValueError(f"REST_Q mismatch: {set(joint_limits) ^ set(rest_q)}")
            with joint_folder:
                for joint_name, (lower, upper) in joint_limits.items():
                    lower = -np.pi if lower is None or lower == -np.inf else lower
                    upper = np.pi if upper is None or upper == np.inf else upper
                    sliders.append(
                        server.gui.add_slider(
                            joint_name, min=lower, max=upper, step=1e-3, initial_value=rest_q[joint_name]
                        )
                    )
            mount_status = "" if not coordinates else " | mount " + ", ".join(x[0] for x in coordinates)
            sphere_count = 0 if robot is None else len(robot.spec.collision_sphere_radii)
            model_format = "MJCF" if isinstance(description, Path) else "URDF"
            status.value = (
                f"Online: {selection[2]} + {selection[3]}"
                + (f" + {selection[4]}" if selection[4] != "None" else "")
                + f" | {model_format} | {sphere_count} spheres{mount_status}"
                if assembling
                else f"{flat.index((selection[1], selection[2])) + 1}/{len(flat)}: "
                f"{model_format} | {sphere_count} spheres{mount_status}"
            )

        if robot_vis is None:
            time.sleep(0.03)
            continue
        q = np.array([slider.value for slider in sliders], dtype=np.float32)
        if previous_q is None or not np.array_equal(q, previous_q):
            robot_vis.update_cfg(q)
            if robot is not None:
                state = robot.transform_collision_spheres(robot.forward_kinematics(robot.state(q=q)))
                positions = state.collision_sphere_centers_world.numpy()[0]
                if len(positions) and sphere_handle is None:
                    sphere_handle = server.scene.add_batched_meshes_simple(
                        "/collision_spheres",
                        vertices=unit_sphere.vertices,
                        faces=unit_sphere.faces,
                        batched_wxyzs=np.tile([1.0, 0.0, 0.0, 0.0], (len(robot.spec.collision_sphere_radii), 1)),
                        batched_positions=positions,
                        batched_scales=robot.spec.collision_sphere_radii,
                        batched_colors=(0.0, 0.8, 0.2),
                    )
                elif sphere_handle is not None:
                    sphere_handle.batched_positions = positions
                for mount_handle, mount_link_name, R_mount_frame, mount_point in coordinate_handles:
                    T_world_mount = state.T_world_link.numpy()[0, robot.spec.link_names.index(mount_link_name)]
                    R_world_mount = quaternion_to_matrix(T_world_mount[3:])
                    mount_handle.position = T_world_mount[:3] + R_world_mount @ mount_point
                    mount_handle.wxyz = matrix_to_quaternion(R_world_mount @ R_mount_frame)
            previous_q = q
        robot_vis.show_visual = show_robot.value
        if sphere_handle is not None:
            sphere_handle.visible = show_spheres.value
        for mount_handle, _, _, _ in coordinate_handles:
            mount_handle.visible = show_mount.value
        time.sleep(0.03)


if __name__ == "__main__":
    main()
