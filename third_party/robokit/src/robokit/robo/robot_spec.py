import hashlib
import io
import json
import logging
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from functools import cached_property, lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union, cast

import numpy as np
import trimesh
import yaml
import yourdfpy
from jaxtyping import Bool, Float, Int
from yourdfpy.urdf import filename_handler_null


if TYPE_CHECKING:
    import torch

    from robokit.robo.robot_spec_tensors import RobotSpecTensors, TorchRobotSpecTensors

from robokit.robo.yourdfpy import URDFWrapper
from robokit.xform.numpy import matrix_to_quaternion, quaternion_to_matrix, rot_tl_to_tf_mat


logger = logging.getLogger("robokit")


class JointType(IntEnum):
    FIXED = 0
    """Fixed joint, no motion."""
    PRISMATIC = 1
    """Prismatic joint, translation along a single axis."""
    REVOLUTE = 2
    """Revolute joint, rotation around a single axis."""


@dataclass
class RobotSpec:
    """Robot specification.

    Examples:
        >>> # xdoctest: +REQUIRES(env:ROBOKIT_ASSETS)
        >>> spec = RobotSpec.parse("tests/fixtures/wx250s.urdf")
        >>> spec.num_actuated_joints
        8
        >>> spec.num_links
        14
    """

    # --- general specification ---
    name: str
    """Name of the robot."""
    robot_description: str
    """Robot description"""
    format: Literal["urdf", "mjcf"]
    """Format of the robot description."""

    # --- joint specification ---
    joint_names: List[str]
    """Names of all joints, including actuated, mimic and fixed joints."""
    nonfixed_joint_names: List[str]
    """Names of non-fixed joints, including actuated and mimic joints."""
    actuated_joint_names: List[str]
    """Names of actuated joints."""
    joint_types: Int[np.ndarray, "num_joints"]
    """Types of all joints, see `JointType`."""
    joint_axes: Float[np.ndarray, "num_joints 3"]
    """Axes of all joints, [x, y, z]."""
    joint_twists: Float[np.ndarray, "num_joints 6"]
    """Twists of all joints, [linear (x, y, z), angular (x, y, z)]."""
    actuated_joint_indices: Int[np.ndarray, "num_joints"]
    """Indices of actuated joints, -1 for mimic and non-actuated joints."""
    parent_joint_indices: Int[np.ndarray, "num_joints"]
    """Indices of parent joints, -1 for joints attached to the base link."""
    parent_joint_transforms: Float[np.ndarray, "num_joints 7"]
    """Transforms of parent joints, [pos (x, y, z), quat (w, x, y, z)]."""

    mimic_actuated_joint_indices: Int[np.ndarray, "num_joints"]
    """Indices of mimicked actuated joints, -1 for non-mimic joints."""
    mimic_multipliers: Float[np.ndarray, "num_joints"]
    """Multipliers of mimicked actuated joints, 1.0 for non-mimic joints."""
    mimic_offsets: Float[np.ndarray, "num_joints"]
    """Offsets of mimicked actuated joints, 0.0 for non-mimic joints."""

    joint_limits: Float[np.ndarray, "num_joints 2"]
    """Limits of all joints, [lower, upper]."""
    joint_velocity_limits: Float[np.ndarray, "num_joints"]
    """Velocity limits of all joints."""
    actuated_joint_limits: Float[np.ndarray, "num_actuated_joints 2"]
    """Limits of actuated joints, [lower, upper]."""
    actuated_joint_velocity_limits: Float[np.ndarray, "num_actuated_joints"]
    """Velocity limits of actuated joints."""

    # --- link specification ---
    link_names: List[str]
    """Names of all links."""
    link_parent_joint_indices: Int[np.ndarray, "num_links"]
    """Indices of parent joints of all links, -1 for the base link."""
    base_link_name: str
    """Base link name (root of the parsed kinematic tree)."""
    ee_link_names: List[str]
    """End-effector (leaf) link names of the parsed kinematic tree."""

    # --- geometry specification ---
    link_visual_geometries: Dict[str, trimesh.Scene]
    """Visual geometries of all links."""
    link_collision_geometries: Dict[str, trimesh.Scene]
    """Collision geometries of all links."""

    local_collision_sphere_centers: Float[np.ndarray, "num_collision_spheres 3"]
    """Collision sphere centers in link-local coordinates."""
    collision_sphere_radii: Float[np.ndarray, "num_collision_spheres"]
    """Collision sphere radii."""
    collision_spheres_link_indices: Int[np.ndarray, "num_collision_spheres"]
    """Link indices (in link_names) for each collision sphere."""
    self_collision_ignored_pairs: Int[np.ndarray, "num_ignored_pairs 2"]
    """Load ignored self-collision link pairs verbatim from `self_collision_ignore_path`, or return none."""

    # --- inertial specification ---
    link_masses: Float[np.ndarray, "num_links"]
    """Mass of each link (0.0 if no inertial data)."""
    link_local_com_positions: Float[np.ndarray, "num_links 3"]
    """Center of mass position in link-local frame."""

    loaded_link_capsule_endpoints: Optional[Float[np.ndarray, "num_links 2 3"]] = None
    """Baked capsule endpoints from `link_capsules_path`; unset means fit them from the collision meshes."""
    loaded_link_capsule_radii: Optional[Float[np.ndarray, "num_links"]] = None
    """Baked capsule radii from `link_capsules_path`."""
    self_collision_link_buffers: Dict[str, float] = field(default_factory=dict)
    """Per-link self-collision radius inflation from the collision-sphere YAML; world radii stay unchanged."""

    # --- parse functions ---
    @staticmethod
    def parse(
        robot_description_or_path: Union[str, Path, yourdfpy.URDF],
        load_meshes: bool = False,
        mesh_dir: Optional[Union[str, Path]] = None,
        load_collision_spheres: bool = False,
        collision_spheres_path: Optional[Union[str, Path]] = None,
        load_link_capsules: bool = False,
        link_capsules_path: Optional[Union[str, Path]] = None,
        self_collision_ignore_path: Optional[Union[str, Path]] = None,
        base_link_name: Optional[str] = None,
        ee_link_names: Optional[Union[Sequence[str], str]] = None,
        keep_mjcf_world_link: bool = False,
    ) -> "RobotSpec":
        format: Literal["urdf", "mjcf"]

        is_str = isinstance(robot_description_or_path, str)
        if isinstance(robot_description_or_path, yourdfpy.URDF) or (is_str and "<robot" in robot_description_or_path):
            format = "urdf"
        elif is_str and "<mujoco" in robot_description_or_path:
            format = "mjcf"
        elif isinstance(robot_description_or_path, (str, Path)):
            path = Path(robot_description_or_path)
            if not path.exists():
                raise FileNotFoundError(f"Robot description file not found: {robot_description_or_path}")
            suffix = path.suffix.lower()
            if suffix == ".urdf":
                format = "urdf"
            elif suffix == ".mjcf":
                format = "mjcf"
            elif suffix == ".xml":
                format = "mjcf" if "<mujoco" in path.read_text() else "urdf"
            else:
                raise ValueError(f"Unsupported file extension: {suffix}")
        else:
            raise ValueError(f"Invalid robot description type: {type(robot_description_or_path)}")

        if format == "urdf":
            spec = RobotSpec.parse_urdf(
                robot_description_or_path,
                load_meshes=load_meshes,
                mesh_dir=mesh_dir,
                load_collision_spheres=load_collision_spheres,
                collision_spheres_path=collision_spheres_path,
                load_link_capsules=load_link_capsules,
                link_capsules_path=link_capsules_path,
                base_link_name=base_link_name,
                ee_link_names=ee_link_names,
            )
        elif format == "mjcf":
            assert not isinstance(robot_description_or_path, yourdfpy.URDF)
            spec = RobotSpec.parse_mjcf(
                robot_description_or_path,
                load_meshes=load_meshes,
                load_collision_spheres=load_collision_spheres,
                collision_spheres_path=collision_spheres_path,
                load_link_capsules=load_link_capsules,
                link_capsules_path=link_capsules_path,
                base_link_name=base_link_name,
                ee_link_names=ee_link_names,
                keep_world_link=keep_mjcf_world_link,
            )
        else:
            raise ValueError(f"Unsupported robot description format: {format}")
        if self_collision_ignore_path is not None:
            spec.self_collision_ignored_pairs = RobotSpec._load_self_collision_ignore(
                spec.link_names, self_collision_ignore_path
            )
        return spec

    @staticmethod
    def _load_self_collision_ignore(
        link_names: List[str], path: Union[str, Path]
    ) -> Int[np.ndarray, "num_ignored_pairs 2"]:
        """Map ignore-file link names to indices, skipping links pruned from the robot spec."""
        link_index = {name: i for i, name in enumerate(link_names)}
        with open(path, "r") as f:
            entries = yaml.safe_load(f)["self_collision_ignore"] or {}
        pairs: Set[Tuple[int, int]] = set()
        for name_a, names_b in entries.items():
            for name_b in names_b:
                if name_a in link_index and name_b in link_index:
                    pairs.add((link_index[name_a], link_index[name_b]))
        if not pairs:
            return np.empty((0, 2), dtype=np.int32)
        return np.array(sorted(pairs), dtype=np.int32)

    @staticmethod
    def _load_collision_spheres(
        link_names: List[str],
        load: bool,
        collision_spheres_path: Optional[Union[str, Path]],
    ) -> Tuple[
        Float[np.ndarray, "num_collision_spheres 3"],
        Float[np.ndarray, "num_collision_spheres"],
        Int[np.ndarray, "num_collision_spheres"],
        Dict[str, float],
    ]:
        """Load collision spheres from YAML (format-agnostic), or none if `load` is off; shared by both parsers."""
        sphere_data = {}
        if load:
            if collision_spheres_path is None:
                raise ValueError("Collision spheres path must be provided when loading collision spheres.")
            with open(collision_spheres_path, "r") as f:
                sphere_data = yaml.safe_load(f)
        all_centers, all_radii, all_link_indices = [], [], []
        for link_name, spheres in sphere_data.get("collision_spheres", {}).items():
            if link_name not in link_names:
                continue
            link_idx = link_names.index(link_name)
            for sphere in spheres:
                center = sphere["center"]
                all_centers.append([center[0], center[1], center[2]])
                all_radii.append(sphere["radius"])
                all_link_indices.append(link_idx)
        centers = np.array(all_centers, dtype=np.float32) if all_centers else np.empty((0, 3), dtype=np.float32)
        radii = np.array(all_radii, dtype=np.float32) if all_radii else np.empty(0, dtype=np.float32)
        link_indices = np.array(all_link_indices, dtype=np.int32) if all_link_indices else np.empty(0, dtype=np.int32)
        return centers, radii, link_indices, sphere_data.get("self_collision_link_buffers", {})

    @staticmethod
    def _load_link_capsules(
        link_names: List[str],
        load: bool,
        link_capsules_path: Optional[Union[str, Path]],
    ) -> Tuple[Optional[Float[np.ndarray, "num_links 2 3"]], Optional[Float[np.ndarray, "num_links"]]]:
        """Load baked per-link capsules from YAML, or none if `load` is off; shared by both parsers."""
        if not load:
            return None, None
        if link_capsules_path is None:
            raise ValueError("Link capsules path must be provided when loading link capsules.")
        with open(link_capsules_path, "r") as f:
            capsule_data = yaml.safe_load(f)
        endpoints = np.zeros((len(link_names), 2, 3), dtype=np.float32)
        radii = np.zeros(len(link_names), dtype=np.float32)
        for link_name, capsule in capsule_data.get("link_capsules", {}).items():
            if link_name not in link_names:
                continue
            link_idx = link_names.index(link_name)
            endpoints[link_idx] = np.asarray([capsule["a"], capsule["b"]], dtype=np.float32)
            radii[link_idx] = float(capsule["radius"])
        return endpoints, radii

    @staticmethod
    def parse_urdf(
        robot_description_or_path: Union[str, Path, yourdfpy.URDF],
        load_meshes: bool = False,
        mesh_dir: Optional[Union[str, Path]] = None,
        load_collision_spheres: bool = False,
        collision_spheres_path: Optional[Union[str, Path]] = None,
        load_link_capsules: bool = False,
        link_capsules_path: Optional[Union[str, Path]] = None,
        base_link_name: Optional[str] = None,
        ee_link_names: Optional[Union[Sequence[str], str]] = None,
    ) -> "RobotSpec":
        # helper functions
        def get_joint_type(joint: yourdfpy.Joint) -> JointType:
            if joint.type in ("revolute", "continuous"):
                return JointType.REVOLUTE
            elif joint.type == "prismatic":
                return JointType.PRISMATIC
            elif joint.type == "fixed":
                return JointType.FIXED
            else:
                logger.warning(
                    f"Unsupported joint type {joint.type} for joint '{joint.name}'. Falling back to fixed joint."
                )
                return JointType.FIXED

        def get_joint_axis(joint: yourdfpy.Joint) -> Float[np.ndarray, "3"]:
            if joint.type in ("revolute", "continuous", "prismatic"):
                return joint.axis
            elif joint.type == "fixed":
                return np.zeros(3)
            else:
                logger.warning(
                    f"Unsupported joint type {joint.type} for joint '{joint.name}'. Falling back to zero axis."
                )
                return np.zeros(3)

        def get_joint_twist(joint: yourdfpy.Joint) -> Float[np.ndarray, "6"]:
            if joint.type in ("revolute", "continuous"):
                return np.concatenate([np.zeros(3), joint.axis])
            elif joint.type == "prismatic":
                return np.concatenate([joint.axis, np.zeros(3)])
            elif joint.type == "fixed":
                return np.zeros(6)
            else:
                logger.warning(
                    f"Unsupported joint type {joint.type} for joint '{joint.name}'. Falling back to fixed joint."
                )
                return np.zeros(6)

        def get_actuated_joint_index(urdf: yourdfpy.URDF, joint: yourdfpy.Joint) -> int:
            if joint.mimic is not None:
                return -1
            elif joint.name in urdf.actuated_joint_names:
                return urdf.actuated_joint_names.index(joint.name)
            else:
                return -1

        def get_mimic_joint_info(urdf: yourdfpy.URDF, joint: yourdfpy.Joint) -> Tuple[int, float, float]:
            mimic_actuated_joint_index, mimic_multiplier, mimic_offset = -1, 1.0, 0.0
            if joint.mimic is not None:
                if joint.mimic.multiplier is not None:
                    mimic_multiplier = joint.mimic.multiplier
                if joint.mimic.offset is not None:
                    mimic_offset = joint.mimic.offset
                mimicked_joint = urdf.joint_map[joint.mimic.joint]
                mimic_actuated_joint_index = urdf.actuated_joint_names.index(mimicked_joint.name)
            return mimic_actuated_joint_index, mimic_multiplier, mimic_offset

        def get_joint_limits(joint: yourdfpy.Joint) -> Tuple[float, float]:
            if joint.limit is not None and joint.limit.lower is not None and joint.limit.upper is not None:
                return joint.limit.lower, joint.limit.upper
            elif joint.type == "continuous":
                # continuous joints are unbounded; [-pi, pi] is a sampling-safe placeholder (inf breaks seeding)
                return -math.pi, math.pi
            elif joint.type == "revolute":
                logger.warning(
                    f"Joint '{joint.name}' ({joint.type}) has no specified limits. Falling back to [-pi, pi]."
                )
                return -math.pi, math.pi
            elif joint.type == "prismatic":
                logger.warning(
                    f"Joint '{joint.name}' ({joint.type}) has no specified limits. Falling back to [-1.0, 1.0]."
                )
                return -1.0, 1.0
            elif joint.type == "fixed":
                return 0.0, 0.0
            else:
                raise ValueError(f"Joint '{joint.name}' ({joint.type}) has no specified limits.")

        def get_joint_velocity_limit(joint: yourdfpy.Joint) -> float:
            if joint.limit is not None and joint.limit.velocity is not None:
                return joint.limit.velocity
            elif joint.type == "fixed":
                return 0.0
            elif joint.type == "continuous":
                return 10.0  # continuous joints rarely specify velocity; default silently
            else:
                logger.warning(
                    f"Joint '{joint.name}' ({joint.type}) has no specified velocity limit. Falling back to 10.0."
                )
                return 10.0

        def get_parent_joint_info(
            urdf: yourdfpy.URDF,
            joint: yourdfpy.Joint,
        ) -> Tuple[int, Float[np.ndarray, "7"]]:
            parent_joint_transform = np.eye(4)
            if joint.origin is not None:
                parent_joint_transform = joint.origin

            child_link_to_joint = {j.child: j for j in urdf.joint_map.values()}
            joint_name_to_idx = {joint_name: i for i, joint_name in enumerate(urdf.joint_names)}

            if joint.parent not in child_link_to_joint:
                parent_index = -1
            else:
                parent_joint = child_link_to_joint[joint.parent]
                parent_index = joint_name_to_idx[parent_joint.name]

            q_wxyz_parent_joint = matrix_to_quaternion(parent_joint_transform[:3, :3])
            parent_joint_pos = parent_joint_transform[:3, 3]
            parent_joint_transform = np.concatenate([parent_joint_pos, q_wxyz_parent_joint], axis=-1)
            return parent_index, parent_joint_transform

        def get_link_geometry(
            urdf: yourdfpy.URDF, link_name: str, mode: Literal["visual", "collision"]
        ) -> trimesh.Scene:
            source_scene = urdf.scene if mode == "visual" else urdf.collision_scene
            link_scene = trimesh.Scene()
            if source_scene is None or (link_name not in source_scene.graph.nodes):
                return link_scene

            children = source_scene.graph.transforms.children.get(link_name, [])

            for child_node in children:
                if child_node in source_scene.geometry:
                    geometry = source_scene.geometry[child_node]
                    transform = source_scene.graph.get(child_node, frame_from=link_name)[0]
                    link_scene.add_geometry(geometry, transform=transform)

            return link_scene

        if isinstance(robot_description_or_path, yourdfpy.URDF):
            urdf = robot_description_or_path
            # caller-passed URDFs often lack a collision scene; rebuild so capsule fitting has geometry
            if load_meshes and urdf.collision_scene is None:
                urdf = yourdfpy.URDF(
                    robot=urdf.robot,
                    filename_handler=urdf._filename_handler,
                    build_scene_graph=True,
                    build_collision_scene_graph=True,
                    load_meshes=True,
                    load_collision_meshes=True,
                )
        else:
            if isinstance(robot_description_or_path, Path) or os.path.isfile(robot_description_or_path):
                # `str`, not `Path`: yourdfpy only derives `mesh_dir` from the filename for string paths.
                source = str(robot_description_or_path)
            elif isinstance(robot_description_or_path, str):
                source = io.BytesIO(robot_description_or_path.encode())
            else:
                raise ValueError(f"Invalid robot description type: {type(robot_description_or_path)}")
            kwargs: Dict[str, Any] = {
                "build_scene_graph": load_meshes,
                "build_collision_scene_graph": load_meshes,
                "load_meshes": load_meshes,
                "load_collision_meshes": load_meshes,
            }
            if mesh_dir is not None:
                kwargs["mesh_dir"] = str(mesh_dir)
            logging.getLogger("yourdfpy").setLevel(logging.ERROR)
            urdf = yourdfpy.URDF.load(source, **kwargs)

        # normalize ee_link_names to list
        if isinstance(ee_link_names, str):
            ee_link_names = [ee_link_names]

        # filter links and joints based on base_link_name and/or ee_link_names
        if base_link_name is not None or ee_link_names is not None:
            child_to_parent = {j.child: j.parent for j in urdf.robot.joints}
            parent_to_children: Dict[str, List[str]] = defaultdict(list)
            for j in urdf.robot.joints:
                parent_to_children[j.parent].append(j.child)

            filtered_links = {link.name for link in urdf.robot.links}

            if base_link_name is not None:
                # keep only descendants of base_link_name
                descendants, stack = set(), [base_link_name]
                while stack:
                    link = stack.pop()
                    descendants.add(link)
                    stack.extend(parent_to_children.get(link, []))
                filtered_links &= descendants

            if ee_link_names is not None:
                # keep only ancestors of ee_link_names (up to base_link_name or root)
                ancestors = {base_link_name} if base_link_name else set()
                for ee in ee_link_names:
                    current: Optional[str] = ee
                    while current is not None and current != base_link_name:
                        ancestors.add(current)
                        current = child_to_parent.get(current)
                filtered_links &= ancestors

            urdf.robot.links = [link for link in urdf.robot.links if link.name in filtered_links]
            urdf.robot.joints = [
                j for j in urdf.robot.joints if j.parent in filtered_links and j.child in filtered_links
            ]
            urdf._create_maps()
            urdf._update_actuated_joints()

        # collect joint specification
        joint_names, actuated_joint_names = urdf.joint_names, urdf.actuated_joint_names
        nonfixed_joint_names = []
        joint_types, joint_axes = [], []
        joint_twists, actuated_joint_indices = [], []
        mimic_actuated_joint_indices, mimic_multipliers, mimic_offsets = [], [], []
        joint_limits, joint_velocity_limits = [], []
        parent_joint_indices, parent_joint_transforms = [], []
        for joint_name in joint_names:
            joint = urdf.joint_map[joint_name]
            joint_types.append(get_joint_type(joint))
            joint_axes.append(get_joint_axis(joint))
            joint_twists.append(get_joint_twist(joint))
            actuated_joint_indices.append(get_actuated_joint_index(urdf, joint))
            mimic_actuated_joint_index, mimic_multiplier, mimic_offset = get_mimic_joint_info(urdf, joint)
            mimic_actuated_joint_indices.append(mimic_actuated_joint_index)
            mimic_multipliers.append(mimic_multiplier)
            mimic_offsets.append(mimic_offset)
            joint_limits.append(get_joint_limits(joint))
            joint_velocity_limits.append(get_joint_velocity_limit(joint))
            parent_joint_index, parent_joint_transform = get_parent_joint_info(urdf, joint)
            parent_joint_indices.append(parent_joint_index)
            parent_joint_transforms.append(parent_joint_transform)
            if joint.type != "fixed":
                nonfixed_joint_names.append(joint_name)

        actuated_joint_limits = [get_joint_limits(urdf.joint_map[joint_name]) for joint_name in actuated_joint_names]
        actuated_joint_velocity_limits = [
            get_joint_velocity_limit(urdf.joint_map[joint_name]) for joint_name in actuated_joint_names
        ]

        # collect link specification
        link_parent_joint_indices = []
        child_link_to_joint = {j.child: j for j in urdf.joint_map.values()}
        link_names = list(urdf.link_map.keys())
        for link_name in link_names:
            if link_name in child_link_to_joint:
                joint_idx = joint_names.index(child_link_to_joint[link_name].name)
                link_parent_joint_indices.append(joint_idx)
            else:
                link_parent_joint_indices.append(-1)
        if base_link_name is None:
            child_links = {j.child for j in urdf.robot.joints}
            base_link_name = next(n for n in link_names if n not in child_links)
        if ee_link_names is None:
            parent_links = {j.parent for j in urdf.robot.joints}
            ee_link_names = [n for n in link_names if n not in parent_links]

        # collect geometry specification
        if load_meshes:
            link_visual_geometries = {
                link_name: get_link_geometry(urdf, link_name, "visual") for link_name in link_names
            }
            link_collision_geometries = {
                link_name: get_link_geometry(urdf, link_name, "collision") for link_name in link_names
            }
        else:
            link_visual_geometries, link_collision_geometries = {}, {}

        (
            local_collision_sphere_centers,
            collision_sphere_radii,
            collision_spheres_link_indices,
            self_collision_link_buffers,
        ) = RobotSpec._load_collision_spheres(link_names, load_collision_spheres, collision_spheres_path)
        loaded_link_capsule_endpoints, loaded_link_capsule_radii = RobotSpec._load_link_capsules(
            link_names, load_link_capsules, link_capsules_path
        )

        # collect inertial specification
        link_masses = []
        link_local_com_positions = []
        for link_name in link_names:
            link = urdf.link_map[link_name]
            if link.inertial is not None and link.inertial.mass is not None:
                link_masses.append(link.inertial.mass)
                origin = link.inertial.origin
                link_local_com_positions.append(origin[:3, 3] if origin is not None else np.zeros(3))
            else:
                link_masses.append(0.0)
                link_local_com_positions.append(np.zeros(3))

        # Note different from the original yourdfpy, we add mimic joint writing in the URDFWrapper
        urdf.__class__ = URDFWrapper
        if not mesh_dir:
            urdf._filename_handler = filename_handler_null
        return RobotSpec(
            name=urdf.robot.name,
            robot_description=urdf.write_xml_string().decode(),
            format="urdf",
            joint_names=joint_names,
            nonfixed_joint_names=nonfixed_joint_names,
            actuated_joint_names=actuated_joint_names,
            joint_types=np.asarray(joint_types, dtype=np.int32),
            joint_axes=np.stack(joint_axes) if joint_axes else np.empty((0, 3)),
            joint_twists=np.stack(joint_twists) if joint_twists else np.empty((0, 6)),
            actuated_joint_indices=np.asarray(actuated_joint_indices, dtype=np.int32),
            parent_joint_indices=np.asarray(parent_joint_indices, dtype=np.int32),
            parent_joint_transforms=np.stack(parent_joint_transforms) if parent_joint_transforms else np.empty((0, 7)),
            mimic_actuated_joint_indices=np.asarray(mimic_actuated_joint_indices, dtype=np.int32),
            mimic_multipliers=np.asarray(mimic_multipliers),
            mimic_offsets=np.asarray(mimic_offsets),
            joint_limits=np.asarray(joint_limits) if joint_limits else np.empty((0, 2)),
            joint_velocity_limits=np.asarray(joint_velocity_limits),
            actuated_joint_limits=np.asarray(actuated_joint_limits) if actuated_joint_limits else np.empty((0, 2)),
            actuated_joint_velocity_limits=np.asarray(actuated_joint_velocity_limits),
            link_names=link_names,
            link_parent_joint_indices=np.asarray(link_parent_joint_indices, dtype=np.int32),
            base_link_name=cast(str, base_link_name),
            ee_link_names=list(ee_link_names),
            link_visual_geometries=link_visual_geometries,
            link_collision_geometries=link_collision_geometries,
            local_collision_sphere_centers=local_collision_sphere_centers,
            collision_sphere_radii=collision_sphere_radii,
            collision_spheres_link_indices=collision_spheres_link_indices,
            self_collision_link_buffers=self_collision_link_buffers,
            loaded_link_capsule_endpoints=loaded_link_capsule_endpoints,
            loaded_link_capsule_radii=loaded_link_capsule_radii,
            self_collision_ignored_pairs=np.empty((0, 2), dtype=np.int32),
            link_masses=np.asarray(link_masses, dtype=np.float64),
            link_local_com_positions=np.stack(link_local_com_positions),
        )

    @staticmethod
    def parse_mjcf(
        robot_description_or_path: Union[str, Path],
        load_meshes: bool = False,
        load_collision_spheres: bool = False,
        collision_spheres_path: Optional[Union[str, Path]] = None,
        load_link_capsules: bool = False,
        link_capsules_path: Optional[Union[str, Path]] = None,
        base_link_name: Optional[str] = None,
        ee_link_names: Optional[Union[Sequence[str], str]] = None,
        keep_world_link: bool = False,
    ) -> "RobotSpec":
        """Parse an MJCF robot: hinge/slide joints, equality mimics; the world body stays if kept or a root has joints. A hinge with an anchor (`jnt_pos`) turns at a synthetic `<body>_anchor` link, then a fixed joint shifts back."""
        import mujoco  # type: ignore  # no wheels/stubs for py<3.10; lazily imported only when parsing MJCF

        def _mjcf_geom_to_trimesh(model: Any, geom_idx: int) -> Optional[trimesh.Trimesh]:
            """Build a link-local trimesh for one MuJoCo geom (primitives + baked meshes); None for non-link geoms."""
            geom_type = int(model.geom_type[geom_idx])
            size = model.geom_size[geom_idx].astype(np.float64)
            if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                return trimesh.creation.icosphere(radius=float(size[0]))
            if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
                return trimesh.creation.box(extents=2.0 * size[:3])
            if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                return trimesh.creation.cylinder(radius=float(size[0]), height=2.0 * float(size[1]))
            if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                return trimesh.creation.capsule(radius=float(size[0]), height=2.0 * float(size[1]))
            if geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                mesh = trimesh.creation.icosphere(radius=1.0)
                mesh.apply_scale(size[:3])
                return mesh
            if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
                data_id = int(model.geom_dataid[geom_idx])
                v0, vn = int(model.mesh_vertadr[data_id]), int(model.mesh_vertnum[data_id])
                f0, fn = int(model.mesh_faceadr[data_id]), int(model.mesh_facenum[data_id])
                vertices = np.asarray(model.mesh_vert[v0 : v0 + vn], dtype=np.float64).reshape(-1, 3)
                faces = np.asarray(model.mesh_face[f0 : f0 + fn], dtype=np.int64).reshape(-1, 3)
                return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            return None

        # one getter for a body's joint; j == -1 means no joint (treated as fixed)
        def get_joint_info(
            model: Any, b: int, j: int
        ) -> Tuple[str, JointType, Float[np.ndarray, "3"], Float[np.ndarray, "6"], Tuple[float, float], float]:
            """(name, type, axis, twist, limits, velocity_limit) for body b's joint; j < 0 means no joint (fixed)."""
            if j < 0:
                body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body_{b}"
                return f"{body}_fixed", JointType.FIXED, np.zeros(3), np.zeros(6), (0.0, 0.0), 0.0
            jnt_type = int(model.jnt_type[j])
            bid = int(model.jnt_bodyid[j])
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body_{bid}"
            if jnt_type not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                raise ValueError(f"Joint on body '{body}' is free/ball; only hinge and slide are supported.")
            axis = model.jnt_axis[j].astype(np.float64)
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
            if jnt_type == int(mujoco.mjtJoint.mjJNT_HINGE):
                jtype, twist, default_limit = (
                    JointType.REVOLUTE,
                    np.concatenate([np.zeros(3), axis]),
                    (-math.pi, math.pi),
                )
            else:
                jtype, twist, default_limit = JointType.PRISMATIC, np.concatenate([axis, np.zeros(3)]), (-1.0, 1.0)
            limits = (
                (float(model.jnt_range[j, 0]), float(model.jnt_range[j, 1])) if model.jnt_limited[j] else default_limit
            )
            return name, jtype, axis, twist, limits, 10.0

        if isinstance(robot_description_or_path, Path) or (
            isinstance(robot_description_or_path, str) and os.path.isfile(robot_description_or_path)
        ):
            path = str(robot_description_or_path)
            robot_description = Path(path).read_text()
            model = mujoco.MjModel.from_xml_path(path)
        else:
            robot_description = str(robot_description_or_path)
            model = mujoco.MjModel.from_xml_string(robot_description)

        nbody = model.nbody
        body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body_{b}" for b in range(nbody)]

        # links are bodies; the world body (id 0) is dropped unless keep_world_link or a root body has joints
        if base_link_name is None and any(model.body_jntnum[b] and model.body_parentid[b] == 0 for b in range(nbody)):
            keep_world_link = True
        link_body_ids = list(range(nbody)) if keep_world_link else list(range(1, nbody))

        # sub-chain pruning like parse_urdf: keep bodies between base and ee links; pruned base re-roots at identity
        keep: Optional[set] = None
        if base_link_name is not None or ee_link_names is not None:
            ee_list = [ee_link_names] if isinstance(ee_link_names, str) else ee_link_names
            name_to_body = {body_names[b]: b for b in range(nbody)}
            children: Dict[int, List[int]] = defaultdict(list)
            for b in range(1, nbody):
                children[int(model.body_parentid[b])].append(b)
            keep = set(link_body_ids)
            base_body = name_to_body[base_link_name] if base_link_name is not None else None
            if base_body is not None:
                descendants, stack = set(), [base_body]
                while stack:
                    x = stack.pop()
                    descendants.add(x)
                    stack.extend(children[x])
                keep &= descendants
            if ee_list is not None:
                ancestors = {0 if base_body is None else base_body}
                for ee in ee_list:
                    cur: Optional[int] = name_to_body[ee]
                    while cur is not None and cur not in (base_body, 0):
                        ancestors.add(cur)
                        cur = int(model.body_parentid[cur])
                keep &= ancestors
            link_body_ids = [b for b in link_body_ids if b in keep]

        link_names = [body_names[b] for b in link_body_ids]
        body_to_link = {b: i for i, b in enumerate(link_body_ids)}

        # one joint per non-world body (hinge/slide or synthetic fixed); the base link has none when world is dropped
        joint_names, joint_types, joint_axes = [], [], []
        joint_twists, joint_limits, joint_velocity_limits = [], [], []
        parent_joint_indices, parent_joint_transforms = [], []
        joint_of_body: Dict[int, int] = {}  # body -> joint whose frame is the body's link
        joint_of_jnt: Dict[int, int] = {}  # MuJoCo joint id -> joint
        anchor_links: List[Tuple[str, int]] = []  # (synthetic anchor link, its parent joint)
        for b in range(1, nbody):
            if keep is not None and b not in keep:
                continue
            parent_body = int(model.body_parentid[b])
            jnt_num = int(model.body_jntnum[b])
            if jnt_num > 1:
                raise ValueError(
                    f"Body '{body_names[b]}' has {jnt_num} joints; only 0 or 1 joint per body is supported."
                )
            if parent_body not in body_to_link:
                # base of the (sub-)chain (parent pruned or world): placement is scene data -- re-root at identity
                if parent_body == 0 and jnt_num != 0:
                    raise ValueError(
                        f"Body '{body_names[b]}' is a non-fixed base; pass keep_world_link=True to parse it."
                    )
                continue

            j = int(model.body_jntadr[b]) if jnt_num else -1
            info = get_joint_info(model, b, j)
            T_parent_joint = np.concatenate([model.body_pos[b], model.body_quat[b]]).astype(np.float64)
            chain = [(info, T_parent_joint)]
            if info[1] == JointType.REVOLUTE and np.any(model.jnt_pos[j] != 0.0):
                # offset hinge: Trans(p) * Rot(q) at the anchor link, then a fixed Trans(-p) to the body's link
                anchor = model.jnt_pos[j].astype(np.float64)
                T_parent_joint[:3] += quaternion_to_matrix(T_parent_joint[3:]) @ anchor
                anchor_links.append((f"{body_names[b]}_anchor", len(joint_names)))
                chain.append((get_joint_info(model, b, -1), np.concatenate([-anchor, [1.0, 0.0, 0.0, 0.0]])))
            if j >= 0:
                joint_of_jnt[j] = len(joint_names)
            parent_joint = joint_of_body.get(parent_body, -1)
            for (name, jtype, axis, twist, limits, vel), T in chain:
                joint_names.append(name)
                joint_types.append(jtype)
                joint_axes.append(axis)
                joint_twists.append(twist)
                joint_limits.append(limits)
                joint_velocity_limits.append(vel)
                parent_joint_indices.append(parent_joint)
                parent_joint_transforms.append(T)
                parent_joint = len(joint_names) - 1
            joint_of_body[b] = parent_joint
        num_joints = len(joint_names)

        # mimic joints from <equality><joint>: q_dep = q_dep0 + a0 + a1 * (q_ind - q_ind0), linear only
        mimic_actuated_joint_indices = [-1] * num_joints
        mimic_multipliers = [1.0] * num_joints
        mimic_offsets = [0.0] * num_joints
        mimic_independent_joint: Dict[int, int] = {}
        for e in range(model.neq):
            if int(model.eq_type[e]) != int(mujoco.mjtEq.mjEQ_JOINT):
                continue
            dep_jnt, ind_jnt = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
            if ind_jnt < 0:
                raise ValueError("Joint equality constraint without an independent joint is not supported.")
            data = model.eq_data[e]
            if not np.allclose(data[2:5], 0.0):
                raise ValueError("Only linear joint equality constraints map to mimic joints.")
            if dep_jnt not in joint_of_jnt or ind_jnt not in joint_of_jnt:
                continue  # a mimic constraint touching a pruned-away joint no longer applies
            a0, a1 = float(data[0]), float(data[1])
            y0 = float(model.qpos0[int(model.jnt_qposadr[dep_jnt])])
            x0 = float(model.qpos0[int(model.jnt_qposadr[ind_jnt])])
            dep = joint_of_jnt[dep_jnt]
            mimic_multipliers[dep] = a1
            mimic_offsets[dep] = y0 + a0 - a1 * x0
            mimic_independent_joint[dep] = joint_of_jnt[ind_jnt]

        # actuated joints are non-fixed and non-mimic; MuJoCo <actuator> tags are irrelevant to kinematics
        nonfixed_joint_names = [joint_names[j] for j in range(num_joints) if joint_types[j] != JointType.FIXED]
        actuated_joint_names = [
            joint_names[j]
            for j in range(num_joints)
            if joint_types[j] != JointType.FIXED and j not in mimic_independent_joint
        ]
        actuated_index = {name: i for i, name in enumerate(actuated_joint_names)}
        actuated_joint_indices = [actuated_index.get(joint_names[j], -1) for j in range(num_joints)]
        for dep, ind in mimic_independent_joint.items():
            ind_actuated = actuated_index.get(joint_names[ind], -1)
            if ind_actuated < 0:
                raise ValueError("Mimic joint references a non-actuated independent joint.")
            mimic_actuated_joint_indices[dep] = ind_actuated
        name_to_joint = {name: j for j, name in enumerate(joint_names)}
        actuated_joint_limits = [joint_limits[name_to_joint[n]] for n in actuated_joint_names]
        actuated_joint_velocity_limits = [joint_velocity_limits[name_to_joint[n]] for n in actuated_joint_names]

        # link specification; synthetic anchor links come after the bodies
        link_names += [name for name, _ in anchor_links]
        link_parent_joint_indices = [joint_of_body.get(b, -1) for b in link_body_ids] + [j for _, j in anchor_links]
        if base_link_name is None:
            if keep_world_link:
                base_link_name = body_names[0]
            else:
                base_link_name = next(body_names[b] for b in link_body_ids if int(model.body_parentid[b]) == 0)
        if ee_link_names is None:
            # leaves of the (possibly pruned) kept set: kept bodies that are no kept body's parent
            parent_body_ids = {int(model.body_parentid[b]) for b in link_body_ids}
            ee_link_names = [body_names[b] for b in link_body_ids if b not in parent_body_ids]
        elif isinstance(ee_link_names, str):
            ee_link_names = [ee_link_names]
        else:
            ee_link_names = list(ee_link_names)

        # geometry: visual = non-colliding geoms (contype==0 and conaffinity==0), collision = the rest
        if load_meshes:
            link_visual_geometries = {name: trimesh.Scene() for name in link_names}
            link_collision_geometries = {name: trimesh.Scene() for name in link_names}
            for g in range(model.ngeom):
                b = int(model.geom_bodyid[g])
                if b not in body_to_link:
                    continue
                geom_mesh = _mjcf_geom_to_trimesh(model, g)
                if geom_mesh is None:
                    continue
                transform = rot_tl_to_tf_mat(
                    quaternion_to_matrix(model.geom_quat[g].astype(np.float64)), model.geom_pos[g].astype(np.float64)
                )
                link_name = link_names[body_to_link[b]]
                if int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0:
                    link_visual_geometries[link_name].add_geometry(geom_mesh, transform=transform)
                else:
                    link_collision_geometries[link_name].add_geometry(geom_mesh, transform=transform)
            # models like xarm7 reuse the collision mesh for display; fall back to it when a link has no visual geom
            for name in link_names:
                if len(link_visual_geometries[name].geometry) == 0:
                    link_visual_geometries[name] = link_collision_geometries[name]
        else:
            link_visual_geometries, link_collision_geometries = {}, {}

        (
            local_collision_sphere_centers,
            collision_sphere_radii,
            collision_spheres_link_indices,
            self_collision_link_buffers,
        ) = RobotSpec._load_collision_spheres(link_names, load_collision_spheres, collision_spheres_path)
        loaded_link_capsule_endpoints, loaded_link_capsule_radii = RobotSpec._load_link_capsules(
            link_names, load_link_capsules, link_capsules_path
        )

        # inertial specification
        link_masses = [float(model.body_mass[b]) for b in link_body_ids] + [0.0] * len(anchor_links)
        link_local_com_positions = [model.body_ipos[b].astype(np.float64) for b in link_body_ids]
        link_local_com_positions += [np.zeros(3)] * len(anchor_links)

        name_match = re.search(r'<mujoco[^>]*\bmodel\s*=\s*"([^"]+)"', robot_description)

        return RobotSpec(
            name=name_match.group(1) if name_match else "robot",
            robot_description=robot_description,
            format="mjcf",
            joint_names=joint_names,
            nonfixed_joint_names=nonfixed_joint_names,
            actuated_joint_names=actuated_joint_names,
            joint_types=np.asarray(joint_types, dtype=np.int32),
            joint_axes=np.stack(joint_axes) if joint_axes else np.empty((0, 3)),
            joint_twists=np.stack(joint_twists) if joint_twists else np.empty((0, 6)),
            actuated_joint_indices=np.asarray(actuated_joint_indices, dtype=np.int32),
            parent_joint_indices=np.asarray(parent_joint_indices, dtype=np.int32),
            parent_joint_transforms=np.stack(parent_joint_transforms) if parent_joint_transforms else np.empty((0, 7)),
            mimic_actuated_joint_indices=np.asarray(mimic_actuated_joint_indices, dtype=np.int32),
            mimic_multipliers=np.asarray(mimic_multipliers),
            mimic_offsets=np.asarray(mimic_offsets),
            joint_limits=np.asarray(joint_limits) if joint_limits else np.empty((0, 2)),
            joint_velocity_limits=np.asarray(joint_velocity_limits),
            actuated_joint_limits=np.asarray(actuated_joint_limits) if actuated_joint_limits else np.empty((0, 2)),
            actuated_joint_velocity_limits=np.asarray(actuated_joint_velocity_limits),
            link_names=link_names,
            link_parent_joint_indices=np.asarray(link_parent_joint_indices, dtype=np.int32),
            base_link_name=base_link_name,
            ee_link_names=ee_link_names,
            link_visual_geometries=link_visual_geometries,
            link_collision_geometries=link_collision_geometries,
            local_collision_sphere_centers=local_collision_sphere_centers,
            collision_sphere_radii=collision_sphere_radii,
            collision_spheres_link_indices=collision_spheres_link_indices,
            self_collision_link_buffers=self_collision_link_buffers,
            loaded_link_capsule_endpoints=loaded_link_capsule_endpoints,
            loaded_link_capsule_radii=loaded_link_capsule_radii,
            self_collision_ignored_pairs=np.empty((0, 2), dtype=np.int32),
            link_masses=np.asarray(link_masses, dtype=np.float64),
            link_local_com_positions=np.stack(link_local_com_positions),
        )

    # --- counts and default configurations ---
    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def num_actuated_joints(self) -> int:
        return len(self.actuated_joint_names)

    @property
    def num_nonfixed_joints(self) -> int:
        return len(self.nonfixed_joint_names)

    @property
    def num_links(self) -> int:
        return len(self.link_names)

    @property
    def zero_q(self) -> Float[np.ndarray, "num_actuated_joints"]:
        return np.zeros(self.num_actuated_joints, dtype=np.float32)

    @property
    def midrange_q(self) -> Float[np.ndarray, "num_actuated_joints"]:
        return 0.5 * (self.actuated_joint_limits[:, 0] + self.actuated_joint_limits[:, 1])

    # --- derived joint properties ---
    @cached_property
    def parent_joint_transforms_matrix(self) -> Float[np.ndarray, "num_joints 4 4"]:
        """Transforms of parent joints as 4x4 matrices."""
        return rot_tl_to_tf_mat(
            quaternion_to_matrix(self.parent_joint_transforms[:, 3:]), self.parent_joint_transforms[:, :3]
        )

    @cached_property
    def has_mimic_joints(self) -> bool:
        return bool(np.any(self.mimic_actuated_joint_indices >= 0))

    @cached_property
    def mimic_joint_names(self) -> List[str]:
        """Names of mimic joints."""
        return [
            self.joint_names[joint_idx]
            for joint_idx in range(self.num_joints)
            if self.mimic_actuated_joint_indices[joint_idx] >= 0
        ]

    @cached_property
    def total_mass(self) -> float:
        return float(np.sum(self.link_masses))

    # --- collision geometry ---
    @cached_property
    def has_collision_spheres(self) -> bool:
        return len(self.local_collision_sphere_centers) > 0 and len(self.collision_spheres_link_indices) > 0

    @property
    def local_collision_sphere_radii(self) -> Float[np.ndarray, "num_collision_spheres"]:
        return self.collision_sphere_radii

    @cached_property
    def _link_bounding_spheres(self) -> Tuple[Float[np.ndarray, "num_links 3"], Float[np.ndarray, "num_links"]]:
        """Per-link bounding spheres enclosing each link's collision spheres (zeros for empty links)."""
        return _compute_link_bounding_spheres(
            self.local_collision_sphere_centers,
            self.collision_sphere_radii,
            self.collision_spheres_link_indices,
            self.num_links,
        )

    @cached_property
    def local_link_bounding_sphere_centers(self) -> Float[np.ndarray, "num_links 3"]:
        """Per-link bounding sphere centers in link-local coordinates (broad-phase filter)."""
        return self._link_bounding_spheres[0]

    @cached_property
    def link_bounding_sphere_radii(self) -> Float[np.ndarray, "num_links"]:
        """Per-link bounding sphere radii (broad-phase filter)."""
        return self._link_bounding_spheres[1]

    @cached_property
    def _link_capsules(self) -> Tuple[Float[np.ndarray, "num_links 2 3"], Float[np.ndarray, "num_links"]]:
        """Fit one capsule per link to its collision mesh (lazy); baked capsules bypass this entirely."""
        endpoints = np.zeros((self.num_links, 2, 3), dtype=np.float32)
        radii = np.zeros(self.num_links, dtype=np.float32)
        for link_idx, link_name in enumerate(self.link_names):
            scene = self.link_collision_geometries.get(link_name)
            if scene is None or len(scene.geometry) == 0:
                continue
            mesh = scene.dump(concatenate=True)
            if len(mesh.vertices) == 0:
                continue
            cylinder = trimesh.bounds.minimum_cylinder(mesh)
            transform = np.asarray(cylinder["transform"], dtype=np.float64)
            center, axis = transform[:3, 3], transform[:3, 2]
            half = 0.5 * float(cylinder["height"])
            endpoints[link_idx, 0] = (center - axis * half).astype(np.float32)
            endpoints[link_idx, 1] = (center + axis * half).astype(np.float32)
            radii[link_idx] = float(cylinder["radius"])
        return endpoints, radii

    @cached_property
    def local_link_capsule_endpoints(self) -> Float[np.ndarray, "num_links 2 3"]:
        """Per-link capsule segment endpoints (a, b) in link-local coordinates; zeros for links without a capsule."""
        if self.loaded_link_capsule_endpoints is not None:
            return self.loaded_link_capsule_endpoints
        return self._link_capsules[0]

    @cached_property
    def link_capsule_radii(self) -> Float[np.ndarray, "num_links"]:
        """Per-link capsule radii; 0.0 for links without a capsule."""
        if self.loaded_link_capsule_radii is not None:
            return self.loaded_link_capsule_radii
        return self._link_capsules[1]

    @cached_property
    def has_link_capsules(self) -> bool:
        return bool(np.any(self.link_capsule_radii > 0.0))

    # --- kinematic structure ---
    @cached_property
    def topological_order_joint_indices(self) -> Int[np.ndarray, "num_joints"]:
        dependencies = [[] for _ in range(self.num_joints)]
        in_degree = [0] * self.num_joints

        for joint_idx, parent_idx in enumerate(self.parent_joint_indices):
            if parent_idx >= 0:
                dependencies[parent_idx].append(joint_idx)
                in_degree[joint_idx] += 1
        # no mimic edges: a mimic joint reads its actuated joint's q directly, not that joint's pose

        topological_order = []
        queue = [joint_idx for joint_idx in range(self.num_joints) if in_degree[joint_idx] == 0]

        while queue:
            joint_idx = queue.pop(0)
            topological_order.append(joint_idx)
            for dependent_idx in dependencies[joint_idx]:
                in_degree[dependent_idx] -= 1
                if in_degree[dependent_idx] == 0:
                    queue.append(dependent_idx)

        if len(topological_order) != self.num_joints:
            raise ValueError("Circular dependency detected in joint relationships")

        return np.asarray(topological_order, dtype=np.int32)

    @cached_property
    def kinematic_tree_str(self) -> str:
        joint_to_child_link_indices = {pj: li for li, pj in enumerate(self.link_parent_joint_indices) if pj >= 0}
        base_links = [li for li, pj in enumerate(self.link_parent_joint_indices) if pj < 0]
        link_to_child_joint_indices = defaultdict(list)
        for j in range(self.num_joints):
            pj = int(self.parent_joint_indices[j])
            if pj >= 0:
                parent_link = joint_to_child_link_indices.get(pj)
            else:
                parent_link = base_links[0] if base_links else None
            if parent_link is None:
                continue
            link_to_child_joint_indices[parent_link].append(j)

        def _joint_type_str(twist: Float[np.ndarray, "6"]) -> str:
            if np.allclose(twist, 0.0):
                return "F"
            elif np.any(twist[3:] != 0):
                return "R"
            elif np.any(twist[:3] != 0):
                return "P"
            return "?"

        def _chain_str(link_idx: int, prefix: str = "", is_last: bool = True) -> str:
            link_name = self.link_names[link_idx]
            parent_joint_idx = int(self.link_parent_joint_indices[link_idx])
            current_prefix = "└──" if is_last else "├──"
            next_prefix = prefix + ("  " if is_last else "│ ")
            if parent_joint_idx >= 0:
                joint_name = self.joint_names[parent_joint_idx]
                joint_type = _joint_type_str(self.joint_twists[parent_joint_idx])
                chain = f"{prefix}{current_prefix} {link_name} ({joint_name}, {joint_type})\n"
            else:
                chain = f"{prefix}{current_prefix} {link_name}\n"
            for i, cj in enumerate(link_to_child_joint_indices.get(link_idx, [])):
                is_last_child = i == len(link_to_child_joint_indices[link_idx]) - 1
                child_link_idx = joint_to_child_link_indices.get(cj)
                if child_link_idx is None:
                    continue
                chain += _chain_str(child_link_idx, next_prefix, is_last_child)
            return chain

        kin_tree_str = ""
        for i, base_link in enumerate(base_links):
            is_last = i == len(base_links) - 1
            kin_tree_str += _chain_str(base_link, "", is_last)
        return kin_tree_str

    @cached_property
    def joints_to_actuated_mapping(self) -> Float[np.ndarray, "num_joints num_actuated_joints"]:
        """Mapping from all joints to actuated joints."""
        mapping = np.zeros((self.num_joints, self.num_actuated_joints), dtype=np.float32)
        for joint_idx in range(self.num_joints):
            actuated_idx = self.actuated_joint_indices[joint_idx]
            mimic_actuated_idx = self.mimic_actuated_joint_indices[joint_idx]
            if actuated_idx >= 0:
                mapping[joint_idx, actuated_idx] = 1.0
            elif mimic_actuated_idx >= 0:
                multiplier = self.mimic_multipliers[joint_idx]
                mapping[joint_idx, mimic_actuated_idx] = multiplier
        return mapping

    @cached_property
    def link_ancestor_joints_mask(self) -> Bool[np.ndarray, "num_links num_joints"]:
        """Boolean mask for ancestor joints of links."""
        mask = np.zeros((self.num_links, self.num_joints), dtype=bool)
        for link_idx in range(self.num_links):
            j = int(self.link_parent_joint_indices[link_idx])
            while j != -1:
                mask[link_idx, j] = True
                j = int(self.parent_joint_indices[j])
        return mask

    @cached_property
    def link_ancestor_links_mask(self) -> Bool[np.ndarray, "num_links num_links"]:
        """Boolean mask for ancestor links of links."""
        # ancestor link iff its parent joint is an ancestor joint; clip(-1 -> 0) for base, column overwritten below
        pj = self.link_parent_joint_indices.clip(min=0)
        mask = self.link_ancestor_joints_mask[:, pj]
        # base link (no parent joint) is ancestor of every non-base link
        mask[:, self.link_names.index(self.base_link_name)] = np.any(self.link_ancestor_joints_mask, axis=1)
        np.fill_diagonal(mask, False)  # a link is not its own ancestor
        return mask

    # --- meshes and representation ---
    def get_link_mesh(
        self, link_name: str, mode: Literal["visual", "collision"] = "collision", return_mesh: bool = True
    ) -> Union[trimesh.Scene, trimesh.Trimesh]:
        """Link geometry: a single concatenated mesh, or the raw `trimesh.Scene` when `return_mesh=False`."""
        if not self.link_visual_geometries or not self.link_collision_geometries:
            raise ValueError("Meshes are not loaded. Please set load_meshes=True when loading the robot.")
        scene = (
            self.link_visual_geometries[link_name] if mode == "visual" else self.link_collision_geometries[link_name]
        )
        return scene.to_mesh() if return_mesh else scene

    def get_link_meshes(
        self, mode: Literal["visual", "collision"] = "collision", return_mesh: bool = True
    ) -> Dict[str, Union[trimesh.Scene, trimesh.Trimesh]]:
        """
        Per-link geometry: concatenated meshes or `trimesh.Scene`s when `return_mesh=False` (needs `load_meshes=True`).

        Example:
            >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
            >>> spec = RobotSpec.parse(load_robot_description("panda_description"), load_meshes=True)
            >>> "panda_hand" in spec.get_link_meshes()
            True
        """
        geometries = self.link_visual_geometries if mode == "visual" else self.link_collision_geometries
        return {
            link_name: scene.to_mesh() if return_mesh else scene
            for link_name, scene in geometries.items()
            if len(scene.geometry) > 0
        }

    # --- hashing and equality ---
    @cached_property
    def _spec_hash(self) -> int:
        hasher = hashlib.sha256()
        hasher.update(self.robot_description.encode())
        # hash kept structure too: a pruned sub-chain shares the description and would alias spec-keyed caches
        hasher.update("\x00".join(self.joint_names).encode())
        hasher.update("\x00".join(self.link_names).encode())
        hasher.update(self.base_link_name.encode())
        hasher.update("\x00".join(self.ee_link_names).encode())
        hasher.update(self.local_collision_sphere_centers.tobytes())
        hasher.update(self.collision_sphere_radii.tobytes())
        hasher.update(self.collision_spheres_link_indices.tobytes())
        return int(hasher.hexdigest(), 16)

    def __hash__(self) -> int:
        return self._spec_hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RobotSpec):
            return False
        return self._spec_hash == other._spec_hash

    def __repr__(self) -> str:
        return (
            f"RobotSpec(name={self.name}, num_actuated_joints={self.num_actuated_joints}, num_links={self.num_links})\n"
            + self.kinematic_tree_str
        )

    def __str__(self) -> str:
        return self.__repr__()

    # --- device tensors and meshes ---
    def get_tensors(self, device: Optional[str] = None) -> "RobotSpecTensors":
        """
        Cached Warp spec tensors for `device` (build on miss).

        Example:
            >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
            >>> spec = RobotSpec.parse(load_robot_description("panda_description"))
            >>> spec.get_tensors() is spec.get_tensors()
            True
        """
        return _get_tensors(self, device)

    def get_tensors_torch(self, device: "Optional[Union[str, torch.device]]" = None) -> "TorchRobotSpecTensors":
        """
        Cached torch spec tensors for `device` (build on miss).

        Example:
            >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
            >>> spec = RobotSpec.parse(load_robot_description("panda_description"))
            >>> spec.get_tensors_torch() is spec.get_tensors_torch()
            True
        """
        return _get_tensors_torch(self, device)


@lru_cache(maxsize=128)
def _get_tensors(spec: RobotSpec, device: Optional[str]) -> "RobotSpecTensors":
    from robokit.robo.robot_spec_tensors import RobotSpecTensors

    return RobotSpecTensors(spec=spec, device=device)


@lru_cache(maxsize=128)
def _get_tensors_torch(spec: RobotSpec, device: "Optional[Union[str, torch.device]]") -> "TorchRobotSpecTensors":
    from robokit.robo.robot_spec_tensors import TorchRobotSpecTensors

    return TorchRobotSpecTensors(spec=spec, device=device)


def _compute_link_bounding_spheres(
    sphere_centers: Float[np.ndarray, "n 3"],
    sphere_radii: Float[np.ndarray, "n"],
    sphere_link_indices: Int[np.ndarray, "n"],
    num_links: int,
) -> Tuple[Float[np.ndarray, "num_links 3"], Float[np.ndarray, "num_links"]]:
    """Per-link bounding sphere over the link's collision spheres (Ritter 2-pass, conservative); zero if none."""
    bound_centers = np.zeros((num_links, 3), dtype=np.float32)
    bound_radii = np.zeros(num_links, dtype=np.float32)
    for link_idx in range(num_links):
        mask = sphere_link_indices == link_idx
        if not np.any(mask):
            continue
        centers = sphere_centers[mask].astype(np.float64)
        radii = sphere_radii[mask].astype(np.float64)
        n = len(radii)
        if n == 1:
            bound_centers[link_idx] = centers[0].astype(np.float32)
            bound_radii[link_idx] = np.float32(radii[0])
            continue
        dists_from_0 = np.linalg.norm(centers - centers[0], axis=1) + radii
        i1 = int(np.argmax(dists_from_0))
        dists_from_i1 = np.linalg.norm(centers - centers[i1], axis=1) + radii
        i2 = int(np.argmax(dists_from_i1))
        c1, r1, c2, r2 = centers[i1], float(radii[i1]), centers[i2], float(radii[i2])
        diff = c2 - c1
        d12 = float(np.linalg.norm(diff))
        if r1 >= d12 + r2:
            C, R = c1.copy(), r1
        elif r2 >= d12 + r1:
            C, R = c2.copy(), r2
        else:
            R = (d12 + r1 + r2) / 2.0
            C = c1 + diff * ((R - r1) / d12)
        for k in range(n):
            diff = centers[k] - C
            d = float(np.linalg.norm(diff))
            overhang = d + float(radii[k]) - R
            if overhang > 1e-12:
                new_R = R + overhang / 2.0
                if d > 1e-12:
                    C = C + diff * ((new_R - R) / d)
                R = new_R
        bound_centers[link_idx] = C.astype(np.float32)
        bound_radii[link_idx] = np.float32(R)
    return bound_centers, bound_radii


def load_link_points(
    link_names: Sequence[str],
    path: Union[str, Path],
    keep_link_names: Optional[Sequence[str]] = None,
) -> Tuple[Float[np.ndarray, "num_points 3"], Int[np.ndarray, "num_points"]]:
    """Load link-attached contact/keypoint JSON into flat points and link indices, filtered by `keep_link_names`.

    >>> import json, tempfile
    >>> path = Path(tempfile.mkdtemp()) / "points.json"
    >>> _ = path.write_text(json.dumps({"thumb": [[0.0, 0.0, 0.01]], "index": [[0.0, 0.01, 0.0], [0.0, 0.02, 0.0]]}))
    >>> points, link_indices = load_link_points(["palm", "thumb", "index"], path)
    >>> points.shape, link_indices.tolist()
    ((3, 3), [1, 2, 2])
    """
    with open(path, "r") as f:
        link_points = json.load(f)
    keep = set(keep_link_names) if keep_link_names is not None else None
    points, indices = [], []
    for link_name, link_pts in link_points.items():
        if link_name not in link_names or (keep is not None and link_name not in keep):
            continue
        link_idx = link_names.index(link_name)
        points.extend(link_pts)
        indices.extend([link_idx] * len(link_pts))
    if not points:
        raise ValueError(f"No link points found in {path} for the given link names")
    return np.asarray(points, dtype=np.float32), np.asarray(indices, dtype=np.int32)
