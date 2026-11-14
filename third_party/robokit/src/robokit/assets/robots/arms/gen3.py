"""Kinova Gen3 arm assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import MountCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/arms/gen3/**",
        "robots/collision_spheres/gen3/**",
        "robots/self_collision/gen3/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/arms/gen3/gen3.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/gen3/collision_spheres.yaml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/gen3/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/gen3/ignore.sphere.yml"
MOUNT_LINK_NAME: str = "end_effector_link"
MOUNT_COORD_SPEC = MountCoordinateSpec(outward_axis="z", right_axis="-x")
REST_Q: Dict[str, float] = {
    **dict.fromkeys(("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"), 0.0),
    "joint_2": 0.26179939,
    "joint_3": 3.14159265,
    "joint_4": -2.26892803,
    "joint_6": 0.95993109,
    "joint_7": 1.57079633,
}
