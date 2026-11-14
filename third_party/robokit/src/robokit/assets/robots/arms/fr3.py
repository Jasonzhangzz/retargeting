"""Franka FR3 arm assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import MountCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/arms/fr3/**",
        "robots/collision_spheres/fr3/**",
        "robots/self_collision/fr3/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/arms/fr3/fr3.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/fr3/collision_spheres.yaml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/fr3/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/fr3/ignore.sphere.yml"
MOUNT_LINK_NAME: str = "fr3_link8"
MOUNT_COORD_SPEC = MountCoordinateSpec(outward_axis="z", right_axis="y")
REST_Q: Dict[str, float] = {
    **dict.fromkeys(
        ("fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4", "fr3_joint5", "fr3_joint6", "fr3_joint7"),
        0.0,
    ),
    "fr3_joint4": -1.57079,
    "fr3_joint6": 1.57079,
    "fr3_joint7": 0.0,
}
