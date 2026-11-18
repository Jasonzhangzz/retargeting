"""Unitree Z1 arm assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import MountCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/arms/z1/**",
        "robots/collision_spheres/z1/**",
        "robots/self_collision/z1/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/arms/z1/z1.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/z1/collision_spheres.yaml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/z1/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/z1/ignore.sphere.yml"
MOUNT_LINK_NAME: str = "link06"
MOUNT_COORD_SPEC = MountCoordinateSpec(outward_axis="x", right_axis="-y")
REST_Q: Dict[str, float] = {
    **dict.fromkeys(("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"), 0.0),
    "joint2": 0.785,
    "joint3": -0.261,
    "joint4": -0.523,
}
