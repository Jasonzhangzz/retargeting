"""Flexiv Rizon4 arm assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import MountCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/arms/rizon4/**",
        "robots/collision_spheres/rizon4/**",
        "robots/self_collision/rizon4/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/arms/rizon4/rizon4.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/rizon4/collision_spheres.yaml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/rizon4/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/rizon4/ignore.sphere.yml"
MOUNT_LINK_NAME: str = "flange"
MOUNT_COORD_SPEC = MountCoordinateSpec(outward_axis="z", right_axis="-y")
REST_Q: Dict[str, float] = {
    **dict.fromkeys(("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"), 0.0),
    "joint2": -0.69813,
    "joint4": 1.570796,
    "joint6": 0.69813,
}
