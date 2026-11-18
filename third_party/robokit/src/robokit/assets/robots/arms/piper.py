"""AgileX PiPER arm assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import MountCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/arms/piper/**",
        "robots/collision_spheres/piper/**",
        "robots/self_collision/piper/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/arms/piper/piper.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/piper/collision_spheres.yaml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/piper/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/piper/ignore.sphere.yml"
MOUNT_LINK_NAME: str = "link6"
MOUNT_COORD_SPEC = MountCoordinateSpec(outward_axis="z", right_axis="-y")
REST_Q: Dict[str, float] = {
    **dict.fromkeys(("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"), 0.0),
    "joint2": 1.57,
    "joint3": -1.3485,
}
