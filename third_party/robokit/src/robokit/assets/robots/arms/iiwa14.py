"""KUKA iiwa 14 arm assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import MountCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/arms/iiwa14/**",
        "robots/collision_spheres/iiwa14/**",
        "robots/self_collision/iiwa14/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/arms/iiwa14/iiwa14.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/iiwa14/collision_spheres.yaml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/iiwa14/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/iiwa14/ignore.sphere.yml"
MOUNT_LINK_NAME: str = "iiwa_link_ee_kuka"
MOUNT_COORD_SPEC = MountCoordinateSpec(outward_axis="z", right_axis="-y")
REST_Q: Dict[str, float] = {
    **dict.fromkeys(
        (
            "iiwa_joint_1",
            "iiwa_joint_2",
            "iiwa_joint_3",
            "iiwa_joint_4",
            "iiwa_joint_5",
            "iiwa_joint_6",
            "iiwa_joint_7",
        ),
        0.0,
    ),
    "iiwa_joint_2": 0.785398,
    "iiwa_joint_4": -1.5708,
}
