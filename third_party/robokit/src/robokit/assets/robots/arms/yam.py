"""YAM arm assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import MountCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/arms/yam/**",
        "robots/collision_spheres/yam/**",
        "robots/self_collision/yam/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/arms/yam/yam.urdf"
_MJ_DIR = "robots/robot_mj_description/arms/yam"
MJCF_PATH: Path = fetch([f"{_MJ_DIR}/**"]) / f"{_MJ_DIR}/yam.xml"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/yam/collision_spheres.yaml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/yam/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/yam/ignore.sphere.yml"
MOUNT_LINK_NAME: str = "link_6"
MOUNT_COORD_SPEC = MountCoordinateSpec(outward_axis="z", right_axis="-x")
REST_Q: Dict[str, float] = {
    **dict.fromkeys(("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"), 0.0),
    "joint2": 1.047,
    "joint3": 1.047,
}
