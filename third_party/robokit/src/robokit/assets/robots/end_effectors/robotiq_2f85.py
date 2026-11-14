"""Robotiq 2F-85 parallel gripper assets (a-price/robotiq_arg85_description, BSD)."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import GripperCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/end_effectors/robotiq_2f85/**",
        "robots/collision_spheres/robotiq_2f85/**",
        "robots/contact_points/robotiq_2f85/**",
        "robots/self_collision/robotiq_2f85/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/end_effectors/robotiq_2f85/robotiq_arg85.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/robotiq_2f85/collision_spheres.yaml"
CONTACT_POINTS_PATH: Path = _DIR / "robots/contact_points/robotiq_2f85/contact_points.json"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/robotiq_2f85/ignore.sphere.yml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/robotiq_2f85/ignore.capsule.yml"
MOUNT_LINK_NAME: str = "robotiq_85_base_link"
EE_COORD_SPEC = GripperCoordinateSpec(outward_axis="z", opening_axis="-x")
MOUNT_COORD_SPEC = EE_COORD_SPEC.mount_spec
REST_Q: Dict[str, float] = {"finger_joint": 0.0}
