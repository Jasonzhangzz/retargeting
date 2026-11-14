"""I2RT YAM flexible_4310 gripper assets (MIT). The YAM arm ships several gripper modules; this is one."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import GripperCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/end_effectors/yam_gripper/**",
        "robots/collision_spheres/yam_gripper/**",
        "robots/contact_points/yam_gripper/**",
        "robots/self_collision/yam_gripper/**",
        "robots/self_collision/yam_gripper_mj/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/end_effectors/yam_gripper/flexible_4310.urdf"
MJCF_PATH: Path = _DIR / "robots/robot_description/end_effectors/yam_gripper/flexible_4310.xml"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/yam_gripper/collision_spheres.yaml"
CONTACT_POINTS_PATH: Path = _DIR / "robots/contact_points/yam_gripper/contact_points.json"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/yam_gripper/ignore.sphere.yml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/yam_gripper/ignore.capsule.yml"
MJ_CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/yam_gripper_mj/ignore.capsule.yml"
MOUNT_LINK_NAME: str = "base"
EE_COORD_SPEC = GripperCoordinateSpec(outward_axis="z", opening_axis="x")
MOUNT_COORD_SPEC = EE_COORD_SPEC.mount_spec
REST_Q: Dict[str, float] = {"tip1": 0.0}
MJCF_MOUNT_LINK_NAME: str = "gripper"
MJCF_EE_COORD_SPEC = GripperCoordinateSpec(outward_axis="-z", opening_axis="y")
MJCF_MOUNT_COORD_SPEC = MJCF_EE_COORD_SPEC.mount_spec
MJCF_REST_Q: Dict[str, float] = {"joint8": 0.0}
