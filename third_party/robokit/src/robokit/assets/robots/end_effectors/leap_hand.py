"""LEAP Hand assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import HandCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/end_effectors/leap_hand/**",
        "robots/collision_spheres/leap_hand_right/**",
        "robots/contact_points/leap_hand_right/**",
        "robots/self_collision/leap_hand_right/**",
    ]
)
URDF_PATH_RIGHT: Path = _DIR / "robots/robot_description/end_effectors/leap_hand/leap_hand_right_glb.urdf"
URDF_PATH_LEFT: Path = _DIR / "robots/robot_description/end_effectors/leap_hand/leap_hand_left_glb.urdf"
URDF_GLB_FLOATING_BASE_PATH_RIGHT: Path = (
    _DIR / "robots/robot_description/end_effectors/leap_hand/leap_hand_right_glb_floating_base.urdf"
)
COLLISION_SPHERE_PATH_RIGHT: Path = _DIR / "robots/collision_spheres/leap_hand_right/collision_spheres.yaml"
CONTACT_POINTS_PATH_RIGHT: Path = _DIR / "robots/contact_points/leap_hand_right/contact_points.json"
SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT: Path = _DIR / "robots/self_collision/leap_hand_right/ignore.sphere.yml"
CAPSULE_SELF_COLLISION_IGNORE_PATH_RIGHT: Path = _DIR / "robots/self_collision/leap_hand_right/ignore.capsule.yml"
MOUNT_LINK_NAME_RIGHT: str = "base"
MOUNT_LINK_NAME_LEFT: str = "base"
EE_COORD_SPEC_RIGHT = HandCoordinateSpec(palm_forward_axis="x", four_fingers_up_axis="z")
EE_COORD_SPEC_LEFT = HandCoordinateSpec(palm_forward_axis="x", four_fingers_up_axis="z")
MOUNT_COORD_SPEC_RIGHT = EE_COORD_SPEC_RIGHT.mount_spec
MOUNT_COORD_SPEC_LEFT = EE_COORD_SPEC_LEFT.mount_spec
REST_Q_RIGHT: Dict[str, float] = dict.fromkeys(
    ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"), 0.0
)
REST_Q_LEFT: Dict[str, float] = dict(REST_Q_RIGHT)
