"""Inspire Hand assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import HandCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/end_effectors/inspire_hand/**",
        "robots/collision_spheres/inspire_hand_right/**",
        "robots/collision_spheres/inspire_hand_left/**",
        "robots/contact_points/inspire_hand_right/**",
        "robots/contact_avoid_points/inspire_hand_right/**",
        "robots/self_collision/inspire_hand_right/**",
        "robots/self_collision/inspire_hand_left/**",
    ]
)
URDF_PATH_RIGHT: Path = _DIR / "robots/robot_description/end_effectors/inspire_hand/inspire_hand_right.urdf"
URDF_PATH_LEFT: Path = _DIR / "robots/robot_description/end_effectors/inspire_hand/inspire_hand_left.urdf"
COLLISION_SPHERE_PATH_RIGHT: Path = _DIR / "robots/collision_spheres/inspire_hand_right/collision_spheres.yaml"
COLLISION_SPHERE_PATH_LEFT: Path = _DIR / "robots/collision_spheres/inspire_hand_left/collision_spheres.yaml"
CONTACT_POINTS_PATH_RIGHT: Path = _DIR / "robots/contact_points/inspire_hand_right/contact_points.json"
CONTACT_AVOID_POINTS_PATH_RIGHT: Path = (
    _DIR / "robots/contact_avoid_points/inspire_hand_right/contact_avoid_points.json"
)
CAPSULE_SELF_COLLISION_IGNORE_PATH_RIGHT: Path = _DIR / "robots/self_collision/inspire_hand_right/ignore.capsule.yml"
URDF_NO_MIMIC_PATH_RIGHT: Path = (
    _DIR / "robots/robot_description/end_effectors/inspire_hand/inspire_hand_right_no_mimic.urdf"
)
URDF_FLOATING_BASE_NO_MIMIC_PATH_RIGHT: Path = (
    _DIR / "robots/robot_description/end_effectors/inspire_hand/inspire_hand_right_floating_base_no_mimic.urdf"
)
SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT: Path = _DIR / "robots/self_collision/inspire_hand_right/ignore.sphere.yml"
CAPSULE_SELF_COLLISION_IGNORE_PATH_LEFT: Path = _DIR / "robots/self_collision/inspire_hand_left/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH_LEFT: Path = _DIR / "robots/self_collision/inspire_hand_left/ignore.sphere.yml"
MOUNT_LINK_NAME_RIGHT: str = "base"
MOUNT_LINK_NAME_LEFT: str = "base"
EE_COORD_SPEC_RIGHT = HandCoordinateSpec(
    palm_forward_axis="x", four_fingers_up_axis="z", mount_point_xyz=(0.0, 0.0, 0.004)
)
EE_COORD_SPEC_LEFT = HandCoordinateSpec(
    palm_forward_axis="x", four_fingers_up_axis="z", mount_point_xyz=(0.0, 0.0, 0.004)
)
MOUNT_COORD_SPEC_RIGHT = EE_COORD_SPEC_RIGHT.mount_spec
MOUNT_COORD_SPEC_LEFT = EE_COORD_SPEC_LEFT.mount_spec
REST_Q_RIGHT: Dict[str, float] = dict.fromkeys(
    (
        "thumb_proximal_yaw_joint",
        "thumb_proximal_pitch_joint",
        "index_proximal_joint",
        "middle_proximal_joint",
        "ring_proximal_joint",
        "pinky_proximal_joint",
    ),
    0.0,
)
REST_Q_RIGHT["thumb_proximal_pitch_joint"] = 0.14
REST_Q_LEFT: Dict[str, float] = dict.fromkeys(REST_Q_RIGHT, 0.0)
