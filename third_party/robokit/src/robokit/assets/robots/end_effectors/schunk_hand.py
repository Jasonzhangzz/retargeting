"""Schunk SVH Hand assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import HandCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/end_effectors/schunk_hand/**",
        "robots/collision_spheres/schunk_hand_right/**",
        "robots/contact_points/schunk_hand_right/**",
        "robots/self_collision/schunk_hand_right/**",
    ]
)
URDF_PATH_RIGHT: Path = _DIR / "robots/robot_description/end_effectors/schunk_hand/schunk_svh_hand_right.urdf"
URDF_PATH_LEFT: Path = _DIR / "robots/robot_description/end_effectors/schunk_hand/schunk_svh_hand_left.urdf"
URDF_GLB_PATH_RIGHT: Path = _DIR / "robots/robot_description/end_effectors/schunk_hand/schunk_svh_hand_right_glb.urdf"
URDF_GLB_FLOATING_BASE_NO_MIMIC_PATH_RIGHT: Path = (
    _DIR / "robots/robot_description/end_effectors/schunk_hand/schunk_hand_right_glb_floating_base_no_mimic.urdf"
)
COLLISION_SPHERE_PATH_RIGHT: Path = _DIR / "robots/collision_spheres/schunk_hand_right/collision_spheres.yaml"
CONTACT_POINTS_PATH_RIGHT: Path = _DIR / "robots/contact_points/schunk_hand_right/contact_points.json"
SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT: Path = _DIR / "robots/self_collision/schunk_hand_right/ignore.sphere.yml"
CAPSULE_SELF_COLLISION_IGNORE_PATH_RIGHT: Path = _DIR / "robots/self_collision/schunk_hand_right/ignore.capsule.yml"
MOUNT_LINK_NAME_RIGHT: str = "right_hand_base_link"
MOUNT_LINK_NAME_LEFT: str = "left_hand_base_link"
EE_COORD_SPEC_RIGHT = HandCoordinateSpec(palm_forward_axis="y", four_fingers_up_axis="z")
EE_COORD_SPEC_LEFT = HandCoordinateSpec(palm_forward_axis="y", four_fingers_up_axis="z")
MOUNT_COORD_SPEC_RIGHT = EE_COORD_SPEC_RIGHT.mount_spec
MOUNT_COORD_SPEC_LEFT = EE_COORD_SPEC_LEFT.mount_spec
REST_Q_RIGHT: Dict[str, float] = dict.fromkeys(
    (
        "right_hand_Thumb_Flexion",
        "right_hand_Thumb_Opposition",
        "right_hand_Index_Finger_Distal",
        "right_hand_Index_Finger_Proximal",
        "right_hand_Middle_Finger_Proximal",
        "right_hand_Middle_Finger_Distal",
        "right_hand_Ring_Finger",
        "right_hand_Pinky",
        "right_hand_Finger_Spread",
    ),
    0.0,
)
REST_Q_LEFT: Dict[str, float] = {name.replace("right_", "left_", 1): value for name, value in REST_Q_RIGHT.items()}
