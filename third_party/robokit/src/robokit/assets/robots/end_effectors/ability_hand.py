"""PSYONIC Ability Hand assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import HandCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/end_effectors/ability_hand/**",
        "robots/collision_spheres/ability_hand_right/**",
        "robots/contact_points/ability_hand_right/**",
        "robots/contact_avoid_points/ability_hand_right/**",
        "robots/self_collision/ability_hand_right/**",
    ]
)
URDF_PATH_RIGHT: Path = _DIR / "robots/robot_description/end_effectors/ability_hand/ability_hand_right.urdf"
URDF_PATH_LEFT: Path = _DIR / "robots/robot_description/end_effectors/ability_hand/ability_hand_left.urdf"
URDF_GLB_PATH_RIGHT: Path = _DIR / "robots/robot_description/end_effectors/ability_hand/ability_hand_right_glb.urdf"
COLLISION_SPHERE_PATH_RIGHT: Path = _DIR / "robots/collision_spheres/ability_hand_right/collision_spheres.yaml"
CONTACT_POINTS_PATH_RIGHT: Path = _DIR / "robots/contact_points/ability_hand_right/contact_points.json"
CONTACT_AVOID_POINTS_PATH_RIGHT: Path = (
    _DIR / "robots/contact_avoid_points/ability_hand_right/contact_avoid_points.json"
)
URDF_GLB_NO_MIMIC_PATH_RIGHT: Path = (
    _DIR / "robots/robot_description/end_effectors/ability_hand/ability_hand_right_glb_no_mimic.urdf"
)
URDF_GLB_FLOATING_BASE_NO_MIMIC_PATH_RIGHT: Path = (
    _DIR / "robots/robot_description/end_effectors/ability_hand/ability_hand_right_glb_floating_base_no_mimic.urdf"
)
URDF_GLB_FLOATING_BASE_NO_MIMIC_PATH_LEFT: Path = (
    _DIR / "robots/robot_description/end_effectors/ability_hand/ability_hand_left_glb_floating_base_no_mimic.urdf"
)
CAPSULE_SELF_COLLISION_IGNORE_PATH_RIGHT: Path = _DIR / "robots/self_collision/ability_hand_right/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT: Path = _DIR / "robots/self_collision/ability_hand_right/ignore.sphere.yml"
MOUNT_LINK_NAME_RIGHT: str = "base"
MOUNT_LINK_NAME_LEFT: str = "base"
EE_COORD_SPEC_RIGHT = HandCoordinateSpec(
    palm_forward_axis="y", four_fingers_up_axis="z", mount_point_xyz=(0.0, 0.0, -0.008)
)
EE_COORD_SPEC_LEFT = HandCoordinateSpec(
    palm_forward_axis="y", four_fingers_up_axis="z", mount_point_xyz=(0.0, 0.0, -0.008)
)
MOUNT_COORD_SPEC_RIGHT = EE_COORD_SPEC_RIGHT.mount_spec
MOUNT_COORD_SPEC_LEFT = EE_COORD_SPEC_LEFT.mount_spec
REST_Q_RIGHT: Dict[str, float] = {
    "thumb_q1": 0.0,
    "thumb_q2": 0.0,
    "index_q1": 0.0,
    "middle_q1": 0.0,
    "ring_q1": 0.0,
    "pinky_q1": 0.0,
}
REST_Q_LEFT: Dict[str, float] = {
    "thumb_q1": 0.0,
    "thumb_q2": 0.0,
    "index_q1": 0.0,
    "middle_q1": 0.0,
    "ring_q1": 0.0,
    "pinky_q1": 0.0,
}
