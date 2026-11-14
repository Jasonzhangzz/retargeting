"""Shadow Hand assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import HandCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/end_effectors/shadow_hand/**",
        "robots/collision_spheres/shadow_hand_right/**",
        "robots/collision_spheres/shadow_hand_mjcf_right/**",
        "robots/robot_description/end_effectors/shadow_hand_mjcf/**",
        "robots/robot_mj_description/end_effectors/shadow_hand/**",
        "robots/collision_spheres/shadow_hand_mj_right/**",
        "robots/self_collision/shadow_hand_right/**",
    ]
)
URDF_PATH_RIGHT: Path = _DIR / "robots/robot_description/end_effectors/shadow_hand/shadow_hand_right.urdf"
URDF_PATH_LEFT: Path = _DIR / "robots/robot_description/end_effectors/shadow_hand/shadow_hand_left.urdf"
COLLISION_SPHERE_PATH_RIGHT: Path = _DIR / "robots/collision_spheres/shadow_hand_right/collision_spheres.yaml"
MJCF_PATH_RIGHT: Path = _DIR / "robots/robot_description/end_effectors/shadow_hand_mjcf/shadow_hand_wrist_free.xml"
MJCF_CONTACT_POINTS_PATH_RIGHT: Path = (
    _DIR / "robots/robot_description/end_effectors/shadow_hand_mjcf/contact_points.json"
)
MJCF_COLLISION_SPHERE_PATH_RIGHT: Path = _DIR / "robots/collision_spheres/shadow_hand_mjcf_right/collision_spheres.yaml"
MJCF_SAPIEN_PATH_RIGHT: Path = _DIR / "robots/robot_description/end_effectors/shadow_hand_mjcf/shadow_hand_sapien.xml"
MJCF_FLOATING_BASE_SAPIEN_PATH_RIGHT: Path = (
    _DIR / "robots/robot_description/end_effectors/shadow_hand_mjcf/shadow_hand_free_sapien.xml"
)
URDF_FLOATING_BASE_PATH_RIGHT: Path = (
    _DIR / "robots/robot_description/end_effectors/shadow_hand/shadow_hand_right_floating_base.urdf"
)
MJ_DESCRIPTION_DIR_RIGHT: Path = _DIR / "robots/robot_mj_description/end_effectors/shadow_hand"
MJ_COLLISION_SPHERE_PATH_RIGHT: Path = _DIR / "robots/collision_spheres/shadow_hand_mj_right/collision_spheres.yaml"
CAPSULE_SELF_COLLISION_IGNORE_PATH_RIGHT: Path = _DIR / "robots/self_collision/shadow_hand_right/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT: Path = _DIR / "robots/self_collision/shadow_hand_right/ignore.sphere.yml"
MOUNT_LINK_NAME_RIGHT: str = "forearm"
MOUNT_LINK_NAME_LEFT: str = "forearm"
EE_COORD_SPEC_RIGHT = HandCoordinateSpec(palm_forward_axis="-y", four_fingers_up_axis="z")
EE_COORD_SPEC_LEFT = HandCoordinateSpec(palm_forward_axis="-y", four_fingers_up_axis="z")
MOUNT_COORD_SPEC_RIGHT = EE_COORD_SPEC_RIGHT.mount_spec
MOUNT_COORD_SPEC_LEFT = EE_COORD_SPEC_LEFT.mount_spec
REST_Q_RIGHT: Dict[str, float] = dict.fromkeys(
    (
        "WRJ2",
        "WRJ1",
        "FFJ4",
        "FFJ3",
        "FFJ2",
        "FFJ1",
        "MFJ4",
        "MFJ3",
        "MFJ2",
        "MFJ1",
        "RFJ4",
        "RFJ3",
        "RFJ2",
        "RFJ1",
        "LFJ5",
        "LFJ4",
        "LFJ3",
        "LFJ2",
        "LFJ1",
        "THJ5",
        "THJ4",
        "THJ3",
        "THJ2",
        "THJ1",
    ),
    0.0,
)
REST_Q_LEFT: Dict[str, float] = dict(REST_Q_RIGHT)
