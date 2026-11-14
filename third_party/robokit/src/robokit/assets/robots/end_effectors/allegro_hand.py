"""Allegro Hand assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import HandCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/end_effectors/allegro_hand/**",
        "robots/collision_spheres/allegro_hand_right/**",
        "robots/contact_points/allegro_hand_right/**",
        "robots/self_collision/allegro_hand_right/**",
    ]
)
URDF_PATH_RIGHT: Path = _DIR / "robots/robot_description/end_effectors/allegro_hand/allegro_hand_right.urdf"
URDF_PATH_LEFT: Path = _DIR / "robots/robot_description/end_effectors/allegro_hand/allegro_hand_left.urdf"
URDF_GLB_PATH_RIGHT: Path = _DIR / "robots/robot_description/end_effectors/allegro_hand/allegro_hand_right_glb.urdf"
URDF_GLB_FLOATING_BASE_PATH_RIGHT: Path = (
    _DIR / "robots/robot_description/end_effectors/allegro_hand/allegro_hand_right_glb_floating_base.urdf"
)
COLLISION_SPHERE_PATH_RIGHT: Path = _DIR / "robots/collision_spheres/allegro_hand_right/collision_spheres.yaml"
CONTACT_POINTS_PATH_RIGHT: Path = _DIR / "robots/contact_points/allegro_hand_right/contact_points.json"
CAPSULE_SELF_COLLISION_IGNORE_PATH_RIGHT: Path = _DIR / "robots/self_collision/allegro_hand_right/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT: Path = _DIR / "robots/self_collision/allegro_hand_right/ignore.sphere.yml"
MOUNT_LINK_NAME_RIGHT: str = "base_link"
MOUNT_LINK_NAME_LEFT: str = "base_link"
EE_COORD_SPEC_RIGHT = HandCoordinateSpec(palm_forward_axis="x", four_fingers_up_axis="z")
EE_COORD_SPEC_LEFT = HandCoordinateSpec(palm_forward_axis="x", four_fingers_up_axis="z")
MOUNT_COORD_SPEC_RIGHT = EE_COORD_SPEC_RIGHT.mount_spec
MOUNT_COORD_SPEC_LEFT = EE_COORD_SPEC_LEFT.mount_spec
REST_Q_RIGHT: Dict[str, float] = dict.fromkeys(
    (
        "joint_0.0",
        "joint_1.0",
        "joint_2.0",
        "joint_3.0",
        "joint_4.0",
        "joint_5.0",
        "joint_6.0",
        "joint_7.0",
        "joint_8.0",
        "joint_9.0",
        "joint_10.0",
        "joint_11.0",
        "joint_12.0",
        "joint_13.0",
        "joint_14.0",
        "joint_15.0",
    ),
    0.0,
)
REST_Q_RIGHT["joint_12.0"] = 0.263
REST_Q_LEFT: Dict[str, float] = dict(REST_Q_RIGHT)
