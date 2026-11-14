"""MANO hand (end-effector URDF) assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import HandCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/end_effectors/mano_hand/**",
        "robots/collision_spheres/mano_hand_right/**",
        "robots/contact_points/mano_hand_right/**",
        "robots/contact_avoid_points/mano_hand_right/**",
        "robots/self_collision/mano_hand_right/**",
    ]
)
URDF_PATH_RIGHT: Path = _DIR / "robots/robot_description/end_effectors/mano_hand/mano_hand.urdf"
URDF_FLOATING_BASE_PATH_RIGHT: Path = (
    _DIR / "robots/robot_description/end_effectors/mano_hand/mano_hand_right_floating_base.urdf"
)
COLLISION_SPHERE_PATH_RIGHT: Path = _DIR / "robots/collision_spheres/mano_hand_right/collision_spheres.yaml"
CONTACT_POINTS_PATH_RIGHT: Path = _DIR / "robots/contact_points/mano_hand_right/contact_points.json"
CONTACT_AVOID_POINTS_PATH_RIGHT: Path = _DIR / "robots/contact_avoid_points/mano_hand_right/contact_avoid_points.json"
CAPSULE_SELF_COLLISION_IGNORE_PATH_RIGHT: Path = _DIR / "robots/self_collision/mano_hand_right/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT: Path = _DIR / "robots/self_collision/mano_hand_right/ignore.sphere.yml"
MOUNT_LINK_NAME_RIGHT: str = "link_0"
EE_COORD_SPEC_RIGHT = HandCoordinateSpec(
    palm_forward_axis="x", four_fingers_up_axis="z", mount_point_xyz=(0.0, 0.0, -0.008)
)
MOUNT_COORD_SPEC_RIGHT = EE_COORD_SPEC_RIGHT.mount_spec
REST_Q_RIGHT: Dict[str, float] = dict.fromkeys(
    (
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "joint_7",
        "joint_8",
        "joint_9",
        "joint_10",
        "joint_11",
        "joint_12",
        "joint_empty",
        "joint_13",
        "joint_14",
        "joint_15",
    ),
    0.0,
)
