"""Booster K1 assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import MountCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/humanoids/booster_k1/**",
        "robots/collision_spheres/booster_k1/**",
        "robots/self_collision/booster_k1/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/humanoids/booster_k1/K1_serial.urdf"
COLLISION_SPHERE_FULL_BODY_PATH: Path = _DIR / "robots/collision_spheres/booster_k1/full_body.yaml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/booster_k1/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/booster_k1/ignore.sphere.yml"
MOUNT_LINK_NAME_RIGHT: str = "right_hand_link"
MOUNT_LINK_NAME_LEFT: str = "left_hand_link"
MOUNT_COORD_SPEC_RIGHT = MountCoordinateSpec(outward_axis="-y", right_axis="-x", mount_point_xyz=(0.0, -0.228, 0.0))
MOUNT_COORD_SPEC_LEFT = MountCoordinateSpec(outward_axis="y", right_axis="x", mount_point_xyz=(0.0, 0.228, 0.0))
REST_Q: Dict[str, float] = {
    "AAHead_yaw": 0.0,
    "Head_pitch": 0.0,
    "ALeft_Shoulder_Pitch": 0.0,
    "Left_Shoulder_Roll": 0.0,
    "Left_Elbow_Pitch": 0.0,
    "Left_Elbow_Yaw": 0.0,
    "ARight_Shoulder_Pitch": 0.0,
    "Right_Shoulder_Roll": 0.0,
    "Right_Elbow_Pitch": 0.0,
    "Right_Elbow_Yaw": 0.0,
    "Left_Hip_Pitch": 0.0,
    "Left_Hip_Roll": 0.0,
    "Left_Hip_Yaw": 0.0,
    "Left_Knee_Pitch": 0.0,
    "Left_Ankle_Pitch": 0.0,
    "Left_Ankle_Roll": 0.0,
    "Right_Hip_Pitch": 0.0,
    "Right_Hip_Roll": 0.0,
    "Right_Hip_Yaw": 0.0,
    "Right_Knee_Pitch": 0.0,
    "Right_Ankle_Pitch": 0.0,
    "Right_Ankle_Roll": 0.0,
}
