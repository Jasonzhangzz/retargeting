"""Tienkung Lite assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import MountCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/humanoids/tienkung_lite/**",
        "robots/collision_spheres/tienkung_lite/**",
        "robots/self_collision/tienkung_lite/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/humanoids/tienkung_lite/tienkung_lite.urdf"
COLLISION_SPHERE_FULL_BODY_PATH: Path = _DIR / "robots/collision_spheres/tienkung_lite/full_body.yaml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/tienkung_lite/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/tienkung_lite/ignore.sphere.yml"
MOUNT_LINK_NAME_RIGHT: str = "elbow_r_link"
MOUNT_LINK_NAME_LEFT: str = "elbow_l_link"
MOUNT_COORD_SPEC_RIGHT = MountCoordinateSpec(outward_axis="-z", right_axis="-x", mount_point_xyz=(0.0, 0.0, -0.3174))
MOUNT_COORD_SPEC_LEFT = MountCoordinateSpec(outward_axis="-z", right_axis="x", mount_point_xyz=(0.0, 0.0, -0.3174))
REST_Q: Dict[str, float] = {
    "hip_roll_l_joint": 0.0,
    "hip_yaw_l_joint": 0.0,
    "hip_pitch_l_joint": 0.0,
    "knee_pitch_l_joint": 0.1745,
    "ankle_pitch_l_joint": 0.0,
    "ankle_roll_l_joint": 0.0,
    "hip_roll_r_joint": 0.0,
    "hip_yaw_r_joint": 0.0,
    "hip_pitch_r_joint": 0.0,
    "knee_pitch_r_joint": 0.1745,
    "ankle_pitch_r_joint": 0.0,
    "ankle_roll_r_joint": 0.0,
    "shoulder_pitch_l_joint": 0.0,
    "shoulder_roll_l_joint": 0.0,
    "shoulder_yaw_l_joint": 0.0,
    "elbow_l_joint": 0.0,
    "shoulder_pitch_r_joint": 0.0,
    "shoulder_roll_r_joint": 0.0,
    "shoulder_yaw_r_joint": 0.0,
    "elbow_r_joint": 0.0,
}
