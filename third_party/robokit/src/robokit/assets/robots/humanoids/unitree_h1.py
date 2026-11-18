"""Unitree H1 assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import MountCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/humanoids/unitree_h1/**",
        "robots/collision_spheres/unitree_h1/**",
        "robots/self_collision/unitree_h1/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/humanoids/unitree_h1/unitree_h1.urdf"
COLLISION_SPHERE_FULL_BODY_PATH: Path = _DIR / "robots/collision_spheres/unitree_h1/full_body.yaml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/unitree_h1/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/unitree_h1/ignore.sphere.yml"
MOUNT_LINK_NAME_RIGHT: str = "right_elbow_link"
MOUNT_LINK_NAME_LEFT: str = "left_elbow_link"
MOUNT_COORD_SPEC_RIGHT = MountCoordinateSpec(outward_axis="x", right_axis="-z", mount_point_xyz=(0.27, 0.0, 0.0))
MOUNT_COORD_SPEC_LEFT = MountCoordinateSpec(outward_axis="x", right_axis="z", mount_point_xyz=(0.27, 0.0, 0.0))
REST_Q: Dict[str, float] = {
    "left_hip_yaw_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_pitch_joint": 0.0,
    "left_knee_joint": 0.0,
    "left_ankle_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_pitch_joint": 0.0,
    "right_knee_joint": 0.0,
    "right_ankle_joint": 0.0,
    "torso_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.0,
}
