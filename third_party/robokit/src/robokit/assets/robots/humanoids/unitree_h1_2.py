"""Unitree H1-2 humanoid assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import MountCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/humanoids/unitree_h1_2/**",
        "robots/collision_spheres/unitree_h1_2/**",
        "robots/self_collision/unitree_h1_2/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/humanoids/unitree_h1_2/h1_2_handless.urdf"
COLLISION_SPHERE_FULL_BODY_PATH: Path = _DIR / "robots/collision_spheres/unitree_h1_2/full_body.yaml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/unitree_h1_2/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/unitree_h1_2/ignore.sphere.yml"
MOUNT_LINK_NAME_RIGHT: str = "right_wrist_yaw_link"
MOUNT_LINK_NAME_LEFT: str = "left_wrist_yaw_link"
MOUNT_COORD_SPEC_RIGHT = MountCoordinateSpec(outward_axis="x", right_axis="-z")
MOUNT_COORD_SPEC_LEFT = MountCoordinateSpec(outward_axis="x", right_axis="z")
REST_Q: Dict[str, float] = {
    **dict.fromkeys(
        (
            "left_hip_yaw_joint",
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_yaw_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "torso_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ),
        0.0,
    ),
    "left_hip_pitch_joint": -0.16,
    "left_knee_joint": 0.36,
    "left_ankle_pitch_joint": -0.2,
    "right_hip_pitch_joint": -0.16,
    "right_knee_joint": 0.36,
    "right_ankle_pitch_joint": -0.2,
    "left_shoulder_pitch_joint": 0.4,
    "left_elbow_joint": 0.3,
    "right_shoulder_pitch_joint": 0.4,
    "right_elbow_joint": 0.3,
}
