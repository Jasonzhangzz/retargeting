"""Fourier GR3 humanoid assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import MountCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/humanoids/fourier_gr3/**",
        "robots/collision_spheres/fourier_gr3/**",
        "robots/self_collision/fourier_gr3/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/humanoids/fourier_gr3/basic_urdf/gr3v2_1_1_dummy_hand.urdf"
URDF_WITHOUT_HAND_PATH: Path = _DIR / "robots/robot_description/humanoids/fourier_gr3/basic_urdf/gr3v2_1_1.urdf"
COLLISION_SPHERE_FULL_BODY_PATH: Path = _DIR / "robots/collision_spheres/fourier_gr3/full_body.yaml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/fourier_gr3/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/fourier_gr3/ignore.sphere.yml"
MOUNT_LINK_NAME_RIGHT: str = "right_end_effector_link"
MOUNT_LINK_NAME_LEFT: str = "left_end_effector_link"
MOUNT_COORD_SPEC_RIGHT = MountCoordinateSpec(outward_axis="-z", right_axis="-x")
MOUNT_COORD_SPEC_LEFT = MountCoordinateSpec(outward_axis="-z", right_axis="x")
REST_Q: Dict[str, float] = dict.fromkeys(
    (
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "head_yaw_joint",
        "head_pitch_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint",
        "right_wrist_yaw_joint",
        "right_wrist_pitch_joint",
        "right_wrist_roll_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_pitch_joint",
        "left_wrist_yaw_joint",
        "left_wrist_pitch_joint",
        "left_wrist_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_pitch_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_pitch_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    ),
    0.0,
)
