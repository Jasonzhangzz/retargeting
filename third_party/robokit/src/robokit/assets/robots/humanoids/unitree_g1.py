"""Unitree G1 humanoid assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import MountCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/humanoids/unitree_g1/**",
        "robots/collision_spheres/g1/**",
        "robots/self_collision/g1/**",
    ]
)
_MODEL = _DIR / "robots/robot_description/humanoids/unitree_g1"
_SPHERES = _DIR / "robots/collision_spheres/g1"

URDF_PATH: Path = _MODEL / "g1_29dof.urdf"
_MJ_DIR = "robots/robot_mj_description/humanoids/g1"
MJCF_PATH: Path = fetch([f"{_MJ_DIR}/**"]) / f"{_MJ_DIR}/g1.xml"
URDF_SPHEREHAND_PATH: Path = _MODEL / "g1_29dof_spherehand.urdf"

COLLISION_SPHERE_PATH: Path = _SPHERES / "collision_spheres.yaml"
COLLISION_SPHERE_FULL_BODY_SPHEREHAND_PATH: Path = _SPHERES / "full_body_spherehand.yaml"
MOUNT_LINK_NAME_RIGHT: str = "right_wrist_yaw_link"
MOUNT_LINK_NAME_LEFT: str = "left_wrist_yaw_link"
MOUNT_COORD_SPEC_RIGHT = MountCoordinateSpec(outward_axis="x", right_axis="-z", mount_point_xyz=(0.0365, 0.0, 0.0))
MOUNT_COORD_SPEC_LEFT = MountCoordinateSpec(outward_axis="x", right_axis="z", mount_point_xyz=(0.0365, 0.0, 0.0))
REST_Q: Dict[str, float] = {
    **dict.fromkeys(
        (
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
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
    "left_hip_pitch_joint": -0.1,
    "left_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "right_hip_pitch_joint": -0.1,
    "right_knee_joint": 0.3,
    "right_ankle_pitch_joint": -0.2,
    "left_shoulder_pitch_joint": 0.35,
    "left_shoulder_roll_joint": 0.18,
    "left_elbow_joint": 0.87,
    "right_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.18,
    "right_elbow_joint": 0.87,
}
# Keep the hand-free G1 base under humanoids/g1/ beside assemblies; grasp chains omit legs but need bare arms.
_NO_HAND_DIR = fetch(["robots/robot_description/humanoids/g1/**"])
URDF_WITHOUT_HAND_PATH: Path = _NO_HAND_DIR / "robots/robot_description/humanoids/g1/g1_29dof_without_hand.urdf"

CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/g1/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/g1/ignore.sphere.yml"
