"""ToddlerBot 2xc assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import MountCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/humanoids/toddlerbot_2xc/**",
        "robots/collision_spheres/toddlerbot_2xc/**",
        "robots/self_collision/toddlerbot_2xc/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/humanoids/toddlerbot_2xc/toddlerbot_2xc.urdf"
COLLISION_SPHERE_FULL_BODY_PATH: Path = _DIR / "robots/collision_spheres/toddlerbot_2xc/full_body.yaml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/toddlerbot_2xc/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/toddlerbot_2xc/ignore.sphere.yml"
MOUNT_LINK_NAME_RIGHT: str = "right_hand"
MOUNT_LINK_NAME_LEFT: str = "left_hand"
MOUNT_COORD_SPEC_RIGHT = MountCoordinateSpec(outward_axis="-z", right_axis="-x")
MOUNT_COORD_SPEC_LEFT = MountCoordinateSpec(outward_axis="-z", right_axis="x")
REST_Q: Dict[str, float] = {
    "neck_yaw_driven": 0.0,
    "neck_pitch": 0.0,
    "waist_yaw": 0.0,
    "waist_roll": 0.0,
    "left_hip_pitch": 0.0,
    "left_hip_roll": 0.0,
    "left_hip_yaw_driven": 0.0,
    "left_knee": 0.0,
    "left_ankle_pitch": 0.0,
    "left_ankle_roll": 0.0,
    "right_hip_pitch": 0.0,
    "right_hip_roll": 0.0,
    "right_hip_yaw_driven": 0.0,
    "right_knee": 8.881784197001252e-15,
    "right_ankle_pitch": 0.0,
    "right_ankle_roll": 0.0,
    "left_shoulder_pitch": 0.0,
    "left_shoulder_roll": 0.0,
    "left_shoulder_yaw_driven": 0.0,
    "left_elbow_roll": 0.0,
    "left_elbow_yaw_driven": 0.0,
    "left_wrist_pitch_driven": 0.0,
    "left_wrist_roll": 0.0,
    "right_shoulder_pitch": 0.0,
    "right_shoulder_roll": 0.0,
    "right_shoulder_yaw_driven": 0.0,
    "right_elbow_roll": 0.0,
    "right_elbow_yaw_driven": 0.0,
    "right_wrist_pitch_driven": 0.0,
    "right_wrist_roll": 0.0,
}
