"""Berkeley Humanoid Lite assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import MountCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/humanoids/berkeley_humanoid_lite/**",
        "robots/collision_spheres/berkeley_humanoid_lite/**",
        "robots/self_collision/berkeley_humanoid_lite/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/humanoids/berkeley_humanoid_lite/berkeley_humanoid_lite.urdf"
COLLISION_SPHERE_FULL_BODY_PATH: Path = _DIR / "robots/collision_spheres/berkeley_humanoid_lite/full_body.yaml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/berkeley_humanoid_lite/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/berkeley_humanoid_lite/ignore.sphere.yml"
MOUNT_LINK_NAME_RIGHT: str = "arm_right_hand_link"
MOUNT_LINK_NAME_LEFT: str = "arm_left_hand_link"
MOUNT_COORD_SPEC_RIGHT = MountCoordinateSpec(outward_axis="-z", right_axis="x")
MOUNT_COORD_SPEC_LEFT = MountCoordinateSpec(outward_axis="-z", right_axis="x")
REST_Q: Dict[str, float] = {
    **dict.fromkeys(
        (
            "arm_left_elbow_roll_joint",
            "arm_left_elbow_pitch_joint",
            "arm_left_shoulder_yaw_joint",
            "arm_left_shoulder_roll_joint",
            "arm_left_shoulder_pitch_joint",
            "arm_right_elbow_roll_joint",
            "arm_right_elbow_pitch_joint",
            "arm_right_shoulder_yaw_joint",
            "arm_right_shoulder_roll_joint",
            "arm_right_shoulder_pitch_joint",
            "leg_left_ankle_roll_joint",
            "leg_left_ankle_pitch_joint",
            "leg_left_knee_pitch_joint",
            "leg_left_hip_pitch_joint",
            "leg_left_hip_yaw_joint",
            "leg_left_hip_roll_joint",
            "leg_right_ankle_roll_joint",
            "leg_right_ankle_pitch_joint",
            "leg_right_knee_pitch_joint",
            "leg_right_hip_pitch_joint",
            "leg_right_hip_yaw_joint",
            "leg_right_hip_roll_joint",
        ),
        0.0,
    ),
    "leg_left_hip_pitch_joint": -0.2,
    "leg_left_knee_pitch_joint": 0.4,
    "leg_left_ankle_pitch_joint": -0.3,
    "leg_right_hip_pitch_joint": -0.2,
    "leg_right_knee_pitch_joint": 0.4,
    "leg_right_ankle_pitch_joint": -0.3,
}
