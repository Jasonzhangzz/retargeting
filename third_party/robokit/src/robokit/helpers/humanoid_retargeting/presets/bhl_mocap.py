"""Berkeley Humanoid Lite interaction preset for MoCap clips."""

from typing import Dict

from robokit.assets.robots.humanoids import berkeley_humanoid_lite
from robokit.helpers.humanoid_retargeting.presets._interaction import _interaction_preset


BHL_HEIGHT = 0.79

MOCAP_TO_BHL_LINK: Dict[str, str] = {
    "Spine1": "imu_2",
    "LeftUpLeg": "leg_left_hip_roll",
    "RightUpLeg": "leg_right_hip_roll",
    "LeftLeg": "leg_left_knee_pitch",
    "RightLeg": "leg_right_knee_pitch",
    "LeftFoot": "leg_left_ankle_roll",
    "RightFoot": "leg_right_ankle_roll",
    "LeftArm": "arm_left_shoulder_roll",
    "RightArm": "arm_right_shoulder_roll",
    "LeftForeArm": "arm_left_elbow_pitch",
    "RightForeArm": "arm_right_elbow_pitch",
    "LeftHandMiddle3": "arm_left_hand_link",
    "RightHandMiddle3": "arm_right_hand_link",
}

bhl_mocap = _interaction_preset(
    MOCAP_TO_BHL_LINK,
    str(berkeley_humanoid_lite.COLLISION_SPHERE_FULL_BODY_PATH),
    str(berkeley_humanoid_lite.URDF_PATH),
)


__all__ = ["BHL_HEIGHT", "MOCAP_TO_BHL_LINK", "bhl_mocap"]
