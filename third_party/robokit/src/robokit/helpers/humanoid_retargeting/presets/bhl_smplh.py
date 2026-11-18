"""Berkeley Humanoid Lite interaction preset for SMPL-H clips."""

from typing import Dict

from robokit.assets.robots.humanoids import berkeley_humanoid_lite
from robokit.helpers.humanoid_retargeting.presets._interaction import _interaction_preset


BHL_HEIGHT = 0.79

SMPLH_TO_BHL_LINK: Dict[str, str] = {
    "Pelvis": "imu_2",
    "L_Hip": "leg_left_hip_roll",
    "R_Hip": "leg_right_hip_roll",
    "L_Knee": "leg_left_knee_pitch",
    "R_Knee": "leg_right_knee_pitch",
    "L_Ankle": "leg_left_ankle_roll",
    "R_Ankle": "leg_right_ankle_roll",
    "L_Shoulder": "arm_left_shoulder_roll",
    "R_Shoulder": "arm_right_shoulder_roll",
    "L_Elbow": "arm_left_elbow_pitch",
    "R_Elbow": "arm_right_elbow_pitch",
    "L_Wrist": "arm_left_hand_link",
    "R_Wrist": "arm_right_hand_link",
}

bhl_smplh = _interaction_preset(
    SMPLH_TO_BHL_LINK,
    str(berkeley_humanoid_lite.COLLISION_SPHERE_FULL_BODY_PATH),
    str(berkeley_humanoid_lite.URDF_PATH),
)


__all__ = ["BHL_HEIGHT", "SMPLH_TO_BHL_LINK", "bhl_smplh"]
