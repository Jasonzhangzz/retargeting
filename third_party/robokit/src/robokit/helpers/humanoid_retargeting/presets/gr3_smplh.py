"""Fourier GR3 interaction preset for SMPL-H clips."""

from typing import Dict

from robokit.assets.robots.humanoids import fourier_gr3
from robokit.helpers.humanoid_retargeting.presets._interaction import _interaction_preset


GR3_HEIGHT = 1.66

SMPLH_TO_GR3_LINK: Dict[str, str] = {
    "Pelvis": "base_link",
    "L_Hip": "left_thigh_roll_link",
    "R_Hip": "right_thigh_roll_link",
    "L_Knee": "left_shank_pitch_link",
    "R_Knee": "right_shank_pitch_link",
    "L_Ankle": "left_foot_roll_link",
    "R_Ankle": "right_foot_roll_link",
    "L_Shoulder": "left_upper_arm_roll_link",
    "R_Shoulder": "right_upper_arm_roll_link",
    "L_Elbow": "left_lower_arm_pitch_link",
    "R_Elbow": "right_lower_arm_pitch_link",
    "L_Wrist": "left_end_effector_link",
    "R_Wrist": "right_end_effector_link",
}

gr3_smplh = _interaction_preset(
    SMPLH_TO_GR3_LINK,
    str(fourier_gr3.COLLISION_SPHERE_FULL_BODY_PATH),
    str(fourier_gr3.URDF_PATH),
)


__all__ = ["GR3_HEIGHT", "SMPLH_TO_GR3_LINK", "gr3_smplh"]
