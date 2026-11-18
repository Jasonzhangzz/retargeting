"""Fourier GR3 interaction preset for MoCap clips."""

from typing import Dict

from robokit.assets.robots.humanoids import fourier_gr3
from robokit.helpers.humanoid_retargeting.presets._interaction import _interaction_preset


GR3_HEIGHT = 1.66

MOCAP_TO_GR3_LINK: Dict[str, str] = {
    "Spine1": "base_link",
    "LeftUpLeg": "left_thigh_roll_link",
    "RightUpLeg": "right_thigh_roll_link",
    "LeftLeg": "left_shank_pitch_link",
    "RightLeg": "right_shank_pitch_link",
    "LeftFoot": "left_foot_roll_link",
    "RightFoot": "right_foot_roll_link",
    "LeftArm": "left_upper_arm_roll_link",
    "RightArm": "right_upper_arm_roll_link",
    "LeftForeArm": "left_lower_arm_pitch_link",
    "RightForeArm": "right_lower_arm_pitch_link",
    "LeftHandMiddle3": "left_end_effector_link",
    "RightHandMiddle3": "right_end_effector_link",
}

gr3_mocap = _interaction_preset(
    MOCAP_TO_GR3_LINK,
    str(fourier_gr3.COLLISION_SPHERE_FULL_BODY_PATH),
    str(fourier_gr3.URDF_PATH),
)


__all__ = ["GR3_HEIGHT", "MOCAP_TO_GR3_LINK", "gr3_mocap"]
