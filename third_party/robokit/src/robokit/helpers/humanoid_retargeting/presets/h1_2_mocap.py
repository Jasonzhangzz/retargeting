"""Unitree H1-2 interaction preset for MoCap clips."""

from typing import Dict

from robokit.assets.robots.humanoids import unitree_h1_2
from robokit.helpers.humanoid_retargeting.presets._interaction import _interaction_preset


H1_2_HEIGHT = 1.79

MOCAP_TO_H1_2_LINK: Dict[str, str] = {
    "Spine1": "pelvis",
    "LeftUpLeg": "left_hip_roll_link",
    "RightUpLeg": "right_hip_roll_link",
    "LeftLeg": "left_knee_link",
    "RightLeg": "right_knee_link",
    "LeftFoot": "left_ankle_roll_link",
    "RightFoot": "right_ankle_roll_link",
    "LeftArm": "left_shoulder_roll_link",
    "RightArm": "right_shoulder_roll_link",
    "LeftForeArm": "left_elbow_link",
    "RightForeArm": "right_elbow_link",
    "LeftHandMiddle3": "left_wrist_yaw_link",
    "RightHandMiddle3": "right_wrist_yaw_link",
}

h1_2_mocap = _interaction_preset(
    MOCAP_TO_H1_2_LINK,
    str(unitree_h1_2.COLLISION_SPHERE_FULL_BODY_PATH),
    str(unitree_h1_2.URDF_PATH),
)


__all__ = ["H1_2_HEIGHT", "MOCAP_TO_H1_2_LINK", "h1_2_mocap"]
