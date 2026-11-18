"""Unitree G1 interaction preset for MoCap clips."""

from typing import Dict

from robokit.assets.robots.humanoids import unitree_g1
from robokit.helpers.humanoid_retargeting.presets._interaction import _interaction_preset


G1_HEIGHT = 1.32

MOCAP_TO_G1_LINK: Dict[str, str] = {
    "Spine1": "pelvis_contour_link",
    "LeftUpLeg": "left_hip_pitch_link",
    "RightUpLeg": "right_hip_pitch_link",
    "LeftLeg": "left_knee_link",
    "RightLeg": "right_knee_link",
    "LeftArm": "left_shoulder_roll_link",
    "RightArm": "right_shoulder_roll_link",
    "LeftForeArm": "left_elbow_link",
    "RightForeArm": "right_elbow_link",
    "LeftFoot": "left_ankle_intermediate_1_link",
    "RightFoot": "right_ankle_intermediate_1_link",
    "LeftToeBase": "left_ankle_roll_sphere_5_link",
    "RightToeBase": "right_ankle_roll_sphere_5_link",
    "LeftHandMiddle3": "left_sphere_hand_link",
    "RightHandMiddle3": "right_sphere_hand_link",
}

g1_mocap = _interaction_preset(
    MOCAP_TO_G1_LINK,
    str(unitree_g1.COLLISION_SPHERE_FULL_BODY_SPHEREHAND_PATH),
    str(unitree_g1.URDF_SPHEREHAND_PATH),
)


__all__ = ["G1_HEIGHT", "MOCAP_TO_G1_LINK", "g1_mocap"]
