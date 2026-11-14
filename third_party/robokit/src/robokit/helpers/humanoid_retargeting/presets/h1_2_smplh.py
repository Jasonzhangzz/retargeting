"""Unitree H1-2 interaction preset for SMPL-H clips."""

from typing import Dict

from robokit.assets.robots.humanoids import unitree_h1_2
from robokit.helpers.humanoid_retargeting.presets._interaction import _interaction_preset


H1_2_HEIGHT = 1.79

SMPLH_TO_H1_2_LINK: Dict[str, str] = {
    "Pelvis": "pelvis",
    "L_Hip": "left_hip_roll_link",
    "R_Hip": "right_hip_roll_link",
    "L_Knee": "left_knee_link",
    "R_Knee": "right_knee_link",
    "L_Ankle": "left_ankle_roll_link",
    "R_Ankle": "right_ankle_roll_link",
    "L_Shoulder": "left_shoulder_roll_link",
    "R_Shoulder": "right_shoulder_roll_link",
    "L_Elbow": "left_elbow_link",
    "R_Elbow": "right_elbow_link",
    "L_Wrist": "left_wrist_yaw_link",
    "R_Wrist": "right_wrist_yaw_link",
}

h1_2_smplh = _interaction_preset(
    SMPLH_TO_H1_2_LINK,
    str(unitree_h1_2.COLLISION_SPHERE_FULL_BODY_PATH),
    str(unitree_h1_2.URDF_PATH),
)


__all__ = ["H1_2_HEIGHT", "SMPLH_TO_H1_2_LINK", "h1_2_smplh"]
