"""Unitree G1 interaction preset for SMPL-H clips."""

from typing import Dict

from robokit.assets.robots.humanoids import unitree_g1
from robokit.helpers.humanoid_retargeting.presets._interaction import _interaction_preset


G1_HEIGHT = 1.32

SMPLH_TO_G1_LINK: Dict[str, str] = {
    "Pelvis": "pelvis_contour_link",
    "L_Hip": "left_hip_pitch_link",
    "R_Hip": "right_hip_pitch_link",
    "L_Knee": "left_knee_link",
    "R_Knee": "right_knee_link",
    "L_Shoulder": "left_shoulder_roll_link",
    "R_Shoulder": "right_shoulder_roll_link",
    "L_Elbow": "left_elbow_link",
    "R_Elbow": "right_elbow_link",
    "L_Ankle": "left_ankle_intermediate_1_link",
    "R_Ankle": "right_ankle_intermediate_1_link",
    "L_Toe": "left_ankle_roll_sphere_5_link",
    "R_Toe": "right_ankle_roll_sphere_5_link",
    "L_Wrist": "left_rubber_hand_link",
    "R_Wrist": "right_rubber_hand_link",
}

g1_smplh = _interaction_preset(
    SMPLH_TO_G1_LINK,
    str(unitree_g1.COLLISION_SPHERE_PATH),
    str(unitree_g1.URDF_PATH),
)


__all__ = ["G1_HEIGHT", "SMPLH_TO_G1_LINK", "g1_smplh"]
