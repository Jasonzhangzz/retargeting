"""G1 + Inspire hands preset for SMPL-H: like `g1_smplh.py` but remaps wrists/ankles to the Inspire links."""

from typing import Dict

from robokit.assets.robots.assembly import g1_inspire
from robokit.helpers.humanoid_retargeting.presets._interaction import _interaction_preset


G1_HEIGHT = 1.32

SMPLH_TO_G1_INSPIRE_LINK: Dict[str, str] = {
    "Pelvis": "pelvis_contour_link",
    "L_Hip": "left_hip_pitch_link",
    "R_Hip": "right_hip_pitch_link",
    "L_Knee": "left_knee_link",
    "R_Knee": "right_knee_link",
    "L_Shoulder": "left_shoulder_roll_link",
    "R_Shoulder": "right_shoulder_roll_link",
    "L_Elbow": "left_elbow_link",
    "R_Elbow": "right_elbow_link",
    "L_Ankle": "left_ankle_pitch_link",
    "R_Ankle": "right_ankle_pitch_link",
    "L_Toe": "left_ankle_roll_link",
    "R_Toe": "right_ankle_roll_link",
    "L_Wrist": "L_hand_base_link",
    "R_Wrist": "R_hand_base_link",
}

g1_inspire_smplh = _interaction_preset(
    SMPLH_TO_G1_INSPIRE_LINK,
    str(g1_inspire.COLLISION_SPHERE_PATH),
    str(g1_inspire.URDF_PATH),
)


__all__ = ["G1_HEIGHT", "SMPLH_TO_G1_INSPIRE_LINK", "g1_inspire_smplh"]
