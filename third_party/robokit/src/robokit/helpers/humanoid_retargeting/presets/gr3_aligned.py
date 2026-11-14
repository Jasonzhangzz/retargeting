"""gr3 aligned preset: learns scale_table + offsets from the base `gr3` preset."""

import dataclasses

import numpy as np

from robokit.helpers.humanoid_retargeting.presets.gr3 import gr3


_OFFSETS = {
    "base_link": np.array([-0.0029, -0.0001, 0.0115], dtype=np.float32),
    "left_thigh_roll_link": np.array([0.002, 0.1115, -0.0022], dtype=np.float32),
    "left_shank_pitch_link": np.array([0.0744, 0.0277, -0.0241], dtype=np.float32),
    "left_foot_roll_link": np.array([0.0146, -0.0051, 0.0375], dtype=np.float32),
    "right_thigh_roll_link": np.array([-0.0005, -0.1101, 0.0014], dtype=np.float32),
    "right_shank_pitch_link": np.array([0.08, -0.0497, -0.0293], dtype=np.float32),
    "right_foot_roll_link": np.array([0.0121, -0.0267, 0.0238], dtype=np.float32),
    "torso_link": np.array([0.0048, 0.0089, 0.0599], dtype=np.float32),
    "left_upper_arm_roll_link": np.array([-0.0136, -0.0015, -0.0099], dtype=np.float32),
    "left_lower_arm_pitch_link": np.array([0.0018, 0.0602, 0.0059], dtype=np.float32),
    "left_end_effector_link": np.array([-0.0054, -0.0019, -0.019], dtype=np.float32),
    "right_upper_arm_roll_link": np.array([-0.0083, -0.0008, -0.0032], dtype=np.float32),
    "right_lower_arm_pitch_link": np.array([-0.0029, -0.0369, 0.0345], dtype=np.float32),
    "right_end_effector_link": np.array([0.0093, -0.0116, 0.0141], dtype=np.float32),
}


gr3_aligned = dataclasses.replace(
    gr3,
    human_height_assumption=1.7189,
    scale_table={
        "pelvis": 0.9752,
        "left_hip": 0.1549,
        "left_knee": 0.8323,
        "left_foot": 0.9486,
        "right_hip": 0.1594,
        "right_knee": 0.8294,
        "right_foot": 0.9265,
        "spine3": 1.1754,
        "left_shoulder": 1.0143,
        "left_elbow": 0.9976,
        "left_wrist": 0.9924,
        "right_shoulder": 1.0293,
        "right_elbow": 1.0156,
        "right_wrist": 1.0195,
    },
    link_mapping={
        link: dataclasses.replace(gr3.link_mapping[link], position_offset=offset) for link, offset in _OFFSETS.items()
    },
)
