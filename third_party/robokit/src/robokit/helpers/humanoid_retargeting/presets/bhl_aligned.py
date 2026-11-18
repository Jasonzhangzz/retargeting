"""bhl aligned preset: learns scale_table + offsets from the base `bhl` preset."""

import dataclasses

import numpy as np

from robokit.helpers.humanoid_retargeting.presets.bhl import bhl


_OFFSETS = {
    "imu_2": np.array([0.0779, 0.0002, 0.2776], dtype=np.float32),
    "leg_left_hip_roll": np.array([-0.0208, -0.0516, 0.1388], dtype=np.float32),
    "leg_left_knee_pitch": np.array([0.008, -0.0005, -0.0277], dtype=np.float32),
    "leg_left_ankle_roll": np.array([-0.0593, -0.0144, -0.0424], dtype=np.float32),
    "leg_right_hip_roll": np.array([-0.0317, -0.0329, -0.1436], dtype=np.float32),
    "leg_right_knee_pitch": np.array([-0.0034, 0.0056, -0.0381], dtype=np.float32),
    "leg_right_ankle_roll": np.array([0.0773, 0.0325, -0.0551], dtype=np.float32),
    "arm_left_elbow_pitch": np.array([-0.0378, -0.0306, -0.03], dtype=np.float32),
    "arm_left_hand_link": np.array([0.0343, 0.0278, 0.1255], dtype=np.float32),
    "arm_right_elbow_pitch": np.array([0.0231, -0.0407, -0.0094], dtype=np.float32),
    "arm_right_hand_link": np.array([-0.026, 0.0281, 0.1399], dtype=np.float32),
}


bhl_aligned = dataclasses.replace(
    bhl,
    human_height_assumption=1.7189,
    scale_table={
        "pelvis": 0.4044,
        "left_hip": -1.0232,
        "left_knee": 0.2213,
        "left_foot": 0.3525,
        "right_hip": -1.0238,
        "right_knee": 0.2151,
        "right_foot": 0.2956,
        "left_elbow": 0.8482,
        "left_wrist": 0.8531,
        "right_elbow": 0.8507,
        "right_wrist": 0.8638,
    },
    link_mapping={
        link: dataclasses.replace(bhl.link_mapping[link], position_offset=offset) for link, offset in _OFFSETS.items()
    },
)
