"""g1 aligned preset: learns scale_table + offsets from the base `g1` preset."""

import dataclasses

import numpy as np

from robokit.helpers.humanoid_retargeting.presets.g1 import g1


_OFFSETS = {
    "pelvis": np.array([-0.0052, 0.0001, 0.0523], dtype=np.float32),
    "left_hip_roll_link": np.array([0.0017, 0.0787, -0.0253], dtype=np.float32),
    "left_knee_link": np.array([0.0196, 0.0216, 0.0185], dtype=np.float32),
    "left_ankle_roll_sphere_5_link": np.array([0.0825, 0.0165, 0.0709], dtype=np.float32),
    "right_hip_roll_link": np.array([0.0017, -0.0751, -0.0141], dtype=np.float32),
    "right_knee_link": np.array([0.0209, -0.027, 0.012], dtype=np.float32),
    "right_ankle_roll_sphere_5_link": np.array([0.0744, -0.007, 0.0639], dtype=np.float32),
    "torso_link": np.array([-0.0151, 0.0039, -0.1037], dtype=np.float32),
    "left_shoulder_yaw_link": np.array([0.0014, 0.0016, -0.1108], dtype=np.float32),
    "left_elbow_link": np.array([-0.01, 0.0573, 0.0476], dtype=np.float32),
    "left_wrist_yaw_link": np.array([-0.0238, 0.0441, 0.0404], dtype=np.float32),
    "right_shoulder_yaw_link": np.array([0.0028, -0.0052, -0.1138], dtype=np.float32),
    "right_elbow_link": np.array([-0.0206, -0.0355, 0.0358], dtype=np.float32),
    "right_wrist_yaw_link": np.array([-0.0319, -0.0438, 0.0381], dtype=np.float32),
}


g1_aligned = dataclasses.replace(
    g1,
    human_height_assumption=1.7189,
    scale_table={
        "pelvis": 0.7822,
        "left_hip": 0.6443,
        "left_knee": 0.8518,
        "left_foot": 0.8605,
        "right_hip": 0.6495,
        "right_knee": 0.8489,
        "right_foot": 0.8509,
        "spine3": 0.6956,
        "left_shoulder": 0.8026,
        "left_elbow": 0.8001,
        "left_wrist": 0.7949,
        "right_shoulder": 0.8061,
        "right_elbow": 0.8004,
        "right_wrist": 0.7949,
    },
    link_mapping={
        link: dataclasses.replace(g1.link_mapping[link], position_offset=offset) for link, offset in _OFFSETS.items()
    },
)
