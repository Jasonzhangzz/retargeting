"""toddy aligned preset: learns scale_table + offsets from the base `toddy` preset."""

import dataclasses

import numpy as np

from robokit.helpers.humanoid_retargeting.presets.toddy import toddy


_OFFSETS = {
    "pelvis_link": np.array([-0.0049, 0.0059, 0.0114], dtype=np.float32),
    "torso": np.array([0.0043, 0.003, -0.1424], dtype=np.float32),
    "left_hip_roll_link": np.array([-0.0206, -0.0333, 0.0078], dtype=np.float32),
    "left_knee_link": np.array([-0.0044, 0.0076, -0.0683], dtype=np.float32),
    "left_ankle_roll_link": np.array([-0.0162, -0.0034, -0.0134], dtype=np.float32),
    "left_elbow_roll_link": np.array([0.0095, 0.0444, 0.0396], dtype=np.float32),
    "left_hand": np.array([0.0145, 0.0595, 0.0222], dtype=np.float32),
    "right_hip_roll_link": np.array([0.0468, 0.0114, -0.0059], dtype=np.float32),
    "right_knee_link": np.array([0.0101, 0.0036, -0.0705], dtype=np.float32),
    "right_ankle_roll_link": np.array([0.0058, -0.0137, -0.0015], dtype=np.float32),
    "right_elbow_roll_link": np.array([-0.0045, -0.0621, 0.0069], dtype=np.float32),
    "right_hand": np.array([-0.0132, -0.0784, -0.0011], dtype=np.float32),
}


toddy_aligned = dataclasses.replace(
    toddy,
    human_height_assumption=1.7189,
    scale_table={
        "pelvis": 0.304,
        "spine3": 0.581,
        "left_hip": 0.5616,
        "left_knee": 0.1206,
        "left_foot": 0.2427,
        "left_elbow": 0.2583,
        "left_wrist": 0.2551,
        "right_hip": 0.5536,
        "right_knee": 0.1369,
        "right_foot": 0.2619,
        "right_elbow": 0.2872,
        "right_wrist": 0.292,
    },
    link_mapping={
        link: dataclasses.replace(toddy.link_mapping[link], position_offset=offset) for link, offset in _OFFSETS.items()
    },
)
