"""talos aligned preset: learns scale_table + offsets from the base `talos` preset."""

import dataclasses

import numpy as np

from robokit.helpers.humanoid_retargeting.presets.talos import talos


_OFFSETS = {
    "base_link": np.array([0.0018, -0.0, 0.1644], dtype=np.float32),
    "torso_2_link": np.array([0.0065, 0.0042, -0.0307], dtype=np.float32),
    "leg_left_2_link": np.array([-0.0497, -0.0761, -0.0027], dtype=np.float32),
    "leg_left_4_link": np.array([0.008, 0.0422, -0.2042], dtype=np.float32),
    "leg_left_6_link": np.array([0.0092, 0.0214, -0.0115], dtype=np.float32),
    "arm_left_4_link": np.array([0.053, -0.0031, -0.0085], dtype=np.float32),
    "arm_left_7_link": np.array([0.0626, -0.014, 0.0362], dtype=np.float32),
    "leg_right_2_link": np.array([0.0987, 0.0391, -0.0118], dtype=np.float32),
    "leg_right_4_link": np.array([0.0852, 0.0238, -0.2033], dtype=np.float32),
    "leg_right_6_link": np.array([-0.0016, -0.0219, 0.0274], dtype=np.float32),
    "arm_right_4_link": np.array([-0.1061, -0.1275, 0.0005], dtype=np.float32),
    "arm_right_7_link": np.array([-0.0651, -0.0966, 0.0579], dtype=np.float32),
}


talos_aligned = dataclasses.replace(
    talos,
    human_height_assumption=1.7189,
    scale_table={
        "pelvis": 0.9593,
        "spine3": 0.9093,
        "left_hip": 0.9654,
        "left_knee": 0.5626,
        "left_foot": 0.8399,
        "left_elbow": 1.1151,
        "left_wrist": 1.1562,
        "right_hip": 0.9362,
        "right_knee": 0.5971,
        "right_foot": 0.8815,
        "right_elbow": 1.2012,
        "right_wrist": 1.2232,
    },
    link_mapping={
        link: dataclasses.replace(talos.link_mapping[link], position_offset=offset) for link, offset in _OFFSETS.items()
    },
)
