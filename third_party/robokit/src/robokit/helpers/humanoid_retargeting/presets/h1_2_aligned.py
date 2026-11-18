"""h1_2 aligned preset: learns scale_table + offsets from the base `h1_2` preset."""

import dataclasses
from copy import deepcopy

import numpy as np

from robokit.helpers.humanoid_retargeting.config import HumanoidContext
from robokit.helpers.humanoid_retargeting.presets.h1_2 import h1_2
from robokit.terms.dense.frame_task import FrameTask


_OFFSETS = {
    "pelvis": np.array([-0.0123, 0.0009, 0.097], dtype=np.float32),
    "left_hip_roll_link": np.array([-0.0004, 0.1214, 0.0001], dtype=np.float32),
    "left_knee_link": np.array([0.0638, 0.0489, -0.0184], dtype=np.float32),
    "left_ankle_roll_link": np.array([-0.0094, 0.014, 0.089], dtype=np.float32),
    "right_hip_roll_link": np.array([0.0037, -0.1169, 0.0066], dtype=np.float32),
    "right_knee_link": np.array([0.0777, -0.0609, -0.0322], dtype=np.float32),
    "right_ankle_roll_link": np.array([-0.007, -0.0266, 0.0517], dtype=np.float32),
    "torso_link": np.array([-0.0166, 0.0055, -0.1049], dtype=np.float32),
    "left_shoulder_roll_link": np.array([0.0376, 0.0012, -0.0125], dtype=np.float32),
    "left_elbow_link": np.array([0.0273, 0.1152, 0.0636], dtype=np.float32),
    "left_wrist_yaw_link": np.array([-0.0469, 0.0597, 0.0622], dtype=np.float32),
    "right_shoulder_roll_link": np.array([0.0567, -0.005, -0.0154], dtype=np.float32),
    "right_elbow_link": np.array([0.0085, -0.0837, 0.0696], dtype=np.float32),
    "right_wrist_yaw_link": np.array([-0.0643, -0.0611, 0.088], dtype=np.float32),
}

_LINK_MAPPING = {
    link: dataclasses.replace(h1_2.link_mapping[link], position_offset=offset) for link, offset in _OFFSETS.items()
}


h1_2_aligned = dataclasses.replace(
    h1_2,
    terms=[
        FrameTask(
            frame_index=HumanoidContext.TARGET_LINKS,
            T_world_target=HumanoidContext.TARGETS,
            position_weight=h1_2.task("target").position_weight,
            orientation_weight={**h1_2.task("target").orientation_weight, "torso_link": 30},
            name="target",
        ),
        *deepcopy(h1_2.terms[1:]),
    ],
    human_height_assumption=1.7189,
    scale_table={
        "pelvis": 0.9878,
        "left_hip": 0.715,
        "left_knee": 0.9211,
        "left_foot": 1.0394,
        "right_hip": 0.7121,
        "right_knee": 0.9137,
        "right_foot": 0.9967,
        "spine3": 0.6921,
        "left_shoulder": 1.2361,
        "left_elbow": 1.2276,
        "left_wrist": 1.2125,
        "right_shoulder": 1.2441,
        "right_elbow": 1.2355,
        "right_wrist": 1.2235,
    },
    link_mapping=_LINK_MAPPING,
)
