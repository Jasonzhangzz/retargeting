"""h1 aligned preset: learns scale_table + offsets from the base `h1` preset."""

import dataclasses
from copy import deepcopy

import numpy as np

from robokit.helpers.humanoid_retargeting.config import HumanoidContext
from robokit.helpers.humanoid_retargeting.presets.h1 import h1
from robokit.terms.dense.frame_task import FrameTask


_OFFSETS = {
    "pelvis": np.array([0.0041, -0.0023, 0.1775], dtype=np.float32),
    "torso_link": np.array([0.0082, 0.0015, -0.0623], dtype=np.float32),
    "left_hip_roll_link": np.array([-0.0615, 0.0366, 0.0171], dtype=np.float32),
    "left_knee_link": np.array([-0.0392, 0.0953, -0.1703], dtype=np.float32),
    "left_ankle_link": np.array([0.0162, 0.057, -0.0557], dtype=np.float32),
    "left_elbow_link": np.array([-0.08164042, 0.11442472, 0.03542803], dtype=np.float32),
    "right_hip_roll_link": np.array([0.0829, 0.0328, -0.0029], dtype=np.float32),
    "right_knee_link": np.array([0.144, 0.0095, -0.1596], dtype=np.float32),
    "right_ankle_link": np.array([0.0177, -0.0183, -0.0111], dtype=np.float32),
    "right_elbow_link": np.array([-0.10463597, -0.08200969, 0.13785432], dtype=np.float32),
}

_LINK_MAPPING = {
    link: dataclasses.replace(h1.link_mapping[link], position_offset=offset) for link, offset in _OFFSETS.items()
}
_LINK_MAPPING["left_elbow_link"] = dataclasses.replace(
    _LINK_MAPPING["left_elbow_link"],
    rotation_offset=np.array([0.92109597, -0.3865862, -0.01326593, 0.04424222], dtype=np.float32),
)
_LINK_MAPPING["right_elbow_link"] = dataclasses.replace(
    _LINK_MAPPING["right_elbow_link"],
    rotation_offset=np.array([0.01000798, 0.03040261, 0.03042694, -0.9990244], dtype=np.float32),
)


h1_aligned = dataclasses.replace(
    h1,
    terms=[
        FrameTask(
            frame_index=HumanoidContext.TARGET_LINKS,
            T_world_target=HumanoidContext.TARGETS,
            position_weight={**h1.task("target").position_weight, "left_elbow_link": 100, "right_elbow_link": 100},
            orientation_weight=h1.task("target").orientation_weight,
            name="target",
        ),
        *deepcopy(h1.terms[1:]),
    ],
    human_height_assumption=1.7189,
    scale_table={
        "pelvis": 0.8943,
        "spine3": 0.8165,
        "left_hip": -0.0251,
        "left_knee": 0.5063,
        "left_foot": 0.7875,
        "left_elbow": 1.252,
        "right_hip": -0.0482,
        "right_knee": 0.4933,
        "right_foot": 0.8221,
        "right_elbow": 1.2558,
    },
    link_mapping=_LINK_MAPPING,
)
