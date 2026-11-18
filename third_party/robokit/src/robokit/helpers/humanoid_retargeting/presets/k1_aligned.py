"""k1 aligned preset: learns scale_table + offsets from the base `k1` preset."""

import dataclasses
from copy import deepcopy

import numpy as np

from robokit.helpers.humanoid_retargeting.config import HumanoidContext
from robokit.helpers.humanoid_retargeting.presets.k1 import k1
from robokit.terms.dense.frame_task import FrameTask


_OFFSETS = {
    "Trunk": np.array([0.002699995, -0.0002, 0.045300007], dtype=np.float32),
    "Left_Hip_Roll": np.array([-0.061900012, -0.0181, 0.023699997], dtype=np.float32),
    "Left_Shank": np.array([-0.030999988, 0.0578, -0.12140001], dtype=np.float32),
    "left_foot_link": np.array([-0.0018000002, 0.0271, 0.0001999998], dtype=np.float32),
    "left_hand_link": np.array([-0.0823, 0.0131, 0.0436], dtype=np.float32),
    "left_hand_end_ball": np.array([-0.0565, 0.0811, 0.0354], dtype=np.float32),
    "Right_Hip_Roll": np.array([0.09560001, 0.031, 0.003400012], dtype=np.float32),
    "Right_Shank": np.array([0.075400025, 0.0346, -0.112600006], dtype=np.float32),
    "right_foot_link": np.array([0.014200003, 0.0041, -0.008699999], dtype=np.float32),
    "right_hand_link": np.array([-0.0878, -0.009, 0.0287], dtype=np.float32),
    "right_hand_end_ball": np.array([-0.0561, -0.0759, 0.0337], dtype=np.float32),
}

_LINK_MAPPING = {
    link: dataclasses.replace(k1.link_mapping[link], position_offset=offset) for link, offset in _OFFSETS.items()
}


k1_aligned = dataclasses.replace(
    k1,
    terms=[
        FrameTask(
            frame_index=HumanoidContext.TARGET_LINKS,
            T_world_target=HumanoidContext.TARGETS,
            position_weight={**k1.task("target").position_weight, "left_hand_link": 300, "right_hand_link": 300},
            orientation_weight=k1.task("target").orientation_weight,
            name="target",
        ),
        *deepcopy(k1.terms[1:]),
    ],
    human_height_assumption=1.7189,
    scale_table={
        "pelvis": 0.4901,
        "left_hip": 0.4915,
        "left_knee": 0.1493,
        "left_foot": 0.4557,
        "left_elbow": 0.4883,
        "left_wrist": 0.5102,
        "right_hip": 0.4575,
        "right_knee": 0.181,
        "right_foot": 0.4525,
        "right_elbow": 0.4921,
        "right_wrist": 0.5166,
    },
    link_mapping=_LINK_MAPPING,
)
