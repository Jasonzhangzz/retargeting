"""t1 aligned preset: learns scale_table + offsets from the base `t1` preset."""

import dataclasses
from copy import deepcopy

import numpy as np

from robokit.helpers.humanoid_retargeting.config import HumanoidContext
from robokit.helpers.humanoid_retargeting.presets.t1 import t1
from robokit.terms.dense.frame_task import FrameTask


_OFFSETS = {
    "waist_yaw_link": np.array([0.00020001756, 0.058, -0.14710002], dtype=np.float32),
    "trunk": np.array([0.0027000213, -0.0022, -0.17430001], dtype=np.float32),
    "left_hip_roll_link": np.array([-0.0381, -0.0963, -0.035800006], dtype=np.float32),
    "left_knee_pitch_link": np.array([-0.065899976, 0.0231, -0.24040003], dtype=np.float32),
    "left_ankle_roll_link": np.array([-0.0817, 0.072, -0.10860002], dtype=np.float32),
    "left_elbow_yaw_link": np.array([-0.03280897, 0.02134491, 0.0384611], dtype=np.float32),
    "left_wrist_roll_link": np.array([0.0038, 0.0315, 0.005], dtype=np.float32),
    "right_hip_roll_link": np.array([0.08640002, 0.2169, -0.036299992], dtype=np.float32),
    "right_knee_pitch_link": np.array([0.09080003, 0.1413, -0.18410002], dtype=np.float32),
    "right_ankle_roll_link": np.array([0.07330001, 0.1095, -0.037099995], dtype=np.float32),
    "right_elbow_yaw_link": np.array([-0.03314885, -0.01628387, 0.01872215], dtype=np.float32),
    "right_wrist_roll_link": np.array([0.0105, -0.0269, 0.0036], dtype=np.float32),
}

_LINK_MAPPING = {
    link: dataclasses.replace(t1.link_mapping[link], position_offset=offset) for link, offset in _OFFSETS.items()
}
_LINK_MAPPING["left_elbow_yaw_link"] = dataclasses.replace(
    _LINK_MAPPING["left_elbow_yaw_link"],
    rotation_offset=np.array([0.5243019, -0.479574, -0.5194244, -0.4746732], dtype=np.float32),
)
_LINK_MAPPING["right_elbow_yaw_link"] = dataclasses.replace(
    _LINK_MAPPING["right_elbow_yaw_link"],
    rotation_offset=np.array([0.4894881, -0.4792946, -0.5202546, -0.509915], dtype=np.float32),
)


t1_aligned = dataclasses.replace(
    t1,
    terms=[
        FrameTask(
            frame_index=HumanoidContext.TARGET_LINKS,
            T_world_target=HumanoidContext.TARGETS,
            position_weight=t1.task("target").position_weight,
            orientation_weight={
                **t1.task("target").orientation_weight,
                "left_elbow_yaw_link": 20,
                "right_elbow_yaw_link": 20,
            },
            name="target",
        ),
        *deepcopy(t1.terms[1:]),
    ],
    human_height_assumption=1.7189,
    scale_table={
        "pelvis": 0.7692,
        "spine3": 0.4875,
        "left_hip": 1.3147,
        "left_knee": 0.3091,
        "left_foot": 0.6096,
        "left_elbow": 0.7447,
        "left_wrist": 0.7737,
        "right_hip": 1.276,
        "right_knee": 0.4129,
        "right_foot": 0.6671,
        "right_elbow": 0.7513,
        "right_wrist": 0.785,
    },
    link_mapping=_LINK_MAPPING,
)
