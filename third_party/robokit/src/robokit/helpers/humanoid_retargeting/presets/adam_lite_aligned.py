"""adam_lite aligned preset: learns scale_table + offsets from the base `adam_lite` preset."""

import dataclasses
from copy import deepcopy

import numpy as np

from robokit.helpers.humanoid_retargeting.config import HumanoidContext
from robokit.helpers.humanoid_retargeting.presets.adam_lite import adam_lite
from robokit.terms.dense.frame_task import FrameTask


_OFFSETS = {
    "pelvis": np.array([0.0031, -0.0, -0.0299], dtype=np.float32),
    "torso": np.array([-0.0082, 0.0034, -0.0797], dtype=np.float32),
    "hipRollLeft": np.array([-0.0936, -0.0898, 0.0251], dtype=np.float32),
    "shinLeft": np.array([0.0029, 0.0716, -0.2537], dtype=np.float32),
    "toeLeft": np.array([0.0026, 0.0063, -0.0134], dtype=np.float32),
    "elbowLeft": np.array([0.00606266, 0.01223958, -0.07616159], dtype=np.float32),
    "wristYawLeft": np.array([-0.0082, 0.0093, 0.1075], dtype=np.float32),
    "hipRollRight": np.array([0.1636, 0.0421, 0.0052], dtype=np.float32),
    "shinRight": np.array([0.0987, 0.0498, -0.2757], dtype=np.float32),
    "toeRight": np.array([0.0081, -0.011, -0.032], dtype=np.float32),
    "elbowRight": np.array([-0.01826955, 0.00728805, -0.05711408], dtype=np.float32),
    "wristYawRight": np.array([-0.0178, -0.0111, 0.1282], dtype=np.float32),
}

_LINK_MAPPING = {
    link: dataclasses.replace(adam_lite.link_mapping[link], position_offset=offset) for link, offset in _OFFSETS.items()
}
_LINK_MAPPING["elbowLeft"] = dataclasses.replace(
    _LINK_MAPPING["elbowLeft"],
    rotation_offset=np.array([0.3041188, -0.4606286, -0.6780801, 0.485325], dtype=np.float32),
)
_LINK_MAPPING["elbowRight"] = dataclasses.replace(
    _LINK_MAPPING["elbowRight"],
    rotation_offset=np.array([0.6655602, -0.4642811, 0.2919593, -0.506194], dtype=np.float32),
)


adam_lite_aligned = dataclasses.replace(
    adam_lite,
    terms=[
        FrameTask(
            frame_index=HumanoidContext.TARGET_LINKS,
            T_world_target=HumanoidContext.TARGETS,
            position_weight=adam_lite.task("target").position_weight,
            orientation_weight={
                **adam_lite.task("target").orientation_weight,
                "torso": 30,
                "elbowLeft": 30,
                "elbowRight": 30,
            },
            name="target",
        ),
        *deepcopy(adam_lite.terms[1:]),
    ],
    human_height_assumption=1.7189,
    scale_table={
        "pelvis": 0.9802,
        "spine3": 0.766,
        "left_hip": 1.1274,
        "left_knee": 0.5173,
        "left_foot": 0.9134,
        "left_elbow": 1.0008,
        "left_wrist": 0.9948,
        "right_hip": 1.0671,
        "right_knee": 0.513,
        "right_foot": 0.8922,
        "right_elbow": 1.0088,
        "right_wrist": 1.0198,
    },
    link_mapping=_LINK_MAPPING,
)
