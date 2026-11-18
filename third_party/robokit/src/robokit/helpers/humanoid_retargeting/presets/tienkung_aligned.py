"""tienkung aligned preset: learns scale_table + offsets from the base `tienkung` preset."""

import dataclasses
from copy import deepcopy

import numpy as np

from robokit.helpers.humanoid_retargeting.config import HumanoidContext
from robokit.helpers.humanoid_retargeting.presets.tienkung import tienkung
from robokit.terms.dense.frame_task import FrameTask


_OFFSETS = {
    "pelvis": np.array([0.0074, -0.0001, 0.0013], dtype=np.float32),
    "waist_link": np.array([0.0101, 0.0024, -0.1534], dtype=np.float32),
    "hip_roll_l_link": np.array([-0.0732, -0.0519, 0.0288], dtype=np.float32),
    "knee_pitch_l_link": np.array([0.0055, 0.0488, -0.24], dtype=np.float32),
    "ankle_roll_l_link": np.array([-0.005, 0.0127, 0.0291], dtype=np.float32),
    "elbow_l_link": np.array([0.05839864, 0.04188077, 0.08612149], dtype=np.float32),
    "hip_roll_r_link": np.array([0.1417, 0.0522, 0.0033], dtype=np.float32),
    "knee_pitch_r_link": np.array([0.0981, 0.0463, -0.2594], dtype=np.float32),
    "ankle_roll_r_link": np.array([0.0089, -0.0062, -0.0176], dtype=np.float32),
    "elbow_r_link": np.array([0.05950546, -0.10176507, 0.11009141], dtype=np.float32),
}

_LINK_MAPPING = {
    link: dataclasses.replace(tienkung.link_mapping[link], position_offset=offset) for link, offset in _OFFSETS.items()
}
_LINK_MAPPING["elbow_l_link"] = dataclasses.replace(
    _LINK_MAPPING["elbow_l_link"],
    rotation_offset=np.array([0.64517897, -0.23493864, -0.66326433, 0.29770514], dtype=np.float32),
)
_LINK_MAPPING["elbow_r_link"] = dataclasses.replace(
    _LINK_MAPPING["elbow_r_link"],
    rotation_offset=np.array([0.00726475, -0.6850481, -0.00562361, -0.7284399], dtype=np.float32),
)


tienkung_aligned = dataclasses.replace(
    tienkung,
    terms=[
        FrameTask(
            frame_index=HumanoidContext.TARGET_LINKS,
            T_world_target=HumanoidContext.TARGETS,
            position_weight={**tienkung.task("target").position_weight, "elbow_l_link": 300, "elbow_r_link": 300},
            orientation_weight={**tienkung.task("target").orientation_weight, "elbow_l_link": 30, "elbow_r_link": 30},
            name="target",
        ),
        *deepcopy(tienkung.terms[1:]),
    ],
    human_height_assumption=1.7189,
    scale_table={
        "pelvis": 0.8708,
        "spine3": 0.5484,
        "left_hip": 0.892,
        "left_knee": 0.4993,
        "left_foot": 0.8605,
        "left_elbow": 1.2505,
        "right_hip": 0.8296,
        "right_knee": 0.4926,
        "right_foot": 0.8082,
        "right_elbow": 1.2646,
    },
    link_mapping=_LINK_MAPPING,
)
