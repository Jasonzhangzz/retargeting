"""pm01 aligned preset: learns scale_table + offsets from the base `pm01` preset."""

import dataclasses

import numpy as np

from robokit.helpers.humanoid_retargeting.presets.pm01 import pm01


_OFFSETS = {
    "LINK_BASE": np.array([0.0092, -0.0003, 0.0434], dtype=np.float32),
    "LINK_TORSO_YAW": np.array([0.0128, 0.0152, -0.0906], dtype=np.float32),
    "LINK_HIP_ROLL_L": np.array([-0.0816, 0.0386, 0.0409], dtype=np.float32),
    "LINK_KNEE_PITCH_L": np.array([-0.012, 0.0757, -0.2231], dtype=np.float32),
    "LINK_ANKLE_ROLL_L": np.array([0.013, 0.0187, -0.0224], dtype=np.float32),
    "LINK_ELBOW_PITCH_L": np.array([-0.00739862, -0.00382795, 0.0055061], dtype=np.float32),
    "LINK_ELBOW_END_L": np.array([0.0222, -0.0084, -0.0893], dtype=np.float32),
    "LINK_HIP_ROLL_R": np.array([0.1263, 0.0825, -0.0045], dtype=np.float32),
    "LINK_KNEE_PITCH_R": np.array([0.1042, 0.0096, -0.1956], dtype=np.float32),
    "LINK_ANKLE_ROLL_R": np.array([0.015, -0.0081, -0.0535], dtype=np.float32),
    "LINK_ELBOW_PITCH_R": np.array([-0.07241805, -0.00249186, 0.00346807], dtype=np.float32),
    "LINK_ELBOW_END_R": np.array([-0.0166, -0.0085, -0.0831], dtype=np.float32),
}

_LINK_MAPPING = {
    link: dataclasses.replace(pm01.link_mapping[link], position_offset=offset) for link, offset in _OFFSETS.items()
}
_LINK_MAPPING["LINK_ELBOW_PITCH_L"] = dataclasses.replace(
    _LINK_MAPPING["LINK_ELBOW_PITCH_L"],
    rotation_offset=np.array([0.5220249, -0.5028737, -0.4751958, 0.4987955], dtype=np.float32),
)
_LINK_MAPPING["LINK_ELBOW_PITCH_R"] = dataclasses.replace(
    _LINK_MAPPING["LINK_ELBOW_PITCH_R"],
    rotation_offset=np.array([0.4768611, -0.4589683, 0.5251929, -0.5349056], dtype=np.float32),
)


pm01_aligned = dataclasses.replace(
    pm01,
    human_height_assumption=1.7189,
    scale_table={
        "pelvis": 0.7999,
        "spine3": 0.7322,
        "left_hip": 0.3665,
        "left_knee": 0.2438,
        "left_foot": 0.7306,
        "left_elbow": 0.783,
        "left_wrist": 0.8015,
        "right_hip": 0.3248,
        "right_knee": 0.3019,
        "right_foot": 0.6996,
        "right_elbow": 0.821,
        "right_wrist": 0.8216,
    },
    link_mapping=_LINK_MAPPING,
)
