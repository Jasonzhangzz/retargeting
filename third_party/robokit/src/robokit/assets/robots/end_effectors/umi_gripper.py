"""UMI gripper assets (MuJoCo Menagerie, MIT). MJCF only: no URDF of this gripper was ever published."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch


_DIR = fetch(["robots/robot_mj_description/end_effectors/umi_gripper/**"])
MJCF_PATH: Path = _DIR / "robots/robot_mj_description/end_effectors/umi_gripper/umi_gripper.xml"
REST_Q: Dict[str, float] = {
    "gripper_joint_x": 0.0,
    "gripper_joint_y": 0.0,
    "gripper_joint_z": 0.0,
    "gripper_joint_rx": 0.0,
    "gripper_joint_ry": 0.0,
    "gripper_joint_rz": 0.0,
    "left_finger_joint": 0.0,
    "right_finger_joint": 0.0,
}
MJCF_REST_Q: Dict[str, float] = dict(REST_Q)
