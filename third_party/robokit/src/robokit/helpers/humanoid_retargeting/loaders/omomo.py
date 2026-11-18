"""OMOMO and climbing MoCap loaders for ordered transform arrays.

>>> from robokit.helpers.humanoid_retargeting.loaders.omomo import MOCAP_DEMO_JOINTS, SMPLH_DEMO_JOINTS
>>> SMPLH_DEMO_JOINTS[0], len(SMPLH_DEMO_JOINTS)
('Pelvis', 52)
>>> len(MOCAP_DEMO_JOINTS)
53
"""

import pickle
from typing import List, Sequence, Tuple

import numpy as np


# SMPL-H 52-joint order (object_interaction / robot_only tasks).
SMPLH_DEMO_JOINTS: List[str] = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Index1",
    "L_Index2",
    "L_Index3",
    "L_Middle1",
    "L_Middle2",
    "L_Middle3",
    "L_Pinky1",
    "L_Pinky2",
    "L_Pinky3",
    "L_Ring1",
    "L_Ring2",
    "L_Ring3",
    "L_Thumb1",
    "L_Thumb2",
    "L_Thumb3",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Index1",
    "R_Index2",
    "R_Index3",
    "R_Middle1",
    "R_Middle2",
    "R_Middle3",
    "R_Pinky1",
    "R_Pinky2",
    "R_Pinky3",
    "R_Ring1",
    "R_Ring2",
    "R_Ring3",
    "R_Thumb1",
    "R_Thumb2",
    "R_Thumb3",
]

# MoCap 53-joint order (climbing task).
MOCAP_DEMO_JOINTS: List[str] = [
    "Hips",
    "Spine",
    "Spine1",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandThumb1",
    "LeftHandThumb2",
    "LeftHandThumb3",
    "LeftHandIndex1",
    "LeftHandIndex2",
    "LeftHandIndex3",
    "LeftHandMiddle1",
    "LeftHandMiddle2",
    "LeftHandMiddle3",
    "LeftHandRing1",
    "LeftHandRing2",
    "LeftHandRing3",
    "LeftHandPinky1",
    "LeftHandPinky2",
    "LeftHandPinky3",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandThumb1",
    "RightHandThumb2",
    "RightHandThumb3",
    "RightHandIndex1",
    "RightHandIndex2",
    "RightHandIndex3",
    "RightHandMiddle1",
    "RightHandMiddle2",
    "RightHandMiddle3",
    "RightHandRing1",
    "RightHandRing2",
    "RightHandRing3",
    "RightHandPinky1",
    "RightHandPinky2",
    "RightHandPinky3",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "LeftFootMod",
    "RightFootMod",
]


SMPLH_TOE_NAMES: Tuple[str, str] = ("L_Toe", "R_Toe")
MOCAP_TOE_NAMES: Tuple[str, str] = ("LeftToeBase", "RightToeBase")
OMOMO_FPS: int = 30
_MOCAP_SOURCE_FPS: int = 120  # climbing capture rate before `load_mocap_clip`'s downsample


def _ground_and_order(
    joints: np.ndarray,
    joint_order: Sequence[str],
    human_joint_names: Sequence[str],
    toe_names: Tuple[str, str],
    scale: float,
) -> np.ndarray:
    """Drop `joints` onto z=0, scale it, and pack the requested joints into `(T, J, 7)` transforms."""
    toe_idx = [joint_order.index(name) for name in toe_names]
    z_min = float(joints[:, toe_idx, 2].min())
    if z_min >= 0.1:  # feet never reach the floor: keep 0.1 m of the original clearance
        z_min -= 0.1
    joints = joints.copy()
    joints[:, :, 2] -= z_min
    joints *= scale

    joint_indices = [joint_order.index(name) for name in human_joint_names]
    T_world_human = np.zeros((len(joints), len(joint_indices), 7), dtype=np.float32)
    T_world_human[..., :3] = joints[:, joint_indices]
    T_world_human[..., 3] = 1.0
    return T_world_human


def load_intermimic_pt(pt_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load Z-up OMOMO `(T, 591)`: joints `(T, 52, 3)`, objects `(T, 7)` as `[qw, qx, qy, qz, x, y, z]`."""
    import torch

    data = torch.load(pt_path, map_location="cpu").detach().numpy()
    human_joints = data[:, 162 : 162 + 52 * 3].reshape(-1, 52, 3)
    T_world_object_wxyz_xyz = data[:, 318:325][:, [6, 3, 4, 5, 0, 1, 2]]
    return human_joints, T_world_object_wxyz_xyz


def load_omomo_clip(
    pt_path: str,
    height_dict_path: str,
    subject: str,
    human_joint_names: Sequence[str],
    robot_height: float = 1.32,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Ground-normalize OMOMO data scaled by `robot_height / human_height`; object XY scales directly, Z as a delta."""
    human_joints, T_world_object_wxyz_xyz = load_intermimic_pt(pt_path)
    with open(height_dict_path, "rb") as f:
        human_height = float(pickle.load(f)[subject])
    scale = robot_height / human_height

    T_world_human = _ground_and_order(human_joints, SMPLH_DEMO_JOINTS, human_joint_names, SMPLH_TOE_NAMES, scale)

    obj = T_world_object_wxyz_xyz.copy()
    obj[:, -3:-1] *= scale
    object_z0 = obj[0, -1]
    obj[:, -1] = object_z0 + (obj[:, -1] - object_z0) * scale
    T_world_object = obj[:, [4, 5, 6, 0, 1, 2, 3]]
    return T_world_human, T_world_object.astype(np.float32), float(OMOMO_FPS), float(scale)


def load_mocap_clip(
    npy_path: str,
    human_joint_names: Sequence[str],
    robot_height: float = 1.32,
    default_human_height: float = 1.78,
    downsample: int = 4,
) -> Tuple[np.ndarray, float, float]:
    """Load a climbing clip as ordered transforms, the downsampled FPS, and the interaction scale."""
    joints = np.load(npy_path)[::downsample].astype(np.float32)
    scale = robot_height / default_human_height
    T_world_human = _ground_and_order(joints, MOCAP_DEMO_JOINTS, human_joint_names, MOCAP_TOE_NAMES, scale)
    return T_world_human, _MOCAP_SOURCE_FPS / downsample, float(scale)
