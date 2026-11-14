"""SMPL-X and MANO model constants: joint counts, joint names, landmark vertex IDs."""

from typing import List, Literal


# --- SMPL-X ----------------------------------------------------------------
Gender = Literal["neutral", "male", "female"]

SMPLX_NUM_VERTICES: int = 10475
SMPLX_NUM_JOINTS: int = 55
SMPLX_NUM_POSE_DIRS: int = 486  # (SMPLX_NUM_JOINTS - 1) * 9
SMPLX_NUM_SHAPE_COEFFS: int = 300

# skeletal joints in SMPL-X kinematic-tree order
SMPLX_JOINT_NAMES: List[str] = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
]

# 21 static landmarks: face(5) + feet(6) + fingertips(10), in MPI `VertexJointSelector` append order
SMPLX_STATIC_LANDMARK_NAMES: List[str] = [
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
]
SMPLX_STATIC_LANDMARK_VERTEX_IDS: List[int] = [
    9120,
    9929,
    9448,
    616,
    6,
    5770,
    5780,
    8846,
    8463,
    8474,
    8635,
    5361,
    4933,
    5058,
    5169,
    5286,
    8079,
    7669,
    7794,
    7905,
    8022,
]


# --- MANO ------------------------------------------------------------------
Side = Literal["left", "right"]

MANO_NUM_VERTICES: int = 778
MANO_NUM_JOINTS: int = 16
MANO_NUM_POSE_DIRS: int = 135  # (MANO_NUM_JOINTS - 1) * 9
MANO_NUM_PCA_COMPONENTS: int = 45
MANO_NUM_SHAPE_COEFFS: int = 10

# wrist + 5 fingers × 3 phalanges, in MPI MANO `kintree_table` order
MANO_JOINT_NAMES: List[str] = [
    "wrist",
    "index1",
    "index2",
    "index3",
    "middle1",
    "middle2",
    "middle3",
    "pinky1",
    "pinky2",
    "pinky3",
    "ring1",
    "ring2",
    "ring3",
    "thumb1",
    "thumb2",
    "thumb3",
]

# fingertip vertex IDs in [thumb, index, middle, ring, pinky] order; sides differ at the middle tip
MANO_FINGERTIP_NAMES: List[str] = ["thumb", "index", "middle", "ring", "pinky"]
MANO_FINGERTIP_VERTEX_IDS_RIGHT: List[int] = [745, 317, 444, 556, 673]
MANO_FINGERTIP_VERTEX_IDS_LEFT: List[int] = [745, 317, 445, 556, 673]


def mano_fingertip_vertex_ids(side: Side) -> List[int]:
    """`[thumb, index, middle, ring, pinky]` mesh vertex IDs for `side`."""
    if side == "right":
        return MANO_FINGERTIP_VERTEX_IDS_RIGHT
    return MANO_FINGERTIP_VERTEX_IDS_LEFT


# MANO-native joint order + 5 fingertips → OpenPose hand-keypoint order (wrist, then thumb→pinky, ×3+tip each)
MANOPTH_KEYPOINT_PERMUTATION: List[int] = [
    0,
    13,
    14,
    15,
    16,
    1,
    2,
    3,
    17,
    4,
    5,
    6,
    18,
    10,
    11,
    12,
    19,
    7,
    8,
    9,
    20,
]
