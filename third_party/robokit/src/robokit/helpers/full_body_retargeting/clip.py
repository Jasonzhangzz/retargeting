"""Dataset-independent clip schema, populated by loaders such as parahome_utils.py.

Coordinate frames:
- Capture world: native motion.
- Body world: robot-scaled translations; rotations match the capture world.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh

from robokit.smplx import SMPLX_JOINT_NAMES, SMPLX_STATIC_LANDMARK_NAMES


CONTACT_MANO_INDICES = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
HANDS = ("right", "left")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


@dataclass
class SceneGeometry:
    """Furniture and untouched objects in the robot-scaled world."""

    names: List[str]
    meshes: List[trimesh.Trimesh]
    T_world_part: np.ndarray  # (parts, frames, 7)
    static: np.ndarray  # (parts,) bool


@dataclass
class HandTrack:
    """One captured hand and its grasped object, if any; free hands hold the wrist pose instead."""

    mano: np.ndarray  # (frames, 21, 3), native world
    q_wxyz_world_wrist: np.ndarray  # (frames, 4), shared by both worlds
    object_name: Optional[str]
    object_mesh: Optional[trimesh.Trimesh]
    T_native_object: np.ndarray  # (frames, 7)
    T_world_object: np.ndarray  # (frames, 7), body-scaled translation
    contact_points: np.ndarray  # (frames, 10, 3), native world
    contact_normals: np.ndarray  # (frames, 10, 3), native world
    contact_mask: np.ndarray  # (frames, 10) bool


@dataclass
class FullBodyClip:
    """One captured body, hand, object, and optional room interaction."""

    T_native_human: np.ndarray  # (frames, 55, 7), SMPL-X joints in the native world
    human_height: float
    scale: float
    fps: int
    hands: Dict[str, HandTrack]
    scene: Optional[SceneGeometry] = None
    frame_indices: Optional[np.ndarray] = None  # retained native frames, when supplied by the loader

    @property
    def num_frames(self) -> int:
        return int(self.T_native_human.shape[0])

    @property
    def sides(self) -> List[str]:
        return [side for side in HANDS if side in self.hands]

    @property
    def object_names(self) -> List[str]:
        names = []
        for side in self.sides:
            name = self.hands[side].object_name
            if name is not None and name not in names:
                names.append(name)
        return names


def mano_index(side: str = "right") -> Tuple[List[int], List[int]]:
    """Return SMPL-X joint and landmark indices in MANO-21 order.

    Example:
        >>> joints, tips = mano_index("left")
        >>> len(joints), joints[0] >= 0, joints[4], tips[4] >= 0
        (21, True, -1, True)
    """
    joints = [SMPLX_JOINT_NAMES.index(f"{side}_wrist")]
    tips = [-1]
    for finger in FINGERS:
        for index in (1, 2, 3):
            joints.append(SMPLX_JOINT_NAMES.index(f"{side}_{finger}{index}"))
            tips.append(-1)
        joints.append(-1)
        tips.append(SMPLX_STATIC_LANDMARK_NAMES.index(f"{side}_{finger}"))
    return joints, tips


__all__ = [
    "CONTACT_MANO_INDICES",
    "FINGERS",
    "HANDS",
    "FullBodyClip",
    "HandTrack",
    "SceneGeometry",
    "mano_index",
]
