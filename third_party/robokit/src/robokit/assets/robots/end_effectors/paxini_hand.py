"""Paxini DexH13 Hand assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import HandCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/end_effectors/paxini_hand/**",
        "robots/collision_spheres/paxini_hand_right/**",
        "robots/contact_points/paxini_hand_right/**",
        "robots/self_collision/paxini_hand_right/**",
    ]
)
URDF_PATH_RIGHT: Path = _DIR / "robots/robot_description/end_effectors/paxini_hand/dexh13_hand_right_description.urdf"
URDF_PATH_LEFT: Path = _DIR / "robots/robot_description/end_effectors/paxini_hand/dexh13_hand_left_description.urdf"
# The tactile pads are separate links whose meshes are fused into their fingers here, so spheres fit once.
URDF_MERGED_PATH_RIGHT: Path = _DIR / "robots/robot_description/end_effectors/paxini_hand/dexh13_hand_right_merged.urdf"
COLLISION_SPHERE_PATH_RIGHT: Path = _DIR / "robots/collision_spheres/paxini_hand_right/collision_spheres.yaml"
CONTACT_POINTS_PATH_RIGHT: Path = _DIR / "robots/contact_points/paxini_hand_right/contact_points.json"
SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT: Path = _DIR / "robots/self_collision/paxini_hand_right/ignore.sphere.yml"
CAPSULE_SELF_COLLISION_IGNORE_PATH_RIGHT: Path = _DIR / "robots/self_collision/paxini_hand_right/ignore.capsule.yml"
MOUNT_LINK_NAME_RIGHT: str = "base_link"
MOUNT_LINK_NAME_LEFT: str = "base_link"
EE_COORD_SPEC_RIGHT = HandCoordinateSpec(palm_forward_axis="x", four_fingers_up_axis="z")
EE_COORD_SPEC_LEFT = HandCoordinateSpec(palm_forward_axis="x", four_fingers_up_axis="z")
MOUNT_COORD_SPEC_RIGHT = EE_COORD_SPEC_RIGHT.mount_spec
MOUNT_COORD_SPEC_LEFT = EE_COORD_SPEC_LEFT.mount_spec
REST_Q_RIGHT: Dict[str, float] = {
    f"right_{finger}_joint_{joint}": 0.0 for finger in ("index", "middle", "ring", "thumb") for joint in range(4)
}
REST_Q_LEFT: Dict[str, float] = {name.replace("right_", "left_", 1): value for name, value in REST_Q_RIGHT.items()}
