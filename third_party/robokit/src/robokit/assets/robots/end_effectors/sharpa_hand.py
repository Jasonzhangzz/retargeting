"""Sharpa Wave Hand assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import HandCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/end_effectors/sharpa_hand/**",
        "robots/robot_mj_description/end_effectors/sharpa_hand/**",
        "robots/collision_spheres/sharpa_hand_right/**",
        "robots/collision_spheres/sharpa_hand_left/**",
        "robots/contact_points/sharpa_hand_right/**",
        "robots/contact_points/sharpa_hand_left/**",
        "robots/self_collision/sharpa_hand_right/**",
        "robots/self_collision/sharpa_hand_left/**",
    ]
)
URDF_PATH_RIGHT: Path = _DIR / "robots/robot_description/end_effectors/sharpa_hand/right_sharpa_wave.urdf"
URDF_PATH_LEFT: Path = _DIR / "robots/robot_description/end_effectors/sharpa_hand/left_sharpa_wave.urdf"
MJCF_PATH_RIGHT: Path = _DIR / "robots/robot_mj_description/end_effectors/sharpa_hand/right_hand.xml"
MJCF_PATH_LEFT: Path = _DIR / "robots/robot_mj_description/end_effectors/sharpa_hand/left_hand.xml"
# the same hands with the finger collision capsules and cylinders refit to the visual meshes (see its CHANGELOG)
MJCF_PATH_RIGHT_FIT: Path = _DIR / "robots/robot_mj_description/end_effectors/sharpa_hand/right_hand_fit.xml"
MJCF_PATH_LEFT_FIT: Path = _DIR / "robots/robot_mj_description/end_effectors/sharpa_hand/left_hand_fit.xml"
COLLISION_SPHERE_PATH_RIGHT: Path = _DIR / "robots/collision_spheres/sharpa_hand_right/collision_spheres.yaml"
COLLISION_SPHERE_PATH_LEFT: Path = _DIR / "robots/collision_spheres/sharpa_hand_left/collision_spheres.yaml"
CONTACT_POINTS_DIR_RIGHT: Path = _DIR / "robots/contact_points/sharpa_hand_right"
CONTACT_POINTS_DIR_LEFT: Path = _DIR / "robots/contact_points/sharpa_hand_left"
SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT: Path = _DIR / "robots/self_collision/sharpa_hand_right/ignore.sphere.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH_LEFT: Path = _DIR / "robots/self_collision/sharpa_hand_left/ignore.sphere.yml"
URDF_FLOATING_BASE_PATH_RIGHT: Path = (
    _DIR / "robots/robot_description/end_effectors/sharpa_hand/sharpa_hand_right_floating_base.urdf"
)
URDF_FLOATING_BASE_PATH_LEFT: Path = (
    _DIR / "robots/robot_description/end_effectors/sharpa_hand/sharpa_hand_left_floating_base.urdf"
)
CAPSULE_SELF_COLLISION_IGNORE_PATH_RIGHT: Path = _DIR / "robots/self_collision/sharpa_hand_right/ignore.capsule.yml"
CAPSULE_SELF_COLLISION_IGNORE_PATH_LEFT: Path = _DIR / "robots/self_collision/sharpa_hand_left/ignore.capsule.yml"
MOUNT_LINK_NAME_RIGHT: str = "right_hand_C_MC"
MOUNT_LINK_NAME_LEFT: str = "left_hand_C_MC"
EE_COORD_SPEC_RIGHT = HandCoordinateSpec(palm_forward_axis="x", four_fingers_up_axis="z")
EE_COORD_SPEC_LEFT = HandCoordinateSpec(palm_forward_axis="x", four_fingers_up_axis="z")
MOUNT_COORD_SPEC_RIGHT = EE_COORD_SPEC_RIGHT.mount_spec
MOUNT_COORD_SPEC_LEFT = EE_COORD_SPEC_LEFT.mount_spec
REST_Q_RIGHT: Dict[str, float] = dict.fromkeys(
    (
        "right_thumb_CMC_FE",
        "right_thumb_CMC_AA",
        "right_thumb_MCP_FE",
        "right_thumb_MCP_AA",
        "right_thumb_IP",
        "right_index_MCP_FE",
        "right_index_MCP_AA",
        "right_index_PIP",
        "right_index_DIP",
        "right_middle_MCP_FE",
        "right_middle_MCP_AA",
        "right_middle_PIP",
        "right_middle_DIP",
        "right_ring_MCP_FE",
        "right_ring_MCP_AA",
        "right_ring_PIP",
        "right_ring_DIP",
        "right_pinky_CMC",
        "right_pinky_MCP_FE",
        "right_pinky_MCP_AA",
        "right_pinky_PIP",
        "right_pinky_DIP",
    ),
    0.0,
)
REST_Q_LEFT: Dict[str, float] = {name.replace("right_", "left_", 1): value for name, value in REST_Q_RIGHT.items()}
