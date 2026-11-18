"""Wuji Hand 2.2 assets (Wuji Technology, MIT): Hand 2.1 plus a tactile pad link on each fingertip; Hand 1 is `wuji_hand`."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import HandCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/end_effectors/wuji_hand_2_2/**",
        "robots/robot_mj_description/end_effectors/wuji_hand_2_2/**",
        "robots/collision_spheres/wuji_hand_2_2_right/**",
        "robots/collision_spheres/wuji_hand_2_2_left/**",
        "robots/contact_points/wuji_hand_2_2_right/**",
        "robots/contact_points/wuji_hand_2_2_left/**",
        "robots/self_collision/wuji_hand_2_2_right/**",
        "robots/self_collision/wuji_hand_2_2_left/**",
    ]
)
URDF_PATH_RIGHT: Path = _DIR / "robots/robot_description/end_effectors/wuji_hand_2_2/wuji_hand_2_2_right.urdf"
URDF_PATH_LEFT: Path = _DIR / "robots/robot_description/end_effectors/wuji_hand_2_2/wuji_hand_2_2_left.urdf"
MJCF_PATH_RIGHT: Path = _DIR / "robots/robot_mj_description/end_effectors/wuji_hand_2_2/wuji_hand_2_2_right.xml"
MJCF_PATH_LEFT: Path = _DIR / "robots/robot_mj_description/end_effectors/wuji_hand_2_2/wuji_hand_2_2_left.xml"
COLLISION_SPHERE_PATH_RIGHT: Path = _DIR / "robots/collision_spheres/wuji_hand_2_2_right/collision_spheres.yaml"
COLLISION_SPHERE_PATH_LEFT: Path = _DIR / "robots/collision_spheres/wuji_hand_2_2_left/collision_spheres.yaml"
CONTACT_POINTS_PATH_RIGHT: Path = _DIR / "robots/contact_points/wuji_hand_2_2_right/contact_points.json"
CONTACT_POINTS_PATH_LEFT: Path = _DIR / "robots/contact_points/wuji_hand_2_2_left/contact_points.json"
SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT: Path = _DIR / "robots/self_collision/wuji_hand_2_2_right/ignore.sphere.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH_LEFT: Path = _DIR / "robots/self_collision/wuji_hand_2_2_left/ignore.sphere.yml"
CAPSULE_SELF_COLLISION_IGNORE_PATH_RIGHT: Path = _DIR / "robots/self_collision/wuji_hand_2_2_right/ignore.capsule.yml"
CAPSULE_SELF_COLLISION_IGNORE_PATH_LEFT: Path = _DIR / "robots/self_collision/wuji_hand_2_2_left/ignore.capsule.yml"
MOUNT_LINK_NAME_RIGHT: str = "r_mount"
MOUNT_LINK_NAME_LEFT: str = "l_mount"
EE_COORD_SPEC_RIGHT = HandCoordinateSpec(palm_forward_axis="y", four_fingers_up_axis="-z")
EE_COORD_SPEC_LEFT = HandCoordinateSpec(palm_forward_axis="-y", four_fingers_up_axis="-z")
MOUNT_COORD_SPEC_RIGHT = EE_COORD_SPEC_RIGHT.mount_spec
MOUNT_COORD_SPEC_LEFT = EE_COORD_SPEC_LEFT.mount_spec
REST_Q_RIGHT: Dict[str, float] = dict.fromkeys(
    ["r_thumb_cmc_flex", "r_thumb_cmc_abd", "r_thumb_mcp", "r_thumb_ip"]
    + [
        f"r_{finger}_{joint}"
        for finger in ("index_finger", "middle_finger", "ring_finger", "pinky")
        for joint in ("mcp_flex", "mcp_abd", "pip", "dip")
    ],
    0.0,
)
REST_Q_LEFT: Dict[str, float] = {"l_" + name[2:]: value for name, value in REST_Q_RIGHT.items()}
