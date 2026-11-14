"""Wuji Hand 1 assets (Wuji Technology, MIT): the URDF and MJCF of the same release; Hand 2.2 is `wuji_hand_2_2`."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import HandCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/end_effectors/wuji_hand/**",
        "robots/robot_mj_description/end_effectors/wuji_hand/**",
        "robots/collision_spheres/wuji_hand_right/**",
        "robots/collision_spheres/wuji_hand_left/**",
        "robots/contact_points/wuji_hand_right/**",
        "robots/contact_points/wuji_hand_left/**",
        "robots/self_collision/wuji_hand_right/**",
        "robots/self_collision/wuji_hand_left/**",
    ]
)
URDF_PATH_RIGHT: Path = _DIR / "robots/robot_description/end_effectors/wuji_hand/wuji_hand_right.urdf"
URDF_PATH_LEFT: Path = _DIR / "robots/robot_description/end_effectors/wuji_hand/wuji_hand_left.urdf"
MJCF_PATH_RIGHT: Path = _DIR / "robots/robot_mj_description/end_effectors/wuji_hand/wuji_hand_right.xml"
MJCF_PATH_LEFT: Path = _DIR / "robots/robot_mj_description/end_effectors/wuji_hand/wuji_hand_left.xml"
COLLISION_SPHERE_PATH_RIGHT: Path = _DIR / "robots/collision_spheres/wuji_hand_right/collision_spheres.yaml"
COLLISION_SPHERE_PATH_LEFT: Path = _DIR / "robots/collision_spheres/wuji_hand_left/collision_spheres.yaml"
CONTACT_POINTS_PATH_RIGHT: Path = _DIR / "robots/contact_points/wuji_hand_right/contact_points.json"
CONTACT_POINTS_PATH_LEFT: Path = _DIR / "robots/contact_points/wuji_hand_left/contact_points.json"
SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT: Path = _DIR / "robots/self_collision/wuji_hand_right/ignore.sphere.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH_LEFT: Path = _DIR / "robots/self_collision/wuji_hand_left/ignore.sphere.yml"
CAPSULE_SELF_COLLISION_IGNORE_PATH_RIGHT: Path = _DIR / "robots/self_collision/wuji_hand_right/ignore.capsule.yml"
CAPSULE_SELF_COLLISION_IGNORE_PATH_LEFT: Path = _DIR / "robots/self_collision/wuji_hand_left/ignore.capsule.yml"
MOUNT_LINK_NAME_RIGHT: str = "right_palm_link"
MOUNT_LINK_NAME_LEFT: str = "left_palm_link"
EE_COORD_SPEC_RIGHT = HandCoordinateSpec(palm_forward_axis="x", four_fingers_up_axis="z")
EE_COORD_SPEC_LEFT = HandCoordinateSpec(palm_forward_axis="x", four_fingers_up_axis="z")
MOUNT_COORD_SPEC_RIGHT = EE_COORD_SPEC_RIGHT.mount_spec
MOUNT_COORD_SPEC_LEFT = EE_COORD_SPEC_LEFT.mount_spec
# finger 1 is the thumb; joints 1-4 run from the palm out, and the thumb's first joint cannot reach 0
REST_Q_RIGHT: Dict[str, float] = {f"right_finger{i}_joint{j}": 0.0 for i in range(1, 6) for j in range(1, 5)}
REST_Q_RIGHT["right_finger1_joint1"] = 0.0475
REST_Q_LEFT: Dict[str, float] = {name.replace("right_", "left_", 1): value for name, value in REST_Q_RIGHT.items()}
