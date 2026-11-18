"""XHand assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import HandCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/end_effectors/xhand/**",
        "robots/collision_spheres/xhand_right/**",
        "robots/contact_points/xhand_right/**",
        "robots/contact_avoid_points/xhand_right/**",
        "robots/self_collision/xhand_right/**",
        "robots/robot_mj_description/end_effectors/xhand/**",
        "robots/contact_points/xarm_xhand_mj/**",
        "robots/contact_avoid_points/xarm_xhand_mj/**",
        "robots/self_collision/xhand_mj_right/**",
    ]
)
URDF_PATH_RIGHT: Path = _DIR / "robots/robot_description/end_effectors/xhand/xhand_right.urdf"
URDF_PATH_LEFT: Path = _DIR / "robots/robot_description/end_effectors/xhand/xhand_left.urdf"
URDF_GLB_PATH_RIGHT: Path = _DIR / "robots/robot_description/end_effectors/xhand/xhand_right_glb.urdf"
URDF_GLB_FLOATING_BASE_PATH_RIGHT: Path = (
    _DIR / "robots/robot_description/end_effectors/xhand/xhand_right_glb_floating_base.urdf"
)
COLLISION_SPHERE_PATH_RIGHT: Path = _DIR / "robots/collision_spheres/xhand_right/collision_spheres.yaml"
CONTACT_POINTS_PATH_RIGHT: Path = _DIR / "robots/contact_points/xhand_right/contact_points.json"
CONTACT_POINTS_TIP_PATH_RIGHT: Path = _DIR / "robots/contact_points/xhand_right/contact_points_tip.json"
CONTACT_AVOID_POINTS_PATH_RIGHT: Path = _DIR / "robots/contact_avoid_points/xhand_right/contact_avoid_points.json"
CAPSULE_SELF_COLLISION_IGNORE_PATH_RIGHT: Path = _DIR / "robots/self_collision/xhand_right/ignore.capsule.yml"
MJCF_PATH_RIGHT: Path = _DIR / "robots/robot_mj_description/end_effectors/xhand/xhand.xml"
MJ_CONTACT_POINTS_PATH_RIGHT: Path = _DIR / "robots/contact_points/xarm_xhand_mj/contact_points.json"
MJ_CONTACT_AVOID_POINTS_PATH_RIGHT: Path = _DIR / "robots/contact_avoid_points/xarm_xhand_mj/contact_avoid_points.json"
SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT: Path = _DIR / "robots/self_collision/xhand_right/ignore.sphere.yml"
MJ_CAPSULE_SELF_COLLISION_IGNORE_PATH_RIGHT: Path = _DIR / "robots/self_collision/xhand_mj_right/ignore.capsule.yml"
MOUNT_LINK_NAME_RIGHT: str = "right_hand_link"
MOUNT_LINK_NAME_LEFT: str = "left_hand_link"
EE_COORD_SPEC_RIGHT = HandCoordinateSpec(
    palm_forward_axis="y", four_fingers_up_axis="-z", mount_point_xyz=(0.0, 0.0, 0.008)
)
EE_COORD_SPEC_LEFT = HandCoordinateSpec(
    palm_forward_axis="y", four_fingers_up_axis="-z", mount_point_xyz=(0.0, 0.0, 0.008)
)
MOUNT_COORD_SPEC_RIGHT = EE_COORD_SPEC_RIGHT.mount_spec
MOUNT_COORD_SPEC_LEFT = EE_COORD_SPEC_LEFT.mount_spec
REST_Q_RIGHT: Dict[str, float] = dict.fromkeys(
    (
        "right_hand_thumb_bend_joint",
        "right_hand_thumb_rota_joint1",
        "right_hand_thumb_rota_joint2",
        "right_hand_index_bend_joint",
        "right_hand_index_joint1",
        "right_hand_index_joint2",
        "right_hand_mid_joint1",
        "right_hand_mid_joint2",
        "right_hand_ring_joint1",
        "right_hand_ring_joint2",
        "right_hand_pinky_joint1",
        "right_hand_pinky_joint2",
    ),
    0.0,
)
REST_Q_LEFT: Dict[str, float] = {name.replace("right_", "left_", 1): value for name, value in REST_Q_RIGHT.items()}
MJCF_MOUNT_LINK_NAME_RIGHT: str = MOUNT_LINK_NAME_RIGHT
MJCF_EE_COORD_SPEC_RIGHT = HandCoordinateSpec(palm_forward_axis="y", four_fingers_up_axis="-z")
MJCF_MOUNT_COORD_SPEC_RIGHT = MJCF_EE_COORD_SPEC_RIGHT.mount_spec
MJCF_REST_Q_RIGHT: Dict[str, float] = {f"{name}_xh": value for name, value in REST_Q_RIGHT.items()}
