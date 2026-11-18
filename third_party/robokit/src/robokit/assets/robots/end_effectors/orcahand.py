"""ORCA Hand assets (ETH Zurich Soft Robotics Lab, MIT). v1 carries the collision geometry; v2 has none."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import HandCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/end_effectors/orcahand/**",
        "robots/collision_spheres/orcahand_right/**",
        "robots/contact_points/orcahand_right/**",
        "robots/self_collision/orcahand_right/**",
    ]
)
URDF_PATH_RIGHT: Path = _DIR / "robots/robot_description/end_effectors/orcahand/orcahand_right.urdf"
COLLISION_SPHERE_PATH_RIGHT: Path = _DIR / "robots/collision_spheres/orcahand_right/collision_spheres.yaml"
CONTACT_POINTS_PATH_RIGHT: Path = _DIR / "robots/contact_points/orcahand_right/contact_points.json"
SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT: Path = _DIR / "robots/self_collision/orcahand_right/ignore.sphere.yml"
CAPSULE_SELF_COLLISION_IGNORE_PATH_RIGHT: Path = _DIR / "robots/self_collision/orcahand_right/ignore.capsule.yml"
MOUNT_LINK_NAME_RIGHT: str = "right_tower"
EE_COORD_SPEC_RIGHT = HandCoordinateSpec(palm_forward_axis="y", four_fingers_up_axis="z")
MOUNT_COORD_SPEC_RIGHT = EE_COORD_SPEC_RIGHT.mount_spec
REST_Q_RIGHT: Dict[str, float] = dict.fromkeys(
    (
        "right_wrist",
        "right_thumb_mcp",
        "right_thumb_abd",
        "right_thumb_pip",
        "right_thumb_dip",
        "right_index_abd",
        "right_index_mcp",
        "right_index_pip",
        "right_middle_abd",
        "right_middle_mcp",
        "right_middle_pip",
        "right_ring_abd",
        "right_ring_mcp",
        "right_ring_pip",
        "right_pinky_abd",
        "right_pinky_mcp",
        "right_pinky_pip",
    ),
    0.0,
)
