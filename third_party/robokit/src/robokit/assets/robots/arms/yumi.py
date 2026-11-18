"""ABB YuMi assets."""

from pathlib import Path
from typing import Dict

from robot_descriptions import yumi_description

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import MountCoordinateSpec


_DIR = fetch(
    [
        "robots/collision_spheres/yumi/**",
        "robots/self_collision/yumi/**",
    ]
)
URDF_PATH: Path = Path(yumi_description.URDF_PATH)
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/yumi/collision_spheres.yaml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/yumi/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/yumi/ignore.sphere.yml"
MOUNT_LINK_NAME_RIGHT: str = "yumi_link_7_r"
MOUNT_LINK_NAME_LEFT: str = "yumi_link_7_l"
MOUNT_COORD_SPEC_RIGHT = MountCoordinateSpec(outward_axis="z", right_axis="-y")
MOUNT_COORD_SPEC_LEFT = MountCoordinateSpec(outward_axis="z", right_axis="-y")
REST_Q: Dict[str, float] = {
    **dict.fromkeys(
        tuple(f"yumi_joint_{joint}_{side}" for side in ("r", "l") for joint in (1, 2, 7, 3, 4, 5, 6))
        + ("gripper_r_joint", "gripper_l_joint"),
        0.0,
    ),
    "yumi_joint_1_l": -1.41,
    "yumi_joint_2_l": -2.1,
    "yumi_joint_3_l": 0.3,
    "yumi_joint_7_l": 0.71,
    "yumi_joint_1_r": 1.41,
    "yumi_joint_2_r": -2.1,
    "yumi_joint_3_r": 0.3,
    "yumi_joint_7_r": -0.71,
}
