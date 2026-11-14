"""Universal Robots UR10 assets."""

from pathlib import Path
from typing import Dict

from robot_descriptions import ur10_description

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import MountCoordinateSpec


_DIR = fetch(
    [
        "robots/collision_spheres/ur10/**",
        "robots/self_collision/ur10/**",
    ]
)
URDF_PATH: Path = Path(ur10_description.URDF_PATH)
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/ur10/generated.yaml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/ur10/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/ur10/ignore.sphere.yml"
MOUNT_LINK_NAME: str = "ee_link"
MOUNT_COORD_SPEC = MountCoordinateSpec(outward_axis="x", right_axis="-y")
REST_Q: Dict[str, float] = {
    **dict.fromkeys(
        (
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ),
        0.0,
    ),
    "shoulder_pan_joint": -1.5708,
    "shoulder_lift_joint": -1.5708,
    "elbow_joint": 1.5708,
    "wrist_1_joint": -1.5708,
    "wrist_2_joint": -1.5708,
}
