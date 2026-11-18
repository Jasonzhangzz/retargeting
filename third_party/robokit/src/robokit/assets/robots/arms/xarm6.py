"""UFactory xArm6 assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import MountCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/arms/xarm6/**",
        "robots/collision_spheres/xarm6/**",
        "robots/self_collision/xarm6/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/arms/xarm6/xarm6.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/xarm6/collision_spheres.yaml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/xarm6/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/xarm6/ignore.sphere.yml"
MOUNT_LINK_NAME: str = "link6"
MOUNT_COORD_SPEC = MountCoordinateSpec(outward_axis="z", right_axis="-y", mount_point_xyz=(0.0, 0.0, 0.010))
REST_Q: Dict[str, float] = dict.fromkeys(("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"), 0.0)
