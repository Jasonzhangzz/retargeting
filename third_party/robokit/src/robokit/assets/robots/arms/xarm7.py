"""UFactory xArm7 assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import MountCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/arms/xarm7/**",
        "robots/collision_spheres/xarm7/**",
        "robots/self_collision/xarm7/**",
        "robots/self_collision/xarm7_mj/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/arms/xarm7/xarm7.urdf"
URDF_GLB_PATH: Path = _DIR / "robots/robot_description/arms/xarm7/xarm7_glb.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/xarm7/collision_spheres.yaml"
MOUNT_LINK_NAME: str = "link7"
MOUNT_COORD_SPEC = MountCoordinateSpec(outward_axis="z", right_axis="-y", mount_point_xyz=(0.0, 0.0, 0.010))
REST_Q: Dict[str, float] = {
    **dict.fromkeys(("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"), 0.0),
    "joint1": -0.03,
    "joint2": 0.13,
    "joint3": 0.03,
    "joint4": 0.24,
    "joint5": 3.14,
    "joint6": 1.46,
    "joint7": 0.0,
}

# MuJoCo Menagerie's bare arm, for pairing with an MJCF end-effector.
_MJ_DIR = fetch(["robots/robot_mj_description/arms/xarm7/**"])
MJCF_PATH: Path = _MJ_DIR / "robots/robot_mj_description/arms/xarm7/xarm7_nohand.xml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/xarm7/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/xarm7/ignore.sphere.yml"
MJ_CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/xarm7_mj/ignore.capsule.yml"
MJCF_MOUNT_LINK_NAME: str = MOUNT_LINK_NAME
MJCF_MOUNT_COORD_SPEC = MOUNT_COORD_SPEC
MJCF_REST_Q: Dict[str, float] = dict(REST_Q)
