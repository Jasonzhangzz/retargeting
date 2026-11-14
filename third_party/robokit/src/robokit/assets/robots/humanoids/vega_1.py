"""DexMate Vega-1 humanoid assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import MountCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/humanoids/vega_1/**",
        "robots/collision_spheres/vega_1/**",
        "robots/self_collision/vega_1/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/humanoids/vega_1/vega_1.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/vega_1/collision_spheres.yaml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/vega_1/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/vega_1/ignore.sphere.yml"
MOUNT_LINK_NAME_RIGHT: str = "R_ee"
MOUNT_LINK_NAME_LEFT: str = "L_ee"
MOUNT_COORD_SPEC_RIGHT = MountCoordinateSpec(outward_axis="z", right_axis="-y", mount_point_xyz=(0.0, 0.0, 0.029))
MOUNT_COORD_SPEC_LEFT = MountCoordinateSpec(outward_axis="z", right_axis="-y", mount_point_xyz=(0.0, 0.0, 0.029))
REST_Q: Dict[str, float] = {
    **dict.fromkeys(
        (
            "B_wheel_j1",
            "B_wheel_j2",
            "R_wheel_j1",
            "R_wheel_j2",
            "L_wheel_j1",
            "L_wheel_j2",
            "torso_j1",
            "torso_j2",
            "torso_j3",
            "head_j1",
            "head_j2",
            "head_j3",
            "L_arm_j1",
            "L_arm_j2",
            "L_arm_j3",
            "L_arm_j4",
            "L_arm_j5",
            "L_arm_j6",
            "L_arm_j7",
            "R_arm_j1",
            "R_arm_j2",
            "R_arm_j3",
            "R_arm_j4",
            "R_arm_j5",
            "R_arm_j6",
            "R_arm_j7",
        ),
        0.0,
    ),
    "torso_j1": 0.8997,
    "torso_j2": 1.5697,
    "torso_j3": 0.0993,
    "head_j1": 0.2062,
    "head_j2": -0.0269,
    "head_j3": 0.0154,
    "L_arm_j1": 0.8399,
    "L_arm_j2": 0.5101,
    "L_arm_j3": 0.3701,
    "L_arm_j4": -1.3,
    "L_arm_j5": -0.6501,
    "L_arm_j6": -0.29,
    "L_arm_j7": -0.0301,
    "R_arm_j1": -0.8399,
    "R_arm_j2": -0.51,
    "R_arm_j3": -0.37,
    "R_arm_j4": -1.3,
    "R_arm_j5": 0.65,
    "R_arm_j6": 0.29,
    "R_arm_j7": 0.0301,
}
