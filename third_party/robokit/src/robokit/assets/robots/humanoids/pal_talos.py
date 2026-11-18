"""PAL Talos assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import MountCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/humanoids/pal_talos/**",
        "robots/collision_spheres/pal_talos/**",
        "robots/self_collision/pal_talos/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/humanoids/pal_talos/pal_talos.urdf"
COLLISION_SPHERE_FULL_BODY_PATH: Path = _DIR / "robots/collision_spheres/pal_talos/full_body.yaml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/pal_talos/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/pal_talos/ignore.sphere.yml"
MOUNT_LINK_NAME_RIGHT: str = "arm_right_7_link"
MOUNT_LINK_NAME_LEFT: str = "arm_left_7_link"
MOUNT_COORD_SPEC_RIGHT = MountCoordinateSpec(outward_axis="-z", right_axis="-x", mount_point_xyz=(0.0, 0.0, -0.0808))
MOUNT_COORD_SPEC_LEFT = MountCoordinateSpec(outward_axis="-z", right_axis="x", mount_point_xyz=(0.0, 0.0, -0.0808))
REST_Q: Dict[str, float] = {
    "torso_1_joint": 0.0,
    "torso_2_joint": 0.0,
    "head_1_joint": 0.0,
    "head_2_joint": 0.0,
    "arm_left_1_joint": 0.0,
    "arm_left_2_joint": 0.00872664625997,
    "arm_left_3_joint": 0.0,
    "arm_left_4_joint": 0.0,
    "arm_left_5_joint": 0.0,
    "arm_left_6_joint": 0.0,
    "arm_left_7_joint": 0.0,
    "arm_right_1_joint": 0.0,
    "arm_right_2_joint": -0.00872664625997,
    "arm_right_3_joint": 0.0,
    "arm_right_4_joint": 0.0,
    "arm_right_5_joint": 0.0,
    "arm_right_6_joint": 0.0,
    "arm_right_7_joint": 0.0,
    "gripper_left_joint": 0.0,
    "gripper_right_joint": 0.0,
    "leg_left_1_joint": 0.0,
    "leg_left_2_joint": 0.0,
    "leg_left_3_joint": 0.0,
    "leg_left_4_joint": 0.0,
    "leg_left_5_joint": 0.0,
    "leg_left_6_joint": 0.0,
    "leg_right_1_joint": 0.0,
    "leg_right_2_joint": 0.0,
    "leg_right_3_joint": 0.0,
    "leg_right_4_joint": 0.0,
    "leg_right_5_joint": 0.0,
    "leg_right_6_joint": 0.0,
}
