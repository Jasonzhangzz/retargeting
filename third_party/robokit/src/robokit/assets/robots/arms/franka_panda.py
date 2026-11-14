"""Franka Emika Panda assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import GripperCoordinateSpec, MountCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/arms/franka_panda/**",
        "robots/collision_spheres/franka_panda/**",
        "robots/self_collision/franka_panda/**",
        "robots/self_collision/franka_panda_mj/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/arms/franka_panda/panda_v2.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/franka_panda/collision_spheres.yaml"
ARM_URDF_PATH: Path = _DIR / "robots/robot_description/arms/franka_panda/panda_arm.urdf"
ARM_COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/franka_panda/arm.yaml"
ARM_MOUNT_LINK_NAME: str = "panda_link8"
ARM_MOUNT_COORD_SPEC = MountCoordinateSpec(outward_axis="z", right_axis="y")
HAND_LINK_NAME: str = "panda_hand"
HAND_EE_COORD_SPEC = GripperCoordinateSpec(outward_axis="z", opening_axis="y")
"""The Franka Hand of `URDF_PATH`, expressed in `HAND_LINK_NAME`."""
ARM_REST_Q: Dict[str, float] = {
    **dict.fromkeys(
        (
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ),
        0.0,
    ),
    "panda_joint4": -1.57079,
    "panda_joint6": 1.57079,
    "panda_joint7": 0.0,
}
REST_Q: Dict[str, float] = {**ARM_REST_Q, "panda_finger_joint1": 0.04}
GRIPPER_URDF_PATH: Path = _DIR / "robots/robot_description/arms/franka_panda/panda_v2_gripper.urdf"
GRIPPER_NO_MIMIC_URDF_PATH: Path = _DIR / "robots/robot_description/arms/franka_panda/panda_v2_no_mimic.urdf"
URDF_NO_MIMIC_PATH: Path = _DIR / "robots/robot_description/arms/franka_panda/panda_v2_no_mimic.urdf"

# MuJoCo Menagerie's bare arm, for pairing with an MJCF end-effector.
_MJ_DIR = fetch(["robots/robot_mj_description/arms/franka_panda/**"])
MJCF_PATH: Path = _MJ_DIR / "robots/robot_mj_description/arms/franka_panda/panda_nohand.xml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/franka_panda/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/franka_panda/ignore.sphere.yml"
MJ_CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/franka_panda_mj/ignore.capsule.yml"
MJCF_MOUNT_LINK_NAME: str = "attachment"
MJCF_MOUNT_COORD_SPEC = MountCoordinateSpec(outward_axis="z", right_axis="x")
MJCF_REST_Q: Dict[str, float] = {name[len("panda_") :]: value for name, value in ARM_REST_Q.items()}
