"""D'Claw Gripper assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import GripperCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/end_effectors/dclaw_gripper/**",
        "robots/collision_spheres/dclaw_gripper/**",
        "robots/contact_points/dclaw_gripper/**",
        "robots/self_collision/dclaw_gripper/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/end_effectors/dclaw_gripper/dclaw_gripper.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/dclaw_gripper/collision_spheres.yaml"
CONTACT_POINTS_PATH: Path = _DIR / "robots/contact_points/dclaw_gripper/contact_points.json"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/dclaw_gripper/ignore.sphere.yml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/dclaw_gripper/ignore.capsule.yml"
MOUNT_LINK_NAME: str = "base"
EE_COORD_SPEC = GripperCoordinateSpec(outward_axis="z", opening_axis="-y")
MOUNT_COORD_SPEC = EE_COORD_SPEC.mount_spec
REST_Q: Dict[str, float] = dict.fromkeys(
    (
        "joint_f1_0",
        "joint_f1_1",
        "joint_f1_2",
        "joint_f2_0",
        "joint_f2_1",
        "joint_f2_2",
        "joint_f3_0",
        "joint_f3_1",
        "joint_f3_2",
    ),
    0.0,
)
