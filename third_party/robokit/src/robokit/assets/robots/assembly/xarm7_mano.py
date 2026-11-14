"""xArm7 + MANO hand assembly assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.assets.robots.arms.xarm7 import REST_Q as XARM7_REST_Q


_DIR = fetch(
    [
        "robots/robot_description/assembly/xarm7_mano/**",
        "robots/robot_description/arms/xarm7/**",
        "robots/robot_description/end_effectors/mano_hand/**",
        "robots/collision_spheres/xarm_mano/**",
        "robots/contact_avoid_points/xarm_mano/**",
        "robots/self_collision/xarm_mano/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/assembly/xarm7_mano/xarm7_mano_right_hand.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/xarm_mano/collision_spheres.yaml"
COLLISION_SPHERE_1_2_PATH: Path = _DIR / "robots/collision_spheres/xarm_mano/collision_spheres_1.2.yaml"
CONTACT_AVOID_POINTS_PATH: Path = _DIR / "robots/contact_avoid_points/xarm_mano/contact_avoid_points.json"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/xarm_mano/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/xarm_mano/ignore.sphere.yml"
REST_Q: Dict[str, float] = {
    **XARM7_REST_Q,
    **dict.fromkeys(
        (
            "mano_joint_1",
            "mano_joint_2",
            "mano_joint_3",
            "mano_joint_4",
            "mano_joint_5",
            "mano_joint_6",
            "mano_joint_7",
            "mano_joint_8",
            "mano_joint_9",
            "mano_joint_10",
            "mano_joint_11",
            "mano_joint_12",
            "mano_joint_empty",
            "mano_joint_13",
            "mano_joint_14",
            "mano_joint_15",
        ),
        0.0,
    ),
}
