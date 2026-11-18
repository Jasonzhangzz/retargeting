"""Unitree G1 + MANO hand assembly assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.assets.robots.humanoids.unitree_g1 import REST_Q as G1_REST_Q


_DIR = fetch(
    [
        "robots/robot_description/assembly/g1_mano/**",
        "robots/robot_description/humanoids/g1/**",
        "robots/robot_description/end_effectors/mano_hand/**",
        "robots/collision_spheres/g1_mano/**",
        "robots/contact_points/g1_mano/**",
        "robots/contact_avoid_points/g1_mano/**",
        "robots/self_collision/g1_mano/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/assembly/g1_mano/g1_mano.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/g1_mano/collision_spheres.yaml"
CONTACT_POINTS_PATH: Path = _DIR / "robots/contact_points/g1_mano/contact_points.json"
CONTACT_AVOID_POINTS_PATH: Path = _DIR / "robots/contact_avoid_points/g1_mano/contact_avoid_points.json"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/g1_mano/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/g1_mano/ignore.sphere.yml"
REST_Q: Dict[str, float] = {
    **G1_REST_Q,
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
