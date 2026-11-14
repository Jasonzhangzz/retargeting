"""Unitree G1 + Inspire hand assembly assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.assets.robots.humanoids.unitree_g1 import REST_Q as G1_REST_Q
from robokit.utils.ee_coord_utils import HandCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/assembly/g1_inspire/**",
        "robots/robot_description/end_effectors/inspire_hand/**",  # assembly URDFs reference these meshes
        "robots/collision_spheres/g1_inspire/**",
        "robots/contact_points/g1_inspire/**",
        "robots/contact_avoid_points/g1_inspire/**",
        "robots/self_collision/g1_inspire/**",
        "robots/robot_description/humanoids/g1_inspire/**",
        "robots/robot_description/humanoids/g1/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/assembly/g1_inspire/g1_inspire.urdf"
URDF_UPPER_HEAD_PATH: Path = _DIR / "robots/robot_description/assembly/g1_inspire/g1_inspire_upper_head.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/g1_inspire/collision_spheres.yaml"
CONTACT_POINTS_PATH: Path = _DIR / "robots/contact_points/g1_inspire/contact_points.json"
CONTACT_AVOID_POINTS_PATH: Path = _DIR / "robots/contact_avoid_points/g1_inspire/contact_avoid_points.json"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/g1_inspire/ignore.capsule.yml"
URDF_NO_MIMIC_PATH: Path = _DIR / "robots/robot_description/assembly/g1_inspire/g1_inspire_no_mimic.urdf"
URDF_UPPER_HEAD_NO_MIMIC_PATH: Path = (
    _DIR / "robots/robot_description/assembly/g1_inspire/g1_inspire_upper_head_no_mimic.urdf"
)
HUMANOID_29DOF_HEAD_URDF_PATH: Path = _DIR / "robots/robot_description/humanoids/g1_inspire/urdf/g1_29dof_head.urdf"
HUMANOID_UPPER_HEAD_URDF_PATH: Path = (
    _DIR / "robots/robot_description/humanoids/g1_inspire/urdf/g1_inspire_upper_head.urdf"
)
HUMANOID_R_HAND_URDF_PATH: Path = _DIR / "robots/robot_description/humanoids/g1_inspire/urdf/r_hand_inspire.urdf"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/g1_inspire/ignore.sphere.yml"
EE_LINK_NAME_RIGHT: str = "R_hand_base_link"
EE_COORD_SPEC_RIGHT = HandCoordinateSpec(palm_forward_axis="-x", four_fingers_up_axis="-y")
"""The right Inspire hand of this assembly (Unitree's model, not `inspire_hand`), expressed in `EE_LINK_NAME_RIGHT`."""
REST_Q: Dict[str, float] = {
    **G1_REST_Q,
    **dict.fromkeys(
        (
            "L_thumb_proximal_yaw_joint",
            "L_thumb_proximal_pitch_joint",
            "L_index_proximal_joint",
            "L_middle_proximal_joint",
            "L_ring_proximal_joint",
            "L_pinky_proximal_joint",
            "R_thumb_proximal_yaw_joint",
            "R_thumb_proximal_pitch_joint",
            "R_index_proximal_joint",
            "R_middle_proximal_joint",
            "R_ring_proximal_joint",
            "R_pinky_proximal_joint",
        ),
        0.0,
    ),
    "L_thumb_proximal_pitch_joint": 0.14,
    "R_thumb_proximal_pitch_joint": 0.14,
}
