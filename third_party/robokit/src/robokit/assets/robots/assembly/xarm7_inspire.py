"""xArm7 + Inspire Hand assembly assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.assets.robots.arms.xarm7 import REST_Q as XARM7_REST_Q
from robokit.assets.robots.end_effectors.inspire_hand import REST_Q_RIGHT as INSPIRE_REST_Q_RIGHT


_DIR = fetch(
    [
        "robots/robot_description/assembly/xarm7_inspire/**",
        "robots/robot_description/arms/xarm7/**",
        "robots/robot_description/end_effectors/inspire_hand/**",
        "robots/collision_spheres/inspire_hand_right/**",
        "robots/contact_points/inspire_hand_right/**",
        "robots/contact_avoid_points/inspire_hand_right/**",
        "robots/self_collision/inspire_hand_right/**",
        "robots/self_collision/xarm_inspire/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/assembly/xarm7_inspire/xarm7_inspire_right.urdf"
HAND_COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/inspire_hand_right/collision_spheres.yaml"
HAND_CONTACT_POINTS_PATH: Path = _DIR / "robots/contact_points/inspire_hand_right/contact_points.json"
HAND_CONTACT_AVOID_POINTS_PATH: Path = _DIR / "robots/contact_avoid_points/inspire_hand_right/contact_avoid_points.json"
HAND_CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/inspire_hand_right/ignore.capsule.yml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/xarm_inspire/ignore.capsule.yml"
REST_Q: Dict[str, float] = {**XARM7_REST_Q, **INSPIRE_REST_Q_RIGHT}
