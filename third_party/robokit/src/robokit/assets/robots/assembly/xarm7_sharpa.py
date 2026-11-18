"""xArm7 + Sharpa Wave Hand assembly assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.assets.robots.arms.xarm7 import REST_Q as XARM7_REST_Q
from robokit.assets.robots.end_effectors.sharpa_hand import REST_Q_RIGHT as SHARPA_REST_Q_RIGHT


_DIR = fetch(
    [
        "robots/robot_description/assembly/xarm7_sharpa/**",
        "robots/robot_description/assembly/xarm6_sharpa/**",
        "robots/robot_description/arms/xarm7/**",
        "robots/collision_spheres/xarm7_sharpa/**",
        "robots/contact_points/sharpa_hand_right/**",
        "robots/self_collision/sharpa_hand_right/**",
        "robots/self_collision/xarm7_sharpa/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/assembly/xarm7_sharpa/xarm7_sharpa_right.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/xarm7_sharpa/collision_spheres.yaml"
HAND_CONTACT_POINTS_DIR: Path = _DIR / "robots/contact_points/sharpa_hand_right"
HAND_SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/sharpa_hand_right/ignore.sphere.yml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/xarm7_sharpa/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/xarm7_sharpa/ignore.sphere.yml"
REST_Q: Dict[str, float] = {**XARM7_REST_Q, **SHARPA_REST_Q_RIGHT}
