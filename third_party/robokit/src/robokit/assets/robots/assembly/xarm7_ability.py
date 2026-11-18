"""xArm7 + Ability Hand assembly assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.assets.robots.arms.xarm7 import REST_Q as XARM7_REST_Q
from robokit.assets.robots.end_effectors.ability_hand import REST_Q_RIGHT as ABILITY_REST_Q_RIGHT


_DIR = fetch(
    [
        "robots/robot_description/assembly/xarm7_ability/**",
        "robots/robot_description/arms/xarm7/**",
        "robots/robot_description/end_effectors/ability_hand/**",
        "robots/collision_spheres/xarm_ability/**",
        "robots/contact_avoid_points/xarm_ability/**",
        "robots/self_collision/xarm_ability/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/assembly/xarm7_ability/xarm7_ability_right_hand.urdf"
URDF_NO_MIMIC_PATH: Path = (
    _DIR / "robots/robot_description/assembly/xarm7_ability/xarm7_ability_right_hand_glb_no_mimic.urdf"
)
URDF_GLB_PATH: Path = _DIR / "robots/robot_description/assembly/xarm7_ability/xarm7_ability_right_hand_glb.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/xarm_ability/collision_spheres.yaml"
CONTACT_AVOID_POINTS_PATH: Path = _DIR / "robots/contact_avoid_points/xarm_ability/contact_avoid_points.json"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/xarm_ability/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/xarm_ability/ignore.sphere.yml"
REST_Q: Dict[str, float] = {**XARM7_REST_Q, **ABILITY_REST_Q_RIGHT}
