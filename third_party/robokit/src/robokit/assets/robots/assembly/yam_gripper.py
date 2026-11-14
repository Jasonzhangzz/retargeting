"""YAM arm with flexible 4310 gripper assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.assets.robots.arms.yam import REST_Q as YAM_REST_Q
from robokit.assets.robots.end_effectors.yam_gripper import REST_Q as GRIPPER_REST_Q


_DIR = fetch(
    [
        "robots/robot_description/assembly/yam_gripper/**",
        "robots/collision_spheres/yam_flexible_4310/**",
        "robots/self_collision/yam_flexible_4310/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/assembly/yam_gripper/yam_flexible_4310.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/yam_flexible_4310/collision_spheres.yaml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/yam_flexible_4310/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/yam_flexible_4310/ignore.sphere.yml"
REST_Q: Dict[str, float] = {**YAM_REST_Q, **GRIPPER_REST_Q}
