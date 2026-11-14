"""xArm6 + Sharpa Wave hand assembly assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.assets.robots.arms.xarm6 import REST_Q as XARM6_REST_Q
from robokit.assets.robots.end_effectors.sharpa_hand import REST_Q_RIGHT as SHARPA_REST_Q_RIGHT


_DIR = fetch(
    [
        "robots/robot_description/assembly/xarm6_sharpa/**",
        "robots/collision_spheres/xarm6_sharpa/**",
        "robots/self_collision/xarm6_sharpa/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/assembly/xarm6_sharpa/xarm6_sharpa_right.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/xarm6_sharpa/collision_spheres.yaml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/xarm6_sharpa/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/xarm6_sharpa/ignore.sphere.yml"
REST_Q: Dict[str, float] = {**XARM6_REST_Q, **SHARPA_REST_Q_RIGHT}
