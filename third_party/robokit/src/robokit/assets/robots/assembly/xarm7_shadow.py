"""xArm7 + Shadow Hand assembly assets. MJCF only: the pairing was authored in MuJoCo."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.assets.robots.arms.xarm7 import REST_Q as XARM7_REST_Q


_DIR = fetch(
    [
        "robots/robot_mj_description/assembly/xarm7_shadow/**",
        "robots/collision_spheres/xarm7_shadow_mj/**",
    ]
)
MJ_ASSEMBLY_DIR: Path = _DIR / "robots/robot_mj_description/assembly/xarm7_shadow"
MJ_COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/xarm7_shadow_mj/collision_spheres.yaml"
REST_Q: Dict[str, float] = dict(XARM7_REST_Q)
