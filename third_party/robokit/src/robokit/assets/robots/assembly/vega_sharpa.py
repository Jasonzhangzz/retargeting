"""Dexmate Vega-1 + bimanual Sharpa Wave hand assets; the combined URDF spans three subtrees, so fetch all three."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.assets.robots.end_effectors.sharpa_hand import REST_Q_LEFT as SHARPA_REST_Q_LEFT
from robokit.assets.robots.end_effectors.sharpa_hand import REST_Q_RIGHT as SHARPA_REST_Q_RIGHT
from robokit.assets.robots.humanoids.vega_1 import REST_Q as VEGA_REST_Q


_DIR = fetch(
    [
        "robots/robot_description/assembly/vega_sharpa/**",
        "robots/robot_description/humanoids/vega_1/**",
        "robots/robot_description/end_effectors/sharpa_hand/**",
        "robots/robot_mj_description/assembly/vega_sharpa/**",
        "robots/collision_spheres/vega_sharpa/**",
        "robots/collision_spheres/vega_sharpa_mj/**",
        "robots/contact_points/vega_sharpa_mj/**",
        "robots/contact_points/vega_sharpa_mj_left/**",
        "robots/self_collision/vega_sharpa/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/assembly/vega_sharpa/vega_sharpa.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/vega_sharpa/collision_spheres.yaml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/vega_sharpa/ignore.sphere.yml"
MJ_ASSEMBLY_DIR: Path = _DIR / "robots/robot_mj_description/assembly/vega_sharpa"
MJ_COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/vega_sharpa_mj/collision_spheres.yaml"
MJ_CONTACT_POINTS_DIR: Path = _DIR / "robots/contact_points/vega_sharpa_mj"
MJ_CONTACT_POINTS_LEFT_DIR: Path = _DIR / "robots/contact_points/vega_sharpa_mj_left"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/vega_sharpa/ignore.capsule.yml"
REST_Q: Dict[str, float] = {**VEGA_REST_Q, **SHARPA_REST_Q_RIGHT, **SHARPA_REST_Q_LEFT}
