"""xArm7 + XHand assembly assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.assets.robots.arms.xarm7 import REST_Q as XARM7_REST_Q
from robokit.assets.robots.end_effectors.xhand import MJCF_REST_Q_RIGHT as XHAND_MJCF_REST_Q_RIGHT
from robokit.assets.robots.end_effectors.xhand import REST_Q_RIGHT as XHAND_REST_Q_RIGHT


_DIR = fetch(
    [
        "robots/robot_description/assembly/xarm7_xhand/**",
        "robots/robot_description/arms/xarm7/**",
        "robots/robot_description/end_effectors/xhand/**",
        "robots/collision_spheres/xarm_xhand/**",
        "robots/contact_points/xarm_xhand/**",
        "robots/contact_avoid_points/xarm_xhand/**",
        "robots/self_collision/xarm_xhand/**",
        "robots/robot_mj_description/assembly/xarm7_xhand/**",
        "robots/self_collision/xarm_xhand_mj/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/assembly/xarm7_xhand/xarm_xhand_righthand.urdf"
URDF_GLB_PATH: Path = _DIR / "robots/robot_description/assembly/xarm7_xhand/xarm7_xhand_right_hand_glb.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/xarm_xhand/collision_spheres.yaml"
CONTACT_POINTS_PATH: Path = _DIR / "robots/contact_points/xarm_xhand/contact_points.json"
CONTACT_POINTS_PINCH_V2_PATH: Path = _DIR / "robots/contact_points/xarm_xhand/contact_points_pinch_v2.json"
CONTACT_AVOID_POINTS_PATH: Path = _DIR / "robots/contact_avoid_points/xarm_xhand/contact_avoid_points.json"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/xarm_xhand/ignore.capsule.yml"
MJ_ASSEMBLY_DIR: Path = _DIR / "robots/robot_mj_description/assembly/xarm7_xhand"
MJCF_PATH: Path = MJ_ASSEMBLY_DIR / "xarm7_xhand.xml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/xarm_xhand/ignore.sphere.yml"
MJ_CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/xarm_xhand_mj/ignore.capsule.yml"
REST_Q: Dict[str, float] = {**XARM7_REST_Q, **XHAND_REST_Q_RIGHT}
MJCF_REST_Q: Dict[str, float] = {**XARM7_REST_Q, **XHAND_MJCF_REST_Q_RIGHT}
