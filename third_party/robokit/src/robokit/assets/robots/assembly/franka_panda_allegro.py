"""DexJoCo Franka Panda with a right Wonik Allegro hand, MuJoCo only."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(
    [
        "robots/robot_mj_description/assembly/franka_panda_allegro/**",
        "robots/collision_spheres/franka_panda_allegro_mj/**",
    ]
)
MJCF_PATH_RIGHT: Path = _DIR / "robots/robot_mj_description/assembly/franka_panda_allegro/panda_allegro_right.xml"
MJ_COLLISION_SPHERE_PATH_RIGHT: Path = _DIR / "robots/collision_spheres/franka_panda_allegro_mj/collision_spheres.yaml"
