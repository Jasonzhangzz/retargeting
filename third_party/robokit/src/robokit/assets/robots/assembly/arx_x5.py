"""RoboTwin ARX X5A arm and gripper assets."""

from pathlib import Path

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import GripperCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/assembly/arx_x5/**",
        "robots/collision_spheres/arx_x5/**",
        "robots/contact_points/arx_x5/**",
        "robots/self_collision/arx_x5/**",
        "robots/robot_mj_description/assembly/arx_x5/**",
        "robots/collision_spheres/arx_x5_mj/**",
        "robots/contact_points/arx_x5_mj/**",
        "robots/self_collision/arx_x5_mj/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/assembly/arx_x5/arx_x5.urdf"
GRIPPER_URDF_PATH: Path = _DIR / "robots/robot_description/assembly/arx_x5/arx_x5_gripper.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/arx_x5/collision_spheres.yaml"
GRIPPER_COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/arx_x5/gripper.yaml"
CONTACT_POINTS_PATH: Path = _DIR / "robots/contact_points/arx_x5/contact_points.json"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/arx_x5/ignore.sphere.yml"
EE_COORD_SPEC = GripperCoordinateSpec(outward_axis="x", opening_axis="y")
# MuJoCo model of the same arm and gripper, as graspkit's MuJoCo benchmark simulates it.
MJCF_PATH: Path = _DIR / "robots/robot_mj_description/assembly/arx_x5/arx_x5.xml"
MJ_COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/arx_x5_mj/collision_spheres.yaml"
MJ_CONTACT_POINTS_PATH: Path = _DIR / "robots/contact_points/arx_x5_mj/contact_points.json"
MJ_SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/arx_x5_mj/ignore.sphere.yml"
