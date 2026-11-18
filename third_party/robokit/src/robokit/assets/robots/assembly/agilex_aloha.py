"""RoboTwin AgileX ALOHA front-right arm and gripper assets."""

from pathlib import Path

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import GripperCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/assembly/agilex_aloha/**",
        "robots/collision_spheres/agilex_aloha/**",
        "robots/contact_points/agilex_aloha/**",
        "robots/self_collision/agilex_aloha/**",
        "robots/robot_mj_description/assembly/agilex_aloha/**",
        "robots/collision_spheres/agilex_aloha_mj/**",
        "robots/contact_points/agilex_aloha_mj/**",
        "robots/self_collision/agilex_aloha_mj/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/assembly/agilex_aloha/agilex_aloha.urdf"
GRIPPER_URDF_PATH: Path = _DIR / "robots/robot_description/assembly/agilex_aloha/agilex_aloha_gripper.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/agilex_aloha/collision_spheres.yaml"
GRIPPER_COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/agilex_aloha/gripper.yaml"
CONTACT_POINTS_PATH: Path = _DIR / "robots/contact_points/agilex_aloha/contact_points.json"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/agilex_aloha/ignore.sphere.yml"
EE_COORD_SPEC = GripperCoordinateSpec(outward_axis="x", opening_axis="y")
# MuJoCo model of the same arm and gripper, as graspkit's MuJoCo benchmark simulates it.
MJCF_PATH: Path = _DIR / "robots/robot_mj_description/assembly/agilex_aloha/agilex_aloha.xml"
MJ_COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/agilex_aloha_mj/collision_spheres.yaml"
MJ_CONTACT_POINTS_PATH: Path = _DIR / "robots/contact_points/agilex_aloha_mj/contact_points.json"
MJ_SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/agilex_aloha_mj/ignore.sphere.yml"
