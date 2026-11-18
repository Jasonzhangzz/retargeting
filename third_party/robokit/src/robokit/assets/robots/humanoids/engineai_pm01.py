"""EngineAI PM01 assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import MountCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/humanoids/engineai_pm01/**",
        "robots/collision_spheres/engineai_pm01/**",
        "robots/self_collision/engineai_pm01/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/humanoids/engineai_pm01/engineai_pm01.urdf"
COLLISION_SPHERE_FULL_BODY_PATH: Path = _DIR / "robots/collision_spheres/engineai_pm01/full_body.yaml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/engineai_pm01/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/engineai_pm01/ignore.sphere.yml"
MOUNT_LINK_NAME_RIGHT: str = "LINK_ELBOW_END_R"
MOUNT_LINK_NAME_LEFT: str = "LINK_ELBOW_END_L"
MOUNT_COORD_SPEC_RIGHT = MountCoordinateSpec(outward_axis="-z", right_axis="-x")
MOUNT_COORD_SPEC_LEFT = MountCoordinateSpec(outward_axis="-z", right_axis="x")
REST_Q: Dict[str, float] = {
    "J00_HIP_PITCH_L": 0.0,
    "J01_HIP_ROLL_L": 0.0,
    "J02_HIP_YAW_L": 0.0,
    "J03_KNEE_PITCH_L": 0.0,
    "J04_ANKLE_PITCH_L": 0.0,
    "J05_ANKLE_ROLL_L": 0.0,
    "J06_HIP_PITCH_R": 0.0,
    "J07_HIP_ROLL_R": 0.0,
    "J08_HIP_YAW_R": 0.0,
    "J09_KNEE_PITCH_R": 0.0,
    "J10_ANKLE_PITCH_R": 0.0,
    "J11_ANKLE_ROLL_R": 0.0,
    "J12_WAIST_YAW": 0.0,
    "J13_SHOULDER_PITCH_L": 0.0,
    "J14_SHOULDER_ROLL_L": 0.0,
    "J15_SHOULDER_YAW_L": 0.0,
    "J16_ELBOW_PITCH_L": 0.0,
    "J17_ELBOW_YAW_L": 0.0,
    "J18_SHOULDER_PITCH_R": 0.0,
    "J19_SHOULDER_ROLL_R": 0.0,
    "J20_SHOULDER_YAW_R": 0.0,
    "J21_ELBOW_PITCH_R": 0.0,
    "J22_ELBOW_YAW_R": 0.0,
}
