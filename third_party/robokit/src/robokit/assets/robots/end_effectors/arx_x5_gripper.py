"""RoboDojo's ARX X5 gripper (MIT): X5A.urdf cut at link6, TCP 0.145 m along +x, with head-camera calibration."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import GripperCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/end_effectors/arx_x5_gripper/**",
        "robots/collision_spheres/arx_x5_gripper/**",
        "robots/contact_points/arx_x5_gripper/**",
        "robots/self_collision/arx_x5_gripper/**",
        "calibration/robodojo_x5/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/end_effectors/arx_x5_gripper/arx_x5_gripper.urdf"
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/arx_x5_gripper/collision_spheres.yaml"
CONTACT_POINTS_PATH: Path = _DIR / "robots/contact_points/arx_x5_gripper/contact_points.json"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/arx_x5_gripper/ignore.sphere.yml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/arx_x5_gripper/ignore.capsule.yml"
HEAD_CAMERA_POSE_PATH: Path = _DIR / "calibration/robodojo_x5/cam_head_pose.json"
HEAD_CAMERA_INTRINSIC_PATH: Path = _DIR / "calibration/robodojo_x5/cam_head_intrinsic.json"
MOUNT_LINK_NAME: str = "link6"
EE_COORD_SPEC = GripperCoordinateSpec(outward_axis="x", opening_axis="y")
MOUNT_COORD_SPEC = EE_COORD_SPEC.mount_spec
REST_Q: Dict[str, float] = {"joint7": 0.044}
