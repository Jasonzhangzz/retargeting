"""PND Adam Lite assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import MountCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/humanoids/pnd_adam_lite/**",
        "robots/collision_spheres/pnd_adam_lite/**",
        "robots/self_collision/pnd_adam_lite/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/humanoids/pnd_adam_lite/pnd_adam_lite.urdf"
COLLISION_SPHERE_FULL_BODY_PATH: Path = _DIR / "robots/collision_spheres/pnd_adam_lite/full_body.yaml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/pnd_adam_lite/ignore.capsule.yml"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/pnd_adam_lite/ignore.sphere.yml"
MOUNT_LINK_NAME_RIGHT: str = "wristYawRight"
MOUNT_LINK_NAME_LEFT: str = "wristYawLeft"
MOUNT_COORD_SPEC_RIGHT = MountCoordinateSpec(outward_axis="-z", right_axis="-x", mount_point_xyz=(0.0, 0.0, -0.3105))
MOUNT_COORD_SPEC_LEFT = MountCoordinateSpec(outward_axis="-z", right_axis="x", mount_point_xyz=(0.0, 0.0, -0.3105))
REST_Q: Dict[str, float] = {
    "hipPitch_Left": 0.0,
    "hipRoll_Left": 0.0,
    "hipYaw_Left": 0.0,
    "kneePitch_Left": 0.052,
    "anklePitch_Left": 0.0,
    "ankleRoll_Left": 0.0,
    "hipPitch_Right": 0.0,
    "hipRoll_Right": 0.0,
    "hipYaw_Right": 0.0,
    "kneePitch_Right": 0.052,
    "anklePitch_Right": 0.0,
    "ankleRoll_Right": 0.0,
    "waistRoll": 0.0,
    "waistPitch": 0.0,
    "waistYaw": 0.0,
    "shoulderPitch_Left": 0.0,
    "shoulderRoll_Left": 0.0,
    "shoulderYaw_Left": 0.0,
    "elbow_Left": 0.0,
    "wristYaw_Left": 0.0,
    "shoulderPitch_Right": 0.0,
    "shoulderRoll_Right": 0.0,
    "shoulderYaw_Right": 0.0,
    "elbow_Right": 0.0,
    "wristYaw_Right": 0.0,
}
