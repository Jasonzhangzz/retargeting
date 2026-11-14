"""Barrett Hand assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.utils.ee_coord_utils import GripperCoordinateSpec


_DIR = fetch(
    [
        "robots/robot_description/end_effectors/barrett_hand/**",
        "robots/collision_spheres/barrett_hand/**",
        "robots/contact_points/barrett_hand/**",
        "robots/self_collision/barrett_hand/**",
    ]
)
URDF_PATH: Path = _DIR / "robots/robot_description/end_effectors/barrett_hand/bhand_model.urdf"
URDF_GLB_PATH: Path = _DIR / "robots/robot_description/end_effectors/barrett_hand/bhand_model_glb_fixed_base.urdf"
URDF_GLB_FLOATING_BASE_NO_MIMIC_PATH: Path = (
    _DIR / "robots/robot_description/end_effectors/barrett_hand/barrett_hand_glb_floating_base_no_mimic.urdf"
)
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/barrett_hand/collision_spheres.yaml"
CONTACT_POINTS_PATH: Path = _DIR / "robots/contact_points/barrett_hand/contact_points.json"
SPHERE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/barrett_hand/ignore.sphere.yml"
CAPSULE_SELF_COLLISION_IGNORE_PATH: Path = _DIR / "robots/self_collision/barrett_hand/ignore.capsule.yml"
MOUNT_LINK_NAME: str = "base_link"
EE_COORD_SPEC = GripperCoordinateSpec(outward_axis="z", opening_axis="y")
MOUNT_COORD_SPEC = EE_COORD_SPEC.mount_spec
REST_Q: Dict[str, float] = dict.fromkeys(
    (
        "finger_1_prox_joint",
        "finger_1_med_joint",
        "finger_1_dist_joint",
        "finger_2_prox_joint",
        "finger_2_med_joint",
        "finger_2_dist_joint",
        "finger_3_med_joint",
        "finger_3_dist_joint",
    ),
    0.0,
)
