"""Unitree H1 + Inspire hand assembly assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.assets.robots.humanoids.unitree_h1_2 import REST_Q as H1_REST_Q


_DIR = fetch(
    [
        "robots/collision_spheres/h1_inspire_hand/**",
        "robots/contact_points/h1_inspire_hand/**",
        "robots/contact_avoid_points/h1_inspire_hand/**",
    ]
)
COLLISION_SPHERE_PATH: Path = _DIR / "robots/collision_spheres/h1_inspire_hand/collision_spheres.yaml"
CONTACT_POINTS_PATH: Path = _DIR / "robots/contact_points/h1_inspire_hand/contact_points.json"
CONTACT_AVOID_POINTS_PATH: Path = _DIR / "robots/contact_avoid_points/h1_inspire_hand/contact_avoid_points.json"
REST_Q: Dict[str, float] = dict(H1_REST_Q)
