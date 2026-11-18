"""Unitree G1 + XHand assembly assets."""

from pathlib import Path
from typing import Dict

from robokit.assets import fetch
from robokit.assets.robots.humanoids.unitree_g1 import REST_Q as G1_REST_Q


_DIR = fetch(
    [
        "robots/collision_spheres/g1_xhand/**",
        "robots/contact_points/g1_xhand/**",
    ]
)
COLLISION_SPHERE_FIXED_PATH: Path = _DIR / "robots/collision_spheres/g1_xhand/collision_spheres_fixed.yaml"
CONTACT_POINTS_PATH: Path = _DIR / "robots/contact_points/g1_xhand/contact_points.json"
REST_Q: Dict[str, float] = dict(G1_REST_Q)
