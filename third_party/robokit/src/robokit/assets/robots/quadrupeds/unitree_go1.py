"""Unitree Go1 quadruped assets."""

from pathlib import Path

from robokit.assets import fetch


_MJ_DIR = "robots/robot_mj_description/quadrupeds/go1"
MJCF_PATH: Path = fetch([f"{_MJ_DIR}/**"]) / f"{_MJ_DIR}/go1.xml"
