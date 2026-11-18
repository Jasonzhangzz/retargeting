"""dex-retargeting motion clips assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(["datasets/dex_retargeting/**"])
DIR: Path = _DIR / "datasets/dex_retargeting"
DEXYCB_MOTION_PATH: Path = DIR / "dexycb_motion.pkl"
