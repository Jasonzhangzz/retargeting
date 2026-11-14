"""DexYCB hand-object dataset assets."""

from pathlib import Path

from robokit.assets import fetch


_DIR = fetch(["datasets/dexycb/**"])
DIR: Path = _DIR / "datasets/dexycb"
