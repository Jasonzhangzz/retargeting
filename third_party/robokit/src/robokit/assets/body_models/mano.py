"""MANO hand body model assets."""

from pathlib import Path

from robokit.assets.body_models import get_body_models_dir


DIR: Path = get_body_models_dir() / "mano"
RIGHT_NPZ_PATH: Path = DIR / "MANO_RIGHT.npz"
LEFT_NPZ_PATH: Path = DIR / "MANO_LEFT.npz"
