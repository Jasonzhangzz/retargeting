"""SMPL-X body model assets."""

from pathlib import Path

from robokit.assets.body_models import get_body_models_dir


DIR: Path = get_body_models_dir() / "smplx"
NEUTRAL_NPZ_PATH: Path = DIR / "SMPLX_NEUTRAL.npz"
MALE_NPZ_PATH: Path = DIR / "SMPLX_MALE.npz"
FEMALE_NPZ_PATH: Path = DIR / "SMPLX_FEMALE.npz"
