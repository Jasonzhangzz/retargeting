"""ParaHome motion data and watertight object collision meshes."""

from pathlib import Path

from robokit.assets import fetch


def data_root() -> Path:
    """Return the directory containing `seq` and `smplx_seq`."""
    return fetch(["datasets/parahome/seq/**", "datasets/parahome/smplx_seq/**"]) / "datasets/parahome"


def mesh_dir() -> Path:
    """Return the directory containing baked part meshes."""
    return fetch(["datasets/parahome/objects/**"]) / "datasets/parahome/objects"
