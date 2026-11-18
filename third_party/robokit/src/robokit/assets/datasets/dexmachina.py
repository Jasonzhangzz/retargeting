"""Canonical DexMachina hand-object clips."""

from pathlib import Path

from robokit.assets import fetch


def data_root() -> Path:
    """Fetch the DexMachina hand-object clips and return their directory.

    >>> data_root().name  # doctest: +SKIP
    'dexmachina'
    """
    return fetch(["datasets/dexmachina/**"]) / "datasets/dexmachina"
