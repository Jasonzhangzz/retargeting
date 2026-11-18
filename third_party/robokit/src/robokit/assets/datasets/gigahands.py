"""Canonical GigaHands hand-object clips."""

from pathlib import Path

from robokit.assets import fetch


def data_root() -> Path:
    """Fetch the GigaHands hand-object clips and return their directory.

    >>> data_root().name  # doctest: +SKIP
    'gigahands'
    """
    return fetch(["datasets/gigahands/**"]) / "datasets/gigahands"
