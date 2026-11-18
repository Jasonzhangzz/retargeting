"""Canonical OakInk2 hand-object clips."""

from pathlib import Path

from robokit.assets import fetch


def data_root() -> Path:
    """Fetch the OakInk2 hand-object clips and return their directory.

    >>> data_root().name  # doctest: +SKIP
    'oakink2'
    """
    return fetch(["datasets/oakink2/**"]) / "datasets/oakink2"
