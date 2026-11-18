"""NIST Assembly Task Board 1 assets from AssembleNIST: six task scenes, point tables and reset banks."""

from pathlib import Path

from robokit.assets import fetch


DIR: Path = fetch(["datasets/nist_assembly/**"]) / "datasets/nist_assembly"
