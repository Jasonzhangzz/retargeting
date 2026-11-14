"""Climbing MoCap sequences (joint positions + the box terrain each was captured on)."""

from pathlib import Path

from robokit.assets import fetch


def sequence_dir(name: str = "mocap_climb_seq_0") -> Path:
    """Directory holding one sequence's `.npy` joint positions and `multi_boxes.obj` terrain."""
    return fetch([f"datasets/climb/{name}/**"]) / f"datasets/climb/{name}"
