"""OMOMO human-object interaction clips and object meshes (InterMimic-processed)."""

from pathlib import Path
from typing import List

from robokit.assets import HF_REPO_ID, REVISION, fetch


NAMES = (
    "clothesstand",
    "floorlamp",
    "largebox",
    "largetable",
    "monitor",
    "plasticbox",
    "smallbox",
    "smalltable",
    "suitcase",
    "trashcan",
    "tripod",
    "whitechair",
    "woodchair",
)


def clip_names() -> List[str]:
    """All clip names, listed over the HF API — no download (the full set is ~1.8 GB)."""
    from huggingface_hub import HfApi

    entries = HfApi().list_repo_tree(
        repo_id=HF_REPO_ID, repo_type="dataset", revision=REVISION, path_in_repo="datasets/omomo", recursive=False
    )
    return sorted(Path(e.path).stem for e in entries if e.path.endswith(".pt"))


def clip_path(name: str) -> Path:
    """Path to one clip `.pt` (e.g. `sub3_largebox_003`); fetches just that clip."""
    return fetch([f"datasets/omomo/{name}.pt"]) / f"datasets/omomo/{name}.pt"


def height_dict_path() -> Path:
    """Path to the per-subject height table the clips are scaled against."""
    return fetch(["datasets/omomo/height_dict.pkl"]) / "datasets/omomo/height_dict.pkl"


def mesh_path(name: str) -> Path:
    """Path to one prop's `.obj`; fetches just that object."""
    assert name in NAMES, f"unknown OMOMO object {name!r}; expected one of {NAMES}"
    return fetch([f"datasets/omomo/objects/{name}/**"]) / f"datasets/omomo/objects/{name}/{name}.obj"
