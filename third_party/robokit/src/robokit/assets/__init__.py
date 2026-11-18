"""RoboKit asset access: import per-asset static paths for checked names; MANO/SMPL-X models remain local. The pinned `openrobokit/robokit` dataset has `robots/` and `datasets/` under three local roles:

- `assets/`: extension-preserving runtime paths created by `fetch`.
- `hf_cache/`: Hugging Face-managed blobs and snapshots.
- `hf_repo/`: writable git clone prepared by `checkout` for authoring.
"""

import fnmatch
import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional


logger = logging.getLogger("robokit")

HF_REPO_ID = "openrobokit/robokit"
ROBOKIT_HOME = Path.home() / ".robokit"
ASSETS_DIR = ROBOKIT_HOME / "assets"
HF_CACHE_DIR = ROBOKIT_HOME / "hf_cache"
HF_REPO_DIR = ROBOKIT_HOME / "hf_repo"
REVISION = "045838f22292b6931f7bc7490959925b1495a77c"


def fetch(
    patterns: Optional[List[str]] = None,
    revision: Optional[str] = None,
    repo_id: str = HF_REPO_ID,
    home: Optional[Path] = None,
) -> Path:
    """Fetch `patterns` at the pinned or supplied `revision` and return the asset root. Other HF dataset repos pass `repo_id`, a pinned `revision`, and their own `home` (default: `ROBOKIT_HOME`).

    Fetch sequence:
    1. Reuse completed local pattern sets without network access.
    2. List the revision's matching files and download missing content-addressed blobs to `<home>/hf_cache`.
    3. Hardlink every listed file under `<home>/assets/<revision>`, preserving loader-required extensions.
    4. Mark the pattern set complete; a file missing from the snapshot raises instead.

    Example:
        >>> d = fetch(["robots/collision_spheres/g1/**"])                          # doctest: +SKIP
        >>> spheres = d / "robots/collision_spheres/g1/collision_spheres.yaml"     # doctest: +SKIP
    """
    if patterns and any(p == "body_models" or p.startswith("body_models/") for p in patterns):
        raise ValueError(
            "Body models are user-provided, not fetched from HF. Use robokit.smplx.load_mano/load_smplx "
            "with ROBOKIT_BODY_MODELS or ~/.robokit/body_models. "
            "Download from https://mano.is.tue.mpg.de/ or https://smpl-x.is.tue.mpg.de/."
        )
    revision = revision or REVISION
    assets_dir, cache_dir = (home / "assets", home / "hf_cache") if home else (ASSETS_DIR, HF_CACHE_DIR)
    local = assets_dir / revision
    key = hashlib.sha1("\n".join(sorted(patterns) if patterns else ["(all)"]).encode()).hexdigest()
    marker = local / ".done" / key
    if marker.exists():  # the warm path: no Hub imports (~0.2 s a process)
        return local
    from filelock import FileLock
    from huggingface_hub import HfApi, snapshot_download

    lock = assets_dir / ".locks" / revision
    lock.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock):
        if marker.exists():
            return local
        logger.debug(f"Fetching {repo_id}@{revision} {patterns if patterns else '(all)'}")
        # List first: an unreachable Hub makes snapshot_download return a partial cached snapshot without error.
        expected = [
            path
            for path in HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision)
            if not path.startswith("body_models/")
            and (patterns is None or any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns))
        ]
        snapshot = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                allow_patterns=patterns,
                ignore_patterns=["body_models/**"],
                cache_dir=str(cache_dir),
            )
        )
        for relative in expected:
            target = local / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link((snapshot / relative).resolve(strict=True), target)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    return local


def checkout(target: Optional[Path] = None) -> Path:
    """Idempotently expose a writable asset checkout at `target`, cloning Git and materializing LFS on first use.

    Example:
        >>> checkout()                    # doctest: +SKIP
    """
    target = target if target is not None else Path.cwd() / "robokit_assets"
    if not HF_REPO_DIR.is_dir():
        import subprocess

        from huggingface_hub import get_token

        token = get_token()
        auth = f"hf:{token}@" if token else ""
        subprocess.run(
            ["git", "clone", f"https://{auth}huggingface.co/datasets/{HF_REPO_ID}", str(HF_REPO_DIR)],
            check=True,
        )
    if target.is_symlink():
        if target.resolve() == HF_REPO_DIR.resolve():
            logger.info(f"Symlink {target} already points to {HF_REPO_DIR}")
            return target
        target.unlink()  # repoint a stale symlink (e.g. the pre-refactor ~/.robokit/assets)
    target.symlink_to(HF_REPO_DIR)
    logger.info(f"Created symlink {target} -> {HF_REPO_DIR}")
    return target
