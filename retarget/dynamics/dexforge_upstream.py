"""Load the vendored, byte-verified DexForge ForceAware implementation lazily."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys


COMMIT = "fa6d671948ef03d18514062748b94e445b63cd49"
ROOT = Path(__file__).resolve().parents[2] / "third_party" / "DexForge"


@lru_cache(maxsize=1)
def verify_upstream():
    """Fail closed if a pinned source file has changed or disappeared."""
    manifest = json.loads((ROOT / "UPSTREAM.json").read_text())
    if manifest["commit"] != COMMIT:
        raise RuntimeError("DexForge source commit differs from the pinned version")
    mismatches = []
    for name, expected in manifest["sha256"].items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            mismatches.append(name)
    if mismatches:
        raise RuntimeError(f"Modified DexForge upstream files: {mismatches}")
    return dict(repository=manifest["repository"], commit=COMMIT,
                verified_files=len(manifest["sha256"]), sha256=manifest["sha256"])


def activate_upstream():
    """Use only this repository's pinned packages, never an external Panda fork."""
    verify_upstream()
    paths = (ROOT / "forceaware", ROOT / "forceaware/third_party/comfree_warp", ROOT)
    for package, directory in zip(("forceaware", "comfree_warp", "tools"), paths):
        loaded = sys.modules.get(package)
        if loaded is not None:
            origin = Path(loaded.__file__).resolve()
            if not origin.is_relative_to(directory):
                raise RuntimeError(f"{package} already imported from a different checkout: {origin}")
    for path in reversed(paths):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    return ROOT


def require_environment():
    """Check the isolated numerical stack before compiling native dynamics."""
    from importlib.metadata import PackageNotFoundError, version

    expected = {"numpy": "1.26.4", "scipy": "1.13.1", "torch": "2.1.2",
                "warp-lang": "1.15.0", "mujoco": "3.6.0"}
    problems = []
    if sys.version_info[:2] != (3, 10):
        problems.append(f"Python {sys.version_info.major}.{sys.version_info.minor} (need 3.10)")
    versions = {}
    for package, wanted in expected.items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            actual = "missing"
        versions[package] = actual
        if actual.split("+")[0] != wanted:
            problems.append(f"{package}={actual} (need {wanted})")
    if problems:
        raise RuntimeError("DexForge requires its isolated environment: " + "; ".join(problems)
                           + ". Run scripts/setup_forceaware.sh and use .venv-forceaware/bin/python.")
    return versions
