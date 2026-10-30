"""Versions and hashes for the actual local inputs and implementation."""

import hashlib
import importlib.metadata
from pathlib import Path
import torch
from retarget.config.run import ROOT


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(config, cases, output):
    versions = {}
    for name in (
        "torch",
        "warp-lang",
        "mujoco",
        "mujoco-warp",
        "numpy",
        "scipy",
        "viser",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    paths = [config.source]
    data = Path(config.task["data"])
    paths.extend(data.glob("*.yaml"))
    paths.extend(data.glob("*.npz"))
    paths.extend(data.glob("*.xml"))
    for case in cases:
        model = Path(case["model"])
        model = model if model.is_absolute() else output / model
        paths.append(model)
    return dict(
        versions=versions,
        gpu=torch.cuda.get_device_name(config.device),
        source_sha256={
            str(p.relative_to(ROOT)): sha256(p)
            for p in (ROOT / "retarget").rglob("*.py")
        },
        input_sha256={str(p): sha256(p) for p in paths if p.is_file()},
        cost_environment_scope="Installed local engines; source-equivalent costs do not imply bitwise upstream dynamics",
    )
