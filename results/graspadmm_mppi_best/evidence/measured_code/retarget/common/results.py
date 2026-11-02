"""Uniform result manifests, safe replacement and strict JSON diagnostics."""

import json, math, shutil
from pathlib import Path
import numpy as np


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def prepare_output(config):
    from retarget.config.run import ROOT

    path = config.destination
    if (
        path.parent != ROOT / "output"
        or (ROOT / "output").is_symlink()
        or path.is_symlink()
    ):
        raise ValueError("Output must be a real direct child of output/")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    import yaml

    (path / "config.yaml").write_text(
        yaml.safe_dump(config.resolved(), sort_keys=False)
    )
    write_json(path / "status.json", dict(state="preparing"))
    return path
