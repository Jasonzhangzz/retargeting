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
        "cyipopt",
        "clarabel",
        "qpsolvers",
        "trimesh",
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
        if config.task.get("type") == "grasp":
            case_root = (output / case["trajectory"]).parent
            paths.extend(
                case_root / name
                for name in (
                    "native_scene/scene_manifest.json",
                    "initialization.json",
                    "primitive_seed_geometry.json",
                    "template.json",
                )
            )
    result = dict(
        versions=versions,
        gpu=torch.cuda.get_device_name(config.device)
        if str(config.device).startswith("cuda") and torch.cuda.is_available()
        else None,
        source_sha256={
            str(p.relative_to(ROOT)): sha256(p)
            for p in (ROOT / "retarget").rglob("*.py")
        },
        input_sha256={str(p): sha256(p) for p in paths if p.is_file()},
        cost_environment_scope="Installed local engines; source-equivalent costs do not imply bitwise upstream dynamics",
    )
    if config.solver.type == "graspadmm":
        import json

        manifest = ROOT / "third_party/Dexonomy/GRASPADMM_UPSTREAM.json"
        result["graspadmm"] = json.loads(manifest.read_text())
        result["graspadmm_source_manifest_sha256"] = sha256(manifest)
        result["cost_environment_scope"] = (
            "Pinned original GraspADMM ADMM/QP/filter/pregrasp/eval implementation; "
            "WUJI2 geometry, joint/frame, rigid pad ownership and input template adapters"
        )
    if config.solver.type == "graspadmm":
        result["evaluation_protocol"] = "graspadmm_native_v1"
        result["evaluation_runtime"] = str(ROOT / ".venv-graspadmm/bin/python")
    return result
