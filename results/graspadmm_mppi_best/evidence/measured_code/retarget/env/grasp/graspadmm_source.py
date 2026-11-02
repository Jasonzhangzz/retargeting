"""Load the unmodified, pinned GraspADMM modules without unrelated operations."""

from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from retarget.config.run import ROOT

COMMIT = "d561ff9df8545deef7a3f9e7cfb4045ef66f6913"
UPSTREAM_ROOT = ROOT / "third_party/Dexonomy"


@lru_cache(maxsize=1)
def require_upstream():
    manifest = json.loads((UPSTREAM_ROOT / "GRASPADMM_UPSTREAM.json").read_text())
    if manifest.get("commit") != COMMIT:
        raise ValueError("Unexpected GraspADMM source revision")
    for relative, expected in manifest["sha256"].items():
        path = UPSTREAM_ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"GraspADMM source integrity check failed: {relative}")
    # Deliberately do not import the upstream op/__init__.py: it eagerly loads
    # unrelated trajectory, USD and viewer operations. The retained modules are
    # byte-for-byte original files in isolated namespace packages instead.
    for name in ("dexonomy", "dexonomy.op", "dexonomy.qp", "dexonomy.util", "dexonomy.data"):
        expected = UPSTREAM_ROOT.joinpath(*name.split("."))
        if name in sys.modules:
            paths = getattr(sys.modules[name], "__path__", ())
            if str(expected) not in paths:
                raise RuntimeError(f"Another Dexonomy package is already loaded: {name}")
            continue
        package = ModuleType(name)
        package.__path__ = [str(expected)]
        package.__package__ = name
        sys.modules[name] = package
    try:
        import clarabel  # noqa: F401
        import qpsolvers  # noqa: F401
        import rtree  # noqa: F401
        from omegaconf import OmegaConf

        core = importlib.import_module("dexonomy.op.gen_grasp_admm")
        initializer = importlib.import_module("dexonomy.op.gen_init")
        sim = importlib.import_module("dexonomy.sim")
        qp = importlib.import_module("dexonomy.qp.qp_single")
        traj = importlib.import_module("dexonomy.util.traj_util")
        obj_loader = importlib.import_module("dexonomy.data.obj_loader")
        hand_loader = importlib.import_module("dexonomy.data.hand_loader")
    except ImportError as error:
        raise RuntimeError("GraspADMM requires requirements-graspadmm.txt in .venv-graspadmm") from error
    return SimpleNamespace(
        core=core, init=initializer, sim=sim, qp=qp, traj=traj,
        obj_loader=obj_loader, hand_loader=hand_loader, OmegaConf=OmegaConf,
        init_config=OmegaConf.load(UPSTREAM_ROOT / "dexonomy/config/op/init.yaml"),
        grasp_config=OmegaConf.load(UPSTREAM_ROOT / "dexonomy/config/op/grasp_admm.yaml"),
        eval_config=OmegaConf.load(UPSTREAM_ROOT / "dexonomy/config/op/eval.yaml"),
        manifest=manifest,
    )
