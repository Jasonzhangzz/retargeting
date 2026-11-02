"""Dispatch only the new task; leave upstream task implementations unchanged."""
from pathlib import Path
import sys


def option(name):
    for i, value in enumerate(sys.argv[1:], 1):
        if value == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return None


def dispatch_train():
    if option("--task") != "hand2-grasp-hold":
        return False
    from .train import main
    main()
    return True


def dispatch_train_config(cfg, folder=None):
    if getattr(cfg, "task", None) != "hand2-grasp-hold":
        return False
    from .train import GraspRunner, TrainConfig
    from .environment import make_env_cfg
    values = {name: getattr(cfg, name) for name in TrainConfig.__dataclass_fields__ if hasattr(cfg, name)}
    values["manifest"] = values.get("manifest") or make_env_cfg()["manifest"]
    if folder is not None:
        values["log_dir"] = str(folder)
    GraspRunner(TrainConfig(**values)).learn()
    return True


def dispatch_play():
    path = option("--checkpoint")
    if not path or not Path(path).expanduser().is_file():
        return False
    import torch
    checkpoint = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    if checkpoint.get("config", {}).get("task") != "hand2-grasp-hold":
        return False
    from .play import main
    main()
    return True
