#!/usr/bin/env python3
"""Install the local adapter and guarded entry-point patches into Dex-Supermimic."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sysconfig


def install(retarget, dex):
    retarget, dex = Path(retarget).resolve(), Path(dex).resolve()
    backup = dex / "logs/grasp_migration" / datetime.now().strftime("backup_%Y%m%d_%H%M%S")
    backup.mkdir(parents=True)
    changed = {}
    def update(relative, replacements):
        path = dex / relative
        source = path.read_text()
        original = source
        for old, new in replacements:
            if new in source:
                continue
            if old not in source:
                raise ValueError(f"Upstream changed; cannot safely patch {path}: {old[:70]}")
            source = source.replace(old, new, 1)
        if source != original:
            dest = backup / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            path.write_text(source)
            changed[relative] = dict(before=hashlib.sha256(original.encode()).hexdigest(),
                                     after=hashlib.sha256(source.encode()).hexdigest())
    base = "src/dex_supermimic/"
    update(base + "rl/dex_supermimic/cli.py", [
        ("def train():\n    cfg =", "def train():\n    from retarget.integrations.supermimic.bridge import dispatch_train\n    if dispatch_train():\n        return\n    cfg ="),
        ("def play():\n    cfg =", "def play():\n    from retarget.integrations.supermimic.bridge import dispatch_play\n    if dispatch_play():\n        return\n    cfg ="),
        ("def train_config(cfg, folder=None):\n", "def train_config(cfg, folder=None):\n    from retarget.integrations.supermimic.bridge import dispatch_train_config\n    if dispatch_train_config(cfg, folder):\n        return\n"),
    ])
    task = base + "tasks/dex_reorient/config/task.py"
    update(task, [
        ('TaskName = Literal["hand2-reorient", "hand1-reorient", "hand2-pen-spin"]',
         'TaskName = Literal["hand2-reorient", "hand1-reorient", "hand2-pen-spin", "hand2-grasp-hold"]'),
        ('    if task == "hand2-pen-spin":',
         '    if task == "hand2-grasp-hold":\n        from retarget.integrations.supermimic.environment import make_env_cfg as factory\n        return factory(num_envs, control_hz, play, manifest=recipe_path)\n    if task == "hand2-pen-spin":'),
        ('("hand1-reorient", "hand2-reorient", "hand2-pen-spin")',
         '("hand1-reorient", "hand2-reorient", "hand2-pen-spin", "hand2-grasp-hold")'),
    ])
    cfg = base + "rl/dex_supermimic/config.py"
    update(cfg, [('("hand1-reorient", "hand2-reorient", "hand2-pen-spin")',
                  '("hand1-reorient", "hand2-reorient", "hand2-pen-spin", "hand2-grasp-hold")')])
    site = Path(sysconfig.get_paths()["purelib"])
    (site / "retarget_grasp_source.pth").write_text(str(retarget) + "\n")
    executable = Path(sysconfig.get_paths()["scripts"]) / "dex-supermimic-evaluate-grasp"
    import sys
    executable.write_text(f"#!{sys.executable}\nfrom retarget.integrations.supermimic.evaluate import main\nmain()\n")
    executable.chmod(0o755)
    (backup / "changes.json").write_text(json.dumps(changed, indent=2))
    print(json.dumps(dict(retarget=str(retarget), dex=str(dex), backup=str(backup), changed=changed), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retarget", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dex", type=Path, required=True)
    args = parser.parse_args()
    install(args.retarget, args.dex)
