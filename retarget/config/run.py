"""One YAML interface; data paths are relative to the configuration file."""

from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
from retarget.solver.base import Settings

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    source: Path
    mode: str
    task: dict
    solver: Settings
    environment: dict
    output: dict
    visualization: dict
    seed: int

    @property
    def destination(self):
        return ROOT / "output" / self.output["name"]

    @property
    def device(self):
        return self.environment.get("device", "cuda:0")

    def resolved(self):
        return dict(
            mode=self.mode,
            task=self.task,
            solver=asdict(self.solver),
            environment=self.environment,
            output=self.output,
            visualization=self.visualization,
            seed=self.seed,
        )


def load(path):
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Task YAML must be a mapping")
    unknown = set(raw) - {
        "mode",
        "task",
        "solver",
        "environment",
        "output",
        "visualization",
        "seed",
    }
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    mode = raw.get("mode", "solve")
    if mode not in ("solve", "replay", "web"):
        raise ValueError("mode must be solve, replay or web")
    task = raw.get("task", {}).copy()
    if mode == "solve" and task.get("type") not in ("grasp", "dex", "humanoid"):
        raise ValueError("task.type must be grasp, dex or humanoid")
    if mode == "solve" and not task.get("data"):
        raise ValueError("task.data is required")
    for key in ("data", "object", "wrist_reference"):
        if task.get(key):
            value = Path(task[key]).expanduser()
            task[key] = str((path.parent / value).resolve())
    options = task.get("options", {}).copy()
    for key in ("physics_reference", "dexonomy_root", "wrist_pose_reference"):
        if options.get(key):
            options[key] = str(
                (path.parent / Path(options[key]).expanduser()).resolve()
            )
    if options:
        task["options"] = options
    output = raw.get("output", {}).copy()
    output.setdefault(
        "name",
        f"{task.get('type', 'view')}_{task.get('name', 'default')}_{raw.get('solver', {}).get('type', 'cem')}",
    )
    name = output["name"]
    if (
        not isinstance(name, str)
        or name in (".", "..", "")
        or Path(name).name != name
        or "/" in name
        or "\\" in name
        or name.startswith(".")
    ):
        raise ValueError("output.name must be one non-hidden directory name")
    if set(output) - {"name"}:
        raise ValueError(
            "Only output.name is supported; output is always inside the repository"
        )
    seed = raw.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    environment = raw.get("environment", {}).copy()
    solver = Settings(**raw.get("solver", {}))
    return Config(
        path,
        mode,
        task,
        solver,
        environment,
        output,
        raw.get("visualization", {}).copy(),
        seed,
    )
