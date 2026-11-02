"""Task-independent, device-resident sampling contract."""

from dataclasses import dataclass
from typing import Callable
import math
import torch


@dataclass(frozen=True)
class Settings:
    type: str = "cem"
    population: int = 1024
    iterations: int = 16
    worlds: int = 128
    elite_fraction: float = 0.05
    temperature: float | None = None
    final_noise: float | None = None
    improvement_threshold: float = 0.01
    improvement_check_steps: int = 1
    exploit_ratio: float = 0.01
    exploit_noise: float = 0.01
    gradient_steps: int = 0
    gradient_step_size: float = 1.0
    gradient_backtracks: int = 6
    scsp: object = None
    dexforge: object = None
    graspadmm: object = None

    def __post_init__(self):
        if self.type not in ("cem", "mppi", "scsp", "dexforge", "graspadmm"):
            raise ValueError("solver.type must be cem, mppi, scsp, dexforge or graspadmm")
        if self.type == "graspadmm":
            from .graspadmm import settings
            object.__setattr__(self, "graspadmm", settings(self.graspadmm))
        elif self.graspadmm is not None:
            raise ValueError("solver.graspadmm requires solver.type: graspadmm")
        if self.type == "dexforge":
            from .dexforge import DexForgeSettings

            values = self.dexforge
            if values is None or isinstance(values, dict):
                try:
                    values = DexForgeSettings(**(values or {}))
                except TypeError as exc:
                    raise ValueError(f"Invalid solver.dexforge: {exc}") from exc
            if not isinstance(values, DexForgeSettings):
                raise ValueError("solver.dexforge must be a mapping")
            object.__setattr__(self, "dexforge", values)
        elif self.dexforge is not None:
            raise ValueError("solver.dexforge requires solver.type: dexforge")
        if self.type == "scsp":
            from .scsp import SCSPSettings

            values = self.scsp
            if values is None or isinstance(values, dict):
                try:
                    values = SCSPSettings(**(values or {}))
                except TypeError as exc:
                    raise ValueError(f"Invalid solver.scsp: {exc}") from exc
            if not isinstance(values, SCSPSettings):
                raise ValueError("solver.scsp must be a mapping")
            object.__setattr__(self, "scsp", values)
        elif self.scsp is not None:
            raise ValueError("solver.scsp requires solver.type: scsp")
        if (
            self.population < (20 if self.type == "mppi" else 2)
            or min(self.iterations, self.worlds, self.improvement_check_steps) < 1
        ):
            raise ValueError("Invalid sampling budget")
        for key, value in [
            ("elite_fraction", self.elite_fraction),
            ("final_noise", self.noise_ratio),
        ]:
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{key} must lie in (0,1]")
        if not math.isfinite(self.temp) or self.temp <= 0:
            raise ValueError("temperature must be positive")
        if (
            not 0 <= self.exploit_ratio < 1
            or not math.isfinite(self.exploit_noise)
            or self.exploit_noise < 0
        ):
            raise ValueError("Invalid MPPI exploitation settings")
        if (
            not math.isfinite(self.improvement_threshold)
            or self.improvement_threshold < 0
        ):
            raise ValueError("Invalid stopping threshold")
        if (
            isinstance(self.gradient_steps, bool)
            or not isinstance(self.gradient_steps, int)
            or self.gradient_steps < 0
            or isinstance(self.gradient_backtracks, bool)
            or not isinstance(self.gradient_backtracks, int)
            or self.gradient_backtracks < 1
            or not math.isfinite(self.gradient_step_size)
            or self.gradient_step_size <= 0
        ):
            raise ValueError("Invalid gradient refinement settings")
        if self.gradient_steps and self.type != "mppi":
            raise ValueError("gradient_steps requires solver.type: mppi")

    @property
    def temp(self):
        return (
            self.temperature
            if self.temperature is not None
            else (0.1 if self.type == "cem" else 1.0)
        )

    @property
    def noise_ratio(self):
        return (
            self.final_noise
            if self.final_noise is not None
            else (0.2 if self.type == "cem" else 0.1)
        )


@dataclass
class Evaluation:
    cost: torch.Tensor
    ranking: torch.Tensor | None = None

    @property
    def valid(self):
        return torch.isfinite(self.cost)

    @property
    def search(self):
        return self.cost if self.ranking is None else self.ranking


@dataclass
class Problem:
    mean: torch.Tensor
    sigma: torch.Tensor
    evaluate: Callable
    project: Callable = lambda x: x


@dataclass
class Result:
    parameters: torch.Tensor
    cost: float
    history: list
    sampled: int
    evaluated: int


class NoValidCandidate(FloatingPointError):
    def __init__(self, history):
        super().__init__(
            "No candidate satisfies the task hard constraints with a finite cost"
        )
        self.history = history


def evaluate(problem, values):
    result = problem.evaluate(values)
    return result if isinstance(result, Evaluation) else Evaluation(result)


def solve(problem, settings, seed=0):
    if settings.type == "graspadmm":
        raise ValueError("GraspADMM requires its dedicated native grasp runner")
    if settings.type == "dexforge":
        from .dexforge import solve as implementation
    elif settings.type == "scsp":
        from .scsp import solve as implementation
    elif settings.type == "cem":
        from .cem import solve as implementation
    else:
        from .mppi import solve as implementation
    with torch.no_grad():
        return implementation(problem, settings, seed)
