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

    def __post_init__(self):
        if self.type not in ("cem", "mppi"):
            raise ValueError("solver.type must be cem or mppi")
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
    if settings.type == "cem":
        from .cem import solve as implementation
    else:
        from .mppi import solve as implementation
    with torch.no_grad():
        return implementation(problem, settings, seed)
