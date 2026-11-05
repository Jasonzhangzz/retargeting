"""Task-independent, device-resident sampling contract."""

from dataclasses import dataclass, field
from typing import Callable
import math
import torch


@dataclass(frozen=True)
class Settings:
    type: str = 'cem'

    def __post_init__(self):
        if self.type not in ('cem','graspadmm'):raise ValueError('solver.type must be cem or graspadmm')

    @property
    def parameters(self):
        if self.type=='graspadmm':
            from .graspadmm import DEFAULTS
        else:
            from .cem import DEFAULTS
        return DEFAULTS

    @property
    def worlds(self):
        from .cem import DEFAULTS
        return DEFAULTS.sampling.worlds

    @property
    def graspadmm(self):
        from .graspadmm import settings
        return settings() if self.type=='graspadmm' else None


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
    observer: Callable | None = None


@dataclass
class Result:
    parameters: torch.Tensor
    cost: float
    history: list
    sampled: int
    evaluated: int
    metadata: dict = field(default_factory=dict)


class NoValidCandidate(FloatingPointError):
    def __init__(self, history):
        super().__init__(
            "No candidate satisfies the task hard constraints with a finite cost"
        )
        self.history = history


def evaluate(problem, values):
    result = problem.evaluate(values)
    return result if isinstance(result, Evaluation) else Evaluation(result)


def solve(problem, settings=Settings(), seed=0):
    if getattr(settings,'type','cem')=='graspadmm':
        from .graspadmm import solve as implementation
        return implementation(problem)
    from .cem import solve as implementation, DEFAULTS
    if not isinstance(settings,Settings):raise TypeError('Pass a method selector; algorithm parameters use the canonical defaults')
    with torch.no_grad():
        return implementation(problem,seed=seed)
