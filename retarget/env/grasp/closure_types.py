"""Configuration and witnesses for the historical four-ray batched force QP."""

from dataclasses import dataclass, field
import math
import torch


@dataclass(frozen=True)
class ClosureSettings:
    friction: float = 0.6
    test_force_n: float | None = None
    certificate_tolerance: float = 0.05
    cuda_graph: bool = True

    def __post_init__(self):
        if not math.isfinite(self.friction) or self.friction < 0:
            raise ValueError("closure.friction must be finite and nonnegative")
        if self.test_force_n is not None and (
            not math.isfinite(self.test_force_n) or self.test_force_n <= 0
        ):
            raise ValueError("closure.test_force_n must be positive or None")
        if (
            not math.isfinite(self.certificate_tolerance)
            or self.certificate_tolerance <= 0
        ):
            raise ValueError("closure.certificate_tolerance must be positive")
        if not isinstance(self.cuda_graph, bool):
            raise ValueError("closure.cuda_graph must be boolean")


@dataclass
class ClosureResult:
    cost: torch.Tensor
    preloads: torch.Tensor
    row_forces: torch.Tensor
    scenario_residuals: torch.Tensor
    scenario_rms: torch.Tensor
    converged: torch.Tensor
    valid: torch.Tensor
    certified: torch.Tensor
    certified_margin: torch.Tensor
    objective_history: torch.Tensor | None = None
    metadata: dict = field(default_factory=dict)
