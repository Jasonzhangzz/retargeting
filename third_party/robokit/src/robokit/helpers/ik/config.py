"""Robot-free IKSolverConfig: terms, scores, solver, knobs; build via `IKSolver(config, robot, target_links)`."""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig, Stage
from robokit.opt.sampler import RobertsSamplerConfig
from robokit.terms.task import Context, ResidualTask


class IKContext(Context):
    """Values an `IKSolver` gives its task templates; arrays are buffers refilled for each solve."""

    TARGET_LINKS = 0
    """Indices of the target links."""
    TARGET_LINK = 1
    """The first target link."""
    TARGET = 2
    """Target poses `(batch, links)` vec7, which candidates share."""
    PREVIOUS_STATE = 3
    """The previous solution `(batch,)`, from `solve(prev_state=...)`."""
    REST_Q = 4
    """Rest configurations `(batch, dofs)`: the robot's zero configuration until `solve(rest_state=...)`."""
    REST_BASE = 5
    """Rest base poses `(batch,)`: identity until `solve(rest_state=...)`; None for a fixed base."""
    NUM_SOLUTIONS = 6
    """Solutions each target returns."""
    OPTIMIZATION_TERMS = 7
    """The built optimization terms, which score goals no candidate makes feasible."""
    SCENE = 8
    """The collision scene, from `IKSolver(scene=...)`; None without one."""
    SCENE_INDICES = 9
    """Each target's scene `(batch,)`: zeros until `solve(scene_indices=...)`."""


def _default_ik_solver_config() -> PopulationSolverConfig:
    return PopulationSolverConfig(
        stages=[
            Stage(6, 64, LMOptimizerConfig(lm_lambda=10.0)),
            Stage(10, 4, LMOptimizerConfig(lm_lambda=1.0)),
            Stage(0, 1),
        ],
        cuda_graph_mode="full",
    )


@dataclass
class IKSolverConfig:
    """Robot-free IK configuration; build it with `IKSolver(config, robot, target_links)`."""

    terms: List[ResidualTask] = field(default_factory=list)
    """Optimization task templates; `IKContext` lists their inputs."""
    score_terms: List[ResidualTask] = field(default_factory=list)
    """Task templates that rank candidates at each select without changing optimization; empty ranks by cost."""
    solver: PopulationSolverConfig = field(default_factory=_default_ik_solver_config)
    """Candidate schedule and LM settings; the last segment's width is the solutions returned per target."""
    sampler: RobertsSamplerConfig = field(default_factory=lambda: RobertsSamplerConfig(keep_first=True))
    """Joint noise of the initial candidates; `scale` is a fraction of each joint's range around `init_state`."""
    base_init_sample_range: Tuple[float, float, float] = (0.1, 0.1, 0.0)
    """Per-axis base translation noise (m) around `init_state`, for a floating base."""
    enable_T_world_base: bool = False
    """Optimize the six floating-base DOFs along with the joints."""
    base_active_mask: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    """Active (1.0) or locked (0.0) base DOFs `[x, y, z, roll, pitch, yaw]`; locking needs `enable_T_world_base`."""

    def __post_init__(self):
        if len(self.base_active_mask) != 6:
            raise ValueError(f"base_active_mask length {len(self.base_active_mask)} != 6")
        if any(m != 1.0 for m in self.base_active_mask) and not self.enable_T_world_base:
            raise ValueError("base_active_mask requires enable_T_world_base=True")

    def add(self, term: ResidualTask, *, name: Optional[str] = None) -> "IKSolverConfig":
        self._assign_name(term, name)
        self.terms.append(term)
        return self

    def add_score(self, term: ResidualTask, *, name: Optional[str] = None) -> "IKSolverConfig":
        self._assign_name(term, name)
        self.score_terms.append(term)
        return self

    def task(self, name: str) -> ResidualTask:
        """Return the term added under `name`, so a caller can complete a preset stub."""
        for term in [*self.terms, *self.score_terms]:
            if term.name == name:
                return term
        raise KeyError(f"No IK term named {name!r}.")

    def _assign_name(self, term: ResidualTask, name: Optional[str]):
        if name is not None:
            term.name = name
            return
        base = term.cost_name
        existing = {t.name for t in self.terms} | {t.name for t in self.score_terms}
        idx = 0
        candidate = f"{base}_{idx}"
        while candidate in existing:
            idx += 1
            candidate = f"{base}_{idx}"
        term.name = candidate


__all__ = ["IKContext", "IKSolverConfig"]
