from typing import TYPE_CHECKING

from robokit.opt.var_values import Var, VarKey, VarValues, VectorVar


if TYPE_CHECKING:
    from robokit.opt.population_solver import PopulationSolver, PopulationSolverConfig, Stage


def __getattr__(name: str):
    if name == "PopulationSolver":
        from robokit.opt.population_solver import PopulationSolver

        return PopulationSolver
    if name in ("PopulationSolverConfig", "Stage"):
        from robokit.opt import population_solver

        return getattr(population_solver, name)
    raise AttributeError(f"module 'robokit.opt' has no attribute {name!r}")


__all__ = [
    "PopulationSolver",
    "PopulationSolverConfig",
    "Stage",
    "Var",
    "VarKey",
    "VarValues",
    "VectorVar",
]
