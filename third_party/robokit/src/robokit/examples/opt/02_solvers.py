# pyright: reportArgumentType=false
# pyright: reportMissingImports=false
"""Swap five solvers on one Rosenbrock problem."""

from enum import Enum
from typing import Optional, cast

import warp as wp

from robokit.opt import PopulationSolver, PopulationSolverConfig, Stage, VarValues, VectorVar
from robokit.opt.gd_optimizer import AnnealedGDConfig, AnnealedGDOptimizer
from robokit.opt.lbfgs_optimizer import LBFGSOptimizer, LBFGSOptimizerConfig
from robokit.opt.lm_optimizer import LMOptimizer, LMOptimizerConfig
from robokit.opt.sampler import GaussianSamplerConfig
from robokit.terms.task import ResidualTask


class Key(Enum):
    X = 0


@wp.kernel
def residual_kernel(x: wp.array2d(dtype=wp.float32), row: int, out: wp.array2d(dtype=wp.float32)):
    i = wp.tid()
    out[i, row] = 1.0 - x[i, 0]
    out[i, row + 1] = 10.0 * (x[i, 1] - x[i, 0] * x[i, 0])


@wp.kernel
def jacobian_kernel(x: wp.array2d(dtype=wp.float32), row: int, out: wp.array3d(dtype=wp.float32)):
    i = wp.tid()
    out[i, row, 0] = -1.0
    out[i, row + 1, 0] = -20.0 * x[i, 0]
    out[i, row + 1, 1] = 10.0


class RosenbrockTask(ResidualTask):
    residual_weight = None
    var_key = Key.X

    @property
    def residual_dim(self) -> int:
        return 2

    def compute_weighted_residual(
        self, var_values: VarValues, out_residual: Optional[wp.array] = None, row_offset: int = 0
    ) -> wp.array:
        x = cast(VectorVar, var_values.get(self.var_key)).x
        if out_residual is None:
            out_residual = wp.empty((var_values.batch_size, 2), dtype=wp.float32, device=x.device)
        wp.launch(residual_kernel, var_values.batch_size, inputs=[x, row_offset], outputs=[out_residual])
        return out_residual

    def compute_weighted_jacobian_analytic(
        self, var_values: VarValues, out_jacobian: Optional[wp.array] = None, row_offset: int = 0
    ) -> wp.array:
        x = cast(VectorVar, var_values.get(self.var_key)).x
        if out_jacobian is None:
            out_jacobian = wp.zeros((var_values.batch_size, 2, 2), dtype=wp.float32, device=x.device)
        wp.launch(jacobian_kernel, var_values.batch_size, inputs=[x, row_offset], outputs=[out_jacobian])
        return out_jacobian


def solve(name: str, iterations: int) -> float:
    task = RosenbrockTask()
    values = VarValues.from_var(VectorVar(wp.array([[-1.2, 1.0]], dtype=wp.float32, device="cuda")), Key.X)
    device = wp.get_device("cuda")
    if name == "Annealed GD":
        solver = AnnealedGDOptimizer(
            [task],
            AnnealedGDConfig(max_iter=iterations, learning_rate=0.01, starting_temperature=0.5, seed=0),
            device,
        )
    elif name == "L-BFGS":
        solver = LBFGSOptimizer(
            [task],
            LBFGSOptimizerConfig(max_iter=iterations),
            device,
        )
    elif name == "LM":
        solver = LMOptimizer(
            [task],
            config=LMOptimizerConfig(max_iter=iterations, early_stopping_interval=0),
            device=device,
        )
    elif name in (
        "MPPI",
        "ES",
    ):  # sampling alone: each round draws 512 candidates around each and moves it by their scores
        sampler = GaussianSamplerConfig(
            var_key=Key.X,
            num_draws=512,
            weighting="softmax" if name == "MPPI" else "fitness_shaping",
            init_std=0.5,
            min_std=0.01 if name == "MPPI" else 0.5,
            temperature=0.1,
            learning_rate=0.1,
            sample_mode="best",
        )
        solver = PopulationSolver(
            [[task]],
            PopulationSolverConfig(stages=[Stage(iterations)], cuda_graph_mode="none", samplers=[sampler]),
            device,
        )
    else:
        raise ValueError(name)
    _, costs = solver.solve(values)
    return 2.0 * costs.numpy()[0]


def main():
    for name in ("L-BFGS", "LM", "MPPI", "ES"):
        print(f"{name}: final cost={solve(name, 32):.3e}")
    # annealing trades iterations for the ability to escape local minima, so it gets a longer budget
    print(f"Annealed GD: final cost={solve('Annealed GD', 5000):.3e}")


if __name__ == "__main__":
    main()
