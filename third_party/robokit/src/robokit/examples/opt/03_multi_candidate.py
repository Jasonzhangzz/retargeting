# pyright: reportArgumentType=false
# pyright: reportMissingImports=false
"""Compare one candidate with multi-candidate optimization on Rastrigin."""

from enum import Enum
from pathlib import Path
from typing import Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import warp as wp

from robokit.opt import VarValues, VectorVar
from robokit.opt.lbfgs_optimizer import LBFGSOptimizerConfig
from robokit.opt.population_solver import PopulationSolver, PopulationSolverConfig, Stage
from robokit.terms.task import GradientTask, ResidualTask


class Key(Enum):
    X = 0


plt.switch_backend("Agg")


@wp.kernel
def rastrigin_kernel(
    x: wp.array2d(dtype=wp.float32),
    cost: wp.array1d(dtype=wp.float32),
    gradient: wp.array2d(dtype=wp.float32),
    need_gradient: wp.bool,
):
    batch_idx = wp.tid()
    value = wp.float32(20.0)
    for dim_idx in range(2):
        q = x[batch_idx, dim_idx]
        value += q * q - 10.0 * wp.cos(2.0 * wp.pi * q)
        if need_gradient:
            gradient[batch_idx, dim_idx] += 2.0 * q + 20.0 * wp.pi * wp.sin(2.0 * wp.pi * q)
    cost[batch_idx] += value


@wp.kernel
def rastrigin_residual_kernel(x: wp.array2d(dtype=wp.float32), row_offset: int, residual: wp.array2d(dtype=wp.float32)):
    batch_idx = wp.tid()
    value = wp.float32(20.0)
    for dim_idx in range(2):
        q = x[batch_idx, dim_idx]
        value += q * q - 10.0 * wp.cos(2.0 * wp.pi * q)
    residual[batch_idx, row_offset] = wp.sqrt(2.0 * max(value, 0.0))


class RastriginTask(ResidualTask, GradientTask):
    var_key = Key.X
    residual_weight = None

    @property
    def residual_dim(self) -> int:
        return 1

    def compute_weighted_residual(
        self, var_values: VarValues, out_residual: Optional[wp.array] = None, row_offset: int = 0
    ) -> wp.array:
        x = cast(VectorVar, var_values.get(self.var_key)).x
        if out_residual is None:
            out_residual = wp.empty((var_values.batch_size, 1), dtype=wp.float32, device=x.device)
        wp.launch(
            rastrigin_residual_kernel,
            var_values.batch_size,
            inputs=[x, row_offset],
            outputs=[out_residual],
            device=x.device,
        )
        return out_residual

    def compute_weighted_cost_and_gradient(
        self,
        var_values: VarValues,
        *args,
        out_cost: wp.array,
        out_gradient: Optional[wp.array] = None,
        **kwargs,
    ):
        del args, kwargs
        x = cast(VectorVar, var_values.get(self.var_key)).x
        wp.launch(
            rastrigin_kernel,
            var_values.batch_size,
            inputs=[x, out_cost, out_gradient, out_gradient is not None],
            device=x.device,
        )


def main():
    num_candidates = 128
    candidates = np.random.default_rng(4).uniform(-5.12, 5.12, (num_candidates, 2)).astype(np.float32)
    task = RastriginTask()
    solutions = {}
    for label, initial in (("single candidate", candidates[:1]), ("128 candidates", candidates)):
        values = VarValues.from_var(VectorVar(wp.array(initial, dtype=wp.float32, device="cuda")), Key.X)
        solver = PopulationSolver(
            terms=[[task]],
            config=PopulationSolverConfig(
                stages=[Stage(300, len(initial), LBFGSOptimizerConfig(gradient_mode="analytic_gradient"))],
                cuda_graph_mode="none",
                use_early_stopping=False,
            ),
            device=wp.get_device("cuda"),
        )
        best, costs = solver.solve(values)
        best_x = cast(VectorVar, best.get(Key.X)).x.numpy()[0]
        solutions[label] = best_x
        print(f"{label}: cost={costs.numpy()[0]:.3e} x={best_x.tolist()}")

    axis = np.linspace(-5.12, 5.12, 300)
    X, Y = np.meshgrid(axis, axis)
    Z = 20.0 + X**2 - 10.0 * np.cos(2.0 * np.pi * X) + Y**2 - 10.0 * np.cos(2.0 * np.pi * Y)
    contours = plt.contourf(X, Y, Z, levels=40, cmap="viridis")
    plt.colorbar(contours, label="Rastrigin cost")
    plt.scatter(candidates[:, 0], candidates[:, 1], s=8, color="white", alpha=0.5, label="initial candidates")
    plt.scatter(
        *solutions["single candidate"],
        marker="x",
        s=100,
        color="red",
        linewidths=2,
        label="single-candidate result / local minimum",
    )
    plt.scatter(
        *solutions["128 candidates"],
        marker="*",
        s=180,
        color="yellow",
        edgecolors="black",
        linewidths=0.8,
        label="multi-candidate result / global minimum",
    )
    plt.title("Multi-candidate optimization avoids local minima")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    out_path = Path("runs") / "03_multi_candidate.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
