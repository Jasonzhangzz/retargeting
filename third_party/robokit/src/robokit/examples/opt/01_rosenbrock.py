# pyright: reportArgumentType=false
# pyright: reportMissingImports=false
"""Write a residual task from scratch and solve Rosenbrock with LM."""

from enum import Enum
from pathlib import Path
from typing import Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import warp as wp

from robokit.opt import VarValues, VectorVar
from robokit.opt.lm_optimizer import LMOptimizer, LMOptimizerConfig
from robokit.terms.task import ResidualTask


class Key(Enum):
    X = 0


plt.switch_backend("Agg")


@wp.kernel
def rosenbrock_residual_kernel(
    x: wp.array2d(dtype=wp.float32),
    row_offset: int,
    out: wp.array2d(dtype=wp.float32),
):
    batch_idx = wp.tid()
    out[batch_idx, row_offset] = 1.0 - x[batch_idx, 0]
    out[batch_idx, row_offset + 1] = 10.0 * (x[batch_idx, 1] - x[batch_idx, 0] * x[batch_idx, 0])


@wp.kernel
def rosenbrock_jacobian_kernel(
    x: wp.array2d(dtype=wp.float32),
    row_offset: int,
    out: wp.array3d(dtype=wp.float32),
):
    batch_idx = wp.tid()
    out[batch_idx, row_offset, 0] = -1.0
    out[batch_idx, row_offset + 1, 0] = -20.0 * x[batch_idx, 0]
    out[batch_idx, row_offset + 1, 1] = 10.0


class RosenbrockTask(ResidualTask):
    residual_weight = None
    var_key = Key.X

    @property
    def residual_dim(self) -> int:
        return 2

    def compute_weighted_residual(
        self,
        var_values: VarValues,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        x = cast(VectorVar, var_values.get(self.var_key)).x
        if out_residual is None:
            out_residual = wp.empty((var_values.batch_size, 2), dtype=wp.float32, device=x.device)
        wp.launch(rosenbrock_residual_kernel, var_values.batch_size, inputs=[x, row_offset], outputs=[out_residual])
        return out_residual

    def compute_weighted_jacobian_analytic(
        self,
        var_values: VarValues,
        out_jacobian: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        x = cast(VectorVar, var_values.get(self.var_key)).x
        if out_jacobian is None:
            out_jacobian = wp.zeros((var_values.batch_size, 2, 2), dtype=wp.float32, device=x.device)
        wp.launch(rosenbrock_jacobian_kernel, var_values.batch_size, inputs=[x, row_offset], outputs=[out_jacobian])
        return out_jacobian


def main():
    start = np.array([[-1.2, 1.0]], dtype=np.float32)
    values = VarValues.from_var(VectorVar(wp.array(start, dtype=wp.float32, device="cuda")), Key.X)
    solver = LMOptimizer(
        [RosenbrockTask()],
        device=wp.get_device("cuda"),
        config=LMOptimizerConfig(max_iter=30, early_stopping_interval=0),
    )
    path = []

    def record_path(_: int, values: VarValues, __: wp.array):
        path.append(cast(VectorVar, values.get(Key.X)).x.numpy()[0].copy())

    solver.solve(values, iteration_callback=record_path)
    path = np.asarray(path)
    final_cost = (1.0 - path[-1, 0]) ** 2 + 100.0 * (path[-1, 1] - path[-1, 0] ** 2) ** 2
    print(f"solution={path[-1].tolist()} cost={final_cost:.3e}")

    grid_x = np.linspace(-2.0, 2.0, 300)
    grid_y = np.linspace(-1.0, 3.0, 300)
    X, Y = np.meshgrid(grid_x, grid_y)
    Z = (1.0 - X) ** 2 + 100.0 * (Y - X**2) ** 2
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), norm="log", cmap="viridis")
    plt.plot(path[:, 0], path[:, 1], "o-", color="tab:red", markersize=3, label="LM")
    plt.scatter(
        [1.0],
        [1.0],
        marker="*",
        s=300,
        color="gold",
        edgecolor="black",
        linewidth=1.5,
        zorder=10,
        label="minimum (1, 1)",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    out_path = Path("runs") / "01_rosenbrock.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
