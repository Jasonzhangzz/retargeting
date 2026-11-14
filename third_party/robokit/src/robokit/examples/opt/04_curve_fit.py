# pyright: reportArgumentType=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportMissingImports=false
"""Fit an exponential curve with analytic least-squares residuals and LM."""

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
    PARAMS = 0


plt.switch_backend("Agg")


@wp.kernel
def curve_residual_kernel(
    params: wp.array2d(dtype=wp.float32),
    t: wp.array1d(dtype=wp.float32),
    y: wp.array1d(dtype=wp.float32),
    row_offset: int,
    out: wp.array2d(dtype=wp.float32),
):
    batch_idx, point_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    amplitude = params[batch_idx, 0]
    rate = params[batch_idx, 1]
    offset = params[batch_idx, 2]
    out[batch_idx, row_offset + point_idx] = amplitude * wp.exp(rate * t[point_idx]) + offset - y[point_idx]


@wp.kernel
def curve_jacobian_kernel(
    params: wp.array2d(dtype=wp.float32),
    t: wp.array1d(dtype=wp.float32),
    row_offset: int,
    out: wp.array3d(dtype=wp.float32),
):
    batch_idx, point_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    amplitude = params[batch_idx, 0]
    exponential = wp.exp(params[batch_idx, 1] * t[point_idx])
    out[batch_idx, row_offset + point_idx, 0] = exponential
    out[batch_idx, row_offset + point_idx, 1] = amplitude * t[point_idx] * exponential
    out[batch_idx, row_offset + point_idx, 2] = 1.0


class CurveFitTask(ResidualTask):
    residual_weight = None
    var_key = Key.PARAMS

    def __init__(self, t: np.ndarray, y: np.ndarray, device: str):
        self.t = wp.array(t, dtype=wp.float32, device=device)
        self.y = wp.array(y, dtype=wp.float32, device=device)

    @property
    def residual_dim(self) -> int:
        return int(self.t.shape[0])

    def compute_weighted_residual(
        self, var_values: VarValues, out_residual: Optional[wp.array] = None, row_offset: int = 0
    ) -> wp.array:
        params = cast(VectorVar, var_values.get(self.var_key)).x
        if out_residual is None:
            out_residual = wp.empty((var_values.batch_size, self.residual_dim), dtype=wp.float32, device=params.device)
        wp.launch(
            curve_residual_kernel,
            (var_values.batch_size, self.residual_dim),
            inputs=[params, self.t, self.y, row_offset],
            outputs=[out_residual],
        )
        return out_residual

    def compute_weighted_jacobian_analytic(
        self, var_values: VarValues, out_jacobian: Optional[wp.array] = None, row_offset: int = 0
    ) -> wp.array:
        params = cast(VectorVar, var_values.get(self.var_key)).x
        if out_jacobian is None:
            out_jacobian = wp.zeros(
                (var_values.batch_size, self.residual_dim, 3), dtype=wp.float32, device=params.device
            )
        wp.launch(
            curve_jacobian_kernel,
            (var_values.batch_size, self.residual_dim),
            inputs=[params, self.t, row_offset],
            outputs=[out_jacobian],
        )
        return out_jacobian


def main():
    rng = np.random.default_rng(5)
    t = np.linspace(0.0, 2.0, 80, dtype=np.float32)
    true_params = np.array([2.5, -1.3, 0.4], dtype=np.float32)
    y = true_params[0] * np.exp(true_params[1] * t) + true_params[2]
    y += rng.normal(0.0, 0.04, t.shape).astype(np.float32)

    initial = np.array([[1.0, -0.2, 0.0]], dtype=np.float32)
    values = VarValues.from_var(VectorVar(wp.array(initial, dtype=wp.float32, device="cuda")), Key.PARAMS)
    solver = LMOptimizer(
        [CurveFitTask(t, y, "cuda")],
        config=LMOptimizerConfig(max_iter=30, early_stopping_interval=0),
        device=wp.get_device("cuda"),
    )
    result, costs = solver.solve(values)
    fitted = cast(VectorVar, result.get(Key.PARAMS)).x.numpy()[0]
    print(f"true={true_params.tolist()} fitted={fitted.tolist()} cost={costs.numpy()[0]:.3e}")

    plt.scatter(t, y, s=14, alpha=0.6, label="observations")
    plt.plot(t, fitted[0] * np.exp(fitted[1] * t) + fitted[2], color="tab:red", linewidth=2, label="LM fit")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    out_path = Path("runs") / "04_curve_fit.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
