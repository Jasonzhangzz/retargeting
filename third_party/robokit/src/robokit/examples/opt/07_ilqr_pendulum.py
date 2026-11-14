# pyright: reportArgumentType=false, reportOperatorIssue=false
"""Pendulum swing-up with iLQR: optimize torques, execute one step, shift, repeat (as `05_mpc_pendulum.py` with MPPI)."""

from pathlib import Path
from typing import Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.artist import Artist

from robokit.opt import VarValues, VectorVar
from robokit.opt.ilqr_optimizer import Dynamics, ILQROptimizer, ILQROptimizerConfig
from robokit.opt.var_values import Var, VarKey
from robokit.terms.task import ResidualTask


INITIAL = np.array([np.pi, 0.0], np.float32)
LIMIT = 2.0


@wp.func
def _step_func(x: wp.vec2, u: float) -> wp.vec2:
    # Angle zero is upright; unit mass/length, gravity 10, semi-implicit Euler.
    velocity = wp.clamp(x[1] + 0.05 * (15.0 * wp.sin(x[0]) + 3.0 * u), -8.0, 8.0)
    return wp.vec2(x[0] + 0.05 * velocity, velocity)


@wp.kernel
def _step_kernel(x: wp.array2d(dtype=float), u: wp.array2d(dtype=float), out: wp.array2d(dtype=float)):
    i = wp.tid()
    y = _step_func(wp.vec2(x[i, 0], x[i, 1]), u[i, 0])
    out[i, 0] = y[0]  # pyright: ignore[reportIndexIssue]
    out[i, 1] = y[1]  # pyright: ignore[reportIndexIssue]


@wp.kernel
def _residual_kernel(x: wp.array2d(dtype=float), u: wp.array2d(dtype=float), o: int, r: wp.array2d(dtype=float)):
    i = wp.tid()
    r[i, o] = wp.sqrt(40.0) * wp.sin(0.5 * x[i, 0])
    r[i, o + 1] = wp.sqrt(0.2) * x[i, 1]
    r[i, o + 2] = 0.2 * u[i, 0]


class Pendulum(Dynamics):
    """States: `VectorVar`s of (angle, velocity); one bounded torque."""

    def __init__(self):
        super().__init__(nu=1, ctrl_range=np.array([[-LIMIT, LIMIT]]))

    def step(self, x: Var, u: wp.array, x_next: Var, finite_differences: bool = False):
        ins = [cast(VectorVar, x).x, u]
        wp.launch(_step_kernel, x.batch_size, inputs=ins, outputs=[cast(VectorVar, x_next).x], device=u.device)


class SwingUpCost(ResidualTask):
    """Per step, `0.5 |r|^2 = 10 (1 - cos angle) + 0.1 velocity^2 + 0.02 torque^2`."""

    @property
    def residual_dim(self) -> int:
        return 3

    def compute_weighted_residual(self, var_values: VarValues, *, out_residual: wp.array, row_offset: int = 0):
        x, u = (cast(VectorVar, var_values.get(key)).x for key in (VarKey.STATE, VarKey.ACTION))
        wp.launch(_residual_kernel, x.shape[0], inputs=[x, u, row_offset], outputs=[out_residual], device=x.device)
        return out_residual


def main():
    device, steps = "cuda" if wp.is_cuda_available() else "cpu", 250
    optimizer = ILQROptimizer(Pendulum(), [SwingUpCost()], ILQROptimizerConfig(max_iter=2), device=device)
    # 64 torques (3.2 s), starting with a push: hanging still, the cost has no gradient for iLQR to follow
    plan = VectorVar(wp.full((1, 64), 0.5, device=device))
    state, states, controls = INITIAL.copy(), [INITIAL.copy()], []
    for step in range(steps):
        optimizer.set_start(VectorVar(wp.array(state[None], dtype=float, device=device)))
        optimizer.solve(plan, shift=None if step == 0 else 1.0)  # then warm-started one step on
        u = float(plan.x.numpy()[0, 0])
        state = np.array(_step_func(wp.vec2(*state), u), np.float32)
        states.append(state.copy())
        controls.append(u)
    states, controls = np.array(states), np.array(controls)
    angle = np.arctan2(np.sin(states[:, 0]), np.cos(states[:, 0]))
    assert np.max(np.abs(angle[-20:])) < 0.2 and np.max(np.abs(states[-20:, 1])) < 0.5
    assert np.max(np.abs(controls)) <= LIMIT
    print(f"pendulum: angle={angle[-1]:.4f} velocity={states[-1, 1]:.4f} torque_max={np.abs(controls).max():.3f}")
    plt.switch_backend("Agg")
    fig, axes = plt.subplot_mosaic(
        [["pendulum", "angle"], ["pendulum", "torque"]], figsize=(9, 4), gridspec_kw={"width_ratios": [1, 1.5]}
    )
    axes["pendulum"].set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), aspect="equal", xlabel="x (m)", ylabel="y (m)")
    axes["pendulum"].plot([0, 0], [0, 1], "--", color="0.7", label="upright")
    (rod,) = axes["pendulum"].plot([], [], "o-", linewidth=3, markersize=10)
    clock = axes["pendulum"].set_title("")
    times = np.arange(steps + 1) * 0.05
    axes["angle"].plot(times, angle)
    axes["angle"].set_ylabel("angle from upright (rad)")
    axes["torque"].sharex(axes["angle"])
    axes["torque"].step(times, np.r_[controls, controls[-1]], where="post")
    axes["torque"].set(xlabel="time (s)", ylabel="torque (Nm)")
    cursors = [axes[name].axvline(0, color="tab:red", alpha=0.7) for name in ("angle", "torque")]

    def update(frame: int) -> Tuple[Artist, ...]:
        rod.set_data([0, np.sin(states[frame, 0])], [0, np.cos(states[frame, 0])])
        clock.set_text(f"t = {times[frame]:.2f} s")
        for cursor in cursors:
            cursor.set_xdata([times[frame], times[frame]])
        return rod, clock, *cursors

    path = Path("runs/07_ilqr_pendulum.webp")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    animation = FuncAnimation(fig, update, frames=len(states), interval=50)
    animation.save(path, writer=PillowWriter(fps=20))
    plt.close(fig)
    print(f"saved=\n{path}")


if __name__ == "__main__":
    main()
