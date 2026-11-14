# pyright: reportArgumentType=false, reportOperatorIssue=false
"""Pendulum swing-up with MPPI: optimize torques, execute one step, shift, repeat."""

from pathlib import Path
from typing import Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.artist import Artist

from robokit.opt import PopulationSolver, PopulationSolverConfig, Stage, VarValues, VectorVar
from robokit.opt.sampler import GaussianSampler, GaussianSamplerConfig
from robokit.opt.var_values import VarKey
from robokit.terms.task import Task


INITIAL = np.array([np.pi, 0.0], np.float32)
STD, LIMIT = 1.5, 2.0


@wp.func
def _step_func(x: wp.vec2, u: wp.vec2) -> wp.vec2:
    # Angle zero is upright; unit mass/length, gravity 10, semi-implicit Euler.
    angle, velocity = float(x[0]), float(x[1])
    velocity = wp.clamp(velocity + 0.05 * (15.0 * wp.sin(angle) + 3.0 * float(u[0])), -8.0, 8.0)
    return wp.vec2(angle + 0.05 * velocity, velocity)


@wp.func
def _cost_func(x: wp.vec2) -> float:
    return 10.0 * (1.0 - wp.cos(float(x[0]))) + 0.1 * float(x[1]) * float(x[1])


@wp.kernel
def _rollout_kernel(actions: wp.array2d(dtype=float), initial: wp.array(dtype=wp.vec2), out: wp.array(dtype=float)):
    row = wp.tid()
    x = wp.vec2(initial[0])
    cost = float(0.0)
    for t in range(actions.shape[1]):
        u = wp.clamp(actions[row, t], -2.0, 2.0)
        x = _step_func(x, wp.vec2(u, 0.0))
        cost += _cost_func(x) + 0.02 * u * u
    out[row] += (cost + 10.0 * _cost_func(x)) / float(actions.shape[1])


class RolloutTask(Task):
    def __init__(self, state: wp.array):
        self.state = state

    def compute_weighted_cost(self, var_values: Optional[VarValues], *, out_cost: wp.array):
        assert var_values is not None
        actions = cast(VectorVar, var_values.get(VarKey.ACTION)).x
        wp.launch(_rollout_kernel, actions.shape[0], [actions, self.state], [out_cost], device=actions.device)


def main():
    device, seed, steps = "cuda", 0, 250
    state = INITIAL.copy()
    task = RolloutTask(wp.array(state[None], dtype=wp.vec2, device=device))
    actions = wp.zeros((1, 64), device=device)
    values = VarValues.from_var(VectorVar(actions), VarKey.ACTION)
    solver = PopulationSolver(
        [task],  # pyright: ignore[reportArgumentType]  The solver also accepts cost-only Tasks.
        PopulationSolverConfig(
            stages=[Stage(4)],
            cuda_graph_mode="none",
            use_early_stopping=False,
            samplers=[GaussianSamplerConfig(var_key=VarKey.ACTION, num_draws=512, init_std=STD, gamma=0.0, seed=seed)],
        ),
        device,
    )
    solver.setup(values)
    sampler = cast(GaussianSampler, solver.samplers[0])
    sampler.lower = wp.full(actions.shape, -LIMIT, device=device)
    sampler.upper = wp.full(actions.shape, LIMIT, device=device)
    solver.solve(values)
    actions.zero_()
    with wp.ScopedCapture(device=device) as capture:
        solver.solve(values)
    graph = capture.graph
    assert graph is not None
    actions.zero_()
    states, controls = [state.copy()], []
    for step in range(steps):
        task.state.assign(state[None])
        sampler.epoch.assign(np.array([step], np.int32))
        wp.capture_launch(graph)
        plan = np.clip(actions.numpy()[0], -LIMIT, LIMIT)
        u = float(plan[0])
        state = np.array(_step_func(wp.vec2(*state), wp.vec2(u, 0.0)), np.float32)
        actions.assign(np.concatenate((plan[1:], plan[-1:]))[None])
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

    path = Path("runs/05_mpc_pendulum.webp")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    animation = FuncAnimation(fig, update, frames=len(states), interval=50)
    animation.save(path, writer=PillowWriter(fps=20))
    plt.close(fig)
    print(f"saved=\n{path}")


if __name__ == "__main__":
    main()
