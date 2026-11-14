# pyright: reportArgumentType=false, reportOperatorIssue=false
"""Navigate a 2D double integrator around a circle with receding-horizon MPPI."""

from pathlib import Path
from typing import Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.artist import Artist
from matplotlib.patches import Circle

from robokit.opt import PopulationSolver, PopulationSolverConfig, Stage, VarValues, VectorVar
from robokit.opt.sampler import GaussianSampler, GaussianSamplerConfig
from robokit.opt.var_values import VarKey
from robokit.terms.task import Task


INITIAL = np.array([-2.0, -0.5, 0.0, 0.0], np.float32)
STD, LIMIT = 0.75, 1.0


@wp.func
def _step_func(x: wp.vec4, u: wp.vec2) -> wp.vec4:
    return wp.vec4(
        float(x[0]) + 0.1 * float(x[2]),
        float(x[1]) + 0.1 * float(x[3]),
        float(x[2]) + 0.1 * float(u[0]),
        float(x[3]) + 0.1 * float(u[1]),
    )


@wp.func
def _cost_func(x: wp.vec4) -> float:
    x0, x1, x2, x3 = float(x[0]), float(x[1]), float(x[2]), float(x[3])
    penetration = wp.max(0.75 - wp.sqrt(x0 * x0 + x1 * x1), 0.0)  # radius 0.6 plus clearance 0.15
    return 2.0 * ((x0 - 2.0) * (x0 - 2.0) + x1 * x1) + 0.2 * (x2 * x2 + x3 * x3) + 1000.0 * penetration * penetration


@wp.kernel
def _rollout_kernel(actions: wp.array2d(dtype=float), initial: wp.array(dtype=wp.vec4), out: wp.array(dtype=float)):
    row = wp.tid()
    x = wp.vec4(initial[0])
    cost = float(0.0)
    horizon = actions.shape[1] // 2
    for t in range(horizon):
        ux, uy = wp.clamp(actions[row, 2 * t], -1.0, 1.0), wp.clamp(actions[row, 2 * t + 1], -1.0, 1.0)
        x = _step_func(x, wp.vec2(ux, uy))
        cost += _cost_func(x) + 0.02 * (ux * ux + uy * uy)
    out[row] += (cost + 10.0 * _cost_func(x)) / float(horizon)


class RolloutTask(Task):
    def __init__(self, state: wp.array):
        self.state = state

    def compute_weighted_cost(self, var_values: Optional[VarValues], *, out_cost: wp.array):
        assert var_values is not None
        actions = cast(VectorVar, var_values.get(VarKey.ACTION)).x
        wp.launch(_rollout_kernel, actions.shape[0], [actions, self.state], [out_cost], device=actions.device)


def main():
    device, seed, steps = "cuda", 0, 100
    state = INITIAL.copy()
    task = RolloutTask(wp.array(state[None], dtype=wp.vec4, device=device))
    actions = wp.zeros((1, 64 * 2), device=device)
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
        plan = np.clip(actions.numpy().reshape(-1, 2), -LIMIT, LIMIT)
        u = plan[0].copy()
        state = np.array(_step_func(wp.vec4(*state), wp.vec2(*u)), np.float32)
        actions.assign(np.concatenate((plan[1:], plan[-1:])).reshape(1, -1))
        states.append(state.copy())
        controls.append(u)
    states, controls = np.array(states), np.array(controls)
    error = np.linalg.norm(states[-1, :2] - [2.0, 0.0])
    clearance = np.linalg.norm(states[:, :2], axis=1).min() - 0.6
    assert error < 0.2 and clearance > 0.0
    assert np.max(np.abs(controls)) <= LIMIT
    print(f"navigation: goal_error={error:.4f} clearance={clearance:.4f} acceleration_max={np.abs(controls).max():.3f}")
    plt.switch_backend("Agg")
    fig, ax = plt.subplots()
    ax.add_patch(Circle((0, 0), 0.6, color="tab:red", alpha=0.5))
    (trajectory,) = ax.plot([], [], label="executed path")
    (robot,) = ax.plot([], [], "o")
    ax.scatter([-2.0, 2.0], [-0.5, 0.0], marker="x", label="start / goal")
    margin = 0.25
    ax.set(
        xlim=(states[:, 0].min() - margin, states[:, 0].max() + margin),
        ylim=(states[:, 1].min() - margin, states[:, 1].max() + margin),
        xlabel="x (m)",
        ylabel="y (m)",
        aspect="equal",
    )
    ax.legend()

    def update(frame: int) -> Tuple[Artist, ...]:
        trajectory.set_data(states[: frame + 1, 0], states[: frame + 1, 1])
        robot.set_data([states[frame, 0]], [states[frame, 1]])
        ax.set_title(f"t = {frame * 0.1:.1f} s")
        return trajectory, robot, ax.title

    path = Path("runs/06_mpc_navigation.webp")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    animation = FuncAnimation(fig, update, frames=len(states), interval=100)
    animation.save(path, writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"saved=\n{path}")


if __name__ == "__main__":
    main()
