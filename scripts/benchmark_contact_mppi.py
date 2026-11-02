"""Compare one MPPI window with MJWarp and cached differentiable contacts.

Run from any directory: python scripts/benchmark_contact_mppi.py --output report.json
Scores from different prediction models are reported separately; only the
cross-evaluation table compares plans under the same scoring model.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/humanoid/move_largebox")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--worlds", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gradient-steps", type=int, default=2)
    parser.add_argument("--gradient-step-size", type=float, default=1.0)
    parser.add_argument("--gradient-backtracks", type=int, default=6)
    parser.add_argument("--physical-contact-weight", type=float, default=1.0)
    parser.add_argument(
        "--action-trust-radius", type=float,
        help="Bound every method's control-knot offsets in raw control units.",
    )
    parser.add_argument(
        "--contact-graph", action="store_true",
        help="Capture frozen-contact scores and their backward pass as CUDA graphs.",
    )
    parser.add_argument(
        "--nominal-correction", action="store_true",
        help="Anchor the local prediction to MJWarp along the nominal trajectory.",
    )
    parser.add_argument(
        "--horizon-steps", type=int,
        help="Optional short smoke horizon; must be a positive multiple of knot_steps.",
    )
    parser.add_argument("--output", type=Path, help="Also save the JSON report here.")
    args = parser.parse_args()

    # Keep --help usable before loading the CUDA simulator dependencies.
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("MUJOCO_GL", "egl")
    import numpy as np
    import torch
    import warp as wp
    from dataclasses import asdict

    from retarget.common.results import clean_json, write_json
    from retarget.env.humanoid.runtime import HumanoidEnvironment
    from retarget.solver.base import Problem, Settings, solve

    settings = Settings(
        type="mppi", population=args.population, iterations=args.iterations,
        worlds=args.worlds, improvement_threshold=0.0,
    )
    guided_settings = Settings(
        **{**asdict(settings), "gradient_steps": args.gradient_steps,
           "gradient_step_size": args.gradient_step_size,
           "gradient_backtracks": args.gradient_backtracks},
    )
    if args.gradient_steps < 1:
        parser.error("--gradient-steps must be positive for the guided case")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if not math.isfinite(args.physical_contact_weight) or args.physical_contact_weight < 0:
        parser.error("--physical-contact-weight must be finite and nonnegative")
    if args.action_trust_radius is not None and (
        not math.isfinite(args.action_trust_radius) or args.action_trust_radius <= 0
    ):
        parser.error("--action-trust-radius must be finite and positive")
    if not torch.cuda.is_available():
        parser.error("A CUDA-capable Torch installation is required")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)

    def timed(task, operation):
        torch.cuda.synchronize(args.device)
        started = time.perf_counter()
        with torch.cuda.stream(task.stream):
            value = operation()
        torch.cuda.synchronize(args.device)
        return value, time.perf_counter() - started

    tasks, setup_seconds = {}, {}
    for model in ("mjwarp", "frozen_contact"):
        torch.cuda.synchronize(args.device)
        started = time.perf_counter()
        task = HumanoidEnvironment(
            "move_largebox", args.data.resolve(), device=args.device,
            batch=args.worlds, record_predictions=False, dynamics=model,
            contact_graph=args.contact_graph and model == "frozen_contact",
            contact_options=(
                dict(
                    nominal_correction=args.nominal_correction,
                    physical_contact_weight=args.physical_contact_weight,
                    action_trust_radius=args.action_trust_radius,
                )
                if model == "frozen_contact" else None
            ),
        )
        if args.horizon_steps is not None:
            steps = args.horizon_steps
            if steps < 1 or steps > task.config.horizon_steps or steps % task.config.knot_steps:
                parser.error(
                    "--horizon-steps must be positive, no longer than the source "
                    "horizon, and divisible by knot_steps"
                )
            task.config.horizon_steps = steps
            task.config.horizon = steps * task.config.sim_dt
            task.nominal = task.nominal[:steps].clone()
            task.knots = steps // task.config.knot_steps
        torch.cuda.synchronize(args.device)
        tasks[model] = task
        setup_seconds[model] = time.perf_counter() - started

    state_names = ("qpos", "qvel", "qacc_warmstart", "ctrl", "time")
    initial_state = {
        name: getattr(tasks["mjwarp"].env.data_wp, name).numpy().copy()
        for name in state_names
    }
    for name, state in initial_state.items():
        np.testing.assert_array_equal(
            getattr(tasks["frozen_contact"].env.data_wp, name).numpy(), state,
            err_msg=f"Prediction models have different initial {name}",
        )
    torch.testing.assert_close(tasks["mjwarp"].nominal, tasks["frozen_contact"].nominal)
    _, cache_seconds = timed(
        tasks["frozen_contact"], tasks["frozen_contact"].prepare_contact_rollout,
    )
    c = tasks["mjwarp"].config
    sigma = (
        torch.logspace(
            math.log10(c.first_ctrl_noise_scale), math.log10(c.last_ctrl_noise_scale),
            tasks["mjwarp"].knots, device=args.device,
        )[:, None].expand(-1, c.nu).flatten()
        * c.joint_noise_scale / tasks["mjwarp"].control_scale
    )
    mean = torch.zeros_like(sigma)

    def project(parameters):
        # Use the identical search domain for all three methods, including the
        # original MJWarp scorer which has no contact_options of its own.
        if args.action_trust_radius is None:
            return parameters
        limit = args.action_trust_radius / tasks["mjwarp"].control_scale
        return parameters.clamp(-limit, limit)

    initial_costs, warmup_seconds = {}, {}
    # Warm up both scorers outside the optimization timing. No execution or
    # horizon shifting occurs, so all three solves reuse the same initial state.
    with torch.no_grad():
        for model, task in tasks.items():
            cost, seconds = timed(task, lambda: task.score(mean[None]))
            initial_costs[model] = float(cost[0])
            warmup_seconds[model] = seconds

    cases = {}
    plans = []
    for label, model, options in (
        ("mjwarp_mppi", "mjwarp", settings),
        ("frozen_mppi", "frozen_contact", settings),
        ("frozen_gradient_mppi", "frozen_contact", guided_settings),
    ):
        task = tasks[model]
        # Each frozen case pays its own lazy graph setup, as a fresh MPC window
        # would. Reusing a previous case's graphs would hide that setup cost.
        if task.contact_graph:
            task._contact_graphs.clear()
        graph_builds_before = task.contact_graph_builds
        graph_seconds_before = task.contact_graph_seconds
        result, seconds = timed(
            task, lambda: solve(
                Problem(mean, sigma, task.score, project), options, args.seed,
            ),
        )
        with torch.cuda.stream(task.stream), torch.no_grad():
            plans.append(task.controls(result.parameters[None])[0].clone())
        cases[label] = dict(
            prediction_model=model, settings=asdict(options),
            contact_graph=task.contact_graph,
            graph_builds=task.contact_graph_builds - graph_builds_before,
            graph_capture_seconds=task.contact_graph_seconds - graph_seconds_before,
            initial_cost=initial_costs[model], final_cost=result.cost,
            optimization_seconds=seconds,
            optimization_seconds_excluding_graph_capture=(
                seconds - (task.contact_graph_seconds - graph_seconds_before)
            ),
            cache_prepare_seconds=cache_seconds if model == "frozen_contact" else 0.0,
            window_seconds_including_cache=(
                seconds + cache_seconds if model == "frozen_contact" else seconds
            ),
            sampled=result.sampled, evaluated=result.evaluated,
            gradient_accepted_steps=sum(
                row.get("gradient_accepted_steps", 0) for row in result.history
            ),
            history=result.history,
        )
        print(
            f"{label}: {seconds:.3f}s optimize, "
            f"cost {initial_costs[model]:.6g} -> {result.cost:.6g}",
            flush=True,
        )

    torch.cuda.synchronize(args.device)
    raw_controls = torch.stack(plans)
    cross_evaluation, cross_seconds = {}, {}
    with torch.no_grad():
        for model, task in tasks.items():
            scored, seconds = timed(task, lambda: task.score_controls(raw_controls))
            cross_evaluation[model] = dict(zip(cases, scored.cpu().tolist()))
            cross_seconds[model] = seconds
    for task in tasks.values():
        for name, state in initial_state.items():
            np.testing.assert_array_equal(
                getattr(task.env.data_wp, name).numpy(), state,
                err_msg=f"Benchmark scoring changed execution {name}",
            )

    report = dict(
        task="move_largebox", windows=1, seed=args.seed, device=args.device,
        gpu=torch.cuda.get_device_name(args.device),
        versions=dict(torch=torch.__version__, warp=wp.__version__),
        horizon_steps=c.horizon_steps, horizon_seconds=c.horizon_steps * c.sim_dt,
        identical_initial_execution_state=True,
        environment_setup_seconds=setup_seconds, scorer_warmup_seconds=warmup_seconds,
        contact_cache_prepare_seconds=cache_seconds,
        contact_cache_builds=tasks["frozen_contact"].contact_cache_builds,
        contact_graph_enabled=args.contact_graph,
        nominal_correction=args.nominal_correction,
        physical_contact_weight=args.physical_contact_weight,
        action_trust_radius=args.action_trust_radius,
        domain_randomization={
            model: dict(enabled=bool(task.config.num_dr), ensembles=len(task.env_params))
            for model, task in tasks.items()
        },
        cases=cases, cross_evaluation=cross_evaluation,
        cross_evaluation_seconds=cross_seconds,
        notes=[
            "Initial/final costs across prediction models use different objectives; "
            "compare plans within each cross_evaluation row.",
            "Frozen cases reuse one measured cache. Per-case window timing adds "
            "that cache preparation cost once, excluding environment/JIT setup.",
            "The same seed, population and iteration count are used in all cases; "
            "the gradient case additionally evaluates mean gradients and line searches.",
            "The same control-offset trust projector is used in all three cases.",
            "One-window optimization timing does not establish full retarget success "
            "or task-wide speed improvement.",
        ],
    )
    if args.output is not None:
        write_json(args.output, report)
    print(json.dumps(clean_json(report), indent=2, allow_nan=False))
    return report


if __name__ == "__main__":
    main()
