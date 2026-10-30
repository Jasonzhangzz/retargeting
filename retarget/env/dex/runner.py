"""Prepared DexYCB tracking using fixed initial state and action knots."""

from dataclasses import replace
from pathlib import Path
import json, time
import numpy as np
import torch
import warp as wp
from retarget.config.dex import load_retarget_config
from retarget.env.mjwarp import load_model
from retarget.env.surface import SurfaceQueries
from .physics import (
    _configure_comfree_physics,
    _collision_output_layout,
)
from .backend import ForwardRollout
from .control import resample_warm_corrections
from retarget.task.dex.reference import (
    build_index,
    load_sequence,
    initial_state,
    make_window,
)
from retarget.solver.base import Problem, solve
from retarget.common.results import write_json, clean_json


def validate(config):
    data = Path(config.task["data"])
    frames = config.environment.get("frames")
    if frames is not None and (not isinstance(frames, int) or frames < 1):
        raise ValueError("environment.frames must be a positive integer")
    for name in (
        "model.yaml",
        "contactaware/wuji_right_qpos.npy",
        "contactaware/object_pose_7.npy",
        "contactaware/contact_guidance.npz",
    ):
        if not (data / name).is_file():
            raise ValueError(
                f"Missing prepared dex input: {data / name}. Raw MANO conversion is not implemented; select data/dexycb/prepared/sugar_box."
            )
    return load_retarget_config(
        data / "sequence", data / "model.yaml", contactaware_dir=data / "contactaware"
    )


def run(config, output):
    started = time.perf_counter()
    cfg = validate(config)
    wp.init()
    wp.set_device(config.device)
    device, model = load_model(str(cfg.sequence.xml), device=config.device)
    _configure_comfree_physics(
        cfg, model, device, timestep=cfg.simulator.time_grid.mpc_dt
    )
    index = build_index(model)
    bodies, enabled = _collision_output_layout(model)
    surfaces = SurfaceQueries(model, index.object_body, device=config.device)
    sequence = load_sequence(
        cfg,
        output_body_ids=bodies,
        enabled_output_indices=enabled,
        object_surface_spheres=surfaces.object_spheres,
    )
    q, v = initial_state(
        model,
        index,
        sequence,
        start_frame=0.0,
        ref_dt=cfg.simulator.time_grid.ref_dt,
        hold=False,
    )
    backend = ForwardRollout(
        model,
        device,
        index,
        cfg,
        sequence.contact.mask.shape[1],
        config.solver.worlds,
        surfaces=surfaces,
    )
    grid = cfg.simulator.time_grid
    full_steps = int(np.ceil((len(sequence.hand) - 1) * grid.ref_dt / grid.mpc_dt))
    frames = config.environment.get("frames")
    max_steps = (
        full_steps
        if frames is None
        else min(
            full_steps,
            max(1, int(frames)) * max(1, int(round(grid.ref_dt / grid.mpc_dt))),
        )
    )
    previous = q[index.hand_qpos].copy()
    mean = torch.zeros(index.action_dim * cfg.prediction.horizon, device=config.device)
    sigma = torch.full_like(mean, float(config.task.get("action_noise", 0.003)))
    qs = [q.copy()]
    vs = [v.copy()]
    controls = []
    records = []
    executed = 0
    torch_stream = torch.cuda.current_stream(config.device)
    with wp.ScopedStream(wp.stream_from_torch(torch_stream)):
        while executed < max_steps:
            targets = make_window(
                sequence,
                executed_steps=executed,
                horizon=cfg.prediction.horizon,
                time_grid=grid,
                start_frame=0.0,
                end_frame=cfg.sequence.end,
                hold_frame=cfg.sequence.hold_frame,
            )
            backend.prepare(q, v, targets, previous, time=executed * grid.mpc_dt)
            torch.cuda.synchronize(config.device)
            tick = time.perf_counter()
            result = solve(
                Problem(mean, sigma, backend.score),
                config.solver,
                config.seed + len(records),
            )
            torch.cuda.synchronize(config.device)
            seconds = time.perf_counter() - tick
            backend.score(result.parameters[None])
            torch.cuda.synchronize(config.device)
            n = min(grid.action_substeps, max_steps - executed)
            for step in backend.steps[:n]:
                qs.append(step.recorded.result.qpos.numpy()[0].copy())
                vs.append(step.recorded.result.qvel.numpy()[0].copy())
                controls.append(step.inputs.dynamics.ctrl.numpy()[0].copy())
            q, v = qs[-1], vs[-1]
            previous = controls[-1][index.hand_ctrl].copy()
            backend.objective.commit_window(previous)
            raw = np.r_[np.zeros(index.action_dim), result.parameters.cpu().numpy()]
            shifted = resample_warm_corrections(
                raw,
                action_dim=index.action_dim,
                horizon=cfg.prediction.horizon,
                executed_dense_steps=n,
                physics_steps_per_knot=grid.knot_substeps,
                warm_scale=cfg.prediction.warm_shift_scale,
            )
            mean = torch.as_tensor(
                shifted[index.action_dim :], device=config.device, dtype=torch.float32
            )
            executed += n
            record = dict(
                window=len(records),
                cost=result.cost,
                solve_seconds=seconds,
                iterations=len(result.history),
                sampled=result.sampled,
                evaluated=result.evaluated,
                history=result.history,
            )
            records.append(record)
            with (output / "iterations.jsonl").open("a") as log:
                log.write(json.dumps(clean_json(record)) + "\n")
            print(
                f"[{config.solver.type} dex step={executed}/{max_steps}] cost={result.cost:.6g}",
                flush=True,
            )
    np.savez_compressed(
        output / "trajectory.npz",
        qpos=qs,
        qvel=vs,
        ctrl=controls,
        time=np.arange(len(qs)) * grid.mpc_dt,
    )
    write_json(
        output / "summary.json",
        dict(
            full_clip_completed=frames is None,
            dynamics="Installed MJWarp + original ComFree contact response",
            collision="MJWarp standard mesh convex collision",
            stiffness=cfg.simulator.stiffness,
            damping=cfg.simulator.damping,
            impedance_width=0.01,
            finite=bool(np.isfinite(qs).all()),
            task_success=None,
            acceptance="No established dex task success criterion; tracking cost and completion are reported",
            windows=len(records),
            solve_seconds=sum(r["solve_seconds"] for r in records),
            total_wall_seconds=time.perf_counter() - started,
        ),
    )
    return [
        dict(
            task="dex",
            name=config.task["name"],
            method=config.solver.type,
            model=str(cfg.sequence.xml),
            trajectory="trajectory.npz",
            summary="summary.json",
            dt=grid.mpc_dt,
        )
    ]
