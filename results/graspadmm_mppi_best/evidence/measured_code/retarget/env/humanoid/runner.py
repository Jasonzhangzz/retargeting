"""Humanoid MPC orchestration with an optional cached-contact predictor."""

import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp

from retarget.common.results import clean_json, write_json
from retarget.solver.base import Problem, solve
from retarget.task.humanoid.metrics import assess_trajectory, timing_summary

from . import physics
from .runtime import HumanoidEnvironment, contact_settings, make_config


def validate(config):
    dynamics = config.environment.get("dynamics", "mjwarp")
    if (config.solver.type == "scsp") != (dynamics == "scsp"):
        raise ValueError("solver.type: scsp and environment.dynamics: scsp must be used together")
    if dynamics == "scsp":
        from .scsp import contact_options
        from retarget.solver.scsp import _load_ipopt

        contact_options(config.environment.get("scsp_contact"))
        _load_ipopt()
    elif config.environment.get("scsp_contact") is not None:
        raise ValueError("environment.scsp_contact requires dynamics: scsp")
    contact_graph = config.environment.get("contact_graph", False)
    if not isinstance(contact_graph, bool):
        raise ValueError("environment.contact_graph must be a boolean")
    if contact_graph and config.environment.get("dynamics", "mjwarp") != "frozen_contact":
        raise ValueError("environment.contact_graph requires dynamics: frozen_contact")
    if config.solver.gradient_steps and config.environment.get("dynamics", "mjwarp") != "frozen_contact":
        raise ValueError(
            "humanoid solver.gradient_steps requires environment.dynamics: frozen_contact"
        )
    contact_settings(
        config.environment.get("dynamics", "mjwarp"),
        config.environment.get("contact_cost"),
    )
    data = Path(config.task["data"])
    for name in ("scene.xml", "reference.npz", "source_config.yaml"):
        if not (data / name).is_file():
            raise ValueError(f"Missing humanoid input: {data / name}")
    if config.task.get("name") not in ("move_largebox", "sprint"):
        raise ValueError("humanoid task.name must be move_largebox or sprint")
    frames = config.environment.get("frames")
    if frames is not None and (not isinstance(frames, int) or frames < 1):
        raise ValueError("environment.frames must be a positive integer")
    c = make_config(data, config.device, config.solver.worlds)
    if dynamics == "scsp":
        if c.embodiment_type != "humanoid_object":
            raise ValueError("SCSP requires a humanoid_object dataset")
        options = contact_options(config.environment.get("scsp_contact"))
        c.point_contact_sites = options.hand_sites + options.foot_sites
        c.point_contact_radius = options.point_radius
        c.point_contact_friction = options.friction
        cpu = physics.setup_mj_model(c)
        for name in options.hand_sites + options.foot_sites:
            if mujoco.mj_name2id(cpu, mujoco.mjtObj.mjOBJ_SITE, name) < 0:
                raise ValueError(f"SCSP palm/sole site does not exist: {name}")
    raw = np.load(data / "reference.npz")
    if (
        raw["qpos"].shape[1] != c.nq
        or raw["ctrl"].shape[1] != c.nu
        or raw["qvel"].shape[1] != c.nv
    ):
        raise ValueError("Reference/model dimension mismatch")
    if not all(np.isfinite(raw[k]).all() for k in ("qpos", "qvel", "ctrl")):
        raise ValueError("Non-finite reference input")
    return data


def run(config, output):
    started = time.perf_counter()
    records = []
    data = Path(config.task["data"])
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    task = HumanoidEnvironment(
        config.task["name"],
        data,
        config.device,
        config.solver.worlds,
        record_predictions=False,
        dynamics=config.environment.get("dynamics", "mjwarp"),
        contact_options=config.environment.get("contact_cost"),
        contact_graph=config.environment.get("contact_graph", False),
        scsp_options=config.environment.get("scsp_contact"),
    )
    c = task.config
    sanity = dict(seconds=3.0, passed=False, policy="record_then_continue")
    try:
        m = physics.setup_mj_model(c)
        physics._initial_state_sanity_check(
            m, mujoco.MjData(m), *task.refs[:3], c, str(output)
        )
        sanity["passed"] = True
    except RuntimeError as exc:
        sanity["error"] = str(exc)
    write_json(output / "sanity_check.json", sanity)
    sigma = (
        torch.logspace(
            math.log10(c.first_ctrl_noise_scale),
            math.log10(c.last_ctrl_noise_scale),
            task.knots,
            device=config.device,
        )[:, None]
        .expand(-1, c.nu)
        .flatten()
        * c.joint_noise_scale
        / task.control_scale
    )
    mean = torch.zeros_like(sigma)
    if task.dynamics != "scsp":
        with torch.cuda.stream(task.stream):
            task.score(mean[None])
    torch.cuda.synchronize(config.device)
    initialized = time.perf_counter() - started
    full_steps = len(task.refs[0]) - c.horizon_steps - c.ctrl_steps
    frames = config.environment.get("frames")
    max_steps = (
        full_steps if frames is None else min(full_steps, int(frames) * c.ref_steps)
    )
    requested = math.ceil(max_steps / c.ctrl_steps)
    d = task.env.data_wp
    qs = [d.qpos.numpy()[0].copy()]
    vs = [d.qvel.numpy()[0].copy()]
    warm = [d.qacc_warmstart.numpy()[0].copy()]
    controls = []
    initial_time = float(d.time.numpy()[0])
    times = [initial_time]
    for window in range(requested):
        torch.cuda.synchronize(config.device)
        tick = time.perf_counter()
        graph_seconds_before = task.contact_graph_seconds
        with torch.cuda.stream(task.stream):
            if task.dynamics == "frozen_contact":
                task.prepare_contact_rollout()
            elif task.dynamics == "scsp":
                scsp_problem = task.prepare_scsp_problem(mean, seed=config.seed + window)
        torch.cuda.synchronize(config.device)
        contact_cache_seconds = time.perf_counter() - tick if task.dynamics in ("frozen_contact", "scsp") else 0.0
        optimization_started = time.perf_counter()
        with torch.cuda.stream(task.stream):
            result = solve(
                scsp_problem if task.dynamics == "scsp" else Problem(mean, sigma, task.score, task.project_parameters),
                config.solver, config.seed + window,
            )
            plan = task.controls(result.parameters[None])[0]
        torch.cuda.synchronize(config.device)
        solve_seconds = time.perf_counter() - tick
        optimization_seconds = time.perf_counter() - optimization_started
        tick = time.perf_counter()
        n = min(c.ctrl_steps, max_steps - task.first_step)
        with torch.cuda.stream(task.stream):
            q, v, w = task.execute(plan[:n])
            controls.extend(plan[:n].cpu().numpy())
            task.shift(plan, n)
        torch.cuda.synchronize(config.device)
        execute_seconds = time.perf_counter() - tick
        qs.extend(q)
        vs.extend(v)
        warm.extend(w)
        first = len(times)
        times.extend(initial_time + (first + i) * c.sim_dt for i in range(n))
        tick = time.perf_counter()
        np.savez_compressed(
            output / "trajectory.npz",
            qpos=qs,
            qvel=vs,
            ctrl=controls,
            qacc_warmstart=warm,
            time=times,
            qpos_ref=task.refs[0][: len(qs)].cpu().numpy(),
        )
        record = dict(
            window=window,
            solve_seconds=solve_seconds,
            contact_cache_seconds=contact_cache_seconds,
            optimization_seconds=optimization_seconds,
            contact_graph_capture_seconds=task.contact_graph_seconds - graph_seconds_before,
            execute_seconds=execute_seconds,
            iterations=len(result.history),
            sampled=result.sampled,
            evaluated=result.evaluated,
            cost=result.cost,
            history=result.history,
            io_seconds=time.perf_counter() - tick,
        )
        if task.dynamics == "scsp":
            model = task.scsp_model
            selection = result.selection
            actual = model.last_prediction
            local, normals = model.select(selection.indices[None])
            name = f"scsp_contacts_{window:05d}.npz"
            np.savez_compressed(
                output / name,
                site_ids=model.contact_ids,
                robot_body_ids=task.env.model_cpu.site_bodyid[list(model.contact_ids)],
                robot_local_points=task.env.model_cpu.site_pos[list(model.contact_ids)],
                robot_point_jacobian=actual.site_jacobian[0].cpu().numpy(),
                real_site_positions=actual.site_xpos[0].cpu().numpy(),
                real_contact_forces=actual.contact_force[0].cpu().numpy(),
                real_contact_active=actual.active_mask[0].cpu().numpy(),
                real_contact_gap=actual.contact_gap[0].cpu().numpy(),
                real_contact_count=actual.actual_contact_count[0].cpu().numpy(),
                real_constraint_count=actual.constraint_count[0].cpu().numpy(),
                operation_mask=model.operation_mask.cpu().numpy(),
                position_consensus_mask=model.contact_position_mask.cpu().numpy(),
                consensus_mask=model.consensus_mask.cpu().numpy(),
                candidate_indices=selection.indices.cpu().numpy(),
                object_local_points=local[0].cpu().numpy(),
                object_local_normals=normals[0].cpu().numpy(),
                force_knots=selection.forces.reshape(model.knots, model.hands, 3).cpu().numpy(),
                object_contact_world=(selection.consensus[..., :3]*model.options.position_scale).cpu().numpy(),
                object_force_world=(selection.consensus[..., 3:]*model.options.force_scale).cpu().numpy(),
                robot_contact_world=(result.consensus[..., :3]*model.options.position_scale).cpu().numpy(),
                robot_force_world=(result.consensus[..., 3:]*model.options.force_scale).cpu().numpy(),
                scaled_dual=result.dual.cpu().numpy(),
            )
            record.update(scsp_converged=result.converged, scsp_contacts=name,
                          contact_universe=121, unique_contact_combinations=result.sampled)
        record["io_seconds"] = time.perf_counter() - tick
        records.append(record)
        with (output / "iterations.jsonl").open("a") as log:
            log.write(json.dumps(clean_json(record), allow_nan=False) + "\n")
        write_json(
            output / "status.json",
            dict(
                state="running",
                completed_windows=len(records),
                requested_windows=requested,
            ),
        )
        print(
            f"[{config.solver.type} {window + 1}/{requested}] cost={result.cost:.6g} solve={solve_seconds:.3f}s",
            flush=True,
        )
    complete = frames is None and len(records) == requested
    summary = assess_trajectory(
        task.env.model_cpu,
        np.asarray(qs),
        task.refs[0][: len(qs)].cpu().numpy(),
        complete,
        c.embodiment_type == "humanoid_object",
    )
    summary.update(
        solver=config.solver.type,
        rollout_model=task.dynamics,
        contact_cache_builds=task.contact_cache_builds,
        contact_graph=task.contact_graph,
        contact_graph_builds=task.contact_graph_builds,
        contact_graph_capture_seconds=task.contact_graph_seconds,
        domain_randomization=bool(c.num_dr),
        seed=config.seed,
        completed_windows=len(records),
        initialize_and_jit_seconds=initialized,
        timing=timing_summary(records),
        sampled=sum(r["sampled"] for r in records),
        evaluated=sum(r["evaluated"] for r in records),
        gradient_accepted_steps=sum(
            h.get("gradient_accepted_steps", 0) for r in records for h in r["history"]
        ),
        gradient_evaluated=sum(
            h.get("gradient_evaluated", 0) for r in records for h in r["history"]
        ),
        scsp_converged_windows=sum(r.get("scsp_converged", False) for r in records),
        total_wall_seconds=time.perf_counter() - started,
        sanity_check_passed=sanity["passed"],
    )
    write_json(output / "summary.json", summary)
    write_json(
        output / "model.json",
        dict(
            physics=asdict(c),
            rollout_model=task.dynamics,
            execution_model="mjwarp",
            domain_randomization=bool(c.num_dr),
            source_num_dr=task.source_num_dr,
            contact_cost=(
                asdict(task.contact_options) if task.contact_options is not None else None
            ),
            contact_graph=task.contact_graph,
            scsp_contact=asdict(task.scsp_options) if task.scsp_options is not None else None,
            source=json.loads((data.parent / "source.json").read_text()),
            input_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [data / "scene.xml", data / "reference.npz"]
            },
            gpu=torch.cuda.get_device_name(config.device),
            versions=dict(
                torch=torch.__version__, warp=wp.__version__, mujoco=mujoco.__version__
            ),
        ),
    )
    return [
        dict(
            task="humanoid",
            name=config.task["name"],
            method=config.solver.type,
            rollout_model=task.dynamics,
            model=str(data / "scene.xml"),
            trajectory="trajectory.npz",
            summary="summary.json",
            dt=c.sim_dt,
        )
    ]
