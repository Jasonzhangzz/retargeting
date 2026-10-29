"""Cached free-object gravity playback using the native Warp executor.

This tests the saved pose and position command directly. It never rebuilds a
grasp, resets to Dexonomy, fixes the object, or invokes an MPC recovery policy.
"""
from retarget.grasp_capture import capture_guard
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp

VERSION = 1


def playback_paths(case, *, duration=3., saved_velocity=False, physics_dt=.0005):
    case = Path(case).resolve()
    report = json.loads((case/'result.json').read_text())
    experiment = case/f"input_{report['selected']['start']:02d}"
    settings = dict(version=VERSION, duration=float(duration), saved_velocity=bool(saved_velocity),
                    physics_dt=float(physics_dt), sample_dt=.02, gravity=[0., 0., -9.81],
                    policy='fixed archived hold_action; free object; no fixture or MPC')
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for path in (case/'grasp.npz', experiment/'forceaware.yaml',
                 experiment/'sequence/scene/wuji_right/primitive_sdf.xml'):
        digest.update(path.read_bytes())
    stem = case/'viser_playback'/digest.hexdigest()[:20]
    return stem.with_suffix('.npz'), stem.with_suffix('.json'), settings


@wp.kernel(enable_backward=False)
def record_state(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
                 positions: wp.array2d(dtype=float), velocities: wp.array2d(dtype=float),
                 index: wp.array(dtype=int)):
    col = wp.tid()
    row = index[0]
    if col < positions.shape[1]:
        positions[row, col] = qpos[0, col]
    if col < velocities.shape[1]:
        velocities[row, col] = qvel[0, col]


@wp.kernel(enable_backward=False)
def next_frame(index: wp.array(dtype=int), condition: wp.array(dtype=int), total: int):
    index[0] += 1
    condition[0] = int(index[0] < total)


def simulate_playback(case, *, duration=3., saved_velocity=False, physics_dt=.0005):
    """One device trajectory graph, one completion wait, then cache the result."""
    from dataclasses import replace
    import comfree_warp
    from forceaware.config import load_retarget_config
    from forceaware.runtime import setup_runtime
    from forceaware.targets import build_index
    from forceaware.time_grid import TimeGrid
    from forceaware.workflow import _configure_comfree_physics
    from forceaware.executor import WarpActionExecutor
    from retarget.grasp_simulation import retention_metrics

    begun = perf_counter()
    case = Path(case).resolve()
    source = json.loads((case/'result.json').read_text())
    if source.get('selected', {}).get('termination') == 'initialization_failed':
        raise ValueError('Initialization failed; no optimized grasp is available for gravity playback')
    path, metadata, settings = playback_paths(case, duration=duration,
        saved_velocity=saved_velocity, physics_dt=physics_dt)
    if path.exists() and metadata.exists():
        return path, json.loads(metadata.read_text())
    experiment = case/f"input_{source['selected']['start']:02d}"
    cfg = load_retarget_config(experiment/'sequence', experiment/'forceaware.yaml',
                              contactaware_dir=experiment/'contactaware')
    frames = int(round(duration/.02))+1
    if frames < 2 or not np.isclose((frames-1)*.02, duration):
        raise ValueError('Playback duration must be a positive multiple of 20 ms')
    grid = TimeGrid.create(mpc_dt=.02, action_dt=.02, knot_dt=.02,
                           exec_dt=physics_dt, ref_dt=cfg.simulator.time_grid.ref_dt)
    cfg = replace(cfg, simulator=replace(cfg.simulator, time_grid=grid))
    setup_runtime(cfg)
    device, model = comfree_warp.load_model(str(cfg.sequence.xml), device='cuda:0')
    _configure_comfree_physics(cfg, model, device, timestep=physics_dt)
    model.opt.gravity[:] = (0., 0., -9.81)
    device.opt.gravity.fill_((0., 0., -9.81))
    index = build_index(model)
    executor = WarpActionExecutor(model, device, index=index, cfg=cfg)
    with np.load(case/'grasp.npz', allow_pickle=False) as saved:
        qpos = saved['qpos'].astype(np.float32)
        qvel = saved['qvel'].astype(np.float32) if saved_velocity else np.zeros(model.nv, np.float32)
        action = saved['hold_action'].astype(np.float32)
    if not all(np.isfinite(value).all() for value in (qpos, qvel, action)):
        raise ValueError('Saved playback pose/velocity/command is nonfinite')
    original_q = wp.array(qpos[None], dtype=float, device=executor.device)
    original_v = wp.array(qvel[None], dtype=float, device=executor.device)
    positions = wp.empty((frames, model.nq), dtype=float, device=executor.device)
    velocities = wp.empty((frames, model.nv), dtype=float, device=executor.device)
    frame = wp.zeros(1, dtype=int, device=executor.device)
    condition = wp.zeros(1, dtype=int, device=executor.device)
    state = executor._initial_state(qpos, qvel)
    executor._assign_action(action)
    executor._clear_status()
    # Load/allocate native kernels before capture; do not retain the warmup state.
    executor._advance(state, 0)
    wp.synchronize_device(executor.device)
    wp.load_module(module=__name__, device=executor.device)

    def advance_frame():
        advanced, _ = executor._advance(state, 0)
        wp.copy(executor.qpos, advanced.qpos)
        wp.copy(executor.qvel, advanced.qvel)
        wp.copy(executor.time, advanced.time)
        wp.launch(record_state, max(model.nq, model.nv), inputs=[executor.qpos, executor.qvel,
            positions, velocities, frame], device=executor.device)
        wp.launch(next_frame, 1, inputs=[frame, condition, frames], device=executor.device)

    # The GPU loop repeats one physical frame graph. Neither simulation state
    # nor its termination decision is read by the host between frames.
    with capture_guard(), wp.ScopedCapture(device=executor.device) as capture:
        wp.copy(executor.qpos, original_q)
        wp.copy(executor.qvel, original_v)
        executor.time.zero_()
        executor._clear_status()
        frame.zero_()
        wp.launch(record_state, max(model.nq, model.nv), inputs=[executor.qpos, executor.qvel,
            positions, velocities, frame], device=executor.device)
        wp.launch(next_frame, 1, inputs=[frame, condition, frames], device=executor.device)
        wp.capture_while(condition, advance_frame)
    graph = capture.graph
    setup_seconds = perf_counter()-begun
    replay = perf_counter()
    wp.capture_launch(graph)
    wp.synchronize_device(executor.device)
    replay_seconds = perf_counter()-replay
    failure = None
    try:
        executor._check_status()
    except Exception as error:
        failure = str(error)
    states, speeds = positions.numpy(), velocities.numpy()
    finite = np.isfinite(states).all(-1)&np.isfinite(speeds).all(-1)
    invalid = np.flatnonzero(~finite)
    if len(invalid):
        stop = max(1, int(invalid[0]))
        states, speeds = states[:stop], speeds[:stop]
        failure = failure or 'Native physics produced a nonfinite state'
    times = np.arange(len(states))*.02
    measures = [retention_metrics(q, states[0], index.object_qpos, distance_mm=30.) for q in states]
    fall = np.array([m['relative_fall_mm'] for m in measures])
    motion = np.array([m['relative_displacement_mm'] for m in measures])
    drops = np.flatnonzero(fall >= 30.)
    escapes = np.flatnonzero(motion >= 30.)
    status = 'simulation_failed' if failure else ('dropped' if len(drops) else ('escaped' if len(escapes) else 'retained'))
    result = dict(settings, status=status, simulated_seconds=float(times[-1]), frames=len(states),
        dropped=bool(len(drops)), escaped=bool(len(escapes)), retained=status == 'retained',
        first_drop_seconds=float(times[drops[0]]) if len(drops) else None,
        first_escape_seconds=float(times[escapes[0]]) if len(escapes) else None,
        maximum_relative_fall_mm=float(fall.max()), maximum_relative_displacement_mm=float(motion.max()),
        maximum_hand_base_shift_mm=float(np.linalg.norm(states[:, :3]-states[0, :3], axis=1).max()*1000),
        drop_threshold_mm=30., backend='native ComFree Warp hard-contact executor',
        initial_state='saved grasp.npz qpos; saved qvel' if saved_velocity else 'saved grasp.npz qpos; zero initial velocity',
        setup_seconds=setup_seconds, replay_seconds=replay_seconds, wall_seconds=perf_counter()-begun,
        simulation_error=failure, source_case=str(case))
    path.parent.mkdir(exist_ok=True)
    temporary = path.with_suffix('.pending.npz')
    np.savez_compressed(temporary, qpos=states, qvel=speeds, time=times,
                        relative_fall_mm=fall, relative_displacement_mm=motion, hold_action=action)
    temporary.replace(path)
    temporary = metadata.with_suffix('.pending.json')
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(metadata)
    return path, result
