"""Independent MuJoCo transfer of controls optimized with native ForceAware."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from retarget.common.results import write_json
from retarget.task.humanoid.metrics import assess_trajectory


def replay_controls(viewer_xml, trajectory_npz, output_dir, *, complete):
    """Replay the saved dense 29-actuator sequence with original geometry.

    The initial qpos/qvel and all control timestamps come from the native FA
    trajectory. Forward kinematics initializes derived fields without advancing
    time. Every subsequent row is an ordinary MuJoCo step, independent of the
    native ComFree model or optimizer. Results are labeled as transfer evidence.
    """
    model = mujoco.MjModel.from_xml_path(str(Path(viewer_xml).resolve()))
    with np.load(trajectory_npz, allow_pickle=False) as saved:
        control = np.asarray(saved["ctrl"], np.float64).copy()
        time = np.asarray(saved["time"], np.float64).copy()
        target = np.asarray(saved["qpos_ref"], np.float64).copy()
        qpos0 = np.asarray(saved["qpos0"] if "qpos0" in saved else saved["qpos"][0], np.float64).copy()
        qvel0 = np.asarray(saved["qvel0"] if "qvel0" in saved else saved["qvel"][0], np.float64).copy()
    if control.ndim != 2 or control.shape[1] != model.nu or model.nu != 29:
        raise ValueError("independent humanoid replay requires exactly 29 actuator controls per step")
    if time.shape != (len(control)+1,) or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("independent replay requires one strictly increasing timestamp per state")
    if qpos0.shape != (model.nq,) or qvel0.shape != (model.nv,) or target.shape != (len(time), model.nq):
        raise ValueError("independent replay initial state/reference dimensions do not match the scene")
    if not all(np.isfinite(value).all() for value in (control, qpos0, qvel0, target)):
        raise ValueError("independent replay inputs must be finite")
    data = mujoco.MjData(model)
    data.qpos[:] = qpos0
    data.qvel[:] = qvel0
    data.time = float(time[0])
    mujoco.mj_forward(model, data)
    positions, velocities = [qpos0.copy()], [qvel0.copy()]
    for step, action in enumerate(control):
        model.opt.timestep = float(time[step+1]-time[step])
        data.ctrl[:] = action
        mujoco.mj_step(model, data)
        positions.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
    positions, velocities = np.asarray(positions), np.asarray(velocities)
    folder = Path(output_dir).resolve() / "independent_mujoco"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "trajectory.npz", qpos=positions, qvel=velocities,
                        ctrl=control, time=time, qpos_ref=target,
                        qpos0=qpos0, qvel0=qvel0)
    summary = assess_trajectory(model, positions, target, bool(complete))
    warnings = {mujoco.mjtWarning(i).name: int(data.warning[i].number)
                for i in range(len(data.warning)) if data.warning[i].number}
    reset_warnings = ("mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC")
    numerical_reset = any(warnings.get(name, 0) > 0 for name in reset_warnings)
    if numerical_reset:
        summary["task_success"] = False
    summary.update(evaluation="independent_mujoco_transfer", source_solver="upstream_forceaware",
                   execution_model="mujoco_original_geometry", execution_integrator="Euler",
                   model_xml=str(Path(viewer_xml).resolve()), source_trajectory=str(Path(trajectory_npz).resolve()),
                   simulated_seconds=float(time[-1]-time[0]), executed_steps=len(control),
                   initialization_prestep=False, numerical_reset=numerical_reset, warnings=warnings)
    write_json(folder / "summary.json", summary)
    return summary
