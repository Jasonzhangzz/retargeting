#!/usr/bin/env python3
"""Assess a saved FA pushing run, including an interrupted diagnostic prefix."""

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retarget.common.results import write_json
from retarget.env.humanoid.dexforge_replay import replay_controls
from retarget.task.humanoid.metrics import assess_trajectory
from retarget.dynamics.dexforge_contact_validation import _hand_object_slots


def assess(model, filename):
    with np.load(filename, allow_pickle=False) as saved:
        qpos, target, times = saved["qpos"], saved["qpos_ref"], saved["time"]
    result = assess_trajectory(model, qpos, target, False)
    result["executed_seconds"] = float(times[-1]-times[0])
    result["root_height_deficit_max_m"] = float(np.maximum(target[:,2]-qpos[:,2], 0).max())
    for key in ("first_failure_frame", "first_balance_failure_frame"):
        frame = result[key]
        result[key.replace("frame", "seconds")] = None if frame is None else float(times[frame])
    displacement = qpos[-1, -7:-4]-qpos[0, -7:-4]
    desired = target[-1, -7:-4]-target[0, -7:-4]
    desired_distance = float(np.linalg.norm(desired))
    result.update(box_displacement_norm_m=float(np.linalg.norm(displacement)),
                  reference_box_displacement_norm_m=desired_distance,
                  box_progress_along_reference_m=float(displacement@desired/max(desired_distance,1e-30)),
                  interpretation="Diagnostic prefix; task_success requires certified full-clip execution")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    folder = args.output.resolve()
    scene = folder/"native_scene/source_scene.xml"
    trajectory = folder/"trajectory.npz"
    model = mujoco.MjModel.from_xml_path(str(scene))
    if not (folder/"independent_mujoco/trajectory.npz").exists():
        replay_controls(scene, trajectory, folder, complete=False)
    status = json.loads((folder/"status.json").read_text())
    gate = json.loads((folder/"gradient_validation.json").read_text())
    with np.load(trajectory) as values:
        timestamps = values["time"].copy()
    dt = float(np.median(np.diff(timestamps)))
    result = dict(run_status=status, native=assess(model, trajectory),
                  independent_mujoco=assess(model, folder/"independent_mujoco/trajectory.npz"),
                  gradient_certified=bool(gate.get("passed", False)))
    rows = [json.loads(row) for row in (folder/"iterations.jsonl").read_text().splitlines() if row.strip()]
    sites = [model.site(name).id for name in ("left_palm", "right_palm")]
    obj = model.body("largebox").id
    samples, hand_times, maxima, prediction_errors = [], [], [], []
    floor = model.geom("floor").id
    robot_depth, box_depth = 0., 0.
    executed_before = 0
    for row in rows:
        count = row["executed_steps"]-executed_before
        with np.load(folder/row["contact_prediction_file"]) as values:
            geoms = values["contact_geom_ids"]
            mask = _hand_object_slots(model, geoms, sites, obj)
            active = values["contact_active"][:count, mask]
            force = np.linalg.norm(values["contact_force_on_geom2"][:count, mask], axis=-1)
            times = (executed_before+np.arange(count))*dt
            for step in np.flatnonzero(active.any(1)):
                hand_times.append(float(times[step]))
            maxima.append(float(force.max(initial=0)))
            samples.extend(active.any(1).tolist())
            floor_slots = np.any(geoms == floor, axis=1)
            object_slots = np.any(model.geom_bodyid[geoms] == obj, axis=1)
            depth = np.maximum(-values["contact_distance"][:count], 0.)
            robot_depth = max(robot_depth, float(depth[:,floor_slots & ~object_slots].max(initial=0)))
            box_depth = max(box_depth, float(depth[:,floor_slots & object_slots].max(initial=0)))
        prediction_errors.append(row["planned_executed_qpos_max"])
        executed_before = row["executed_steps"]
    result["executed_hand_object_contacts"] = dict(
        first_contact_seconds=min(hand_times) if hand_times else None,
        last_contact_seconds=max(hand_times) if hand_times else None,
        active_steps=int(np.count_nonzero(samples)), total_steps=len(samples),
        maximum_force_n=max(maxima, default=0),
        stage="native pre-integration geometry of executed steps; future prediction excluded")
    result["planning_execution_qpos_max_difference"] = max(prediction_errors, default=None)
    result["executed_max_penetration_m"] = dict(robot_floor=robot_depth, box_floor=box_depth)
    result["completed_windows"] = len(rows)
    result["mean_solve_seconds"] = float(np.mean([row["solve_seconds"] for row in rows])) if rows else None
    write_json(folder/"effect_evaluation.json", result)
    # A failed solve still has a useful, explicitly partial saved trajectory.
    if not (folder/"manifest.json").exists():
        write_json(folder/"manifest.json", dict(version=1, task="humanoid", method="dexforge",
            cases=[dict(task="humanoid", name="move_largebox_diagnostic_prefix", method="dexforge",
                        model=str(scene), trajectory="trajectory.npz", summary="effect_evaluation.json", dt=dt)]))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
