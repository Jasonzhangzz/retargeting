import numpy as np
import mujoco
from retarget.common.results import clean_json as json_finite

ACCEPTANCE = dict(
    box_position_max_m=0.1,
    box_rotation_max_rad=0.3,
    root_height_deficit_max_m=0.18,
    torso_tilt_max_deg=60.0,
)


def assess_trajectory(model, actual, target, complete, humanoid_object=True):
    """External acceptance only; these thresholds never enter the task cost."""
    finite = bool(np.isfinite(actual).all())
    d, dr = mujoco.MjData(model), mujoco.MjData(model)
    torso = model.body("torso_link").id
    tilt = np.full(len(actual), np.inf)
    for i, (q, r) in enumerate(zip(actual, target)):
        if not np.isfinite(q).all():
            continue
        d.qpos[:], dr.qpos[:] = q, r
        mujoco.mj_forward(model, d)
        mujoco.mj_forward(model, dr)
        dot = d.xmat[torso].reshape(3, 3)[:, 2] @ dr.xmat[torso].reshape(3, 3)[:, 2]
        tilt[i] = np.degrees(np.arccos(np.clip(dot, -1, 1)))
    balance = (
        target[:, 2] - actual[:, 2] > ACCEPTANCE["root_height_deficit_max_m"]
    ) | (tilt > ACCEPTANCE["torso_tilt_max_deg"])
    failure = balance | ~np.isfinite(actual).all(1)
    result = dict(
        acceptance=ACCEPTANCE,
        full_clip_completed=bool(complete),
        finite=finite,
        root_position_rms_m=float(
            np.sqrt(np.mean(np.sum((actual[:, :3] - target[:, :3]) ** 2, axis=1)))
        ),
        torso_tilt_max_deg=float(tilt.max()),
        balance_failure_frames=int(balance.sum()),
        first_balance_failure_frame=int(np.flatnonzero(balance)[0])
        if balance.any()
        else None,
    )
    if humanoid_object:
        position = np.linalg.norm(actual[:, -7:-4] - target[:, -7:-4], axis=1)
        # Physical quaternion angle is sign invariant, in radians.
        qa, qr = actual[:, -4:], target[:, -4:]
        norm = np.linalg.norm(qa, axis=1) * np.linalg.norm(qr, axis=1)
        dot = np.abs(np.sum(qa * qr, axis=1) / np.maximum(norm, 1.0e-30))
        angle = 2 * np.arccos(np.clip(dot, 0, 1))
        failure |= (
            (position > ACCEPTANCE["box_position_max_m"])
            | (angle > ACCEPTANCE["box_rotation_max_rad"])
            | (norm < 1.0e-12)
        )
        result.update(
            box_position_rms_m=float(np.sqrt(np.mean(position**2))),
            box_position_max_m=float(position.max()),
            box_final_error_m=float(position[-1]),
            box_rotation_max_rad=float(angle.max()),
            box_final_rotation_rad=float(angle[-1]),
            box_displacement_m=(actual[-1, -7:-4] - actual[0, -7:-4]).tolist(),
            reference_box_displacement_m=(
                target[-1, -7:-4] - target[0, -7:-4]
            ).tolist(),
        )
    result.update(
        task_success=bool(complete and finite and not failure.any()),
        first_failure_frame=int(np.flatnonzero(failure)[0]) if failure.any() else None,
        acceptance_origin="Project operational criteria, not an official SPIDER success definition",
    )
    return json_finite(result)


def timing_summary(records):
    solve = np.asarray([r["solve_seconds"] for r in records])
    steady = solve[1:]
    return dict(
        first_solve_seconds=float(solve[0]) if len(solve) else None,
        subsequent_solve_mean_seconds=float(steady.mean()) if len(steady) else None,
        subsequent_solve_p50_seconds=float(np.median(steady)) if len(steady) else None,
        subsequent_solve_p95_seconds=float(np.percentile(steady, 95))
        if len(steady)
        else None,
        solve_total_seconds=float(solve.sum()),
        execute_total_seconds=sum(r["execute_seconds"] for r in records),
        io_total_seconds=sum(r["io_seconds"] for r in records),
    )
