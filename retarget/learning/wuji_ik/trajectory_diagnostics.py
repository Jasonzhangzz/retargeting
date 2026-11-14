"""Within-segment trajectory errors, jump residuals, and paired temporal drift."""
import numpy as np


def bootstrap_mean(values, seed=42, samples=1000):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return dict(mean=None, ci95=[None, None])
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(samples, len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()), ci95=np.quantile(means, [.025, .975]).tolist())


def quantiles(values):
    values = np.asarray(values)
    return dict(zip(('p50', 'p95', 'p99', 'p999', 'max'),
                    np.quantile(values, [.5, .95, .99, .999, 1.]).tolist())) if values.size else {}


def analyze_trajectories(prediction, target, offsets):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    offsets = np.asarray(offsets)
    if (prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 20
            or offsets[0] != 0 or offsets[-1] != len(prediction) or np.any(np.diff(offsets) <= 0)
            or not np.isfinite(prediction).all() or not np.isfinite(target).all()):
        raise ValueError('Expected finite N x 20 trajectories and nonempty segment offsets')
    pred = np.rad2deg(prediction); truth = np.rad2deg(target); error = pred-truth
    frame_mae = np.abs(error).mean(axis=1)
    rows = []; jumps = []; true_jumps = []; delta_errors = []; spike_strengths = []
    progress = []; drift = []; eligible = []
    for index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        e = error[start:end]; n = len(e); mae = frame_mae[start:end]
        dq = np.diff(pred[start:end], axis=0); dt = np.diff(truth[start:end], axis=0); de = np.diff(e, axis=0)
        jumps.append(dq); true_jumps.append(dt); delta_errors.append(de)
        strength = np.minimum(np.abs(de[:-1]), np.abs(de[1:]))*(de[:-1]*de[1:] < 0)
        spike_strengths.append(strength)
        k = max(1, n//5)
        frame, joint = np.unravel_index(np.argmax(np.abs(de)), de.shape) if de.size else (0, 0)
        spike_frame, spike_joint = np.unravel_index(np.argmax(strength), strength.shape) if strength.size else (0, 0)
        row = dict(index=index, frames=n, mae_deg=float(mae.mean()), rmse_deg=float(np.sqrt(np.square(e).mean())),
            early_mae_deg=float(mae[:k].mean()), late_mae_deg=float(mae[-k:].mean()),
            late_minus_early_deg=float(mae[-k:].mean()-mae[:k].mean()),
            max_prediction_jump_deg=float(np.abs(dq).max()) if dq.size else 0.,
            max_sdk_jump_deg=float(np.abs(dt).max()) if dt.size else 0.,
            max_delta_error_deg=float(np.abs(de).max()) if de.size else 0.,
            max_spike_strength_deg=float(strength.max()) if strength.size else 0.,
            spike_frames_over_5deg=int(np.any(strength>5, axis=1).sum()) if strength.size else 0,
            worst_delta_frame=int(frame+1), worst_delta_joint=int(joint),
            worst_spike_frame=int(spike_frame+1), worst_spike_joint=int(spike_joint))
        rows.append(row)
        if n >= 40:
            eligible.append(index); drift.append(row['late_minus_early_deg'])
            progress.append([float(x.mean()) for x in np.array_split(mae, 20)])
    dq = np.concatenate(jumps); dt = np.concatenate(true_jumps); de = np.concatenate(delta_errors)
    strength = np.concatenate(spike_strengths)
    thresholds = {}
    for threshold in (2, 5, 10, 20):
        counts = dict(prediction=int(np.any(np.abs(dq)>threshold, axis=1).sum()),
                      sdk=int(np.any(np.abs(dt)>threshold, axis=1).sum()),
                      delta_error=int(np.any(np.abs(de)>threshold, axis=1).sum()))
        thresholds[str(threshold)] = dict(frame_pair_counts=counts,
            frame_pair_rates={key:value/max(1,len(de)) for key,value in counts.items()})
    prog = np.asarray(progress)
    rng = np.random.default_rng(42)
    if len(prog):
        boot = prog[rng.integers(len(prog), size=(500,len(prog)))].mean(axis=1)
        progress_report = dict(segments=len(prog), minimum_frames=40,
            mean_deg=prog.mean(axis=0).tolist(), median_deg=np.median(prog,axis=0).tolist(),
            ci95_low_deg=np.quantile(boot,.025,axis=0).tolist(), ci95_high_deg=np.quantile(boot,.975,axis=0).tolist())
    else:
        progress_report = dict(segments=0, minimum_frames=40)
    horizon = next((n for n in (200,100,50,20) if sum(row['frames']>=n for row in rows)>=20), None)
    absolute_report = None
    if horizon:
        fixed = np.asarray([frame_mae[start:start+horizon] for start,end in zip(offsets[:-1],offsets[1:]) if end-start>=horizon])
        boot = fixed[rng.integers(len(fixed),size=(500,len(fixed)))].mean(axis=1)
        absolute_report = dict(horizon=horizon, segments=len(fixed), mean_deg=fixed.mean(axis=0).tolist(),
            ci95_low_deg=np.quantile(boot,.025,axis=0).tolist(), ci95_high_deg=np.quantile(boot,.975,axis=0).tolist())
    result = dict(frames=len(prediction), segments=len(rows), consecutive_frame_pairs=len(de),
        mae_deg=float(np.abs(error).mean()), rmse_deg=float(np.sqrt(np.square(error).mean())),
        prediction_joint_delta_deg=quantiles(np.abs(dq)), sdk_joint_delta_deg=quantiles(np.abs(dt)),
        delta_error_joint_deg=quantiles(np.abs(de)), isolated_error_spike_strength_deg=quantiles(strength),
        thresholds_degrees_per_frame=thresholds,
        isolated_spike_frames_over_5deg=int(np.any(strength>5,axis=1).sum()) if strength.size else 0,
        eligible_frame_triples=len(strength),
        paired_drift=dict(segments=len(drift), minimum_frames=40, early_fraction=.2, late_fraction=.2,
            late_minus_early_deg=bootstrap_mean(drift),
            median_difference_deg=float(np.median(drift)) if drift else None,
            fraction_late_worse=float(np.mean(np.asarray(drift)>0)) if drift else None),
        normalized_progress=progress_report, fixed_history_cohort=absolute_report,
        definitions=dict(delta_error='(q_pred[t]-q_pred[t-1])-(q_SDK[t]-q_SDK[t-1]); never crosses a segment boundary',
            isolated_spike='Error increments on both sides exceed 5 degrees and reverse sign at an interior frame',
            drift='Paired comparison of last 20% minus first 20% mean absolute error in trajectories >=40 frames',
            confidence_intervals='95% trajectory bootstrap; frames within a trajectory are not independent',
            time='Frame index; no assumed FPS; no q feedback or closed-loop dynamics'))
    return result, rows, dict(frame_mae_deg=frame_mae, prediction_delta_deg=dq, sdk_delta_deg=dt, delta_error_deg=de)
