"""Bounded-cost branch discovery and velocity-limited source-IK transitions.

Exploration never changes the objective, model, limits or published history.
A better full-range solution is a destination, not a publishable warm-frame
candidate. Crossing a local minimum can temporarily increase tracking error;
the transition is reported explicitly and uses the original frame speed box.
"""
import numpy as np

REVISION = 'source_ik_branch_recovery_v2_calibration_seeds'
PROBE_INTERVAL_FRAMES = 8
MIN_RELATIVE_IMPROVEMENT = .05


class BranchRecovery:
    def __init__(self, kinematics, config, bank=None):
        self.kinematics = kinematics
        self.config = config
        lo, hi = kinematics.limits.T
        # Generic fallback seeds. Calibrated poses below are starting points;
        # runtime selection never reads gesture labels or uses pose targets.
        self.seeds = tuple(lo + fraction*(hi-lo) for fraction in (.5, .25, .75))
        self.calibration_seeds = tuple(np.asarray(s['joint_positions_rad'], float)
                                       for s in (bank or {}).get('seeds', []))
        self.calibration_sha256 = (bank or {}).get('sha256')
        self.reset()

    def reset(self):
        self.targets = None
        self.valid = np.zeros(5, dtype=bool)
        self.transitioning = np.zeros(5, dtype=bool)
        self.frame_index = 0
        self.next_probe_frame = 0

    def policy(self):
        return dict(revision=REVISION, seeds=len(self.seeds),
                    probe_interval_frames=PROBE_INTERVAL_FRAMES,
                    position_trigger_m=self.config['retry_position_m'],
                    rotation_trigger_rad=self.config['retry_rotation_rad'],
                    minimum_relative_improvement=MIN_RELATIVE_IMPROVEMENT,
                    calibration_seed_count=len(self.calibration_seeds),
                    calibration_seed_sha256=self.calibration_sha256,
                    calibrated_seed_policy='best_current_full_cost_per_finger_replaces_third_generic_probe',
                    trigger='large_residual_and_stalled_local_descent',
                    selection='same_full_objective_and_frame_reference',
                    transition='joint_space_segment_with_original_frame_speed_box')

    def solve(self, native, start, positions, rotations, dt, history):
        q, base = native.solve(start, positions, rotations, dt, history)
        failed = ((base[:, 0] > max(self.config['retry_position_m'], 1e-8)) |
                  (base[:, 1] > max(self.config['retry_rotation_rad'], 1e-8)))
        # Large residual can simply mean a fast hand is moving faster than
        # this frame's speed box. Preserve a healthy local descent; discover
        # branches only after it stalls, or continue an already chosen path.
        eligible = failed.copy()
        if history and failed.any():
            previous = native.score(start, positions, rotations, start, history)
            stalled = (previous[:, 3]-base[:, 3]) <= np.maximum(1e-6, .01*previous[:, 3])
            eligible &= stalled
        continuing = failed & self.transitioning
        candidates_for = eligible | continuing
        targets = start.copy() if self.targets is None else self.targets.copy()
        valid = self.valid.copy()
        scores = base[:, 3].copy()
        iterations = base[:, 2].copy()
        probe_calls = 0
        next_probe = self.next_probe_frame
        periodic = np.asarray([name in getattr(native, 'periodic_joints', ())
                               for name in self.kinematics.joint_names])

        def consider(candidate, report, mask):
            for f in np.flatnonzero(mask):
                if report[f, 3] < scores[f]:
                    ids = self.kinematics.finger_indices[f]
                    targets[ids] = candidate[ids]
                    scores[f] = report[f, 3]
                    valid[f] = True

        if self.targets is not None and (candidates_for & valid).any():
            cached = native.score(targets, positions, rotations, start, history)
            consider(targets.copy(), cached, candidates_for & valid)
        calibration_scores = 0
        if eligible.any() and self.frame_index >= next_probe:
            starts = list(self.seeds)
            if self.calibration_seeds:
                calibrated = start.copy(); best = np.full(5, np.inf)
                for seed in self.calibration_seeds:
                    report = native.score(seed, positions, rotations, start, history)
                    calibration_scores += 1
                    for f, ids in enumerate(self.kinematics.finger_indices):
                        if report[f, 3] < best[f]:
                            best[f] = report[f, 3]; calibrated[ids] = seed[ids]
                # Preserve the three-solve online exploration budget.
                starts[-1] = calibrated
            for seed in starts:
                seed = seed.copy()
                seed[periodic] = start[periodic] + (seed[periodic]-start[periodic]+np.pi) % (2*np.pi)-np.pi
                candidate, report = native.solve(seed, positions, rotations, dt, history,
                                                  reference=start, probe=True)
                iterations += report[:, 2]
                probe_calls += 1
                consider(candidate, report, eligible)
            next_probe = self.frame_index + PROBE_INTERVAL_FRAMES

        improved = (base[:, 3]-scores) > np.maximum(1e-6, MIN_RELATIVE_IMPROVEMENT*base[:, 3])
        selected = candidates_for & valid & improved
        per_finger = []
        for f, ids in enumerate(self.kinematics.finger_indices):
            fraction = 0.
            if selected[f]:
                delta = targets[ids]-start[ids]
                largest = float(np.max(np.abs(delta)))
                fraction = min(1., self.config['v_max']*dt/largest) if history and largest else 1.
                # A raw exploratory pose must never bypass the previous-frame
                # velocity box or become the next history value prematurely.
                q[ids] = start[ids] + fraction*delta
            per_finger.append(dict(triggered=bool(eligible[f]), selected=bool(selected[f]),
                transitioning=bool(selected[f] and fraction < 1.),
                step_fraction=fraction, local_cost=float(base[f, 3]),
                destination_cost=float(scores[f]) if selected[f] else None))

        output = native.score(q, positions, rotations, start, history) if selected.any() else base.copy()
        output[:, 2] = iterations
        if not np.isfinite(q).all() or not np.isfinite(output).all():
            raise RuntimeError('Nonfinite source-IK branch recovery result')
        self.targets = targets
        self.valid = valid
        self.transitioning = np.array([r['transitioning'] for r in per_finger])
        self.next_probe_frame = next_probe
        self.frame_index += 1
        return q, output, dict(revision=REVISION, probe_calls=probe_calls,
            calibration_seed_scores=calibration_scores,
            calibration_seed_sha256=self.calibration_sha256,
            probe_candidates=probe_calls*5, selected_fingers=int(selected.sum()),
            transitioning=any(r['transitioning'] for r in per_finger),
            frame_index=self.frame_index, fingers=per_finger)
