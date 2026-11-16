"""Stateful two-stage retarget reconstruction, independent of wuji_sdk."""
from enum import Enum
import threading
import numpy as np
import nlopt
from .kinematics import HandKinematics
from .preprocess import preprocess, prepare_targets
from .objectives import FrameObjective


class HandModel(Enum):
    WujiHand = 'WujiHand'
    WujiHand2 = 'WujiHand2'


class Handedness(Enum):
    Left = 'Left'
    Right = 'Right'


def _name(value):
    return str(getattr(value, 'name', value)).split('.')[-1]


def builtin_parameters(model):
    """Values extracted from the two YAML resources of the pinned SDK."""
    params = dict(lambda_warm=1., lambda_lap=150., lambda_pose=.1,
                  lambda_smooth=.01, lambda_antirev=20.,
                  first_frame_stage1_maxeval=50, later_frame_stage1_maxeval=1,
                  first_frame_stage2_maxeval=80, later_frame_stage2_maxeval=10,
                  nlopt_ftol_abs=1.e-4, edge_eps=1.e-8,
                  topo_distance_scale=50., thumb_free=True,
                  thumb_rel_full_rad=.35, robot_keypoint_backoff_m=0.)
    params.update(lambda_thumb_rel=5. if model == 'WujiHand' else .5,
                  thumb_rel_off_rad=.80 if model == 'WujiHand' else .50,
                  thumb_rel_skip_proximal=1 if model == 'WujiHand' else 2)
    return params


class RetargetSession:
    """21 MediaPipe points -> 20 float32 robot angles, with persistent state.

    Complete independent data path: preparation, FK/J, objective, two bounded
    SLSQP solves and float32 low-pass state. NLopt is the numerical backend.
    No SDK intermediate values are used. last_trace records stage diagnostics.
    """
    def __init__(self, hand_model, side):
        self.model = _name(hand_model)
        self.side = _name(side).lower()
        self.kinematics = HandKinematics(self.model, self.side)
        self.model = self.kinematics.model
        self.params = builtin_parameters(self.model)
        self.alpha = np.float32(.2)
        self._lock = threading.Lock()
        self._warm = None
        self._filtered = None
        self.last_trace = None

    @classmethod
    def for_hand(cls, hand_model, side):
        return cls(hand_model, side)

    def reset(self):
        with self._lock:
            self._warm = None
            self._filtered = None
            self.last_trace = None

    def _project(self, q):
        return np.clip(q, self.kinematics.lower, self.kinematics.upper)

    def _solve(self, objective, start, stage, maxeval):
        # Float32 warm storage may round a bound slightly outside its f64
        # interval. The native stage setup clamps its working vector again.
        start = self._project(start)
        opt = nlopt.opt(nlopt.LD_SLSQP, 20)
        opt.set_lower_bounds(self.kinematics.lower)
        opt.set_upper_bounds(self.kinematics.upper)
        opt.set_ftol_abs(self.params['nlopt_ftol_abs'])
        opt.set_maxeval(maxeval)
        evaluations = []
        callback_error = [None]

        def callback(q, grad):
            try:
                points, jacobians = self.kinematics.forward(q)
                cost, gradient, terms = objective.evaluate(q, points, jacobians, stage)
                if grad.size:
                    grad[:] = gradient
                evaluations.append(dict(cost=float(cost), q=q.tolist(), terms=terms))
                return cost
            except Exception as exc:
                callback_error[0] = exc
                raise

        opt.set_min_objective(callback)
        fallback = False
        try:
            result = opt.optimize(start.copy())
            status = int(opt.last_optimize_result())
        except Exception:
            # The native session discards best-so-far on all negative statuses
            # (-5..-1). Its jump tables restore this stage's starting vector.
            # Never hide an exception from our FK/objective implementation.
            if callback_error[0] is not None:
                raise callback_error[0]
            status = int(opt.last_optimize_result())
            if status >= 0:
                raise
            result = start.copy()
            fallback = True
        if status <= 0:
            result = start.copy()
            fallback = True
        if not np.isfinite(result).all():
            raise RuntimeError('SLSQP produced nonfinite angles')
        return result, dict(stage=stage, status=status, maxeval=maxeval, fallback=fallback,
                            evaluations=evaluations, output=result.tolist())

    def step(self, keypoints):
        with self._lock:
            source = preprocess(keypoints, self.model, self.side)
            targets = prepare_targets(source, thumb_free=self.params['thumb_free'],
                eps=self.params['edge_eps'], distance_scale=self.params['topo_distance_scale'],
                thumb_rel_skip_proximal=self.params['thumb_rel_skip_proximal'])
            first = self._warm is None
            initial = self._project(self.kinematics.bounds.mean(axis=1)) if first else self._warm.copy()
            reference = initial.copy()
            objective = FrameObjective(targets['S'], targets['edges'],
                directions=targets['directions'], laplacian=targets['L'],
                differential=targets['D'], weights=targets['row_weights'],
                thumb_triplets=targets['thumb_triplets'], params=self.params,
                q_ref=reference, antirev_indices=self.kinematics.antirev_indices)
            prefix = 'first_frame' if first else 'later_frame'
            q1, trace1 = self._solve(objective, initial, 1,
                self.params[prefix + '_stage1_maxeval'])
            q2, trace2 = self._solve(objective, q1, 2,
                self.params[prefix + '_stage2_maxeval'])
            raw = self._project(q2).astype(np.float32)
            # Preserve unfiltered warm state, rounded f32 then promoted to f64.
            warm = raw.astype(np.float64)
            if self._filtered is None:
                output = raw.copy()
            else:
                delta = np.subtract(raw, self._filtered, dtype=np.float32)
                weighted = np.multiply(self.alpha, delta, dtype=np.float32)
                output = np.add(self._filtered, weighted, dtype=np.float32)
            self._warm = warm
            self._filtered = output.copy()
            self.last_trace = dict(first=first, source=source.tolist(),
                initial=initial.tolist(), q_ref=reference.tolist(),
                stages=[trace1, trace2], unfiltered=raw.tolist())
            return output
