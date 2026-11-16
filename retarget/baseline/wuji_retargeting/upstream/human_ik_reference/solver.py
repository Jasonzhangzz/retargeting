"""Independent EMF receiver-pose to human-hand IK reconstruction.

The algorithm/parameters follow the pinned SDK's observed CQQ offline path.
No wuji_sdk import, SDK angle seed, or per-frame SDK intermediate is used.
Input: five TX-frame receiver positions (metres) and XYZW quaternions.
"""
import copy
import math
import threading
import numpy as np
from .kinematics import HumanHandKinematics
from .sdk_rotation import sdk_pose_residual_jacobian as pose_residual_jacobian
from .parameters import default_parameters
from .boxqp import solve_box_qp


def quaternion_matrix_xyzw(value):
    q = np.asarray(value, dtype=float)
    if q.shape != (4,): raise ValueError('quaternion must have four components')
    x,y,z,w = [float(v) for v in q]
    norm = math.sqrt(0. + ((w*w+y*y)+(z*z+x*x)))
    if q.shape != (4,) or not math.isfinite(norm) or norm == 0:
        raise ValueError('quaternion must contain four finite nonzero components')
    x,y,z,w = x/norm,y/norm,z/norm,w/norm
    xx,yy,zz,ww = x*x,y*y,z*z,w*w
    xy,xz,yz,xw,yw,zw = x*y,x*z,y*z,x*w,y*w,z*w
    xy,xz,yz,xw,yw,zw = xy+xy,xz+xz,yz+yz,xw+xw,yw+yw,zw+zw
    return np.array([[((xx+ww)-yy)-zz, xy-zw, xz+yw],
                     [zw+xy, (yy+(ww-xx))-zz, yz-xw],
                     [xz-yw, xw+yz, ((ww-xx)-yy)+zz]])


class HumanIKSession:
    """Stateful five-finger constrained IK, returning 21 human angles in radians.

    Explicit CQQ URDFs and finite EMF poses are the validated scope. reset()
    restores constructor histories. last_trace exposes pose errors, retries,
    candidate evaluations and QP diagnostics for inspection.
    """
    def __init__(self, side='right', urdf_path=None, parameters=None):
        # Native constructor reroots the URDF at TX and serializes the inverse
        # TX joint's XYZ/RPY with ten decimals before loading the IK model.
        self.kinematics = HumanHandKinematics(side, urdf_path, sdk_tx_rounding=True)
        self.side = self.kinematics.side
        self.parameters = default_parameters(self.side)
        # Native physiological IK limits are independent of the CQQ URDF's
        # broad +/-pi limits; a URDF override must not replace these bounds.
        if parameters is not None:
            self.parameters.update(copy.deepcopy(parameters))
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self._history = [np.array(q, dtype=float) for q in self.parameters['seed']]
            self._history2 = [np.array(q, dtype=float) for q in self.parameters['history2']]
            self.last_trace = None

    def _evaluate(self, finger, q, target):
        pose, jp, jw = self.kinematics.finger_pose_jacobian(finger, q)
        residual, jac = pose_residual_jacobian(pose, target, jp, jw)
        scale = 1000. * self.parameters['error_to_m']
        residual[:3] *= scale
        jac[:3] *= scale
        weights = np.array([self.parameters['w_pos']]*3 + [self.parameters['w_rot']]*3)
        weighted = weights * residual
        cost = float(sum(float(x)*float(x) for x in weighted))
        position_error = float(np.linalg.norm(target[:3,3]-pose[:3,3])) * self.parameters['error_to_m']
        angle_error = float(np.linalg.norm(residual[3:]))
        return cost, weighted, weights[:,None]*jac, position_error, angle_error

    def _core(self, finger, target, initial, history2, base, budget, verbose):
        p = self.parameters
        q = initial.copy()
        reference = np.asarray(p['reference'][finger], dtype=float)
        bounds = np.asarray(p['bounds'][finger], dtype=float)
        n = q.size
        count = 0
        steps = []
        current = self._evaluate(finger, q, target)
        reason = 'budget'
        while count < budget:
            cost, weighted, jac, position_error, angle_error = current
            if not math.isfinite(cost):
                reason = 'nonfinite'; break
            if position_error < 1e-8 and angle_error < 1e-6:
                reason = 'pose_tolerance'; break
            drive = -p['error_gain'] * weighted
            damping = p['damping_scale'] * sum(float(x)*float(x) for x in drive) + p['damping_floor']
            # Scalar accumulation keeps the native six-row arithmetic order.
            hessian = np.empty((n,n)); gradient = np.empty(n)
            for j in range(n):
                gradient[j] = -sum(float(jac[k,j])*float(drive[k]) for k in range(6))
                for i in range(n):
                    hessian[i,j] = sum(float(jac[k,i])*float(jac[k,j]) for k in range(6))
                hessian[j,j] += damping
            posture = p['posture_weight'] * (p['thumb_posture_scale'] if finger == 0 else 1.)
            if p['posture_weight'] > 0 and posture > 0:
                for j in range(n):
                    if finger != 0 and j == 3: continue
                    hessian[j,j] += posture
                    gradient[j] += posture * (q[j] - (reference[j] if j < reference.size else 0.))
            if p['history_weight'] > 0:
                for j in range(n):
                    hessian[j,j] += p['history_weight']
                    gradient[j] += p['history_weight'] * ((q[j] - (initial[j] + initial[j])) + history2[j])
            cap = base * p['max_step_scale']
            lower = np.maximum(p['bound_scale']*(bounds[:,0]-q), -cap)
            upper = np.minimum(p['bound_scale']*(bounds[:,1]-q), cap)
            delta = solve_box_qp(hessian, gradient, lower, upper)
            threshold = cost + 1e-12*max(1.,abs(cost))
            accepted = False
            trials = []
            for alpha in (1., .5, .25, .125):
                if count >= budget: break
                candidate = q + alpha*delta
                evaluation = self._evaluate(finger, candidate, target)
                count += 1
                accepted = math.isfinite(evaluation[0]) and evaluation[0] <= threshold
                if verbose: trials.append(dict(alpha=alpha, cost=evaluation[0], accepted=accepted))
                if accepted:
                    q = candidate
                    current = evaluation
                    break
            if verbose:
                steps.append(dict(cost=cost, damping=damping, hessian=hessian.tolist(),
                    gradient=gradient.tolist(), lower=lower.tolist(), upper=upper.tolist(),
                    delta=delta.tolist(), trials=trials))
            if not accepted:
                reason = 'line_search'; break
        return q, dict(position_error_m=current[3], angle_error_rad=current[4],
            cost=current[0], evaluations=count, stop=reason, steps=steps)

    def _finger(self, finger, target, verbose):
        initial = self._history[finger].copy()
        q, trace = self._core(finger, target, initial, self._history2[finger],
                              self.parameters['retry_base'], 40, verbose)
        attempts = [trace]; selected = 0
        if trace['position_error_m'] > .005:
            for multiplier in (2.,4.):
                candidate, attempt = self._core(finger, target, initial, initial,
                    multiplier*self.parameters['retry_base'], 36, verbose)
                attempts.append(attempt)
                if attempt['position_error_m'] < trace['position_error_m']:
                    q, trace, selected = candidate, attempt, len(attempts)-1
                if trace['position_error_m'] <= .005: break
        self._history[finger] = q.copy()
        self._history2[finger] = q.copy()
        return q, dict(selected=selected, attempts=attempts,
            position_error_m=trace['position_error_m'], angle_error_rad=trace['angle_error_rad'])

    def step(self, positions, orientations_xyzw, confidence=None, *, trace=False):
        """Consume one new frame; returns q, transforms and confidence arrays.

        Values first pass through float32, matching the native EMF record.
        Every call advances the solver. The OfflinePipeline adapter supplies
        the frame cache shared by angles, tips and skeleton compute calls.
        """
        with self._lock:
            positions = np.asarray(positions, dtype=np.float32).astype(float)
            orientations = np.asarray(orientations_xyzw, dtype=np.float32).astype(float)
            sensor = np.ones(5) if confidence is None else np.asarray(confidence,dtype=np.float32).astype(float)
            if positions.shape != (5,3) or orientations.shape != (5,4) or sensor.shape != (5,):
                raise ValueError('expected positions (5,3), XYZW (5,4), confidence (5,)')
            if not np.isfinite(positions).all() or not np.isfinite(orientations).all() or not np.isfinite(sensor).all():
                raise ValueError('EMF inputs must be finite')
            targets = np.tile(np.eye(4), (5,1,1))
            targets[:,:3,3] = positions * self.parameters['target_scale']
            for f in range(5): targets[f,:3,:3] = quaternion_matrix_xyzw(orientations[f])
            fingers=[];traces=[]
            for f in range(5):
                q, detail = self._finger(f, targets[f], trace)
                fingers.append(q); traces.append(detail)
            q21 = np.concatenate(fingers)
            transforms = self.kinematics.forward(q21)
            errors = np.array([t['position_error_m'] for t in traces])
            ik_confidence = np.maximum(0.,1.-np.minimum(errors/.02,1.))
            combined = (sensor + ik_confidence)*.5
            padded = np.zeros((5,5))
            for f,q in enumerate(fingers): padded[f,:len(q)] = q
            self.last_trace = dict(fingers=traces, q=q21.tolist())
            return dict(q=q21, angles=padded, confidence=combined,
                ik_confidence=ik_confidence, position_error_m=errors,
                angle_error_rad=np.array([t['angle_error_rad'] for t in traces]),
                skeleton_confidence=np.r_[np.mean(combined),np.repeat(combined,4)],
                **transforms)
