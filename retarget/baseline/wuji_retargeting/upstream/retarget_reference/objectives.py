"""Independent NumPy reconstruction of Wuji SDK 2026.8.31 retarget losses.

Evidence: retarget-math-supplement.txt and retarget-final-supplement.txt.
No SDK imports or native calls. FK positions and analytic Jacobians are inputs;
the optimizer and frame preprocessing are deliberately outside this module.
All objective arithmetic is float64, matching the native optimization core.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import numpy as np
from ._numeric import dot3, norm3, cross3, jacobian_transpose_vector, ordered_matmul


DEFAULTS = {
    "lambda_warm": 1.0,
    "lambda_lap": 150.0,
    "lambda_pose": 0.1,
    "lambda_smooth": 0.01,
    "lambda_thumb_rel": 5.0,
    "lambda_antirev": 20.0,
    "thumb_rel_full_rad": 0.35,
    "thumb_rel_off_rad": 0.80,
    "edge_eps": 1.0e-8,
}


def direction_loss(points, jacobians, edges, directions, weight, edge_eps):
    """Half weighted squared unit-edge error and its joint gradient."""
    n = jacobians.shape[2]
    if len(edges) == 0 or weight == 0:
        return 0.0, np.zeros(n)
    cost = 0.0
    gradient = np.zeros(n)
    half_weight = 0.5 * weight
    for (a, b), direction in zip(edges, directions):
        delta = points[b] - points[a]
        length = norm3(delta)
        if length <= edge_eps:
            continue
        unit = delta / length
        residual = unit - direction
        projected = (residual - unit * dot3(unit, residual)) / length
        edge_gradient = jacobian_transpose_vector(jacobians[b] - jacobians[a], projected)
        gradient += edge_gradient * weight
        cost += dot3(residual, residual) * half_weight
    return float(cost), gradient


def laplacian_loss(points, jacobians, laplacian, differential, weights, weight):
    """Row mask/weight occurs once in the quadratic, not squared."""
    if weight == 0:
        return 0.0, np.zeros(jacobians.shape[2])
    residual = ordered_matmul(laplacian, points) - differential
    weighted = residual * weights[:, None]
    squared = 0.0
    for i in range(len(points)):
        for c in range(3):
            squared += float(weighted[i, c]) * float(residual[i, c])
    cost = (weight * 0.5) * squared
    point_gradient = ordered_matmul(laplacian.T, weighted)
    gradient = np.zeros(jacobians.shape[2])
    for jacobian, point_grad in zip(jacobians, point_gradient):
        gradient += jacobian_transpose_vector(jacobian, point_grad)
    gradient *= weight
    return float(cost), gradient


def smooth_loss(q, q_ref, weight):
    delta = q - q_ref
    squared = 0.0
    for value in delta:
        squared += float(value) * float(value)
    return float((weight * 0.5) * squared), weight * delta


def antireverse_loss(q, indices, weight):
    """Penalize only negative values of constructor-selected joint indices."""
    gradient = np.zeros_like(q)
    cost = 0.0
    if weight <= 0:
        return cost, gradient
    half_weight = weight * 0.5
    for index in indices:
        value = max(-float(q[index]), 0.0)
        cost += (half_weight * value) * value
        gradient[index] = -weight * value
    return float(cost), gradient


def prepare_thumb_targets(source, triplets, weight, full_rad, off_rad, edge_eps):
    """Freeze source bend-plane, signed-sine target and source-angle gate.

    Exactly straight source triplets are skipped: their plane is undefined.
    The gate is source-only and therefore has no derivative with respect to q.
    """
    targets = []
    gate_enabled = 0.0 <= full_rad < off_rad
    if gate_enabled:
        off_cos = math.cos(off_rad)
        denom = math.cos(full_rad) - off_cos
    for a, b, c in triplets:
        u, v = source[b] - source[a], source[c] - source[b]
        lu, lv = norm3(u), norm3(v)
        if lu <= edge_eps or lv <= edge_eps:
            continue
        u, v = u / lu, v / lv
        cross = cross3(u, v)
        sine = norm3(cross)
        if sine <= edge_eps:
            continue
        cosine = dot3(u, v)
        effective = float(weight)
        if gate_enabled:
            # Embedded thresholds are in [0, pi], with a positive denominator.
            # Degenerate non-embedded thresholds receive explicit validation.
            if denom <= 0 or not np.isfinite(denom):
                raise ValueError("thumb gate needs cos(full) > cos(off)")
            t = float(np.clip((cosine - off_cos) / denom, 0.0, 1.0))
            effective = ((3.0 - (t + t)) * (t * t)) * effective
        if effective > 0:
            targets.append((int(a), int(b), int(c), float(cosine),
                            float(sine), cross / sine, effective))
    return tuple(targets)


def thumb_relative_loss(points, jacobians, targets, edge_eps):
    total = 0.0
    gradient = np.zeros(jacobians.shape[2])
    for a, b, c, cosine, sine, normal, weight in targets:
        du, dv = points[b] - points[a], points[c] - points[b]
        lu, lv = norm3(du), norm3(dv)
        if lu <= edge_eps or lv <= edge_eps:
            continue
        u, v = du / lu, dv / lv
        robot_cosine = dot3(u, v)
        rc = robot_cosine - cosine
        rs = dot3(normal, cross3(u, v)) - sine
        dcu = (v - u * robot_cosine) / lu
        dcv = (u - v * robot_cosine) / lv
        dsu_raw = cross3(v, normal)
        dsv_raw = cross3(normal, u)
        dsu = (dsu_raw - u * dot3(u, dsu_raw)) / lu
        dsv = (dsv_raw - v * dot3(v, dsv_raw)) / lv
        # The native helper differentiates at each of the three points before
        # Jacobian multiplication. Distributing J differences changes rounding.
        ga = dcu * (-rc) + dsu * (-rs)
        gb = (dcu - dcv) * rc + (dsu - dsv) * rs
        gc = dcv * rc + dsv * rs
        triplet_gradient = jacobian_transpose_vector(jacobians[a], ga)
        triplet_gradient += jacobian_transpose_vector(jacobians[b], gb)
        triplet_gradient += jacobian_transpose_vector(jacobians[c], gc)
        gradient += triplet_gradient * weight
        total += (weight * 0.5) * (rc * rc + rs * rs)
    return float(total), gradient


class FrameObjective:
    """Objective fixed for one frame, evaluated at successive trial angles.

    Parameters
    ----------
    source : (N,3), already aligned and pinch-adjusted
    edges : (E,2), constructor-selected source edges
    directions : optional (E,3), source unit directions
    laplacian : (N,N), prepared source graph matrix
    differential : optional (N,3), defaults to laplacian @ source
    weights : optional (N,), defaults to ones
    thumb_triplets : source/robot point index triples
    params : topo parameter mapping, or mapping containing a 'topo' mapping
    q_ref : previous *unfiltered*, float32-rounded solution promoted to float64
    antirev_indices : constructor-selected indices, NOT all joint indices

    evaluate returns (cost, gradient, term_costs). Stage 1 uses lambda_warm;
    stage 2 uses lambda_pose and additionally enables the Laplacian loss.
    The caller owns q initialization, reference selection and state transitions.
    """

    def __init__(self, source, edges, *, directions=None, laplacian=None,
                 differential=None, weights=None, thumb_triplets=(),
                 params=None, q_ref=None, antirev_indices=()):
        self.params = dict(DEFAULTS)
        if params is not None:
            if not isinstance(params, Mapping):
                raise TypeError("params must be a mapping")
            self.params.update(params.get("topo", params))
        self.source = np.array(source, dtype=np.float64, copy=True)
        if self.source.ndim != 2 or self.source.shape[1] != 3:
            raise ValueError("source must have shape (N,3)")
        if not np.all(np.isfinite(self.source)):
            raise ValueError("source must be finite")
        count = len(self.source)
        self.edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2).copy()
        if np.any(self.edges < 0) or np.any(self.edges >= count):
            raise ValueError("edge index outside source")
        self.edge_eps = float(self.params["edge_eps"])
        if directions is None:
            delta = self.source[self.edges[:, 1]] - self.source[self.edges[:, 0]]
            norms = np.array([norm3(vector) for vector in delta])
            keep = norms > self.edge_eps
            self.edges = self.edges[keep]
            self.directions = delta[keep] / norms[keep, None]
        else:
            self.directions = np.array(directions, dtype=np.float64, copy=True)
            if self.directions.shape != (len(self.edges), 3):
                raise ValueError("directions must have shape (len(edges),3)")
        self.laplacian = None if laplacian is None else np.array(laplacian, dtype=np.float64, copy=True)
        self.differential = None
        if self.laplacian is not None:
            if self.laplacian.shape != (count, count):
                raise ValueError("laplacian must have shape (N,N)")
            self.differential = (ordered_matmul(self.laplacian, self.source) if differential is None
                                 else np.array(differential, dtype=np.float64, copy=True))
            if self.differential.shape != self.source.shape:
                raise ValueError("differential must have shape (N,3)")
        self.weights = np.ones(count) if weights is None else np.array(weights, dtype=np.float64, copy=True)
        if self.weights.shape != (count,):
            raise ValueError("weights must have shape (N,)")
        triplets = np.asarray(thumb_triplets, dtype=np.int64).reshape(-1, 3)
        if np.any(triplets < 0) or np.any(triplets >= count):
            raise ValueError("thumb triplet index outside source")
        self.thumb_targets = prepare_thumb_targets(
            self.source, triplets, self.params["lambda_thumb_rel"],
            self.params["thumb_rel_full_rad"], self.params["thumb_rel_off_rad"],
            self.edge_eps)
        self.q_ref = None if q_ref is None else np.array(q_ref, dtype=np.float64, copy=True)
        self.antirev_indices = np.asarray(antirev_indices, dtype=np.int64).reshape(-1).copy()

    def evaluate(self, q, points, jacobians, stage=2):
        q = np.asarray(q, dtype=np.float64)
        points = np.asarray(points, dtype=np.float64)
        jacobians = np.asarray(jacobians, dtype=np.float64)
        if q.ndim != 1 or points.shape != self.source.shape:
            raise ValueError("expected q (nq,) and points matching source (N,3)")
        if jacobians.shape != (len(points), 3, len(q)):
            raise ValueError("jacobians must have shape (N,3,nq)")
        if stage not in (1, 2):
            raise ValueError("stage must be 1 or 2")
        if np.any(self.antirev_indices < 0) or np.any(self.antirev_indices >= len(q)):
            raise ValueError("antirev index outside q")
        if self.q_ref is not None and self.q_ref.shape != q.shape:
            raise ValueError("q_ref shape must match q")
        params = self.params
        gradient = np.zeros_like(q)
        terms = {}

        def add(name, result):
            cost, term_gradient = result
            terms[name] = float(cost)
            gradient[:] += term_gradient

        if stage == 2:
            if self.laplacian is None:
                if params["lambda_lap"] != 0:
                    raise ValueError("stage 2 requires laplacian or lambda_lap=0")
                terms["laplacian"] = 0.0
            else:
                add("laplacian", laplacian_loss(
                    points, jacobians, self.laplacian, self.differential,
                    self.weights, params["lambda_lap"]))
        add("direction", direction_loss(
            points, jacobians, self.edges, self.directions,
            params["lambda_warm" if stage == 1 else "lambda_pose"], self.edge_eps))
        if self.q_ref is None:
            if params["lambda_smooth"] != 0:
                raise ValueError("q_ref required when lambda_smooth is nonzero")
            terms["smooth"] = 0.0
        else:
            add("smooth", smooth_loss(q, self.q_ref, params["lambda_smooth"]))
        add("thumb_relative", thumb_relative_loss(
            points, jacobians, self.thumb_targets, self.edge_eps))
        add("antireverse", antireverse_loss(q, self.antirev_indices, params["lambda_antirev"]))
        return float(sum(terms.values())), gradient, terms
