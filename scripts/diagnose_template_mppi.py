#!/usr/bin/env python3
"""Independent fixed-anchor CPU IK controls for saved template MPPI results.

This diagnostic never changes templates, saved results or the production
solver. Failure of this local optimizer is NOT a proof of IK infeasibility.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import mujoco
from scipy.optimize import least_squares, minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retarget.common.results import write_json
from retarget.env.geometry import ObjectMeshSurface, closest_triangles
from retarget.env.load_support import skew


class FixedAnchorIK:
    """Exact MuJoCo FK/Jacobians and refreshed convex triangle projections."""
    def __init__(self, case, *, collisions=True, normals=True, expanded=False):
        self.case = Path(case)
        self.model = mujoco.MjModel.from_xml_path(str(self.case / "scene.xml"))
        self.data = mujoco.MjData(self.model)
        template = json.loads((self.case / "template_contacts.json").read_text())
        self.bodies = np.array([self.model.body(n).id for n in template["body_names"]])
        self.local = np.asarray(template["local_points"])
        self.normals = np.asarray(template["local_normals"])
        self.object_id = self.model.body("grasp_object").id
        self.surface = ObjectMeshSurface.from_model(self.model, self.object_id)
        self.collisions, self.use_normals = collisions, normals
        self.collision_thresholds = np.array([.00005, .00005])
        self.scale = np.r_[[.02] * 3, [.3] * 23]
        initialization = json.loads((self.case / "initialization.json").read_text())
        self.initial = np.asarray(initialization["initial_qpos"])
        joints = np.array([np.flatnonzero(self.model.jnt_qposadr == i)[0] for i in range(26)])
        self.low, self.high = self.model.jnt_range[joints].T.copy()
        radius = np.r_[[.25 if expanded else .1] * 3, [math.pi if expanded else 1.] * 3]
        self.low[:6] = np.maximum(self.low[:6], self.initial[:6] - radius)
        self.high[:6] = np.minimum(self.high[:6], self.initial[:6] + radius)
        self.low += 1.e-7
        self.high -= 1.e-7
        self.cached = None
        self.seen = []

    def query(self, points):
        nearest = closest_triangles(points, self.surface.triangles)
        distances = ((nearest - points[:, None]) ** 2).sum(-1)
        ids = distances.argmin(1)
        projected = nearest[np.arange(len(points)), ids]
        outward = self.surface.normals[ids]
        delta = points - projected
        length = np.linalg.norm(delta, axis=1)
        # Supported primitives are convex. This half-space test handles edge
        # projections without using one arbitrarily selected face for the sign.
        inside = np.all(np.einsum("ntd,td->nt", points[:, None] - self.surface.triangles[None, :, 0],
                                  self.surface.normals) <= 1.e-10, axis=1)
        sign = np.where(inside, -1., 1.)
        gradient = np.where(length[:, None] > 1.e-10,
            sign[:, None] * delta / np.maximum(length[:, None], 1.e-10), outward)
        return projected, outward, sign * length, gradient

    def state(self, q):
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        rotation = self.data.xmat[self.bodies].reshape(-1, 3, 3)
        points = self.data.xpos[self.bodies] + np.einsum("nij,nj->ni", rotation, self.local)
        normals = np.einsum("nij,nj->ni", rotation, self.normals)
        _, outward, phi, gradient = self.query(points)
        cosine = np.clip(-(normals * outward).sum(1), -1., 1.)
        angle = np.rad2deg(np.arccos(cosine))
        return points, normals, outward, phi, gradient, angle

    def collision(self, *, jacobian=True, thresholds=None):
        mujoco.mj_collision(self.model, self.data)
        depth, rows = np.zeros(2), np.zeros((2, 26))
        for contact in self.data.contact:
            bodies = self.model.geom_bodyid[contact.geom]
            if np.all(bodies == self.object_id):
                continue
            kind = int(self.object_id not in bodies)
            limits = self.collision_thresholds if thresholds is None else thresholds
            deficit = limits[kind] - contact.dist
            if deficit <= depth[kind]:
                continue
            depth[kind] = deficit
            if jacobian:
                first, second = np.zeros((3, 26)), np.zeros((3, 26))
                angular = np.zeros_like(first)
                mujoco.mj_jac(self.model, self.data, first, angular, contact.pos, int(bodies[0]))
                mujoco.mj_jac(self.model, self.data, second, angular, contact.pos, int(bodies[1]))
                rows[kind] = -contact.frame[:3] @ (second - first)
        return depth, rows

    def evaluate(self, q):
        if self.cached is not None and np.array_equal(q, self.cached[0]):
            return self.cached[1:]
        p, normal, outward, phi, gradient, angle = self.state(q)
        jp, jr = np.empty((len(p), 3, 26)), np.empty((len(p), 3, 26))
        for i, (body, point) in enumerate(zip(self.bodies, p)):
            mujoco.mj_jac(self.model, self.data, jp[i], jr[i], point, int(body))
        normal_scale = math.sqrt(2. * (1. - math.cos(math.radians(30.))))
        slot = (phi / .002) ** 2
        if self.use_normals:
            slot += ((normal + outward) ** 2).sum(1) / normal_scale ** 2
        k = math.ceil(.8 * len(p))
        selected = np.argsort(slot)[:k]
        weights = np.full(len(p), .05 / len(p))
        weights[selected] += 1. / k
        weight = np.sqrt(weights)
        residual = [weight * phi / .002]
        jacobian = [weight[:, None] * np.einsum("ni,nij->nj", gradient, jp) / .002]
        if self.use_normals:
            residual.append((weight[:, None] * (normal + outward) / normal_scale).ravel())
            # The selected mesh face normal is piecewise constant. Its discrete
            # identity is refreshed for every trial, not frozen across the solve.
            jacobian.append((-weight[:, None, None] * (skew(normal) @ jr) / normal_scale).reshape(-1, 26))
        if self.collisions:
            depth, collision_jac = self.collision()
            residual.append(100. * depth / .001)
            jacobian.append(100. * collision_jac / .001)
        r, j = np.concatenate(residual), np.concatenate(jacobian)
        self.cached = (q.copy(), r, j)
        return r, j

    def diagnostics(self, q):
        _, _, _, phi, _, angle = self.state(q)
        penetration, _ = self.collision(jacobian=False, thresholds=np.zeros(2))
        return dict(coverage=float(np.mean((abs(phi) <= .002) & (angle <= 30.))),
                    distance_coverage=float(np.mean(abs(phi) <= .002)),
                    normal_coverage=float(np.mean(angle <= 30.)),
                    median_gap_mm=float(np.median(abs(phi)) * 1000),
                    max_gap_mm=float(np.max(abs(phi)) * 1000),
                    normal_error_deg=angle.tolist(), signed_distance_m=phi.tolist(),
                    max_hand_object_penetration_m=float(penetration[0]),
                    max_hand_hand_penetration_m=float(penetration[1]),
                    collision_free=bool(penetration.max() <= 1.e-5))


def constrained_control(case, output, *, iterations=300, object_allowance_m=None, start=None):
    """Sequential QP with explicit near-collision inequalities, no large penalty.

    Positive collision margins reveal impending collisions. Final geometry is
    checked against the original model again; its physical shapes are unchanged.
    """
    problem = FixedAnchorIK(case, collisions=False)
    original_margin = problem.model.geom_margin.copy()
    problem.model.geom_margin[problem.model.geom_contype != 0] = .005
    if object_allowance_m is not None:
        # Self-contact is allowed, self-penetration is not. Only hand-object
        # pairs receive the requested penetration tolerance.
        problem.collision_thresholds = np.array([-object_allowance_m, 0.])
    q = np.load(Path(case) / "grasp.npz")["qpos"].copy() if start is None else np.asarray(start).copy()
    q = np.clip(q, problem.low, problem.high)
    history, trace = [], [q.copy()]
    trust, damping = .05, 1.e-3
    begun = time.perf_counter()
    for iteration in range(iterations):
        r, jac = problem.evaluate(q)
        cost = float(r @ r)
        # evaluate() may use its cache; refresh MuJoCo data for the constraint rows.
        problem.state(q)
        mujoco.mj_collision(problem.model, problem.data)
        distances, rows = [], []
        for contact in problem.data.contact:
            bodies = problem.model.geom_bodyid[contact.geom]
            if np.all(bodies == problem.object_id):
                continue
            first, second = np.zeros((3, 26)), np.zeros((3, 26))
            angular = np.zeros_like(first)
            mujoco.mj_jac(problem.model, problem.data, first, angular, contact.pos, int(bodies[0]))
            mujoco.mj_jac(problem.model, problem.data, second, angular, contact.pos, int(bodies[1]))
            distances.append(contact.dist)
            rows.append(contact.frame[:3] @ (second - first))
        j = jac * problem.scale
        hessian = j.T @ j + damping * np.eye(26)
        gradient = j.T @ r
        low = np.maximum((problem.low - q) / problem.scale, -trust)
        high = np.minimum((problem.high - q) / problem.scale, trust)
        constraints = []
        if rows:
            a = np.asarray(rows) * problem.scale
            kinds = np.array([int(problem.object_id not in problem.model.geom_bodyid[c.geom])
                              for c in problem.data.contact
                              if not np.all(problem.model.geom_bodyid[c.geom] == problem.object_id)])
            bound = problem.collision_thresholds[kinds] - np.asarray(distances)
            constraints.append(dict(type="ineq", fun=lambda x: a @ x - bound, jac=lambda x: a))
        result = minimize(lambda x: .5 * x @ hessian @ x + gradient @ x,
                          np.zeros(26), jac=lambda x: hessian @ x + gradient,
                          bounds=list(zip(low, high)), constraints=constraints,
                          method="SLSQP", options=dict(maxiter=100, ftol=1.e-10))
        accepted, alpha = False, 1.
        for _ in range(12):
            trial = np.clip(q + alpha * problem.scale * result.x, problem.low, problem.high)
            trial_r, _ = problem.evaluate(trial)
            depth, _ = problem.collision(jacobian=False)
            trial_cost = float(trial_r @ trial_r)
            if np.isfinite(trial_cost) and trial_cost < cost - 1.e-10 and depth.max() <= 1.e-6:
                q, accepted = trial, True
                trace.append(q.copy())
                break
            alpha *= .5
        diagnostic = problem.diagnostics(q)
        history.append(dict(iteration=iteration, cost=cost, qp_success=bool(result.success),
                            accepted=accepted, alpha=alpha, trust=trust, **diagnostic))
        if accepted:
            trust = min(.15, trust * (1.15 if alpha >= .5 else .7))
        else:
            trust *= .5
        if trust < 1.e-6:
            break
    problem.model.geom_margin[:] = original_margin
    diagnostic = problem.diagnostics(q)
    report = dict(final=diagnostic, iterations=len(history), seconds=time.perf_counter() - begun,
                  history=history, physical_shapes_unchanged=True, final_checked_with_original_margin=True,
                  object_allowance_m=object_allowance_m, self_penetration_allowed=False)
    destination = Path(output) / Path(case).parent.name
    destination.mkdir(parents=True, exist_ok=True)
    name = "constrained_control" if object_allowance_m is None else f"object_allowance_{object_allowance_m * 1000:g}mm"
    np.savez_compressed(destination / f"{name}.npz", qpos=np.asarray(trace))
    write_json(destination / f"{name}.json", report)
    return dict(case=str(case), **report)


def solve_case(case, output, *, starts=4, iterations=300):
    case, output = Path(case), Path(output)
    saved = np.load(case / "grasp.npz")["qpos"]
    source = json.loads((case / "summary.json").read_text())
    name = source.get("case_id", case.parent.name)
    target = output / name
    target.mkdir(parents=True, exist_ok=True)
    report = dict(case=str(case), template=source["template"], primitive=source["primitive"],
        original=source["geometry_diagnostics"], experiments={},
        scope="fixed original slots; no closure solve or point reselection; local IK failure does not prove infeasibility")
    for mode in ("contact_only", "collision_contact", "position_only", "expanded_collision_contact"):
        begun = time.perf_counter()
        problem = FixedAnchorIK(case, collisions=mode != "contact_only", normals=mode != "position_only",
                                expanded=mode == "expanded_collision_contact")
        rng = np.random.default_rng(20261102)
        initial = [saved.copy(), problem.initial.copy()]
        for _ in range(max(0, starts - 2)):
            noise = rng.normal(size=26) * np.r_[[.008] * 3, [.15] * 3, [.3] * 20]
            initial.append(saved + noise)
        if mode.startswith("expanded"):
            candidates = np.load(case / "initialization_candidates.npz")["qpos"]
            costs = np.asarray(json.loads((case / "initialization.json").read_text())["candidate_costs"])
            for index in np.argsort(costs):
                if all(np.linalg.norm(candidates[index, 3:6] - q[3:6]) > .7 for q in initial):
                    initial.append(candidates[index])
                if len(initial) >= starts + 3:
                    break
        trials, configurations = [], []
        for seed, value in enumerate(initial[:starts + (3 if mode.startswith("expanded") else 0)]):
            q = np.clip(value, problem.low + 1.e-8, problem.high - 1.e-8)
            before = float(np.sum(problem.evaluate(q)[0] ** 2))
            result = least_squares(lambda q: problem.evaluate(q)[0], q,
                jac=lambda q: problem.evaluate(q)[1], bounds=(problem.low, problem.high),
                x_scale=problem.scale, max_nfev=iterations, ftol=1.e-9, xtol=1.e-9, gtol=1.e-7)
            diagnostic = problem.diagnostics(result.x)
            trials.append(dict(seed=seed, initial_cost=before, cost=float(2. * result.cost),
                nfev=result.nfev, status=result.status, optimality=float(result.optimality), **diagnostic))
            configurations.append(result.x)
        # Report the objective optimum and best valid geometric witness separately.
        best = min(range(len(trials)), key=lambda i: trials[i]["cost"])
        witness = max(range(len(trials)), key=lambda i: (trials[i]["collision_free"], trials[i]["coverage"], -trials[i]["cost"]))
        np.savez_compressed(target / f"{mode}.npz", qpos=np.asarray(configurations),
                            best_qpos=configurations[best], witness_qpos=configurations[witness])
        report["experiments"][mode] = dict(best=best, witness=witness, trials=trials,
                                          seconds=time.perf_counter() - begun)
        write_json(target / "diagnosis.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--types", type=int, nargs="+", default=[1, 6, 10, 11, 14, 15, 17, 26, 28, 30, 33])
    parser.add_argument("--starts", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    entries = json.loads((args.batch / "manifest.json").read_text())["cases"]
    selected = []
    for typ in args.types:
        cases = [e for e in entries if int(e["template"].split("_")[0]) == typ]
        entry = min(cases, key=lambda e: abs(e["primitive"]["size_scale"] - 1.))
        selected.append((args.batch / entry["summary"]).parent.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(solve_case, case, args.output, starts=args.starts, iterations=args.iterations)
                   for case in selected]
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            print(report["template"], report["original"]["coverage"],
                  {mode: [(round(t["coverage"], 3), round(t["max_hand_object_penetration_m"] * 1000, 2))
                           for t in value["trials"]] for mode, value in report["experiments"].items()}, flush=True)
            write_json(args.output / "ik_experiments.json", reports)


if __name__ == "__main__":
    main()
