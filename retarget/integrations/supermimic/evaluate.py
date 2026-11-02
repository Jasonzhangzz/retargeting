"""Closed-loop six-direction hold protocol, separate from old static evaluation."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from .environment import GraspHoldEnv, make_env_cfg, physical_checks

PROTOCOL = "wuji2_policy_force_closure_hold_v1"


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return None if isinstance(value, float) and not np.isfinite(value) else value


def exact_closure(points, normals, friction, center, radius):
    """Use the pinned circular-cone solver and retain the ACTUAL solver status."""
    if len(points) < 3:
        return dict(success=False, reason="fewer_than_three_loaded_contacts")
    import clarabel
    from retarget.config.run import ROOT
    package = str(ROOT / "third_party/DexForge/contactaware")
    if package not in sys.path:
        sys.path.insert(0, package)
    from contactaware.contact.force_margin import solve_force_margin
    status = []
    class Solver:
        def __init__(self, *args):
            self.solver = clarabel.DefaultSolver(*args)
        def solve(self):
            result = self.solver.solve()
            status.append(str(result.status))
            return result
    try:
        result = solve_force_margin(points, normals, center=center, radius=radius, friction=friction,
                                    solver_factory=Solver, settings_factory=clarabel.DefaultSettings,
                                    equality_cone=clarabel.ZeroConeT, circular_cone=clarabel.SecondOrderConeT)
        actual = status[-1] if status else "NotRun"
        success = (actual == "Solved" and result.wrench_rank == 6 and result.reserve is not None
                   and result.reserve > 1e-4 and result.max_balance_error < 1e-6)
        return dict(success=bool(success), status=actual, reserve=result.reserve,
                    rank=result.wrench_rank, balance_error=result.max_balance_error)
    except Exception as error:
        return dict(success=False, reason="solver_error", error=repr(error))


@torch.no_grad()
def evaluate(env, actor, *, split, repeats, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    n, ds = env.num_envs, env.dataset
    case_ids = sorted(set(env.assignment.tolist()))
    replicas = n // len(case_ids)
    ranks = np.zeros(n, int)
    for case in case_ids:
        ids = np.flatnonzero(env.assignment == case)
        ranks[ids] = np.arange(len(ids))
    pools = ds.pools(split)
    outcomes = np.ones((len(ds.cases), repeats, 6), bool)
    seen = np.zeros_like(outcomes)
    directions = np.r_[np.eye(3), -np.eye(3)]
    force_scale = env.force_scale
    rng = copy.deepcopy(env.rng.bit_generator.state)
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(env.device)
    old_training = actor.training if actor is not None else False
    if actor is not None:
        actor.eval()
    env.force_scale = 1.
    trace_cases = set()
    # Pre-import before starting worker threads; all QPs use independent solvers.
    from retarget.config.run import ROOT
    sys.path.insert(0, str(ROOT / "third_party/DexForge/contactaware"))
    import contactaware.contact.force_margin  # noqa: F401
    details = (output / "directions.jsonl").open("w")
    try:
        with ThreadPoolExecutor(max_workers=8) as workers:
            for batch in range((repeats * 6 + replicas - 1) // replicas):
                trial = batch * replicas + ranks
                valid_world = trial < repeats * 6
                repeat = np.minimum(trial // 6, repeats - 1)
                direction = trial % 6
                refs = [pools[c][repeat[w] % len(pools[c])] for w, c in enumerate(env.assignment)]
                torch.manual_seed(81173 + batch)
                torch.cuda.manual_seed(81173 + batch)
                env.reset(split=split, reference_ids=refs, perturb=True,
                          directions=torch.as_tensor(directions[direction], device=env.device, dtype=torch.float32))
                alive = valid_world & env.reference_valid.cpu().numpy()
                reference_valid = env.reference_valid.cpu().numpy()
                reasons = [[] if reference_valid[w] else ["invalid_cached_reference"] for w in range(n)]
                worst_p = np.zeros(n)
                worst_r = np.zeros(n)
                min_load = np.full((n, 6), np.inf)
                max_forbidden = np.zeros((n, 6))
                min_reserve = np.full(n, np.inf)
                trace = []
                controls = []
                for step in range(110):
                    action = actor(env.observe()) if actor is not None else torch.zeros((n, 20), device=env.device)
                    tr = env.step(action, training=False)
                    trace.append(torch.cat((env.qpos[:, env.joints], env.qpos[:, env.object_q:env.object_q + 7]), -1).cpu().numpy())
                    controls.append(tr["u"].cpu().numpy())
                    valid = tr["valid"].cpu().numpy()
                    for w in np.flatnonzero(alive & ~valid):
                        reasons[w].append("invalid_or_contact_overflow")
                    alive &= valid
                    if step < 10:
                        continue
                    ctx = env.context()
                    checks = physical_checks(tr["x1"], ctx, tr["aux"])
                    for key, value in checks.items():
                        ok = value.cpu().numpy()
                        for w in np.flatnonzero(alive & ~ok):
                            reasons[w].append(key)
                        alive &= ok
                    aux = tr["aux"].cpu().numpy()
                    min_load = np.minimum(min_load, aux[:, :6])
                    max_forbidden = np.maximum(max_forbidden, aux[:, 6:12])
                    states = tr["x1"].cpu().numpy()
                    contexts = ctx.cpu().numpy()
                    displacement = np.linalg.norm(states[:, 40:43] - contexts[:, 197:200], axis=1)
                    worst_p = np.maximum(worst_p, np.nan_to_num(displacement, nan=np.inf, posinf=np.inf))
                    from scipy.spatial.transform import Rotation
                    rot = states[:, 43:52].reshape(-1, 3, 3)
                    target = contexts[:, 200:209].reshape(-1, 3, 3)
                    finite_rotation = np.isfinite(rot).all((1, 2)) & np.isfinite(target).all((1, 2))
                    matrices = target.transpose(0, 2, 1) @ rot
                    matrices[~finite_rotation] = np.eye(3)
                    angles = np.linalg.norm(Rotation.from_matrix(matrices).as_rotvec(), axis=1)
                    angles[~finite_rotation] = np.inf
                    worst_r = np.maximum(worst_r, angles)
                    active = np.flatnonzero(alive)
                    if len(active):
                        contacts = env.meter.snapshots()
                        def solve(w):
                            count = int(contacts["count"][w])
                            accepted = (contacts["accepted"][w, :count] > 0) & (contacts["pressures"][w, :count] > 1e-6)
                            mu = contacts["mus"][w, :count][accepted]
                            return exact_closure(contacts["points"][w, :count][accepted],
                                                 contacts["normals"][w, :count][accepted],
                                                 float(mu.min()) if len(mu) else 0.,
                                                 states[w, 40:43], float(env.radii[w]))
                        for w, result in zip(active, workers.map(solve, active)):
                            if not result["success"]:
                                alive[w] = False
                                reasons[w].append("force_closure:" + result.get("reason", result.get("status", "failed")))
                            elif result.get("reserve") is not None:
                                min_reserve[w] = min(min_reserve[w], result["reserve"])
                for w in np.flatnonzero(valid_world):
                    c, r, d = int(env.assignment[w]), int(repeat[w]), int(direction[w])
                    outcomes[c, r, d] = alive[w]
                    seen[c, r, d] = True
                    record = dict(case_id=ds.cases[c]["case_id"], repeat=r, direction=d,
                                  reference=ds.records[refs[w]], success=bool(alive[w]), reasons=reasons[w],
                                  max_position_m=float(worst_p[w]), max_rotation_deg=float(np.degrees(worst_r[w])),
                                  min_group_loads_n=min_load[w].tolist(), max_forbidden_loads_n=max_forbidden[w].tolist(),
                                  minimum_reserve=float(min_reserve[w]) if np.isfinite(min_reserve[w]) else None)
                    details.write(json.dumps(json_safe(record), allow_nan=False) + "\n")
                    tag = (c, bool(alive[w]))
                    if tag not in trace_cases:
                        trace_cases.add(tag)
                        np.savez_compressed(output / f"{ds.cases[c]['case_id']}_{'pass' if alive[w] else 'fail'}.npz",
                                            qpos=np.asarray(trace)[:, w], ctrl=np.asarray(controls)[:, w],
                                            case_index=c, reference_id=refs[w], direction=directions[d], dt=.02)
                details.flush()
                print(f"[{split}] directions {int(seen.sum())}/{len(case_ids) * repeats * 6}", flush=True)
        reports = []
        for c in case_ids:
            if not seen[c].all():
                raise RuntimeError("Incomplete evaluation denominator")
            success = outcomes[c].all(-1)
            reports.append(dict(case_id=ds.cases[c]["case_id"], type_id=ds.cases[c]["type_id"],
                                primitive=ds.cases[c]["primitive"], trials=repeats,
                                successes=int(success.sum()), success_rate=float(success.mean())))
        result = dict(protocol=PROTOCOL, split=split, policy="zero_residual" if actor is None else "shared_actor",
                      dataset_signature=ds.signature, cases=reports, total_cases=len(reports),
                      total_trials=len(reports) * repeats, total_directions=int(seen.sum()),
                      minimum_case_success_rate=min(r["success_rate"] for r in reports),
                      macro_success_rate=float(np.mean([r["success_rate"] for r in reports])),
                      all_cases_passed=all(r["successes"] == repeats for r in reports),
                      elapsed_seconds=time.monotonic() - started)
        (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        import csv
        with (output / "by_case.csv").open("w") as out:
            writer = csv.DictWriter(out, fieldnames=["case_id", "type_id", "trials", "successes", "success_rate"])
            writer.writeheader()
            writer.writerows([{k: r[k] for k in writer.fieldnames} for r in reports])
        return result
    finally:
        details.close()
        env.force_scale = force_scale
        env.rng.bit_generator.state = rng
        torch.set_rng_state(torch_rng)
        torch.cuda.set_rng_state(cuda_rng, env.device)
        if actor is not None:
            actor.train(old_training)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint")
    p.add_argument("--manifest", default=make_env_cfg()["manifest"])
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--num-envs", type=int, default=644)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--split", choices=("validation", "test"), default="test")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case-ids", type=int, nargs="+")
    p.add_argument("--baseline", action="store_true")
    args = p.parse_args(argv)
    env = GraspHoldEnv(make_env_cfg(args.num_envs, manifest=args.manifest, case_ids=args.case_ids), args.device)
    actor = None
    if not args.baseline:
        if not args.checkpoint:
            p.error("Require --checkpoint or --baseline")
        from dex_supermimic.rl.dex_supermimic.networks import Actor
        from .train import FEATURE_SCHEMA, FORMAT
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        if checkpoint["format_version"] != FORMAT or checkpoint["feature_schema"] != FEATURE_SCHEMA or checkpoint["dataset_signature"] != env.dataset.signature:
            raise ValueError("Checkpoint/cache feature mismatch")
        actor = Actor(checkpoint["obs_dim"]).to(args.device)
        actor.load_state_dict(checkpoint["modules"]["actor"])
    print(json.dumps(evaluate(env, actor, split=args.split, repeats=args.repeats, output=args.output), indent=2))


if __name__ == "__main__":
    main()
