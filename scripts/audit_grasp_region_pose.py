#!/usr/bin/env python3
"""Audit the type 6–9 constraints separately from native physical success."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from retarget.common.results import write_json
from retarget.env.grasp.contact_regions import orientation_error_deg, region_mask, region_preference
from retarget.env.grasp.graspadmm_seed_geometry import _input_digest


def audit(batch):
    batch = Path(batch).resolve()
    plan = json.loads((batch / "plan.json").read_text())
    rows = json.loads((batch / "summary.json").read_text())["jobs"]
    errors, checks, groups, types = [], Counter(), defaultdict(list), defaultdict(list)

    def check(value, label, case):
        checks[label] += 1
        if not value:
            errors.append(dict(case=case, check=label))

    check({r["id"] for r in rows} == set(plan["case_ids"]), "planned_coverage", "batch")
    max_angle = 0.
    for row in rows:
        cid = row["id"]
        source = Path(row["source_summary"]).parent
        meta = json.loads((source / "native_candidate.json").read_text())
        types[row["type_id"]].append(row)
        check(not row.get("error"), "no_runtime_exception", cid)
        initial = np.load(source / "initial_grasp.npy", allow_pickle=True).item()
        if row["type_id"] in (6, 7, 8):
            region = meta["contact_regions"]
            diagnostic = json.loads((source / "contact_region_diagnostics.json").read_text())
            check(diagnostic["initial_anchors"]["semantic_success"], "initial_anchors_in_region", cid)
            with np.load(source / "admm_trace.npz", allow_pickle=False) as trace:
                for slot, name in enumerate(initial["hand_cbody"]):
                    anchors = trace["hand_cpn_b"][:, slot]
                    if region.get("mode") == "soft":
                        check(bool(np.isfinite(region_preference(name, anchors[:, :3], anchors[:, 3:], region)).all()),
                              "finite_soft_cost_every_admm_anchor", cid)
                    else:
                        check(bool(region_mask(name, anchors[:, :3], anchors[:, 3:], region).all()),
                              "every_admm_anchor_in_region", cid)
            check("semantic_success" in row, "actual_contact_result_recorded", cid)
        elif row["type_id"] == 9:
            geometry = json.loads((source / "primitive_seed_geometry.json").read_text())
            reference = geometry["reference"]
            reference_path = Path(reference["path"])
            raw = np.load(reference_path, allow_pickle=True).item()
            check(hashlib.sha256(reference_path.read_bytes()).hexdigest() == reference["sha256"],
                  "reference_file_hash", cid)
            check(_input_digest(raw) == geometry["reference_input_sha256"], "reference_input_hash", cid)
            check(reference["role"] == "native_initial_seed_pose_only_never_optimized_output",
                  "initial_seed_not_final_pose", cid)
            check(reference["identity"]["primitive"]["size_scale"] == 1.5, "reference_size_1p5x", cid)
            check(geometry["rotation_axis_palm"] == [0., 1., 0.] and geometry["rotation_deg"] == 90.,
                  "palm_positive_90_degree_rotation", cid)
            check(np.array_equal(initial["grasp_qpos"][:, 7:], raw["grasp_qpos"][:, 7:]),
                  "reference_finger_angles_reused", cid)
            check(min(geometry["object_target_cap_clearance_m"]) > 0., "targets_on_cylinder_side", cid)
            reference_quat = meta["reference_orientation_wxyz"]
            with np.load(source / "simulation_trace.npz", allow_pickle=False) as trace:
                angle = max(orientation_error_deg(q, reference_quat) for q in trace["qpos"])
            max_angle = max(max_angle, angle, row.get("reference_orientation_error_deg", 0.))
            check(angle < 1e-4, "every_admm_simulation_record_orientation_locked", cid)
            groups[row["seed"]].append((geometry["reference_input_sha256"], initial["grasp_qpos"].copy()))

    for seed, values in groups.items():
        check(len(values) == 5, "five_sizes_per_reference_seed", f"seed{seed}")
        check(len({v[0] for v in values}) == 1, "same_reference_across_sizes", f"seed{seed}")
        check(all(np.array_equal(values[0][1], v[1]) for v in values),
              "same_initial_pose_across_sizes", f"seed{seed}")

    result = dict(passed=not errors, trials=len(rows), checks=dict(checks), errors=errors,
                  max_type9_orientation_error_deg=max_angle,
                  by_type=[dict(type_id=number, trials=len(values),
                                native_successes=sum(bool(v.get("task_success")) for v in values),
                                semantic_evaluated=sum("semantic_success" in v for v in values),
                                actual_center_contact_successes=sum(bool(v.get("semantic_success")) for v in values))
                           for number, values in sorted(types.items())],
                  interpretation="Constraint integrity and physical/actual-region outcomes are separate; failed grasps remain in the denominator")
    write_json(batch / "region_pose_audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, type=Path)
    result = audit(parser.parse_args().batch)
    print(json.dumps(result))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
