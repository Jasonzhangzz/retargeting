"""CPU-only scoring of saved viewer trajectories; does not solve IK.

Robot FK vs SDK, robot FK vs EMF-derived retarget landmarks, and glove FK vs
measured EMF are separate metrics. No robot sensor extrinsic is fabricated.
"""
import argparse
import csv
import importlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch

from retarget.learning.wuji_ik.batched_lm import FastFingerURDF, prepare_robot_targets, so3_log


def load_reference_modules(reference):
    # Scoring needs FK only, not the package __init__ that imports NLopt.
    for name in ("human_ik_reference", "retarget_reference"):
        package = types.ModuleType(name)
        package.__path__ = [str(reference / name)]
        sys.modules[name] = package
    return (importlib.import_module("human_ik_reference.kinematics").HumanHandKinematics,
            importlib.import_module("human_ik_reference.parameters").default_parameters,
            importlib.import_module("retarget_reference.kinematics").HandKinematics,
            importlib.import_module("retarget_reference.preprocess"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    Human, parameters, Robot, pre = load_reference_modules(root / "reference")
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float64)
    records = [r for r in json.loads((root / "selection.json").read_text())["records"] if r["split"] == "development"]
    rows = []
    modes = [("SDK", "q_sdk_filtered"), ("Network initial guess", "q_network"),
             ("Causal prefix 32", "source_causal_human_from_emf__causal_prefix__32"),
             ("Frozen parallel 32", "source_causal_human_from_emf__frozen_parallel__32"),
             ("Frozen history previous 10", "source_causal_human_from_emf__frozen_history_previous__10"),
             ("Frozen history network 10", "source_causal_human_from_emf__frozen_history_network__10"),
             ("Joint LM 32 FUTURE", None)]
    for glove in ("builtin", "hl_urdf"):
        for side in ("left", "right"):
            curves = np.load(root / "oneway_comparison" / f"{glove}_{side}" / "curves.npz", allow_pickle=False)
            human = Human(side, root / "assets" / f"{glove}_{side}.urdf", sdk_tx_rounding=True)
            robot = Robot("WujiHand2", side)
            hfk = FastFingerURDF(human.urdf_path if hasattr(human, "urdf_path") else root / "assets" / f"{glove}_{side}.urdf", human.joint_names,
                                [f"{f}_rx_coil" for f in ("thumb", "index", "middle", "ring", "pinky")], root_transform=human._tx_inverse)
            sfk = FastFingerURDF(root / "assets" / f"{glove}_{side}.urdf", human.joint_names, human.skeleton_links)
            rfk = FastFingerURDF(robot.urdf_path, robot.joint_names, robot.landmark_names)
            seed = np.concatenate(parameters(side)["seed"])
            neutral = human.forward(seed)["skeleton"][:, :3, 3].astype(np.float32).astype(np.float64)
            operator = pre.OPERATOR2MANO_RIGHT if side == "right" else pre.OPERATOR2MANO_LEFT
            rz = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
            alignment = tensor(pre.estimate_frame(neutral) @ operator @ rz.T)
            for ri, rec in enumerate(records):
                idx = rec["index"]
                lo, hi = curves["record_offsets"][ri:ri+2]
                assert curves["record_indices"][ri] == idx
                data = np.load(root / "inputs" / f"trajectory_{idx}_{side}.npz", allow_pickle=False)
                sdk = np.load(root / "sdk" / glove / f"trajectory_{idx}_{side}.npz", allow_pickle=False)
                joint = np.load(root / "jobs" / f"development_banded_lm_{glove}_{side}" / "budget_32" / f"trajectory_{idx}_{side}.npz", allow_pickle=False)
                assert np.array_equal(sdk["q_sdk_filtered"], curves["q_sdk_filtered"][lo:hi])
                assert np.array_equal(joint["timestamp_ns"], data["timestamp_ns"])
                target_p = tensor(data["position"])
                from scipy.spatial.transform import Rotation
                target_r = tensor(Rotation.from_quat(data["quaternion"].reshape(-1, 4)).as_matrix().reshape(-1, 5, 3, 3))
                human_scores, robot_targets = {}, {}
                for hname, hq in (("sdk_causal", sdk["human_q"]), ("joint", joint["human_q"])):
                    p, r, _, _ = hfk.forward(tensor(hq))
                    ep = (p-target_p).norm(dim=-1)
                    er = so3_log(r.transpose(-1, -2) @ target_r).norm(dim=-1)
                    human_scores[hname] = dict(human_emf_position_rmse_m=float(ep.square().mean().sqrt()),
                                              human_emf_rotation_rmse_rad=float(er.square().mean().sqrt()))
                    robot_targets[hname] = prepare_robot_targets(sfk.forward(tensor(hq))[0], alignment)[0]
                sdk_p, sdk_r, _, _ = rfk.forward(tensor(sdk["q_sdk_filtered"]))
                tips = [4, 8, 12, 16, 20]
                for name, key in modes:
                    q = tensor(joint["q_filtered"] if key is None else curves[key][lo:hi])
                    p, r, _, _ = rfk.forward(q)
                    dp = (p-sdk_p).norm(dim=-1)
                    dr = so3_log(r[:, tips].transpose(-1, -2) @ sdk_r[:, tips]).norm(dim=-1)
                    target_kind = "joint" if key is None else "sdk_causal"
                    source = robot_targets[target_kind]
                    target_tip = (p[:, tips]-source[:, tips]).norm(dim=-1)
                    common_tip = (p[:, tips]-robot_targets["sdk_causal"][:, tips]).norm(dim=-1)
                    qe = q-tensor(sdk["q_sdk_filtered"])
                    row = dict(record=idx, collection=rec["collection"], side=side, glove=glove, method=name, frames=len(q),
                               q_sdk_rmse_rad=float(qe.square().mean().sqrt()),
                               robot_landmark_sdk_rmse_m=float(dp.square().mean().sqrt()),
                               robot_tip_orientation_sdk_rmse_rad=float(dr.square().mean().sqrt()),
                               robot_tip_common_emf_derived_target_rmse_m=float(common_tip.square().mean().sqrt()),
                               robot_tip_own_emf_derived_target_rmse_m=float(target_tip.square().mean().sqrt()),
                               **human_scores[target_kind])
                    if name == "Network initial guess":
                        row["human_emf_position_rmse_m"] = None
                        row["human_emf_rotation_rmse_rad"] = None
                    rows.append(row)
                print("SCORED", glove, side, idx, flush=True)
            # Cross-check independently recomputed joint human EMF aggregates.
            saved = json.loads((root / "jobs" / f"development_banded_lm_{glove}_{side}" / "budget_32" / "result.json").read_text())["human_emf_error"]
            matched = [r for r in rows if r["glove"] == glove and r["side"] == side and r["method"] == "Joint LM 32 FUTURE"]
            for metric, expected in (("human_emf_position_rmse_m", saved["position_rmse_m"]), ("human_emf_rotation_rmse_rad", saved["rotation_rmse_rad"])):
                actual = np.sqrt(sum(r[metric]**2*r["frames"] for r in matched)/sum(r["frames"] for r in matched))
                assert abs(actual-expected) < 1e-7, (metric, actual, expected)
    aggregate = []
    for name, _ in modes:
        matched = [r for r in rows if r["method"] == name]
        total = sum(r["frames"] for r in matched)
        agg = dict(method=name, frames=total)
        for k in matched[0]:
            if "rmse" in k:
                agg[k] = None if matched[0][k] is None else float(np.sqrt(sum(r[k]**2*r["frames"] for r in matched)/total))
        aggregate.append(agg)
    report = dict(rows=rows, aggregate=aggregate, definitions={
        "human_emf": "glove RX-coil FK vs measured TX-frame EMF, position Euclidean norm (m), rotation SO(3) geodesic angle (rad); not robot accuracy",
        "causal_human": "SDK cached human states used for scoring causal routes: separately verified source-vs-SDK human q RMSE <=1.7e-11 rad; network-only has no human IK",
        "robot_target": "five robot fingertip positions vs palm-aligned and pinch-processed EMF-derived human skeleton; morphology differs, this is NOT direct EMF sensor pose RMSE",
        "common_target": "every method scored against SDK causal human retarget targets, including joint LM which used its own joint human solve",
        "direct_robot_emf_rmse": "not computed in this report; see robot_emf/metrics.json for user-requested fixed glove tip->RX mount transfer to robot FK",
        "aggregate": "frame-weighted squared errors over 5 complete development records x2 hands x2 glove configs; configs are repeated observations, not independent new data"},
        scoring_device="cpu", solver_executions=0, neural_optimizer_updates=0,
        independent_joint_human_metric_crosscheck_passed=True)
    out = root / "viser_viewer"
    (out / "pose_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    with (out / "pose_metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
