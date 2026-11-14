"""Score original stage-2 Wuji2 objective on screw-out right-hand trajectories.

All methods share SDK-human targets; history uses each method's previous raw q.
No IK/model runs. Exact source raw SDK states have verified identical SDK output.
"""
import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.score_wuji_pose_comparison import load_reference_modules
from scripts.score_wuji_robot_emf import METHODS
from retarget.learning.wuji_ik.batched_lm import FastFingerURDF, RobotRetargetObjective, prepare_robot_targets

TERMS = ("laplacian", "direction", "thumb_relative", "antireverse", "history", "total")
PARAMS = dict(lambda_warm=1., lambda_lap=150., lambda_pose=.1, lambda_smooth=.01,
              lambda_thumb_rel=.5, lambda_antirev=20., thumb_rel_full_rad=.35,
              thumb_rel_off_rad=.5, edge_eps=1e-8)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    out = root / "viser_viewer" / "objective_screw_right"
    out.mkdir(parents=True, exist_ok=True)
    Human, parameters, Robot, pre = load_reference_modules(root / "reference")
    FrameObjective = importlib.import_module("retarget_reference.objectives").FrameObjective
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float64)
    record = json.loads((root / "selection.json").read_text())["records"][0]
    assert record["index"] == 0 and "screw_out_the_screw" in record["collection"]
    report = dict(record=record["collection"], record_index=0, side="right", parameters=PARAMS, configurations={},
                  target_scope="SDK causal human skeleton common to every method; joint LM own-human targets are secondary only",
                  history_scope="previous raw q of the same method; reset uses original robot midpoint reference",
                  filtered_scope="evaluate filtered output q_t with the SAME previous raw q history as raw scoring; diagnostic, not optimization cost",
                  sdk_raw_scope="exact source q_source_raw from source_slsqp_stopping, whose filtered output was independently verified bitwise SDK-equal",
                  network_scope="network direct output; no hidden raw state or LM; same q scored as raw and output",
                  solver_executions=0, scoring_device="cpu")
    for glove in ("builtin", "hl_urdf"):
        dst = out / glove
        dst.mkdir(exist_ok=True)
        side = "right"
        path = root / "assets" / f"{glove}_{side}.urdf"
        human, robot = Human(side, path, sdk_tx_rounding=True), Robot("WujiHand2", side)
        sfk = FastFingerURDF(path, human.joint_names, human.skeleton_links)
        rfk = FastFingerURDF(robot.urdf_path, robot.joint_names, robot.landmark_names)
        neutral = human.forward(np.concatenate(parameters(side)["seed"]))["skeleton"][:, :3, 3].astype(np.float32).astype(np.float64)
        rz = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
        alignment = tensor(pre.estimate_frame(neutral) @ pre.OPERATOR2MANO_RIGHT @ rz.T)
        sdk = np.load(root / "sdk" / glove / "trajectory_0_right.npz", allow_pickle=False)
        joint = np.load(root / "jobs" / f"development_banded_lm_{glove}_right" / "budget_32" / "trajectory_0_right.npz", allow_pickle=False)
        curves = np.load(root / "oneway_comparison" / f"{glove}_right" / "curves.npz", allow_pickle=False)
        source = np.load(root / "oneway_comparison" / "source_slsqp_stopping" / f"{glove}_right.npz", allow_pickle=False)
        lo, hi = curves["record_offsets"][:2]
        assert lo == 0 and hi == len(sdk["q_sdk_filtered"])
        assert np.array_equal(curves["q_sdk_filtered"][lo:hi], sdk["q_sdk_filtered"])
        assert np.array_equal(source["q_source_filtered"][lo:hi], sdk["q_sdk_filtered"])
        n = int(hi-lo)
        offsets = joint["segment_offsets"]
        starts = offsets[:-1]
        raw, filtered = [], []
        for name, key in METHODS:
            if name == "SDK":
                raw.append(source["q_source_raw"][lo:hi]); filtered.append(sdk["q_sdk_filtered"])
            elif key is None:
                raw.append(joint["q_raw"]); filtered.append(joint["q_filtered"])
            else:
                raw.append(curves[key if name == "Network initial guess" else key+"__raw"][lo:hi])
                filtered.append(curves[key][lo:hi])
        raw, filtered = map(np.stack, (raw, filtered))
        target = prepare_robot_targets(sfk.forward(tensor(sdk["human_q"]))[0], alignment)
        spatial = RobotRetargetObjective(rfk, *target)
        own_spatial = RobotRetargetObjective(rfk, *prepare_robot_targets(sfk.forward(tensor(joint["human_q"]))[0], alignment))
        midpoint = robot.bounds.mean(1)
        previous = np.concatenate((raw[:, :1], raw[:, :-1]), axis=1)
        previous[:, starts] = midpoint
        sdk_previous = np.concatenate((raw[0, :1], raw[0, :-1]), axis=0)
        sdk_previous[starts] = midpoint
        slices = ((0, 63), (63, 111), (111, 113), (113, 123))
        arrays = {}
        for label, qs in (("raw", raw), ("filtered", filtered)):
            costs, fixed_history = [], []
            for i, q in enumerate(qs):
                residual = spatial(tensor(q))[0]
                parts = np.stack([(.5*residual[:, a:b].square().sum(1)).numpy() for a,b in slices], axis=1)
                hist = .005*np.square(q-previous[i]).sum(1)
                c = np.column_stack((parts, hist, parts.sum(1)+hist))
                costs.append(c)
                fixed_history.append(parts.sum(1)+.005*np.square(q-sdk_previous).sum(1))
            arrays["cost_"+label] = np.stack(costs)
            arrays["cost_"+label+"_fixed_sdk_history"] = np.stack(fixed_history)
        # Spot-check batched scores against the original scalar FrameObjective.
        errors = []
        check_frames = sorted(set([0, 1, int(np.argmax(arrays["cost_raw"][0, :, -1])), n//2, n-1]))
        for t in check_frames:
            ptarget, lap, diff, edges, directions = target
            obj = FrameObjective(ptarget[t].numpy(), edges.numpy(), directions=directions[t].numpy(),
                laplacian=lap[t].numpy(), differential=diff[t].numpy(), weights=np.r_[1.,0.,0.,0.,np.ones(17)],
                thumb_triplets=np.array([[2,3,4]]), params=PARAMS, q_ref=previous[0,t], antirev_indices=robot.antirev_indices)
            for i in (0, 1, 2, 6):
                obj.q_ref = previous[i,t]
                p, j = robot.forward(raw[i,t])
                value, _, terms = obj.evaluate(raw[i,t], p, j, 2)
                diffcost = abs(value-arrays["cost_raw"][i,t,-1])
                assert diffcost < 1e-8, (glove,t,i,value,arrays["cost_raw"][i,t,-1],terms)
                errors.append(diffcost)
        own_residual = own_spatial(tensor(raw[-1]))[0]
        arrays["joint_own_target_cost_raw"] = (.5*own_residual.square().sum(1)).numpy() + .005*np.square(raw[-1]-previous[-1]).sum(1)
        rows = []
        for i, (name, _) in enumerate(METHODS):
            row = dict(method=name, frames=int(n), raw_total=float(arrays["cost_raw"][i,:,-1].sum()),
                raw_mean=float(arrays["cost_raw"][i,:,-1].mean()), raw_max=float(arrays["cost_raw"][i,:,-1].max()),
                filtered_total=float(arrays["cost_filtered"][i,:,-1].sum()),
                fixed_sdk_history_raw_total=float(arrays["cost_raw_fixed_sdk_history"][i].sum()),
                raw_terms={k:float(arrays["cost_raw"][i,:,j].sum()) for j,k in enumerate(TERMS[:-1])},
                filtered_terms={k:float(arrays["cost_filtered"][i,:,j].sum()) for j,k in enumerate(TERMS[:-1])})
            row["raw_vs_sdk_percent"] = 100*(row["raw_total"]/arrays["cost_raw"][0,:,-1].sum()-1)
            rows.append(row)
        report["configurations"][glove] = dict(rows=rows, frames=int(n), continuous_segments=(offsets[1:]-offsets[:-1]).tolist(),
            scalar_reference_max_abs_error=max(errors), reference_check_frames=check_frames,
            joint_own_target_raw_total=float(arrays["joint_own_target_cost_raw"].sum()))
        np.savez_compressed(dst / "curves.npz", **arrays, timestamp_ns=joint["timestamp_ns"], segment_offsets=offsets,
                            method_names=np.array([n for n,_ in METHODS]), term_names=np.array(TERMS))
        print(glove, json.dumps(rows, indent=2), flush=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
