"""Compare virtual EMF receivers on Wuji2 FK against measured EMF (CPU only).

Transfer glove fingertip->RX extrinsics to corresponding robot fingertip frames.
The primary metric attaches these unchanged fixed transforms to robot tip links.
Zero-pose orientation bridging is an additional diagnostic, not the primary metric.
No trajectory-dependent alignment, offset fit, IK, or optimization is performed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from scripts.score_wuji_pose_comparison import load_reference_modules
from retarget.learning.wuji_ik.batched_lm import FastFingerURDF, so3_log
from retarget.learning.wuji_ik.features import wrist_tx_transform

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
TIPS = ("thumb_tip", "index_finger_tip", "middle_finger_tip", "ring_finger_tip", "pinky_tip")
METHODS = [("SDK", "q_sdk_filtered"), ("Network initial guess", "q_network"),
           ("Causal prefix 32", "source_causal_human_from_emf__causal_prefix__32"),
           ("Frozen parallel 32", "source_causal_human_from_emf__frozen_parallel__32"),
           ("Frozen history previous 10", "source_causal_human_from_emf__frozen_history_previous__10"),
           ("Frozen history network 10", "source_causal_human_from_emf__frozen_history_network__10"),
           ("Joint LM 32 FUTURE", None)]


def sensor_extrinsics(path):
    children = {j.find("child").get("link"): j for j in ET.parse(path).getroot().findall("joint")}
    rotations, positions = [], []
    for finger, tip in zip(FINGERS, TIPS):
        j = children[finger + "_rx_coil"]
        if j.get("type") != "fixed" or j.find("parent").get("link") != tip:
            raise ValueError("Expected fixed tip -> RX transform for " + finger)
        o = j.find("origin")
        positions.append(np.fromstring(o.get("xyz", "0 0 0"), sep=" "))
        rotations.append(Rotation.from_euler("xyz", np.fromstring(o.get("rpy", "0 0 0"), sep=" ")).as_matrix())
    return np.stack(positions), np.stack(rotations)


def aggregate(rows, **selectors):
    result = []
    for method, _ in METHODS:
        selected = [r for r in rows if r["method"] == method and all(r[k] == v for k, v in selectors.items())]
        n = sum(r["frames"] for r in selected)
        row = dict(method=method, frames=n, **selectors)
        for name in ("position_rmse_m", "orientation_rmse_rad", "zero_pose_bridged_position_rmse_m"):
            row[name] = float(np.sqrt(sum(r[name]**2*r["frames"] for r in selected)/n))
        row["position_max_m"] = max(r["position_max_m"] for r in selected)
        row["per_finger_position_rmse_m"] = np.sqrt(sum(np.square(r["per_finger_position_rmse_m"])*r["frames"] for r in selected)/n).tolist()
        result.append(row)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    out = root / "viser_viewer" / "robot_emf"
    out.mkdir(parents=True, exist_ok=True)
    Human, parameters, Robot, pre = load_reference_modules(root / "reference")
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float64)
    records = [r for r in json.loads((root / "selection.json").read_text())["records"] if r["split"] == "development"]
    rows, bridges, validation = [], {}, []
    for glove in ("builtin", "hl_urdf"):
        for side in ("left", "right"):
            path = root / "assets" / f"{glove}_{side}.urdf"
            human, robot = Human(side, path, sdk_tx_rounding=True), Robot("WujiHand2", side)
            tip_fk = FastFingerURDF(path, human.joint_names, TIPS)
            coil_fk = FastFingerURDF(path, human.joint_names, [f+"_rx_coil" for f in FINGERS])
            rfk = FastFingerURDF(robot.urdf_path, robot.joint_names, robot.landmark_names)
            tip_ids = [4, 8, 12, 16, 20]
            offset_p, offset_r = map(tensor, sensor_extrinsics(path))
            wrist_T_tx = tensor(wrist_tx_transform(path))
            seed = np.concatenate(parameters(side)["seed"])
            neutral = human.forward(seed)["skeleton"][:, :3, 3].astype(np.float32).astype(np.float64)
            operator = pre.OPERATOR2MANO_RIGHT if side == "right" else pre.OPERATOR2MANO_LEFT
            rz = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
            A = tensor((pre.estimate_frame(neutral) @ operator @ rz.T).T)
            human_zero_r = tip_fk.forward(torch.zeros(1, 21, dtype=torch.float64))[1][0]
            _, robot_zero_r, _, _ = rfk.forward(torch.zeros(1, 20, dtype=torch.float64))
            robot_wrist_r = robot_zero_r[0, 0]
            B = robot_zero_r[0, tip_ids].transpose(-1, -2) @ A @ human_zero_r
            assert torch.allclose(B.transpose(-1, -2) @ B, torch.eye(3, dtype=torch.float64).expand(5, -1, -1), atol=1e-12)
            bridges[f"{glove}_{side}"] = dict(urdf_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                robot_urdf_sha256=hashlib.sha256(robot.urdf_path.read_bytes()).hexdigest(),
                robot_base_R_glove_wrist=A.tolist(), robot_tip_R_glove_tip_zero_pose=B.tolist(),
                glove_tip_p_rx_m=offset_p.tolist(), glove_tip_R_rx=offset_r.tolist(), wrist_T_tx=wrist_T_tx.tolist())
            # Verify extracted sensor extrinsics against the full glove FK tree.
            random_q = tensor(np.random.default_rng(42).normal(0, .2, (4, 21)))
            tp, tr, _, _ = tip_fk.forward(random_q)
            cp, cr, _, _ = coil_fk.forward(random_q)
            pcheck = float((tp + (tr @ offset_p[..., None])[..., 0] - cp).abs().max())
            rcheck = float((tr @ offset_r - cr).abs().max())
            assert max(pcheck, rcheck) < 1e-12
            validation.append(dict(glove=glove, side=side, tip_to_rx_position_chain_max_abs_m=pcheck, tip_to_rx_rotation_chain_max_abs=rcheck))
            curves = np.load(root / "oneway_comparison" / f"{glove}_{side}" / "curves.npz", allow_pickle=False)
            for ri, rec in enumerate(records):
                idx = rec["index"]
                lo, hi = curves["record_offsets"][ri:ri+2]
                assert curves["record_indices"][ri] == idx
                data = np.load(root / "inputs" / f"trajectory_{idx}_{side}.npz", allow_pickle=False)
                joint = np.load(root / "jobs" / f"development_banded_lm_{glove}_{side}" / "budget_32" / f"trajectory_{idx}_{side}.npz", allow_pickle=False)
                assert np.array_equal(data["timestamp_ns"], joint["timestamp_ns"])
                assert np.array_equal(curves["q_sdk_filtered"][lo:hi], joint["q_sdk_filtered"])
                # Both observations and predictions are expressed in ROBOT wrist coordinates.
                wrist_p = (wrist_T_tx[:3, :3] @ tensor(data["position"])[..., None])[..., 0] + wrist_T_tx[:3, 3]
                target_p = (robot_wrist_r.T @ A @ wrist_p[..., None])[..., 0]
                measured_r = tensor(Rotation.from_quat(data["quaternion"].reshape(-1, 4)).as_matrix().reshape(-1, 5, 3, 3))
                target_r = robot_wrist_r.T @ A @ wrist_T_tx[:3, :3] @ measured_r
                predictions, errors, rotation_errors = [], [], []
                for name, key in METHODS:
                    q = tensor(joint["q_filtered"] if key is None else curves[key][lo:hi])
                    p, r, _, _ = rfk.forward(q)
                    tip_p, tip_r = p[:, tip_ids], r[:, tip_ids]
                    sensor_p_base = tip_p + (tip_r @ offset_p[..., None])[..., 0]
                    pred_p = (robot_wrist_r.T @ (sensor_p_base-p[:, :1])[..., None])[..., 0]
                    pred_r = robot_wrist_r.T @ tip_r @ offset_r
                    ep = (pred_p-target_p).norm(dim=-1)
                    er = so3_log(pred_r.transpose(-1, -2) @ target_r).norm(dim=-1)
                    # Diagnostic only: an alternative fixed zero-pose orientation bridge.
                    bridged_p = tip_p + (tip_r @ B @ offset_p[..., None])[..., 0]
                    bridged_p = (robot_wrist_r.T @ (bridged_p-p[:, :1])[..., None])[..., 0]
                    row = dict(record=idx, collection=rec["collection"], glove=glove, side=side,
                               method=name, frames=len(q), position_rmse_m=float(ep.square().mean().sqrt()),
                               position_max_m=float(ep.max()), orientation_rmse_rad=float(er.square().mean().sqrt()),
                               per_finger_position_rmse_m=ep.square().mean(0).sqrt().tolist(),
                               zero_pose_bridged_position_rmse_m=float((bridged_p-target_p).square().sum(-1).mean().sqrt()))
                    rows.append(row)
                    predictions.append(pred_p.numpy()); errors.append(ep.numpy()); rotation_errors.append(er.numpy())
                np.savez_compressed(out / f"{glove}_{side}_record_{idx}.npz", method_names=np.array([n for n, _ in METHODS]),
                    timestamp_ns=data["timestamp_ns"], segment_offsets=data["segment_offsets"], measured_emf_p_wrist=target_p.numpy(),
                    predicted_emf_p_wrist=np.stack(predictions), position_error_m=np.stack(errors), orientation_error_rad=np.stack(rotation_errors))
                print("SCORED ROBOT EMF", glove, side, idx, flush=True)
    report = dict(rows=rows, aggregate=aggregate(rows), by_configuration={g: aggregate(rows, glove=g) for g in ("builtin", "hl_urdf")},
        formula="p_hat = wrist_R_base * (FK_base_T_tip(q) * [glove_tip_p_rx; 1]); p_measured = wrist_R_base * A * glove_wrist_T_TX * p_TX",
        rmse_definition="sqrt(mean_over_frames_and_5_receivers(||p_hat - p_measured||_2^2)); no division by 3; units m",
        orientation_definition="SO(3) geodesic angle RMSE, rad; not included in position RMSE",
        mapping="Unchanged fixed glove tip->RX transforms attached to corresponding robot tip links; no added zero-pose rotation, translation/scale fit or framewise alignment; reference palm alignment reused",
        scope="Robot virtual EMF receiver metric, including SDK; virtual mounts are a transferred glove convention, not a calibrated physical robot sensor installation",
        zero_pose_bridged_definition="Diagnostic only: prepend a fixed rotation obtained by matching URDF zero-pose tip orientations; excluded from primary scores",
        output_q="SDK/solvers filtered q; network direct q", finger_order=FINGERS,
        zero_pose_bridges=bridges, validation=validation, solver_executions=0, neural_optimizer_updates=0,
        aggregate_scope="5 full development records x2 hands x2 glove configurations; frame weighted, configurations repeat observations")
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    with (out / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
