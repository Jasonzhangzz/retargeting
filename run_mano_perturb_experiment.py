#!/usr/bin/env python3
"""Generate MANO finger-perturbed DexYCB sequences and run ContactAware retarget.

The source MANO trajectory is PCA encoded. Each variant is perturbed in decoded
15-joint axis-angle space, then re-encoded before the original MANO->Wuji
retargeter is called. This script only prepares sequences; the retarget runner
is kept separate so the same variants can be passed to different optimizers.
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

FINGER_POSE = {
    "index": (0, 3), "middle": (3, 6), "pinky": (6, 9),
    "ring": (9, 12), "thumb": (12, 15),
}
# First contact in the matching 72-frame sugar-box demonstration.
CONTACT_FRAMES = {"index": 21, "middle": 22, "ring": 21, "thumb": 21, "pinky": 21}


def make_variants(source: Path, out: Path, amplitude: float) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    raw = source / "mano_raw"
    pose = np.load(raw / "hand_pose_51.npy").astype(np.float64)
    pose = pose.reshape(len(pose), -1)
    if pose.shape[1] != 51:
        raise ValueError(f"unexpected MANO pose shape {pose.shape}")
    # Decode the PCA hand pose using the same MANO model stored with DexForge.
    model_path = Path(os.environ.get("MANO_MODEL_ROOT", "/home/zz/GraspSONIC/thirdparty/DexForge/assets/mano")) / "MANO_RIGHT.pkl"
    import pickle, inspect
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    for name, value in {"bool": bool, "int": int, "float": float, "complex": complex, "object": object, "str": str, "unicode": str}.items():
        if not hasattr(np, name): setattr(np, name, value)
    with model_path.open("rb") as f:
        model = pickle.load(f, encoding="latin1")
    comp = np.asarray(model["hands_components"].r if hasattr(model["hands_components"], "r") else model["hands_components"], float)
    mean = np.asarray(model["hands_mean"].r if hasattr(model["hands_mean"], "r") else model["hands_mean"], float)
    if comp.shape != (45, 45):
        raise ValueError(f"unexpected MANO PCA basis {comp.shape}")
    pinv = np.linalg.pinv(comp)
    decoded = pose[:, 3:48] @ comp + mean
    report = {"source": str(source), "amplitude_rad": amplitude, "variants": {}}
    for finger, (j0, j1) in FINGER_POSE.items():
        case = out / finger
        if case.exists():
            raise FileExistsError(case)
        shutil.copytree(source, case)
        pert = decoded.copy()
        frame = CONTACT_FRAMES[finger]
        # Bend each of the three joints around its local y axis. The sign is
        # chosen from the current pose so the perturbation initially increases
        # the existing bend magnitude rather than undoing it.
        for joint in range(j0, j1):
            sl = 3 * joint + 1
            sign = 1.0 if decoded[frame, sl] >= 0.0 else -1.0
            pert[frame:, sl] += sign * amplitude
        encoded = (pert - mean) @ pinv
        new_pose = pose.copy()
        new_pose[:, 3:48] = encoded
        np.save(case / "mano_raw/hand_pose_51.npy", new_pose.astype(np.float32))
        meta = {
            "finger": finger, "contact_frame": frame, "axis": "local_y",
            "decoded_joint_range": [j0 + 1, j1], "amplitude_rad": amplitude,
            "perturbation_policy": "from_first_contact_frame_through_sequence_end",
        }
        (case / "mano_perturbation.json").write_text(json.dumps(meta, indent=2) + "\n")
        report["variants"][finger] = meta
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--amplitude", type=float, default=0.25)
    a = p.parse_args()
    report = make_variants(a.source.resolve(), a.output_root.resolve(), a.amplitude)
    (a.output_root / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
