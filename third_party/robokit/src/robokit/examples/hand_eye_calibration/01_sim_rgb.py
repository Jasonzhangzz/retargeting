# pyright: reportMissingImports=false
"""Basic hand-eye calibration example: solve T_camera_base from robot masks."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import warp as wp

from robokit.assets import fetch
from robokit.helpers.hand_eye_calibration import HandEyeCalibrator, compute_robot_masks, presets
from robokit.robo import Robot


SAMPLES = 10  # solve on the first 10 samples; overlay the held-out rest


def main():
    wp.init()

    # --- data: baked ManiSkill StackCube-v1 samples (fixed camera) ---
    root = fetch(["calibration/**"])
    data = np.load(root / "calibration" / "maniskill_stackcube.npz")
    robot = Robot.load(str(root / str(data["urdf"])), load_meshes=True)
    masks, q, intrinsic, initial, gt = (
        torch.from_numpy(np.float32(data[k])).cuda()
        for k in ("masks", "q", "intrinsic", "initial_extrinsic", "gt_extrinsic")
    )
    H, W = masks.shape[1:]

    # --- solve ---
    hec = HandEyeCalibrator(presets.rgb, robot, camera_intrinsic=intrinsic, height=H, width=W)
    predicted = hec.solve(initial, target_masks=masks[:SAMPLES], q=q[:SAMPLES])
    print(f"Predicted camera extrinsic (OpenCV T_camera_base):\n{predicted.cpu().numpy()!r}")

    # --- overlay the first held-out sample under each extrinsic ---
    extrinsics = {"Initial": initial, "Predicted": predicted, "Ground Truth": gt}
    rendered = {
        label: compute_robot_masks(robot, q[SAMPLES : SAMPLES + 1], intrinsic, ext, H, W)
        for label, ext in extrinsics.items()
    }
    fig, axes = plt.subplots(1, 3, figsize=(21, 8))
    for ax, (label, m) in zip(axes, rendered.items()):
        overlay = data["rgb"][SAMPLES].copy()
        overlay[m[0].cpu().numpy() > 0] //= 4
        ax.imshow(overlay)
        ax.set_title(label)
        ax.axis("off")
    out_path = Path("runs") / "01_sim_rgb.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Overlay saved to {out_path}")


if __name__ == "__main__":
    main()
