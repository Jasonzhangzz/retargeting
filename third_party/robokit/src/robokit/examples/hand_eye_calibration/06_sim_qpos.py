# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "robokit",
#   "torch==2.11.0",
#   "nvdiffrast==0.3.3+pt2.11.0cu126",
#   "warp-lang==1.10.1",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
#
# [[tool.uv.index]]
# name = "torch-packages-builder"
# url = "https://jianglongye.github.io/torch_packages_builder"
# explicit = true
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# torch = { index = "pytorch-cu126" }
# nvdiffrast = { index = "torch-packages-builder" }
# ///
"""Compare independent frame fits with a coupled trajectory from calibrated silhouettes; saves source/per-frame/trajectory masks and metrics under runs/hand_eye_calibration/.

Run with CUDA: uv run src/robokit/examples/hand_eye_calibration/06_sim_qpos.py
"""

import json
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import torch

from robokit.assets.robots.arms import xarm7
from robokit.helpers.hand_eye_calibration import QposTrajectoryCalibrator, compute_robot_masks, presets
from robokit.robo import Robot


def main():
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- data: two cameras observing an unknown joint trajectory ---
    robot = Robot.load(str(xarm7.URDF_PATH), load_meshes=True)
    height = width = 128
    intrinsic = torch.tensor([[115.0, 0.0, 64.0], [0.0, 115.0, 64.0], [0.0, 0.0, 1.0]], device="cuda")
    cameras = []
    for eye in [np.array([1.1, -0.9, 0.9]), np.array([1.1, 0.9, 0.9])]:
        forward = np.array([0.0, 0.0, 0.35]) - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        T_base_camera = np.eye(4, dtype=np.float32)
        T_base_camera[:3, :3] = np.stack([right, np.cross(forward, right), forward], axis=1)
        T_base_camera[:3, 3] = eye
        cameras.append(np.linalg.inv(T_base_camera))
    cameras = torch.tensor(np.stack(cameras), device="cuda")
    times = torch.linspace(0, 2, 9, device="cuda")
    truth = torch.tensor([0.1, -0.5, 0.15, 1.0, -0.1, 0.6, 0.2], device="cuda")[None].repeat(len(times), 1)
    truth[:, 0] += 0.2 * torch.sin(times)
    truth[:, 3] += 0.1 * times
    masks = torch.stack([compute_robot_masks(robot, truth, intrinsic, c, height, width) for c in cameras], dim=1)
    # --- solve: independent frame estimates initialize the coupled trajectory ---
    config = replace(presets.rgb_qpos_trajectory, frame=replace(presets.rgb_qpos, max_restarts=0))
    helper = QposTrajectoryCalibrator(config, robot, intrinsic, height, width)
    fitted_cameras = helper.solve(cameras, masks, times)
    # --- compare reconstruction and joint errors ---
    figure, axes = plt.subplots(3, len(times), figsize=(16, 6))
    rows = []
    for row, (name, q, cams) in enumerate(
        [
            ("Source", truth, cameras),
            ("Independent frames", helper.last_frame_q, cameras),
            ("Coupled trajectory", helper.last_q, fitted_cameras),
        ]
    ):
        predicted = torch.stack([compute_robot_masks(robot, q, intrinsic, c, height, width) for c in cams], dim=1)
        a, b = predicted > 0.5, masks > 0.5
        iou = ((a & b).sum((-2, -1)) / (a | b).sum((-2, -1)).clamp(min=1)).mean()
        stats = dict(
            method=name,
            iou=float(iou),
            joint_mae_deg=float(torch.rad2deg((q - truth).abs()).mean()),
            joint_wrapped_mae_deg=float(
                torch.rad2deg(((q - truth + torch.pi) % (2 * torch.pi) - torch.pi).abs()).mean()
            ),
            peak_jump_deg=float(torch.rad2deg(q.diff(dim=0).abs()).max()),
        )
        rows.append(stats)
        print(json.dumps(stats), flush=True)
        for frame in range(len(times)):
            axes[row, frame].imshow(predicted[frame, 0].cpu(), cmap="gray", vmin=0, vmax=1)
            axes[row, frame].axis("off")
            axes[row, frame].set_title(f"{times[frame]:.2f}s" if row == 0 else "")
        axes[row, 0].set_title(f"{name}\n{times[0]:.2f}s")
    output = Path("runs/hand_eye_calibration/qpos_trajectory")
    output.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output / "comparison.png", dpi=150)
    plt.close(figure)
    (output / "metrics.json").write_text(json.dumps(rows, indent=2))
    np.savez(
        output / "trajectories.npz",
        timestamps=times.cpu().numpy(),
        reference_q=truth.cpu().numpy(),
        frame_q=helper.last_frame_q.cpu().numpy(),
        trajectory_q=helper.last_q.cpu().numpy(),
        cameras=fitted_cameras.cpu().numpy(),
    )
    print(f"Saved image to {output / 'comparison.png'}")


if __name__ == "__main__":
    main()
