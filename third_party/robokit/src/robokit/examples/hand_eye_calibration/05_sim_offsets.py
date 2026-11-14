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
# pyright: reportMissingImports=false
"""Fit camera extrinsics together with joint encoder bias or an arm/hand mount offset; both modes compare against camera-only fitting.

Run with CUDA: uv run src/robokit/examples/hand_eye_calibration/05_sim_offsets.py --offset joints; use --offset mount for the xArm7+Inspire adapter.
"""

from dataclasses import replace
from typing import Literal

import numpy as np
import torch
import tyro
import warp as wp

from robokit.assets.robots.arms import xarm7
from robokit.assets.robots.assembly import xarm7_inspire
from robokit.helpers.hand_eye_calibration import HandEyeCalibrator, compute_robot_masks, presets
from robokit.helpers.hand_eye_calibration.mask_alignment_task import MaskAlignmentOffsetTask
from robokit.lie.se3 import se3_identity, se3_to_matrix
from robokit.opt.var_values import VarKey
from robokit.robo import Robot
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.xform.warp.torch_wrappers import (
    axis_angle_to_matrix,
    matrix_to_quaternion,
    quaternion_to_axis_angle,
)


def se3_error(predicted: torch.Tensor, gt: torch.Tensor) -> str:
    """Format the translation and rotation error of `gt^-1 @ predicted`."""
    d = torch.linalg.inv(gt) @ predicted
    axis_angle = quaternion_to_axis_angle(matrix_to_quaternion(d[:3, :3]))
    return (
        f"{float(torch.linalg.norm(d[:3, 3])) * 1000:.2f} mm / {float(torch.linalg.norm(axis_angle).rad2deg()):.3f} deg"
    )


SAMPLES = 12
H = W = 512
OFFSET_DEG = 2.0
MOUNT_LINK = "hand_base_link"


def main(offset: Literal["joints", "mount"] = "joints"):
    wp.init()
    rng = np.random.default_rng(0)
    robot = Robot.load(str(xarm7.URDF_PATH if offset == "joints" else xarm7_inspire.URDF_PATH), load_meshes=True)
    n_dofs = robot.spec.num_actuated_joints

    # --- data: raised arm and, when attached, half-closed hand ---
    q_nom = robot.spec.actuated_joint_limits.mean(axis=1).astype(np.float32)
    q_nom[:7] = [0.0, -0.5, 0.0, 1.0, 0.0, 0.6, 0.0]
    q = q_nom + rng.uniform(-0.5, 0.5, (SAMPLES, n_dofs)).astype(np.float32)
    limits = robot.spec.actuated_joint_limits.astype(np.float32)
    q = torch.from_numpy(np.clip(q, limits[:, 0], limits[:, 1])).cuda()

    # --- camera looking at the arm from the front-left (OpenCV convention) ---
    eye, target = np.array([1.1, -0.9, 0.9]), np.array([0.0, 0.0, 0.35])
    fwd = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    T_base_cam = np.eye(4, dtype=np.float32)
    T_base_cam[:3, :3] = np.stack([right, np.cross(fwd, right), fwd], axis=1)
    T_base_cam[:3, 3] = eye
    gt_extrinsic = torch.from_numpy(np.linalg.inv(T_base_cam).astype(np.float32)).cuda()
    intrinsic = torch.tensor([[0.9 * W, 0, W / 2], [0, 0.9 * W, H / 2], [0, 0, 1]]).float().cuda()

    # --- targets: render the true offset, then solve using the nominal robot ---
    if offset == "joints":
        gt_offset = torch.from_numpy((rng.uniform(-1, 1, 7) * np.deg2rad(OFFSET_DEG)).astype(np.float32)).cuda()
        masks = compute_robot_masks(robot, q + gt_offset, intrinsic, gt_extrinsic, H, W)
        task = MaskAlignmentOffsetTask(offset_joints=list(range(7)))
    else:
        gt_offset = torch.eye(4, device="cuda")
        mount_axis = torch.tensor([0.2, -0.9, 0.4], device="cuda")
        gt_offset[:3, :3] = axis_angle_to_matrix(mount_axis / mount_axis.norm() * np.deg2rad(OFFSET_DEG))
        gt_offset[:3, 3] = torch.tensor([0.006, -0.004, 0.008], device="cuda")
        masks = compute_robot_masks(
            robot, q, intrinsic, gt_extrinsic, H, W, mount_link=MOUNT_LINK, T_mount_offset=gt_offset
        )
        task = MaskAlignmentOffsetTask(offset_link=MOUNT_LINK)

    delta = torch.eye(4).cuda()
    axis = torch.tensor([0.3, -0.8, 0.5]).float().cuda()
    delta[:3, :3] = axis_angle_to_matrix(axis / axis.norm() * np.deg2rad(15.0))
    delta[:3, 3] = torch.tensor([0.06, -0.05, 0.05]).float().cuda()
    initial = gt_extrinsic @ delta

    # --- solve: camera only, then camera plus the selected offset ---
    camera_only = HandEyeCalibrator(presets.rgb, robot, intrinsic, H, W)
    predicted_camera_only = camera_only.solve(initial, target_masks=masks, q=q)

    # A one-sigma offset costs ~2.5e-4/pixel, keeping weakly observable variables near nominal.
    scale = float(np.sqrt(2.5e-4 * masks.sum().item()))
    if offset == "joints":
        rest = ConfigurationTask(
            rest_q=np.zeros(7, dtype=np.float32), weight=scale / np.deg2rad(OFFSET_DEG), var_key=VarKey.QPOS_OFFSET
        )
    else:
        rest = ConfigurationTask(
            T_world_base_rest=se3_identity(shape=(1,), device="cuda:0"),
            base_weight=[scale / 0.01] * 3 + [scale / np.deg2rad(OFFSET_DEG)] * 3,
            include_joints=False,
            var_key=VarKey.MOUNT_OFFSET,
        )
    hec = HandEyeCalibrator(replace(presets.rgb, terms=[task, rest]), robot, intrinsic, H, W)
    with_offset = hec.solve(initial, target_masks=masks, q=q)

    # --- report ---
    for label, calibration, predicted in (
        ("camera only", camera_only, predicted_camera_only),
        (f"camera + {offset}", hec, with_offset),
    ):
        print(f"{label}: cost {calibration.last_final_cost:.1f}   camera {se3_error(predicted, gt_extrinsic)}")
    assert hec.last_var is not None
    if offset == "joints":
        solved = wp.to_torch(hec.last_var.get(VarKey.QPOS_OFFSET).x)[0]
        for label, error in (("nominal", gt_offset.abs()), ("fitted", (solved - gt_offset).abs())):
            print(f"{label} joint error: {error.rad2deg().mean():.3f} deg mean / {error.rad2deg().max():.3f} deg max")
    else:
        solved = wp.to_torch(se3_to_matrix(hec.last_var.get(VarKey.MOUNT_OFFSET).xyz_wxyz))[0]
        print(f"nominal mount error: {se3_error(torch.eye(4, device='cuda'), gt_offset)}")
        print(f"fitted mount error: {se3_error(solved, gt_offset)}")


if __name__ == "__main__":
    tyro.cli(main)
