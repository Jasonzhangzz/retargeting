# pyright: reportMissingImports=false
"""Physical xArm + RealSense variant of `01_sim_rgb.py` (SAM2, Warp-LM); saved pose JSONs hold `T_base_camera` (OpenCV).

Usage:
    xArm7:
        uv run src/robokit/examples/hand_eye_calibration/04_real_xarm.py \
            --arm xarm7 \
            --qpos-file path/to/xarm7_qpos.npy \
            --no-auto-load-cache

    xArm6:
        uv run src/robokit/examples/hand_eye_calibration/04_real_xarm.py \
            --arm xarm6 \
            --qpos-file path/to/xarm6_qpos.npy \
            --no-auto-load-cache

    Reuse a saved initial camera pose:
        uv run src/robokit/examples/hand_eye_calibration/04_real_xarm.py \
            --arm xarm6 \
            --qpos-file path/to/xarm6_qpos.npy \
            --initial-camera-pose-file runs/real_xarm6/<run_id>/initial_camera_pose_opencv.json
"""

import datetime
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
import viser
import warp as wp
from viser.extras import ViserUrdf

from robokit.assets.robots.arms import xarm6, xarm7  # noqa: E402
from robokit.helpers.hand_eye_calibration import (  # noqa: E402
    HandEyeCalibrator,
    HandEyeCalibratorConfig,
    build_robot_render_data,
)
from robokit.helpers.hand_eye_calibration.config import PyramidLevel  # noqa: E402
from robokit.helpers.hand_eye_calibration.mask_alignment_task import render_mask  # noqa: E402
from robokit.robo import Robot as WarpRobot  # noqa: E402
from robokit.xform.numpy.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix  # noqa: E402


def pick_initial_extrinsic_viser(
    urdf_path: str,
    first_qpos: np.ndarray,
    image: np.ndarray,
    intrinsic: np.ndarray,
    initial_guess: Optional[np.ndarray] = None,
) -> np.ndarray:
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2.0, height=2.0)
    urdf_vis = ViserUrdf(server, Path(urdf_path), root_node_name="/robot")
    urdf_vis.update_cfg(np.asarray(first_qpos, dtype=np.float64))

    H, W = image.shape[:2]
    fy = float(intrinsic[1, 1])
    fov = 2.0 * np.arctan2(H / 2.0, fy)
    aspect = W / H

    if initial_guess is not None:
        T_world_cam = np.linalg.inv(initial_guess)
        position = tuple(T_world_cam[:3, 3])
        wxyz = tuple(matrix_to_quaternion(T_world_cam[:3, :3]))
    else:
        position = (0.8, 0.0, 0.5)
        wxyz = (0.5, -0.5, -0.5, 0.5)

    controls = server.scene.add_transform_controls("/camera", scale=0.3, position=position, wxyz=wxyz)
    server.scene.add_camera_frustum("/camera/frustum", fov=fov, aspect=aspect, scale=0.2, image=image, variant="filled")

    xyz_label = server.gui.add_text(
        "Camera xyz (m)",
        initial_value=f"{position[0]:+.3f}, {position[1]:+.3f}, {position[2]:+.3f}",
        disabled=True,
    )

    done = {"flag": False}
    btn = server.gui.add_button("Use this pose")

    @btn.on_click
    def _(_evt):
        done["flag"] = True

    while not done["flag"]:
        p = controls.position
        xyz_label.value = f"{p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}"
        time.sleep(0.05)

    T_world_cam = np.eye(4)
    T_world_cam[:3, :3] = quaternion_to_matrix(np.asarray(controls.wxyz, dtype=np.float64))
    T_world_cam[:3, 3] = np.asarray(controls.position, dtype=np.float64)
    server.stop()
    return np.linalg.inv(T_world_cam)


def verify_calibration_viser(
    urdf_path: str,
    first_qpos: np.ndarray,
    first_rgb: np.ndarray,
    first_depth_m: np.ndarray,
    intrinsic: np.ndarray,
    predicted_opencv: np.ndarray,
    max_points: int = 200_000,
):
    """Show the URDF at first_qpos and a point cloud from predicted T_camera_base in viser."""
    H, W = first_depth_m.shape
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = first_depth_m > 0
    z = first_depth_m[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=-1)
    T_base_cam = np.linalg.inv(predicted_opencv)
    pts_base = pts_cam @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]
    colors = first_rgb[valid]
    if pts_base.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(pts_base.shape[0], size=max_points, replace=False)
        pts_base = pts_base[idx]
        colors = colors[idx]

    centroid = pts_base.mean(0)
    bbox_min = pts_base.min(0)
    bbox_max = pts_base.max(0)
    print(f"Point cloud: {pts_base.shape[0]} pts, centroid={centroid}, bbox=[{bbox_min}, {bbox_max}]")
    cam_pos_base = T_base_cam[:3, 3]
    print(f"Camera position in base frame: {cam_pos_base} (distance={np.linalg.norm(cam_pos_base):.3f} m)")

    server = viser.ViserServer()
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=2.0, height=2.0)
    urdf_vis = ViserUrdf(server, Path(urdf_path), root_node_name="/robot")
    urdf_vis.update_cfg(np.asarray(first_qpos, dtype=np.float64))
    server.scene.add_point_cloud(
        "/pcd",
        pts_base.astype(np.float32),
        colors.astype(np.uint8),
        point_size=0.003,
        point_shape="circle",
    )
    while True:
        time.sleep(1.0)


Stage = Literal["capture", "segment", "init_cam_guess", "solve", "visualize", "verify_viser"]
Arm = Literal["xarm6", "xarm7"]


@dataclass
class Args:
    qpos_file: str
    """NumPy .npy file of shape (N, 6) or (N, 7) holding joint configurations (radians)."""
    arm: Arm = "xarm7"
    """Robot arm model."""
    stages: Optional[List[Stage]] = None
    """Select fresh stages in capture -> segment -> init_cam_guess -> solve -> visualize -> verify_viser order.

    Stage policies:
    - Listed: run fresh; verify_viser blocks forever and requires explicit selection.
    - Unlisted: load cached output, failing if missing.
    - None: defer to --auto-load-cache.
    """
    auto_load_cache: bool = True
    """Control cache use only when --stages is unspecified.

    Modes:
    - True: load every stage's cache, failing if any is missing.
    - False: run every stage fresh.
    """
    initial_camera_pose_file: Optional[str] = None
    """Use this JSON `"T_base_camera"` 4x4 OpenCV pose (inverse extrinsic) instead of init_cam_guess's viser picker."""
    urdf_path: Optional[str] = None
    """Path to the xArm URDF. Defaults by --arm."""
    xarm_ip: Optional[str] = None
    realsense_serial: Optional[str] = None
    output_dir: Optional[str] = None
    """Override output dir. Defaults to runs/real_<arm>."""
    run_id: str = field(default_factory=lambda: datetime.date.today().isoformat())
    """Run subfolder under the output dir; defaults to YYYY-MM-DD, or accepts a custom rig/experiment tag."""


def capture(args: Args) -> dict:
    import cv2
    import pyrealsense2 as rs
    from retargetingkit.targets.xarm import XArm7, XArm7Config
    from retargetingkit.targets.xarm6 import XArm6, XArm6Config

    qpos_list = list(np.load(args.qpos_file).astype(np.float64))

    xarm_ip = args.xarm_ip or ("192.168.1.228" if args.arm == "xarm6" else "192.168.1.242")
    if args.arm == "xarm6":
        arm = XArm6(XArm6Config(ip=xarm_ip, max_velocity=np.full(6, 0.25)))
    else:
        arm = XArm7(XArm7Config(ip=xarm_ip, max_velocity=np.full(7, 0.25)))
    arm.start()

    pipeline = rs.pipeline()
    config = rs.config()
    if args.realsense_serial is not None:
        config.enable_device(args.realsense_serial)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    profile = pipeline.start(config)
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    intrinsic = np.array(
        [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
    align = rs.align(rs.stream.color)

    images: List[np.ndarray] = []
    depths: List[np.ndarray] = []
    for i, q in enumerate(qpos_list):
        print(f"[{i + 1}/{len(qpos_list)}] moving to qpos={np.round(q, 3).tolist()}")
        arm.set_qpos(q)
        t0 = time.time()
        while np.linalg.norm(arm.get_qpos() - q) > 1e-2:
            if time.time() - t0 > 15.0:
                raise RuntimeError(
                    f"xArm did not reach target qpos within 15s; last err={np.linalg.norm(arm.get_qpos() - q):.4f}"
                )
            time.sleep(0.02)
        time.sleep(0.5)

        for _ in range(5):
            pipeline.wait_for_frames()
        frames = align.process(pipeline.wait_for_frames())
        color_bgr = np.asanyarray(frames.get_color_frame().get_data())
        depth_u16 = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.uint16)
        images.append(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB))
        depths.append(depth_u16)

    arm.stop()
    pipeline.stop()
    return dict(
        images=np.stack(images),
        depths=np.stack(depths),
        intrinsic=intrinsic,
        depth_scale=depth_scale,
    )


def main(args: Args):
    wp.init()
    default_out = Path("runs") / f"real_{args.arm}"
    out_dir = (Path(args.output_dir) if args.output_dir else default_out) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.urdf_path is not None:
        urdf_path = args.urdf_path
    elif args.arm == "xarm6":
        urdf_path = str(xarm6.URDF_PATH)
    else:
        urdf_path = str(xarm7.URDF_PATH)
    robot = WarpRobot.load(urdf_path, base_link_name="link_base", load_meshes=True)

    img_path = out_dir / "images.npy"
    depth_path = out_dir / "depths.npy"
    depth_scale_path = out_dir / "depth_scale.json"
    intr_path = out_dir / "camera_intrinsic.json"
    capture_files = [img_path, depth_path, depth_scale_path, intr_path]
    mask_path = out_dir / "mask.npy"
    init_guess_path = out_dir / "initial_camera_pose_opencv.json"
    predicted_path = out_dir / "camera_pose_opencv.json"

    def decide(stage: Stage, cache_paths: List[Path]) -> Tuple[bool, str]:
        if args.stages is not None:
            if stage in args.stages:
                return True, "explicit --stages"
            missing = [p.name for p in cache_paths if not p.exists()]
            if missing:
                raise FileNotFoundError(f"Stage '{stage}' is skipped but its cache is missing: {missing}")
            return False, "cache present"
        if stage == "verify_viser":
            return False, "blocking; opt in via --stages verify_viser"
        if not args.auto_load_cache:
            return True, "--no-auto-load-cache"
        if not cache_paths:
            return True, "no cache files"
        missing = [p.name for p in cache_paths if not p.exists()]
        if missing:
            return True, f"cache incomplete (missing: {', '.join(missing)})"
        return False, "cache present"

    plan = {
        "capture": decide("capture", capture_files),
        "segment": decide("segment", [mask_path]),
        "init_cam_guess": decide("init_cam_guess", [init_guess_path]),
        "solve": decide("solve", [predicted_path]),
        "visualize": decide("visualize", []),
        "verify_viser": decide("verify_viser", []),
    }
    print(f"\nOutput dir: {out_dir}")
    print("Pipeline plan:")
    for name, (run, reason) in plan.items():
        print(f"  [{'RUN ' if run else 'SKIP'}] {name:<14}  {reason}")
    print()

    if plan["capture"][0]:
        print(">>> capture: running")
        data = capture(args)
        images = data["images"]
        depths = data["depths"]
        depth_scale = data["depth_scale"]
        intrinsic = data["intrinsic"]
        np.save(img_path, images)
        np.save(depth_path, depths)
        with open(depth_scale_path, "w") as f:
            json.dump({"depth_scale": float(depth_scale)}, f, indent=4)
        with open(intr_path, "w") as f:
            json.dump({"intrinsic": intrinsic.tolist()}, f, indent=4)
        print("<<< capture: done")
    else:
        print(">>> capture: loading cache")
        images = np.load(img_path)
        depths = np.load(depth_path)
        with open(depth_scale_path) as f:
            depth_scale = float(json.load(f)["depth_scale"])
        with open(intr_path) as f:
            intrinsic = np.asarray(json.load(f)["intrinsic"], dtype=np.float64)
        print("<<< capture: loaded")

    H, W = images.shape[1], images.shape[2]
    qpos_all = np.load(args.qpos_file).astype(np.float64)
    qpos_first = qpos_all[0]

    if plan["segment"][0]:
        from robokit.helpers.hand_eye_calibration import sam2_web
        from robokit.helpers.hand_eye_calibration.sam2_web import segment

        print(">>> segment: running")
        proc = subprocess.Popen(
            [sys.executable, str(Path(sam2_web.__file__).parent / "server.py")],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        print("SAM2 server starting... open http://localhost:8000 in a browser once loaded.")
        masks = np.stack([segment(img) for img in images])
        proc.terminate()
        proc.wait()
        np.save(mask_path, masks)
        print("<<< segment: done")
    else:
        print(">>> segment: loading cache")
        masks = np.load(mask_path)
        print("<<< segment: loaded")

    if plan["init_cam_guess"][0]:
        print(">>> init_cam_guess: running")
        if args.initial_camera_pose_file is not None:
            with open(args.initial_camera_pose_file) as f:
                T_base_camera_init = np.asarray(json.load(f)["T_base_camera"], dtype=np.float64)
            initial_extrinsic = np.linalg.inv(T_base_camera_init)
        else:
            initial_extrinsic = pick_initial_extrinsic_viser(urdf_path, qpos_first, images[0], intrinsic)
        with open(init_guess_path, "w") as f:
            json.dump({"T_base_camera": np.linalg.inv(initial_extrinsic).tolist()}, f, indent=4)
        print("<<< init_cam_guess: done")
    else:
        print(">>> init_cam_guess: loading cache")
        with open(init_guess_path) as f:
            initial_extrinsic = np.linalg.inv(np.asarray(json.load(f)["T_base_camera"], dtype=np.float64))
        print("<<< init_cam_guess: loaded")
    print(f"Initial extrinsic (OpenCV T_camera_base):\n{repr(initial_extrinsic)}")

    device = torch.device("cuda")
    q_t = torch.from_numpy(qpos_all.astype(np.float32)).to(device)
    initial_extrinsic_t = torch.from_numpy(initial_extrinsic).float().to(device)

    if plan["solve"][0]:
        print(">>> solve: running")
        hec = HandEyeCalibrator(
            camera_intrinsic=torch.from_numpy(intrinsic).float().to(device),
            robot=robot,
            height=H,
            width=W,
            config=HandEyeCalibratorConfig(
                candidate_noise_t=0.6,
                candidate_noise_r=1.2,
                levels=[
                    PyramidLevel(downscale=16, blur_sigma=2.5, patience=5, obs_subset_size=5),
                    PyramidLevel(downscale=8, blur_sigma=1.8, patience=4, obs_subset_size=5),
                    PyramidLevel(downscale=4, blur_sigma=1.2, patience=4, obs_subset_size=8),
                    PyramidLevel(downscale=2, blur_sigma=0.6, patience=4),
                    PyramidLevel(downscale=1, blur_sigma=0.0, patience=8),
                ],
                solver=replace(
                    HandEyeCalibratorConfig().solver,
                    stages=[
                        replace(stage, iters=iters, num_candidates=width)
                        for stage, iters, width in zip(
                            HandEyeCalibratorConfig().solver.stages, (30, 20, 20, 20, 60), (64, 16, 8, 4, 1)
                        )
                    ],
                ),
            ),
        )
        predicted_opencv_t = hec.solve(
            initial_extrinsic_t, target_masks=torch.from_numpy(masks).float().to(device), q=q_t
        )
        predicted_opencv = predicted_opencv_t.cpu().numpy()
        with open(predicted_path, "w") as f:
            json.dump({"T_base_camera": np.linalg.inv(predicted_opencv).tolist()}, f, indent=4)
        print(f"HEC final cost (0.5*||render-target||^2): {hec.last_final_cost:.1f}")
        print("<<< solve: done")
    else:
        print(">>> solve: loading cache")
        with open(predicted_path) as f:
            predicted_opencv = np.linalg.inv(np.asarray(json.load(f)["T_base_camera"], dtype=np.float64))
        predicted_opencv_t = torch.from_numpy(predicted_opencv).float().to(device)
        print("<<< solve: loaded")
    print(f"Predicted camera extrinsic (OpenCV T_camera_base):\n{repr(predicted_opencv)}")

    if plan["visualize"][0]:
        import nvdiffrast.torch as dr

        print(">>> visualize: running")
        glctx = dr.RasterizeCudaContext()
        intrinsic_t = torch.from_numpy(intrinsic).float().to(device)
        T_base_link, link_vertices, link_faces = build_robot_render_data(robot, q_t)
        extrinsics = [initial_extrinsic_t, predicted_opencv_t]
        labels = ["Initial Extrinsic Guess", "Predicted Extrinsic", "Target (SAM2)"]
        for i in range(images.shape[0]):
            plt.rcParams.update({"font.size": 16})
            fig, axes = plt.subplots(1, 3, figsize=(7 * 3, 8))
            for j, ext in enumerate(extrinsics):
                link_masks = []
                for mi in range(len(link_vertices)):
                    m = render_mask(
                        glctx, link_vertices[mi], link_faces[mi], intrinsic_t, ext @ T_base_link[i, mi], H, W
                    )
                    link_masks.append(m)
                mask_np = torch.stack(link_masks).sum(0).clamp(max=1.0).detach().cpu().numpy() > 0
                overlay = images[i].copy()
                overlay[mask_np] = overlay[mask_np] // 4
                axes[j].imshow(overlay)
                axes[j].set_title(labels[j])
                axes[j].axis("off")
            target_bool = masks[i].astype(bool)
            target_overlay = images[i].copy()
            target_overlay[target_bool] = target_overlay[target_bool] // 4
            axes[2].imshow(target_overlay)
            axes[2].set_title(labels[2])
            axes[2].axis("off")
            plt.tight_layout()
            fig.savefig(out_dir / f"{i}.png")
            plt.close()
        print(f"Visualizations saved to {out_dir}")
        print("<<< visualize: done")

    if plan["verify_viser"][0]:
        print(">>> verify_viser: running")
        first_depth_m = depths[0].astype(np.float32) * depth_scale
        verify_calibration_viser(urdf_path, qpos_first, images[0], first_depth_m, intrinsic, predicted_opencv)


if __name__ == "__main__":
    main(tyro.cli(Args))
