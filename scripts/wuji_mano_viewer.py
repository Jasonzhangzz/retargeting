"""Browser visualization and fixed-shape online MANO IK for the calibration CLI.

Run through calibrate_wuji_mano.py view / capture --visualize, not directly.
The SDK skeleton is displayed separately: its frame and assumed anatomy may
differ from raw EMF. It is never used as the calibration ground truth.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from scripts import calibrate_wuji_mano as cal

COLORS = np.array([[245, 130, 80], [90, 180, 255], [100, 220, 150],
                   [230, 180, 60], [200, 130, 245]], dtype=np.uint8)
SDK_EDGES = np.array([(0, 1+4*f) for f in range(5)] +
                     [(1+4*f+j, 2+4*f+j) for f in range(5) for j in range(3)])


class OnlineMANO:
    """Track articulation only; keep calibrated shape/mounting/root fixed."""
    def __init__(self, args):
        torch.set_num_threads(args.threads)
        self.model = cal.PersonalizedMANO(args.mano_model, args.side, args.model_unit).to(args.device)
        self.iterations = args.ik_steps
        self.min_confidence = args.min_confidence
        self.dtype, self.device = self.model.template.dtype, self.model.template.device
        self.quality = "未附带标定报告"
        self.expected_frame = None
        report_path = args.parameters.parent / "calibration.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if report["side"] != args.side or report["model"]["model_unit"] != args.model_unit:
                raise ValueError("Calibration handedness/model units do not match viewer")
            if report["model"]["sha256"] != hashlib.sha256(self.model.path.read_bytes()).hexdigest():
                raise ValueError("Calibration source-model hash differs from --mano-model")
            self.quality = "残差检查通过" if report["fit_residuals_passed"] else "候选：标定残差未通过"
            self.expected_frame = report.get("frame_id")
        with np.load(args.parameters, allow_pickle=False) as data:
            def value(name, shape):
                array = np.array(data[name], dtype=float)
                if array.shape != shape or not np.isfinite(array).all():
                    raise ValueError(f"Invalid personalized MANO parameter: {name}")
                return torch.tensor(array, dtype=self.dtype, device=self.device)
            if list(data["finger_order"]) != list(cal.FINGERS):
                raise ValueError("Personalized MANO finger order mismatch")
            self.beta = value("betas", (self.model.shapedirs.shape[-1],))
            scale = value("global_scale", ())
            lengths, radial = value("finger_length_scales", (5,)), value("finger_radial_scales", (5,))
            if scale <= 0 or (lengths <= 0).any() or (radial <= 0).any():
                raise ValueError("Personalized MANO scales must be positive")
            self.scale, self.lengths, self.radial = scale.log(), lengths.log(), radial.log()
            self.root = cal.rotmat(value("root_axis_angle", (3,)))
            self.translation = value("translation_m", (3,))
            self.offset = value("sensor_offsets_local_m", (5, 3))
            self.mounting = cal.rotmat(value("sensor_rotations_local_axis_angle", (5, 3)))
            poses = np.asarray(data["poses_axis_angle"])
            if poses.ndim != 3 or poses.shape[1:] != (15, 3) or not np.isfinite(poses).all():
                raise ValueError("Invalid saved articulation")
            self.pose = torch.nn.Parameter(torch.tensor(poses[:1], dtype=self.dtype, device=self.device))
        self.optimizer = torch.optim.Adam([self.pose], lr=.025)
        self.distal_ids = [c[-1] for c in cal.CHAINS]

    def predict(self):
        v, j, r = self.model(self.beta, self.scale, self.lengths, self.radial, self.pose)
        distal = r[:, self.distal_ids]
        sensor = v[:, list(cal.TIPS)] + (distal @ self.offset[None, :, :, None])[..., 0]
        return (v[0] @ self.root.T + self.translation,
                j[0] @ self.root.T + self.translation,
                sensor[0] @ self.root.T + self.translation,
                self.root[None] @ distal[0] @ self.mounting)

    def update(self, row):
        if self.expected_frame and row["frame_id"] != self.expected_frame:
            raise ValueError("Live EMF frame differs from the calibration transmitter frame")
        p = torch.tensor(row["positions_m"], dtype=self.dtype, device=self.device)
        r = torch.tensor(Rotation.from_quat(row["quaternions_xyzw"]).as_matrix(), dtype=self.dtype, device=self.device)
        w = torch.tensor(row["confidence"], dtype=self.dtype, device=self.device)
        valid = w >= self.min_confidence
        if not valid.all():
            # Missing fingers cannot be inferred reliably; keep last pose.
            return None
        previous = self.pose.detach().clone()
        for _ in range(self.iterations):
            self.optimizer.zero_grad()
            _, _, sensor, sensor_r = self.predict()
            pos = ((sensor-p)/.002).square().sum(-1)
            orient = (sensor_r-r).square().sum((-1, -2)) / (.14**2 * 2)
            centred = self.pose.reshape(1, 45) - self.model.pose_mean
            basis = self.model.components[:15]
            outside = centred - (centred @ basis.T) @ basis
            loss = ((pos + .3*orient) * w).mean() + .1*(self.pose-previous).square().mean() + .05*outside.square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite online MANO tracking loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self.pose], 100.)
            self.optimizer.step()
            with torch.no_grad():
                self.pose.clamp_(-2.5, 2.5)
        with torch.no_grad():
            vertices, joints, sensor, _ = self.predict()
            arrays = [x.cpu().numpy() for x in (vertices, joints, sensor)]
        error = np.linalg.norm(arrays[2] - np.asarray(row["positions_m"]), axis=-1)
        return dict(vertices=arrays[0], joints=arrays[1], sensors=arrays[2],
                    rmse_mm=float(np.sqrt(np.mean(error**2))*1000))


class LiveViewer:
    def __init__(self, args):
        try:
            import viser
        except ImportError as exc:
            raise RuntimeError("Visualization requires: python -m pip install 'viser==1.1.1'") from exc
        self.args = args
        self.tracker = OnlineMANO(args) if args.parameters else None
        self.lock = threading.Lock()
        self.latest = {"emf": None, "skeleton": None}
        self.seen = {}
        self.stage = "等待手套数据"
        self.error = ""
        self.stop_event = threading.Event()
        self.server = viser.ViserServer(host=args.host, port=args.port, label="WUJI Glove · MANO", verbose=False)
        self.server.scene.set_up_direction("+z")
        self.server.scene.add_frame("/emf", axes_length=.035, axes_radius=.001)
        self.server.scene.add_label("/emf/title", "原始 EMF / 个性化 MANO", position=(0., 0., .15))
        self.server.scene.add_frame("/sdk", position=(.25, 0., 0.), axes_length=.035, axes_radius=.001)
        self.server.scene.add_label("/sdk/title", "SDK 骨架（预设手型，单独坐标系）", position=(0., 0., .15))
        self.status = self.server.gui.add_markdown("等待手套数据……")
        self.server.gui.add_markdown("左：五个 EMF 传感器及 MANO。右：SDK 骨架。\n\n鼠标拖动旋转、滚轮缩放；终端 Ctrl+C 退出。")
        self.show_sdk = self.server.gui.add_checkbox("显示 SDK 骨架", initial_value=True)
        self.show_mesh = self.server.gui.add_checkbox("显示 MANO mesh", initial_value=True)
        self.mesh = self.sensor_points = self.sdk_points = self.sdk_lines = self.error_lines = None
        self.frames = [self.server.scene.add_frame(f"/emf/{name}", axes_length=.018, axes_radius=.0007)
                       for name in cal.FINGERS]
        @self.server.on_client_connect
        def camera(client):
            client.camera.position = (.25, -.50, .30)
            client.camera.look_at = (.10, 0., .03)
        self.thread = threading.Thread(target=self._run, name="wuji-mano-viewer", daemon=True)
        self.thread.start()
        url = f"http://127.0.0.1:{self.server.get_port()}"
        print(f"可视化：{url}", flush=True)
        if not args.no_browser:
            webbrowser.open(url)

    def push_emf(self, row):
        p, q, c = np.asarray(row["positions_m"]), np.asarray(row["quaternions_xyzw"]), np.asarray(row["confidence"])
        if p.shape != (5, 3) or q.shape != (5, 4) or c.shape != (5,) or not all(np.isfinite(x).all() for x in (p, q, c)):
            raise ValueError("Malformed visualization EMF observation")
        if (np.linalg.norm(q, axis=-1) < .5).any() or (c < 0).any() or (c > 1).any() or not row["frame_id"]:
            raise ValueError("Invalid visualization quaternion/confidence/frame")
        with self.lock:
            old = self.latest["emf"]
            if old is not None and old[0]["frame_id"] != row["frame_id"]:
                raise ValueError("EMF coordinate frame changed; restart viewer")
            if old is not None and (old[0]["seq"], old[0]["timestamp_us"]) == (row["seq"], row["timestamp_us"]):
                return  # A repeated cached frame must not refresh stream liveness.
            self.latest["emf"] = (row, time.monotonic())

    def push_skeleton(self, frame):
        try:
            p = np.asarray([j.pose.position for j in frame.joints], dtype=float) * self.args.position_scale
            if p.shape != (21, 3) or not np.isfinite(p).all():
                raise ValueError("Expected a finite 21-joint SDK skeleton")
            with self.lock:
                self.latest["skeleton"] = (p, str(frame.header.frame_id), time.monotonic())
        except (ValueError, AttributeError, TypeError) as exc:
            self.error = str(exc)

    def _run(self):
        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                self._render()
            except Exception as exc:
                self.error = f"可视化错误：{exc}"
                self.status.content = self.error
            self.stop_event.wait(max(0., 1/self.args.fps - (time.monotonic()-started)))

    def _render(self):
        with self.lock:
            emf, skeleton = self.latest["emf"], self.latest["skeleton"]
        if emf is not None:
            row, received = emf
            fresh = time.monotonic()-received < 1.
            key = (row["seq"], row["timestamp_us"])
            if self.seen.get("emf") != key:
                self.seen["emf"] = key
                points = np.asarray(row["positions_m"], dtype=np.float32)
                colors = COLORS.copy()
                colors[np.asarray(row["confidence"]) < self.args.min_confidence] = [120, 120, 120]
                if self.sensor_points is None:
                    self.sensor_points = self.server.scene.add_point_cloud("/emf/sensors", points=points, colors=colors, point_size=.007, precision="float32")
                else:
                    self.sensor_points.points, self.sensor_points.colors = points, colors
                for f, handle in enumerate(self.frames):
                    handle.position = points[f]
                    q = row["quaternions_xyzw"][f]
                    handle.wxyz = (q[3], q[0], q[1], q[2])
                if self.tracker and fresh:
                    tracked = self.tracker.update(row)
                    if tracked:
                        rmse = tracked["rmse_mm"]
                        self.seen["rmse"] = rmse
                        vertices = tracked["vertices"].astype(np.float32)
                        if self.mesh is None:
                            self.mesh = self.server.scene.add_mesh_simple("/emf/mano", vertices=vertices, faces=self.tracker.model.faces,
                                                                        color=(225, 175, 140), opacity=.8, side="double")
                        else:
                            self.mesh.vertices = vertices
                        lines = np.stack((points, tracked["sensors"]), 1).astype(np.float32)
                        if self.error_lines is None:
                            self.error_lines = self.server.scene.add_line_segments("/emf/fit_error", points=lines, colors=(255, 90, 80), thickness=.001)
                        else:
                            self.error_lines.points = lines
            state = "接收中" if fresh else "数据已过期 / 等待新帧"
            if min(row["confidence"]) < self.args.min_confidence:
                state += " · 置信度不足，MANO 暂停更新"
            quality = self.tracker.quality if self.tracker else "未加载个性化 MANO 参数"
            error = self.seen.get("rmse")
            self.status.content = (f"**{self.stage}**\n\n{state} · seq={row['seq']}\n\n"
                                   f"EMF 坐标系：`{row['frame_id']}`\n\n"
                                   f"最低置信度：{min(row['confidence']):.2f}\n\n{quality}" +
                                   (f"\n\nMANO 传感器拟合 RMS：{error:.2f} mm" if error is not None else "") +
                                   (f"\n\n{self.error}" if self.error else ""))
        else:
            self.status.content = f"**{self.stage}**\n\n等待 EMF 数据……\n\n{self.error}"
        if skeleton is not None:
            p, frame, received = skeleton
            if self.sdk_points is None:
                self.sdk_points = self.server.scene.add_point_cloud("/sdk/joints", points=p.astype(np.float32), colors=(100, 225, 150), point_size=.004)
                self.sdk_lines = self.server.scene.add_line_segments("/sdk/bones", points=p[SDK_EDGES].astype(np.float32), colors=(100, 225, 150), thickness=.002)
            else:
                self.sdk_points.points = p.astype(np.float32)
                self.sdk_lines.points = p[SDK_EDGES].astype(np.float32)
            self.sdk_points.visible = self.sdk_lines.visible = self.show_sdk.value and time.monotonic()-received < 1.
        if self.mesh is not None:
            self.mesh.visible = self.show_mesh.value

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        self.server.stop()


def view(args):
    if args.parameters and args.mano_model is None:
        raise ValueError("MANO visualization requires --mano-model together with --parameters")
    viewer = LiveViewer(args)
    glove, subscriptions = None, []
    try:
        if args.capture:
            data = json.loads(args.capture.read_text())
            if data.get("units") != "metres" or data.get("side") != args.side or data.get("finger_order") != list(cal.FINGERS):
                raise ValueError("Replay units/side/finger order mismatch")
            rows = [(stage["name"], row) for stage in data["poses"] for row in stage["frames"]]
            if not rows:
                raise ValueError("Replay contains no captured frames")
            started = time.monotonic()
            while args.duration is None or time.monotonic()-started < args.duration:
                for name, row in rows:
                    viewer.stage = f"文件回放：{name}（非实时手套）"
                    viewer.push_emf(row)
                    time.sleep(1/args.fps)
                    if args.duration is not None and time.monotonic()-started >= args.duration:
                        return
        else:
            try:
                import wuji_sdk as sdk
            except ImportError as exc:
                raise RuntimeError("Live view requires: python -m pip install 'wuji-sdk==2026.10.26'") from exc
            manager = sdk.SdkManager.instance()
            devices = [d for d in manager.scan() if d.device_type == sdk.DeviceType.WujiGlove and (not args.sn or d.sn == args.sn)]
            if len(devices) != 1:
                raise ValueError(f"Expected one selected glove; found {[d.sn for d in devices]}; use --sn")
            glove = manager.connect(sn=devices[0].sn, device_name="mano_live_view")
            if args.side not in str(glove.hand_side().get()).lower():
                raise ValueError("Connected glove handedness mismatch")
            def on_emf(frame):
                try:
                    viewer.push_emf(cal.decode_emf(frame, args.position_scale))
                except (ValueError, AttributeError, TypeError) as exc:
                    viewer.error = str(exc)
            subscriptions.append(glove.emf_poses().subscribe_with_callback(on_emf, on_error=lambda e: setattr(viewer, "error", str(e))))
            subscriptions.append(glove.hand_skeleton().subscribe_with_callback(viewer.push_skeleton, on_error=lambda e: setattr(viewer, "error", str(e))))
            viewer.stage = f"实时手套：{devices[0].sn}"
            started = time.monotonic()
            while args.duration is None or time.monotonic()-started < args.duration:
                time.sleep(.1)
    finally:
        try:
            for sub in subscriptions:
                sub.close()
        finally:
            try:
                if glove is not None:
                    glove.disconnect()
            finally:
                viewer.close()
