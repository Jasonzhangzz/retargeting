"""Read-only Viser comparison of saved Wuji2 motion and all 20 joint angles.

python scripts/view_wuji_trajectory_comparison.py --port 8088
No IK or neural model is run by this viewer. FK uses the exact named URDF tree.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import viser
import trimesh

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
COLORS = np.array([(245, 135, 80), (80, 175, 250), (80, 215, 145),
                   (240, 195, 70), (195, 125, 245)], dtype=np.uint8)
METHODS = [
    ("SDK", "q_sdk_filtered"),
    ("Network initial guess", "q_network"),
    ("Causal prefix · 32", "source_causal_human_from_emf__causal_prefix__32"),
    ("Frozen parallel · 32", "source_causal_human_from_emf__frozen_parallel__32"),
    ("Frozen history / previous · 10", "source_causal_human_from_emf__frozen_history_previous__10"),
    ("Frozen history / network · 10", "source_causal_human_from_emf__frozen_history_network__10"),
    ("Joint LM · 32 · FUTURE", "joint"),
]
PLOT_COLORS = ("#e4e8ef", "#f8a65a", "#59bdff", "#5ee0a4", "#deb9ff", "#ff80ae", "#ffd24f")


def joint_names(side):
    out = []
    for f in FINGERS:
        stem = f if f in ("thumb", "pinky") else f + "_finger"
        suffix = ("cmc_flex", "cmc_abd", "mcp", "ip") if f == "thumb" else ("mcp_flex", "mcp_abd", "pip", "dip")
        out.extend(f"{side[0]}_{stem}_{s}" for s in suffix)
    return out


class SkeletonFK:
    def __init__(self, path, side):
        self.names = joint_names(side)
        tree = ET.parse(path).getroot()
        self.links = [x.get("name") for x in tree.findall("link")]
        self.index = {n: i for i, n in enumerate(self.links)}
        joints = tree.findall("joint")
        children = {j.find("child").get("link") for j in joints}
        self.root, = set(self.links) - children
        self.joints, available = [], {self.root}
        self.edges, self.edge_colors, self.point_colors = [], [], np.full((len(self.links), 3), 180, np.uint8)
        found = set()
        while joints:
            ready = [j for j in joints if j.find("parent").get("link") in available]
            if not ready:
                raise ValueError("Disconnected/cyclic URDF")
            for j in ready:
                name, kind = j.get("name"), j.get("type")
                if kind not in ("revolute", "fixed") or j.find("mimic") is not None:
                    raise ValueError("Unsupported joint " + name)
                qi = self.names.index(name) if name in self.names else None
                if kind != "fixed" and qi is None:
                    raise ValueError("Unmapped joint " + name)
                if qi is not None:
                    found.add(qi)
                origin, axis = j.find("origin"), j.find("axis")
                xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
                rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
                a = np.fromstring(axis.get("xyz", "1 0 0") if axis is not None else "1 0 0", sep=" ")
                a /= np.linalg.norm(a)
                parent, child = j.find("parent").get("link"), j.find("child").get("link")
                self.joints.append((parent, child, xyz, Rotation.from_euler("xyz", rpy).as_matrix(), a, qi))
                self.edges.append((self.index[parent], self.index[child]))
                color = next((COLORS[k] for k, f in enumerate(FINGERS) if f in name), np.array([180, 180, 180]))
                self.edge_colors.append(color)
                self.point_colors[self.index[child]] = color
                available.add(child)
                joints.remove(j)
        if found != set(range(20)):
            raise ValueError("Incomplete named joint order")
        self.edges = np.array(self.edges)
        self.edge_colors = np.repeat(np.array(self.edge_colors)[:, None], 2, axis=1)

    def forward(self, q, *, return_rotations=False):
        q = np.asarray(q, dtype=float)
        if q.ndim != 2 or q.shape[1] != 20 or not np.isfinite(q).all():
            raise ValueError("Expected finite [frames,20] q in radians")
        p = {self.root: np.zeros((len(q), 3))}
        r = {self.root: np.broadcast_to(np.eye(3), (len(q), 3, 3))}
        for parent, child, xyz, rot, axis, qi in self.joints:
            p[child] = p[parent] + np.einsum("nij,j->ni", r[parent], xyz)
            r[child] = r[parent] @ rot
            if qi is not None:
                r[child] = r[child] @ Rotation.from_rotvec(q[:, qi, None] * axis).as_matrix()
        positions = np.stack([p[n] for n in self.links], 1).astype(np.float32)
        if return_rotations:
            return positions, np.stack([r[n] for n in self.links], 1)
        return positions


def load_visual_meshes(path):
    """Load URDF visual geometry, including mesh scale and visual origin.

    STL assets are uploaded once; playback updates only rigid link transforms.
    """
    visuals = []
    for link in ET.parse(path).getroot().findall("link"):
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                raise ValueError("Expected Wuji2 mesh visual: " + link.get("name"))
            filename = (path.parent / mesh.get("filename")).resolve()
            geometry = trimesh.load_mesh(filename, process=True)
            if not isinstance(geometry, trimesh.Trimesh) or len(geometry.faces) == 0:
                raise ValueError("Invalid Wuji2 visual mesh: " + str(filename))
            scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            origin = visual.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            vertices = (geometry.vertices * scale) @ Rotation.from_euler("xyz", rpy).as_matrix().T + xyz
            rgba = visual.find("material/color")
            color = tuple(np.rint(np.fromstring(rgba.get("rgba"), sep=" ")[:3]*255).astype(int)) if rgba is not None else (180, 185, 195)
            visuals.append(dict(link=link.get("name"), vertices=vertices.astype(np.float32),
                                faces=np.array(geometry.faces, dtype=np.uint32), color=color))
    return visuals


class ComparisonViewer:
    def __init__(self, args):
        self.root = args.root.resolve()
        self.records = {r["index"]: r for r in json.loads((self.root / "selection.json").read_text())["records"] if r["split"] == "development"}
        self.record_options = {f'{i}: {r["collection"]}': i for i, r in self.records.items()}
        self.lock, self.dirty, self.last_ui = threading.RLock(), True, 0.
        self.cache, self.mesh_cache, self.mesh_nodes = {}, {}, []
        self.mesh_side = None
        self.server = viser.ViserServer(host=args.host, port=args.port, label="Wuji2 · trajectory comparison")
        s, g = self.server.scene, self.server.gui
        g.configure_theme(dark_mode=True, control_width="large", show_share_button=False)
        s.set_up_direction("+z")
        g.add_markdown("## Wuji2 Hand 动作轨迹对比\n真实 Wuji2 URDF 网格 + 20 关节角。鼠标旋转 / 缩放视角。")
        self.record = g.add_dropdown("动作 / collection", tuple(self.record_options))
        self.side = g.add_dropdown("手侧", ("left", "right"), initial_value=getattr(args, "side", "right"))
        self.glove = g.add_dropdown("Glove 配置", ("builtin", "hl_urdf"))
        self.play = g.add_checkbox("播放", initial_value=False)
        self.loop = g.add_checkbox("循环当前动作", initial_value=True)
        self.speed = g.add_dropdown("播放速度", ("0.25", "0.5", "1", "2"), initial_value="1")
        self.frame = g.add_slider("帧", min=0, max=1, step=1, initial_value=0)
        self.previous = g.add_button("上一帧")
        self.next = g.add_button("下一帧")
        self.method = g.add_dropdown("关节角详情 / 标注方法", tuple(n for n, _ in METHODS), initial_value=METHODS[2][0])
        self.units = g.add_dropdown("角度单位", ("rad", "deg"))
        self.show_hand = g.add_checkbox("显示 Wuji2 Hand 网格", initial_value=True)
        self.show_skeleton = g.add_checkbox("显示辅助骨骼", initial_value=False)
        self.overlay = g.add_checkbox("叠加 SDK 灰色参考骨骼", initial_value=False)
        self.show_emf = g.add_checkbox("显示 EMF 点：红=测量，绿=FK", initial_value=False)
        self.annotations = g.add_checkbox("在所选手模型上标注关节角", initial_value=False)
        self.joint = g.add_dropdown("角度曲线关节", tuple(joint_names("left")))
        self.seek_peak = g.add_button("跳至所选方法最大误差帧")
        self.reset_camera = g.add_button("重置视角")
        self.status = g.add_markdown("")
        self.emf_metrics = g.add_html("")
        self.cost_metrics = g.add_html("")
        self.plot = g.add_html("")
        self.angles = g.add_html("")
        g.add_markdown("五指颜色：橙=拇指、蓝=食指、绿=中指、黄=无名指、紫=小指。\n\nSDK/求解器显示 **filtered q**，网络显示直接预测 q。Joint LM 使用未来帧且人手联合轨迹模型不同。其余路径的人手重建为因果过程。全程不插值、不删除尖点，缺帧/reset 仅标记，不跨段连线。")
        self.nodes = []
        self.emf_nodes = []
        for i, (name, _) in enumerate(METHODS):
            offset = ((i % 4) * .29, 0., .34 if i < 4 else 0.)
            base = f"/method_{i}"
            s.add_frame(base, position=offset, show_axes=False)
            title = s.add_label(base + "/title", name, position=(0, 0, .25))
            lines = s.add_line_segments(base + "/bones", points=np.zeros((1, 2, 3), np.float32), colors=(255, 255, 255), thickness=4., thickness_units="screen")
            ghost = s.add_line_segments(base + "/sdk", points=np.zeros((1, 2, 3), np.float32), colors=(110, 120, 135), thickness=1.5, thickness_units="screen")
            points = s.add_point_cloud(base + "/joints", points=np.zeros((1, 3), np.float32), colors=(255, 255, 255), point_size=.004)
            labels = [s.add_label(base + f"/q_{j}", "", visible=False) for j in range(20)]
            self.nodes.append((title, lines, ghost, points, labels))
            measured = s.add_point_cloud(base + "/emf_measured", points=np.zeros((5, 3), np.float32), colors=(255, 85, 85), point_size=.005, visible=False)
            predicted = s.add_point_cloud(base + "/emf_fk", points=np.zeros((5, 3), np.float32), colors=(70, 240, 165), point_size=.005, visible=False)
            self.emf_nodes.append((measured, predicted))

        @self.server.on_client_connect
        def connect(client):
            self.camera(client)

        @self.reset_camera.on_click
        def reset(_):
            for client in self.server.get_clients().values():
                self.camera(client)

        for control in (self.record, self.side, self.glove):
            control.on_update(self.reload)
        for control in (self.frame, self.method, self.units, self.show_hand, self.show_skeleton, self.show_emf, self.overlay, self.annotations, self.joint):
            control.on_update(self.mark_dirty)
        self.previous.on_click(lambda _: self.step(-1))
        self.next.on_click(lambda _: self.step(1))

        @self.seek_peak.on_click
        def peak(_):
            with self.lock:
                i = [n for n, _ in METHODS].index(self.method.value)
                self.play.value = False
                self.frame.value = int(np.abs(self.data["q"][i] - self.data["q"][0]).max(1).argmax())
                self.dirty = True

        self.reload(None)

    @staticmethod
    def camera(client):
        client.camera.position = (.48, -1.30, .68)
        client.camera.look_at = (.40, 0., .29)
        client.camera.up_direction = (0., 0., 1.)

    def mark_dirty(self, _):
        self.dirty = True

    def step(self, delta):
        with self.lock:
            self.play.value = False
            self.frame.value = int(np.clip(self.frame.value + delta, 0, len(self.data["time"]) - 1))
            self.dirty = True

    def install_hand_meshes(self, side):
        if self.mesh_side == side:
            return
        path = self.root / "viser_viewer" / "assets" / f"WujiHand2_{side}.urdf"
        if side not in self.mesh_cache:
            self.mesh_cache[side] = load_visual_meshes(path)
        for nodes in self.mesh_nodes:
            for _, frame, meshes in nodes:
                for mesh in meshes:
                    mesh.remove()
                frame.remove()
        self.mesh_nodes = []
        fk = self.data["fk"]
        for i in range(len(METHODS)):
            nodes = {}
            for vi, visual in enumerate(self.mesh_cache[side]):
                link = visual["link"]
                if link not in nodes:
                    frame = self.server.scene.add_frame(f"/method_{i}/hand/{link}", show_axes=False)
                    nodes[link] = (fk.index[link], frame, [])
                frame, meshes = nodes[link][1:]
                meshes.append(self.server.scene.add_mesh_simple(
                    f"/method_{i}/hand/{link}/visual_{vi}", vertices=visual["vertices"],
                    faces=visual["faces"], color=visual["color"], material="standard",
                    flat_shading=False, side="double"))
            self.mesh_nodes.append(list(nodes.values()))
        self.mesh_side = side

    def reload(self, _):
        with self.lock:
            self.play.value = False
            rid, side, glove = self.record_options[self.record.value], self.side.value, self.glove.value
            key = (rid, side, glove)
            if key not in self.cache:
                path = self.root / "viser_viewer" / "assets" / f"WujiHand2_{side}.urdf"
                fk = SkeletonFK(path, side)
                with np.load(self.root / "oneway_comparison" / f"{glove}_{side}" / "curves.npz", allow_pickle=False) as c:
                    ri, = np.flatnonzero(c["record_indices"] == rid)
                    lo, hi = c["record_offsets"][ri:ri+2]
                    q = [np.array(c[k][lo:hi]) for _, k in METHODS[:-1]]
                with np.load(self.root / "jobs" / f"development_banded_lm_{glove}_{side}" / "budget_32" / f"trajectory_{rid}_{side}.npz", allow_pickle=False) as j:
                    q.append(np.array(j["q_filtered"]))
                    if not np.array_equal(q[0], j["q_sdk_filtered"]) or not np.array_equal(q[1], j["q_network"]):
                        raise ValueError("Method frame alignment failed")
                    timestamps = np.array(j["timestamp_ns"])
                    segments = np.array(j["segment_offsets"])
                if any(v.shape != (len(timestamps), 20) for v in q) or np.any(np.diff(timestamps) <= 0):
                    raise ValueError("Invalid trajectory lengths/timestamps")
                self.cache[key] = dict(fk=fk, q=np.stack(q), xyz=np.stack([fk.forward(v) for v in q]),
                                       time=(timestamps-timestamps[0])/1e9, segments=segments)
                emf_path = self.root / "viser_viewer" / "robot_emf" / f"{glove}_{side}_record_{rid}.npz"
                if emf_path.exists():
                    with np.load(emf_path, allow_pickle=False) as e:
                        if not np.array_equal(e["timestamp_ns"], timestamps) or e["predicted_emf_p_wrist"].shape != (len(METHODS), len(timestamps), 5, 3):
                            raise ValueError("EMF scoring frame/method alignment mismatch")
                        self.cache[key]["emf"] = {k: np.array(e[k]) for k in ("measured_emf_p_wrist", "predicted_emf_p_wrist", "position_error_m")}
                cost_path = self.root / "viser_viewer" / "objective_screw_right" / glove / "curves.npz"
                if rid == 0 and side == "right" and cost_path.exists():
                    with np.load(cost_path, allow_pickle=False) as c:
                        if not np.array_equal(c["timestamp_ns"], timestamps):
                            raise ValueError("Objective score timestamps mismatch")
                        self.cache[key]["cost"] = {k:np.array(c[k]) for k in ("cost_raw", "cost_filtered")}
            self.data = self.cache[key]
            self.install_hand_meshes(side)
            self.frame.max = len(self.data["time"]) - 1
            self.frame.value = 0
            self.joint.options = tuple(self.data["fk"].names)
            self.joint.value = self.data["fk"].names[0]
            self.dirty = True

    def render(self):
        d, f = self.data, int(self.frame.value)
        fk, selected = d["fk"], [n for n, _ in METHODS].index(self.method.value)
        factor = 180/np.pi if self.units.value == "deg" else 1.
        mesh_p, mesh_r = fk.forward(d["q"][:, f], return_rotations=True)
        mesh_q = Rotation.from_matrix(mesh_r.reshape(-1, 3, 3)).as_quat().reshape(len(METHODS), len(fk.links), 4)
        side = self.side.value
        wrist_R = mesh_r[0, fk.index[side[0] + "_wrist"]]
        wrist_p = mesh_p[0, fk.index[side[0] + "_wrist"]]
        for i, nodes in enumerate(self.mesh_nodes):
            for li, link_frame, _ in nodes:
                link_frame.position = mesh_p[i, li]
                link_frame.wxyz = mesh_q[i, li, [3, 0, 1, 2]]
                link_frame.visible = self.show_hand.value
        for i, (title, lines, ghost, points, labels) in enumerate(self.nodes):
            xyz = d["xyz"][i, f]
            lines.points, lines.colors = xyz[fk.edges], fk.edge_colors
            points.points, points.colors = xyz, fk.point_colors
            lines.visible = points.visible = self.show_skeleton.value
            ghost.points, ghost.visible = d["xyz"][0, f][fk.edges], self.overlay.value and i != 0
            error = d["q"][i, f] - d["q"][0, f]
            title.text = f"{METHODS[i][0]}\nq vs SDK: RMSE {np.sqrt(np.mean(error**2)):.4f} / max {np.abs(error).max():.4f} rad"
            measured, predicted = self.emf_nodes[i]
            measured.visible = predicted.visible = self.show_emf.value and "emf" in d
            if "emf" in d:
                measured.points = (d["emf"]["measured_emf_p_wrist"][f] @ wrist_R.T + wrist_p).astype(np.float32)
                predicted.points = (d["emf"]["predicted_emf_p_wrist"][i, f] @ wrist_R.T + wrist_p).astype(np.float32)
                title.text += f"\nEMF position RMSE: {np.sqrt(np.mean(d['emf']['position_error_m'][i, f]**2))*1000:.3f} mm"
            for j, label in enumerate(labels):
                label.visible = self.annotations.value and selected == i
                if label.visible:
                    child = next(child for _, child, _, _, _, qi in fk.joints if qi == j)
                    label.position = xyz[fk.index[child]]
                    label.text = f"q{j}: {d['q'][i,f,j]*factor:.3f}"
        now = time.monotonic()
        if now - self.last_ui < .15 and self.play.value:
            return
        self.last_ui = now
        q, sdk = d["q"][selected], d["q"][0]
        err = q - sdk
        segment = min(int(np.searchsorted(d["segments"], f, side="right") - 1), len(d["segments"]) - 2)
        self.status.content = (f"**帧 {f} / {len(q)-1} · {d['time'][f]:.3f} / {d['time'][-1]:.3f} s · 连续段 {segment}**\n\n"
                               f"所选方法全动作：q RMSE **{np.sqrt(np.mean(err**2)):.6f} rad**，最大误差 **{np.abs(err).max():.6f} rad**。")
        if "emf" in d:
            emf = d["emf"]["position_error_m"]
            metrics_rows = "".join(f"<tr><td>{html.escape(name)}</td><td>{np.sqrt(np.mean(emf[i]**2))*1000:.3f}</td><td>{np.sqrt(np.mean(emf[i,f]**2))*1000:.3f}</td></tr>" for i,(name,_) in enumerate(METHODS))
            self.emf_metrics.content = ("<div style='font-size:11px'><b>Wuji2 FK → EMF 位置 RMSE [mm]</b>"
                "<table style='width:100%;text-align:right'><tr><th>方法</th><th>完整动作</th><th>当前帧</th></tr>" + metrics_rows +
                "</table>相同 glove tip→RX 固定 TF 接到各方法的 Wuji2 指尖 FK；测量值按固定 TX→wrist 和原始掌面旋转转换。"
                "不拟合平移/尺度，不逐帧配准，不添加零位旋转修正。此项包含机器人/人手形态差异，SDK 同样计算。红=测量 EMF，绿=FK 虚拟 EMF。</div>")
        else:
            self.emf_metrics.content = ""
        if "cost" in d:
            raw, filtered = d["cost"]["cost_raw"][:, :, -1], d["cost"]["cost_filtered"][:, :, -1]
            rows = "".join(f"<tr><td>{html.escape(name)}</td><td>{raw[i].sum():.5f}</td><td>{filtered[i].sum():.5f}</td><td>{raw[i,f]:.5f}</td></tr>" for i,(name,_) in enumerate(METHODS))
            high = max(float(raw.max())*1.03, 1e-10)
            paths = []
            for i in range(len(METHODS)):
                pts = " ".join(f"{20+440*k/max(len(raw[i])-1,1):.2f},{130-110*v/high:.2f}" for k,v in enumerate(raw[i]))
                paths.append(f'<polyline fill="none" stroke="{PLOT_COLORS[i]}" stroke-width="1" points="{pts}"/>')
            cursor = 20+440*f/max(raw.shape[1]-1,1)
            self.cost_metrics.content = ("<div style='font-size:11px'><b>拧螺丝右手 · 原始阶段2 F(q)</b><table style='width:100%;text-align:right'><tr><th>方法</th><th>raw 总和</th><th>filtered 总和</th><th>当前帧 raw</th></tr>" + rows +
                "</table>统一使用 SDK 人手目标；history 参考各方法上一帧 raw q。filtered 评分也保持相同 raw history。网络为直接输出。"
                "<svg viewBox='0 0 480 150' style='width:100%;background:#171d28'>" + "".join(paths) +
                f"<line x1='{cursor}' x2='{cursor}' y1='15' y2='135' stroke='white' stroke-dasharray='3 3'/></svg>"
                "曲线显示逐帧 raw F(q)，颜色与下方关节曲线一致。完整分项和交互曲线见 objective_screw_right/comparison.html。</div>")
        else:
            self.cost_metrics.content = ""
        rows = "".join(f"<tr><td>{html.escape(n)}</td><td>{sdk[f,j]*factor:.4f}</td><td>{q[f,j]*factor:.4f}</td><td style='color:{'#ffad82' if abs(err[f,j])>1e-4 else '#7be6ab'}'>{err[f,j]*factor:+.4f}</td></tr>" for j,n in enumerate(fk.names))
        self.angles.content = f"<div style='font-size:11px;overflow-x:auto'><b>{html.escape(self.method.value)} · {self.units.value}</b><table style='width:100%;font-variant-numeric:tabular-nums;text-align:right'><tr><th>joint</th><th>SDK</th><th>selected</th><th>Δ</th></tr>{rows}</table></div>"
        j = fk.names.index(self.joint.value)
        vals = d["q"][:, :, j] * factor
        low, high = float(vals.min()), float(vals.max())
        margin = max((high-low)*.08, .01)
        low, high = low-margin, high+margin
        def xy(t, v):
            return 30 + t/max(d["time"][-1], 1e-9)*440, 145 - (v-low)/(high-low)*120
        paths = []
        for i in range(len(METHODS)):
            for a, b in zip(d["segments"][:-1], d["segments"][1:]):
                coords = " ".join(f"{x:.2f},{y:.2f}" for x,y in (xy(d['time'][k], vals[i,k]) for k in range(a,b)))
                paths.append(f'<polyline fill="none" stroke="{PLOT_COLORS[i]}" stroke-width="{1.7 if i==selected else .8}" points="{coords}"/>')
        x, _ = xy(d["time"][f], low)
        legend = " · ".join(f'<span style="color:{c}">{html.escape(n)}</span>' for (n,_),c in zip(METHODS,PLOT_COLORS))
        self.plot.content = (f'<div style="font-size:11px">{html.escape(self.joint.value)} [{self.units.value}]<svg viewBox="0 0 500 170" style="width:100%;background:#171d28">'
                             f'<text x="2" y="22" fill="#aab5c5" font-size="10">{high:.2f}</text><text x="2" y="147" fill="#aab5c5" font-size="10">{low:.2f}</text>'
                             + "".join(paths) + f'<line x1="{x}" x2="{x}" y1="20" y2="147" stroke="white" stroke-dasharray="3 3"/>'
                             f'<text x="30" y="164" fill="#aab5c5" font-size="10">0 s</text><text x="435" y="164" fill="#aab5c5" font-size="10">{d["time"][-1]:.1f} s</text></svg>{legend}</div>')

    def run(self):
        print(f"VIEWER_READY http://127.0.0.1:{self.server.get_port()}", flush=True)
        last = time.monotonic()
        while True:
            with self.lock:
                now = time.monotonic()
                if self.play.value:
                    # Keep fractional progress; round-trip through frame indices would lose time.
                    if not getattr(self, "was_playing", False) or self.frame.value != getattr(self, "last_frame", None):
                        self.play_time = self.data["time"][self.frame.value]
                    self.play_time += (now-last)*float(self.speed.value)
                    if self.play_time > self.data["time"][-1]:
                        if self.loop.value:
                            self.play_time = 0.
                        else:
                            self.play.value = False
                            self.play_time = self.data["time"][-1]
                    self.frame.value = int(np.clip(np.searchsorted(self.data["time"], self.play_time, side="right")-1, 0, self.frame.max))
                    self.dirty = True
                self.was_playing, self.last_frame = self.play.value, self.frame.value
                if self.dirty:
                    with self.server.atomic():
                        self.render()
                    self.dirty = False
                last = now
            time.sleep(1/40)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "results/wuji_sparse_trajectory_20261112")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    args = parser.parse_args()
    ComparisonViewer(args).run()


if __name__ == "__main__":
    main()
