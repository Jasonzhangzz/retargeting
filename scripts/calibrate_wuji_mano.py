#!/usr/bin/env python3
"""Personalize a MANO mesh using Wuji Glove EMF observations and measurements.

Python >=3.10; numpy, scipy, torch; live capture additionally needs wuji-sdk.
  python scripts/calibrate_wuji_mano.py init --output output/my_hand
  # Fill output/my_hand/measurements.json (millimetres), then:
  python scripts/calibrate_wuji_mano.py calibrate --mano-model /path/MANO_RIGHT.pkl \
      --measurements output/my_hand/measurements.json --output output/my_hand
  python scripts/calibrate_wuji_mano.py fit --capture output/my_hand/capture.json \
      --mano-model /path/MANO_RIGHT.pkl --measurements output/my_hand/measurements.json \
      --output output/my_hand_fit
  python scripts/calibrate_wuji_mano.py view
  python scripts/calibrate_wuji_mano.py view --mano-model /path/MANO_RIGHT.pkl \
      --parameters output/my_hand/personalized_mano.npz

Reads raw emf_poses(), not SDK hand_skeleton(): the latter already assumes a
hand model. Never commands a robotic hand or modifies the SDK user calibration.
The extension adds independent finger length/radial scales to MANO beta/LBS.
Those parameters MUST accompany beta when reusing the fitted model. Width-only
measurements imply isotropic radial scaling; depth is not independently fitted.
Sensor mounting transforms and per-pose articulation are nuisance variables.
Calibration motions alone do not uniquely determine full surface anatomy.

Sources (checked against SDK 2026.10.26, examples commit b6be39b1c923b822f627ac8564dd91ceb38b303b):
  https://github.com/wuji-technology/wuji-sdk/tree/main/examples/python/wuji_glove
  https://mano.is.tue.mpg.de/  (licensed model weights are not bundled here)
  https://github.com/vchoutas/smplx/blob/main/smplx/vertex_ids.py
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
CHAINS = ((13, 14, 15), (1, 2, 3), (4, 5, 6), (10, 11, 12), (7, 8, 9))
TIPS = (744, 320, 443, 554, 671)  # Official smplx MANO landmark convention.
POSES = (
    ("flat_open", "手掌摊平，四指伸直、自然分开；拇指自然展开"),
    ("flat_together", "手掌摊平，四指伸直并拢；不要用另一只手压"),
    ("pinch_index", "拇指指腹轻触食指指腹（靠近指尖），不要用力挤压"),
    ("pinch_middle", "拇指指腹轻触中指指腹（靠近指尖）"),
    ("pinch_ring", "拇指指腹轻触无名指指腹（靠近指尖）"),
    ("pinch_pinky", "拇指指腹轻触小指指腹（靠近指尖）"),
    ("hook", "四指弯成钩状，掌指关节尽量伸直，拇指放松"),
    ("loose_fist", "自然虚握拳，不用力；拇指不要压住传感器"),
    ("pinch_index_bent", "再次拇食对指，这次将掌指关节多弯一些"),
    ("flat_open_repeat", "再次摊平，四指伸直、自然分开"),
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, path)


class _ChValue:
    """Decode only the numeric Ch/Select nodes used in older MANO PKLs.

    Avoid importing obsolete chumpy or monkey-patching numpy/inspect. PKL inputs
    must still be trusted: this is a compatibility reader, not a safe unpickler.
    """
    def __setstate__(self, state):
        self.__dict__.update(state)

    def array(self):
        if "x" in self.__dict__:
            return np.asarray(self.x)
        if "a" in self.__dict__ and "idxs" in self.__dict__:
            return as_array(self.a).ravel()[self.idxs].reshape(self.preferred_shape)
        raise ValueError("Unsupported legacy chumpy expression; convert the official model to numpy first")


class _ManoReader(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            if (module, name) not in {("chumpy.ch", "Ch"), ("chumpy.reordering", "Select")}:
                raise ValueError(f"Unsupported chumpy node {module}.{name}")
            return _ChValue
        if module in {"scipy.sparse.csc", "scipy.sparse.csr"}:
            from scipy import sparse
            return getattr(sparse, name)
        return super().find_class(module, name)


def as_array(value):
    if isinstance(value, _ChValue):
        return value.array()
    if hasattr(value, "toarray"):
        return value.toarray()
    return np.asarray(value)


def rotmat(aa):
    """Stable differentiable Rodrigues formula, including at zero angle."""
    angle = torch.linalg.vector_norm(aa, dim=-1)
    x, y, z = aa.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*aa.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device)
    a = torch.sinc(angle / math.pi)[..., None, None]
    b = (0.5 * torch.sinc(angle / (2 * math.pi)).square())[..., None, None]
    return eye + a * skew + b * (skew @ skew)


class PersonalizedMANO(torch.nn.Module):
    """MANO shape/pose blend shapes + optional rest-mesh finger scaling + LBS.

    Public output: vertices (metres), 16 MANO joints, 16 global joint rotations.
    pose contains 15 absolute local axis-angle rotations (no hands_mean added).
    length/radial scales are in thumb,index,middle,ring,pinky order.
    """
    def __init__(self, path, side="right", model_unit="m"):
        super().__init__()
        self.path = Path(path).expanduser().resolve()
        self.side = side
        with self.path.open("rb") as stream:
            data = _ManoReader(stream, encoding="latin1").load()
        unit = 1.0 if model_unit == "m" else .001
        for key, name in (("v_template", "template"), ("shapedirs", "shapedirs"),
                          ("posedirs", "posedirs"), ("J_regressor", "regressor"),
                          ("weights", "weights"), ("hands_components", "components"),
                          ("hands_mean", "pose_mean")):
            value = as_array(data[key]).astype(np.float64)
            if key in {"v_template", "shapedirs", "posedirs"}:
                value *= unit
            self.register_buffer(name, torch.tensor(value))
        self.faces = as_array(data["f"]).astype(np.int64)
        if "retarget_rest_joints" in data:
            joints = np.asarray(data["retarget_rest_joints"], dtype=np.float64)
            if joints.shape != (16, 3) or not np.isfinite(joints).all():
                raise ValueError("Invalid calibrated MANO rest joints")
            self.register_buffer("calibrated_rest_joints", torch.tensor(joints * unit))
        self.appearance = dict(data.get("retarget_appearance", {}))
        self.asset_metadata = dict(data.get("retarget_metadata", {}))
        tree = as_array(data["kintree_table"])
        lookup = {int(label): i for i, label in enumerate(tree[1])}
        self.parents = [-1] + [lookup[int(label)] for label in tree[0, 1:]]
        if self.template.shape != (778, 3) or self.weights.shape != (778, 16):
            raise ValueError("Expected standard MANO topology: 778 vertices / 16 skinning joints")
        if self.shapedirs.shape[:2] != (778, 3) or self.posedirs.shape != (778, 3, 135):
            raise ValueError("Invalid MANO blend-shape dimensions")
        if not all(self.parents[j] < j for j in range(1, 16)):
            raise ValueError("MANO joints must be topologically ordered")
        extent = float(torch.linalg.vector_norm(self.template.max(0).values - self.template.min(0).values))
        if not .08 < extent < .5:
            raise ValueError("Unexpected model scale; select --model-unit m or mm explicitly")
        rest = self.calibrated_rest_joints if hasattr(self, "calibrated_rest_joints") else self.regressor @ self.template
        self.register_buffer("finger_weights", torch.stack([self.weights[:, list(c)].sum(-1) for c in CHAINS], -1))
        # Palm lateral axis and a palm normal, computed from anatomical landmarks.
        lateral = torch.nn.functional.normalize(rest[1] - rest[7], dim=0)
        forward = torch.nn.functional.normalize(rest[4] - rest[0], dim=0)
        normal = torch.nn.functional.normalize(torch.linalg.cross(lateral, forward), dim=0)
        self.register_buffer("palm_lateral", lateral)
        self.register_buffer("palm_normal", normal)
        sections, pads = [], []
        for f, chain in enumerate(CHAINS):
            # MANO thumb chain starts at CMC; finger widths are measured on
            # the proximal phalanx (MCP->IP), excluding the thumb metacarpal.
            section_chain = chain[1:] if f == 0 else chain
            root = rest[section_chain[0]]
            axis = torch.nn.functional.normalize(self.template[TIPS[f]] - root, dim=0)
            u = ((self.template - root) * axis).sum(-1)
            proximal = float(((rest[section_chain[1]] - root) * axis).sum())
            membership = self.finger_weights[:, f] > .65
            candidates = torch.where(membership & (u > 0) & (u < proximal))[0]
            if len(candidates) < 4:
                raise ValueError(f"Cannot locate proximal cross-section for {FINGERS[f]}")
            section = candidates[torch.argsort((u[candidates] - proximal * .5).abs())[:20]]
            sections.append(section)
            distal = torch.where(self.weights[:, chain[-1]] > .5)[0]
            # A local surface patch near the official fingertip landmark; mesh
            # contact is a weak cue, never treated as complete collision testing.
            distance = torch.linalg.vector_norm(self.template[distal] - self.template[TIPS[f]], dim=-1)
            pads.append(distal[torch.argsort(distance)[:24]])
        for i, ids in enumerate(sections):
            self.register_buffer(f"section_{i}", ids)
        for i, ids in enumerate(pads):
            self.register_buffer(f"pad_{i}", ids)

    def rest_shape(self, beta, log_scale, log_lengths, log_radial):
        v = self.template + torch.einsum("vck,k->vc", self.shapedirs, beta)
        j = self.calibrated_rest_joints if hasattr(self, "calibrated_rest_joints") else self.regressor @ v
        delta_v = torch.zeros_like(v)
        new_j = [j[0]] + [None] * 15
        for f, chain in enumerate(CHAINS):
            root = j[chain[0]]
            axis = torch.nn.functional.normalize(v[TIPS[f]] - root, dim=0)
            offset = v - root
            axial = (offset * axis).sum(-1, keepdim=True) * axis
            change = axial * torch.expm1(log_lengths[f]) + (offset - axial) * torch.expm1(log_radial[f])
            delta_v = delta_v + change * self.finger_weights[:, f, None]
            for index in chain:
                d = j[index] - root
                longitudinal = (d * axis).sum() * axis
                new_j[index] = j[index] + longitudinal * torch.expm1(log_lengths[f])
        return (v + delta_v) * log_scale.exp(), torch.stack(new_j) * log_scale.exp()

    def forward(self, beta, log_scale, log_lengths, log_radial, pose):
        v, j = self.rest_shape(beta, log_scale, log_lengths, log_radial)
        batch = len(pose)
        local_r = rotmat(torch.cat((torch.zeros_like(pose[:, :1]), pose), dim=1))
        feature = (local_r[:, 1:] - torch.eye(3, dtype=v.dtype, device=v.device)).reshape(batch, 135)
        offsets = torch.einsum("bp,vcp->bvc", feature, self.posedirs) * log_scale.exp()
        posed = v[None] + offsets
        transforms = []
        for index, parent in enumerate(self.parents):
            relative = j[index] if parent == -1 else j[index] - j[parent]
            top = torch.cat((local_r[:, index], relative[None, :, None].expand(batch, -1, -1)), -1)
            bottom = torch.zeros((batch, 1, 4), dtype=v.dtype, device=v.device)
            bottom[:, :, 3] = 1
            transform = torch.cat((top, bottom), 1)
            transforms.append(transform if parent == -1 else transforms[parent] @ transform)
        g = torch.stack(transforms, 1)
        translation = g[:, :, :3, 3] - (g[:, :, :3, :3] @ j[None, :, :, None]).squeeze(-1)
        top = torch.cat((g[:, :, :3, :3], translation[..., None]), -1)
        bottom = torch.zeros((batch, 16, 1, 4), dtype=v.dtype, device=v.device)
        bottom[..., 3] = 1
        corrected = torch.cat((top, bottom), -2)
        blended = torch.einsum("vj,bjkl->bvkl", self.weights, corrected)
        homogeneous = torch.cat((posed, torch.ones_like(posed[..., :1])), -1)
        vertices = (blended @ homogeneous[..., None])[..., :3, 0]
        return vertices, g[:, :, :3, 3], g[:, :, :3, :3]

    def dimensions(self, v, j):
        values = {}
        for f, chain in enumerate(CHAINS):
            axis = torch.nn.functional.normalize(v[TIPS[f]] - j[chain[0]], dim=0)
            side = torch.nn.functional.normalize(torch.linalg.cross(axis, self.palm_normal), dim=0)
            projection = v[getattr(self, f"section_{f}")] @ side
            values[f"{FINGERS[f]}_width_mm"] = (projection.max() - projection.min()) * 1000
            measured_chain = chain[1:] if f == 0 else chain
            points = torch.cat((j[list(measured_chain)], v[TIPS[f]][None]), 0)
            values[f"{FINGERS[f]}_length_mm"] = torch.linalg.vector_norm(points[1:] - points[:-1], dim=-1).sum() * 1000
        # Anatomical MCP centre span, deliberately distinguished from outer palm width.
        values["palm_mcp_span_mm"] = torch.linalg.vector_norm(j[1] - j[7]) * 1000
        # Outer mesh width at the midpoint between wrist and middle MCP.
        forward = torch.nn.functional.normalize(j[4] - j[0], dim=0)
        u = (v - j[0]) @ forward
        length = (j[4] - j[0]) @ forward
        ids = (self.finger_weights.sum(-1) < .5) & (u > length * .45) & (u < length * .75)
        projection = v[ids] @ self.palm_lateral
        if len(projection) < 2:
            raise ValueError("Cannot locate palm width section")
        values["palm_width_mm"] = (projection.max() - projection.min()) * 1000
        return values


def measurements_template():
    return {
        "description": "单位毫米。空值表示未测量；请替换五指宽度及掌宽。宽度需脱下手套测量。",
        "width_definition": "各手指近节中段的左右宽度：MCP到PIP中点（拇指MCP到IP中点）；不要量周长。",
        "length_definition": "可选：伸直手指MCP关节中心到指尖的长度，拇指从MCP量起，不含掌骨。",
        "palm_definition": "掌部中段外缘宽度，约腕横纹到中指MCP距离的60%，不包含拇指。",
        "widths_mm": {f: None for f in FINGERS},
        "lengths_mm": {f: None for f in FINGERS},
        "palm_width_mm": None,
        "measurement_sigma_mm": 2.0,
    }


def load_measurements(path):
    if path is None:
        return {}, 2.0
    data = json.loads(Path(path).read_text())
    targets = {}
    for kind in ("widths", "lengths"):
        for finger, value in data.get(kind + "_mm", {}).items():
            if finger not in FINGERS:
                raise ValueError(f"Unknown finger in measurements: {finger}")
            if value is not None:
                upper = 45 if kind == "widths" else 140
                if not isinstance(value, (float, int)) or not math.isfinite(value) or not 3 < value < upper:
                    raise ValueError(f"Invalid {finger} {kind} in millimetres: {value}")
                targets[f"{finger}_{'width' if kind == 'widths' else 'length'}_mm"] = float(value)
    for key in ("palm_width_mm", "palm_mcp_span_mm"):
        value = data.get(key)
        if value is not None:
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 20 < value < 160:
                raise ValueError(f"Invalid {key}")
            targets[key] = float(value)
    sigma = float(data.get("measurement_sigma_mm", 2))
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("measurement_sigma_mm must be positive")
    return targets, sigma


def decode_emf(frame, position_scale=1.0):
    if len(frame.poses) != 5:
        raise ValueError("Expected five EMF sensors in thumb,index,middle,ring,pinky order")
    p = np.asarray([item.pose.position for item in frame.poses], dtype=float) * position_scale
    q = np.asarray([[item.pose.orientation.x, item.pose.orientation.y,
                     item.pose.orientation.z, item.pose.orientation.w] for item in frame.poses])
    confidence = np.asarray([item.confidence for item in frame.poses], dtype=float)
    if p.shape != (5, 3) or not all(np.isfinite(x).all() for x in (p, q, confidence)):
        raise ValueError("Non-finite or malformed EMF frame")
    norm = np.linalg.norm(q, axis=-1)
    if np.any(norm < .5) or np.any(norm > 1.5) or np.any(confidence < 0) or np.any(confidence > 1):
        raise ValueError("Invalid EMF quaternion/confidence")
    return {"seq": int(frame.header.seq), "timestamp_us": int(frame.header.timestamp_us),
            "frame_id": str(frame.header.frame_id), "positions_m": p.tolist(),
            "quaternions_xyzw": (q / norm[:, None]).tolist(), "confidence": confidence.tolist()}


def stable_frames(frames, min_confidence=.7, position_std_mm=2.0, angle_std_deg=5.0):
    if len(frames) < 10:
        return False
    if min(min(f["confidence"]) for f in frames) < min_confidence:
        return False
    p = np.asarray([f["positions_m"] for f in frames])
    if np.linalg.norm(p.std(0), axis=-1).max() * 1000 > position_std_mm:
        return False
    q = np.asarray([f["quaternions_xyzw"] for f in frames])
    for finger in range(5):
        rotations = Rotation.from_quat(q[:, finger])
        if np.sqrt(np.mean((rotations.mean().inv() * rotations).magnitude() ** 2)) * 180 / math.pi > angle_std_deg:
            return False
    return True


def visualization_module():
    # Support both direct script execution and imports from repository tests.
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from scripts import wuji_mano_viewer
    return wuji_mano_viewer


def capture(args):
    try:
        import wuji_sdk as sdk
    except ImportError as exc:
        raise RuntimeError("Live capture requires: python -m pip install 'wuji-sdk==2026.10.26'") from exc
    destination = args.output / "capture.json"
    if destination.exists():
        raise ValueError(f"Capture already exists: {destination}; choose a new --output")
    manager = sdk.SdkManager.instance()
    devices = [d for d in manager.scan() if d.device_type == sdk.DeviceType.WujiGlove]
    if args.sn:
        devices = [d for d in devices if d.sn == args.sn]
    if len(devices) != 1:
        raise ValueError(f"Expected one selected Wuji Glove; found {[d.sn for d in devices]}. Use --sn.")
    glove = manager.connect(sn=devices[0].sn, device_name="mano_calibration")
    sub, skeleton_sub, viewer = None, None, None
    frames = collections.deque(maxlen=1000)
    errors = collections.deque(maxlen=10)
    lock = threading.Lock()
    result = {"schema_version": 1, "status": "capture_incomplete", "side": args.side,
              "source": "wuji_sdk.emf_poses", "sdk_version": importlib.metadata.version("wuji-sdk"),
              "serial_number": devices[0].sn, "units": "metres", "finger_order": list(FINGERS),
              "position_scale": args.position_scale, "poses": []}

    def callback(frame):
        try:
            row = decode_emf(frame, args.position_scale)
            row["received_monotonic_s"] = time.monotonic()
            with lock:
                frames.append(row)
            if viewer is not None:
                viewer.push_emf(row)
        except (ValueError, TypeError, AttributeError) as exc:
            with lock:
                errors.append(str(exc))

    try:
        side = str(glove.hand_side().get()).lower()
        if args.side not in side:
            raise ValueError(f"Connected glove reports {side}; requested {args.side}")
        result["hand_side_reported"] = side
        if getattr(args, "visualize", False):
            viewer = visualization_module().LiveViewer(args)
            skeleton_sub = glove.hand_skeleton().subscribe_with_callback(viewer.push_skeleton)
        sub = glove.emf_poses().subscribe_with_callback(callback, on_error=lambda message: errors.append(str(message)))
        write_json(destination, result)
        print("保持手套及传感器佩戴位置不变，远离金属干扰。每个动作按回车后保持不动。", flush=True)
        frame_id = None
        for name, guide in POSES:
            if viewer is not None:
                viewer.stage = f"标定 {len(result['poses'])+1}/{len(POSES)}：{guide}"
            input(f"\n[{len(result['poses'])+1}/{len(POSES)}] {guide}\n准备好后按回车：")
            start = time.monotonic()
            accepted = None
            while time.monotonic() - start < args.timeout:
                now = time.monotonic()
                with lock:
                    window = [row for row in frames if row["received_monotonic_s"] >= max(start, now - args.hold_seconds)]
                    latest_errors = list(errors)
                if window:
                    # Count sensor frames, not repeated cached observations.
                    unique = {(row["seq"], row["timestamp_us"]): row for row in window}
                    window = list(unique.values())
                    stamps = np.asarray([row["timestamp_us"] for row in window])
                    ids = {row["frame_id"] for row in window}
                    if len(ids) != 1 or not next(iter(ids)):
                        raise ValueError("EMF coordinate frame is empty or changed during collection")
                    if frame_id is not None and ids != {frame_id}:
                        raise ValueError("EMF transmitter coordinate frame changed; restart calibration")
                    duration = window[-1]["received_monotonic_s"] - window[0]["received_monotonic_s"]
                    if (now - window[-1]["received_monotonic_s"] < .25 and duration > .8 * args.hold_seconds
                            and np.all(np.diff(stamps) > 0) and stable_frames(window, args.min_confidence)):
                        accepted = window
                        frame_id = window[0]["frame_id"]
                        break
                time.sleep(.03)
            if accepted is None:
                raise RuntimeError(f"动作 {name} 超时：没有稳定、高置信度的新EMF帧。最近错误：{latest_errors}")
            result["poses"].append({"name": name, "guide": guide, "frames": accepted})
            write_json(destination, result)
            print(f"已记录 {len(accepted)} 帧", flush=True)
        result["status"] = "capture_complete"
        result["frame_id"] = frame_id
        write_json(destination, result)
        return destination
    finally:
        try:
            if sub is not None:
                sub.close()
            if skeleton_sub is not None:
                skeleton_sub.close()
        finally:
            try:
                glove.disconnect()
            finally:
                if viewer is not None:
                    viewer.close()


def observations(path, side):
    data = json.loads(Path(path).read_text())
    if (data.get("schema_version") != 1 or data.get("units") != "metres"
            or data.get("finger_order") != list(FINGERS) or data.get("side") != side):
        raise ValueError("Capture schema/units/finger order/handedness mismatch")
    if data.get("status") != "capture_complete" or len(data["poses"]) < 6:
        raise ValueError("Capture is incomplete; collect all guided poses before fitting")
    names, p, q, c = [], [], [], []
    frame_ids = set()
    for stage in data["poses"]:
        frames = stage["frames"]
        if not frames:
            raise ValueError("Empty capture stage")
        positions = np.asarray([row["positions_m"] for row in frames])
        quaternions = np.asarray([row["quaternions_xyzw"] for row in frames])
        confidence = np.asarray([row["confidence"] for row in frames])
        if (positions.shape[1:] != (5, 3) or quaternions.shape[1:] != (5, 4)
                or confidence.shape[1:] != (5,) or not all(np.isfinite(x).all() for x in (positions, quaternions, confidence))):
            raise ValueError("Invalid saved EMF observations")
        if np.any(confidence < 0) or np.any(confidence > 1) or np.any(np.linalg.norm(quaternions, axis=-1) < .5):
            raise ValueError("Invalid saved EMF confidence/quaternion")
        frame_ids.update(row["frame_id"] for row in frames)
        names.append(stage["name"])
        p.append(np.median(positions, axis=0))
        q.append(np.stack([Rotation.from_quat(quaternions[:, f]).mean().as_matrix() for f in range(5)]))
        c.append(np.mean(confidence, axis=0))
    if len(frame_ids) != 1 or not next(iter(frame_ids)):
        raise ValueError("Saved observations must share one nonempty EMF transmitter frame")
    if "flat_open" not in names:
        raise ValueError("Missing flat_open initialization pose")
    return data, names, np.asarray(p), np.asarray(q), np.asarray(c)


def rigid_alignment(source, target):
    a, b = source - source.mean(0), target - target.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    r = vt.T @ correction @ u.T
    return r, target.mean(0) - source.mean(0) @ r.T


def fit(args, capture_path):
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    model = PersonalizedMANO(args.mano_model, args.side, args.model_unit).to(args.device)
    targets, measure_sigma = load_measurements(args.measurements)
    data, names, positions, orientations, confidences = observations(capture_path, args.side)
    dtype, device = model.template.dtype, model.template.device
    tensor = lambda x: torch.as_tensor(x, dtype=dtype, device=device)
    param = lambda x: torch.nn.Parameter(tensor(x).clone())
    count = len(names)
    beta = param(np.zeros(model.shapedirs.shape[-1]))
    log_scale, length, radial = param(0.), param(np.zeros(5)), param(np.zeros(5))
    pose = param(np.zeros((count, 15, 3)))
    with torch.no_grad():
        v0, j0 = model.rest_shape(beta, log_scale, length, radial)
    index = names.index("flat_open")
    r, t = rigid_alignment(v0[list(TIPS)].cpu().numpy(), positions[index])
    root = param(Rotation.from_matrix(r).as_rotvec())
    translation = param(t)
    mounting_pos = param(np.zeros((5, 3)))
    mounting_rot = param(Rotation.from_matrix(r.T[None] @ orientations[index]).as_rotvec())
    # Initialize articulation from measured changes in sensor orientation.
    for i in range(count):
        delta = r.T[None] @ orientations[i] @ np.swapaxes(orientations[index], -1, -2) @ r[None]
        aa = Rotation.from_matrix(delta).as_rotvec()
        with torch.no_grad():
            for finger, chain in enumerate(CHAINS):
                for joint in chain:
                    pose[i, joint-1] = tensor(aa[finger] / 3)
    observed_p, observed_r, weights = tensor(positions), tensor(orientations), tensor(confidences)
    flat_ids = [i for i, name in enumerate(names) if name.startswith("flat_")]
    # PCA prior for the 45-D local rotation vector; keep MANO's absolute pose convention.
    basis = model.components[:15]
    distal_ids = [chain[-1] for chain in CHAINS]
    contact_pairs = []
    for i, name in enumerate(names):
        if name.startswith("pinch_"):
            finger = "index" if name == "pinch_index_bent" else name.removeprefix("pinch_")
            if finger in FINGERS[1:]:
                contact_pairs.append((i, FINGERS.index(finger)))
    parameters = [beta, log_scale, length, radial, pose, root, translation, mounting_pos, mounting_rot]
    # Position parameters are in metres; the dimensionless pose/shape learning
    # rate would otherwise produce centimetre-sized translation jumps.
    optimizer = torch.optim.Adam([
        {"params": [beta, log_scale, length, radial, pose, root, mounting_rot]},
        {"params": [translation, mounting_pos], "lr": args.learning_rate * .03},
    ], lr=args.learning_rate)
    best, best_values = float("inf"), None
    started = time.monotonic()

    def evaluate():
        v, j, joint_r = model(beta, log_scale, length, radial, pose)
        rroot = rotmat(root)
        distal = joint_r[:, distal_ids]
        sensor = v[:, list(TIPS)] + (distal @ mounting_pos[None, :, :, None]).squeeze(-1)
        predicted_p = sensor @ rroot.T + translation
        predicted_r = rroot[None, None] @ distal @ rotmat(mounting_rot)[None]
        error = (predicted_p - observed_p) / .002
        # Smooth-L1 downweights occasional EMF distortions.
        pos_loss = (torch.nn.functional.smooth_l1_loss(error, torch.zeros_like(error), reduction="none").sum(-1) * weights).sum() / weights.sum()
        rot_error = (predicted_r - observed_r).square().sum((-1, -2))
        rotation_loss = (rot_error * weights).sum() / weights.sum() / (math.radians(8) ** 2 * 2)
        rest_v, rest_j = model.rest_shape(beta, log_scale, length, radial)
        dims = model.dimensions(rest_v, rest_j)
        measurement_loss = sum(((dims[key] - value) / measure_sigma).square() for key, value in targets.items())
        flat_loss = pose[flat_ids][:, [j-1 for c in CHAINS[1:] for j in c[1:]]].square().mean() / .08**2
        contact_loss = torch.zeros((), dtype=dtype, device=device)
        for i, finger in contact_pairs:
            a, b = v[i, model.pad_0], v[i, getattr(model, f"pad_{finger}")]
            distance = torch.cdist(a[None], b[None]).squeeze(0)
            contact_loss = contact_loss + distance.min().square() / .003**2
        contact_loss = contact_loss / max(1, len(contact_pairs))
        centred = pose.reshape(count, 45) - model.pose_mean
        outside_pca = centred - (centred @ basis.T) @ basis
        prior = (.03 * beta.square().mean() + .1 * (log_scale / .20).square()
                 + .2 * (length / .20).square().mean() + .2 * (radial / .20).square().mean()
                 + .02 * pose.square().mean() + .1 * outside_pca.square().mean()
                 + .2 * (mounting_pos / .012).square().mean())
        total = pos_loss + .3 * rotation_loss + measurement_loss + .2 * flat_loss + .15 * contact_loss + prior
        return total, (v, j, predicted_p, predicted_r, dims, pos_loss, rotation_loss, contact_loss)

    for step in range(args.steps):
        optimizer.zero_grad()
        loss, _ = evaluate()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite fit loss; capture and partial outputs are retained")
        if float(loss.detach()) < best:
            best = float(loss.detach())
            best_values = [p.detach().clone() for p in parameters]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 100.)
        optimizer.step()
        with torch.no_grad():
            beta.clamp_(-3, 3)
            log_scale.clamp_(math.log(.65), math.log(1.5))
            length.clamp_(math.log(.65), math.log(1.5))
            radial.clamp_(math.log(.6), math.log(1.6))
            mounting_pos.clamp_(-.035, .035)
            pose.clamp_(-2.5, 2.5)
        if step % 100 == 0 or step == args.steps-1:
            print(f"fit {step+1}/{args.steps}: loss={float(loss.detach()):.5f}", flush=True)
    with torch.no_grad():
        loss, values = evaluate()
        if float(loss) < best:
            best_values = [p.detach().clone() for p in parameters]
        for p, value in zip(parameters, best_values):
            p.copy_(value)
        loss, (v, j, predicted_p, predicted_r, dims, pl, rl, cl) = evaluate()
        rest_v, rest_j = model.rest_shape(beta, log_scale, length, radial)
        arr = lambda x: x.detach().cpu().numpy()
        position_error_mm = np.linalg.norm(arr(predicted_p) - positions, axis=-1) * 1000
        relative = np.swapaxes(orientations, -1, -2) @ arr(predicted_r)
        orientation_error_deg = Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude().reshape(count, 5) * 180 / math.pi
        dimensions = {key: float(value) for key, value in dims.items()}
        measurement_errors = {key: dimensions[key] - value for key, value in targets.items()}
        position_passed = (float(np.sqrt(np.mean(position_error_mm ** 2))) <= args.max_fit_rmse_mm
                           and float(position_error_mm.max()) <= 3 * args.max_fit_rmse_mm)
        orientation_passed = float(np.sqrt(np.mean(orientation_error_deg ** 2))) <= args.max_fit_orientation_deg
        measurement_passed = all(abs(e) <= 2 * measure_sigma for e in measurement_errors.values())
        contact_gap_mm = float(cl.sqrt()) * 3 if contact_pairs else None
        contact_passed = contact_gap_mm is None or contact_gap_mm <= args.max_pinch_gap_mm
        fit_passed = position_passed and measurement_passed and orientation_passed and contact_passed
        output = args.output
        output.mkdir(parents=True, exist_ok=True)
        arrays = {"betas": arr(beta), "global_scale": arr(log_scale.exp()),
                  "finger_length_scales": arr(length.exp()), "finger_radial_scales": arr(radial.exp()),
                  "poses_axis_angle": arr(pose), "root_axis_angle": arr(root),
                  "translation_m": arr(translation), "sensor_offsets_local_m": arr(mounting_pos),
                  "sensor_rotations_local_axis_angle": arr(mounting_rot),
                  "rest_vertices_m": arr(rest_v), "rest_joints_m": arr(rest_j),
                  "pose_vertices_m": arr(v), "pose_joints_m": arr(j), "faces": model.faces,
                  "tip_vertex_ids": np.asarray(TIPS), "pose_names": np.asarray(names),
                  "finger_order": np.asarray(FINGERS)}
        np.savez_compressed(output / "personalized_mano.npz", **arrays)
        export_obj(output / "personalized_rest.obj", arr(rest_v), model.faces)
        export_obj(output / "personalized_flat.obj", arr(v[index]), model.faces)
        report = {
            "schema_version": 1, "status": "fit_residuals_passed" if fit_passed else "fit_residuals_failed",
            "fit_residuals_passed": fit_passed, "anatomical_identification_verified": False,
            "hardware_validation": "not_performed_by_offline_fit", "side": args.side,
            "model": {"path": str(model.path), "sha256": hashlib.sha256(model.path.read_bytes()).hexdigest(),
                      "source": "https://mano.is.tue.mpg.de/", "official_archive_byte_identity_verified": False,
                      "model_unit": args.model_unit, "extension": "MANO + global/finger longitudinal/radial rest-shape scales"},
            "capture_sha256": hashlib.sha256(Path(capture_path).read_bytes()).hexdigest(),
            "capture_source": data.get("source"), "frame_id": data.get("frame_id"),
            "dimensions_mm": dimensions, "measurement_targets_mm": targets,
            "measurement_errors_mm": measurement_errors,
            "position_rmse_mm": float(np.sqrt(np.mean(position_error_mm ** 2))),
            "position_max_error_mm": float(position_error_mm.max()),
            "orientation_rmse_deg": float(np.sqrt(np.mean(orientation_error_deg ** 2))),
            "declared_pinch_surface_gap_rms_mm": contact_gap_mm,
            "by_pose": {name: {"sensor_position_errors_mm": position_error_mm[i].tolist(),
                               "sensor_orientation_errors_deg": orientation_error_deg[i].tolist()} for i, name in enumerate(names)},
            "parameters": {key: value.tolist() for key, value in arrays.items() if key in {
                "betas", "global_scale", "finger_length_scales", "finger_radial_scales",
                "root_axis_angle", "translation_m", "sensor_offsets_local_m", "sensor_rotations_local_axis_angle"}},
            "fit_seconds": time.monotonic() - started, "optimizer_steps": args.steps,
            "loss": float(loss), "max_fit_rmse_mm": args.max_fit_rmse_mm,
            "max_fit_orientation_deg": args.max_fit_orientation_deg,
            "max_pinch_gap_mm": args.max_pinch_gap_mm,
            "checks": {"sensor_position": position_passed, "sensor_orientation": orientation_passed,
                       "measurements": measurement_passed, "declared_pinch_surface_gap": contact_passed},
            "limitations": ["Regularized shape estimate; motions alone do not uniquely identify anatomy.",
                            "Unmeasured widths remain prior-dependent; depth follows the radial-scale assumption.",
                            "Sensor mounting offsets can trade off against shape; inspect measurement and pose residuals.",
                            "Local fingertip mesh distance is a weak contact cue, not full self-collision validation.",
                            "Extended shape requires this decoder plus all scales; betas alone do not reproduce it.",
                            "OBJ/NPZ outputs inherit the source MANO model licence."],
        }
        write_json(output / "calibration.json", report)
        print(json.dumps({key: report[key] for key in ("status", "position_rmse_mm", "orientation_rmse_deg", "fit_seconds")}, indent=2))
        return report


def export_obj(path, vertices, faces):
    with Path(path).open("w") as stream:
        stream.write("# Personalized MANO; vertex units metres; original model licence applies.\n")
        for v in vertices:
            stream.write("v %.9f %.9f %.9f\n" % tuple(v))
        for face in faces:
            stream.write("f %d %d %d\n" % tuple(face + 1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("init", "capture", "fit", "calibrate", "model-info", "view"))
    parser.add_argument("--output", type=Path, default=Path("output/wuji_mano"))
    parser.add_argument("--mano-model", type=Path, help="Licensed MANO_RIGHT.pkl or MANO_LEFT.pkl")
    parser.add_argument("--model-unit", choices=("m", "mm"), default="m")
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--sn", help="Select a glove by serial number")
    parser.add_argument("--position-scale", type=float, default=1., help="SDK EMF positions are metres; explicit conversion multiplier")
    parser.add_argument("--hold-seconds", type=float, default=1.5)
    parser.add_argument("--timeout", type=float, default=45., help="Timeout per guided pose, seconds")
    parser.add_argument("--min-confidence", type=float, default=.7)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=.015)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-fit-rmse-mm", type=float, default=5.)
    parser.add_argument("--max-fit-orientation-deg", type=float, default=20.)
    parser.add_argument("--max-pinch-gap-mm", type=float, default=5.)
    parser.add_argument("--visualize", action="store_true", help="Open live visualization while capturing/calibrating")
    parser.add_argument("--parameters", type=Path, help="personalized_mano.npz for fixed-shape live MANO tracking")
    parser.add_argument("--host", default="127.0.0.1", help="Visualization bind address")
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--fps", type=float, default=20., help="Visualization/replay rate ceiling")
    parser.add_argument("--ik-steps", type=int, default=3, help="Warm-started articulation updates per visualization frame")
    parser.add_argument("--no-browser", action="store_true", help="Print visualization URL without launching a browser")
    parser.add_argument("--duration", type=float, help="Stop view after this many seconds; default runs until Ctrl+C")
    args = parser.parse_args(argv)
    for key in ("position_scale", "hold_seconds", "timeout", "learning_rate", "max_fit_rmse_mm", "max_fit_orientation_deg", "max_pinch_gap_mm"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be finite and positive")
    if args.hold_seconds < .4 or args.timeout <= args.hold_seconds or not 0 < args.min_confidence <= 1 or args.steps < 1 or args.threads < 1:
        parser.error("Invalid capture/fitting limits")
    if not math.isfinite(args.fps) or not 1 <= args.fps <= 120 or args.ik_steps < 1 or not 1 <= args.port <= 65535:
        parser.error("Invalid visualization rate/IK steps/port")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be finite and positive")
    try:
        if args.parameters is not None and args.mano_model is None:
            parser.error("--parameters requires --mano-model")
        if args.command == "view":
            visualization_module().view(args)
            return 0
        if args.measurements is not None:
            load_measurements(args.measurements)
        if args.command == "init":
            path = args.output / "measurements.json"
            if path.exists():
                raise ValueError(f"Already exists: {path}")
            write_json(path, measurements_template())
            print(f"请填写五指宽度与掌宽（毫米）：{path}")
            return 0
        if args.command in {"fit", "calibrate", "model-info"}:
            if args.mano_model is None:
                parser.error("--mano-model is required; obtain licensed weights at https://mano.is.tue.mpg.de/")
            expected = f"MANO_{args.side.upper()}"
            if args.mano_model.stem.upper().startswith("MANO_") and args.mano_model.stem.upper() != expected:
                raise ValueError("MANO model filename and --side disagree")
            if args.command in {"fit", "calibrate"} and (args.output / "calibration.json").exists():
                raise ValueError("Fit output already exists; choose another --output")
            model = PersonalizedMANO(args.mano_model, args.side, args.model_unit)
            if args.command == "model-info":
                print(json.dumps({"path": str(model.path), "vertices": 778, "faces": len(model.faces),
                                  "shape_components": model.shapedirs.shape[-1], "side": args.side,
                                  "sha256": hashlib.sha256(model.path.read_bytes()).hexdigest()}, indent=2))
                return 0
        if args.command in {"capture", "calibrate"}:
            source = capture(args)
        else:
            source = args.capture
        if args.command in {"fit", "calibrate"}:
            if source is None:
                parser.error("fit requires --capture")
            return 0 if fit(args, source)["fit_residuals_passed"] else 2
        return 0
    except KeyboardInterrupt:
        print("\n已取消；已采集的数据保留。", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
