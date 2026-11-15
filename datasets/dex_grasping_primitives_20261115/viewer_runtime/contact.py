"""Per-device tactile tare, editable anatomical patches and contact hysteresis."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from pathlib import Path

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
LABELS = ("拇指", "食指", "中指", "无名指", "小指")
SEGMENTS = {"distal": "末端", "intermediate": "中段", "proximal": "近端"}
PATCHES = [{"id": f"{f}_{part}", "label": f"{label} · {SEGMENTS[part]}"}
           for f, label in zip(FINGERS, LABELS)
           for part in (("distal", "proximal") if f == "thumb" else ("distal", "intermediate", "proximal"))]
PATCHES += [{"id": f"palm_{r}_{c}", "label": f"掌心 · {('腕侧', '中央', '指根侧')[r]}{('拇指侧', '小指侧')[c]}"}
            for r in range(3) for c in range(2)]
PATCH_IDS = [p["id"] for p in PATCHES]
EMA_TAU = .040
DETECTOR = {"version": "ema-light-touch-v2", "ema_tau_ms": EMA_TAU * 1000,
            "enter_adc": 1.5, "enter_sigma": 1.5, "exit_adc": .6, "exit_sigma": .75,
            "enter_hold_ms": 30, "exit_hold_ms": 80}


def default_mapping(side):
    result = []
    for sid in range(460):
        row, col = divmod(sid, 20)
        physical_col = col if side == "left" else 19 - col
        finger = FINGERS[physical_col // 4]
        if row >= 15:
            part = "distal" if row >= 19 else "proximal" if row <= 16 or finger == "thumb" else "intermediate"
            patch = f"{finger}_{part}"
        else:
            patch = f"palm_{row // 5}_{physical_col // 10}"
        result.append(PATCH_IDS.index(patch))
    return result


def validate_mapping(assignments, confirmed):
    if not isinstance(assignments, list) or len(assignments) != 460:
        raise ValueError("映射必须包含全部 460 个 SID")
    if any(type(v) is not int or not 0 <= v < 20 for v in assignments):
        raise ValueError("patch 索引必须为 0–19")
    if set(assignments) != set(range(20)):
        raise ValueError("20 个 patch 均须保留至少一个单元")
    if not isinstance(confirmed, list) or any(p not in PATCH_IDS for p in confirmed):
        raise ValueError("确认列表包含未知 patch")
    return list(assignments), sorted(set(confirmed))


def valid_tare(tare):
    return (isinstance(tare, dict) and all(isinstance(tare.get(k), list) and len(tare[k]) == 460
            and all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in tare[k])
            for k in ("baseline", "sigma")))


class ContactEngine:
    def __init__(self, folder: Path):
        self.path = folder / "contact_profiles.json"
        try:
            self.profiles = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.profiles = {}
        if not isinstance(self.profiles, dict):
            self.profiles = {}
        self.key = None
        self.profile = {}
        self.collection = None
        self.tare_status = {"state": "idle", "progress": 0, "error": None}
        self.reset_detection()

    def reset_detection(self):
        self.active = [False] * 20
        self.pending = [None] * 20
        self.latest = None
        self.latest_time = None
        self.filtered = [0.] * 460
        self.filter_time = None

    def bind(self, serial, side):
        key = f"{serial}:{side}"
        if key == self.key:
            return
        self.cancel_tare("设备或手侧改变，去皮采集中止")
        self.key = key
        self.profile = self.profiles.setdefault(key, {})
        if not isinstance(self.profile, dict):
            self.profile = self.profiles[key] = {}
        try:
            validate_mapping(self.profile["assignments"], self.profile.get("confirmed", []))
        except (KeyError, ValueError):
            self.profile["assignments"] = default_mapping(side)
            self.profile["confirmed"] = []
        self.rebuild_mapping()
        self.reset_detection()

    def rebuild_mapping(self):
        self.groups = [[i for i, p in enumerate(self.profile["assignments"]) if p == patch] for patch in range(20)]
        self.neighbors = []
        for group in self.groups:
            sids = set(group)
            self.neighbors.append([(s, n) for s in group for n in (s + 20, s + 1)
                                   if n in sids and (n == s + 20 or s // 20 == n // 20)])
        self.version = hashlib.sha256(json.dumps([self.profile["assignments"], self.profile.get("confirmed", [])]).encode()).hexdigest()[:12]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.profiles))
        tmp.replace(self.path)

    def mapping(self):
        if not self.key:
            return {"ready": False, "patches": PATCHES}
        return {"ready": True, "version": self.version, "patches": PATCHES,
                "assignments": list(self.profile["assignments"]), "confirmed": list(self.profile.get("confirmed", [])),
                "note": "初始近端和掌心分区为示意，请逐区触碰核对 SID"}

    def set_mapping(self, assignments, confirmed):
        if not self.key:
            raise ValueError("请先连接设备")
        assignments, confirmed = validate_mapping(assignments, confirmed)
        self.profile.update(assignments=assignments, confirmed=confirmed)
        self.rebuild_mapping()
        self.reset_detection()
        self.save()

    def restore_mapping(self):
        if not self.key:
            raise ValueError("请先连接设备")
        self.set_mapping(default_mapping(self.key.rsplit(":", 1)[1]), [])

    def start_tare(self, now=None):
        if not self.key:
            raise ValueError("请先连接设备")
        self.collection = {"start": time.monotonic() if now is None else now, "samples": []}
        self.tare_status = {"state": "collecting", "progress": 0, "error": None, "samples": 0}
        self.reset_detection()

    def cancel_tare(self, reason="去皮采集中止"):
        if self.collection is not None:
            self.collection = None
            self.tare_status = {"state": "failed", "progress": 0, "error": reason}
        self.reset_detection()

    def clear_tare(self):
        self.collection = None
        self.profile.pop("tare", None)
        self.tare_status = {"state": "idle", "progress": 0, "error": None}
        self.reset_detection()
        self.save()

    def feed(self, raw, now):
        if len(raw) != 460 or not self.key:
            return
        if self.latest_time is not None and now - self.latest_time > 0.25:
            self.cancel_tare("数据流间断，请重新去皮")
        self.latest_time = now
        if self.collection is not None:
            elapsed = now - self.collection["start"]
            if elapsed >= 0:
                self.collection["samples"].append(list(raw))
            samples = self.collection["samples"]
            self.tare_status.update(progress=min(1, elapsed / 3), samples=len(samples))
            if elapsed >= 3:
                if len(samples) < 240:
                    self.cancel_tare("3 秒内有效帧不足 240，已保留原去皮配置")
                else:
                    columns = list(zip(*samples))
                    med = [statistics.median(v) for v in columns]
                    sigma = [max(1., 1.4826 * statistics.median(abs(x - m) for x in v)) for m, v in zip(med, columns)]
                    self.profile["tare"] = {"baseline": med, "sigma": sigma, "samples": len(samples), "timestamp": time.time()}
                    self.collection = None
                    self.tare_status = {"state": "ready", "progress": 1, "error": None, "samples": len(samples)}
                    self.reset_detection()
                    self.latest_time = now
                    self.save()
            if self.collection is not None:
                return
        tare = self.profile.get("tare")
        if not valid_tare(tare):
            return
        sigma = tare["sigma"]
        # Smooth signed residuals before rectification: rectifying first would
        # turn zero-mean baseline noise into a positive bias at low thresholds.
        dt = 1 / 120 if self.filter_time is None else max(0., now - self.filter_time)
        alpha = -math.expm1(-dt / EMA_TAU)
        self.filtered = [old + alpha * (float(v) - b - old)
                         for old, v, b in zip(self.filtered, raw, tare["baseline"])]
        self.filter_time = now
        delta = [max(v, 0.) for v in self.filtered]
        normalized = [round(min(d / max(12., 6 * s), 1.), 4) for d, s in zip(delta, sigma)]
        enter = [d >= max(DETECTOR["enter_adc"], DETECTOR["enter_sigma"] * s) for d, s in zip(delta, sigma)]
        leave = [d >= max(DETECTOR["exit_adc"], DETECTOR["exit_sigma"] * s) for d, s in zip(delta, sigma)]
        patches = []
        cells = [False] * 460
        for i, definition in enumerate(PATCHES):
            mask = leave if self.active[i] else enter
            supported = any(mask[a] and mask[b] for a, b in self.neighbors[i])
            if supported == self.active[i]:
                self.pending[i] = None
            else:
                if self.pending[i] is None:
                    self.pending[i] = now
                hold = DETECTOR["exit_hold_ms" if self.active[i] else "enter_hold_ms"] / 1000
                if now - self.pending[i] >= hold - 1e-9:
                    self.active[i] = supported
                    self.pending[i] = None
            patches.append({**definition, "contact": self.active[i],
                            "response": max(normalized[s] for s in self.groups[i]),
                            "active_cells": sum(mask[s] for s in self.groups[i])})
            # Only supported neighboring cells in an active patch turn yellow.
            # An isolated spike must not paint a taxel as confirmed contact.
            if self.active[i]:
                for a, b in self.neighbors[i]:
                    if leave[a] and leave[b]:cells[a] = cells[b] = True
        self.latest = {"ready": True, "status": "ready", "normalized": normalized, "patches": patches,
                       "cell_contact": cells,
                       "active_patches": [p["id"] for p in patches if p["contact"]],
                       "count": sum(self.active), "timestamp_monotonic": now, "timestamp_ms": time.time() * 1000}

    def snapshot(self, live=True, now=None):
        now = time.monotonic() if now is None else now
        result = dict(self.latest or {})
        ready = bool(live and self.latest and self.latest_time is not None and now - self.latest_time < .25 and self.collection is None)
        result.update(ready=ready, status="ready" if ready else "collecting" if self.collection is not None else "needs_tare" if not valid_tare(self.profile.get("tare")) else "unknown",
                      tare=dict(self.tare_status), mapping_version=getattr(self, "version", None),
                      mapping_confirmed=len(self.profile.get("confirmed", [])), detector=dict(DETECTOR))
        if not ready:
            result.update(count=None, active_patches=[], patches=[{**p, "contact": None, "response": None} for p in PATCHES], normalized=[0.] * 460, cell_contact=[None] * 460)
        return result
