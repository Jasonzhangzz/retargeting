"""Nonblocking process bridge to the official SDK 1.7 Skeleton runtime."""
from __future__ import annotations

import copy
import json
import math
import os
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path

JOINT_CHAINS = [
    ["wrist", "thumb_metacarpal", "thumb_proximal", "thumb_distal", "thumb_tip"],
    *[["wrist"] + [f"{f}_{s}" for s in ("metacarpal", "proximal", "intermediate", "distal", "tip")]
      for f in ("index", "middle", "ring", "little")],
]
JOINT_NAMES = ["wrist"] + [name for chain in JOINT_CHAINS for name in chain[1:]]


def validate_pose(pose):
    if pose.get("coordinate_frame") != "wrist_local" or pose.get("hand_side") != "left":
        raise ValueError("骨架坐标系或手侧不匹配")
    names, positions, orientations = (pose.get(k, []) for k in ("names", "positions_local", "orientations_local_xyzw"))
    if len(names) != 25 or set(names) != set(JOINT_NAMES) or len(positions) != 25 or len(orientations) != 25:
        raise ValueError("骨架必须包含完整的 25 个命名节点")
    if not all(len(v) == size and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)
               for seq, size in ((positions, 3), (orientations, 4)) for v in seq):
        raise ValueError("骨架含无效坐标")
    if any(abs(sum(x*x for x in q) - 1) > .05 for q in orientations):
        raise ValueError("骨架四元数不是单位四元数")
    if not math.isfinite(pose.get("timestamp_ms", float("nan"))):
        raise ValueError("骨架缺少毫秒时间戳")
    wrist = names.index("wrist")
    if any(abs(x) > 1e-5 for x in positions[wrist]):
        raise ValueError("wrist 必须为局部坐标原点")
    return pose


def raw_input(telemetry):
    """Take pre-filter SDK 6DoF data; never substitute held/invalid fingertips."""
    raw = telemetry.quality.get("input_6dof", {}).get("fingertip_local", {})
    result = {}
    for name in ("thumb", "index", "middle", "ring", "pinky"):
        node = raw.get(name, {})
        if not node.get("valid"):
            return None
        pos, q = node.get("position", ()), node.get("quaternion_wxyz", ())
        if len(pos) != 3 or len(q) != 4 or not all(math.isfinite(x) for x in (*pos, *q)):
            return None
        norm = math.sqrt(sum(x*x for x in q))
        if norm < 1e-8:
            return None
        result[name] = {"position": list(pos), "quaternion_xyzw": [q[i] / norm for i in (1, 2, 3, 0)]}
    timestamp_ms = telemetry.meta.timestamp * 1000.
    if not math.isfinite(timestamp_ms):
        return None
    return {"type": "frame", "timestamp_ms": timestamp_ms, "fingers": result}


class SkeletonBridge:
    def __init__(self, root, folder):
        self.root, self.folder = Path(root), Path(folder)
        self.python = self.root / ".venv-moxian-ik/bin/python"
        self.lock = threading.Lock()
        self.state = {"status": "not_started", "pose": None, "error": None, "capture": None, "calibration": None}
        self.pose_at = 0
        self.proc = None
        self.serial = None
        self.generation = 0
        self.frames = queue.Queue(maxsize=1)
        self.controls = queue.Queue(maxsize=32)
        self.pending = {}
        self.threads = []
        self.log = None

    def start(self, serial):
        if self.proc and self.proc.poll() is None and self.serial == serial:
            return
        self.stop()
        self.serial = serial
        if not self.python.is_file():
            with self.lock:
                self.state.update(status="unavailable", error="请运行 scripts/setup_moxian_ik.sh 安装官方骨骼运行时")
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        self.log = (self.folder / "skeleton-worker.log").open("a")
        read_fd, write_fd = os.pipe()
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        serial_key = __import__("hashlib").sha256(serial.encode()).hexdigest()[:16]
        home = self.folder / "ik" / serial_key
        args = [str(self.python), "-I", str(self.root / "moxian_web/ik_worker.py"),
                "--event-fd", str(write_fd), "--home", str(home), "--serial", serial]
        try:
            proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=self.log, stderr=self.log,
                                    text=True, bufsize=1, pass_fds=(write_fd,), env=env)
        except Exception:
            os.close(read_fd)
            self.log.close()
            self.log = None
            raise
        finally:
            os.close(write_fd)
        self.proc = proc
        with self.lock:
            self.state = {"status": "starting", "error": None, "pose": None, "capture": None, "calibration": None}
        generation = self.generation
        self.threads = [threading.Thread(target=self._read, args=(read_fd, proc, generation), daemon=True),
                        threading.Thread(target=self._write, args=(proc, generation), daemon=True)]
        for thread in self.threads:
            thread.start()

    def send(self, frame):
        if not self.proc or self.proc.poll() is not None:
            return
        try:
            self.frames.put_nowait(frame)
        except queue.Full:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frames.put_nowait(frame)
            except queue.Full:
                pass

    def control(self, action, data):
        if not self.proc or self.proc.poll() is not None:
            raise ValueError("骨骼运行时未启动，请重连手套")
        request_id = uuid.uuid4().hex
        reply = queue.Queue(maxsize=1)
        with self.lock:
            self.pending[request_id] = reply
        try:
            self.controls.put_nowait({"type": "control", "action": action, "id": request_id,
                                      "request_id": data.get("request_id"), "stage_id": data.get("stage_id")})
            result = reply.get(timeout=4)
            if result.get("error"):
                raise ValueError(result["error"])
        except queue.Full:
            raise ValueError("骨架控制队列已满")
        except queue.Empty:
            raise ValueError("骨架控制超时，请等待初始化完成")
        finally:
            with self.lock:
                self.pending.pop(request_id, None)

    def snapshot(self, connected=True):
        with self.lock:
            result = copy.deepcopy(self.state)
            age = (time.monotonic() - self.pose_at) * 1000 if self.pose_at else None
        if self.proc and self.proc.poll() is not None:
            result.update(status="error", error=result.get("error") or f"骨骼进程退出 ({self.proc.returncode})，请重连或重启骨架")
        result["age_ms"] = round(age, 1) if age is not None else None
        result["ready"] = bool(connected and result.get("status") == "ready" and result.get("pose") and age is not None and age < 1000)
        if result.get("pose") and not result["ready"] and result["status"] != "error":
            result["status"] = "stale"
        result["chains"] = JOINT_CHAINS
        result["hand_side"] = "left"
        return result

    def _read(self, fd, proc, generation):
        try:
            with os.fdopen(fd) as stream:
                for line in stream:
                    if generation != self.generation:
                        return
                    message = json.loads(line)
                    if message.get("type") == "ack":
                        with self.lock:
                            reply = self.pending.get(message.get("id"))
                        if reply:
                            reply.put_nowait(message)
                        continue
                    if message.get("pose"):
                        validate_pose(message["pose"])
                    with self.lock:
                        if message.get("pose"):
                            self.pose_at = time.monotonic()
                        self.state.update({k: v for k, v in message.items() if k != "type"})
        except Exception as exc:
            if generation == self.generation:
                with self.lock:
                    self.state.update(status="error", error=str(exc))
        finally:
            if generation == self.generation and proc.poll() is not None:
                with self.lock:
                    self.state.update(status="error", error=f"骨骼进程退出 ({proc.returncode})，详见 skeleton-worker.log")

    def _write(self, proc, generation):
        try:
            while proc.poll() is None and generation == self.generation:
                try:
                    message = self.controls.get_nowait()
                except queue.Empty:
                    try:
                        message = self.frames.get(timeout=.03)
                    except queue.Empty:
                        continue
                proc.stdin.write(json.dumps(message, allow_nan=False) + "\n")
                proc.stdin.flush()
        except (OSError, ValueError):
            pass

    def stop(self):
        self.generation += 1
        proc, self.proc = self.proc, None
        if proc:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
            if proc.stdin:
                proc.stdin.close()
        for thread in self.threads:
            thread.join(timeout=.1)
        self.threads = []
        for q in (self.controls, self.frames):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        if self.log:
            self.log.close()
            self.log = None
        with self.lock:
            self.state.update(status="not_started", pose=None, capture=None, calibration=None)
            self.pose_at = 0
