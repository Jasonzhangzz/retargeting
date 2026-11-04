"""Run with .venv-moxian/bin/python -m moxian_web.server."""
from __future__ import annotations

import argparse
import copy
import json
import logging
import mimetypes
import queue
import signal
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .protocol import ROOT, install_v1_adapter
from .contact import ContactEngine
from .skeleton import SkeletonBridge, raw_input
from .hand_model import HandModelBridge, HandInputSynchronizer
from .contact import PATCH_IDS
from qnbot_sdk.core import DataKeys, QnBotSDK
from qnbot_sdk.core.config import CalibrationConfig, DeviceSpec, QnBotConfig, SerialConnection, Side
from qnbot_sdk.glove import GloveModule
from qnbot_sdk.glove.device import GloveDevice
from qnbot_sdk.glove.haptics import GloveHapticsWriter
from qnbot_sdk.glove.serial_endpoint import PySerialEndpoint
from serial.tools.list_ports import comports

LOG = logging.getLogger("moxian")
STATIC = Path(__file__).parent / "static"
OUTPUT = ROOT / "output" / "moxian"


class PassiveHapticsWriter(GloveHapticsWriter):
    def submit(self, command):
        # Core disconnect submits zero-strength commands even for read-only
        # devices. Accept that bookkeeping without writing to a removed USB.
        if any(command.strengths.values()):
            raise ValueError("此控制台只读，不支持振动输出")

    def clear(self):
        pass


class ReadOnlyGloveModule(GloveModule):
    def _create_device(self, spec, driver):
        writer = PassiveHapticsWriter(source=spec.source, serial_writer=driver)
        return GloveDevice(spec=spec, driver=driver, command_writers=(writer,))


def ports():
    return [{"port": p.device, "serial": p.serial_number or p.device,
             "name": p.description, "vid": f"{p.vid:04x}", "pid": f"{p.pid:04x}",
             "usb_side": "left" if p.pid == 0x5739 else "right"}
            for p in comports() if p.vid is not None and p.pid in (0x5739, 0x5740)]


class Runtime:
    def __init__(self, port=None):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        self.parser = install_v1_adapter()
        self.contact = ContactEngine(OUTPUT)
        self.ik = SkeletonBridge(ROOT, OUTPUT)
        self.hand = HandModelBridge(ROOT, OUTPUT)
        self.last_mano_send = 0.
        self.hand_inputs = HandInputSynchronizer()
        self.preferred_port = port
        self.commands = queue.Queue()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.wanted = True
        self.sdk = None
        self.endpoint = None
        self.device = None
        self.spec = None
        self.state = {"status": "connecting", "error": None, "pose": None, "tactile": None}
        self.last_frame = 0.0
        self.last_seq = None
        self.samples = deque(maxlen=250)
        self.baselines = self._load_baselines()
        self.baseline_samples = None
        self.record_file = None
        self.record_path = None
        self.record_count = 0
        self.record_start = 0
        self.record_model_version = None
        self.thread = threading.Thread(target=self.run, name="moxian-sdk", daemon=True)
        self.thread.start()

    def _load_baselines(self):
        try:
            return json.loads((OUTPUT / "baselines.json").read_text())
        except (OSError, ValueError):
            return {}

    def snapshot(self):
        with self.lock:
            result = dict(self.state)
            age = (time.monotonic() - self.last_frame) * 1000 if self.last_frame else None
            result.update(age_ms=round(age, 1) if age is not None else None,
                          connected=bool(age is not None and age < 1000 and self.sdk),
                          recording={"active": self.record_file is not None,
                                     "frames": self.record_count,
                                     "seconds": round(time.monotonic() - self.record_start, 1) if self.record_file else 0,
                                     "file": self.record_path.name if self.record_path else None},
                          baseline_collecting=self.baseline_samples is not None)
            if result["status"] == "streaming" and age is not None and age >= 1000:
                result["status"] = "stale"
                result["hz"] = 0
            result["contact"] = copy.deepcopy(result.get("contact", {"ready": False, "status": "needs_tare"}))
            contact_time = result["contact"].get("timestamp_monotonic")
            if not result["connected"] or contact_time is not None and time.monotonic() - contact_time >= .25:
                result["contact"].update(ready=False, status="unknown", count=None, active_patches=[], normalized=[0.] * 460)
                for patch in result["contact"].get("patches", []):
                    patch.update(contact=None, response=None)
        result["skeleton"] = self.ik.snapshot(result["connected"])
        result["hand_model"] = self.hand.snapshot(result["connected"])
        return result

    def command(self, name, data=None):
        reply = queue.Queue(maxsize=1)
        self.commands.put((name, data or {}, reply))
        try:
            result = reply.get(timeout=5)
        except queue.Empty:
            raise ValueError("SDK 操作超时，请检查设备状态")
        if isinstance(result, Exception):
            raise ValueError(str(result))
        return result

    def disconnect(self):
        self.hand_inputs = HandInputSynchronizer()
        if hasattr(self, "contact"):
            self.contact.cancel_tare("设备已断开，去皮中止；原配置已保留")
        if hasattr(self, "ik"):
            self.ik.stop()
        if hasattr(self, "hand"):
            self.hand.stop()
        try:
            if self.sdk:
                self.sdk.disconnect()
        except Exception as exc:
            LOG.warning("SDK disconnect during device loss: %s", exc)
        finally:
            if self.endpoint:
                try:
                    self.endpoint.close()
                except Exception as exc:
                    LOG.warning("Serial close: %s", exc)
                self.endpoint = None
        with self.lock:
            self.sdk = None
            self.last_frame = 0
            self.state.update(status="disconnected", pose=None, tactile=None, hz=0)
        self.baseline_samples = None
        self.samples.clear()
        self.last_seq = None
        self.parser.buffer.clear()
        if hasattr(self.parser, "samples"):
            self.parser.samples.clear()

    def apply_command(self, name, data=None):
        data = data or {}
        if name == "connect":
            self.wanted = True
        elif name == "disconnect":
            self.wanted = False
            self.stop_recording()
            self.disconnect()
        elif name == "baseline":
            if not self.snapshot()["connected"]:
                raise ValueError("请先连接有实时数据的手套")
            self.baseline_samples = []
        elif name == "clear_baseline":
            self.baseline_samples = None
            if self.device:
                self.baselines.pop(self.device["serial"], None)
                self.save_baselines()
        elif name == "record_start":
            if not self.snapshot()["connected"]:
                raise ValueError("没有实时数据，无法开始录制")
            if not self.record_file:
                folder = OUTPUT / "recordings"
                folder.mkdir(exist_ok=True)
                self.record_path = folder / (datetime.now(timezone.utc).strftime("glove_%Y%m%dT%H%M%S_%fZ") + ".jsonl")
                self.record_file = self.record_path.open("x")
                self.record_count = 0
                self.record_start = time.monotonic()
                self.record_model_version = None
        elif name == "record_stop":
            self.stop_recording()
        elif name == "mapping_get":
            return self.contact.mapping()
        elif name in ("tare", "clear_tare", "mapping_save", "mapping_restore"):
            if not self.snapshot()["connected"]:
                raise ValueError("请先连接有实时数据的手套")
            if name == "tare":
                # Drop pre-command frames so a 3-second capture begins now.
                self.parser.samples.clear()
                self.contact.start_tare()
            elif name == "clear_tare":
                self.contact.clear_tare()
            elif name == "mapping_save":
                self.contact.set_mapping(data.get("assignments"), data.get("confirmed", []))
            else:
                self.contact.restore_mapping()
        elif name in ("ik_confirm", "ik_cancel", "ik_retry"):
            self.ik.control(name.removeprefix("ik_"), data)
        elif name == "ik_restart":
            if not self.device or not self.sdk:
                raise ValueError("请先连接手套")
            self.ik.stop()
            self.ik.start(self.device["serial"])
        elif name == "mano_restart":
            if not self.device or not self.sdk:
                raise ValueError("请先连接手套")
            self.hand.stop()
            self.hand.start(self.device["serial"])
        elif name == "mano_verify":
            patches = data.get("patch_ids", [])
            if not isinstance(patches, list) or not patches or any(p not in PATCH_IDS for p in patches):
                raise ValueError("请选择待核对分区")
            mapping = self.contact.mapping()
            self.contact.set_mapping(mapping["assignments"], sorted(set(mapping["confirmed"] + patches)))
        elif name.startswith("mano_"):
            if name == "mano_capture_start" and (not self.contact.snapshot()["ready"] or not self.ik.snapshot()["ready"]):
                raise ValueError("请先完成触觉去皮并等待官方骨架在线")
            return self.hand.control(name.removeprefix("mano_"), data)
        else:
            raise ValueError("未知操作")
        return {"ok": True}

    def save_baselines(self):
        dest = OUTPUT / "baselines.json"
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.baselines))
        tmp.replace(dest)

    def stop_recording(self):
        if self.record_file:
            self.record_file.close()
            self.record_file = None

    def connect(self):
        choices = ports()
        if self.preferred_port:
            choices = [p for p in choices if p["port"] == self.preferred_port]
        if not choices:
            raise ValueError("未发现墨现手套，请检查 USB 连接")
        self.device = choices[0]
        self.spec = DeviceSpec.glove(side=Side.LEFT,
                                     connection=SerialConnection(port=self.device["port"], baudrate=115200))
        def serial_factory(spec):
            connection = spec.connection
            self.endpoint = PySerialEndpoint(port=connection.port, baudrate=connection.baudrate, timeout_sec=0.05)
            return self.endpoint
        sdk = QnBotSDK.create(QnBotConfig(devices=(self.spec,), targets=(),
                             calibration=CalibrationConfig(run_if_missing=False)),
                             modules=(ReadOnlyGloveModule(serial_factory=serial_factory),))
        try:
            sdk.connect()
        except Exception:
            try:
                sdk.disconnect()
            except Exception:
                pass
            if self.endpoint:
                self.endpoint.close()
                self.endpoint = None
            raise
        with self.lock:
            self.sdk = sdk
            self.state.update(status="waiting", device=self.device, error=None)
        LOG.info("SDK connected: %s", self.device)
        self.contact.bind(self.device["serial"], "left")
        try:
            self.ik.start(self.device["serial"])
        except Exception as exc:
            with self.ik.lock:
                self.ik.state.update(status="error", error=str(exc))
        try:
            self.hand.start(self.device["serial"])
        except Exception as exc:
            with self.hand.lock:
                self.hand.state.update(status="error", error=str(exc))

    def update(self):
        self.sdk.update()
        for _ in range(1024):
            try:
                sample_time, raw_sample = self.parser.samples.popleft()
            except IndexError:
                break
            self.contact.feed(raw_sample, sample_time)
        buffer = self.sdk.system.buffer
        telemetry = buffer.latest(DataKeys.source_telemetry(self.spec.source))
        tactile = buffer.latest(DataKeys.source_tactile(self.spec.source))
        if telemetry is None or tactile is None or telemetry.meta.sequence == self.last_seq:
            return
        if tactile.get("seq") != telemetry.meta.sequence:
            return  # The SDK receive thread is still publishing this frame.
        self.last_seq = telemetry.meta.sequence
        now = time.monotonic()
        self.samples.append((now, self.parser.frames))
        while len(self.samples) > 2 and self.samples[0][0] < now - 1:
            self.samples.popleft()
        elapsed = now - self.samples[0][0]
        hz = (self.parser.frames - self.samples[0][1]) / elapsed if elapsed > 0.1 else 0
        raw = tactile["matrix"]
        if self.baseline_samples is not None:
            self.baseline_samples.append(list(raw))
            if len(self.baseline_samples) >= 50:
                self.baselines[self.device["serial"]] = {
                    "side": tactile["hand_side"],
                    "values": [sum(col) / len(col) for col in zip(*self.baseline_samples)]}
                self.save_baselines()
                self.baseline_samples = None
        baseline = self.baselines.get(self.device["serial"], {})
        values = baseline.get("values", []) if baseline.get("side") == tactile["hand_side"] else []
        corrected = [max(0, round(v - b)) for v, b in zip(raw, values)] if len(values) == 460 else list(raw)
        pose = telemetry.payload.to_dict()
        ik_frame = raw_input(telemetry) if pose.get("hand_side") == "left" else None
        if ik_frame:
            self.ik.send(ik_frame)
        contact_state = self.contact.snapshot(now=now)
        if ik_frame:
            self.hand_inputs.push({**ik_frame, "hand_side": "left", "raw": list(raw),
                                   "tactile_timestamp_ms": contact_state.get("timestamp_ms", 0),
                                   "contact": contact_state})
        hand_input = None
        if ik_frame and now - self.last_mano_send >= 1/60:
            bone = self.ik.snapshot()
            skeleton = bone.get("pose") if bone.get("ready") else None
            hand_input = self.hand_inputs.match(skeleton, time.time()*1000)
            if hand_input:
                self.last_mano_send = now
                hand_input.update(mapping=self.contact.mapping(),
                                  tare_profile=copy.deepcopy(self.contact.profile.get("tare")),
                                  skeleton_targets=bone.get('targets'),calibration_fingerprint=bone.get('calibration_fingerprint'))
                self.hand.send(hand_input)
        state = {"status": "streaming", "error": None, "device": self.device,
                 "side": tactile["hand_side"], "side_mismatch": tactile["hand_side"] != self.device["usb_side"],
                 "sequence": telemetry.meta.sequence, "frames": self.parser.frames,
                 "hz": round(hz, 1), "crc_errors": self.parser.crc_errors,
                 "timestamp": time.time(), "pose": pose,
                 "contact": contact_state,
                 "tactile": {"raw": list(raw), "values": corrected, "rows": 23, "cols": 20,
                             "has_baseline": len(values) == 460, "unit": "ADC",
                             "peak": max(corrected), "sum": sum(corrected),
                             "active": sum(v > 5 for v in corrected)},
                 "imu": {k: v for k, v in self.parser.extra.get("imu", {}).items()
                         if k in ("acc", "gyro", "valid", "acc_g")}}
        with self.lock:
            self.state = state
            self.last_frame = now
        if self.record_file:
            hand_state = self.hand.snapshot()
            topology = None
            version = (hand_state.get("geometry_version"), hand_state.get("binding_version"))
            if hand_state.get("ready") and version != self.record_model_version:
                topology = self.hand.model()
                if topology.get("ready"):
                    self.record_model_version = version
            self.record_file.write(json.dumps({**state, "skeleton": self.ik.snapshot(),
                                              "hand_model": hand_state, "hand_model_topology": topology, "raw_6dof": ik_frame,
                                              "hand_model_input": hand_input}, separators=(",", ":")) + "\n")
            self.record_count += 1
            if self.record_count % 50 == 0:
                self.record_file.flush()

    def run(self):
        next_retry = 0
        connected_at = 0
        try:
            while not self.stop.is_set():
                try:
                    name, data, reply = self.commands.get_nowait()
                except queue.Empty:
                    pass
                else:
                    try:
                        reply.put(self.apply_command(name, data))
                        next_retry = 0
                    except Exception as exc:
                        reply.put(exc)
                try:
                    if self.wanted and not self.sdk and time.monotonic() >= next_retry:
                        self.connect()
                        connected_at = time.monotonic()
                    if self.sdk:
                        self.update()
                        if time.monotonic() - (self.last_frame or connected_at) > 3:
                            raise ValueError("数据流中断，正在重新连接")
                except Exception as exc:
                    LOG.warning("SDK: %s", exc)
                    self.stop_recording()
                    self.disconnect()
                    with self.lock:
                        self.state.update(status="connecting" if self.wanted else "disconnected", error=str(exc))
                    next_retry = time.monotonic() + 2
                self.stop.wait(0.01)
        finally:
            self.stop_recording()
            self.disconnect()


def handler(runtime):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, status, data, content_type="application/json; charset=utf-8", filename=None):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            route = urlparse(self.path).path
            if route == "/api/state":
                return self.respond(200, runtime.snapshot())
            if route == "/api/ports":
                return self.respond(200, {"ports": ports()})
            if route == "/api/mapping":
                return self.respond(200, runtime.command("mapping_get"))
            if route == "/api/hand_model/model":
                return self.respond(200, runtime.hand.model())
            if route == "/api/recording":
                if runtime.record_file:
                    return self.respond(409, {"error": "请先停止录制，再导出完整文件"})
                path = runtime.record_path
                if not path or not path.exists():
                    return self.respond(404, {"error": "尚无本次会话录制"})
                return self.respond(200, path.read_bytes(), "application/x-ndjson", path.name)
            file = STATIC / {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css",
                             "/gestures/together.png": "gestures/together.png",
                             "/gestures/spread.png": "gestures/spread.png",
                             "/gestures/knuckle-and-fist.png": "gestures/knuckle-and-fist.png"}.get(route, "__missing__")
            extra = {"/mano-view.js": "mano-view.js", "/vendor/three.module.min.js": "vendor/three.module.min.js",
                     "/vendor/OrbitControls.js": "vendor/OrbitControls.js"}
            if route in extra:
                file = STATIC / extra[route]
            if file.is_file():
                return self.respond(200, file.read_bytes(), mimetypes.guess_type(file)[0] or "text/plain")
            self.respond(404, {"error": "Not found"})

        def do_POST(self):
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                return self.respond(403, {"error": "Origin mismatch"})
            if urlparse(self.path).path != "/api/command":
                return self.respond(404, {"error": "Not found"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size < 32768:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(size))
                self.respond(200, runtime.command(body["command"], body))
            except (ValueError, KeyError, TypeError) as exc:
                self.respond(400, {"error": str(exc)})
    return Handler


def main():
    parser = argparse.ArgumentParser(description="墨现手套 SDK + WebUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    # Bind first, so a second invocation never competes for the glove serial port.
    server = ThreadingHTTPServer((args.host, args.port), BaseHTTPRequestHandler)
    runtime = Runtime(args.device)
    server.RequestHandlerClass = handler(runtime)
    LOG.info("WebUI http://%s:%s", args.host, args.port)
    def terminate(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runtime.stop.set()
        runtime.thread.join(timeout=5)
        server.server_close()


if __name__ == "__main__":
    main()
