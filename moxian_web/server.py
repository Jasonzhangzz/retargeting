"""Run with .venv-moxian/bin/python -m moxian_web.server."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import mimetypes
try:
    import orjson
except ImportError:
    orjson=None
import os
import queue
import signal
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,parse_qs

from .protocol import ROOT, install_v1_adapter, bound_live_sdk_history
from .contact import ContactEngine
from .skeleton import SkeletonBridge, OfficialSkeletonBridge, raw_input
from .hand_side import device_side
from .hand_model import HandModelBridge, HandInputSynchronizer
from .contact import PATCH_IDS
from .checkpoints import CheckpointCatalog
from .shared_capture import OfficialSensorRecorder
from .recording import SnapshotRecorder
from .wuji_bridge import WujiBridge
from .contact_prediction_bridge import ContactPredictionBridge
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
        self.official_ik = OfficialSkeletonBridge(ROOT, OUTPUT)
        self.hand = HandModelBridge(ROOT, OUTPUT)
        self.checkpoints = CheckpointCatalog(OUTPUT)
        self.wuji = WujiBridge(ROOT, OUTPUT)
        self.contact_prediction = ContactPredictionBridge(ROOT, OUTPUT)
        self.contact_prediction.start()
        self.last_mano_send = 0.
        self.hand_inputs = HandInputSynchronizer()
        self.shared_recorder = None
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
        self.recorder=SnapshotRecorder(OUTPUT/'recordings')
        self.view_demand={}
        self.software_control_lock=threading.Lock()
        self.thread = threading.Thread(target=self.run, name="moxian-sdk", daemon=True)
        self.thread.start()

    def _load_baselines(self):
        try:
            return json.loads((OUTPUT / "baselines.json").read_text())
        except (OSError, ValueError):
            return {}

    def snapshot(self,view=None,packed_wuji=False):
        with self.lock:
            selected=view if view in ('mano','raw','skeleton','official_skeleton','wuji2') else 'all'
            self.view_demand[selected]=time.monotonic()
            result = dict(self.state)
            age = (time.monotonic() - self.last_frame) * 1000 if self.last_frame else None
            result.update(age_ms=round(age, 1) if age is not None else None,
                          connected=bool(age is not None and age < 1000 and self.sdk),
                          recording=self.recorder.snapshot(),
                          baseline_collecting=self.baseline_samples is not None)
            if result["status"] == "streaming" and age is not None and age >= 1000:
                result["status"] = "stale"
                result["hz"] = 0
            result["contact"] = copy.deepcopy(result.get("contact", {"ready": False, "status": "needs_tare"}))
            contact_time = result["contact"].get("timestamp_monotonic")
            if not result["connected"] or contact_time is not None and time.monotonic() - contact_time >= .25:
                result["contact"].update(ready=False, status="unknown", count=None, active_patches=[], normalized=[0.] * 460, cell_contact=[None] * 460)
                for patch in result["contact"].get("patches", []):
                    patch.update(contact=None, response=None)
        result["skeleton"] = self.ik.snapshot(result["connected"])
        if hasattr(self, 'official_ik'):
            result['skeleton_official'] = self.official_ik.snapshot(result['connected'])
        result["hand_model"] = self.hand.snapshot(result["connected"],mesh_transport=view in ('mano','raw','skeleton','official_skeleton','wuji2'))
        mesh_hash=result['hand_model'].get('model_hash')
        bone_hash=(result['skeleton'].get('pose') or {}).get('model_hash')
        if (mesh_hash or bone_hash) and mesh_hash!=bone_hash:
            result['hand_model'].update(ready=False,status='waiting_model',error=None)
        wuji=self.wuji.snapshot(result["connected"],copy_state=view not in ('mano','raw','skeleton','official_skeleton'),
            mesh_transport=packed_wuji or view in ('mano','raw','skeleton','official_skeleton'))
        if view in ('mano','raw','skeleton','official_skeleton'):
            from .web_state import inactive_wuji
            wuji=inactive_wuji(wuji)
        result["wuji2"] = wuji
        if result["skeleton"].get("status")=="input_invalid":
            result["hand_model"].update(ready=False,status="input_invalid",error=result["skeleton"]["error"])
        if hasattr(self,'contact_prediction'):
            result['contact_prediction']=self.contact_prediction.snapshot(result['connected'] and result['hand_model'].get('ready',False))
        return result

    def command(self, name, data=None):
        # These controls do not own the serial device. Waiting for the MANO
        # worker must never stall delivery of synchronized EMF/ADC packets.
        software={'record_start','record_stop','mano_restart','mano_capture_start','mano_capture_confirm','mano_capture_resume','mano_capture_retry','mano_capture_cancel','mano_checkpoint_preview','mano_checkpoint_end','mano_fit','mano_fit_cancel'}
        software.update(('wuji2_solver_set','wuji2_object_set','wuji2_object_clear','wuji2_mapping_rebuild'))
        if name in software:
            with self.software_control_lock:return self.apply_command(name,data)
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
        if self.shared_recorder:self.shared_recorder.close()
        if hasattr(self, "contact"):
            self.contact.cancel_tare("设备已断开，去皮中止；原配置已保留")
        if hasattr(self, "ik"):
            self.ik.stop()
        if hasattr(self, 'official_ik'):
            self.official_ik.stop()
        if hasattr(self, "hand"):
            self.hand.stop()
        if hasattr(self, "wuji"):
            self.wuji.stop()
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
        if name == 'contact_prediction_reset':
            return self.contact_prediction.reset_metrics()
        elif name == 'contact_prediction_restart':
            self.contact_prediction.stop();self.contact_prediction.start();return {'ok':True}
        elif name == "connect":
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
            with self.lock:connected=bool(self.sdk and self.last_frame and time.monotonic()-self.last_frame<1.)
            if not connected:raise ValueError('没有实时数据，无法开始录制')
            return {'ok':True,'recording':self.recorder.start()}
        elif name == "record_stop":
            return {'ok':True,'recording':self.stop_recording()}
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
            self.ik.start(self.device["serial"],self.device["hand_side"])
        elif name == 'ik_official_restart':
            if not self.device or not self.sdk:
                raise ValueError('请先连接手套')
            self.official_ik.stop()
            self.official_ik.start(self.device['serial'], self.device['hand_side'])
        elif name in ("ik_recalibrate","ik_restore_calibration"):
            if not self.device or not self.sdk:
                raise ValueError("请先连接手套")
            self.ik.recalibrate(self.device["serial"],restore=name=="ik_restore_calibration")
        elif name == "mano_restart":
            if not self.device or not self.sdk:
                raise ValueError("请先连接手套")
            self.hand.stop()
            self.hand.start(self.device["serial"],self.device["hand_side"],surface_backend=data.get('surface_backend'))
        elif name == 'wuji2_restart':
            if not self.device or not self.sdk:raise ValueError('请先连接手套')
            self.wuji.stop();self.wuji.start(self.device['serial'])
        elif name in ('wuji2_object_set','wuji2_object_clear','wuji2_solver_set'):
            return self.wuji.control(name.removeprefix('wuji2_'),data)
        elif name == 'wuji2_mapping_rebuild':
            return self.wuji.control('rebuild',{})
        elif name == "mano_verify":
            patches = data.get("patch_ids", [])
            if not isinstance(patches, list) or not patches or any(p not in PATCH_IDS for p in patches):
                raise ValueError("请选择待核对分区")
            mapping = self.contact.mapping()
            self.contact.set_mapping(mapping["assignments"], sorted(set(mapping["confirmed"] + patches)))
        elif name == 'mano_checkpoint_preview':
            if not self.device:raise ValueError('请先连接手套')
            item,candidate=self.checkpoints.get(data.get('id'),self.device['serial'],self.state['side'])
            from .hand_model import public_fit_report
            candidate['report']=public_fit_report(candidate.get('report'))
            return self.hand.control('checkpoint_preview',{'checkpoint':candidate,'info':item})
        elif name.startswith("mano_"):
            if name in ("mano_apply","mano_preview") and not self.ik.snapshot().get("ready"):
                raise ValueError(self.ik.snapshot().get("error") or "等待有效官方骨架后再拟合或应用手型")
            if name == "mano_capture_start" and not self.ik.snapshot(copy_state=False)["ready"]:
                raise ValueError("请等待官方骨架在线；新协议允许触觉未知")
            return self.hand.control(name.removeprefix("mano_"), data)
        else:
            raise ValueError("未知操作")
        return {"ok": True}

    def save_baselines(self):
        dest = OUTPUT / "baselines.json"
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.baselines))
        tmp.replace(dest)

    def stop_recording(self,wait=False):
        result=self.recorder.stop()
        if wait:self.recorder.close()
        return result

    def connect(self):
        choices = ports()
        if self.preferred_port:
            choices = [p for p in choices if p["port"] == self.preferred_port]
        if not choices:
            raise ValueError("未发现墨现手套，请检查 USB 连接")
        self.device = choices[0]
        self.device["hand_side"]=device_side(self.device)
        self.spec = DeviceSpec.glove(side=Side.LEFT if self.device["hand_side"]=="left" else Side.RIGHT,
                                     connection=SerialConnection(port=self.device["port"], baudrate=115200))
        def serial_factory(spec):
            connection = spec.connection
            self.endpoint = PySerialEndpoint(port=connection.port, baudrate=connection.baudrate, timeout_sec=0.05)
            return self.endpoint
        sdk = QnBotSDK.create(QnBotConfig(devices=(self.spec,), targets=(),
                             calibration=CalibrationConfig(run_if_missing=False)),
                             modules=(ReadOnlyGloveModule(serial_factory=serial_factory),))
        bound_live_sdk_history(sdk.system.buffer)
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
        self.contact.bind(self.device["serial"], self.device["hand_side"])
        key=__import__('hashlib').sha256(self.device['serial'].encode()).hexdigest()[:16]
        self.shared_recorder=OfficialSensorRecorder(OUTPUT/'ik'/key/'shared-captures')
        try:
            self.ik.start(self.device["serial"],self.device["hand_side"])
        except Exception as exc:
            with self.ik.lock:
                self.ik.state.update(status="error", error=str(exc))
        try:
            self.hand.start(self.device["serial"],self.device["hand_side"])
        except Exception as exc:
            with self.hand.lock:
                self.hand.state.update(status="error", error=str(exc))
        self.start_official_reference()
        try:
            self.wuji.start(self.device['serial'])
        except Exception as exc:
            with self.wuji.lock:self.wuji.state.update(status='error',ready=False,error=str(exc))

    def start_official_reference(self):
        if not hasattr(self, 'official_ik'):
            return
        try:
            self.official_ik.start(self.device['serial'], self.device['hand_side'])
        except Exception as exc:
            with self.official_ik.lock:
                self.official_ik.state.update(status='error', error=str(exc))

    def dispatch_hand_input(self):
        # Poll completed IK independently of receiving another serial packet.
        # The synchronizer suppresses duplicate skeleton stamps and preserves
        # the original raw/tactile observation instead of substituting latest.
        bone=self.ik.snapshot();skeleton=bone.get('pose') if bone.get('ready') else None
        hand_input=self.hand_inputs.match(skeleton,time.time()*1000)
        if hand_input:
            self.last_mano_send=time.monotonic()
            hand_input.update(mapping=self.contact.mapping(),tare_profile=copy.deepcopy(self.contact.profile.get('tare')),
                              skeleton_targets=bone.get('targets'),calibration_fingerprint=bone.get('calibration_fingerprint'),
                              source_solver_diagnostics=(bone.get('alignment') or {}).get('ik'),
                              skeleton_runtime_revision=(bone.get('runtime_policy') or {}).get('runtime_policy_sha256'),dispatch_timestamp_ms=time.time()*1000)
            self.hand.send(hand_input)
        return hand_input

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
        reported_side=pose.get("hand_side")
        if reported_side in ("left","right") and reported_side!=self.device["hand_side"]:
            self.device["hand_side"]=reported_side
            self.contact.bind(self.device["serial"],reported_side)
            self.ik.start(self.device["serial"],reported_side)
            self.start_official_reference()
            self.hand.start(self.device["serial"],reported_side)
            self.hand_inputs=HandInputSynchronizer()
        ik_frame = raw_input(telemetry) if reported_side in ("left","right") else None
        if ik_frame:ik_frame["hand_side"]=reported_side
        contact_state = self.contact.snapshot(now=now)
        if ik_frame:
            self.wuji.observe_source_input(ik_frame['timestamp_ms'])
            self.ik.send({**ik_frame,'contact':{k:contact_state.get(k) for k in ('ready','timestamp_ms','patches')},'tactile_timestamp_ms':ik_frame['timestamp_ms']})
            if hasattr(self, 'official_ik'):
                self.official_ik.send(ik_frame)
        if ik_frame:
            observation={**ik_frame, "hand_side": reported_side, "raw": list(raw),
                                   # This raw ADC matrix shares the SDK's
                                   # combined-packet sequence with the pose.
                                   # Detector readiness has its own timestamp;
                                   # no tare must not suppress MANO pose input.
                                   "tactile_timestamp_ms": ik_frame['timestamp_ms'],
                                   "contact": contact_state}
            self.hand_inputs.push(observation)
            with self.ik.lock:capture=copy.deepcopy(self.ik.state.get('capture'))
            if self.shared_recorder:self.shared_recorder.feed(observation,capture,self.contact.mapping(),copy.deepcopy(self.contact.profile.get('tare')))
        hand_input = self.dispatch_hand_input()
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
        if self.recorder.active:
            # Worker outputs replace nested fields rather than mutate published
            # arrays. Keep shallow, same-instant references; encode off-thread.
            hand_state=self.hand.snapshot(copy_state=False,include_audit=False,mesh_transport=True)
            if hand_state.get('surface_backend')=='qnbot' and hand_state.get('ready'):
                hand_state=dict(hand_state,_mesh_f32=self.hand.mesh_frame(sequence=hand_state['mesh_frame']['sequence']))
            with self.hand.lock:topology=self.hand.topology
            with self.wuji.lock:wuji_topology=self.wuji.topology
            self.recorder.submit({**state,'skeleton':self.ik.snapshot(copy_state=False),
                                  'skeleton_official':self.official_ik.snapshot(copy_state=False),
                                  'hand_model':hand_state,'hand_model_topology':topology,'raw_6dof':ik_frame,
                                  'hand_model_input':hand_input,'wuji2':self.wuji.snapshot(copy_state=False),
                                  'contact_prediction':self.contact_prediction.snapshot(copy_state=False),
                                  'wuji2_topology':wuji_topology})

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
                        self.dispatch_hand_input()
                        from .web_state import needs_wuji
                        with self.lock:views=dict(self.view_demand)
                        if needs_wuji(views,time.monotonic(),self.recorder.active):self.wuji.dispatch(self.hand)
                        self.contact_prediction.dispatch(self.hand)
                        if time.monotonic() - (self.last_frame or connected_at) > 3:
                            raise ValueError("数据流中断，正在重新连接")
                except Exception as exc:
                    LOG.warning("SDK: %s", exc)
                    self.stop_recording()
                    self.disconnect()
                    with self.lock:
                        self.state.update(status="connecting" if self.wanted else "disconnected", error=str(exc))
                    next_retry = time.monotonic() + 2
                self.stop.wait(0.005)
        finally:
            self.stop_recording(wait=True)
            self.disconnect()
            self.contact_prediction.stop()

def handler(runtime):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, status, data, content_type="application/json; charset=utf-8", filename=None, fast=False):
            if not isinstance(data, bytes):
                # Worker snapshots have already passed strict finite JSON/pose
                # validation. Serialize their large visual arrays in native
                # code so concurrent legacy tabs cannot monopolize the GIL.
                data = orjson.dumps(data) if fast and orjson is not None else json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
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
                query=parse_qs(urlparse(self.path).query);view=query.get('view',[None])[0]
                return self.respond(200, runtime.snapshot(view=view,packed_wuji=query.get('geometry',[None])[0]=='packed'),fast=True)
            if route == "/api/ports":
                return self.respond(200, {"ports": ports()})
            if route == "/api/diagnostics/pose":
                report = runtime.parser.pose_diagnostics()
                report['connected'] = runtime.snapshot()['connected']
                return self.respond(200, report)
            if route == "/api/mapping":
                return self.respond(200, runtime.command("mapping_get"))
            if route == '/api/hand_model/checkpoints':
                device=runtime.device
                if not device:return self.respond(200,{'items':[],'error':'等待手套连接'})
                side=runtime.state.get('side')
                if side not in ('left','right'):return self.respond(200,{'items':[],'error':'等待手侧初始化'})
                bone=runtime.ik.snapshot();policy=bone.get('runtime_policy') or {}
                frame={'calibration_fingerprint':bone.get('calibration_fingerprint'),'skeleton_runtime_revision':policy.get('runtime_policy_sha256')}
                return self.respond(200,{'items':runtime.checkpoints.list(device['serial'],side,frame)})
            if route == '/api/hand_model/training_trace':
                query=parse_qs(urlparse(self.path).query)
                try:after=max(0,int(query.get('after',['0'])[0]))
                except ValueError:return self.respond(400,{'error':'无效的曲线游标'})
                try:return self.respond(200,runtime.hand.training_history_snapshot(after,query.get('run_id',[None])[0],query.get('method',[None])[0]))
                except ValueError as exc:return self.respond(400,{'error':str(exc)})
            if route == "/api/hand_model/model":
                return self.respond(200, runtime.hand.model())
            if route == '/api/hand_model/mesh':
                query=parse_qs(urlparse(self.path).query)
                try:
                    after=int(query['after'][0]) if 'after' in query else None
                    sequence=int(query['sequence'][0]) if 'sequence' in query else None
                except ValueError:return self.respond(400,{'error':'无效的mesh帧游标'})
                payload=runtime.hand.mesh_frame(after,sequence)
                if payload is None:return self.respond(404,{'error':'尚无官方mesh帧'})
                return self.respond(204 if not payload else 200,payload,'application/octet-stream')
            if route == '/api/wuji2/model':
                return self.respond(200, runtime.wuji.model())
            if route == '/api/hand_model/capture':
                serial=runtime.hand.serial
                if not serial:return self.respond(404,{'error':'尚无标定设备'})
                path=OUTPUT/'hand_model'/hashlib.sha256(serial.encode()).hexdigest()[:16]/'motion-capture.json'
                if runtime.hand.surface_backend=='qnbot':
                    folder=path.parent/'qnbot'
                    protocol=(runtime.hand.snapshot(copy_state=False).get('capture') or {}).get('protocol')
                    native_capture=folder/'wuji-calibration-capture.json'
                    if protocol=='wuji_studio_native_v1' and native_capture.exists():path=native_capture
                    elif (folder/'motion-capture.json').exists():path=folder/'motion-capture.json'
                try:stream=path.open('rb')
                except FileNotFoundError:return self.respond(404,{'error':'标定数据尚在首次保存，请稍候'})
                # Atomic checkpoints let a download retain one consistent
                # inode without allocating another whole-session byte buffer.
                with stream:
                    self.send_response(200);self.send_header('Content-Type','application/json; charset=utf-8')
                    self.send_header('Content-Length',str(os.fstat(stream.fileno()).st_size))
                    self.send_header('Content-Disposition','attachment; filename="motion-capture.json"');self.send_header('Cache-Control','no-store');self.end_headers()
                    try:
                        while chunk:=stream.read(262144):self.wfile.write(chunk)
                    except (BrokenPipeError,ConnectionResetError):pass
                return
            if route == "/api/recording":
                if runtime.recorder.snapshot()['active'] or runtime.recorder.snapshot()['saving']:
                    return self.respond(409, {"error": "请先停止录制，再导出完整文件"})
                path = runtime.recorder.path
                if not path or not path.exists():
                    return self.respond(404, {"error": "尚无本次会话录制"})
                return self.respond(200, path.read_bytes(), "application/x-ndjson", path.name)
            file = STATIC / {"/": "index.html", "/app.js": "app.js", "/motion-guide.js":"motion-guide.js", "/calibration-workflow.js":"calibration-workflow.js", "/style.css": "style.css",
                             "/gestures/together.png": "gestures/together.png",
                             "/gestures/spread.png": "gestures/spread.png",
                             "/gestures/knuckle-and-fist.png": "gestures/knuckle-and-fist.png"}.get(route, "__missing__")
            extra = {"/training-view.js":"training-view.js","/contact-prediction-view.js":"contact-prediction-view.js","/wuji-view.js":"wuji-view.js","/mano-view.js": "mano-view.js", "/taxel-view.js": "taxel-view.js", "/checkpoint-view.js":"checkpoint-view.js", "/curved-surface.js": "curved-surface.js", "/vendor/three.module.min.js": "vendor/three.module.min.js",
                     "/vendor/OrbitControls.js": "vendor/OrbitControls.js"}
            if route in extra:
                file = STATIC / extra[route]
            guides={f'/wuji-calibration-guides/{stage}-{side}.png':STATIC/'gestures/wuji_native'/f'{stage}-{side}.png' for stage in ('pinch_index','pinch_middle','pinch_ring','pinch_pinky','four_finger_bend_90','flat_open') for side in ('left','right')}
            if route in guides:file=guides[route]
            if route == '/contact-prediction.css':file=STATIC/'contact-prediction.css'
            if route == '/calibration-contact-targets.json':file=STATIC/'calibration-contact-targets.json'
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
        runtime.thread.join(timeout=35)
        server.server_close()


if __name__ == "__main__":
    main()
