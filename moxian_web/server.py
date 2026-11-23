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
        from .geort_service import GeoRTService
        from .cart_service import CartService
        self.cart = CartService()
        self.geort = GeoRTService(workspace=True)
        self.hand.frame_observer = self.observe_hand_output
        if self.geort.selected:
            try:
                self.wuji.geort_model_path=str(self.geort.model_folder(self.geort.selected))
                self.wuji.backend='geort'
            except (OSError,ValueError):pass
        if self.cart.selected:
            try:
                self.wuji.cart_model_path=str(self.cart.model_folder(self.cart.selected));self.wuji.backend='cart'
            except (OSError,ValueError):pass
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
        self.geort.on_model_published=self.reload_geort_model
        from .geort_diagnostics_service import DiagnosticService
        self.geort.diagnostics=DiagnosticService(self.geort_diagnostic_input,self.geort.root)
        self.geort_diagnostic_context=None
        from .wuji_sdf_service import CollisionComparisonService
        self.wuji_collision = CollisionComparisonService(self.wuji_collision_input, OUTPUT / 'wuji2_collision')
        self.wuji_collision.configure(False)
        self.thread = threading.Thread(target=self.run, name="moxian-sdk", daemon=True)
        self.thread.start()

    def reload_geort_model(self,key):
        with self.software_control_lock:
            folder=self.geort.model_folder(key);self.wuji.geort_model_path=str(folder)
            if self.wuji.backend=='geort':
                try:self.geort_command('geort_model_select',dict(model=key))
                except (OSError,ValueError) as exc:
                    self.wuji.stop()
                    with self.wuji.lock:self.wuji.state=dict(status='model_incompatible',ready=False,display_available=False,solver_backend='geort',error=str(exc))
                    raise
            self.geort.model_update=dict(model=key,status='loaded' if self.wuji.backend=='geort' else 'ready',run_id=self.geort.published_run)

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
        wuji=self.wuji.snapshot(result["connected"] or self.geort.replay_active or self.cart.replay_active,copy_state=view not in ('mano','raw','skeleton','official_skeleton'),
            mesh_transport=packed_wuji or view in ('mano','raw','skeleton','official_skeleton'))
        if view in ('mano','raw','skeleton','official_skeleton'):
            from .web_state import inactive_wuji
            wuji=inactive_wuji(wuji)
        result["wuji2"] = wuji
        result['wuji_collision'] = self.wuji_collision.snapshot(wuji)
        if result['wuji_collision'].get('status') == 'waiting_input':
            if wuji.get('error'):
                result['wuji_collision']['error'] = '等待有效 Wuji2 姿态：' + str(wuji['error'])
            elif result['skeleton'].get('status') == 'needs_calibration':
                result['wuji_collision']['error'] = '请先完成源骨架初始化；当前没有可供碰撞对照的 Wuji2 姿态'
        result['geort'] = self.geort.snapshot()
        result['cart'] = self.cart.snapshot(result['connected'],include_mesh=view=='mano')
        if (self.geort.replay_active and self.geort.replay_topology) or (self.cart.replay_active and self.cart.replay_topology):
            top=self.cart.replay_topology if self.cart.replay_active else self.geort.replay_topology
            result['hand_model'].update(geometry_version=top['geometry_version'],binding_version=top['binding_version'],status='geort_replay')
        if result["skeleton"].get("status")=="input_invalid":
            result["hand_model"].update(ready=False,status="input_invalid",error=result["skeleton"]["error"])
        return result

    def command(self, name, data=None):
        # These controls do not own the serial device. Waiting for the MANO
        # worker must never stall delivery of synchronized EMF/ADC packets.
        software={'record_start','record_stop','mano_restart','mano_capture_start','mano_capture_confirm','mano_capture_resume','mano_capture_retry','mano_capture_cancel','mano_checkpoint_preview','mano_checkpoint_end','mano_fit','mano_fit_cancel'}
        software.update(('wuji2_solver_set','wuji2_object_set','wuji2_object_clear','wuji2_mapping_rebuild'))
        software.add('wuji2_collision_set')
        if name in software or name.startswith(('geort_','cart_')):
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
        if hasattr(self,'geort'):self.geort.stop_recording('设备断开，已停止录制')
        if hasattr(self,'cart'):self.cart.recorder.stop_recording('设备断开，已停止录制');self.cart.stop_preview()
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
        if hasattr(self, "wuji") and not (getattr(getattr(self,'geort',None),'replay_active',False) or getattr(getattr(self,'cart',None),'replay_active',False)):
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
        if name == 'mano_fit' and data.get('method') not in ('moxian_wuji','wuji_native'):
            raise ValueError('网页仅支持当前六动作标定；旧实验方法已移除')
        if name == 'mano_restart' and data.get('surface_backend') not in (None,'qnbot'):
            raise ValueError('网页仅使用当前 Qnbot 手部模型')
        if name == 'wuji2_collision_set':
            return self.wuji_collision.configure(data.get('enabled'))
        if name.startswith('cart_'):return self.cart_command(name,data)
        if name.startswith('geort_'):
            return self.geort_command(name,data)
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
            if name=='wuji2_solver_set' and data.get('solver')!='cart' and self.cart.replay_active:
                self.cart_command('cart_replay_stop',{})
            if name=='wuji2_solver_set' and data.get('solver')!='geort' and self.geort.replay_active:
                self.geort_command('geort_replay_stop',{})
            if name=='wuji2_solver_set' and data.get('solver')=='cart':
                key=self.cart.selected
                if not key:
                    imported=[m['id'] for m in self.cart.catalog('models') if m.get('imported')]
                    if len(imported)==1:key=imported[0]
                if not key:raise ValueError('请先在 CART 面板选择模型')
                return self.cart_command('cart_model_select',dict(model=key))
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

    def observe_hand_output(self, frame, topology, joints, serial):
        self.geort.observe(frame,topology,joints,serial)
        self.cart.observe(frame,topology,joints,serial)
        return dict(ready=False,status='cart_diff_v5')

    def wuji_collision_input(self):
        from .wuji_sdf_service import comparison_packet
        with self.lock:
            connected = bool(self.sdk and self.last_frame and time.monotonic() - self.last_frame < 1.)
        replay = self.geort.replay_active or self.cart.replay_active
        return comparison_packet(self.wuji.snapshot(connected or replay, copy_state=False, mesh_transport=True))

    def geort_diagnostic_input(self):
        """A sampled paired result, never a fresh Glove mixed with an old robot."""
        import base64,struct
        from retarget.baseline.geort.adapter import digest
        w=self.wuji.snapshot(True,copy_state=False,mesh_transport=True)
        paired=bool(w.get('solver_backend')=='geort' and w.get('ready'))
        playback=w.get('playback') or {}
        if paired:
            frame=w;stamp=w['source_timestamp_ms'];geometry=w['geometry_version'];binding=w['source_binding_version']
            replay=bool(playback.get('active'))
            if replay:
                from .geort_service import safe_id
                folder=self.geort.root/'recordings'/safe_id(playback['id'])
                profile_path=folder/'profile.json';topology=self.geort.replay_topology
            else:profile_path=None;topology=None
        else:
            hand=self.hand.snapshot(copy_state=False,include_audit=False,mesh_transport=True)
            if not hand.get('ready') or not hand.get('calibrated') or hand.get('preview'):return
            stamp=hand['timestamp_ms'];geometry=hand['geometry_version'];binding=hand['binding_version'];replay=False
            payload=self.hand.mesh_frame(sequence=(hand.get('mesh_frame') or {}).get('sequence'))
            if not payload or abs(struct.unpack_from('<d',payload,24)[0]-stamp)>.01:return
            frame=dict(timestamp_ms=stamp,geometry_version=geometry,binding_version=binding,
                       mesh_f32_b64=base64.b64encode(payload).decode('ascii'))
            profile_path=None;topology=None
        if profile_path is None:
            serial=self.hand.serial
            if not serial:return
            profile_path=OUTPUT/'hand_model'/hashlib.sha256(serial.encode()).hexdigest()[:16]/'qnbot/current.json'
        key=(geometry,binding,str(profile_path))
        cached=self.geort_diagnostic_context
        if not cached or cached[0]!=key:
            topology=topology or self.hand.model()
            if topology.get('geometry_version')!=geometry or topology.get('binding_version')!=binding:return
            profile=json.loads(profile_path.read_text())
            context=dict(topology=topology,profile=profile)
            cached=(key,digest([geometry,binding,profile.get('model_hash')]),context)
            self.geort_diagnostic_context=cached
        return dict(context_key=cached[1],context=cached[2],frame=frame,robot=paired,timestamp_ms=stamp,
                    input_kind='replay' if replay else 'live',
                    robot_reason=None if paired else '当前没有有效的GeoRT结果，仅诊断源Glove')

    def cart_command(self,name,data):
        from .geort_service import safe_id
        if name in ('cart_recordings_delete','cart_models_delete'):
            return self.delete_learning_assets(self.cart,'recordings' if name=='cart_recordings_delete' else 'models',
                                               data.get('ids'),'cart')
        if name=='cart_record_start':
            if self.geort.replay_active or self.cart.replay_active:raise ValueError('请先返回实时输入')
            if self.geort.recording.get('active') or self.geort.recording.get('saving'):raise ValueError('GeoRT 正在录制或保存')
            return self.cart.start_recording(data)
        if name=='cart_record_stop':return self.cart.recorder.stop_recording()
        if name in ('cart_stage_set','cart_configure','cart_preview'):
            result=self.cart.configure(data);self.cart.start_preview();return result
        if name=='cart_stage_next':
            raise ValueError('CART 已改为任意动作采集，无需切换子动作')
        if name=='cart_preview_stop':self.cart.stop_preview();return dict(stopped=True)
        if name in ('cart_preprocess','cart_train','cart_evaluate'):
            return self.cart.start_job(name.removeprefix('cart_'),data)
        if name=='cart_cancel':return self.cart.cancel()
        if name=='cart_action_set':
            raise ValueError('CART 可微策略不使用动作类别条件')
        if name=='cart_model_select':
            key=safe_id(data.get('model'));folder=self.cart.model_folder(key)
            config=json.loads((folder/'config.json').read_text())
            if self.cart.recorder.latest and config['source']!=self.cart.recorder.latest[0]['source']:
                raise ValueError('CART 模型与当前实时标定不匹配；可选择配套录制并直接点击 CART 回放')
            if self.cart.recorder.latest and config.get('source_binding_version')!=self.cart.recorder.latest[1].get('binding_version'):
                raise ValueError('CART 模型与当前触觉材料绑定不匹配')
            if not self.wuji.proc or self.wuji.proc.poll() is not None:self.wuji.start(config['source']['serial'])
            self.wuji.control('model_set',dict(model_path=str(folder),solver='cart'))
            self.wuji.backend='cart';self.wuji.source_version=None;self.wuji.cart_model_path=str(folder);self.cart.activate(key)
            return dict(selected_model=key,solver='cart')
        if name=='cart_replay_load':
            from .cart_data import manifest
            key=data.get('model') or self.cart.selected
            if not key:raise ValueError('请先选择 CART 模型')
            folder,meta=manifest(self.cart.root,safe_id(data.get('dataset')))
            model_folder=self.cart.model_folder(safe_id(key))
            config=json.loads((model_folder/'config.json').read_text())
            if config['source']!=meta['source']:raise ValueError('回放与模型源不兼容')
            if config.get('source_binding_version')!=json.loads((folder/'topology.json').read_text()).get('binding_version'):
                raise ValueError('回放与模型的触觉材料绑定不兼容')
            if self.cart.recorder.recording.get('active'):raise ValueError('请先停止录制')
            if self.geort.replay_active:self.geort_command('geort_replay_stop',{})
            if not self.cart.replay_active:self.cart.previous_solver=self.wuji.backend
            if not self.wuji.proc or self.wuji.proc.poll() is not None:self.wuji.start(config['source']['serial'])
            self.wuji.control('model_set',dict(model_path=str(model_folder),solver='cart'))
            self.wuji.cart_model_path=str(model_folder);self.cart.selected=safe_id(key)
            self.wuji.control('replay_load',dict(folder=str(folder)))
            self.cart.replay_active=True;self.cart.replay_topology=json.loads((folder/'topology.json').read_text())
            return dict(loaded=meta['id'],selected_model=key,solver='cart')
        if name in ('cart_replay_pause','cart_replay_seek'):
            if name.endswith('seek') and (isinstance(data.get('index'),bool) or not isinstance(data.get('index'),int) or data['index']<0):raise ValueError('定位帧号必须为非负整数')
            return self.wuji.control(name.removeprefix('cart_'),data)
        if name=='cart_replay_stop':
            self.wuji.control('replay_stop',{});self.cart.replay_active=False;self.cart.replay_topology=None;self.wuji.source_version=None
            previous=getattr(self.cart,'previous_solver','cart')
            if previous!='cart':self.wuji.control('solver_set',dict(solver=previous))
            return dict(stopped=True,solver=previous)
        raise ValueError('未知 CART 命令')

    def geort_command(self, name, data):
        from .geort_service import safe_id
        from retarget.baseline.geort.training import load_recordings
        if name=='geort_diagnostics_set':return self.geort.diagnostics.configure(data.get('enabled'))
        if name=='geort_coverage_select':return self.geort.select_coverage(data.get('datasets',[]))
        if name=='geort_workspace_rebuild':
            self.geort.poll_workspace(force=True);return dict(status='starting')
        if name in ('geort_recordings_delete','geort_models_delete'):
            return self.delete_learning_assets(self.geort,'recordings' if name=='geort_recordings_delete' else 'models',
                                               data.get('ids'),'geort')
        if name=='geort_record_start':
            if self.geort.replay_active or self.cart.replay_active:raise ValueError('请先返回实时输入再录制动作')
            if self.cart.recorder.recording.get('active') or self.cart.recorder.recording.get('saving'):raise ValueError('CART 正在录制或保存')
            return self.geort.start_recording(data.get('label',''))
        if name=='geort_record_stop':return self.geort.stop_recording()
        if name=='geort_train':return self.geort.start_training(data.get('datasets',[]),data.get('epochs',200),data.get('fk_epochs',200),data.get('mode','official'))
        if name=='geort_train_cancel':return self.geort.cancel_training()
        if name=='geort_evaluate':return self.geort.start_evaluation(data.get('model'),data.get('datasets',[]))
        if name=='geort_model_select':
            key=safe_id(data.get('model'));folder=self.geort.model_folder(key)
            config=json.loads((folder/'config.json').read_text())
            if self.geort.latest and not self.geort.replay_active and config['source']!=self.geort.latest[0]['source']:
                raise ValueError('模型与当前 Moxian 设备或标定不兼容')
            if not self.wuji.proc or self.wuji.proc.poll() is not None:
                serial=config['source']['serial'];self.wuji.start(serial)
            self.wuji.control('model_set',dict(model_path=str(folder)))
            self.wuji.backend='geort';self.wuji.source_version=None;self.wuji.geort_model_path=str(folder);self.geort.activate(key)
            return dict(selected=key,solver='geort')
        if name=='geort_replay_load':
            if not self.geort.selected:raise ValueError('请先选择 GeoRT 模型')
            key=safe_id(data.get('dataset'))
            _,source,_=load_recordings(self.geort.root,[key])
            config=json.loads((self.geort.model_folder(self.geort.selected)/'config.json').read_text())
            if source!=config['source']:raise ValueError('回放录制与所选模型不兼容')
            folder=self.geort.root/'recordings'/key
            self.wuji.control('replay_load',dict(folder=str(folder)))
            self.geort.replay_active=True
            self.geort.replay_topology=json.loads((folder/'topology.json').read_text())
            return dict(loaded=key)
        if name in ('geort_replay_pause','geort_replay_seek'):
            if name=='geort_replay_seek' and (isinstance(data.get('index'),bool) or not isinstance(data.get('index'),int) or data['index']<0):
                raise ValueError('回放位置必须为非负整数帧号')
            if name=='geort_replay_pause' and not isinstance(data.get('paused',True),bool):
                raise ValueError('暂停状态必须为布尔值')
            return self.wuji.control(name.removeprefix('geort_'),data)
        if name=='geort_replay_stop':
            self.wuji.control('replay_stop',{})
            self.geort.replay_active=False;self.geort.replay_topology=None;self.wuji.source_version=None
            return dict(stopped=True)
        raise ValueError('未知 GeoRT 操作')

    def delete_learning_assets(self,service,kind,ids,solver):
        def unload():
            if kind!='models' or service.selected not in ids:return
            if self.wuji.backend==solver:
                self.wuji.stop();self.wuji.topology=None;self.wuji.source_version=None
                with self.wuji.lock:
                    self.wuji.state=dict(status='model_missing',ready=False,display_available=False,
                                         solver_backend=solver,error='已删除当前模型，请选择或训练新模型')
            setattr(self.wuji,'cart_model_path' if solver=='cart' else 'geort_model_path','')
            service.replay_active=False;service.replay_topology=None
        return service.delete_assets(kind,ids,before_delete=unload)

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
                        if not (self.geort.replay_active or self.cart.replay_active) and needs_wuji(views,time.monotonic(),self.recorder.active):self.wuji.dispatch(self.hand)
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
            self.wuji_collision.close()
            self.geort.close()
            self.cart.close()
            self.wuji.stop()

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
            if route in ('/api/wuji_collision/model', '/api/wuji_collision/examples', '/api/wuji_collision/example'):
                try:
                    kind = route.rsplit('/', 1)[-1]
                    index = int(parse_qs(urlparse(self.path).query).get('index', ['0'])[0]) if kind == 'example' else None
                    return self.respond(200, runtime.wuji_collision.visual_asset(kind, index))
                except (ValueError, OSError) as exc:
                    return self.respond(400, {'error':str(exc)})
            if route in ('/api/geort/recordings','/api/geort/models'):
                return self.respond(200,{'items':runtime.geort.catalog(route.rsplit('/',1)[-1])})
            if route=='/api/geort/coverage':
                finger=parse_qs(urlparse(self.path).query).get('finger',['thumb'])[0]
                if finger not in ('thumb','index','middle','ring','pinky'):return self.respond(400,{'error':'无效手指'})
                return self.respond(200,runtime.geort.coverage_snapshot(include_cloud=True,finger=finger))
            if route in ('/api/geort/workspace/pose','/api/geort/workspace/model'):
                query=parse_qs(urlparse(self.path).query)
                try:
                    sample_id=int(query.get('sample_id',[''])[0]) if route.endswith('/pose') else None
                    return self.respond(200,runtime.geort.workspace_preview(
                        query.get('reference_id',[''])[0],sample_id,query.get('finger',['thumb'])[0]))
                except (ValueError,OSError) as exc:return self.respond(400,{'error':str(exc)})
            if route in ('/api/cart/recordings','/api/cart/models'):
                return self.respond(200,{'items':runtime.cart.catalog(route.rsplit('/',1)[-1])})
            if route=='/api/cart/preview':
                return self.respond(200,runtime.cart.preview_packet())
            if route=='/api/cart/preview/topology':
                return self.respond(200,runtime.cart.preview_topology or {'ready':False})
            if route=='/api/cart/ik/viewer/close':
                runtime.cart.ik_view.close()
                return self.respond(200,{'closed':True})
            if route in ('/api/cart/ik/viewer/topology','/api/cart/ik/viewer/frame'):
                query=parse_qs(urlparse(self.path).query)
                try:
                    index=int(query.get('index',['0'])[0]) if route.endswith('/frame') else None
                    return self.respond(200,runtime.cart.ik_view.request(query.get('id',[''])[0],index),fast=True)
                except (OSError,ValueError) as exc:return self.respond(400,{'error':str(exc)})
            if route in ('/api/cart/recording','/api/cart/ik','/api/cart/geometry'):
                from .geort_service import safe_id
                try:
                    key=safe_id(parse_qs(urlparse(self.path).query).get('id',[None])[0])
                    folder=runtime.cart.root/'recordings'/key
                    if not folder.exists():folder=runtime.geort.root/'recordings'/key
                    meta=json.loads((folder/'manifest.json').read_text())
                    if meta.get('capture_mode') not in ('cart','cart_v2','cart_v3','cart_v4','cart_v5'):return self.respond(404,{'error':'不是 CART 录制'})
                    if not meta.get('complete'):return self.respond(409,{'error':'请先停止录制并等待保存完成'})
                    ik=route.endswith('/ik');geometry=route.endswith('/geometry')
                    path=folder/'cart_capture.jsonl' if geometry else folder/'cart_ik.npz' if ik else folder/('cart_targets.jsonl' if (folder/'cart_targets.jsonl').exists() else 'frames.jsonl')
                    if not path.exists():return self.respond(404,{'error':'请等待 wrist 几何保存完成，或点击准备观测补算' if geometry else '请先准备观测、触点与 IK'})
                    with path.open('rb') as stream:
                        self.send_response(200);self.send_header('Content-Type','application/octet-stream' if ik else 'application/x-ndjson')
                        self.send_header('Content-Length',str(path.stat().st_size));self.send_header('Cache-Control','no-store')
                        suffix='_wrist_geometry.jsonl' if geometry else '_ik.npz' if ik else '.jsonl'
                        self.send_header('Content-Disposition',f'attachment; filename="cart_{key}{suffix}"');self.end_headers()
                        import shutil
                        shutil.copyfileobj(stream,self.wfile,64*1024)
                    return
                except (OSError,ValueError) as exc:return self.respond(400,{'error':str(exc)})
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
                return self.respond(200, runtime.cart.replay_topology if runtime.cart.replay_active else runtime.geort.replay_topology if runtime.geort.replay_active else runtime.hand.model())
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
            extra = {"/geort-workspace-fk.js":"geort-workspace-fk.js","/geort-workspace-pose.js":"geort-workspace-pose.js","/cart-preview.js":"cart-preview.js","/geort-coverage.js":"geort-coverage.js","/geort.js":"geort.js","/cart.js":"cart.js","/cart-overlay.js":"cart-overlay.js","/training-view.js":"training-view.js","/wuji-view.js":"wuji-view.js","/mano-view.js": "mano-view.js", "/taxel-view.js": "taxel-view.js", "/checkpoint-view.js":"checkpoint-view.js", "/curved-surface.js": "curved-surface.js", "/vendor/three.module.min.js": "vendor/three.module.min.js",
                     "/vendor/OrbitControls.js": "vendor/OrbitControls.js"}
            if route in extra:
                file = STATIC / extra[route]
            guides={f'/wuji-calibration-guides/{stage}-{side}.png':STATIC/'gestures/wuji_native'/f'{stage}-{side}.png' for stage in ('pinch_index','pinch_middle','pinch_ring','pinch_pinky','four_finger_bend_90','flat_open') for side in ('left','right')}
            if route in guides:file=guides[route]
            if route == '/calibration-contact-targets.json':file=STATIC/'calibration-contact-targets.json'
            if route == '/cart-ik-view.js':file=STATIC/'cart-ik-view.js'
            if route == '/wuji-collision.js':file=STATIC/'wuji-collision.js'
            if route == '/wuji-collision-view.js':file=STATIC/'wuji-collision-view.js'
            if route == '/wuji-collision-colors.js':file=STATIC/'wuji-collision-colors.js'
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
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.respond(400, {"error": str(exc)})
    return Handler


def main():
    parser = argparse.ArgumentParser(description="墨现手套 SDK + WebUI")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址；默认允许局域网访问")
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
