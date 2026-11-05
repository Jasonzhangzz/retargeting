"""Standalone entrypoint: run only inside .venv-moxian-ik with Python -I.

SDK logs use stdout/stderr; structured events use a separate inherited FD.
"""
from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import math
import os
import queue
import signal
import sys
import threading
import time
import hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moxian_web.native_skeleton import NativeSkeleton
from moxian_web.kinematic_policy import build_kinematic_config


def json_value(value):
    if dataclasses.is_dataclass(value):
        return {f.name: json_value(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-fd", type=int, required=True)
    parser.add_argument("--home", required=True)
    parser.add_argument("--serial", required=True)
    args = parser.parse_args()
    os.environ["QNBOT_SDK_HOME"] = os.path.abspath(args.home)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    event_file = os.fdopen(args.event_fd, "w", buffering=1)
    event_lock = threading.Lock()

    def emit(**message):
        with event_lock:
            event_file.write(json.dumps(json_value(message), allow_nan=False, ensure_ascii=False) + "\n")

    controls = queue.Queue(maxsize=32)
    latest = queue.Queue(maxsize=1)

    def input_reader():
        try:
            for line in sys.stdin:
                message = json.loads(line)
                if message.get("type") == "control":
                    controls.put(message)
                elif message.get("type") == "frame":
                    try:
                        latest.get_nowait()
                    except queue.Empty:
                        pass
                    latest.put_nowait(message)
        except Exception as exc:
            emit(type="state", status="error", error=str(exc))
        finally:
            stop.set()

    threading.Thread(target=input_reader, daemon=True).start()
    sdk = None
    native = None
    native_fingerprint = None
    runtime_policy = None
    subscriptions = []
    try:
        from qnbot_sdk import AlgorithmsConfig, CaptureConfig, CaptureInteractionMode, Sdk, Side, TargetName
        from qnbot_sdk.glove import ExternalConnection, ExternalGloveFrame, GloveConfig, GloveNodePose
        from qnbot_sdk.glove._resources import builtin_skeleton_package_root

        sdk = Sdk(devices=(GloveConfig(name="moxian", side=Side.LEFT, connection=ExternalConnection()),),
                  algorithms=AlgorithmsConfig(capture=CaptureConfig(interaction=CaptureInteractionMode.EXTERNAL,
                                                                    operator_id="local")))
        glove = sdk.glove()
        device = glove.device()
        skeleton = device.skeleton()
        pose_cache = queue.Queue(maxsize=1)

        def on_pose(sample):
            try:
                pose_cache.get_nowait()
            except queue.Empty:
                pass
            pose_cache.put_nowait({**json_value(sample.value), "sequence": sample.sequence})

        subscriptions.append(skeleton.pose().subscribe(on_pose))
        capture_channel = device.capture_progress()
        calibration_channel = device.calibration_progress(name=TargetName.SKELETON)
        control = device.capture_control()
        device.start()
        glove.run_background()
        emit(type="state", status="needs_calibration", error=None, sdk_version="1.7.0")
        last_capture = last_calibration = None
        last_timestamp = -1
        pose_at = 0
        next_pose_emit = 0
        paused = False
        input_frames = 0
        next_input_report = 0
        while not stop.is_set():
            try:
                message = controls.get_nowait()
            except queue.Empty:
                pass
            else:
                try:
                    action = message["action"]
                    progress = capture_channel.latest()
                    stage = progress.value.stage if progress else None
                    if action == "confirm":
                        if not stage or not stage.request_id or message.get("request_id") != stage.request_id:
                            raise ValueError("采集请求已更新，请按网页最新提示操作")
                        control.confirm(stage.request_id)
                    elif action == "cancel":
                        if not stage or message.get("stage_id") != stage.stage_id:
                            raise ValueError("采集阶段已更新，请按最新提示取消")
                        if stage.request_id:
                            if message.get("request_id") != stage.request_id:
                                raise ValueError("采集请求已更新")
                            control.cancel(stage.request_id)
                        # Stop scheduling and input even when the SDK has no
                        # pending confirmation (e.g. collecting a gesture).
                        paused = True
                        glove.request_stop()
                        glove.join()
                        device.stop()
                        value = json_value(progress.value)
                        value["session_state"] = "cancelled"
                        value["stage"].update(state="pending", request_id=None)
                        emit(type="state", status="cancelled", capture=value, pose=None, error=None)
                    elif action == "retry":
                        if not stage or message.get("stage_id") != stage.stage_id:
                            raise ValueError("采集阶段已更新")
                        control.retry(stage.stage_id)
                    else:
                        raise ValueError("未知骨架操作")
                    emit(type="ack", id=message["id"], ok=True)
                except Exception as exc:
                    emit(type="ack", id=message["id"], error=str(exc))
            try:
                message = latest.get(timeout=.01)
            except queue.Empty:
                message = None
            if paused:
                continue
            if message is not None:
                timestamp = float(message["timestamp_ms"])
                if not math.isfinite(timestamp) or timestamp <= last_timestamp:
                    continue
                nodes = {}
                for name in ("thumb", "index", "middle", "ring", "pinky"):
                    node = message["fingers"][name]
                    position, quat = node["position"], node["quaternion_xyzw"]
                    if len(position) != 3 or len(quat) != 4 or not all(math.isfinite(x) for x in (*position, *quat)):
                        raise ValueError("收到无效外部位姿输入")
                    nodes[name] = GloveNodePose(position=tuple(position), quaternion_xyzw=tuple(quat))
                if native is None:
                    device.push_frame(ExternalGloveFrame(**nodes, timestamp_ms=timestamp))
                else:
                    result=native.step(message)
                    if time.monotonic()>=next_pose_emit:
                        nodes_by_name=dict(zip(result['pose']['names'],result['pose']['positions_local']))
                        alignment={f:math.dist(nodes_by_name[f+'_tip'],p)*1000 for f,p in zip(('thumb','index','middle','ring','little'),result['targets']['positions_local'])}
                        emit(type='state',status='ready',pose=result['pose'],targets=result['targets'],physics=result['physics'],
                             alignment={'tip_errors_mm':alignment},runtime_policy=runtime_policy,calibration_fingerprint=native_fingerprint,pose_source='official_native_kinematic',error=None)
                        pose_at=time.monotonic();next_pose_emit=pose_at+1/60
                last_timestamp = timestamp
                input_frames += 1
                if time.monotonic() >= next_input_report:
                    emit(type="state", input_frames=input_frames, input_timestamp_ms=timestamp)
                    next_input_report = time.monotonic() + 1
            progress = capture_channel.latest()
            if progress and progress.sequence != last_capture:
                last_capture = progress.sequence
                value = json_value(progress.value)
                session = value["session_state"]
                status = "error" if session == "failed" else "cancelled" if session == "cancelled" else "needs_calibration" if session != "completed" else "calibrating"
                failure = value.get("failure")
                emit(type="state", status=status, capture=value, error=failure.get("message") if failure else None)
            progress = calibration_channel.latest()
            if progress and progress.sequence != last_calibration:
                last_calibration = progress.sequence
                value = json_value(progress.value)
                job = value["job"]
                failure = job.get("failure")
                status = {"completed": "waiting_pose", "failed": "error", "cancelled": "cancelled"}.get(job["state"], "calibrating")
                emit(type="state", status=status, calibration=value, error=failure.get("message") if failure else None)
            if native is None and time.monotonic() >= next_pose_emit:
                try:
                    pose = pose_cache.get_nowait()
                except queue.Empty:
                    pass
                else:
                    artifacts=list(Path(args.home).glob('glove/calibrations/*/artifacts/hand_system.yaml'))
                    if artifacts:
                        artifact=max(artifacts,key=lambda p:p.stat().st_mtime)
                        text=artifact.read_text();native_fingerprint=hashlib.sha256(text.encode()).hexdigest()
                        runtime=Path(args.home)/'runtime';runtime.mkdir(exist_ok=True)
                        derived,runtime_policy=build_kinematic_config(text,builtin_skeleton_package_root(),runtime)
                        native=NativeSkeleton(builtin_skeleton_package_root(),derived)
                        if native.lib.qh_engine_physics_enabled(native.handle):raise RuntimeError('运动学骨架不允许开启动力学')
                        # Calibration is complete. Stop the unused SDK engine,
                        # including its previously configured physics branch.
                        glove.request_stop();glove.join();device.stop()
                        emit(type='state',status='waiting_pose',pose=None,targets=None,pose_source='official_native_kinematic',calibration_fingerprint=native_fingerprint,error=None)
                    else:
                        emit(type="state", status="ready", pose=pose, targets=None,error=None)
                    pose_at = time.monotonic()
                    next_pose_emit = pose_at + 1 / 60
    except Exception as exc:
        emit(type="state", status="error", error=f"官方 Skeleton 运行时：{exc}")
        raise
    finally:
        if native:native.close()
        if sdk:
            try:
                sdk.request_stop()
                sdk.join()
                for subscription in subscriptions:
                    subscription.unsubscribe()
                sdk.stop()
                sdk.close()
            except Exception as exc:
                print(f"Skeleton cleanup: {exc}", file=sys.stderr)
        event_file.close()


if __name__ == "__main__":
    main()
