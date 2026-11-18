"""Standalone entrypoint: run only inside .venv-moxian-ik with Python -I.

SDK logs use stdout/stderr; structured events use a separate inherited FD.
"""
from __future__ import annotations

import argparse
from functools import partial
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
import numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moxian_web.native_skeleton import NativeSkeleton
from moxian_web.emf_slsqp import SLSQPSkeleton
from moxian_web.emf_qp import QPSkeleton,REVISION as QP_REVISION
from moxian_web.studio_emf_solver import StudioQPSkeleton,REVISION as STUDIO_QP_REVISION
from moxian_web.wuji_native_ik import WujiNativeSkeleton
from moxian_web.bone_extrinsics import read_active
from moxian_web.thumb_coordinates import build_thumb_coordinate_config
from moxian_web.kinematic_policy import build_kinematic_config as build_official_config


def latest_official_artifact(home):
    return max(Path(home).glob('glove/calibrations/*/artifacts/hand_system.yaml'),
               key=lambda p:p.stat().st_mtime,default=None)


def json_value(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return json_value(value.tolist())
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
    parser.add_argument("--side",choices=("left","right"),default="left")
    args = parser.parse_args()
    build_kinematic_config=build_thumb_coordinate_config if args.side=="left" else partial(build_official_config,side=args.side)
    joint_profile=Path(args.home).parent.parent/'hand_model'/Path(args.home).name/'current.json'
    active_home=joint_profile.parent/'qnbot'
    def joint_revision():
        def signature(path):
            try:
                info=path.stat();return (info.st_ino,info.st_size,info.st_mtime_ns)
            except FileNotFoundError:return None
        return signature(joint_profile),signature(active_home/'active.json')
    def personalized_model(text):
        if not joint_profile.exists():return None
        from moxian_web.hand_model import calibration_report_valid
        profile=json.loads(joint_profile.read_text())
        if (profile.get('serial')!=args.serial or profile.get('side')!=args.side or
            profile.get('calibration_fingerprint')!=hashlib.sha256(text.encode()).hexdigest() or
            not profile.get('calibrated') or not calibration_report_valid(profile.get('report',{}))):return None
        # GPU refinement is calibrated against an immutable upstream IK seed.
        # Its personalized skeleton is emitted alongside the MANO mesh; changing
        # the seed here would make online inference differ from holdout replay.
        return profile.get('skeleton_seed_model') if profile.get('runtime_backend') in ('gpu_mano_v1','fixed_mano_v1') else profile.get('skeleton_model')
    def calibrated_engine(text, package, runtime):
        derived, policy = build_kinematic_config(text, package, runtime)
        selected=read_active(active_home)
        if selected and selected.get('serial')==args.serial and selected.get('side')==args.side and selected.get('calibration_fingerprint')==hashlib.sha256(text.encode()).hexdigest():
            if selected.get('wuji_native'):
                engine=WujiNativeSkeleton(selected)
            else:engine=StudioQPSkeleton(selected)
            return engine,engine.policy(policy)
        derived.write_text(derived.read_text()+'\n__moxian_slsqp__: true\n')
        engine = SLSQPSkeleton(package, derived, side=args.side,personalized=personalized_model(text))
        return engine, engine.policy(policy)
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
        from qnbot_sdk import (AlgorithmsConfig, CaptureConfig, CaptureInteractionMode, Sdk, Side, TargetName,
                               TargetConfig, TargetType, TargetAlgorithm, BuiltinAlgorithm, DeviceSelector)
        from qnbot_sdk.glove import ExternalConnection, ExternalGloveFrame, GloveConfig, GloveNodePose
        from qnbot_sdk.glove._resources import builtin_skeleton_package_root

        sdk_side=Side.LEFT if args.side=="left" else Side.RIGHT
        pose_cache = queue.Queue(maxsize=1)

        def on_pose(sample):
            try:
                pose_cache.get_nowait()
            except queue.Empty:
                pass
            pose_cache.put_nowait({**json_value(sample.value), "sequence": sample.sequence})

        capture_channel=calibration_channel=control=None
        artifact=latest_official_artifact(args.home)
        if artifact:
            # Same native solver and original mounting calibration, with the
            # explicitly reported left-thumb straight-zero model definition.
            text=artifact.read_text();native_fingerprint=hashlib.sha256(text.encode()).hexdigest()
            native,runtime_policy=calibrated_engine(text,builtin_skeleton_package_root(),Path(args.home)/'runtime')
            if native.input_registration:raise RuntimeError('官方模式不允许加载本地输入变换')
            if isinstance(native,SLSQPSkeleton) and native.mapper.lib.qh_engine_physics_enabled(native.mapper.handle):raise RuntimeError('运动学骨架不允许开启动力学')
            emit(type='state',status='waiting_pose',pose=None,targets=None,sdk_version='1.7.0',
                 pose_source='wuji_sdk_native21' if isinstance(native,WujiNativeSkeleton) else STUDIO_QP_REVISION if isinstance(native,StudioQPSkeleton) else QP_REVISION if isinstance(native,QPSkeleton) else 'emf_slsqp_kinematic',runtime_policy=runtime_policy,
                 calibration_fingerprint=native_fingerprint,registration=None,error=None)
        else:
            # Fresh eight-gesture collection stays on the SDK's official
            # capture/calibration flow. Use the same thumb coordinate model
            # during collection/calibration and the realtime native engine.
            package=Path(builtin_skeleton_package_root())
            bootstrap,_=build_kinematic_config((package/'share/qnbot_hand/hands/dynamic_openxr_hand/dynamic_openxr_hand_system.yaml').read_text(),
                                               package,Path(args.home)/'runtime/capture-bootstrap')
            target=TargetConfig(type=TargetType.HAND,name=TargetName.SKELETON,side=sdk_side,
                                source=DeviceSelector(type='glove',name='moxian',side=sdk_side),
                                algorithms=(TargetAlgorithm(id=BuiltinAlgorithm.SKELETON,algorithm_config_path=bootstrap),))
            sdk = Sdk(devices=(GloveConfig(name="moxian", side=sdk_side, connection=ExternalConnection()),),
                      targets=(target,),algorithms=AlgorithmsConfig(capture=CaptureConfig(interaction=CaptureInteractionMode.EXTERNAL,
                                                                                        operator_id="local")))
            glove = sdk.glove();device = glove.device("moxian",side=sdk_side);skeleton = device.skeleton()
            subscriptions.append(skeleton.pose().subscribe(on_pose))
            capture_channel = device.capture_progress();calibration_channel = device.calibration_progress(name=TargetName.SKELETON)
            control = device.capture_control();device.start();glove.run_background()
            emit(type="state", status="needs_calibration", error=None, sdk_version="1.7.0")
        last_capture = last_calibration = None
        last_timestamp = -1
        pose_at = 0
        next_pose_emit = 0
        paused = False
        input_frames = 0
        next_input_report = 0
        capture_signature=None;capture_changed_at=time.monotonic();capture_stall_reported=False
        joint_checked=0.;joint_signature=joint_revision()
        while not stop.is_set():
            if native is not None and time.monotonic()-joint_checked>.5:
                joint_checked=time.monotonic();signature=joint_revision()
                if signature!=joint_signature:
                    artifact=latest_official_artifact(args.home)
                    if artifact:
                        replacement,replacement_policy=calibrated_engine(artifact.read_text(),builtin_skeleton_package_root(),Path(args.home)/'runtime')
                        native.close();native=replacement;runtime_policy=replacement_policy
                        emit(type='state',pose=None,targets=None,status='waiting_pose',runtime_policy=runtime_policy,error=None)
                    joint_signature=signature
            try:
                message = controls.get_nowait()
            except queue.Empty:
                pass
            else:
                try:
                    action = message["action"]
                    if capture_channel is None:raise ValueError('当前使用已保存的官方标定；如需采集请点击重新八动作标定')
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
                if message.get("hand_side",args.side)!=args.side:
                    emit(type="state",status="input_invalid",error="输入帧手侧与标定器不匹配")
                    continue
                timestamp = float(message["timestamp_ms"])
                if not math.isfinite(timestamp) or timestamp <= last_timestamp:
                    continue
                nodes = {}; invalid_input=False
                for name in ("thumb", "index", "middle", "ring", "pinky"):
                    node = message["fingers"][name]
                    position, quat = node["position"], node["quaternion_xyzw"]
                    if len(position) != 3 or len(quat) != 4 or not all(math.isfinite(x) for x in (*position, *quat)):
                        invalid_input=True;break
                    nodes[name] = GloveNodePose(position=tuple(position), quaternion_xyzw=tuple(quat))
                if invalid_input:
                    emit(type="state",status="input_invalid",error="收到无效外部位姿输入，保留上一有效姿态")
                    continue
                if native is None:
                    device.push_frame(ExternalGloveFrame(**nodes, timestamp_ms=timestamp))
                else:
                    try:result=native.step(message)
                    except ValueError as exc:
                        emit(type="state",status="input_invalid",error=str(exc))
                        continue
                    # The compiled Moxian QP is cheaper than the input period.
                    # Publish each consumed latest frame; a relative 60 Hz
                    # gate otherwise drops alternating ~60 Hz source packets.
                    if isinstance(native,StudioQPSkeleton) or time.monotonic()>=next_pose_emit:
                        nodes_by_name=dict(zip(result['pose']['names'],result['pose']['positions_local']))
                        alignment={f:math.dist(nodes_by_name[f+'_tip'],p)*1000 for f,p in zip(('thumb','index','middle','ring','little'),result['targets']['positions_local'])}
                        if native.personalized:
                            alignment={('little' if f=='pinky' else f):r['position_error_mm'] for f,r in result['pose']['tracking_errors'].items()}
                        emit(type='state',status='ready',pose=result['pose'],targets=result['targets'],physics=result['physics'],
                             alignment={'tip_errors_mm':alignment,'joint_limits':result['joint_limit_diagnostics'],'continuity':result['continuity'],
                                        'thumb_unreliable':alignment['thumb']>20 and sum(v['joint'].startswith(args.side+'_thumb_') for v in result['joint_limit_diagnostics'])>=2,
                                        'ik':result['ik_diagnostics']},
                             runtime_policy=runtime_policy,calibration_fingerprint=native_fingerprint,pose_source='wuji_sdk_native21' if isinstance(native,WujiNativeSkeleton) else STUDIO_QP_REVISION if isinstance(native,StudioQPSkeleton) else QP_REVISION if isinstance(native,QPSkeleton) else 'emf_slsqp_kinematic',error=None)
                        pose_at=time.monotonic();next_pose_emit=pose_at+1/60
                last_timestamp = timestamp
                input_frames += 1
                if time.monotonic() >= next_input_report:
                    emit(type="state", input_frames=input_frames, input_timestamp_ms=timestamp)
                    next_input_report = time.monotonic() + 1
            progress = capture_channel.latest() if capture_channel else None
            if progress and progress.sequence != last_capture:
                last_capture = progress.sequence
                value = json_value(progress.value)
                session = value["session_state"]
                status = "error" if session == "failed" else "cancelled" if session == "cancelled" else "needs_calibration" if session != "completed" else "calibrating"
                failure = value.get("failure")
                emit(type="state", status=status, capture=value, error=failure.get("message") if failure else None)
            if progress:
                watch=json_value(progress.value);stage_watch=watch.get('stage') or {}
                signature=(watch.get('session_id'),stage_watch.get('stage_id'),stage_watch.get('state'),stage_watch.get('collected_sample_count'))
                if signature!=capture_signature:
                    capture_signature=signature;capture_changed_at=time.monotonic();capture_stall_reported=False
                if watch.get('session_state')=='running' and stage_watch.get('state')=='collecting' and time.monotonic()-capture_changed_at>5 and not capture_stall_reported:
                    capture_stall_reported=True
                    emit(type='state',status='input_invalid',capture=watch,
                         error=f'采集超过 5 秒没有新增有效样本（标定器 {args.side}，已接收输入 {input_frames} 帧）。请检查手侧、五指定位有效性，或重试当前动作。')
            progress = calibration_channel.latest() if calibration_channel else None
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
                    artifact=latest_official_artifact(args.home)
                    if artifact:
                        text=artifact.read_text();native_fingerprint=hashlib.sha256(text.encode()).hexdigest()
                        runtime=Path(args.home)/'runtime';runtime.mkdir(exist_ok=True)
                        # Use exactly the official calibration result. No
                        # secondary fitting or local registration follows it.
                        glove.request_stop();glove.join();device.stop()
                        native,runtime_policy=calibrated_engine(text,builtin_skeleton_package_root(),runtime)
                        if native.input_registration:raise RuntimeError('官方模式不允许加载本地输入变换')
                        if isinstance(native,SLSQPSkeleton) and native.mapper.lib.qh_engine_physics_enabled(native.mapper.handle):raise RuntimeError('运动学骨架不允许开启动力学')
                        # Calibration is complete. Stop the unused SDK engine,
                        # including its previously configured physics branch.
                        emit(type='state',status='waiting_pose',pose=None,targets=None,pose_source='wuji_sdk_native21' if isinstance(native,WujiNativeSkeleton) else STUDIO_QP_REVISION if isinstance(native,StudioQPSkeleton) else QP_REVISION if isinstance(native,QPSkeleton) else 'emf_slsqp_kinematic',runtime_policy=runtime_policy,calibration_fingerprint=native_fingerprint,registration=None,calibration_input_quality=None,error=None)
                    else:
                        emit(type="state", status="waiting_pose", pose=None, targets=None,error=None)
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
