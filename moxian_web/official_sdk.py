"""Unmodified QnBot public Python SDK skeleton output for comparison.

Runs in a separate process/interpreter. Uses ExternalConnection only, copies
the existing official capture/calibration into a private SDK home, and reads
Skeleton.pose()/joint_angles() without local pose, joint or MANO corrections.
"""
import dataclasses
import enum
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import tempfile
import time
from importlib.metadata import version


def plain(value):
    if dataclasses.is_dataclass(value):return {f.name:plain(getattr(value,f.name)) for f in dataclasses.fields(value)}
    if isinstance(value,enum.Enum):return value.value
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return value


def calibration_artifact(home):
    return max(Path(home).glob('glove/calibrations/*/artifacts/hand_system.yaml'),
               key=lambda p:p.stat().st_mtime,default=None)


def calibration_digest(home):
    artifact=calibration_artifact(home)
    return hashlib.sha256(artifact.read_bytes()).hexdigest() if artifact else None


class OfficialSDK:
    def __init__(self, calibration_home, scratch, side):
        self.side=side;self.samples=queue.Queue();self.sdk=None
        source=Path(calibration_home)
        artifact=calibration_artifact(source)
        if artifact is None:raise ValueError('官方对照需要已保存的官方八动作标定')
        if '__moxian_' in artifact.read_text():
            raise ValueError('官方对照拒绝含本地修正标记的派生标定；必须使用官方原始 artifact')
        Path(scratch).mkdir(parents=True,exist_ok=True)
        self.private=tempfile.TemporaryDirectory(prefix='sdk-reference-',dir=scratch)
        home=Path(self.private.name);shutil.copytree(source/'glove',home/'glove')
        copied=home/artifact.relative_to(source)
        if copied.read_bytes()!=artifact.read_bytes():raise ValueError('Official calibration copy mismatch')
        os.environ['QNBOT_SDK_HOME']=str(home)
        from qnbot_sdk import (Sdk,Side,TargetName,TargetType,TargetConfig,TargetAlgorithm,
            BuiltinAlgorithm,DeviceSelector,AlgorithmsConfig,CaptureConfig,CaptureInteractionMode)
        from qnbot_sdk.glove import GloveConfig,ExternalConnection
        sdk_side=Side.LEFT if side=='left' else Side.RIGHT
        target=TargetConfig(type=TargetType.HAND,name=TargetName.SKELETON,side=sdk_side,
            source=DeviceSelector(type='glove',name='moxian',side=sdk_side),
            algorithms=(TargetAlgorithm(id=BuiltinAlgorithm.SKELETON,algorithm_config_path=copied),))
        self.sdk=Sdk(devices=(GloveConfig(name='moxian',side=sdk_side,connection=ExternalConnection()),),
            targets=(target,),algorithms=AlgorithmsConfig(capture=CaptureConfig(interaction=CaptureInteractionMode.EXTERNAL,operator_id='local')))
        self.glove=self.sdk.glove();self.device=self.glove.device('moxian',side=sdk_side);self.skeleton=self.device.skeleton()
        self.subscription=self.skeleton.pose().subscribe(self.samples.put)
        self.device.start();self.glove.run_background()
        from qnbot_sdk.glove._resources import builtin_skeleton_package_root
        package=Path(builtin_skeleton_package_root())
        resource_root=package/'share/qnbot_hand'
        for line in artifact.read_text().splitlines():
            if line.startswith('__resource_root__:'):
                value=line.split(':',1)[1].strip()
                resource_root=Path(json.loads(value) if value.startswith('"') else value)
        self.policy={'mode':'official_sdk','revision':'qnbot_public_sdk_reference_v1','sdk_version':version('qnbot-sdk-glove'),
            'core_version':version('qnbot-sdk-core'),'entrypoint':'GloveDevice.push_frame -> Skeleton.pose / joint_angles',
            'configuration':'byte_identical_saved_official_calibration','calibration_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),
            'calibration_source':str(artifact.resolve()),'calibration_resource_root':str(resource_root),
            'local_ik_modifications':False,'local_pose_corrections':False,'local_MANO_conversion':False,
            'native_library_sha256':hashlib.sha256((package/'lib/libqnbot_hand.so').read_bytes()).hexdigest(),
            'assets_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (resource_root/'hands/dynamic_openxr_hand/resources/pinocchio').glob(side+'*.urdf')}}
        self.previous_stamp=None

    def step(self, frame, timeout=2.):
        from qnbot_sdk.glove import ExternalGloveFrame,GloveNodePose
        if frame.get('hand_side',self.side)!=self.side:raise ValueError('Official reference hand side mismatch')
        stamp=float(frame['timestamp_ms'])
        if self.previous_stamp is not None and stamp<=self.previous_stamp:raise ValueError('Official reference needs increasing source timestamps')
        nodes={f:GloveNodePose(position=tuple(n['position']),quaternion_xyzw=tuple(n['quaternion_xyzw'])) for f,n in frame['fingers'].items()}
        began=time.perf_counter();self.device.push_frame(ExternalGloveFrame(**nodes,timestamp_ms=stamp))
        deadline=time.monotonic()+timeout
        while True:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('官方 SDK 未返回同帧骨架；检查官方标定是否兼容')
            try:sample=self.samples.get(timeout=remaining)
            except queue.Empty as exc:raise TimeoutError('官方 SDK 未返回同帧骨架；检查官方标定是否兼容') from exc
            pose=plain(sample.value)
            if abs(pose['timestamp_ms']-stamp)<.001:break
            if pose['timestamp_ms']>stamp:raise ValueError('Official SDK skipped requested source frame')
        joint=self.skeleton.joint_angles().latest()
        angles=plain(joint.value) if joint else None
        if angles is not None and abs(angles.get('timestamp_ms',angles.get('timestamp',-1))-stamp)>=.001:angles=None
        self.previous_stamp=stamp
        return {'pose':pose,'joint_angles':angles,'processing_ms':(time.perf_counter()-began)*1000,
                'sdk_sequence':sample.sequence,'source_timestamp_ms':stamp}

    def close(self):
        try:
            if self.sdk:
                self.glove.request_stop();self.glove.join();self.subscription.unsubscribe();self.sdk.close()
        finally:self.private.cleanup()
