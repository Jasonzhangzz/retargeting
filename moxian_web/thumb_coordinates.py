"""Left-thumb straight-zero coordinates for the vendor native IK.

This is an explicit model change, not an output pose correction. The vendor
solver, targets, mounting calibration and temporal settings remain unchanged.
The installed resources and original calibration are never edited.
"""
from pathlib import Path
import hashlib
import json
import math
import re
import shutil
import xml.etree.ElementTree as ET

from .kinematic_policy import build_kinematic_config

REVISION = 'left_thumb_straight_zero_v2'


def build_thumb_coordinate_config(text, package_root, runtime):
    runtime = Path(runtime)
    baseline, original = build_kinematic_config(text, package_root, runtime / 'official-baseline')
    source = Path(original['resource_root'])
    key = hashlib.sha256((REVISION + original['runtime_policy_sha256']).encode()).hexdigest()
    destination = runtime / ('thumb-resources-' + key[:16])
    manifest = destination / 'thumb-coordinates.json'
    if not manifest.exists():
        staging = runtime / ('thumb-resources-' + key[:16] + '.tmp')
        shutil.copytree(source, staging, dirs_exist_ok=True)
        folder = staging / 'hands/dynamic_openxr_hand'
        pinocchio = folder / 'resources/pinocchio'
        joints = {j.get('name'): j for j in ET.parse(pinocchio / 'left.urdf').getroot().findall('joint')}
        axial = joints['left_thumb_mcp_abduction']
        if axial.find('axis').get('xyz') != '1 0 0':
            raise ValueError('不支持的拇指轴向关节定义')
        if joints['left_thumb_mcp_flexion'].find('origin').get('rpy') != '0 0 0':
            raise ValueError('不支持的拇指 MCP 零位')
        offsets = {'left_thumb_mcp_flexion': float(axial.find('origin').get('rpy').split()[2]),
                   'left_thumb_ip_flexion': float(joints['left_thumb_ip_flexion'].find('origin').get('rpy').split()[2])}
        limits = {}
        for name, offset in offsets.items():
            joint = joints[name]
            if joint.find('axis').get('xyz') != '0 0 1':
                raise ValueError('不支持的拇指屈伸轴')
            # q=0 is extension; share the explicit bidirectional thumb range.
            # Preserve the original maximum physical bend, not its raw angle.
            from .thumb_limits import THUMB_EXTENSION_DEG
            limits[name] = [-math.radians(THUMB_EXTENSION_DEG), float(joint.find('limit').get('upper')) + offset]
        for filename in ('left.urdf', 'left_skeleton.urdf'):
            tree = ET.parse(pinocchio / filename)
            for joint in tree.getroot().findall('joint'):
                name = joint.get('name')
                if name in ('left_thumb_mcp_abduction', 'left_thumb_ip_flexion'):
                    # Axial rotation belongs to the incoming straight bone.
                    # Remove the fixed bend AND its tiny fixed roll/pitch.
                    joint.find('origin').set('rpy', '0 0 0')
                if name in limits:
                    joint.find('limit').set('lower', str(limits[name][0]))
                    joint.find('limit').set('upper', str(limits[name][1]))
            ET.indent(tree)
            tree.write(pinocchio / filename, encoding='unicode')
        metadata = pinocchio / 'metadata.yaml'
        content = metadata.read_text()
        for name, bounds in limits.items():
            content, count = re.subn(r'(?m)^(\s*' + name + r':\s*)\[[^\]]+\]',
                                     lambda m: m[1] + json.dumps(bounds), content)
            if count != 1:
                raise ValueError('拇指限位元数据不唯一：' + name)
        metadata.write_text(content)
        library = folder / 'pose_library.yaml'
        content = library.read_text()
        def rebase_pose(match):
            block = match[0]
            flat = match[1] in ('open', 'five_close')
            for joint, offset in offsets.items():
                name = joint.removeprefix('left_')
                block, count = re.subn(r'(?m)^(      ' + name + r':\s*)([-+\d.eE]+)',
                    lambda m: m[1] + str(0. if flat else float(m[2]) + offset), block)
                if count != 1:
                    raise ValueError('姿态库拇指角不唯一：' + name)
            return block
        content, count = re.subn(r'(?m)^  - name: (\S+)\n(?:(?!^  - name:)[\s\S])*', rebase_pose, content)
        if count != 4:
            raise ValueError('官方姿态库结构已改变，停止修改')
        library.write_text(content)
        report = {**original, 'mode': 'thumb_straight_zero', 'revision': REVISION,
                  'runtime_policy_sha256': key, 'local_ik_modifications': True,
                  'resource_root': str(destination.resolve()),
                  'joint_limits_source': 'thumb_straight_zero_other_joints_vendor',
                  'joint_limits': {n: {'q_min_rad': b[0], 'q_max_rad': b[1],
                                       'old_origin_yaw_rad': offsets[n]} for n, b in limits.items()},
                  'exceptions': ['physics_disabled_by_user', 'left_thumb_straight_zero_coordinates'],
                  'coordinate_definition': 'MCP: Rx(axial) Rz(flexion); IP: Rz(flexion); zero=straight',
                  'validation': 'geometry_verified_physical_retest_pending',
                  'baseline_asset_sha256': original['asset_sha256']}
        report['asset_sha256'] = {p: hashlib.sha256((staging / p).read_bytes()).hexdigest() for p in original['asset_sha256']}
        (staging / manifest.name).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        staging.rename(destination)
    report = json.loads(manifest.read_text())
    content, count = re.subn(r'(?m)^__resource_root__:.*$',
                            lambda _: '__resource_root__: ' + json.dumps(str(destination.resolve())), baseline.read_text())
    if count != 1:
        raise ValueError('官方资源路径不唯一')
    report['configuration_sha256'] = hashlib.sha256(content.encode()).hexdigest()
    # Keep the established derived path used by the offline MANO fit reader.
    path = runtime / ('kinematic-' + original['calibration_sha256'][:16] + '.yaml')
    temporary = path.with_suffix('.tmp')
    temporary.write_text(content)
    temporary.replace(path)
    return path, report
