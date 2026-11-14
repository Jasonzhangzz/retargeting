"""Straight-zero thumb geometry, native IK and fixed MANO binding agree."""
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from moxian_web.native_skeleton import NativeSkeleton
from moxian_web.official_fk import JOINT_NAMES, urdf_reference
from moxian_web.thumb_coordinates import build_thumb_coordinate_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def corrected(tmp_path):
    packages = list((ROOT / '.venv-moxian-ik/lib').glob('python*/site-packages/qnbot_sdk/glove/_private/algorithms/skeleton/*'))
    if len(packages) != 1:
        pytest.skip('local official SDK required')
    package = packages[0]
    source = package / 'share/qnbot_hand/hands/dynamic_openxr_hand/dynamic_openxr_hand_system.yaml'
    path, report = build_thumb_coordinate_config(source.read_text(), package, tmp_path)
    return package, source, path, report


def fk(path, angles):
    joints = ET.parse(path).getroot().findall('joint')
    frames = {'left_wrist': np.eye(4)}
    remaining = [j for j in joints if j.find('child').get('link') != 'left_wrist']
    while remaining:
        previous = len(remaining)
        for j in remaining[:]:
            parent = j.find('parent').get('link')
            if parent not in frames:
                continue
            o = j.find('origin'); t = np.eye(4)
            if o is not None:
                t[:3, 3] = np.fromstring(o.get('xyz', '0 0 0'), sep=' ')
                t[:3, :3] = Rotation.from_euler('xyz', np.fromstring(o.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            if j.find('axis') is not None:
                t[:3, :3] = t[:3, :3] @ Rotation.from_rotvec(np.fromstring(j.find('axis').get('xyz'), sep=' ') * angles.get(j.get('name'), 0.)).as_matrix()
            frames[j.find('child').get('link')] = frames[parent] @ t
            remaining.remove(j)
        assert len(remaining) < previous
    return frames


def bends(frames):
    p = np.array([frames['left_thumb_' + name][:3, 3] for name in ('metacarpal', 'proximal', 'distal', 'tip')])
    d = np.diff(p, axis=0); d /= np.linalg.norm(d, axis=1)[:, None]
    return np.degrees(np.arccos(np.clip(np.sum(d[:-1] * d[1:], axis=1), -1, 1)))


def test_only_thumb_geometry_limits_and_reference_coordinates_change(corrected):
    package, source, path, report = corrected
    vendor = package / 'share/qnbot_hand'
    resource = Path(report['resource_root'])
    relative = Path('hands/dynamic_openxr_hand/resources/pinocchio')
    a, b = yaml.safe_load(source.read_text()), yaml.safe_load(path.read_text())
    a['__resource_root__'] = str(resource)
    for section in ('viewer', 'qnbot_sdk'):
        a[section]['physics_enabled'] = False
    assert a == b  # includes mounts, target mapping, weights, GN, smoothing/reseed
    assert report['input_registration'] is False
    assert report['mode'] == 'thumb_straight_zero'
    assert report['joint_smoother'] and report['open_error_reseed']
    assert report['fixed_input_registration'] is None
    assert set(report['joint_limits']) == {'left_thumb_mcp_flexion', 'left_thumb_ip_flexion'}
    allowed = {'left_thumb_mcp_abduction', 'left_thumb_mcp_flexion', 'left_thumb_ip_flexion'}
    def attributes(element):
        return element.tag, element.attrib, [attributes(child) for child in element]
    for name in ('left.urdf', 'left_skeleton.urdf'):
        before = {j.get('name'): j for j in ET.parse(vendor / relative / name).getroot().findall('joint')}
        after = {j.get('name'): j for j in ET.parse(resource / relative / name).getroot().findall('joint')}
        for joint in before:
            if joint not in allowed:
                assert attributes(before[joint]) == attributes(after[joint])
            for tag in ('parent', 'child', 'axis'):
                old, new = before[joint].find(tag), after[joint].find(tag)
                assert (old.attrib if old is not None else None) == (new.attrib if new is not None else None)
            assert before[joint].find('origin').get('xyz') == after[joint].find('origin').get('xyz')
    for name in ('right.urdf', 'right_skeleton.urdf'):
        assert (vendor / relative / name).read_bytes() == (resource / relative / name).read_bytes()
    metadata = yaml.safe_load((resource / relative / 'metadata.yaml').read_text())
    # Both native and display URDFs must use the same new limits.
    for filename in ('left.urdf', 'left_skeleton.urdf'):
        for j in ET.parse(resource / relative / filename).getroot().findall('joint'):
            if j.get('name') in report['joint_limits']:
                limits = [float(j.find('limit').get(k)) for k in ('lower', 'upper')]
                assert limits == metadata['left']['joint_limits_rad'][j.get('name')]


@pytest.mark.parametrize('axial', [-.6, -.3, 0., .3])
@pytest.mark.parametrize('cmc', [-.3, 0., .7])
def test_zero_flexion_is_straight_even_with_axial_twist(corrected, axial, cmc):
    _, _, _, report = corrected
    path = Path(report['resource_root']) / 'hands/dynamic_openxr_hand/resources/pinocchio/left.urdf'
    f = fk(path, {'left_thumb_mcp_abduction': axial, 'left_thumb_cmc_opposition': cmc})
    assert np.max(bends(f)) < 2e-6


@pytest.mark.parametrize('angle', [10., 35., 65.])
def test_flexion_angles_are_physical_bends_including_signed_direction(corrected, angle):
    _, _, _, report = corrected
    path = Path(report['resource_root']) / 'hands/dynamic_openxr_hand/resources/pinocchio/left.urdf'
    f = fk(path, {'left_thumb_mcp_abduction': -.4, 'left_thumb_mcp_flexion': np.radians(angle), 'left_thumb_ip_flexion': np.radians(angle / 2)})
    np.testing.assert_allclose(bends(f), [angle, angle / 2], atol=2e-6)
    # The parent-to-child flexion has the documented positive local-z sign.
    relative = f['left_thumb_proximal'][:3, :3].T @ f['left_thumb_distal'][:3, :3]
    assert Rotation.from_matrix(relative).as_rotvec()[2] > 0


def test_reference_poses_are_rebased_and_original_resources_untouched(corrected, tmp_path):
    package, source, path, report = corrected
    vendor = package / 'share/qnbot_hand'
    relative = Path('hands/dynamic_openxr_hand/pose_library.yaml')
    before = (vendor / relative).read_bytes()
    old = yaml.safe_load(before)['poses']
    new = yaml.safe_load((Path(report['resource_root']) / relative).read_text())['poses']
    for a, b in zip(old, new):
        for key, value in a['joint_angles_rad'].items():
            if key not in ('thumb_mcp_flexion', 'thumb_ip_flexion'):
                assert b['joint_angles_rad'][key] == value
            elif a['name'] in ('open', 'five_close'):
                assert b['joint_angles_rad'][key] == 0.
            else:
                offset = report['joint_limits']['left_' + key]['old_origin_yaw_rad']
                assert b['joint_angles_rad'][key] == pytest.approx(value + offset)
    path2, report2 = build_thumb_coordinate_config(source.read_text(), package, tmp_path)
    assert path2 == path and report2 == report
    assert (vendor / relative).read_bytes() == before


def test_native_solver_uses_corrected_frames_and_mano_binding_axes_stay_fixed(corrected):
    package, _, path, report = corrected
    resource = Path(report['resource_root'])
    new_urdf = resource / 'hands/dynamic_openxr_hand/resources/pinocchio/left_skeleton.urdf'
    old_urdf = package / 'share/qnbot_hand/hands/dynamic_openxr_hand/resources/pinocchio/left_skeleton.urdf'
    old, new = urdf_reference(old_urdf), urdf_reference(new_urdf)
    # MANO uses a fixed local bone-direction/pad-axis TF. Straight-zero changes
    # joint origins, not either link's local +x bone direction or +y pad axis.
    for pose in (old, new):
        p = np.array(pose['positions_local']); r = Rotation.from_quat(pose['orientations_local_xyzw']).as_matrix()
        for start, end in zip(('thumb_metacarpal', 'thumb_proximal', 'thumb_distal'), ('thumb_proximal', 'thumb_distal', 'thumb_tip')):
            i, j = [pose['names'].index(n) for n in (start, end)]
            d = r[i].T @ (p[j] - p[i]); d /= np.linalg.norm(d)
            np.testing.assert_allclose(d, [1, 0, 0], atol=1e-9)
    h = ROOT / 'output/moxian/ik/9c0dfad82edffc57'
    captures = list(h.glob('glove/captures/*.json'))
    if not captures:
        pytest.skip('retained eight-action capture required')
    frame = json.loads(captures[0].read_text())['payload']['stage_samples'][0]['frames'][0]['payload']
    e = NativeSkeleton(package, path)
    try:
        # Preserve the official smoother; let its public commands settle before
        # comparing them with the native forward-kinematics snapshot.
        for _ in range(160):
            result = e.step({'timestamp_ms': frame['timestamp_ms'], 'fingers': copy.deepcopy(frame['fingertip_local'])})
        pose = result['pose']; q = dict(zip(pose['joint_names'], pose['joint_positions_rad']))
        expected = fk(new_urdf, q)
        r = Rotation.from_quat(pose['orientations_local_xyzw']).as_matrix()
        # The vendor may apply an anthropometric CMC/palm transform. Compare
        # the MCP/IP relative rotations, independent of that upstream layout.
        for parent, child in zip(('thumb_metacarpal', 'thumb_proximal', 'thumb_distal'), ('thumb_proximal', 'thumb_distal', 'thumb_tip')):
            a, b = [pose['names'].index(name) for name in (parent, child)]
            actual = r[a].T @ r[b]
            wanted = expected['left_' + parent][:3, :3].T @ expected['left_' + child][:3, :3]
            np.testing.assert_allclose(actual, wanted, atol=1e-7)
        assert result['continuity']['reused_engine']
        assert result['physics'] == {'enabled': False, 'steps': 0, 'contacts': 0, 'actuators': 0}
        assert not e.input_registration
    finally:
        e.close()
