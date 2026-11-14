"""The preserved official-baseline builder must exactly match the vendor IK.

The live thumb-coordinate exception is tested separately.
"""
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from moxian_web.kinematic_policy import build_kinematic_config
from moxian_web.native_skeleton import NativeSkeleton
from moxian_web.skeleton import SkeletonBridge

ROOT=Path(__file__).resolve().parents[1]


def vendor_package():
    packages=list((ROOT/'.venv-moxian-ik/lib').glob('python*/site-packages/qnbot_sdk/glove/_private/algorithms/skeleton/*'))
    if len(packages)!=1:pytest.skip('installed official SDK needed')
    return packages[0]


def vendor_config(package):
    return package/'share/qnbot_hand/hands/dynamic_openxr_hand/dynamic_openxr_hand_system.yaml'


def test_live_policy_preserves_all_ik_options_and_ignores_even_malformed_local_registrations(tmp_path):
    package=vendor_package();source=vendor_config(package);original=source.read_bytes()
    (tmp_path/'ik-registration.json').write_text('malformed historical local fit')
    (tmp_path/'thumb-registration.json').write_text('malformed historical thumb fit')
    assets=package/'share/qnbot_hand/hands/dynamic_openxr_hand/resources/pinocchio'
    before={p.name:p.read_bytes() for p in assets.iterdir() if p.is_file()}
    path,report=build_kinematic_config(source.read_text(),package,tmp_path)
    expected=yaml.safe_load(original);actual=yaml.safe_load(path.read_text())
    for section in ('viewer','qnbot_sdk'):expected[section]['physics_enabled']=False
    expected['__resource_root__']=str((package/'share/qnbot_hand').resolve())
    assert actual==expected  # all weights, mounts, smoother, reseed, limits and pinch policy
    assert '__moxian_input_registration__' not in actual
    assert report['mode']=='official' and report['local_ik_modifications'] is False
    assert report['input_registration'] is False and report['joint_limits_source']=='vendor_original'
    assert report['joint_smoother']==expected['joint_smoother']['enabled']
    assert report['open_error_reseed']==expected['solver']['error_reseed']['enabled']
    assert source.read_bytes()==original
    assert {p.name:p.read_bytes() for p in assets.iterdir() if p.is_file()}==before
    assert not list(tmp_path.glob('resources-*'))
    path2,report2=build_kinematic_config(source.read_text(),package,tmp_path)
    assert path2==path and report2==report


def test_live_policy_rejects_already_modified_input_instead_of_silently_reusing_it(tmp_path):
    package=vendor_package()
    with pytest.raises(ValueError,match='原始标定文件'):
        build_kinematic_config(vendor_config(package).read_text()+'\n__moxian_input_registration__: {}\n',package,tmp_path)


def test_official_wrapper_matches_direct_vendor_ik_on_same_raw_sequence(tmp_path):
    package=vendor_package();home=ROOT/'output/moxian/ik/9c0dfad82edffc57'
    artifacts=list(home.glob('glove/calibrations/*/artifacts/hand_system.yaml'))
    if not artifacts:pytest.skip('retained official calibration required')
    source=max(artifacts,key=lambda p:p.stat().st_mtime);text=source.read_text()
    config=yaml.safe_load(text)
    for section in ('viewer','qnbot_sdk'):config[section]['physics_enabled']=False
    config['__resource_root__']=str((package/'share/qnbot_hand').resolve())
    direct=tmp_path/'direct-official.yaml';direct.write_text(yaml.safe_dump(config,sort_keys=False))
    path,report=build_kinematic_config(text,package,tmp_path/'runtime')
    doc=json.loads(next(home.glob('glove/captures/*.json')).read_text())
    engines=[NativeSkeleton(package,direct),NativeSkeleton(package,path)]
    count=0
    try:
        assert not engines[1].input_registration
        for group in doc['payload']['stage_samples']:
            for sample in group['frames']:
                payload=sample['payload'];frame={'timestamp_ms':payload['timestamp_ms'],'fingers':copy.deepcopy(payload['fingertip_local'])}
                a,b=[engine.step(frame) for engine in engines]
                np.testing.assert_allclose(a['pose']['joint_positions_rad'],b['pose']['joint_positions_rad'],atol=1e-11,rtol=0)
                np.testing.assert_allclose(a['pose']['positions_local'],b['pose']['positions_local'],atol=1e-11,rtol=0)
                np.testing.assert_allclose(a['pose']['orientations_local_xyzw'],b['pose']['orientations_local_xyzw'],atol=1e-11,rtol=0)
                np.testing.assert_allclose(a['targets']['positions_local'],b['targets']['positions_local'],atol=1e-11,rtol=0)
                assert b['physics']=={'enabled':False,'steps':0,'contacts':0,'actuators':0}
                assert b['pose']['timestamp_ms']==frame['timestamp_ms'] and b['pose']['hand_side']=='left'
                count+=1
        assert count>=240
    finally:
        for engine in engines:engine.close()


def test_new_capture_uses_sdk_eight_gesture_flow_without_local_fit(tmp_path):
    vendor_package()
    bridge=SkeletonBridge(ROOT,tmp_path);bridge.start('isolated-official-capture')
    try:
        deadline=time.monotonic()+8;state={};i=0
        while time.monotonic()<deadline:
            bridge.send({'type':'frame','timestamp_ms':1000+i*16,'fingers':{f:{'position':[.03,.02,.1],'quaternion_xyzw':[0.,0.,0.,1.]}for f in ('thumb','index','middle','ring','pinky')}})
            i+=1;time.sleep(.025);state=bridge.snapshot()
            if (state.get('capture') or {}).get('stage',{}).get('state')=='awaiting_confirmation':break
            if state['status']=='error':break
        assert state['status']=='needs_calibration',state
        assert state['capture']['stage']['state']=='awaiting_confirmation',state
        assert state['capture']['stage']['stage_id']=='close'
        assert not state.get('registration') and not state.get('ready')
    finally:bridge.stop()
