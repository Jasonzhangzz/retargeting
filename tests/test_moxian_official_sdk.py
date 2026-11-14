"""The comparison must use the public SDK, exact source stamps and no edits."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

import pytest

from moxian_web.skeleton import OfficialSkeletonBridge
from moxian_web.official_sdk import OfficialSDK
from scripts.compare_moxian_ik import compare_frames

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'output/moxian/hand_model/348640923d9e2f3f/shared-capture.json'
HOME=ROOT/'output/moxian/ik/348640923d9e2f3f'
PY17=ROOT/'.venv-moxian-ik/bin/python'
PY18=ROOT/'.venv-moxian-official/bin/python'


def test_reference_rejects_local_derived_calibration_before_loading_sdk(tmp_path):
    artifact=tmp_path/'glove/calibrations/record-test/artifacts/hand_system.yaml'
    artifact.parent.mkdir(parents=True)
    artifact.write_text('__moxian_slsqp__: true\n')
    with pytest.raises(ValueError,match='派生标定'):
        OfficialSDK(tmp_path,tmp_path/'scratch','right')


def test_reference_is_isolated_and_cannot_recalibrate(tmp_path):
    bridge=OfficialSkeletonBridge(ROOT,tmp_path)
    assert bridge.folder==tmp_path/'official-baseline'
    assert bridge.python==PY18
    with pytest.raises(ValueError,match='只读'):bridge.control('confirm',{})
    with pytest.raises(ValueError,match='只读'):bridge.recalibrate('test')


@pytest.mark.skipif(not PY18.is_file(),reason='Official SDK 1.8 environment required')
def test_missing_calibration_waits_without_starting_capture_or_blocking(tmp_path):
    bridge=OfficialSkeletonBridge(ROOT,tmp_path)
    try:
        bridge.start('no-calibration','right');deadline=time.monotonic()+6
        while bridge.snapshot()['status']=='starting' and time.monotonic()<deadline:time.sleep(.02)
        state=bridge.snapshot()
        assert state['status']=='needs_calibration',state
        assert not state['ready'] and state['capture'] is None
        assert not list(tmp_path.glob('ik/**/*'))
        started=time.monotonic()
        for i in range(5000):bridge.send({'type':'frame','timestamp_ms':i})
        assert time.monotonic()-started<1.
    finally:bridge.stop()


@pytest.mark.skipif(not (SOURCE.is_file() and PY17.is_file() and PY18.is_file()),reason='Local raw EMF and official SDKs required')
def test_public_sdk_versions_agree_with_original_native_on_same_raw_frames(tmp_path):
    rows=json.loads(SOURCE.read_text())['rows']
    # Include both flat holds and a pinch, preserving recorded ordering.
    rows=[r for stage in ('together','open','pinch_index') for r in [x for x in rows if x['stage']==stage][:4]]
    raw=tmp_path/'raw.json';raw.write_text(json.dumps({'rows':rows}))
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    before={str(p):digest(p) for p in (HOME/'glove').rglob('*') if p.is_file()}
    outputs={}
    for name,python in [('sdk17',PY17),('sdk18',PY18)]:
        output=tmp_path/(name+'.json')
        subprocess.run([str(python),'-I',str(ROOT/'moxian_web/official_worker.py'),
            '--home',str(tmp_path/name),'--calibration-home',str(HOME),'--side','right',
            '--replay',str(raw),'--output',str(output)],check=True,capture_output=True,timeout=20)
        outputs[name]=json.loads(output.read_text())
        assert outputs[name]['policy']['local_ik_modifications'] is False
        assert outputs[name]['policy']['local_pose_corrections'] is False
        for src,result in zip(rows,outputs[name]['frames']):
            assert result['pose']['timestamp_ms']==src['timestamp_ms']==result['source_timestamp_ms']
            assert result['joint_angles']['timestamp']==src['timestamp_ms']
    output=tmp_path/'native.json'
    subprocess.run([str(PY17),'-I',str(ROOT/'scripts/compare_moxian_ik.py'),'--native-worker','cabi17',
        '--input',str(raw),'--calibration-home',str(HOME),'--side','right','--output',str(output)],
        check=True,capture_output=True,timeout=20)
    native=json.loads(output.read_text())
    for reference in outputs.values():
        diff=compare_frames(reference['frames'],native['frames'])
        assert diff['max_position_coordinate_difference_m']<1e-11
        assert diff['max_joint_difference_rad']<1e-10
        assert diff['max_quaternion_sign_invariant_difference']<1e-10
    assert before=={str(p):digest(p) for p in (HOME/'glove').rglob('*') if p.is_file()}
    assert json.loads(raw.read_text())['rows']==rows
    assert not list((tmp_path/'sdk18').glob('sdk-reference-*'))
    # Out-of-order input must fail explicitly, not return the next available pose.
    raw.write_text(json.dumps({'rows':[rows[0],rows[0]]}))
    failed=subprocess.run([str(PY18),'-I',str(ROOT/'moxian_web/official_worker.py'),
        '--home',str(tmp_path/'bad'),'--calibration-home',str(HOME),'--side','right',
        '--replay',str(raw),'--output',str(tmp_path/'bad.json')],capture_output=True,text=True,timeout=20)
    assert failed.returncode!=0 and 'increasing source timestamps' in failed.stderr
