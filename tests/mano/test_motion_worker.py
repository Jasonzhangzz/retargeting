"""Exercise the actual IPC worker using retained observations, without USB."""
import copy
import hashlib
import json
import shutil
import time
from pathlib import Path

import pytest
import torch

from moxian_web.hand_model import HandModelBridge,DEFAULT_MANO


def test_motion_controls_record_raw_data_and_cancel_without_applying(tmp_path):
    root=Path(__file__).resolve().parents[2];serial='2DDDF3AC0DA5';key=hashlib.sha256(serial.encode()).hexdigest()[:16]
    retained=root/'output/moxian/hand_model'/key/'fit-wrist-targets.json'
    if not torch.cuda.is_available() or not retained.exists() or not DEFAULT_MANO.with_name('MANO_RIGHT.pkl').exists():pytest.skip('Local retained observations / licensed MANO / CUDA required')
    row=json.loads(retained.read_text())['rows'][0];device=root/'output/moxian/ik'/key
    matching=next((p.parent.parent.parent.parent for p in device.glob('**/glove/calibrations/*/artifacts/hand_system.yaml')
                   if hashlib.sha256(p.read_bytes()).hexdigest()==row['calibration_fingerprint']),None)
    if matching is None:pytest.skip('Matching retained official calibration required')
    shutil.copytree(matching,tmp_path/'ik'/key/'glove')
    shutil.copytree(device/'runtime',tmp_path/'ik'/key/'runtime')
    bridge=HandModelBridge(root,tmp_path)
    def send(unknown_tactile=False):
        frame=copy.deepcopy(row);stamp=time.time()*1000
        frame.update(type='frame',timestamp_ms=stamp,tactile_timestamp_ms=stamp)
        frame['skeleton']['timestamp_ms']=stamp;frame['skeleton_targets']['timestamp_ms']=stamp
        frame['contact']['timestamp_ms']=stamp
        if unknown_tactile:frame.update(tactile_timestamp_ms=None,raw=None,contact={'ready':False,'status':'unknown','patches':[]})
        bridge.send(frame)
    def wait_capture(state,source=None):
        deadline=time.monotonic()+45
        while time.monotonic()<deadline:
            value=bridge.snapshot().get('capture') or {}
            if value.get('state')==state and (source is None or value.get('source')==source):return value
            time.sleep(.01)
        pytest.fail('Worker did not publish expected capture state: '+str(bridge.snapshot().get('capture')))
    try:
        bridge.start(serial,side='right');deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            send();time.sleep(.04)
            if bridge.snapshot().get('ready'):break
        assert bridge.snapshot().get('ready'),bridge.snapshot().get('error')
        began=time.perf_counter();bridge.control('capture_start',{});assert time.perf_counter()-began<.5
        snapshot=wait_capture('ready','continuous_motion')
        began=time.perf_counter();bridge.control('capture_confirm',dict(request_id=snapshot['request_id']));assert time.perf_counter()-began<.5
        for _ in range(30):send();time.sleep(.02)
        for _ in range(5):send(unknown_tactile=True);time.sleep(.02)
        # Keep sampling over an entire window while checkpoints encode in the
        # background. Slow saves must not manufacture a 250ms input gap.
        deadline=time.monotonic()+4.5
        while time.monotonic()<deadline:send();time.sleep(.02)
        current=wait_capture('paused')
        assert '断流' not in (current.get('error') or '') and '缓冲' not in (current.get('error') or '')
        assert current['samples']>=340
        deadline=time.monotonic()+2
        while bridge.snapshot()['capture']['samples']<260 and time.monotonic()<deadline:time.sleep(.03)
        assert bridge.snapshot()['capture']['samples']>=260,bridge.snapshot()['capture']
        bridge.control('capture_cancel',dict(stage_id=snapshot['stage_id']))
        wait_capture('cancelled')
        bridge.control('capture_retry',dict(stage_id=snapshot['stage_id']))
        snapshot=wait_capture('ready')
        assert snapshot['samples']>=260
        assert snapshot['protocol']=='contact_motion_v3' and snapshot['actions_per_pass']==17
        assert snapshot['detected_repetitions']==0
        home=tmp_path/'hand_model'/key
        deadline=time.monotonic()+8
        while time.monotonic()<deadline:
            try:
                document=json.loads((home/'motion-capture.json').read_text())
                if document['state']['state']=='ready' and all(not r.get('training_eligible',True) for r in document['rows'] if not r.get('initialization')):break
            except (FileNotFoundError,ValueError):pass
            time.sleep(.03)
        assert document['protocol']=='contact_motion_v3'
        assert all(not r.get('training_eligible',True) for r in document['rows'] if not r.get('initialization'))
        assert (home/'contact-motion-v3.json').exists()
        assert any(r.get('tactile_timestamp_ms') is None and not r.get('initialization') for r in document['rows'])
        assert not (home/'current.json').exists() and not (home/'candidate.json').exists()
    finally:bridge.stop()
