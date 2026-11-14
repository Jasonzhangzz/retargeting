"""Real live-worker comparison swaps whole candidate models, then restores."""
import copy,json,threading,time
from pathlib import Path
import numpy as np
import pytest
import torch
from moxian_web.hand_model import HandModelBridge,DEFAULT_MANO


def test_two_real_checkpoint_previews_restore_original_without_publishing(tmp_path):
    root=Path(__file__).resolve().parents[2]
    source=root/'output/moxian/hand_model/348640923d9e2f3f/fit-wrist-targets.json'
    paths=[root/'output/moxian/debug/pair-local-ema-v1/parameters-warm-3/candidate.json',root/'output/moxian/debug/pair-local-ema-v1/flashsac-warm-3/candidate.json']
    if not torch.cuda.is_available() or not source.exists() or not all(p.exists() for p in paths) or not DEFAULT_MANO.exists():pytest.skip('Local retained observations, candidates and CUDA required')
    row=next(r for r in json.loads(source.read_text())['rows'] if r['stage']=='open')
    weights=[json.loads(p.read_text()) for p in paths]
    bridge=HandModelBridge(root,tmp_path);bridge.start(weights[0]['serial'],'right',surface_backend='mano');stop=threading.Event()
    def feed():
        while not stop.is_set():
            frame=copy.deepcopy(row);stamp=time.time()*1000
            frame.update(type='frame',timestamp_ms=stamp,tactile_timestamp_ms=stamp,contact={'ready':False,'cell_contact':[None]*460,'patches':[]})
            frame['skeleton']['timestamp_ms']=stamp
            if isinstance(frame.get('skeleton_targets'),dict):frame['skeleton_targets']['timestamp_ms']=stamp
            bridge.send(frame);stop.wait(.04)
    thread=threading.Thread(target=feed,daemon=True);thread.start()
    def wait(predicate,seconds=60):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            state=bridge.snapshot()
            if predicate(state):return state
            time.sleep(.05)
        raise AssertionError(bridge.snapshot())
    try:
        baseline=wait(lambda h:h.get('ready'));original=baseline['betas'];original_geometry=baseline['geometry_version']
        for index,weight in enumerate(weights):
            bridge.control('checkpoint_preview',{'checkpoint':weight,'info':{'id':str(index),'label':weight['report']['method'],'accepted':False}})
            shown=wait(lambda h:h.get('ready') and h.get('checkpoint_preview',{}).get('id')==str(index))
            assert shown['preview'] and not shown['calibrated']
            np.testing.assert_allclose(shown['betas'],weight['shape'][:10],atol=1e-7)
            assert shown['runtime_backend']=='fixed_mano_v1' and shown['shared_skeleton_pose']
            assert shown['mano_solver']['optimizer_steps']==0 and not shown['mano_solver']['tactile_pose_feedback']
        bridge.control('checkpoint_end',{})
        restored=wait(lambda h:h.get('ready') and not h.get('preview') and h.get('checkpoint_preview') is None)
        np.testing.assert_array_equal(restored['betas'],original)
        assert restored['runtime_backend']=='fixed_mano_v1' and not restored['calibrated']
        assert not list(tmp_path.glob('hand_model/*/current.json'))
        assert not list(tmp_path.glob('hand_model/*/candidate.json'))
    finally:
        stop.set();thread.join(timeout=2);bridge.stop()
