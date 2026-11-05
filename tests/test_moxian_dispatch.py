import unittest
from types import SimpleNamespace
from unittest.mock import patch
from moxian_web.hand_model import HandInputSynchronizer
try:
    from moxian_web.server import Runtime
except ImportError:
    Runtime=None

@unittest.skipIf(Runtime is None,'Run in .venv-moxian with the installed acquisition SDK')
class DispatchTests(unittest.TestCase):
    def test_completed_ik_dispatches_without_waiting_for_another_sensor_frame(self):
        runtime=object.__new__(Runtime);runtime.hand_inputs=HandInputSynchronizer();sent=[]
        frame={'timestamp_ms':1000.,'tactile_timestamp_ms':1004.,'hand_side':'left','raw':[10]*460,'fingers':{f:{'position':[.01,.02,.03],'quaternion_xyzw':[0,0,0,1]} for f in ('thumb','index','middle','ring','pinky')}}
        runtime.hand_inputs.push(frame);bone={'ready':False};runtime.ik=SimpleNamespace(snapshot=lambda:bone)
        runtime.contact=SimpleNamespace(mapping=lambda:{'version':'v'},profile={'tare':{'baseline':[10]*460}});runtime.hand=SimpleNamespace(send=lambda data:sent.append(data))
        with patch('moxian_web.server.time.time',return_value=1.05):
            self.assertIsNone(runtime.dispatch_hand_input())
            # A result arrives later; no additional raw frame is pushed.
            bone.update(ready=True,pose={'timestamp_ms':1000.},targets={'timestamp_ms':1000.},calibration_fingerprint='c',runtime_policy={'runtime_policy_sha256':'p'})
            actual=runtime.dispatch_hand_input();self.assertEqual(len(sent),1)
            self.assertEqual(actual['timestamp_ms'],1000.);self.assertEqual(actual['tactile_timestamp_ms'],1004.)
            self.assertEqual(actual['skeleton_runtime_revision'],'p');self.assertEqual(actual['dispatch_timestamp_ms'],1050.)
            self.assertIsNone(runtime.dispatch_hand_input());self.assertEqual(len(sent),1)
            self.assertNotIn('mapping',frame)
