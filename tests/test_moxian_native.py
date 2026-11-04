"""Native target contract and a read-only replay of the retained calibration."""
import copy,hashlib,json,math,shutil,tempfile,time,unittest
from pathlib import Path
from moxian_web.native_skeleton import quaternion_matrix,matrix_quaternion
from moxian_web.skeleton import SkeletonBridge

ROOT=Path(__file__).resolve().parents[1]
CAPTURE=ROOT/'output/moxian/hand_model/9c0dfad82edffc57/capture.json'
HOME=ROOT/'output/moxian/ik/9c0dfad82edffc57'
class NativeTests(unittest.TestCase):
    def test_rotation_roundtrip_including_half_turns(self):
        for q in ([0,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,1,0],[.2,-.3,.4,.5],[.7,-.4,.2,.01]):
            m=quaternion_matrix(q);actual=quaternion_matrix(matrix_quaternion(m))
            self.assertLess(max(abs(a-b) for a,b in zip(m,actual)),1e-10)
    def test_invalid_quaternions(self):
        for q in ([0,0,0,0],[1,2,3],[float('nan'),0,0,1]):
            with self.assertRaises(ValueError):quaternion_matrix(q)

    @unittest.skipUnless(CAPTURE.exists() and HOME.exists() and (ROOT/'.venv-moxian-ik/bin/python').exists(),'Retained local SDK capture needed')
    def test_calibrated_bridge_same_frame_targets_no_dynamics_or_source_mutation(self):
        sources={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in HOME.glob('glove/calibrations/**/*') if p.is_file()}
        row=json.loads(CAPTURE.read_text())['rows'][10]
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);shutil.copytree(HOME/'glove',folder/'ik/9c0dfad82edffc57/glove')
            bridge=SkeletonBridge(ROOT,folder);self.addCleanup(bridge.stop);bridge.start('2DF5F3B00DA5')
            deadline=time.monotonic()+12;i=0;state={}
            while time.monotonic()<deadline:
                frame={k:copy.deepcopy(row[k]) for k in ('fingers','timestamp_ms')};frame.update(type='frame',timestamp_ms=row['timestamp_ms']+i*16);bridge.send(frame);i+=1;time.sleep(.02)
                state=bridge.snapshot()
                if state.get('targets') and state.get('ready'):break
            self.assertTrue(state.get('ready'),state);self.assertEqual(state.get('pose_source'),'official_native_kinematic')
            self.assertEqual(state['targets']['timestamp_ms'],state['pose']['timestamp_ms'])
            self.assertEqual(state['targets']['hand_side'],'left');self.assertEqual(state['targets']['coordinate_frame'],'wrist_local')
            self.assertEqual(len(state['targets']['positions_local']),5);self.assertEqual(len(state['pose']['names']),25)
            self.assertEqual(state['physics'],{'enabled':False,'steps':0,'contacts':0,'actuators':0})
            self.assertGreater(max(state['alignment']['tip_errors_mm'].values()),1.)
            bridge.stop()
        for p,digest in sources.items():self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),digest)
if __name__=='__main__':unittest.main()
