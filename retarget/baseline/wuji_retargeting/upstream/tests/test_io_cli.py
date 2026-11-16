"""File/CLI contracts, independent of the SDK and recorded user dataset."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from emf_retarget.io import iter_emf_frames
from emf_retarget.cli import main
from emf_retarget.pipeline import matrix_to_xyzw
from human_ik_reference import HumanHandKinematics
from human_ik_reference.parameters import default_parameters


def sample():
    h=HumanHandKinematics('right')
    poses=h.forward(np.concatenate(default_parameters('right')['seed']))['tx_rx']
    return poses[:,:3,3],np.array([matrix_to_xyzw(t[:3,:3]) for t in poses]),np.ones(5)


class IOTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.path=Path(self.directory.name)

    def test_npz_roundtrip_and_cli(self):
        p,q,c=sample();source=self.path/'in.npz';output=self.path/'out.npz'
        np.savez(source,positions=np.array([p,p]),orientations_xyzw=np.array([q,q]),confidence=np.array([c,c]),seq=np.array([41,42]),timestamp_us=np.array([5000,6000]))
        frames=list(iter_emf_frames(source,'right',start=1,limit=1))
        self.assertEqual((len(frames),frames[0].seq,frames[0].timestamp_us),(1,42,6000))
        self.assertEqual(main([str(source),'--side','right','--limit','1','--output',str(output)]),0)
        with np.load(output,allow_pickle=False) as result:
            self.assertEqual(result['right_human_q'].shape,(1,21))
            self.assertEqual(result['right_wuji1_q'].shape,(1,20))
            self.assertEqual(result['right_wuji2_q'].dtype,np.float32)
            self.assertEqual(len(result['right_wuji1_joint_names']),20)
            self.assertTrue(np.isfinite(result['right_wuji2_q']).all())
        report=json.loads(output.with_suffix('.json').read_text())
        self.assertEqual(report['sides']['right']['frames'],1)
        with self.assertRaises(SystemExit):main([str(source),'--side','right','--output',str(output)])

    def test_hdf5_header_units_and_selection(self):
        try:import h5py
        except ImportError:self.skipTest('h5py optional dependency absent')
        p,q,c=sample();source=self.path/'in.h5'
        with h5py.File(source,'w') as h:
            g=h.create_group('right')
            g['emf_poses/position']=np.array([p,p]);g['emf_poses/orientation_xyzw']=np.array([q,q]);g['emf_poses/confidence']=np.array([c,c])
            g['source_seq']=[1,2];g['source_timestamp_ns']=[1234567,2345678];g['source_frame_id']=np.array([b'r0',b'r1'])
        frames=list(iter_emf_frames(source,'right',start=1,limit=1))
        self.assertEqual((frames[0].timestamp_us,frames[0].frame_id),(2345,'r1'))
        with self.assertRaises(ValueError):list(iter_emf_frames(source,'left'))

    def test_malformed_npz_is_rejected(self):
        p,q,c=sample();source=self.path/'bad.npz'
        np.savez(source,positions=p,orientations_xyzw=q,confidence=c)
        with self.assertRaises(ValueError):list(iter_emf_frames(source,'right'))


if __name__=='__main__':unittest.main()
