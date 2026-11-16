"""Unified boundary/state tests using only independently generated EMF poses."""
import copy
from dataclasses import replace
import pickle
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from emf_retarget import EmfFrame, EmfRetargetPipeline
from emf_retarget.pipeline import matrix_to_xyzw
from human_ik_reference import HumanIKSession


def synthetic_frame(side='right', phase=0., seq=0):
    human = HumanIKSession(side)
    q = np.concatenate(human.parameters['seed'])
    q += .035*np.sin(np.arange(21)+phase)
    poses = [human.kinematics.finger_pose_jacobian(f,q[s])[0]
             for f,s in enumerate(human.kinematics.finger_slices)]
    return EmfFrame(np.array([t[:3,3] for t in poses]),
                    np.array([matrix_to_xyzw(t[:3,:3]) for t in poses]),
                    np.linspace(.6,1.,5), seq=seq, timestamp_us=1000+seq,
                    frame_id='synthetic_tx')


def state_bytes(pipeline):
    human=pipeline.human
    state=(human._history,human._history2,human.last_trace,
           {name:(r._warm,r._filtered,r.last_trace) for name,r in pipeline.robots.items()},
           pipeline._key,pipeline._result,pipeline.frames_computed)
    return pickle.dumps(state,protocol=5)


def result_arrays(result):
    return {**{name:value for name,value in vars(result).items() if isinstance(value,np.ndarray)},
            **{'robot_'+name:value for name,value in result.robot_q.items()}}


class PipelineTests(unittest.TestCase):
    def assertResultsEqual(self,a,b):
        self.assertEqual((a.seq,a.timestamp_us,a.frame_id,a.side),(b.seq,b.timestamp_us,b.frame_id,b.side))
        self.assertEqual(set(result_arrays(a)),set(result_arrays(b)))
        for name,value in result_arrays(a).items():
            self.assertEqual(value.tobytes(),result_arrays(b)[name].tobytes(),name)

    def test_shapes_bounds_and_no_sdk_import(self):
        self.assertFalse(any(k=='wuji_sdk' or k.startswith('wuji_sdk.') for k in sys.modules))
        for side in ('left','right'):
            p=EmfRetargetPipeline(side)
            r=p.process(synthetic_frame(side))
            for name,shape in [('human_q',(21,)),('human_confidence',(5,)),
                ('skeleton_position',(21,3)),('skeleton_orientation_xyzw',(21,4)),
                ('skeleton_confidence',(21,)),('tip_position',(5,3)),
                ('tip_orientation_xyzw',(5,4)),('human_position_error_m',(5,))]:
                value=getattr(r,name)
                self.assertEqual(value.shape,shape,name)
                self.assertTrue(np.isfinite(value).all(),name)
            self.assertEqual(set(r.robot_q),{'wuji1','wuji2'})
            human_bounds=np.concatenate([np.asarray(b) for b in p.human.parameters['bounds']])
            self.assertTrue(np.all(r.human_q>=human_bounds[:,0]-1e-8))
            self.assertTrue(np.all(r.human_q<=human_bounds[:,1]+1e-8))
            for name,q in r.robot_q.items():
                self.assertEqual(q.shape,(20,));self.assertEqual(q.dtype,np.float32)
                fk=p.robots[name].kinematics
                self.assertTrue(np.all(q>=fk.lower-2e-7))
                self.assertTrue(np.all(q<=fk.upper+2e-7))
            np.testing.assert_allclose(np.linalg.norm(r.skeleton_orientation_xyzw,axis=1),1.,atol=1e-6)

    def test_dual_models_equal_separate_stateful_pipelines(self):
        dual=EmfRetargetPipeline('right')
        singles={m:EmfRetargetPipeline('right',models=m) for m in dual.models}
        self.assertIsNot(dual.robots['wuji1'],dual.robots['wuji2'])
        for i in range(3):
            frame=synthetic_frame(phase=i*.4,seq=i)
            result=dual.process(frame)
            for name,one in singles.items():
                separate=one.process(frame)
                np.testing.assert_array_equal(result.human_q,separate.human_q)
                np.testing.assert_array_equal(result.robot_q[name],separate.robot_q[name])
                np.testing.assert_array_equal(dual.robots[name]._warm,one.robots[name]._warm)

    def test_duplicate_frame_is_cached_without_state_advance(self):
        p=EmfRetargetPipeline()
        frame=synthetic_frame()
        first=p.process(frame);before=state_bytes(p)
        with patch.object(p.human,'step',side_effect=AssertionError('cache called human solver')):
            second=p.process(replace(frame))
        self.assertResultsEqual(first,second)
        self.assertEqual(state_bytes(p),before)
        self.assertEqual(p.frames_computed,1)

    def test_header_and_signed_zero_are_part_of_cache_identity(self):
        p=EmfRetargetPipeline()
        frame=synthetic_frame()
        confidence=frame.confidence.copy();confidence[0]=0.
        frame=replace(frame,confidence=confidence)
        p.process(frame)
        for changed in [replace(frame,seq=9),replace(frame,timestamp_us=9000),
                        replace(frame,frame_id='other_tx')]:
            before=p.frames_computed;p.process(changed)
            self.assertEqual(p.frames_computed,before+1)
        p.process(frame);before=p.frames_computed
        confidence[0]=-0.
        changed=replace(frame,confidence=confidence)
        self.assertNotEqual(changed.cache_key(),frame.cache_key())
        p.process(changed)
        self.assertEqual(p.frames_computed,before+1)

    def test_output_and_input_ownership(self):
        frame=synthetic_frame();positions=frame.positions.copy()
        copied=replace(frame,positions=positions)
        positions[:]=0.
        np.testing.assert_array_equal(copied.positions,frame.positions)
        with self.assertRaises(ValueError):copied.positions[0,0]=1.
        p=EmfRetargetPipeline();first=p.process(frame)
        pristine=copy.deepcopy(first);before=state_bytes(p)
        for array in result_arrays(first).values():array[:]=1234.
        first.robot_q['invented']=np.zeros(20)
        self.assertResultsEqual(p.process(frame),pristine)
        self.assertEqual(state_bytes(p),before)

    def test_reset_replays_entire_chain(self):
        for side in ('left','right'):
            p=EmfRetargetPipeline(side)
            frames=[synthetic_frame(side,phase=i*.5,seq=i) for i in range(2)]
            first=[p.process(f) for f in frames]
            p.reset()
            self.assertEqual(p.frames_computed,0)
            self.assertIsNone(p._key);self.assertIsNone(p._result)
            self.assertTrue(all(r._warm is None and r._filtered is None for r in p.robots.values()))
            for old,frame in zip(first,frames):self.assertResultsEqual(old,p.process(frame))

    def test_second_model_failure_rolls_back_every_component(self):
        p=EmfRetargetPipeline();control=EmfRetargetPipeline()
        first=synthetic_frame(seq=0);second=synthetic_frame(phase=.8,seq=1)
        p.process(first);control.process(first)
        before=state_bytes(p)
        with patch.object(p.robots['wuji2'],'step',side_effect=RuntimeError('injected second-model error')):
            with self.assertRaisesRegex(RuntimeError,'injected second-model error'):p.process(second)
        self.assertEqual(state_bytes(p),before)
        self.assertResultsEqual(p.process(second),control.process(second))

    def test_invalid_input_preserves_state(self):
        p=EmfRetargetPipeline();frame=synthetic_frame();p.process(frame)
        before=state_bytes(p)
        with self.assertRaises(TypeError):p.process({'positions':frame.positions})
        for change in [dict(positions=np.zeros((4,3))),dict(positions=np.full((5,3),np.nan)),
                       dict(orientations_xyzw=np.zeros((5,4))),dict(confidence=np.full(5,np.inf)),
                       dict(seq=-1),dict(timestamp_us=2**64),dict(seq=True)]:
            with self.assertRaises(ValueError):p.process(replace(frame,**change))
        self.assertEqual(state_bytes(p),before)


if __name__=='__main__':
    unittest.main(verbosity=2)
