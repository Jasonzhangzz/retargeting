import copy,math,tempfile,unittest
from pathlib import Path
from moxian_web.hand_model import pair_mechanics,positive_gain,material_config,aligned_frame,fusion_state,validate_hand_output,HandModelBridge,HandInputSynchronizer,confirmed_contact_points
from moxian_web.hand_capture import HandCapture
from moxian_web.contact import PATCHES

def input_frame(t=1000,touch=False):
    return {'timestamp_ms':t,'tactile_timestamp_ms':t+4,'hand_side':'left','raw':[10]*460,
            'skeleton':{'timestamp_ms':t,'hand_side':'left','coordinate_frame':'wrist_local'},
            'fingers':{f:{'position':[.01,.02,.03],'quaternion_xyzw':[0.,0.,0.,1.]} for f in ['thumb','index','middle','ring','pinky']},
            'contact':{'ready':True,'patches':[{**p,'contact':touch and p['id'] in ['thumb_distal','index_distal'],'response':.3 if touch else 0} for p in PATCHES]}}

class HandContractTests(unittest.TestCase):
    def test_contact_markers_only_confirmed_ready_and_no_input_mutation(self):
        pair={'patch_a':'thumb_distal','patch_b':'index_distal','state':'confirmed','point_a_m':[.01,.02,.03],'point_b_m':[.011,.021,.031]}
        fusion={'ready':True,'pairs':[pair,{**pair,'state':'candidate'}]};before=copy.deepcopy(fusion)
        markers=confirmed_contact_points(fusion)
        self.assertEqual(len(markers),2);self.assertEqual(markers[0]['position_m'],pair['point_a_m']);self.assertEqual(markers[1]['patch'],'index_distal')
        self.assertEqual(fusion,before);self.assertEqual(confirmed_contact_points({**fusion,'ready':False}),[])
        self.assertEqual(confirmed_contact_points({'ready':True,'pairs':[{**pair,'point_a_m':[float('nan'),0,0],'point_b_m':[0,0]}]}),[])

    def test_series_compliance_units_caps_and_material_validation(self):
        p=pair_mechanics(-.001,1e-4,1e-4,{})
        self.assertAlmostEqual(p['stiffness_n_m'],500)
        self.assertAlmostEqual(p['model_force_n'],.5)
        self.assertAlmostEqual(p['compression_a_m'],.0005)
        p=pair_mechanics(-.009,1e-4,2e-4,{})
        self.assertLessEqual(p['compression_a_m'],.002)
        self.assertEqual(pair_mechanics(.001,1e-4,1e-4,{})['model_force_n'],0)
        for m in [{'young_pa':0},{'thickness_m':float('nan')},{'max_compression_m':.01}]:
            with self.assertRaises(ValueError):material_config(m)
        with self.assertRaises(ValueError):pair_mechanics(0,0,1,{})
    def test_relative_gain_without_force_claim(self):
        self.assertEqual(positive_gain([1,2,3],[2,4,6]),2)
        self.assertIsNone(positive_gain([0,0],[1,1]))
        with self.assertRaises(ValueError):positive_gain([1],[-1])
    def test_timestamp_units_side_and_invalid_pose(self):
        f=input_frame();self.assertTrue(aligned_frame(f))
        for key,value in [('tactile_timestamp_ms',1026),('hand_side','right'),('timestamp_ms',float('inf'))]:
            q=copy.deepcopy(f);q[key]=value;self.assertFalse(aligned_frame(q))
        f['fingers']['thumb']['quaternion_xyzw']=[0,0,0,0];self.assertFalse(aligned_frame(f))
    def test_geometry_motion_and_bilateral_touch_fusion(self):
        self.assertEqual(fusion_state(True,.001,-.8,True,True,True),'confirmed')
        self.assertEqual(fusion_state(True,.001,-.8,True,True,False),'candidate')
        self.assertEqual(fusion_state(True,.001,-.8,True,False,False),'candidate')
        for native,gap,facing,motion in [(False,0,-1,True),(True,.03,-1,True),(True,0,1,True),(True,0,-1,False)]:
            self.assertEqual(fusion_state(native,gap,facing,motion,True,True),'conflict')
        self.assertEqual(fusion_state(True,0,-1,True,True,True,False),'unknown')
    def test_output_rejects_bad_units_side_topology(self):
        p={'hand_side':'left','coordinate_frame':'wrist_local','vertices_m':[[0,0,0]]*778,'joints_m':[[0,0,0]]*21,'timestamp_ms':1234}
        validate_hand_output(p)
        for key,value in [('hand_side','right'),('vertices_m',[[1,1,1]]*778),('joints_m',[[0,0,0]]*20)]:
            q=copy.deepcopy(p);q[key]=value
            with self.assertRaises(ValueError):validate_hand_output(q)
    def test_missing_gpu_interpreter_does_not_block_sdk(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=HandModelBridge(Path(tmp),Path(tmp));b.start('test');self.assertEqual(b.snapshot()['status'],'unavailable')
            for _ in range(10000):b.send({'type':'frame'})
            self.assertFalse(b.snapshot()['ready']);b.stop()

class CaptureTests(unittest.TestCase):
    def test_async_skeleton_joins_original_observation_and_capture_completes(self):
        sync=HandInputSynchronizer();c=HandCapture();c.start();c.confirm(c.request_id)
        for i in range(110):
            t=1000+i*20;sync.push(input_frame(t))
            # The async IK result is 80 ms behind the current serial input.
            if i>=4:
                matched=sync.match(input_frame(t-80)['skeleton'],t)
                self.assertEqual(matched['timestamp_ms'],t-80)
                self.assertEqual(matched['tactile_timestamp_ms'],t-76)
                if i%3==0:c.feed(matched)
        self.assertEqual(c.complete_groups,['open']);self.assertEqual(c.index,1)
        self.assertIsNone(sync.match(input_frame(1000)['skeleton'],5000))
        self.assertIsNone(sync.match(None,5000))

    def test_short_gap_restarts_stable_segment_without_failing_or_changing_request(self):
        c=HandCapture();c.start();c.confirm(c.request_id);rid=c.request_id
        for i in range(10):c.feed(input_frame(1000+i*50))
        c.feed(input_frame(1900))
        self.assertEqual(c.status,'collecting');self.assertEqual(c.request_id,rid)
        self.assertEqual(c.samples,[]);self.assertIn('短暂缺帧',c.error)
        for i in range(33):c.feed(input_frame(1950+i*50))
        self.assertEqual(c.complete_groups,['open'])

    def test_manual_confirmation_stability_and_retry(self):
        c=HandCapture();c.start()
        with self.assertRaises(ValueError):c.confirm('stale')
        c.confirm(c.request_id)
        for i in range(40):c.feed(input_frame(1000+i*50))
        self.assertEqual(c.index,1);self.assertEqual(c.status,'awaiting_confirmation');self.assertEqual(c.complete_groups,['open'])
        c.confirm(c.request_id);c.feed(input_frame(4000));c.feed(input_frame(6500))
        self.assertEqual(c.status,'failed');self.assertEqual(c.complete_groups,['open'])
        c.retry(c.request_id);self.assertEqual(c.status,'awaiting_confirmation');c.cancel();self.assertEqual(c.status,'cancelled')
    def test_two_pinch_cycles_required_and_no_stale_samples(self):
        c=HandCapture();c.start();c.index=5;c.confirm(c.request_id)
        for i in range(145):
            seconds=i*.05;c.feed(input_frame(1000+i*50,touch=(1<seconds<2 or 3<seconds<4)))
        self.assertEqual(c.index,6);self.assertEqual(c.complete_groups,['pinch_index'])
        c.confirm(c.request_id);bad=input_frame(10000);bad['tactile_timestamp_ms']+=100;c.feed(bad);self.assertEqual(len(c.samples),0)
        self.assertIn('时间',c.error)

if __name__=='__main__':unittest.main()
