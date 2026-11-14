import copy,math,tempfile,unittest,json
from pathlib import Path
from moxian_web.hand_model import pair_mechanics,positive_gain,material_config,aligned_frame,fusion_state,validate_hand_output,HandModelBridge,HandInputSynchronizer,confirmed_contact_points,persist_calibrated_profile,tracking_registration_matches
from moxian_web.contact import PATCHES
from moxian_web.hand_model import MANO_FIT_REVISION

def input_frame(t=1000,touch=False):
    return {'timestamp_ms':t,'tactile_timestamp_ms':t+4,'hand_side':'left','raw':[10]*460,
            'skeleton':{'timestamp_ms':t,'hand_side':'left','coordinate_frame':'wrist_local'},
            'fingers':{f:{'position':[.01,.02,.03],'quaternion_xyzw':[0.,0.,0.,1.]} for f in ['thumb','index','middle','ring','pinky']},
            'contact':{'ready':True,'patches':[{**p,'contact':touch and p['id'] in ['thumb_distal','index_distal'],'response':.3 if touch else 0} for p in PATCHES]}}

class HandContractTests(unittest.TestCase):
    def test_native_policy_change_invalidates_mount_even_with_same_capture(self):
        saved={'calibration_fingerprint':'capture','skeleton_runtime_revision':'old'}
        frame={'calibration_fingerprint':'capture','skeleton_runtime_revision':'new'}
        self.assertFalse(tracking_registration_matches(saved,frame))
        saved['skeleton_runtime_revision']='new';self.assertTrue(tracking_registration_matches(saved,frame))
        self.assertFalse(tracking_registration_matches({},frame))
        self.assertFalse(tracking_registration_matches({'calibration_fingerprint':'capture'},frame))
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
        for key,value in [('tactile_timestamp_ms',1026),('hand_side','unknown'),('timestamp_ms',float('inf'))]:
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
        for key,value in [('hand_side','unknown'),('vertices_m',[[1,1,1]]*778),('joints_m',[[0,0,0]]*20)]:
            q=copy.deepcopy(p);q[key]=value
            with self.assertRaises(ValueError):validate_hand_output(q)
    def test_offline_audit_reports_actual_applied_profile_not_stale_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);home=root/'hand_model'/'device';home.mkdir(parents=True)
            audit={'dimensions_applied':False,'active_dimensions_mm':{'index':{'surface_length_mm':80}},'capture_complete':True,'candidate_report':{'accepted':False}}
            (home/'calibration-audit.json').write_text(json.dumps(audit))
            (home/'candidate.json').write_text(json.dumps({'dimension_mode':'official_mano_betas_v1','report':{'accepted':True,'dimensions_after_mm':{'index':{'surface_length_mm':85}}}}))
            b=HandModelBridge(root,root);state=b.snapshot(False)
            self.assertFalse(state['ready']);self.assertFalse(state['calibrated']);self.assertTrue(state['fit']['accepted'])
            self.assertEqual(state['dimensions_mm']['index']['surface_length_mm'],80)
            (home/'current.json').write_text(json.dumps({'dimension_mode':'official_mano_betas_v1','calibrated':True,'revision':'applied','dimension_parameters':[0.]*20,'report':{'dimensions_after_mm':{'index':{'surface_length_mm':85}}}}))
            b.audit_checked=0;state=b.snapshot(False)
            self.assertFalse(state['calibrated']);self.assertFalse(state['calibration_process']['dimensions_applied'])
            self.assertEqual(state['dimensions_mm']['index']['surface_length_mm'],85)
            self.assertEqual(state['calibration_audit']['applied_revision'],'applied')

    def test_auto_application_persists_calibrated_version_and_restore_backup(self):
        from moxian_web.hand_model import MANO_SOLVER_REVISION
        report=dict(fit_revision=MANO_FIT_REVISION,solver_revision=MANO_SOLVER_REVISION,accepted=True,validation_pose_source='runtime_mano_sensor_only',contact_validation_passed=True,
                    together_validation_passed=True,flat_validation_passed=True,penetration_validation_passed=True,
                    runtime_validation_passed=True,per_stage_validation_passed=True,holdout_validation_passed=True,
                    capture_protocol='official_eight_v1',max_penetration_mm=1.,runtime_max_penetration_mm=1.)
        candidate={'report':report,'dimension_parameters':[.02]*20,'calibrated':False}
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp);old={'calibrated':False,'dimension_parameters':[0]*20}
            applied=persist_calibrated_profile(home,candidate,True,old)
            self.assertTrue(applied['calibrated']);self.assertTrue(applied['automatic_application']);self.assertTrue(applied['revision'])
            self.assertEqual(json.loads((home/'current.json').read_text()),applied)
            self.assertEqual(json.loads((home/'previous.json').read_text()),old)
            self.assertFalse(candidate['calibrated'])
            newer=persist_calibrated_profile(home,candidate,True)
            self.assertNotEqual(newer['revision'],applied['revision'])
            self.assertEqual(json.loads((home/'previous.json').read_text()),applied)
            invalid=copy.deepcopy(candidate);invalid['report']['together_validation_passed']=False
            with self.assertRaises(ValueError):persist_calibrated_profile(home,invalid,True)
            self.assertEqual(json.loads((home/'current.json').read_text()),newer)

    def test_missing_gpu_interpreter_does_not_block_sdk(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=HandModelBridge(Path(tmp),Path(tmp));b.start('test');self.assertEqual(b.snapshot()['status'],'unavailable')
            for _ in range(10000):b.send({'type':'frame'})
            self.assertFalse(b.snapshot()['ready']);b.stop()


if __name__=='__main__':unittest.main()


def test_polling_summary_keeps_acceptance_and_hides_large_fit_arrays():
    from moxian_web.hand_model import public_fit_report
    report={'accepted':False,'position_rmse_mm':2.,'collision_frames':[{'depth':9}]*1000,
            'runtime_solver_frames':[{'status':'bad'}]*1000,
            'solver':{'parameter_snapshot':{'vector':[0]*10000},'phases':[{'phase':'reference','steps':5,'parameter_vector':[0]*10000}]}}
    before=copy.deepcopy(report);summary=public_fit_report(report)
    assert summary['accepted'] is False and summary['position_rmse_mm']==2.
    assert summary['solver']['phases']==[{'phase':'reference','steps':5}]
    assert 'collision_frames' not in summary and 'parameter_snapshot' not in summary['solver']
    assert len(json.dumps(summary))<500 and report==before
