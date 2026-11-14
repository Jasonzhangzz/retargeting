import hashlib,json,math,tempfile,unittest,xml.etree.ElementTree as ET
from pathlib import Path
from moxian_web.kinematic_policy import build_experimental_kinematic_config as build_kinematic_config,apply_thumb_registration,load_thumb_registration,POLICY_REVISION

class RetiredExperimentalPolicyTests(unittest.TestCase):
    """Historical ablations only; production uses the separate official policy."""
    def test_thumb_twist_precedes_static_bend_in_actual_solver_asset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);package=root/'p';folder=package/'share/qnbot_hand/hands/dynamic_openxr_hand/resources/pinocchio';folder.mkdir(parents=True)
            joints='''<robot>
<joint name="left_thumb_mcp_abduction"><origin xyz="0.04 0 0" rpy="0.001 0.002 0.5"/><axis xyz="1 0 0"/><limit lower="-0.6" upper="0.3"/></joint>
<joint name="left_thumb_mcp_flexion"><origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-0.35" upper="1.2"/></joint>
<joint name="left_thumb_ip_flexion"><origin xyz="0.03 0 0" rpy="0 0 0.3"/><axis xyz="0 0 1"/><limit lower="-0.26" upper="1.4"/></joint></robot>'''
            for name in ('left.urdf','left_skeleton.urdf'):(folder/name).write_text(joints)
            (folder/'metadata.yaml').write_text('left_thumb_mcp_flexion: [-0.35, 1.2]\nleft_thumb_ip_flexion: [-0.26, 1.4]\n')
            config='joint_smoother:\n  enabled: true\nerror_reseed:\n  enabled: true\n__resource_root__: /old\n'
            _,report=build_kinematic_config(config,package,root/'runtime')
            asset=root/'runtime'/('resources-'+report['runtime_policy_sha256'][:16])/ 'hands/dynamic_openxr_hand/resources/pinocchio'
            a=ET.parse(asset/'left.urdf').getroot();b=ET.parse(asset/'left_skeleton.urdf').getroot()
            self.assertEqual(ET.tostring(a),ET.tostring(b))
            js={j.get('name'):j for j in a.findall('joint')}
            self.assertEqual(js['left_thumb_mcp_abduction'].find('origin').get('rpy'),'0 0 0')
            self.assertEqual(js['left_thumb_mcp_flexion'].find('origin').get('rpy'),'0.001 0.002 0.5')
            self.assertAlmostEqual(float(js['left_thumb_mcp_flexion'].find('limit').get('lower'))+.5,-math.radians(3))
            for name in ('left.urdf','left_skeleton.urdf'):self.assertEqual((folder/name).read_text(),joints)

    def test_fixed_thumb_mount_changes_only_left_and_preserves_other_fingers(self):
        text='''tip_mapping:
  right:
    input_tip_offset_local:
      thumb: [1, 2, 3]
  left:
    input_tip_rotation_offset_local:
      first: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    input_tip_offset_local:
      first:
        - 0.008
        - 0
        - -0.005
      thumb:
        - 0.005
        - 0
        - -0.01
solver:
  unchanged: true
'''
        r={'input_tip_offset_local_m':[.02,.001,-.02],'input_tip_rotation_offset_local':[[1,0,0],[0,1,0],[0,0,1]]}
        output=apply_thumb_registration(text,r)
        self.assertIn('thumb: [0.02, 0.001, -0.02]',output)
        self.assertEqual(text.split('  left:')[0],output.split('  left:')[0])
        self.assertIn('      first:\n        - 0.008\n        - 0\n        - -0.005',output)
        self.assertEqual(output.split('solver:')[1],text.split('solver:')[1])

    def test_mount_is_bound_to_calibration_and_rejects_reflection_or_unvalidated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'thumb-registration.json'
            data={'calibration_sha256':'a','policy_revision':POLICY_REVISION,'accepted':True,'input_tip_offset_local_m':[.02,0,0],'input_tip_rotation_offset_local':[[1,0,0],[0,1,0],[0,0,1]]}
            p.write_text(json.dumps(data));self.assertEqual(load_thumb_registration(root,'a'),data)
            self.assertIsNone(load_thumb_registration(root,'b'))
            data['input_tip_rotation_offset_local'][0][0]=-1;p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):load_thumb_registration(root,'a')
            data['accepted']=False;p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):load_thumb_registration(root,'a')

    def test_native_sdk_reads_the_modified_solver_geometry(self):
        # Catch the real integration error: changing only *_skeleton.urdf can
        # pass XML tests while the native solver still reads left.urdf.
        from moxian_web.native_skeleton import NativeSkeleton
        from moxian_web.official_fk import urdf_reference
        root=Path(__file__).resolve().parents[1]
        packages=list((root/'.venv-moxian-ik/lib').glob('python*/site-packages/qnbot_sdk/glove/_private/algorithms/skeleton/*'))
        if len(packages)!=1:self.skipTest('local official SDK required')
        package=packages[0];source=package/'share/qnbot_hand/hands/dynamic_openxr_hand'
        with tempfile.TemporaryDirectory() as tmp:
            import yaml
            config=yaml.safe_load((source/'dynamic_openxr_hand_system.yaml').read_text());config['__resource_root__']='/old'
            config['joint_smoother']['enabled']=True;config['solver']['error_reseed']['enabled']=True
            path,report=build_kinematic_config(yaml.safe_dump(config,sort_keys=False),package,Path(tmp))
            resource=Path(tmp)/('resources-'+report['runtime_policy_sha256'][:16])/'hands/dynamic_openxr_hand/resources/pinocchio'
            original=urdf_reference(source/'resources/pinocchio/left_skeleton.urdf');updated=urdf_reference(resource/'left.urdf')
            self.assertEqual(original['positions_local'],updated['positions_local'])
            self.assertEqual(original['orientations_local_xyzw'],updated['orientations_local_xyzw'])
            tree=ET.parse(resource/'left.urdf')
            for joint in tree.getroot().findall('joint'):
                if joint.get('name')=='left_thumb_cmc_abduction':joint.find('origin').set('xyz','0.012345 0.023456 0.034567')
            tree.write(resource/'left.urdf')
            engine=NativeSkeleton(package,path)
            try:
                result=engine.step({'timestamp_ms':1000.,'fingers':{f:{'position':[.03,.02,.1],'quaternion_xyzw':[0.,0.,0.,1.]}for f in ('thumb','index','middle','ring','pinky')}})
                pose=result['pose'];p=pose['positions_local'][pose['names'].index('thumb_metacarpal')]
                for actual,expected in zip(p,[.012345,.023456,.034567]):self.assertAlmostEqual(actual,expected,places=8)
                self.assertFalse(result['physics']['enabled']);self.assertEqual(result['physics']['steps'],0)
            finally:engine.close()

    def test_physical_extension_accounts_for_static_origin_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);package=root/'package';folder=package/'share/qnbot_hand/hands/dynamic_openxr_hand/resources/pinocchio';folder.mkdir(parents=True)
            text='<robot><joint name="left_index_pip_flexion"><origin rpy="0 0 0.37"/><axis xyz="0 0 1"/><limit lower="-0.7" upper="1.9"/></joint></robot>'
            urdf=folder/'left_skeleton.urdf';urdf.write_text(text);meta=folder/'metadata.yaml';meta.write_text('left:\n  joint_limits_rad:\n    left_index_pip_flexion: [-0.7, 1.9]\n')
            config='qnbot_sdk:\n  physics_enabled: true\njoint_smoother:\n  enabled: true\nsolver:\n  error_reseed:\n    enabled: true\n__resource_root__: /old\n'
            original={p:p.read_bytes() for p in (urdf,meta)};path,report=build_kinematic_config(config,package,root/'runtime');derived=path.read_text()
            self.assertIn('physics_enabled: false',derived);self.assertNotIn('enabled: true',derived)
            limit=report['joint_limits']['left_index_pip_flexion'];self.assertAlmostEqual(limit['q_min_rad']+.37,-math.radians(3))
            self.assertLess(limit['q_min_rad'],0)
            for p,before in original.items():self.assertEqual(p.read_bytes(),before)
            p2,r2=build_kinematic_config(config,package,root/'runtime');self.assertEqual(p2,path);self.assertEqual(report,r2)
            self.assertEqual(path.read_text(),derived)
    def test_unknown_joint_axis_fails_instead_of_inventing_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'p/share/qnbot_hand/hands/dynamic_openxr_hand/resources/pinocchio';folder.mkdir(parents=True)
            (folder/'left_skeleton.urdf').write_text('<robot><joint name="left_index_pip_flexion"><origin rpy="0 0 0"/><axis xyz="1 0 0"/><limit lower="0" upper="1"/></joint></robot>')
            (folder/'metadata.yaml').write_text('left_index_pip_flexion: [0, 1]')
            with self.assertRaises(ValueError):build_kinematic_config('',root/'p',root/'runtime')
