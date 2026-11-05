import hashlib,json,math,tempfile,unittest,xml.etree.ElementTree as ET
from pathlib import Path
from moxian_web.kinematic_policy import build_kinematic_config

class PolicyTests(unittest.TestCase):
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
