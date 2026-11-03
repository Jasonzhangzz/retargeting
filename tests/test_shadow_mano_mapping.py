"""Real-model consistency checks for the contact-mapping inspection."""
from pathlib import Path
import json,unittest
import numpy as np
import torch,mujoco
from scipy.spatial.transform import Rotation
from retarget.env.grasp.mano_solver import ManoDecoder,sha
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'output/dexonomy_shadow_mano_mapping_20261103'
class MappingInspectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.raw=json.loads((FOLDER/'data.json').read_text())
        cls.decoder=ManoDecoder(ROOT/'data/grasp/assets/mano/MANO_RIGHT.pkl')
        cls.faces=cls.decoder.base.faces
        cls.shadow=mujoco.MjModel.from_xml_path(str(ROOT/'data/dexonomy/shadow_assets/right.xml'))
    def test_coverage_and_model_hash(self):
        self.assertEqual(len(self.raw['mappings']),31)
        self.assertEqual(len(self.raw['cases']),62)
        self.assertEqual(self.raw['model_sha256'],sha(self.decoder.base.path))
        self.assertFalse(self.raw['dataset_contact_labels_available'])
    def test_barycentric_surface_and_normals(self):
        for row in self.raw['cases']:
            vertices=np.array(row['mano_vertices']);points=np.array(row['mano_contacts']);normals=np.array(row['mano_normals'])
            for i,rec in enumerate(row['contact_mapping']):
                b=np.array(rec['mano_barycentric']);tri=vertices[self.faces[rec['mano_face_id']]]
                self.assertAlmostEqual(float(b.sum()),1.,places=6);self.assertGreaterEqual(float(b.min()),-1e-5)
                np.testing.assert_allclose(b@tri,points[i],atol=2e-7)
                normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);normal/=np.linalg.norm(normal)
                np.testing.assert_allclose(normal,normals[i],atol=2e-7)
    def test_no_cross_finger_mapping(self):
        chains={'thumb':[13,14,15],'index':[1,2,3],'middle':[4,5,6],'ring':[10,11,12],'pinky':[7,8,9]}
        weights=self.decoder.base.weights.cpu().numpy()
        for mapping in self.raw['mappings'].values():
            for rec in mapping:
                if rec['finger']=='palm':continue
                self.assertIn(rec['mano_joint_id'],chains[rec['finger']])
                patch=weights[np.array(rec['mano_vertex_ids'])][:,chains[rec['finger']]].sum(-1)
                self.assertGreaterEqual(int((patch>.65).sum()),2)
    def test_exact_source_fk(self):
        for row in self.raw['cases']:
            q=np.array(row['source_qpos']);data=mujoco.MjData(self.shadow);data.qpos[:]=q[7:]
            mujoco.mj_kinematics(self.shadow,data);root=Rotation.from_quat(q[3:7],scalar_first=True).as_matrix()
            for i,rec in enumerate(row['contact_mapping']):
                body=self.shadow.body(rec['shadow_body']).id;r=data.xmat[body].reshape(3,3)
                point=(data.xpos[body]+r@rec['shadow_local_point'])@root.T+q[:3]
                np.testing.assert_allclose(point,row['shadow_contacts'][i],atol=1e-8)
    def test_posed_normals_do_not_flip(self):
        for row in self.raw['cases']:
            source=np.array(row['shadow_normals']);target=np.array(row['mano_normals'])
            cosine=(source*target).sum(-1)/(np.linalg.norm(source,axis=-1)*np.linalg.norm(target,axis=-1))
            self.assertGreater(float(cosine.min()),0.,row['case'])
    def test_mano_pose_reconstruction(self):
        variables=torch.tensor([r['variables'] for r in self.raw['cases']],dtype=torch.float32)
        with torch.no_grad():v,j,_=self.decoder(variables,full=True)
        np.testing.assert_allclose(v.numpy(),[r['mano_vertices'] for r in self.raw['cases']],atol=3e-6)
        np.testing.assert_allclose(j.numpy(),[r['mano_joints'] for r in self.raw['cases']],atol=3e-6)
if __name__=='__main__':unittest.main()
