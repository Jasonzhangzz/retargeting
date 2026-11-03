"""Mapped MANO LM/CEM: exact skin slots, mesh transforms and real cached outputs."""

from tests._paths import ROOT as _PROJECT_ROOT, TEST_ROOT as _TEST_ROOT
from pathlib import Path
import json,unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from retarget.env.grasp.mano_solver import ManoDecoder,FixedContacts,SurfaceBatch
ROOT=_PROJECT_ROOT
SMOKE=ROOT/'results/dexonomy_mano_solver_smoke_20261103'
class MappedSolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.raw=json.loads((ROOT/'output/dexonomy_shadow_mano_mapping_20261103/data.json').read_text())
        cls.decoder=ManoDecoder(ROOT/'data/grasp/assets/mano/MANO_RIGHT.pkl')
        cls.decoder.configure_contact_vertices(cls.raw['mappings'])
    def test_fixed_slots_match_full_and_sampled_decoder(self):
        x=torch.randn(3,30,generator=torch.Generator().manual_seed(71))*.2
        with torch.no_grad():full,_,_=self.decoder(x,full=True);sample,_,_=self.decoder(x)
        for name,records in self.raw['mappings'].items():
            fixed=FixedContacts(self.decoder,records)
            a,an=fixed.evaluate(full,full=True);b,bn=fixed.evaluate(sample)
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5);torch.testing.assert_close(an,bn,atol=1e-5,rtol=1e-4)
    def test_surface_mesh_transforms_and_barycentrics(self):
        import trimesh
        mesh=trimesh.creation.box(extents=[2,2,2]);g=dict(vertices=np.asarray(mesh.vertices,np.float32),faces=np.asarray(mesh.faces,np.int32))
        rotation=Rotation.from_rotvec([.3,-.1,.4]);quat=rotation.as_quat(scalar_first=True);origin=np.array([.2,-.1,.3])
        batch=SurfaceBatch([g],[0],np.r_[origin,quat][None],np.array([.2]),'cpu')
        local=np.array([[2.,0,0],[0,0,0],[0,2,0]],np.float32)
        world=local*.2@rotation.as_matrix().T+origin
        p,n,d,f,b=batch.query(torch.tensor(world[None],dtype=torch.float32),torch.tensor([0]))
        self.assertGreater(float(d[0,0]),0);self.assertLess(float(d[0,1]),0)
        np.testing.assert_allclose(d[0,[0,2]].numpy(),.2,atol=1e-6)
        rebuilt=(g['vertices'][g['faces'][f.numpy()[0]]]*b.numpy()[0,:,:,None]).sum(1)*.2@rotation.as_matrix().T+origin
        np.testing.assert_allclose(rebuilt,p.numpy()[0],atol=1e-6)
    def test_real_output_schema_correspondences_and_monotone_cost(self):
        manifests=list(SMOKE.glob('*/shards/*/manifest.json'));self.assertEqual(len(manifests),6)
        for path in manifests:
            manifest=json.loads(path.read_text());name=manifest['cases'][0]['grasp_type'];mapping=self.raw['mappings'][name]
            with np.load(path.parent/'poses.npz',allow_pickle=False) as a:
                self.assertEqual(a['qpos'].shape,(2,51));self.assertEqual(a['grasp_qpos'].shape,(2,52))
                faces=np.array([m['mano_vertex_ids'] for m in mapping])
                bary=np.array([m['mano_barycentric'] for m in mapping])
                rebuilt=(a['mano_vertices'][:,faces]*bary[None,:,:,None]).sum(-2)
                np.testing.assert_allclose(rebuilt,a['mapped_hand_points'],atol=2e-6)
                np.testing.assert_array_equal(a['projected_points'],a['mapped_object_points'])
                self.assertTrue(np.all(a['final_cost']<=a['initial_cost']+1e-5))
                self.assertTrue(np.all(np.diff(a['cem_best_cost_history'],axis=1)<=1e-5))
                with np.load(Path(manifest['cases'][0]['source_manifest']).parent/'poses.npz') as source:
                    np.testing.assert_array_equal(a['source_shadow_grasp_qpos'],source['grasp_qpos'][:2])
            with np.load(path.parent/'contacts.npz') as c:
                self.assertEqual(c['offsets'].shape,(3,))
                self.assertEqual(c['mapped_hand_points_world'].shape,(2,len(mapping),3))
    def test_resume_returns_verified_shard_and_rejects_incompatible_config(self):
        from scripts.retarget_dexonomy_mano import export_shard
        for path in SMOKE.glob('*/shards/*/verification.json'):
            v=json.loads(path.read_text());manifest=json.loads((path.parent/'manifest.json').read_text())
            group=dict(type=manifest['cases'][0]['grasp_type'],part=int(path.parent.name),path=str(Path(manifest['cases'][0]['source_manifest']).parent))
            reused=export_shard(group,SMOKE,None,None,None,None,v['configuration_fingerprint'])
            self.assertEqual(reused['samples'],2)
            with self.assertRaisesRegex(ValueError,'Incompatible'):
                export_shard(group,SMOKE,None,None,None,None,'wrong')
if __name__=='__main__':unittest.main()
