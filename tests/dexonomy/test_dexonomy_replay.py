"""Coordinate/scale/contact regressions for the remote Shadow replay cache."""
import tempfile,unittest
from pathlib import Path
import numpy as np
import mujoco
from retarget.env.grasp.dexonomy_replay import contacts,relative_pose,load_record,resolve_scene

class ReplayTests(unittest.TestCase):
    def test_relative_pose_rotated_hand(self):
        from scipy.spatial.transform import Rotation as R
        h=np.r_[[1.,2.,3.],R.from_euler('z',90,degrees=True).as_quat(scalar_first=True)]
        o=np.r_[[1.,3.,3.],h[3:]]
        actual=relative_pose(h,o)
        np.testing.assert_allclose(actual[:3],[1,0,0],atol=1e-12)
        np.testing.assert_allclose(actual[3:],[1,0,0,0],atol=1e-12)

    def test_numpy_record_roundtrip_and_extra_scene_scale(self):
        import json
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);scene=root/'extracted/objects/scene_cfg/object/floating/scale010.npy';scene.parent.mkdir(parents=True)
            asset=scene.parent/'mesh.obj';asset.write_text('v 0 0 0\n')
            info=scene.parent/'info.json';info.write_text(json.dumps({'mass':1.,'density':1000.,'scale':1.}))
            np.save(scene,{'scene':{'object':{'type':'rigid_object','file_path':'mesh.obj','info_path':'info.json','scale':np.ones(3)*.1,'pose':np.array([1.,2.,3.,1.,0.,0.,0.])}},'task':{'obj_name':'object'}})
            record={'scene_path':'assets/object/objects/scene_cfg/object/floating/scale010.npy','scene_scale':np.array([2.]),'grasp_qpos':np.zeros((1,29))}
            result,_,factor=resolve_scene(root,record,0)
            np.testing.assert_allclose(result['scene']['object']['scale'],[.2]*3)
            np.testing.assert_allclose(result['scene']['object']['pose'][:3],[2,4,6])
            self.assertEqual(factor,2.)
            self.assertAlmostEqual(result['scene']['object']['density']*.001*.2**3,.1)

    def test_contact_endpoints_normals_and_link_frames(self):
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="hand" pos="0.019 0 0"><freejoint/><geom type="sphere" size=".01"/></body><body name="object"><geom type="sphere" size=".01"/></body></worldbody></mujoco>')
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        obs=contacts(model,data,model.body('object').id)
        self.assertEqual(len(obs['points']),1)
        np.testing.assert_allclose(obs['normals'],[[1,0,0]],atol=1e-10)
        np.testing.assert_allclose(obs['hand_points']-obs['object_points'],obs['distances'][:,None]*obs['normals'],atol=1e-12)
        body=int(obs['hand_body_ids'][0]);world=data.xpos[body]+data.xmat[body].reshape(3,3)@obs['hand_local_points'][0]
        np.testing.assert_allclose(world,obs['hand_points'][0],atol=1e-12)
        self.assertAlmostEqual(obs['object_penetration_m'],.001)
        self.assertGreater(float(obs['force_on_object_world'][0]@obs['normals'][0]*-1),0)

    def test_closure_uses_massless_object_subtree(self):
        from retarget.env.grasp.dexonomy_replay import closure
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="objectroot"><freejoint/><body name="objectmass"><geom type="sphere" size=".05" mass=".1"/></body></body></worldbody></mujoco>')
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        normal=np.concatenate([np.eye(3),-np.eye(3)])
        report,arrays=closure(model,data,model.body('objectroot').id,{'points':normal*.05,'object_points':normal*.05,'normals':normal})
        self.assertEqual(model.body_mass[model.body('objectroot').id],0.)
        self.assertAlmostEqual(report['closure_model']['object_mass_kg'],.1)
        self.assertAlmostEqual(report['closure_model']['radius_m'],.05)
        self.assertTrue(report['certified'])

    def test_repository_closure_symmetric_contacts(self):
        import torch
        from retarget.env.grasp.projected_comfree import ProjectedComFreeClosure,ClosureSettings
        torch.set_num_threads(1)
        normals=torch.cat([torch.eye(3),-torch.eye(3)]).double()[None]
        solver=ProjectedComFreeClosure(ClosureSettings(cuda_graph=False),center=[0,0,0],radius=.05,object_mass=.1,object_inertia=[.0001]*3)
        result=solver.solve(normals*.05,normals)
        self.assertTrue(bool(result.certified[0]))
        self.assertTrue(bool(result.valid[0]))
        force,torque=solver.contact_wrenches(result.row_forces,normals)
        wrench=torch.cat([force.sum(-2),(torch.linalg.cross((normals*.05)[:,None],force)+torque).sum(-2)/.05],dim=-1)[0]
        loads=torch.cat([torch.zeros(1,6),torch.eye(6),-torch.eye(6)]).double()*solver.test_force_n
        self.assertLess(float(torch.linalg.vector_norm((wrench+loads)/solver.test_force_n,dim=-1).max()),.05)

if __name__=='__main__':unittest.main()
