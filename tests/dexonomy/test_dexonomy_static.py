"""Static geometry must preserve inputs without invoking dynamics or forces."""
import unittest,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import mujoco
from retarget.env.grasp.dexonomy_static import ROOT,StaticModels,decode_record,static_sample,geometry_contacts
class StaticTests(unittest.TestCase):
    def test_geometric_contact_frames(self):
        m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="hand" pos=".019 0 0"><freejoint/><geom type="sphere" size=".01"/></body><body name="object"><geom type="sphere" size=".01"/></body></worldbody></mujoco>')
        d=mujoco.MjData(m);mujoco.mj_kinematics(m,d);mujoco.mj_collision(m,d)
        with patch('mujoco.mj_contactForce',side_effect=AssertionError('force query forbidden')):
            c,od,sd=geometry_contacts(m,d,m.body('object').id)
        self.assertEqual(len(c['points_world']),1)
        np.testing.assert_allclose(c['object_normals_world'],[[1,0,0]],atol=1e-7)
        np.testing.assert_allclose(c['hand_points_world']-c['object_points_world'],c['distances_m'][:,None]*c['object_normals_world'],atol=1e-8)
        self.assertAlmostEqual(od,.001);self.assertFalse(any('force' in key for key in c))
    def test_real_pose_has_no_simulation_calls(self):
        plan=json.loads((ROOT/'results/dexonomy_shadow_pilot100_20261103/plan.json').read_text())
        p=Path(plan[0]['path']);p=p if p.is_absolute() else ROOT/p
        record=decode_record(p.read_bytes());row=plan[0]['row']
        with patch('mujoco.mj_step',side_effect=AssertionError('step forbidden')),patch('mujoco.mj_forward',side_effect=AssertionError('forward dynamics forbidden')),patch('mujoco.mj_contactForce',side_effect=AssertionError('contact forces forbidden')):
            out=static_sample(StaticModels(ROOT/'data/dexonomy'),record,row)
        for i,stage in enumerate(('pregrasp','grasp','squeeze')):
            np.testing.assert_array_equal(out['source_stage_hand_qpos'][i],record[stage+'_qpos'][row])
            np.testing.assert_array_equal(out['stage_qpos'][i,:29],record[stage+'_qpos'][row])
        self.assertEqual(out['stage_qpos'].shape,(3,36))
    def test_visual_and_collision_only_contacts_match(self):
        plan=json.loads((ROOT/'results/dexonomy_shadow_pilot100_20261103/plan.json').read_text())
        reduced=StaticModels(ROOT/'data/dexonomy',visual=False)
        visual=StaticModels(ROOT/'data/dexonomy',visual=True)
        for item in plan[::20]:
            path=Path(item['path']);path=path if path.is_absolute() else ROOT/path
            record=decode_record(path.read_bytes());a=static_sample(reduced,record,item['row']);b=static_sample(visual,record,item['row'])
            np.testing.assert_array_equal(a['stage_qpos'],b['stage_qpos'])
            for ca,cb in zip(a['contacts'],b['contacts']):
                self.assertEqual(len(ca['points_world']),len(cb['points_world']))
                for key in ['points_world','object_normals_world','hand_points_world','object_points_world','hand_local_points']:
                    np.testing.assert_allclose(ca[key],cb[key],atol=2e-7)

    def test_completed_preview_shards(self):
        from retarget.viewer.snapshot import get_snapshot
        root=ROOT/'results/dexonomy_shadow_static_preview_20261103';count=0
        for p in root.glob('*/shards/*/manifest.json'):
            snapshot=get_snapshot(p.parent);m=snapshot.model(0);d=mujoco.MjData(m)
            d.qpos[:]=snapshot.poses['qpos'][0];mujoco.mj_kinematics(m,d)
            with np.load(p.parent/'contacts.npz',allow_pickle=False) as c:
                self.assertEqual(len(c['offsets']),len(snapshot.manifest['cases'])*3+1)
                self.assertEqual(c['offsets'][-1],len(c['points_world']))
                a,b=map(int,c['offsets'][1:3]);ids=c['hand_body_ids'][a:b]
                if len(ids):
                    world=d.xpos[ids]+np.einsum('nij,nj->ni',d.xmat[ids].reshape(-1,3,3),c['hand_local_points'][a:b])
                    np.testing.assert_allclose(world,c['hand_points_world'][a:b],atol=2e-7)
                self.assertFalse(any('force' in key for key in c.files))
            count+=len(snapshot.manifest['cases'])
        self.assertEqual(count,100)
if __name__=='__main__':unittest.main()
