"""Whole-hand references, native validity and size-grid provenance."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from retarget.env.grasp.graspadmm_gesture import (snapshot_type12,load_type12,enabled,
    attach_reference,reference_diagnostics,make_reference)
from retarget.env.grasp.graspadmm_scene import to_upstream_qpos
from retarget.env.grasp.primitives import size_specifications
from scripts.run_graspadmm_batch import sample_jobs,verify_snapshot,digest


def test_preview_inventory_and_type12_grid():
    jobs,_=sample_jobs(seeds=[0],types=[9,10,12,22,23,30,31])
    assert len(jobs)==41 and {j['seed'] for j in jobs}=={0}
    sizes=size_specifications(12,'disk')
    np.testing.assert_allclose([p['size_scale'] for p in sizes],[1,1.125,1.25,1.375,1.5])
    assert all(p['height_m']==.016 and p['size_policy_version']==3 for p in sizes)


def test_approved_reference_uses_case_id_selected_pose_and_frozen_hash(tmp_path):
    package=tmp_path/'archive';(package/'cases').mkdir(parents=True)
    cid='t12_disk_s04_seed00'
    (package/'cases'/f'{cid}.json').write_text(json.dumps(dict(primitive=dict(size_scale=1.))))
    q=to_upstream_qpos(np.r_[[.01,.02,.03],[.1,-.2,.3],np.linspace(.1,.5,20)])
    np.savez(package/'candidates.npz',case_ids=np.array(['decoy',cid]),
        selected_qpos=np.stack([q+1,q]),mppi_qpos=np.stack([q+1,q]),admm_qpos=np.stack([q+2,q+3]))
    target=tmp_path/'reference/type12.json'
    record=snapshot_type12(package,target)
    np.testing.assert_array_equal(record['grasp_qpos'],q)
    np.testing.assert_allclose(record['object_center_palm'],-Rotation.from_quat(q[3:7],scalar_first=True).inv().apply(q[:3]))
    loaded,info=load_type12(SimpleNamespace(gesture_reference_root=target.parent))
    assert loaded['selected_stage']=='mppi' and info['sha256']==digest(target)
    plan=dict(runtime=str(tmp_path),gesture_reference_hashes={str(target):digest(target)})
    verify_snapshot(plan)
    target.write_text('{}')
    with pytest.raises(ValueError,match='Frozen gesture reference'):verify_snapshot(plan)


def test_optional_policy_and_independent_reference_do_not_lock_orientation():
    assert enabled(SimpleNamespace(gesture_policy='whole_hand_v1'),'23_Adduction_Grip')
    assert not enabled(SimpleNamespace(gesture_policy='whole_hand_v1'),'6_Pinch')
    assert not enabled(SimpleNamespace(gesture_policy='legacy'),'9_Palmar_Pinch')
    assert not enabled(SimpleNamespace(gesture_policy='whole_hand_v1',primitive_seed_policy='upstream'),'12_Precision_Disk')
    q=to_upstream_qpos(np.r_[np.zeros(6),np.linspace(.1,.5,20)])
    reference=dict(hand_qpos=np.r_[np.zeros(6),np.ones(20)*.2].tolist(),policy='whole_hand_v1')
    raw=dict(grasp_qpos=q[None],reference_orientation_wxyz=[1,0,0,0])
    attached=attach_reference(raw,reference,q)
    assert 'reference_orientation_wxyz' not in attached
    np.testing.assert_array_equal(attached['gesture_reference']['hand_qpos'][6:],reference['hand_qpos'][6:])
    assert 'gesture_reference' not in raw and reference['hand_qpos'][:6]==[0.]*6
    moved=q.copy();moved[3:7]=Rotation.from_euler('Y',90,degrees=True).as_quat(scalar_first=True)
    diagnostics=reference_diagnostics(moved,attached['gesture_reference'])
    assert diagnostics['orientation_error_deg']==pytest.approx(90)
    assert diagnostics['semantic_status']=='pending_user_review'
    approved=dict(reference,orientation_source='approved_snapshot')
    retained=attach_reference(raw,approved,moved)
    np.testing.assert_allclose(retained['gesture_reference']['hand_qpos'][3:6],reference['hand_qpos'][3:6])


def test_viewer_never_mixes_historical_type12_scale_indices():
    from retarget.viewer.runtime import preferred_batch_rows
    def row(batch,case,version):
        return dict(batch=Path(batch),name=case,case_id=case,task='grasp',hand='wuji2',
                    template='12_Precision_Disk',primitive=dict(size_policy_version=version))
    old=[row('old','t12_disk_s04_seed00',2),row('old','t12_disk_s00_seed00',2)]
    preview=row('preview','t12_disk_s00_seed00',3)
    assert preferred_batch_rows([*old,preview],'preview','old')==[preview]


@pytest.mark.parametrize('number',[23,31])
def test_gpu_whole_hand_conversion_keeps_participating_fingers_and_joint_limits(tmp_path,number):
    import torch
    if not torch.cuda.is_available():pytest.skip('whole-hand CUDA initialization')
    from retarget.config.run import ROOT
    from retarget.env.grasp.primitives import catalog,primitive_mesh
    from retarget.env.grasp.graspadmm_inputs import fixed_template
    name=next(r['template'] for r in catalog(ROOT/'data/grasp/templates') if r['type_id']==number)
    p=next(p for p in size_specifications(number) if p['size_scale']==1.)
    mesh=primitive_mesh(tmp_path/'object.obj',p['kind'],p['radius_m'],p['height_m'])
    args=SimpleNamespace(dexonomy_root=ROOT/'data/grasp/templates',scale=1.,init_device='cuda:0',
                         primitive_seed_policy='taxonomy',gesture_policy='whole_hand_v1')
    hand,q,data,_,info=fixed_template(args,name,mesh,tmp_path/'template')
    fit=info['gesture_reference']['source']['whole_hand_fit']
    assert fit['all_twenty_joints_fitted_together'] and fit['final_cost']<=fit['initial_cost']+1.e-10
    assert np.isfinite(q).all()
    joint_ids=[np.flatnonzero(hand.model.jnt_qposadr==i)[0] for i in range(6,26)]
    low,high=hand.model.jnt_range[joint_ids].T
    assert np.all(q[6:]>=low-1.e-8) and np.all(q[6:]<=high+1.e-8)
    fingers={int(body.split('finger')[1][0]) for body in data['hand_cbody'] if 'finger' in body}
    assert fingers==({2,3} if number==23 else {1,2,3})
    assert data['required_cbody']==info['required_cbody']
    assert 'reference_orientation_wxyz' not in data


def test_reference_canonical_landmarks_are_independent_of_world_root(tmp_path):
    import trimesh
    from retarget.env.grasp.inputs import build_scene,StaticHand
    mesh=tmp_path/'object.obj';trimesh.creation.box().export(mesh)
    model,surface,_,_=build_scene(mesh,hand_model='wuji2')
    bodies=np.array([model.body('right_finger2_link4').id])
    hand=StaticHand(model,surface,bodies,np.zeros((1,3)),np.array([[0.,1.,0.]]))
    q=np.r_[np.zeros(6),np.ones(20)*.2]
    a=make_reference(hand,q,'23_Adduction_Grip',{})
    q[:6]=[.1,.2,.3,.4,.5,.6]
    b=make_reference(hand,q,'23_Adduction_Grip',{})
    np.testing.assert_allclose(a['landmark_positions_palm'],b['landmark_positions_palm'],atol=1.e-12)


def test_approved_type12_actual_contacts_survive_template_slot_normal_failure(tmp_path):
    import torch
    import mujoco
    if mujoco.__version__!='3.3.3':pytest.skip('native integration uses .venv-graspadmm (MuJoCo 3.3.3 / Warp 1.8)')
    if not torch.cuda.is_available():pytest.skip('native CUDA initialization predicates')
    from retarget.config.run import ROOT
    from retarget.env.grasp.primitives import primitive_mesh
    from retarget.env.grasp.graspadmm_inputs import fixed_template
    from retarget.env.grasp.graspadmm_gesture_init import candidate_data,validate_candidates
    from retarget.env.grasp.graspadmm_scene import write_scene,from_upstream_qpos
    source=json.loads((Path(__file__).parent/'fixtures/graspadmm_type12_reference.json').read_text())
    mesh=primitive_mesh(tmp_path/'object.obj','disk',source['radius_m'],source['height_m'])
    args=SimpleNamespace(dexonomy_root=ROOT/'data/grasp/templates',scale=1.,init_device='cuda:0',
                         primitive_seed_policy='taxonomy',gesture_policy='legacy')
    hand,_,template,_,_=fixed_template(args,'12_Precision_Disk',mesh,tmp_path/'template')
    scene=write_scene(tmp_path/'scene',mesh)
    q=from_upstream_qpos(np.asarray(source['grasp_qpos']))
    candidate=candidate_data(hand,q,template,scene)
    result=validate_candidates(hand,[candidate],args,scene)[0]
    assert result['checks']['loss'] is False
    assert result['native_static_grasp_valid'] is True
    assert result['valid'] is True
    assert set(result['actual_contact_bodies'])>=set(g[0] for g in template['required_cbody'] if len(g)==1)
