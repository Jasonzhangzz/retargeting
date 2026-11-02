"""CA proposals are seeds; geometry, six-group contacts and closure certify MPPI."""
import json
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
import mujoco
from test_grasp_contactaware import actual_seed
from retarget.env.grasp.spider_seed import POWER_TYPES,palm_region,export_seed,load_seed
from retarget.env.grasp.spider_reference import read_input
from retarget.env.grasp.spider_geometry import exact_closure,ContactRequirements
from retarget.env.grasp.forceaware_reference import _read_ca
from retarget.config.run import ROOT,load
from retarget.env.grasp.runner import validate


def test_power_classes_do_not_force_precision_or_intermediate_palm():
    assert {1,5,10,11,15,17,18,26,28,30,31} <= POWER_TYPES
    assert not ({6,7,8,9,12,13,14,16,20,21,22,23,24,25,27,29,32,33}&POWER_TYPES)


def test_failed_ca_with_no_active_contacts_is_a_valid_spider_seed(actual_seed,tmp_path):
    hand,guess,_=actual_seed
    hand.ik_reference_qpos=guess.hand_qpos.copy()
    hand.semantic_contact_local=hand.local.copy();hand.semantic_contact_normals=hand.normals.copy()
    guess.metadata['contactaware']=dict(success=False,required_finger_contact={})
    guess.contact_mask=np.zeros(len(hand.bodies),bool)
    args=validate(load(ROOT/'configs/grasp_contactaware_mppi.yaml'))
    out=export_seed(tmp_path/'proposal',hand,guess,args,template='1_Large_Diameter',primitive=dict(kind='cylinder',radius_m=.035,height_m=.12))
    _,metadata,seed,model=load_seed(out)
    assert metadata['ca_success'] is False and seed['required_groups'].all()
    assert not seed['ca_active_mask'].any()
    ref=read_input(out)
    assert ref.reference_kind=='ca_proposal' and ref.required_groups.all()
    np.testing.assert_allclose(ref.qpos[:26],guess.hand_qpos)
    with pytest.raises(ValueError,match='accepted DexForge'):_read_ca(out)
    with np.load(out/'card_input.npz') as raw:assert 'hand_reference_qpos' not in raw
    region=palm_region(model)
    names={model.body(int(i)).name for i in region['bodies']}
    assert 'right_palm_link' in names and len(names)>3
    assert all(not name.endswith(('link3','link4')) for name in names)
    requirements=ContactRequirements(model,seed)
    body=int(next(b for b in region['bodies'] if model.body(int(b)).name=='right_finger1_link1'))
    index=int(np.flatnonzero(region['bodies']==body)[0]);point=region['local'][index];normal=region['normals'][index]
    data=mujoco.MjData(model)
    for angle in (0.,.4):
        data.qpos[:26]=seed['qpos'][:26];data.qpos[6]+=angle;mujoco.mj_kinematics(model,data)
        R=data.xmat[body].reshape(3,3);world=data.xpos[body]+R@point
        assert requirements.classify(data,body,world,R@normal)==5
        assert requirements.classify(data,body,world,-R@normal)==-1


def test_virtual_or_rank_deficient_contacts_cannot_certify_force_closure():
    for points in [np.empty((0,3)),np.asarray([[1.,0,0],[-1.,0,0]])]:
        report=exact_closure(dict(points=points,normals=-points,friction=1.),center=np.zeros(3),radius=1.)
        assert not report['success']
    points=np.r_[np.eye(3),-np.eye(3)]
    report=exact_closure(dict(points=points,normals=-points,friction=1.),center=np.zeros(3),radius=1.)
    assert report['success'] and report['wrench_rank']==6 and report['reserve']>1e-4


@pytest.mark.parametrize('throws',[False,True])
def test_numerical_ca_failure_preserves_pre_ca_geometry_only_in_seed_mode(actual_seed,monkeypatch,throws):
    from retarget.env.grasp.inputs import contactaware_proposal
    import retarget.env.grasp.contactaware as ca
    hand,guess,args=actual_seed;original=guess.hand_qpos.copy();args.ca_as_seed=True
    def failure(*a,**kw):
        if throws:raise FloatingPointError('numerical failure')
        result=copy.deepcopy(guess);result.hand_qpos[:]=np.nan;return result
    monkeypatch.setattr(ca,'refine_grasp_reference',failure)
    proposal=contactaware_proposal(hand,guess,args,dict(kind='cylinder'))
    np.testing.assert_array_equal(proposal.hand_qpos,original)
    assert not proposal.metadata['contactaware']['success']
    assert 'pre-CA geometric' in proposal.metadata['contactaware']['fallback']
    args.ca_as_seed=False
    if throws:
        with pytest.raises(FloatingPointError):contactaware_proposal(hand,guess,args,dict(kind='cylinder'))


def test_malformed_proposal_rejected_without_consulting_ca_flag(actual_seed,tmp_path):
    hand,guess,_=actual_seed;hand.ik_reference_qpos=guess.hand_qpos.copy()
    hand.semantic_contact_local=hand.local.copy();hand.semantic_contact_normals=hand.normals.copy()
    guess.metadata['contactaware']=dict(success=False)
    args=validate(load(ROOT/'configs/grasp_contactaware_mppi.yaml'))
    out=export_seed(tmp_path/'proposal',hand,guess,args,template='6_Prismatic_4_Finger',primitive=dict(kind='cylinder',radius_m=.035,height_m=.12))
    _,meta,values,_=load_seed(out)
    assert not meta['power']  # Explicit template palm annotations remain requirements.
    values['qpos'][0]=np.nan;np.savez_compressed(out/'spider_seed.npz',**values)
    with pytest.raises(ValueError,match='input_invalid'):load_seed(out)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Native pose MPPI requires CUDA')
def test_pose_evaluation_keeps_object_fixed_and_retains_missing_finger_requirements(actual_seed,tmp_path):
    from retarget.env.grasp.spider_acquisition import PoseObjective,AcquisitionSettings
    hand,guess,_=actual_seed;hand.ik_reference_qpos=guess.hand_qpos.copy()
    hand.semantic_contact_local=hand.local.copy();hand.semantic_contact_normals=hand.normals.copy()
    guess.metadata['contactaware']=dict(success=False);guess.contact_mask=np.zeros(len(hand.bodies),bool)
    args=validate(load(ROOT/'configs/grasp_contactaware_mppi.yaml'))
    out=export_seed(tmp_path/'proposal',hand,guess,args,template='1_Large_Diameter',primitive=dict(kind='cylinder',radius_m=.035,height_m=.12))
    objective=PoseObjective(out,AcquisitionSettings(worlds=2,coarse_population=20,fine_population=20))
    palm=objective.cpu.body('right_palm_link').id
    geoms=objective.cpu.geom_bodyid==palm
    assert np.all(objective.geometry._group.numpy()[geoms]==5)
    assert np.all(objective.geometry._base.numpy()[geoms]==1)
    controls=objective.initial[:26].expand(2,1,26).clone();controls[1,0,0]+=.005
    costs=objective.evaluate(controls)
    assert torch.isfinite(costs).all() and objective.required.all()
    torch.testing.assert_close(objective.physics.qpos[:,26:],objective.initial[26:].expand(2,-1))
    for world in range(2):
        data=mujoco.MjData(objective.cpu);data.qpos[:]=objective.seed['qpos'];data.qpos[:26]=controls[world,0].cpu().numpy()
        mujoco.mj_kinematics(objective.cpu,data)
        np.testing.assert_allclose(objective.physics.xpos[world].cpu().numpy(),data.xpos,atol=1e-6)
    # Retaining all requested anchor slots cannot manufacture physical support.
    controls[:,:,0]+=1.
    objective.evaluate(controls)
    assert not objective.geometry.count.numpy().any()
    torch.testing.assert_close(objective.geometry.wrench_cost(),torch.full((2,),1.25,device='cuda:0'))
