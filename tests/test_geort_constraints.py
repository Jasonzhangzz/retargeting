"""Non-official pose-domain guarantees and actual mesh-loss derivatives."""
import copy
import numpy as np
import pytest
import torch
from retarget.baseline.geort.adapter import RobotAdapter
from retarget.baseline.geort.constrained_network import DomainIK,domain,create_ik


def test_domain_is_structural_in_saved_network_and_does_not_change_native_model():
    robot=RobotAdapter();native=robot.lower.copy();spec=domain(robot)
    net=DomainIK(robot,spec).eval();x=torch.randn(64,5,3)*.1
    with torch.no_grad():q=robot.unnormalize(net(x).numpy())
    assert (q>=np.array(spec['lower'])-1e-6).all() and (q<=np.array(spec['upper'])+1e-6).all()
    np.testing.assert_array_equal(np.array(spec['lower']).reshape(5,4)[1:,[2,3]],0)
    np.testing.assert_array_equal(np.array(spec['lower'])[[2,3]],0)
    np.testing.assert_array_equal(robot.lower,native)
    loaded=create_ik(robot,dict(training_mode='contact_v1',output_domain=spec)).eval()
    loaded.load_state_dict(copy.deepcopy(net.state_dict()))
    torch.testing.assert_close(loaded(x),net(x),rtol=0,atol=0)
    with pytest.raises(ValueError):create_ik(robot,{'training_mode':'unknown'})
    broken=copy.deepcopy(spec);broken['lower'][0]=-100
    with pytest.raises(ValueError):DomainIK(robot,broken)


def test_completed_constrained_model_catalog_accepts_mode_in_status_and_config(tmp_path):
    import json
    from moxian_web.geort_service import GeoRTService
    folder=tmp_path/'models/current';folder.mkdir(parents=True)
    (folder/'config.json').write_text(json.dumps(dict(training_mode='contact_v1',reproduction_profile='contact_v1')))
    (folder/'status.json').write_text(json.dumps(dict(status='completed',training_mode='contact_v1',run_id='new')))
    service=GeoRTService(tmp_path)
    try:
        item=service.catalog('models')[0]
        assert item['training_mode']=='contact_v1' and item['run_id']=='new' and item['id']=='current'
    finally:service.close()


def test_rare_thumb_contact_cannot_be_diluted_by_long_adjacent_finger_contact():
    from retarget.baseline.geort.constraints import balanced_contact_mean,sample_contact_frames
    # One thumb-index miss of 8 mm and 99 perfect adjacent-finger contacts.
    # Repeating the latter must not hide the former or remove its gradient.
    values=torch.zeros(100,10,requires_grad=True)
    contacts=torch.zeros(100,10,dtype=torch.bool);contacts[0,0]=True;contacts[1:,4]=True
    errors=values+torch.nn.functional.one_hot(torch.tensor(0),10)*16
    loss=balanced_contact_mean(errors,contacts)
    assert loss==8
    loss.backward();assert values.grad[0,0]==.5
    indices=sample_contact_frames(contacts.numpy(),32,np.random.default_rng(714))
    assert (indices==0).sum()>1 and (indices!=0).any()
    silent=balanced_contact_mean(values,torch.zeros_like(contacts))
    assert silent==0 and torch.isfinite(silent)


def test_contact_intent_keeps_official_pinch_even_when_surface_label_is_absent():
    from retarget.baseline.geort.constraints import contact_intent
    points=np.zeros((1,5,3));points[0,:,0]=[0,.014,.04,.08,.12]
    physical=np.zeros((1,10),bool);physical[0,9]=True
    intent=contact_intent(points,physical)
    assert intent[0,0] and intent[0,9] and intent.sum()==2


def test_true_fk_pinch_has_metric_scale_and_pulls_both_fingers_together_only_on_intent():
    from retarget.baseline.geort.constraints import true_fk_pinch_loss
    source=torch.zeros(1,21,3);source[:,[4,8,12,16,20],0]=torch.tensor([0.,.014,.05,.09,.13])
    tips=source[:,[4,8,12,16,20]].clone();tips[0,1,0]=.03;tips.requires_grad_()
    loss=true_fk_pinch_loss(tips,source)
    torch.testing.assert_close(loss,torch.tensor(4.))
    loss.backward()
    assert tips.grad[0,0,0]<0 and tips.grad[0,1,0]>0
    assert tips.grad[0,2:].abs().sum()==0
    source[0,8,0]=.025
    assert true_fk_pinch_loss(tips,source)==0


def test_closing_contact_can_adapt_shape_without_disabling_flat_hand_orientation():
    from retarget.baseline.geort.constraints import contact_shape_weights
    contacts=torch.zeros(2,10,dtype=torch.bool);contacts[:,0]=True
    bend=torch.zeros(2,5,2);bend[1,2]=1.
    weight=contact_shape_weights(contacts,bend)
    torch.testing.assert_close(weight[0],torch.full((5,),10.))
    torch.testing.assert_close(weight[1],torch.tensor([.1,.1,1.,1.,1.]))


def test_flat_distal_orientation_is_not_diluted_by_other_bones_or_active_when_closing():
    from retarget.baseline.geort.constraints import flat_direction_loss
    source=torch.tensor([1.,0,0]).expand(2,5,3)
    angle=np.deg2rad(15.);robot=torch.tensor([np.cos(angle),np.sin(angle),0.],dtype=torch.float32).repeat(2,5,1).requires_grad_()
    bends=torch.zeros(2,5,2);bends[1,1]=1.
    loss=flat_direction_loss(robot,source,bends)
    torch.testing.assert_close(loss,torch.tensor(1.),atol=1e-5,rtol=1e-5)
    loss.backward();assert robot.grad[0].abs().sum()>0 and robot.grad[1].abs().sum()==0


def test_native_mesh_penetration_and_contact_have_finite_nonzero_joint_gradients():
    from retarget.baseline.geort.constraints import ContactGeometry,WEIGHTS
    torch.set_num_threads(1);robot=RobotAdapter();geometry=ContactGeometry(robot,'cpu')
    # A deterministic native pose with crossing fingers, evaluated without a
    # teacher optimizer or a learned-FK substitute.
    q=torch.tensor(robot.lower+(robot.upper-robot.lower)*.9)[None]
    norm=torch.tensor(robot.normalize(q.numpy()),requires_grad=True)
    source=torch.randn(1,21,3)*.03;contact=torch.ones(1,10,dtype=torch.bool)
    loss,parts=geometry.losses(norm,source,contact)
    assert set(parts)==set(WEIGHTS) and parts['penetration']>0 and parts['surface_contact']>0
    parts['penetration'].backward(retain_graph=True)
    assert torch.isfinite(norm.grad).all() and norm.grad.abs().sum()>0
    norm.grad=None;parts['surface_contact'].backward(retain_graph=True)
    assert torch.isfinite(norm.grad).all() and norm.grad.abs().sum()>0
    norm.grad=None;loss.backward();assert torch.isfinite(norm.grad).all()
    # No contact attraction when the source has no corresponding contact intent.
    _,open_parts=geometry.losses(norm,source,torch.zeros_like(contact))
    assert open_parts['surface_contact']==0 and open_parts['surface_normal']==0
    # Signed Glove extension must remain extension; abs/acos targets would
    # incorrectly turn this into positive flexion.
    extended=q.clone();extended.reshape(1,5,4)[:,:,[2,3]]=-.1
    extension=extended.reshape(1,5,4)[:,:,[2,3]]
    x=torch.tensor(robot.normalize(extended.numpy()))
    _,matched=geometry.losses(x,source,torch.zeros_like(contact),extension)
    _,wrong_sign=geometry.losses(x,source,torch.zeros_like(contact),-extension)
    assert matched['bending']<1e-10 and wrong_sign['bending']>.1
