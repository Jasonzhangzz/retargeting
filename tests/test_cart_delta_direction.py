import torch
from moxian_web.cart_delta_direction import perturb_finger_observations,motion_direction_loss


def test_direction_sign_zero_response_and_gradient():
    delta=torch.tensor([[[.002,0.,0.]]]);valid=torch.ones(1,1,dtype=torch.bool)
    for response,expected in [(delta,0.),(-delta,2.),(torch.zeros_like(delta),1.)]:
        loss,_=motion_direction_loss(delta,response,valid)
        torch.testing.assert_close(loss,torch.tensor(expected))
    zero=torch.zeros_like(delta,requires_grad=True)
    motion_direction_loss(delta,zero,valid)[0].backward()
    assert torch.isfinite(zero.grad).all() and zero.grad[0,0,0]<0


def test_invalid_fingers_have_no_gradient():
    delta=torch.randn(2,5,3)*.002;response=torch.randn(2,5,3,requires_grad=True)
    valid=torch.zeros(2,5,dtype=torch.bool);valid[0,2]=True
    loss,_=motion_direction_loss(delta,response,valid);loss.backward()
    assert response.grad[0,2].norm()>0
    assert torch.count_nonzero(response.grad[~valid])==0
    loss,_=motion_direction_loss(delta,response,torch.zeros_like(valid))
    assert loss==0 and torch.isfinite(loss)


def test_coherent_perturbation_masks_palm_orientation_time_and_no_mutation():
    groups=torch.arange(460)%6
    inputs=dict(future_emf=torch.randn(2,10,5,9),emf_valid=torch.ones(2,10,5,dtype=torch.bool),
                tactile_positions=torch.randn(2,10,460,3),tactile_state=torch.ones(2,10,460),
                tactile_valid=torch.ones(2,10,460,dtype=torch.bool),future_dt=torch.rand(2,10))
    inputs['emf_valid'][0,3,2]=False;inputs['tactile_valid'][1,4,0]=False
    before={k:v.clone() for k,v in inputs.items()};delta=torch.randn(2,5,3)*.002
    moved=perturb_finger_observations(inputs,groups,delta)
    for k in before:torch.testing.assert_close(before[k],inputs[k])
    torch.testing.assert_close(moved['future_emf'][...,3:],inputs['future_emf'][...,3:])
    for k in ['future_dt','tactile_state','tactile_valid','emf_valid']:torch.testing.assert_close(moved[k],inputs[k])
    torch.testing.assert_close(moved['tactile_positions'][:,:,groups==5],inputs['tactile_positions'][:,:,groups==5])
    torch.testing.assert_close(moved['future_emf'][1,7,1,:3],inputs['future_emf'][1,7,1,:3]+delta[1,1])
    torch.testing.assert_close(moved['tactile_positions'][1,7,1],inputs['tactile_positions'][1,7,1]+delta[1,1])
    torch.testing.assert_close(moved['tactile_positions'][0,3,groups==2],inputs['tactile_positions'][0,3,groups==2])
    torch.testing.assert_close(moved['tactile_positions'][1,4,0],inputs['tactile_positions'][1,4,0])
